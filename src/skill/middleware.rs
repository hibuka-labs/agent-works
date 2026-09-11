//! Per-turn system prompt refresh for the skills catalog (skill-injection M3a).
//!
//! Host apps embed the catalog in the stored system prompt at build time and
//! let this middleware swap in a fresh one before each LLM call
//! (`refresh_catalog` — idempotent, in place), so resolver changes show up
//! without restarting and the prompt never accumulates duplicate catalogs.
//!
//! Uses `agent_base::Middleware::on_pre_llm`, which runs before every LLM
//! call — zero cross-repo changes needed on the engine side.

use std::sync::Arc;

use agent_base::llm_trait::ChatMessage;
use agent_base::{AgentResult, Middleware, PreLlmCtx};
use async_trait::async_trait;

use super::catalog::render_catalog;
use super::resolver::SkillResolver;

/// Middleware that refreshes the system prompt (base + skills catalog) before
/// each LLM call. This lets the model see skill catalog changes (additions,
/// removals, description edits) without restarting the session.
///
/// The resolver is `Arc`-shared with the host app's slash path and the
/// `SkillTool` — all three read the same immutable skill list. To pick up
/// filesystem changes, the resolver would need to be rebuilt (future work;
/// the crate's `hot-reload` feature is the natural home); this middleware
/// handles the *wiring* half: ensuring the prompt the model sees always
/// reflects the resolver's current state, without duplication.
pub struct SkillCatalogRefreshMiddleware {
    resolver: Arc<SkillResolver>,
}

impl SkillCatalogRefreshMiddleware {
    pub fn new(resolver: Arc<SkillResolver>) -> Self {
        Self { resolver }
    }
}

#[async_trait]
impl Middleware for SkillCatalogRefreshMiddleware {
    async fn on_pre_llm(&self, ctx: &mut PreLlmCtx) -> AgentResult<()> {
        // Only touch the system message when there are skills to inject.
        // Without skills, the builder's original prompt is left byte-identical.
        let Some(catalog) = render_catalog(&self.resolver) else {
            return Ok(());
        };
        match ctx.messages.first_mut() {
            Some(ChatMessage::System { content, .. }) => {
                // Idempotent refresh: swap the previously injected catalog (the
                // builder already composed one into the stored prompt) for the
                // current one, in place — trailing sections keep their position.
                // A plain append made the model see a duplicate catalog on every
                // LLM call.
                *content = super::catalog::refresh_catalog(content, &catalog);
            }
            // 静默跳过会掩盖接线错误：skills 已装配、catalog 却从不刷新，
            // 模型看到的目录会悄悄过期——这是难以察觉的回归，至少留日志。
            _ => tracing::warn!(
                "skill catalog refresh skipped: first message is not System \
                 (skills are loaded but the catalog cannot be refreshed)"
            ),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::SessionId;
    use std::path::Path;

    const BASE: &str = "You are a coding agent.";

    fn make_skill_dir(tmp: &Path, name: &str, body: &str) {
        let dir = tmp.join("skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: test\nuser-invocable: true\n---\n\n{body}"),
        )
        .unwrap();
    }

    fn make_pre_llm_ctx(system_content: &str) -> PreLlmCtx {
        PreLlmCtx {
            session_id: SessionId {
                id: 1,
                external_id: None,
            },
            messages: vec![ChatMessage::System {
                content: system_content.to_string(),
                ephemeral: false,
            }],
            tools: vec![],
            emit_fn: None,
            turn_count: 1,
            max_turns: 10,
        }
    }

    /// builder 产物的等价构造：base + "\n\n" + catalog —— 宿主应用在 build()
    /// 时就是这样把 catalog 组进存盘 system prompt 的。
    fn stored_prompt(resolver: &SkillResolver) -> String {
        format!("{BASE}\n\n{}", render_catalog(resolver).unwrap())
    }

    #[tokio::test]
    async fn refresh_replaces_system_message_with_catalog() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));

        let mw = SkillCatalogRefreshMiddleware::new(resolver);
        let mut ctx = make_pre_llm_ctx("original prompt");

        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert!(
                    content.starts_with("original prompt"),
                    "base prompt must be preserved"
                );
                assert!(content.contains("## Skills"), "catalog must be appended");
                assert!(content.contains("- alpha: test"), "skill must appear");
            }
            _ => panic!("expected System message"),
        }
    }

    #[tokio::test]
    async fn refresh_without_skills_keeps_original_prompt() {
        let tmp = tempfile::tempdir().unwrap();
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("nonexistent")]));

        let mw = SkillCatalogRefreshMiddleware::new(resolver);
        let original = "base prompt only";
        let mut ctx = make_pre_llm_ctx(original);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert_eq!(content, original, "must be byte-identical without skills");
            }
            _ => panic!("expected System message"),
        }
    }

    #[tokio::test]
    async fn refresh_reflects_newly_added_skill() {
        let tmp = tempfile::tempdir().unwrap();
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));
        let mw = SkillCatalogRefreshMiddleware::new(resolver.clone());

        // First turn: no skills
        let mut ctx = make_pre_llm_ctx("prompt");
        mw.on_pre_llm(&mut ctx).await.unwrap();
        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert!(!content.contains("## Skills"));
            }
            _ => panic!(),
        }

        // Add a skill and rebuild resolver
        make_skill_dir(tmp.path(), "new-skill", "new body");
        let resolver2 = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));
        let mw2 = SkillCatalogRefreshMiddleware::new(resolver2);

        // Second turn: skill appears
        let mut ctx = make_pre_llm_ctx("prompt");
        mw2.on_pre_llm(&mut ctx).await.unwrap();
        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert!(
                    content.contains("- new-skill: test"),
                    "new skill must appear"
                );
            }
            _ => panic!(),
        }
    }

    // ── Catalog duplication regression (build-time M1 × per-turn M3a) ──
    //
    // build() 已经把 catalog 组进存盘的 system prompt；middleware 必须先
    // 换旧再注入，而不是再追加一份（模型每次 LLM 调用曾看到两份目录）。

    #[tokio::test]
    async fn refresh_strips_builder_embedded_catalog_instead_of_duplicating() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));

        // 真实接线：builder 存进 session 的就是含 catalog 的 prompt
        let stored = stored_prompt(&resolver);
        assert!(stored.contains("## Skills"));

        let mw = SkillCatalogRefreshMiddleware::new(resolver.clone());
        let mut ctx = make_pre_llm_ctx(&stored);
        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert_eq!(
                    content.matches("## Skills").count(),
                    1,
                    "catalog must appear exactly once, not duplicated per LLM call"
                );
                assert_eq!(
                    content,
                    &stored_prompt(&resolver),
                    "unchanged resolver → byte-identical recomposition"
                );
            }
            _ => panic!("expected System message"),
        }
    }

    #[tokio::test]
    async fn refresh_preserves_trailing_sections_after_catalog() {
        // token-budget 类宿主：prompt suffix 以 push_str 直接拼在 catalog 之后
        // （suffix 自带 \n\n 前导），刷新旧目录时不得吃掉它后面的内容。
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));

        let suffix = "\n\n## Context Management\n\nuse history and notes tools.";
        let stored = format!("{}{suffix}", stored_prompt(&resolver));

        let mw = SkillCatalogRefreshMiddleware::new(resolver.clone());
        let mut ctx = make_pre_llm_ctx(&stored);
        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => {
                assert_eq!(content.matches("## Skills").count(), 1, "{content}");
                assert!(
                    content.ends_with(suffix),
                    "trailing section must survive: {content}"
                );
                assert_eq!(
                    content, &stored,
                    "unchanged resolver + suffix → byte-identical to builder output"
                );
            }
            _ => panic!("expected System message"),
        }
    }

    #[tokio::test]
    async fn refresh_is_idempotent_across_repeated_calls() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));

        let mw = SkillCatalogRefreshMiddleware::new(resolver.clone());
        let mut ctx = make_pre_llm_ctx(&stored_prompt(&resolver));
        mw.on_pre_llm(&mut ctx).await.unwrap();

        let after_first = match &ctx.messages[0] {
            ChatMessage::System { content, .. } => content.clone(),
            _ => panic!(),
        };

        // 第二次调用（下一个 LLM call，messages 来自 session 的重新 clone）
        let mut ctx2 = make_pre_llm_ctx(&after_first);
        mw.on_pre_llm(&mut ctx2).await.unwrap();

        match &ctx2.messages[0] {
            ChatMessage::System { content, .. } => {
                assert_eq!(
                    content, &after_first,
                    "repeated refresh must be a fixed point"
                );
                assert_eq!(content.matches("## Skills").count(), 1);
            }
            _ => panic!("expected System message"),
        }
    }

    // ── 非 System 开头的健壮性：跳过 + 告警日志，不 panic、不误改 ──

    #[tokio::test]
    async fn non_system_first_message_left_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));
        let mw = SkillCatalogRefreshMiddleware::new(resolver);

        let mut ctx = make_pre_llm_ctx("unused");
        ctx.messages = vec![ChatMessage::User {
            content: "hi".to_string(),
            images: vec![],
            ephemeral: false,
        }];
        mw.on_pre_llm(&mut ctx).await.unwrap();

        match &ctx.messages[0] {
            ChatMessage::User { content, .. } => {
                assert_eq!(content, "hi", "user message must survive")
            }
            other => panic!("expected User message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_messages_do_not_panic() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "alpha", "alpha body");
        let resolver = Arc::new(SkillResolver::from_dirs(&[tmp.path().join("skills")]));
        let mw = SkillCatalogRefreshMiddleware::new(resolver);

        let mut ctx = make_pre_llm_ctx("unused");
        ctx.messages.clear();
        mw.on_pre_llm(&mut ctx).await.unwrap();
        assert!(ctx.messages.is_empty());
    }
}
