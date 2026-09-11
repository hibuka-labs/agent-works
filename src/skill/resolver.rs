//! Filesystem-backed skill discovery and resolution.
//!
//! `SkillResolver` scans SKILL.md directories (later dirs override earlier
//! ones on name collision), resolves `/name args` slash input with fuzzy
//! matching, and serves exact-name lookups for the model-facing `skill` tool.
//! Host apps decide *which* directories to scan (policy); this module only
//! knows *how* to scan and resolve (mechanism).

use std::collections::HashMap;
use std::path::PathBuf;

use super::Skill;
use super::prompt_skill::PromptSkill;

/// Scanned skill set backing slash lookup and the `skill` tool.
pub struct SkillResolver {
    skills: Vec<PromptSkill>,
}

/// `resolve_by_name` 的失败原因（`skill` 工具据此生成不同错误文案）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillLookupError {
    /// 没有这个名字的 skill。
    NotFound,
    /// skill 标记了 `disable-model-invocation`：对模型拒绝（用户的斜杠
    /// 路径不受影响——该 flag 只约束模型端触发）。
    ModelInvocationDenied,
}

impl SkillResolver {
    /// 从多个目录扫描 skills（后扫描的优先级更高，同名覆盖）。
    pub fn from_dirs(dirs: &[PathBuf]) -> Self {
        let mut by_name: HashMap<String, PromptSkill> = HashMap::new();

        for dir in dirs {
            match PromptSkill::scan_dir(dir) {
                Ok(skills) => {
                    for skill in skills {
                        let name = skill.name().to_string();
                        by_name.insert(name, skill); // 后者覆盖前者
                    }
                }
                Err(e) => {
                    tracing::warn!(dir = %dir.display(), error = %e, "failed to scan skill directory");
                }
            }
        }

        let mut skills: Vec<PromptSkill> = by_name.into_values().collect();
        skills.sort_by(|a, b| a.name().cmp(b.name()));
        Self { skills }
    }

    /// 解析用户输入，如果是 `/skill-name args` 则返回解析后的 skill body。
    ///
    /// 返回 `None` 表示输入不是斜杠命令（或未匹配到 skill），应原样提交。
    pub fn resolve(&self, input: &str) -> Option<String> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return None;
        }

        // 拆分：`/skill-name arg1 arg2 ...` → name="skill-name", rest="arg1 arg2 ..."
        let without_slash = &trimmed[1..];
        let (name, raw_args) = match without_slash.split_once(char::is_whitespace) {
            Some((n, rest)) => (n, rest.trim()),
            None => (without_slash, ""),
        };

        // 名字不能为空、不能包含空格（防止 "/ " 被当成 skill 命令）
        if name.is_empty() || name.contains(char::is_whitespace) {
            return None;
        }

        // 模糊匹配：exact → suffix → contains → word-overlap，同级取最短名
        let skill = self.fuzzy_find(name)?;

        if !skill.is_user_invocable() {
            tracing::info!(
                name = name,
                "skill is not user-invocable, passing through as text"
            );
            return None;
        }

        // 暂不解析命名参数，只传 raw_arguments
        let params = HashMap::new();
        let body = skill.resolve_body(&params, raw_args);

        tracing::info!(
            input = name,
            matched = skill.name(),
            args = raw_args,
            body_len = body.len(),
            "resolved /skill command"
        );

        Some(body)
    }

    /// 按优先级模糊查找 skill：exact → suffix → contains → word-overlap。
    /// 同级取名字最短的（最具体）。返回 `None` 表示完全没匹配。
    fn fuzzy_find(&self, query: &str) -> Option<&PromptSkill> {
        // 候选：user_invocable 的才参与匹配
        let candidates: Vec<&PromptSkill> = self
            .skills
            .iter()
            .filter(|s| s.is_user_invocable())
            .collect();

        // 1. 精确匹配
        if let Some(s) = candidates.iter().find(|s| s.name() == query) {
            return Some(s);
        }

        // 2. 后缀匹配：skill name 以 query 结尾（如 requesting-code-review 以 code-review 结尾）
        let mut suffix_hits: Vec<&&PromptSkill> = candidates
            .iter()
            .filter(|s| s.name().ends_with(query) && s.name() != query)
            .collect();
        if !suffix_hits.is_empty() {
            suffix_hits.sort_by_key(|s| s.name().len());
            return Some(suffix_hits[0]);
        }

        // 3. 子串匹配：skill name 包含 query
        let mut contains_hits: Vec<&&PromptSkill> = candidates
            .iter()
            .filter(|s| s.name().contains(query))
            .collect();
        if !contains_hits.is_empty() {
            contains_hits.sort_by_key(|s| s.name().len());
            return Some(contains_hits[0]);
        }

        // 4. 词级匹配：query 拆词后是 skill name 拆词的子集
        //    如 "review-pr" → ["review", "pr"] ⊆ ["pre", "landing", "pr", "review"]
        let query_words: Vec<&str> = query.split('-').collect();
        let mut word_hits: Vec<(&&PromptSkill, usize)> = candidates
            .iter()
            .filter_map(|s| {
                let name_words: Vec<&str> = s.name().split('-').collect();
                let all_match = query_words.iter().all(|qw| name_words.contains(qw));
                if all_match {
                    Some((s, name_words.len()))
                } else {
                    None
                }
            })
            .collect();
        if !word_hits.is_empty() {
            // 取词数最少的（最具体的）
            word_hits.sort_by_key(|(_, len)| *len);
            return Some(word_hits[0].0);
        }

        None
    }

    /// 已加载的 skill 数量（catalog 构建与测试用）。
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// 是否为空（catalog 注入判空与测试用）。
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// 只读访问已加载的 skills（按名字排序）。
    ///
    /// 供 catalog 渲染等同 crate 消费方遍历；不暴露 `&mut`——resolver 是
    /// 共享快照，变更走重建（热重载）。
    pub fn entries(&self) -> &[PromptSkill] {
        &self.skills
    }

    /// 列出所有已加载 skill 的名字（含 `user_invocable: false`；目前只用于
    /// 宿主日志行）。
    pub fn skill_names(&self) -> Vec<&str> {
        self.skills.iter().map(|s| s.name()).collect()
    }

    /// 列出所有已加载 skill 的 (name, description) 摘要（供 `/` picker 展示）。
    pub fn skill_summaries(&self) -> Vec<(String, String)> {
        self.skills
            .iter()
            .map(|s| (s.name().to_string(), s.brief_description()))
            .collect()
    }

    /// Exact-match by name (skill tool uses this; `/` slash path uses `resolve`).
    ///
    /// Unlike `resolve`, this does NOT filter on `user_invocable` — model tools
    /// can load every skill, and the catalog lists every skill regardless.
    /// Skills flagged `disable-model-invocation` are refused: the trait contract
    /// ("the LLM cannot auto-trigger this skill") is enforced at both model
    /// surfaces — the catalog omits them and this lookup denies them.
    /// Returns `(body, name)` so the caller can trace which skill was served.
    pub fn resolve_by_name(
        &self,
        name: &str,
        args: &str,
    ) -> Result<(String, &str), SkillLookupError> {
        let skill = self.skills.iter().find(|s| s.name() == name);
        let Some(skill) = skill else {
            return Err(SkillLookupError::NotFound);
        };
        if skill.disable_model_invocation() {
            return Err(SkillLookupError::ModelInvocationDenied);
        }
        let params = HashMap::new();
        let body = skill.resolve_body(&params, args);
        Ok((body, skill.name()))
    }

    /// Absolute path of the `SKILL.md` that backs a named skill (for the
    /// overflow fallback: tool returns truncated body + path for read_file).
    pub fn source_path_for(&self, name: &str) -> Option<&std::path::Path> {
        self.skills.iter().find(|s| s.name() == name)?.source_path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    /// 创建临时 skill 目录结构：
    ///   tmp_dir/skills/test-skill/SKILL.md
    fn make_skill_dir(tmp: &Path, name: &str, body: &str, user_invocable: bool) {
        let skill_dir = tmp.join("skills").join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        let invocable = if user_invocable { "true" } else { "false" };
        let content = format!(
            "---\nname: {name}\ndescription: test skill\nuser-invocable: {invocable}\n---\n\n{body}"
        );
        fs::write(skill_dir.join("SKILL.md"), content).unwrap();
    }

    #[test]
    fn resolve_slash_command() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "commit", "Run tests then commit.", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert_eq!(resolver.len(), 1);

        let result = resolver.resolve("/commit");
        assert!(result.is_some());
        assert!(result.unwrap().contains("Run tests then commit."));
    }

    #[test]
    fn resolve_slash_command_with_args() {
        let tmp = tempfile::tempdir().unwrap();
        let skill_dir = tmp.path().join("skills").join("commit");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: commit\ndescription: d\nuser-invocable: true\n---\n\nRun tests then commit.\nArgs: $ARGUMENTS",
        )
        .unwrap();

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        let result = resolver.resolve("/commit --amend");
        assert!(result.is_some());
        let body = result.unwrap();
        assert!(
            body.contains("--amend"),
            "body should contain raw arguments"
        );
    }

    #[test]
    fn non_slash_input_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "commit", "body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert!(resolver.resolve("hello world").is_none());
        assert!(resolver.resolve("no slash").is_none());
    }

    #[test]
    fn unknown_skill_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "commit", "body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert!(resolver.resolve("/nonexistent").is_none());
    }

    #[test]
    fn not_user_invocable_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "internal", "body", false);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert!(resolver.resolve("/internal").is_none());
    }

    #[test]
    fn bare_slash_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        assert!(resolver.resolve("/").is_none());
        assert!(resolver.resolve("/ ").is_none());
    }

    #[test]
    fn project_level_overrides_user_level() {
        let tmp = tempfile::tempdir().unwrap();
        let user_dir = tmp.path().join("user_skills");
        let proj_dir = tmp.path().join("proj_skills");

        // 同名 skill，不同 body
        let sd = user_dir.join("commit").join("SKILL.md");
        fs::create_dir_all(sd.parent().unwrap()).unwrap();
        fs::write(
            &sd,
            "---\nname: commit\ndescription: d\nuser-invocable: true\n---\n\nuser body",
        )
        .unwrap();

        let sd = proj_dir.join("commit").join("SKILL.md");
        fs::create_dir_all(sd.parent().unwrap()).unwrap();
        fs::write(
            &sd,
            "---\nname: commit\ndescription: d\nuser-invocable: true\n---\n\nproject body",
        )
        .unwrap();

        // 后扫描的优先级更高
        let resolver = SkillResolver::from_dirs(&[user_dir, proj_dir]);
        assert_eq!(resolver.len(), 1);
        let body = resolver.resolve("/commit").unwrap();
        assert!(
            body.contains("project body"),
            "project-level should override user-level"
        );
    }

    #[test]
    fn empty_dir_yields_empty_resolver() {
        let tmp = tempfile::tempdir().unwrap();
        let resolver = SkillResolver::from_dirs(&[tmp.path().join("nonexistent")]);
        assert!(resolver.is_empty());
        assert!(resolver.resolve("/anything").is_none());
    }

    // ── 模糊匹配测试 ──

    #[test]
    fn fuzzy_suffix_match() {
        // 模拟 Claude Code 场景：skill 叫 requesting-code-review，用户输入 /code-review
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(
            tmp.path(),
            "requesting-code-review",
            "review body here",
            true,
        );

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);
        let result = resolver.resolve("/code-review");
        assert!(
            result.is_some(),
            "suffix match should find requesting-code-review"
        );
        assert!(result.unwrap().contains("review body here"));
    }

    #[test]
    fn fuzzy_contains_match() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "design-review", "design body", true);
        make_skill_dir(tmp.path(), "requesting-code-review", "code body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        // "review" 是子串，但不是后缀 → 走 contains 分支
        // 两个都 contains "review"，取最短的（design-review）
        let result = resolver.resolve("/review");
        assert!(result.is_some());
        assert!(
            result.unwrap().contains("design body"),
            "contains match should pick shortest name"
        );
    }

    #[test]
    fn fuzzy_exact_takes_priority() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "review", "exact body", true);
        make_skill_dir(tmp.path(), "code-review", "suffix body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        // 精确匹配 "review" 应优先于后缀匹配 "code-review"
        let result = resolver.resolve("/review");
        assert!(result.is_some());
        assert!(
            result.unwrap().contains("exact body"),
            "exact match should win over suffix"
        );
    }

    #[test]
    fn fuzzy_suffix_picks_shortest() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "requesting-code-review", "long body", true);
        make_skill_dir(tmp.path(), "code-review", "short body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        // 两个都以 "code-review" 结尾，取最短的
        let result = resolver.resolve("/code-review");
        assert!(result.is_some());
        assert!(
            result.unwrap().contains("short body"),
            "suffix match should pick shortest name"
        );
    }

    #[test]
    fn fuzzy_word_overlap() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(
            tmp.path(),
            "finishing-a-development-branch",
            "finish body",
            true,
        );

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        // "dev-branch" → ["dev", "branch"]，skill 拆词 ["finishing","a","development","branch"]
        // "dev" 不在 skill 词里（skill 用的是 "development"），所以不匹配
        let result = resolver.resolve("/dev-branch");
        assert!(result.is_none(), "partial word should not match");

        // "development-branch" → 完整词匹配
        let result = resolver.resolve("/development-branch");
        assert!(result.is_some());
        assert!(result.unwrap().contains("finish body"));
    }

    // ── resolve_by_name（`skill` 工具路径）──

    #[test]
    fn resolve_by_name_denies_disable_model_invocation() {
        let tmp = tempfile::tempdir().unwrap();
        let deploy_dir = tmp.path().join("skills").join("deploy");
        fs::create_dir_all(&deploy_dir).unwrap();
        fs::write(
            deploy_dir.join("SKILL.md"),
            "---\nname: deploy\ndescription: d\nuser-invocable: true\ndisable-model-invocation: true\n---\n\nship it",
        )
        .unwrap();
        make_skill_dir(tmp.path(), "commit", "commit body", true);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        assert_eq!(
            resolver.resolve_by_name("deploy", ""),
            Err(SkillLookupError::ModelInvocationDenied),
            "disable-model-invocation skill must be denied on the model path"
        );
        assert_eq!(
            resolver.resolve_by_name("nope", ""),
            Err(SkillLookupError::NotFound)
        );
        let (body, name) = resolver.resolve_by_name("commit", "").unwrap();
        assert_eq!(name, "commit");
        assert!(body.contains("commit body"));
    }

    #[test]
    fn skill_summaries_cover_all_skills_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        make_skill_dir(tmp.path(), "zeta", "z body", true);
        make_skill_dir(tmp.path(), "internal-helper", "h body", false);

        let resolver = SkillResolver::from_dirs(&[tmp.path().join("skills")]);

        let summaries = resolver.skill_summaries();
        let names: Vec<&str> = summaries.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["internal-helper", "zeta"],
            "summaries sorted by name"
        );
        // 非 user-invocable 的 skill 也要在册（斜杠路径才过滤，picker 要全量）。
        assert!(names.contains(&"internal-helper"));
        assert!(summaries.iter().all(|(_, d)| !d.is_empty()));
    }

    #[test]
    #[cfg(unix)]
    fn unreadable_dir_does_not_block_later_dirs() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let bad_dir = tmp.path().join("bad_skills");
        let good_dir = tmp.path().join("good_skills");
        fs::create_dir_all(bad_dir.join("broken")).unwrap();
        fs::create_dir_all(bad_dir.join("locked")).unwrap();
        // good 目录里放一个真 skill
        let sd = good_dir.join("commit");
        fs::create_dir_all(&sd).unwrap();
        fs::write(
            sd.join("SKILL.md"),
            "---\nname: commit\ndescription: d\nuser-invocable: true\n---\n\nlater dir body",
        )
        .unwrap();

        // bad_dir 下放进一个会被锁死的 skill 目录，再把 bad_dir 本身 chmod 000
        fs::write(
            bad_dir.join("broken").join("SKILL.md"),
            "---\nname: broken\ndescription: d\nuser-invocable: true\n---\n\nshould be unreadable",
        )
        .unwrap();
        fs::set_permissions(&bad_dir, fs::Permissions::from_mode(0o000)).unwrap();

        // 无论扫描结果如何，先恢复权限，保证 tempdir 清理不残留。
        let result = std::panic::catch_unwind(|| {
            SkillResolver::from_dirs(&[bad_dir.clone(), good_dir.clone()])
        });
        fs::set_permissions(&bad_dir, fs::Permissions::from_mode(0o755)).unwrap();

        let resolver = result.expect("from_dirs must not panic on unreadable dir");
        assert_eq!(
            resolver.len(),
            1,
            "later readable dir must still be scanned"
        );
        assert!(resolver.resolve("/commit").is_some());
    }
}
