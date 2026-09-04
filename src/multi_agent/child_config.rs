//! Per-child spawn configuration (design doc §5.1).
//!
//! Semantics convention: `Option::None` means *inherit* (the parent agent /
//! the global [`MultiAgentConfig`](super::MultiAgentConfig) value). Every
//! `Option` field of [`Default`] is `None` — "set nothing" is "inherit
//! everything".
//!
//! There is deliberately no typestate magic here (six fields don't warrant
//! it, §5.1): the required-field check (`system_prompt`) runs at spawn time
//! and fails fast with `ConfigError` on the first call.

use std::collections::BTreeSet;

/// Configuration for one spawned child agent.
///
/// Built fluently via [`ChildBuilder`](super::child_builder::ChildBuilder)
/// (stage-1 API, same feature gate), or constructed directly.
#[derive(Debug, Clone, Default)]
pub struct ChildConfig {
    /// System prompt. `ChildBuilder::spawn` validates at runtime that this
    /// is set (directly or via a preset) and errors otherwise.
    pub system_prompt: Option<String>,

    /// Tool whitelist: picked from the parent's business tools by their
    /// real registered names.
    ///
    /// `None` = inherit all of the parent's business tools (still subject to
    /// the global `child_excluded_tools`). Validation runs against the
    /// **post-exclusion registered set** (design §5.4, review M-3):
    /// - a whitelisted name that resolves nowhere → spawn fails with
    ///   `ToolNotFound` (never silently degrade to a smaller set);
    /// - a whitelisted name that the deployment intentionally excluded →
    ///   `tracing::warn!` + the spawn echoes the actually-registered set.
    pub tool_names: Option<BTreeSet<String>>,

    /// Max turns → `AgentBuilder::execution_max_turns(u32)` (agent-base
    /// builder.rs:223).
    pub max_turns: Option<u32>,

    /// Context window → `AgentBuilder::context_window(usize)` (agent-base
    /// builder.rs:180).
    pub context_window: Option<usize>,

    /// Overrides this spawn's permission (takes effect only when
    /// `child_permission_mode = PerSpawn`; keeping the knob does not open a
    /// privilege door for the LLM, see design §10.1 B4).
    pub full_permission: Option<bool>,

    /// Requested model override for this child, verbatim from the spawn
    /// arguments (`spawn_agent`'s `model` field). `None` = inherit the
    /// parent's model (the only behaviour today).
    ///
    /// TODO(layer-3): **currently inert.** The model name is baked into the
    /// shared `LlmProvider` (`LlmConfig.model` → `create_provider`), and
    /// `llm_trait::ChatRequest` has no per-request model field, so there is
    /// no route for this value to influence LLM calls yet. Layer-3 work:
    /// add `ChatRequest.model: Option<String>` (llm-trait), honour it in
    /// each protocol (llm-providers), then thread this field into the child
    /// builder. Until then this exists only to accept and carry the LLM's
    /// choice so the schema does not churn again.
    pub model: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_inherits_everything() {
        // "All None" is the whole point: a default ChildConfig carries no
        // intent, so the child behaves exactly like the legacy spawn path.
        let c = ChildConfig::default();
        assert!(c.system_prompt.is_none());
        assert!(c.tool_names.is_none());
        assert!(c.max_turns.is_none());
        assert!(c.context_window.is_none());
        assert!(c.full_permission.is_none());
        assert!(c.model.is_none());
    }

    #[test]
    fn clone_is_deep_and_independent() {
        let c = ChildConfig {
            tool_names: Some(BTreeSet::from(["read_file".to_string()])),
            ..Default::default()
        };
        let c2 = c.clone();
        assert_eq!(c2.tool_names.as_ref().unwrap().len(), 1);
        drop(c);
        assert!(c2.tool_names.is_some());
    }
}
