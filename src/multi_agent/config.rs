//! Multi-agent configuration.
//!
//! Controls whether multi-agent capabilities are enabled and sets resource limits.
//! Multi-agent is enabled by default — users who don't need it can disable via
//! `.without_multi_agent()` or feature gate.

use agent_base::ReasoningEffort;

/// Permission mode for spawned child agents.
///
/// Controls whether sub-agents run with full tool access or are restricted by
/// the parent's [`ToolPolicy`](agent_base::ToolPolicy). The mode is resolved at
/// the runtime layer on every spawn, so it cannot be bypassed by the LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChildPermissionMode {
    /// Child agents have full permission — every tool auto-approves.
    ///
    /// This matches the historical behaviour (children carried no tool policy),
    /// so it is the default to avoid surprising existing users.
    #[default]
    Full,
    /// Child agents have no permission — the parent's "dangerous tools" are
    /// denied when the parent has a [`ToolPolicy`](agent_base::ToolPolicy); if
    /// the parent has none, every tool is denied via [`DenyAllToolPolicy`](agent_base::DenyAllToolPolicy).
    None,
    /// The parent agent decides per-spawn via `SpawnAgentArgs.full_permission`.
    PerSpawn,
}

/// Configuration for the multi-agent subsystem.
///
/// # Default
///
/// Multi-agent is **enabled** by default with limits of 8 sub-agents and depth 1.
/// Disable it with:
///
/// ```rust,ignore
/// use agent_works::AgentBuilder;
///
/// let agent = AgentBuilder::new(client)
///     .without_multi_agent()
///     .build()?;
/// ```
///
/// # Limits
///
/// Limits only take effect when `enabled` is `true`.
#[derive(Clone, Debug)]
pub struct MultiAgentConfig {
    /// Enable multi-agent capabilities.
    ///
    /// When `true` (default): all 6 multi-agent tools are registered, and the main
    /// agent's system prompt is augmented with usage guidance.
    ///
    /// When `false`: no multi-agent tools are registered, and the system
    /// prompt does not mention multi-agent capabilities.
    pub enabled: bool,

    /// Maximum number of concurrent sub-agents.
    ///
    /// Default: 8. Only enforced when `enabled` is `true`.
    pub max_sub_agents: usize,

    /// Maximum nesting depth for sub-agents.
    ///
    /// Default: 1 (only direct children of root can spawn). A value of 0 means
    /// no sub-agents can be spawned at all. Only enforced when `enabled` is `true`.
    pub max_agent_depth: i32,

    /// Permission mode for spawned child agents.
    ///
    /// Default: [`ChildPermissionMode::Full`].
    pub child_permission_mode: ChildPermissionMode,

    /// Business tool names to EXCLUDE from child agents.
    ///
    /// By default children inherit every business tool registered on the parent.
    /// Some tools are inherently root-level (e.g. `decompose`/`merge` orchestration):
    /// a leaf agent that lacks `spawn_agent` should not be handed a "split this into
    /// parallel sub-agents" tool — it would plan work it cannot execute. Mark such
    /// tools here so `build_child_runtime` skips them, and children do the work
    /// inline instead of trying to orchestrate further.
    pub child_excluded_tools: Vec<String>,

    /// Optional reasoning-effort override applied to every child agent.
    ///
    /// Child agents do narrow, focused slices, so they rarely need the parent's
    /// full reasoning depth — and on reasoning-heavy models (deepseek-v4-pro) an
    /// unbounded child can "think" itself into a runaway (30KB+ of reasoning with
    /// no tool call or answer). Setting e.g. [`ReasoningEffort::Low`] caps that
    /// cost. `None` (default) leaves the child on the framework default.
    pub child_reasoning_effort: Option<ReasoningEffort>,

    /// Whether child agents are *recommended* to be read-only.
    ///
    /// The framework is domain-agnostic: it cannot classify which business tools
    /// mutate state and which merely read, so this is a **prompt-level suggestion
    /// only** — it does not remove or gate any tool. When `true` (default),
    /// [`MultiAgentRuntime::build_child_runtime`] appends a read-only nudge to
    /// every child's system prompt, telling it to investigate and report rather
    /// than mutate.
    ///
    /// For a *hard* guarantee, exclude mutating tools at the business layer via
    /// [`MultiAgentConfig::child_excluded_tools`]; the framework only *suggests*
    /// read-only, it cannot enforce it. Set this to `false` when children are
    /// meant to write (the codex-style symmetric model).
    pub child_read_only: bool,
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_sub_agents: 8,
            max_agent_depth: 1,
            child_permission_mode: ChildPermissionMode::Full,
            child_excluded_tools: Vec::new(),
            child_reasoning_effort: None,
            child_read_only: true,
        }
    }
}

impl MultiAgentConfig {
    /// Create a config with multi-agent enabled and default limits.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Create a config with multi-agent enabled and custom limits.
    pub fn with_limits(max_sub_agents: usize, max_agent_depth: i32) -> Self {
        Self {
            enabled: true,
            max_sub_agents,
            max_agent_depth,
            child_permission_mode: ChildPermissionMode::Full,
            child_excluded_tools: Vec::new(),
            child_reasoning_effort: None,
            child_read_only: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_enabled() {
        let config = MultiAgentConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_sub_agents, 8);
        assert_eq!(config.max_agent_depth, 1);
    }

    #[test]
    fn default_children_are_read_only() {
        assert!(MultiAgentConfig::default().child_read_only);
        assert!(MultiAgentConfig::enabled().child_read_only);
        assert!(MultiAgentConfig::with_limits(4, 2).child_read_only);
    }

    #[test]
    fn default_permission_mode_is_full() {
        assert_eq!(
            MultiAgentConfig::default().child_permission_mode,
            ChildPermissionMode::Full
        );
        assert_eq!(
            MultiAgentConfig::enabled().child_permission_mode,
            ChildPermissionMode::Full
        );
        assert_eq!(
            MultiAgentConfig::with_limits(4, 2).child_permission_mode,
            ChildPermissionMode::Full
        );
    }

    #[test]
    fn enabled_shortcut() {
        let config = MultiAgentConfig::enabled();
        assert!(config.enabled);
        assert_eq!(config.max_sub_agents, 8);
        assert_eq!(config.max_agent_depth, 1);
    }

    #[test]
    fn with_limits() {
        let config = MultiAgentConfig::with_limits(4, 2);
        assert!(config.enabled);
        assert_eq!(config.max_sub_agents, 4);
        assert_eq!(config.max_agent_depth, 2);
    }
}
