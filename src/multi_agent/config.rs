//! Multi-agent configuration.
//!
//! Controls whether multi-agent capabilities are enabled and sets resource limits.
//! Multi-agent is enabled by default — users who don't need it can disable via
//! `.without_multi_agent()` or feature gate.

use std::time::Duration;

use agent_base::ReasoningEffort;

/// Deployment autonomy mode (design doc §7.5, v3.2).
///
/// A **deployment-level** bit — it is deliberately *not* in `ChildConfig`:
/// the LLM must not choose its own autonomy level (same attack-surface logic
/// as B4 removing `depth`/`full_permission`). `Manual` expands at the spawn
/// gate into the three existing layers (§9.3) — hard exclusion, approval
/// floor, read-only nudge — and never into a fourth permission mechanism.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AgentAutonomy {
    /// Children may write: mutating tools are available per the normal
    /// layers, and writes follow `ChildPermissionMode` (≈ today's
    /// `enabled()` defaults). The default — behaviour unchanged.
    #[default]
    Auto,
    /// Children are hard read-only: [`ControlConfig::write_tools`] is merged
    /// into the exclusion set for every child, effective permission is
    /// forced to `None`-mode semantics (policy with the parent, approvals up
    /// to the parent handler, DenyAll otherwise), and the read-only nudge is
    /// forced on. All mutations belong to the parent (human-in-the-loop at
    /// the parent's approval layer).
    Manual,
}

/// Session-level control-plane configuration (see design doc §7.4 / §7.5).
///
/// Every budget/limit knob is `Option`: `None` means "not enabled", which
/// keeps behaviour byte-identical to the pre-multi-agent-API runtime.
/// Defaults are all `None` by design — the control plane never silently
/// tightens a deployment.
///
/// Note on overlap with [`MultiAgentConfig::max_sub_agents`] (§7.3): both the
/// registry gate and `max_concurrency` count *live* (spawned-but-not-closed)
/// children. `max_sub_agents` stays the primary count/depth gate;
/// `max_concurrency` is a separate session-level knob for temporarily
/// tightening concurrency without touching deployment config.
#[derive(Clone, Debug)]
pub struct ControlConfig {
    /// Token budget for **child** agents (the parent's own usage is not
    /// metered). `None` = unlimited (current behaviour).
    pub child_max_tokens: Option<u64>,

    /// Cumulative spawn count allowed in a rollout. `None` = only bounded by
    /// `max_sub_agents` (live count), a different dimension (§7.2).
    pub max_spawns: Option<usize>,

    /// Live-concurrency cap (children alive, not tasks executing).
    /// `None` = gate not enabled (§7.3).
    pub max_concurrency: Option<usize>,

    /// Per-task execution timeout for a child's `run_turn` (§9.2).
    /// `None` = no timeout.
    pub task_timeout: Option<Duration>,

    /// Deployment autonomy mode (§7.5). `Auto` (default) keeps the layers
    /// independent, exactly as today; `Manual` hard-read-only-izes every
    /// child at the spawn gate.
    pub autonomy: AgentAutonomy,

    /// The mutating-tool set that `Manual` mode excludes from every child.
    ///
    /// Read/write classification is **deployment knowledge** (§9.3 killed v2's
    /// `PermissionChecker` for guessing tool names — guessing is a hole,
    /// declaring is a policy). The default covers the phi-kernel-tools write
    /// set (`write_file` / `edit_file` / `execute_command`); deployments with
    /// custom mutating tools add them here. In `Auto` mode this list is
    /// unused.
    pub write_tools: Vec<String>,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            child_max_tokens: None,
            max_spawns: None,
            max_concurrency: None,
            task_timeout: None,
            autonomy: AgentAutonomy::default(),
            write_tools: super::preset::default_write_tools(),
        }
    }
}

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
    /// When `true` (default): the multi-agent tools are registered (via the
    /// tool factory — 5 with the reference factory, `followup_task` deprecated
    /// per §8.3), and the main agent's system prompt is augmented with usage
    /// guidance.
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
    /// `build_child_runtime` appends a read-only nudge to
    /// every child's system prompt, telling it to investigate and report rather
    /// than mutate.
    ///
    /// For a *hard* guarantee, exclude mutating tools at the business layer via
    /// [`MultiAgentConfig::child_excluded_tools`]; the framework only *suggests*
    /// read-only, it cannot enforce it. Set this to `false` when children are
    /// meant to write (the codex-style symmetric model).
    pub child_read_only: bool,

    /// Control-plane limits (token budget, cumulative spawns, live
    /// concurrency, per-task timeout). All `None` by default — behaviour
    /// unchanged unless a knob is explicitly set (see [`ControlConfig`]).
    pub control: ControlConfig,
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
            control: ControlConfig::default(),
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
            control: ControlConfig::default(),
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

    #[test]
    fn control_defaults_to_all_none() {
        let c = ControlConfig::default();
        assert!(c.child_max_tokens.is_none());
        assert!(c.max_spawns.is_none());
        assert!(c.max_concurrency.is_none());
        assert!(c.task_timeout.is_none());
        // MultiAgentConfig default carries all-None control → behaviour
        // unchanged (design doc §7.4).
        assert!(
            MultiAgentConfig::default()
                .control
                .max_concurrency
                .is_none()
        );
    }

    #[test]
    fn multi_agent_config_carries_control() {
        let config = MultiAgentConfig {
            control: ControlConfig {
                max_concurrency: Some(3),
                ..Default::default()
            },
            ..MultiAgentConfig::enabled()
        };
        assert_eq!(config.control.max_concurrency, Some(3));
    }
}
