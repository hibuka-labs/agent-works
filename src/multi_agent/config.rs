//! Multi-agent configuration.
//!
//! Controls whether multi-agent capabilities are enabled and sets resource limits.
//! Multi-agent is enabled by default — users who don't need it can disable via
//! `.without_multi_agent()` or feature gate.

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
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_sub_agents: 8,
            max_agent_depth: 1,
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
