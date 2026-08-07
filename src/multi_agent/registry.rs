//! Agent lifecycle registry.
//!
//! The [`AgentRegistry`] tracks all active sub-agents, enforces spawn limits
//! (total count and depth), and manages the lifecycle state machine
//! (`idle → running → done`).

use std::collections::HashMap;

use super::config::MultiAgentConfig;
use super::path::AgentPath;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Lifecycle status of a sub-agent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentStatus {
    /// Agent is spawned but has not yet received a task.
    Idle,
    /// Agent is currently executing a task.
    Running,
    /// Agent has completed its task and is idle, waiting for a new task or close.
    Done,
}

/// A registered agent entry in the registry.
#[derive(Clone, Debug)]
pub struct AgentEntry {
    /// The agent's tree path.
    pub path: AgentPath,
    /// Current lifecycle status.
    pub status: AgentStatus,
    /// Depth from root (root=0, direct child=1, etc.).
    pub depth: i32,
    /// Number of tools registered on this agent (for `list_agents` output).
    pub tool_count: usize,
}

/// Errors that can occur during spawn attempts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpawnError {
    /// Spawning would exceed the maximum number of sub-agents.
    MaxAgentsReached { max: usize },
    /// Spawning would exceed the maximum agent nesting depth.
    DepthLimitReached { max: i32, attempted: i32 },
    /// An agent with this path already exists.
    AlreadyExists,
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxAgentsReached { max } => {
                write!(f, "max agents reached (limit: {})", max)
            }
            Self::DepthLimitReached { max, attempted } => {
                write!(
                    f,
                    "agent depth limit reached (max: {}, attempted: {})",
                    max, attempted
                )
            }
            Self::AlreadyExists => {
                write!(f, "agent with this path already exists")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AgentRegistry
// ---------------------------------------------------------------------------

/// Tracks active sub-agents and enforces spawn limits.
///
/// # Lifecycle
///
/// ```text
/// register() → status=Idle
///   → set_status(Running) on followup_task
///     → set_status(Done) on completion
///       → close() removes from registry
/// ```
///
/// Done agents still count toward the quota until explicitly `close()`d.
/// There is no automatic garbage collection in v1.
pub struct AgentRegistry {
    config: MultiAgentConfig,
    agents: HashMap<AgentPath, AgentEntry>,
}

impl AgentRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: MultiAgentConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
        }
    }

    /// Check whether a new agent can be spawned at the given depth.
    ///
    /// Returns `Ok(())` if spawning is allowed, or a [`SpawnError`] describing
    /// which limit was exceeded.
    pub fn can_spawn(&self, depth: i32) -> Result<(), SpawnError> {
        // Check total count limit
        if self.config.enabled && self.agents.len() >= self.config.max_sub_agents {
            return Err(SpawnError::MaxAgentsReached {
                max: self.config.max_sub_agents,
            });
        }

        // Check depth limit
        if self.config.enabled && depth > self.config.max_agent_depth {
            return Err(SpawnError::DepthLimitReached {
                max: self.config.max_agent_depth,
                attempted: depth,
            });
        }

        Ok(())
    }

    /// Register a new agent.
    ///
    /// Returns `Ok(())` on success, or a [`SpawnError`] if limits are exceeded
    /// or the path already exists.
    pub fn register(
        &mut self,
        path: &AgentPath,
        depth: i32,
        tool_count: usize,
    ) -> Result<(), SpawnError> {
        self.can_spawn(depth)?;

        if self.agents.contains_key(path) {
            return Err(SpawnError::AlreadyExists);
        }

        self.agents.insert(
            path.clone(),
            AgentEntry {
                path: path.clone(),
                status: AgentStatus::Idle,
                depth,
                tool_count,
            },
        );

        Ok(())
    }

    /// Close (remove) an agent from the registry.
    ///
    /// Returns the removed entry, or `None` if the agent was not registered.
    /// This releases the agent's quota slot.
    pub fn close(&mut self, path: &AgentPath) -> Option<AgentEntry> {
        self.agents.remove(path)
    }

    /// Update the lifecycle status of an agent.
    ///
    /// Returns `true` if the agent was found and updated.
    pub fn set_status(&mut self, path: &AgentPath, status: AgentStatus) -> bool {
        match self.agents.get_mut(path) {
            Some(entry) => {
                entry.status = status;
                true
            }
            None => false,
        }
    }

    /// Get an agent entry by path.
    pub fn get(&self, path: &AgentPath) -> Option<&AgentEntry> {
        self.agents.get(path)
    }

    /// List all registered agents (all statuses).
    pub fn list(&self) -> Vec<&AgentEntry> {
        let mut entries: Vec<&AgentEntry> = self.agents.values().collect();
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        entries
    }

    /// Return the total number of registered agents.
    pub fn count(&self) -> usize {
        self.agents.len()
    }

    /// Return the number of agents with a specific status.
    pub fn count_by_status(&self, status: &AgentStatus) -> usize {
        self.agents.values().filter(|e| e.status == *status).count()
    }

    /// Return whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Check if an agent path is registered.
    pub fn contains(&self, path: &AgentPath) -> bool {
        self.agents.contains_key(path)
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &MultiAgentConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MultiAgentConfig {
        MultiAgentConfig::enabled()
    }

    fn test_path(name: &str) -> AgentPath {
        AgentPath::root().join(name)
    }

    #[test]
    fn register_and_close() {
        let mut reg = AgentRegistry::new(test_config());
        let path = test_path("worker");

        assert!(reg.register(&path, 1, 5).is_ok());
        assert_eq!(reg.count(), 1);
        assert!(reg.contains(&path));

        let entry = reg.get(&path).unwrap();
        assert_eq!(entry.status, AgentStatus::Idle);
        assert_eq!(entry.depth, 1);
        assert_eq!(entry.tool_count, 5);

        let closed = reg.close(&path).unwrap();
        assert_eq!(closed.path, path);
        assert_eq!(reg.count(), 0);
        assert!(!reg.contains(&path));
    }

    #[test]
    fn duplicate_register_fails() {
        let mut reg = AgentRegistry::new(test_config());
        let path = test_path("worker");

        assert!(reg.register(&path, 1, 3).is_ok());
        assert_eq!(
            reg.register(&path, 1, 3).unwrap_err(),
            SpawnError::AlreadyExists
        );
    }

    #[test]
    fn max_agents_limit() {
        let config = MultiAgentConfig::with_limits(2, 1);
        let mut reg = AgentRegistry::new(config);

        assert!(reg.register(&test_path("a"), 1, 1).is_ok());
        assert!(reg.register(&test_path("b"), 1, 1).is_ok());
        assert_eq!(
            reg.register(&test_path("c"), 1, 1).unwrap_err(),
            SpawnError::MaxAgentsReached { max: 2 }
        );
    }

    #[test]
    fn depth_limit() {
        let config = MultiAgentConfig::with_limits(8, 1);
        let reg = AgentRegistry::new(config);

        // Depth 1 is allowed
        assert!(reg.can_spawn(1).is_ok());

        // Depth 2 exceeds limit
        assert_eq!(
            reg.can_spawn(2).unwrap_err(),
            SpawnError::DepthLimitReached {
                max: 1,
                attempted: 2
            }
        );
    }

    #[test]
    fn lifecycle_states() {
        let mut reg = AgentRegistry::new(test_config());
        let path = test_path("worker");

        reg.register(&path, 1, 3).unwrap();
        assert_eq!(reg.count_by_status(&AgentStatus::Idle), 1);

        reg.set_status(&path, AgentStatus::Running);
        assert_eq!(reg.count_by_status(&AgentStatus::Idle), 0);
        assert_eq!(reg.count_by_status(&AgentStatus::Running), 1);
        assert_eq!(reg.get(&path).unwrap().status, AgentStatus::Running);

        reg.set_status(&path, AgentStatus::Done);
        assert_eq!(reg.count_by_status(&AgentStatus::Running), 0);
        assert_eq!(reg.count_by_status(&AgentStatus::Done), 1);

        // Done agent still counts toward quota
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn close_frees_quota() {
        let config = MultiAgentConfig::with_limits(1, 1);
        let mut reg = AgentRegistry::new(config);

        let path = test_path("worker");
        reg.register(&path, 1, 3).unwrap();
        assert_eq!(reg.count(), 1);

        // Can't spawn another — quota full
        assert!(reg.can_spawn(1).is_err());

        // Close frees the slot
        reg.close(&path);
        assert_eq!(reg.count(), 0);
        assert!(reg.can_spawn(1).is_ok());
    }

    #[test]
    fn list_sorted() {
        let mut reg = AgentRegistry::new(test_config());
        reg.register(&test_path("b"), 1, 1).unwrap();
        reg.register(&test_path("a"), 1, 1).unwrap();

        let list = reg.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].path.name(), "a");
        assert_eq!(list[1].path.name(), "b");
    }

    #[test]
    fn set_status_nonexistent() {
        let mut reg = AgentRegistry::new(test_config());
        assert!(!reg.set_status(&test_path("ghost"), AgentStatus::Running));
    }

    #[test]
    fn disabled_config_allows_spawn() {
        // When config.enabled is false, limits are NOT checked
        // (spawning still works, it's just that the 6 tools aren't registered)
        let config = MultiAgentConfig::default(); // enabled=false
        let reg = AgentRegistry::new(config);

        // can_spawn still returns Ok even with 0 sub_agents allowed
        // because limits are only enforced when enabled=true
        assert!(reg.can_spawn(999).is_ok());
    }

    #[test]
    fn spawn_error_display() {
        assert_eq!(
            SpawnError::MaxAgentsReached { max: 8 }.to_string(),
            "max agents reached (limit: 8)"
        );
        assert_eq!(
            SpawnError::DepthLimitReached {
                max: 1,
                attempted: 2
            }
            .to_string(),
            "agent depth limit reached (max: 1, attempted: 2)"
        );
        assert_eq!(
            SpawnError::AlreadyExists.to_string(),
            "agent with this path already exists"
        );
    }
}
