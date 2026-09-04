//! Agent lifecycle registry.
//!
//! The [`AgentRegistry`] tracks all active sub-agents, enforces spawn limits
//! (total count and depth), and manages the lifecycle state machine
//! (`idle → running → done`).

use std::collections::HashMap;
use std::time::Instant;

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
    /// Tool calls the agent has actually executed (monotonic). Surfaced by
    /// `list_agents` so the parent can distinguish "working" from "stalled":
    /// a live agent's count grows; a frozen count with a stale
    /// `last_activity` is the real stall signal.
    ///
    /// (This deliberately replaced the former static `tool_count` — the size
    /// of the child's tool *inventory*, fixed at spawn — which the parent
    /// misread as progress: "stuck at 9 tool calls".)
    pub tool_calls: usize,
    /// Last observed activity (task start or tool call). `None` until the
    /// agent receives its first task.
    pub last_activity: Option<Instant>,
    /// The task the agent is currently assigned (first `send_task` wins).
    /// Recorded so `list_agents` can show *what* each agent is doing, not
    /// just its lifecycle status.
    pub task: Option<String>,
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
///   → set_status(Running) on send_message(trigger=true)
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
    pub fn register(&mut self, path: &AgentPath, depth: i32) -> Result<(), SpawnError> {
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
                tool_calls: 0,
                last_activity: None,
                task: None,
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

    /// Record the agent's assigned task (first write wins).
    ///
    /// `list_agents` surfaces this so the parent — and the user asking
    /// "what is that agent doing?" — can tell agents apart by *what* they
    /// were asked to do, not just by name and lifecycle status.
    pub fn set_task(&mut self, path: &AgentPath, task: String) -> bool {
        match self.agents.get_mut(path) {
            Some(entry) => {
                if entry.task.is_none() {
                    entry.task = Some(task);
                }
                true
            }
            None => false,
        }
    }

    /// Mark the agent as active (liveness heartbeat) without counting a tool
    /// call — used on task start, where there is work but no tool call yet.
    ///
    /// Returns `true` if the agent was found and updated.
    pub fn touch(&mut self, path: &AgentPath) -> bool {
        match self.agents.get_mut(path) {
            Some(entry) => {
                entry.last_activity = Some(Instant::now());
                true
            }
            None => false,
        }
    }

    /// Record one executed tool call: bumps the monotonic counter and the
    /// activity timestamp.
    ///
    /// Called from the child event bridge on every `ToolCallStarted`, so
    /// `list_agents` can show real progress (see [`AgentEntry::tool_calls`]).
    /// Returns `true` if the agent was found and updated.
    pub fn record_tool_call(&mut self, path: &AgentPath) -> bool {
        match self.agents.get_mut(path) {
            Some(entry) => {
                entry.tool_calls += 1;
                entry.last_activity = Some(Instant::now());
                true
            }
            None => false,
        }
    }

    /// Number of agents currently executing a task (`Running`).
    ///
    /// The fan-in coordinator uses this as its quiescence signal: a batch of
    /// results is complete when nothing is `Running` anymore.
    pub fn running_count(&self) -> usize {
        self.count_by_status(&AgentStatus::Running)
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

        assert!(reg.register(&path, 1).is_ok());
        assert_eq!(reg.count(), 1);
        assert!(reg.contains(&path));

        let entry = reg.get(&path).unwrap();
        assert_eq!(entry.status, AgentStatus::Idle);
        assert_eq!(entry.depth, 1);
        // Fresh agent: no tool calls executed, no activity yet.
        assert_eq!(entry.tool_calls, 0);
        assert!(entry.last_activity.is_none());

        let closed = reg.close(&path).unwrap();
        assert_eq!(closed.path, path);
        assert_eq!(reg.count(), 0);
        assert!(!reg.contains(&path));
    }

    #[test]
    fn duplicate_register_fails() {
        let mut reg = AgentRegistry::new(test_config());
        let path = test_path("worker");

        assert!(reg.register(&path, 1).is_ok());
        assert_eq!(
            reg.register(&path, 1).unwrap_err(),
            SpawnError::AlreadyExists
        );
    }

    #[test]
    fn max_agents_limit() {
        let config = MultiAgentConfig::with_limits(2, 1);
        let mut reg = AgentRegistry::new(config);

        assert!(reg.register(&test_path("a"), 1).is_ok());
        assert!(reg.register(&test_path("b"), 1).is_ok());
        assert_eq!(
            reg.register(&test_path("c"), 1).unwrap_err(),
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

        reg.register(&path, 1).unwrap();
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
        reg.register(&path, 1).unwrap();
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
        reg.register(&test_path("b"), 1).unwrap();
        reg.register(&test_path("a"), 1).unwrap();

        let list = reg.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].path.name(), "a");
        assert_eq!(list[1].path.name(), "b");
    }

    #[test]
    fn tool_call_counter_is_monotonic() {
        // Session 20260903_0cf95e79 regression: list_agents showed a frozen
        // `tool_count: 9` (static inventory size) while the children actually
        // executed 100+ calls — the parent read it as "stuck". The executed
        // count must start at zero and only grow.
        let mut reg = AgentRegistry::new(test_config());
        let path = test_path("worker");
        reg.register(&path, 1).unwrap();

        for _ in 0..3 {
            assert!(reg.record_tool_call(&path));
        }
        assert_eq!(reg.get(&path).unwrap().tool_calls, 3);
        assert!(reg.get(&path).unwrap().last_activity.is_some());

        assert!(reg.record_tool_call(&path));
        assert_eq!(reg.get(&path).unwrap().tool_calls, 4);

        // Unknown paths are a no-op (event bridge may race cleanup).
        assert!(!reg.record_tool_call(&test_path("ghost")));
    }

    #[test]
    fn touch_marks_activity_without_counting() {
        let mut reg = AgentRegistry::new(test_config());
        let path = test_path("worker");
        reg.register(&path, 1).unwrap();

        assert!(reg.touch(&path));
        assert!(reg.get(&path).unwrap().last_activity.is_some());
        assert_eq!(reg.get(&path).unwrap().tool_calls, 0);
        assert!(!reg.touch(&test_path("ghost")));
    }

    #[test]
    fn running_count_tracks_lifecycle() {
        // Fan-in quiescence signal: the batch is complete when nothing is
        // Running anymore (Done children awaiting close must not block it).
        let mut reg = AgentRegistry::new(test_config());
        let a = test_path("a");
        let b = test_path("b");
        reg.register(&a, 1).unwrap();
        reg.register(&b, 1).unwrap();
        assert_eq!(reg.running_count(), 0);

        reg.set_status(&a, AgentStatus::Running);
        reg.set_status(&b, AgentStatus::Running);
        assert_eq!(reg.running_count(), 2);

        reg.set_status(&a, AgentStatus::Done);
        assert_eq!(reg.running_count(), 1);

        reg.set_status(&b, AgentStatus::Done);
        assert_eq!(reg.running_count(), 0);
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
        let config = MultiAgentConfig {
            enabled: false,
            ..MultiAgentConfig::default()
        };
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
