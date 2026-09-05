//! Agent lifecycle registry — the single source of truth for child state.
//!
//! Session 20260904_c6559510 redesign: an agent's status is **derived from
//! observable facts**, never stored as a marker. The old `set_status` protocol
//! made every caller responsible for marking state at the right moment; one
//! forgotten call produced a phantom-idle registry and the watcher fired the
//! fan-in batch early (7 reports → 53,276 tokens of re-synthesis). Under the
//! fact model there is nothing to forget: the two facts below are maintained
//! at exactly the two places the runtime structure forces work through.
//!
//! # Facts (per agent)
//!
//! ```text
//! queue_len      : tasks enqueued but not yet dequeued (send_task / dequeue)
//! in_flight      : a dequeued task is executing and its result is unposted
//! results_posted : monotonic count of results this agent has delivered
//! results_handed_over : results included in a fired fan-in batch
//! ```
//!
//! `results_posted` vs `results_handed_over` is the **delivery gap**: how
//! many of an agent's reports exist but have not reached the parent yet.
//! Session 20260904_841ed65b: a mid-turn parent saw `done` children but no
//! reports (Progress never wakes the parent; the batch flushes at turn end)
//! and concluded "the system didn't deliver" — then re-did the work itself.
//! Surfacing the gap through `list_agents` turns that blind faith into a
//! checkable fact.
//!
//! | Fact change      | Maintained by       | Timing                          |
//! |------------------|---------------------|---------------------------------|
//! | `queue_len += 1` | `note_enqueued`     | when `send_task` queues a task  |
//! | `queue_len -= 1` | `note_dequeued`     | when the child loop dequeues    |
//! | `in_flight=true` | `note_dequeued`     | same dequeue                    |
//! | `in_flight=false`| `note_posted`       | **before** `post_result` bumps the seq — the watcher wakes on that bump and must see the settled facts |
//! | `results_posted+=1`| `note_posted`     | same delivery                   |
//!
//! # Derived status
//!
//! [`AgentStatus`] is a pure function of the facts — see
//! [`derive_status`]. There is deliberately no `Idle`: a spawned-but-never-
//! tasked agent has the same facts (`queue_len==0 && !in_flight`) as a
//! finished one, and forcing the model to distinguish the two only breeds
//! polling.
//!
//! # Quiescence (fan-in)
//!
//! [`AgentRegistry::quiescent`] is the watcher's batch predicate: nobody is
//! working (`!in_flight && queue_len==0` everywhere) **and** every registered
//! agent has delivered at least once. The delivery clause seals the
//! spawn→send window: a freshly spawned agent (registered, zero deliveries)
//! cannot satisfy quiescence, so a batch can never fire while a sibling's
//! `send_task` is still in flight.
//!
//! # Lifecycle broadcast
//!
//! Every fact-driven transition emits an [`AgentLifecycleEvent`] (ring buffer,
//! [`AgentRegistry::recent_events`]) and republishes a [`RegistrySnapshot`]
//! through a `tokio::sync::watch` channel ([`AgentRegistry::subscribe`]).
//! Consumers (UI panel, `list_agents`, logging) read "what is each agent's
//! status *now*" from the snapshot; the event ring is for diagnostics.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::watch;

use super::config::MultiAgentConfig;
use super::path::AgentPath;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Lifecycle status of a sub-agent — **derived**, never stored.
///
/// See [`derive_status`] for the derivation and the registry module docs for
/// why no variant carries a marker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentStatus {
    /// Task(s) queued but nothing executing (before the first dequeue, or
    /// between posts while another task waits).
    Queued,
    /// Executing a dequeued task (from dequeue to `note_posted`).
    Running,
    /// Queue empty, nothing in flight — results delivered, waiting for new
    /// tasks or close. A freshly spawned agent also lands here (its facts are
    /// indistinguishable until the first `send_task`).
    Done,
    /// Unregistered (closed or cleaned up).
    Closed,
    /// Reserved for Phase 6+ (user/guard pause). Not produced yet.
    Paused,
}

impl AgentStatus {
    /// Lowercase name used on the wire (`list_agents`, close results).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Done => "done",
            Self::Closed => "closed",
            Self::Paused => "paused",
        }
    }
}

/// Derive a sub-agent's status from the observable facts (pure function).
///
/// ```text
/// (unregistered)            → Closed
/// in_flight                 → Running   (queued siblings don't change this)
/// queue_len == 0            → Done      (delivered and nothing pending)
/// else                      → Queued    (work waiting, nothing executing)
/// ```
pub fn derive_status(registered: bool, in_flight: bool, queue_len: usize) -> AgentStatus {
    match (registered, in_flight, queue_len) {
        (false, _, _) => AgentStatus::Closed,
        (_, true, _) => AgentStatus::Running,
        (_, false, 0) => AgentStatus::Done,
        (_, false, _) => AgentStatus::Queued,
    }
}

/// A registered agent entry in the registry.
///
/// The fields below `results_posted` are the facts; `status()` derives the
/// lifecycle from them. `tool_calls` / `last_activity` remain plain metrics
/// (stall diagnostics) and play no part in the derivation.
#[derive(Clone, Debug)]
pub struct AgentEntry {
    /// The agent's tree path.
    pub path: AgentPath,
    /// Depth from root (root=0, direct child=1, etc.).
    pub depth: i32,
    /// Tasks enqueued but not yet dequeued.
    pub queue_len: usize,
    /// A dequeued task is executing and its result is unposted.
    pub in_flight: bool,
    /// When the current `Running` period began (the last dequeue). `None`
    /// outside `Running`; feeds stall detection (Phase 6 reaper).
    pub running_since: Option<Instant>,
    /// Monotonic count of results this agent has delivered. Drives the
    /// fan-in delivery-completeness clause — a registered agent with zero
    /// deliveries (spawn→send window) can never be part of a settled batch.
    pub results_posted: usize,
    /// Results already included in a fired batch (see field docs).
    pub results_handed_over: usize,
    /// Results of this agent that have been included in a fired fan-in
    /// batch (the watcher stamps every batch member when it drains). The
    /// delivery gap `results_posted - results_handed_over` is what
    /// `list_agents` surfaces as `pending_results`: reports that exist but
    /// have not reached the parent yet.
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

impl AgentEntry {
    /// Derived lifecycle status (this entry is by definition registered).
    pub fn status(&self) -> AgentStatus {
        derive_status(true, self.in_flight, self.queue_len)
    }
}

/// One status transition, emitted when a fact change moves an agent between
/// derived states. The event stream is diagnostics only — state consumers
/// read the [`RegistrySnapshot`].
#[derive(Clone, Debug)]
pub struct AgentLifecycleEvent {
    /// The agent's tree path.
    pub path: AgentPath,
    /// Previously derived status.
    pub from: AgentStatus,
    /// Newly derived status.
    pub to: AgentStatus,
    /// Human-readable cause: `"task_enqueued"` / `"task_dequeued"` /
    /// `"result_posted"` / `"unregistered"`.
    pub reason: &'static str,
}

/// Point-in-time view of every registered agent, published through the
/// registry's `watch` channel on each transition. This is what read-only
/// consumers (UI panel, `list_agents`) subscribe to.
#[derive(Clone, Debug, Default)]
pub struct RegistrySnapshot {
    /// One entry per registered agent, sorted by path.
    pub agents: Vec<AgentSnapshot>,
}

/// Per-agent row of a [`RegistrySnapshot`].
#[derive(Clone, Debug)]
pub struct AgentSnapshot {
    /// Agent path as a string (e.g. `"root/worker"`).
    pub path: String,
    /// Derived status name (`queued` / `running` / `done`; `paused` is
    /// reserved and never produced yet). `closed` cannot appear here —
    /// snapshot rows exist only for registered agents; unregistration shows
    /// up as the agent leaving the snapshot.
    pub status: String,
    /// Seconds spent in the current `Running` period; `None` unless running.
    pub running_secs: Option<u64>,
    /// Seconds since the agent's last observed activity (metrics — task
    /// start or tool call; `None` until the first task).
    pub last_activity_secs: Option<u64>,
    /// Executed tool calls (monotonic) — metrics, not lifecycle.
    pub tool_calls: usize,
    /// What the agent was asked to do (first task wins).
    pub task: Option<String>,
    /// Reports that exist but have not been included in a fired batch yet
    /// (`results_posted - results_handed_over`). Zero is the common case.
    pub pending_results: usize,
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
/// # Lifecycle (derived, not marked)
///
/// ```text
/// register()                    → facts (0,false)   → Done
/// note_enqueued  (send_task)    → queue_len += 1    → Queued
/// note_dequeued  (child loop)   → -1, in_flight     → Running
/// note_posted    (child loop)   → in_flight=false   → Done (or Queued if more queued)
/// close()                       → entry removed     → Closed (event ring only —
///                                                    never a snapshot row)
/// ```
///
/// Done agents still count toward the quota until explicitly `close()`d.
/// There is no automatic garbage collection in v1.
pub struct AgentRegistry {
    config: MultiAgentConfig,
    agents: HashMap<AgentPath, AgentEntry>,
    /// Lifecycle event ring (diagnostics; oldest dropped at capacity).
    events: VecDeque<AgentLifecycleEvent>,
    /// Watch channel carrying the latest [`RegistrySnapshot`].
    snapshot_tx: watch::Sender<Arc<RegistrySnapshot>>,
}

/// Ring-buffer capacity for lifecycle events. Same order of magnitude as
/// agent-base's event bus default (2048, inline in `builder.rs` — the value
/// is duplicated here, not shared).
const EVENT_RING_CAP: usize = 2048;

impl AgentRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: MultiAgentConfig) -> Self {
        let (snapshot_tx, _) = watch::channel(Arc::new(RegistrySnapshot::default()));
        Self {
            config,
            agents: HashMap::new(),
            events: VecDeque::new(),
            snapshot_tx,
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
                depth,
                queue_len: 0,
                in_flight: false,
                running_since: None,
                results_posted: 0,
                results_handed_over: 0,
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
    /// This releases the agent's quota slot. Emits an `unregistered`
    /// transition (→ `Closed`) and republishes the snapshot without the agent.
    pub fn close(&mut self, path: &AgentPath) -> Option<AgentEntry> {
        let entry = self.agents.remove(path)?;
        self.push_event(AgentLifecycleEvent {
            path: path.clone(),
            from: entry.status(),
            to: AgentStatus::Closed,
            reason: "unregistered",
        });
        self.publish_snapshot();
        Some(entry)
    }

    /// Record a fact change and emit a transition event if the derived status
    /// moved. Shared body of the three `note_*` methods.
    fn transition(
        &mut self,
        path: &AgentPath,
        reason: &'static str,
        apply: impl FnOnce(&mut AgentEntry),
    ) -> bool {
        let Some(entry) = self.agents.get_mut(path) else {
            return false;
        };
        let from = entry.status();
        apply(entry);
        let to = entry.status();
        if from != to {
            self.push_event(AgentLifecycleEvent {
                path: path.clone(),
                from,
                to,
                reason,
            });
            self.publish_snapshot();
        }
        true
    }

    /// Record that a task was enqueued for the agent (fact: `queue_len += 1`).
    ///
    /// Called by `send_task` after the mailbox accepted the task. Returns
    /// `true` if the agent was found.
    pub fn note_enqueued(&mut self, path: &AgentPath) -> bool {
        self.transition(path, "task_enqueued", |e| {
            e.queue_len += 1;
        })
    }

    /// Record that the child loop dequeued a task (facts: `queue_len -= 1`,
    /// `in_flight = true`, `running_since = now`).
    ///
    /// Returns `true` if the agent was found.
    pub fn note_dequeued(&mut self, path: &AgentPath) -> bool {
        self.transition(path, "task_dequeued", |e| {
            e.queue_len = e.queue_len.saturating_sub(1);
            e.in_flight = true;
            e.running_since = Some(Instant::now());
        })
    }

    /// Record that the child loop delivered a task's result (facts:
    /// `in_flight = false`, `results_posted += 1`, `running_since = None`).
    ///
    /// Must be called **before** `post_result` bumps the mailbox sequence:
    /// the watcher wakes on that bump and must observe the settled facts —
    /// this ordering is what makes "result drained ⇒ producer quiescent"
    /// hold structurally instead of by caller discipline.
    ///
    /// Returns `true` if the agent was found.
    pub fn note_posted(&mut self, path: &AgentPath) -> bool {
        self.transition(path, "result_posted", |e| {
            e.in_flight = false;
            e.results_posted += 1;
            e.running_since = None;
        })
    }

    /// Roll back a `note_enqueued` whose `send_task` subsequently failed
    /// (mailbox entry vanished, or the task channel rejected the task). The
    /// enqueue fact is always recorded **before** the task is sent, so a
    /// failed send must give the count back — otherwise a phantom
    /// `queue_len` would block quiescence forever (nothing will ever dequeue
    /// or deliver it).
    ///
    /// Returns `true` if the agent was found.
    pub fn note_send_failed(&mut self, path: &AgentPath) -> bool {
        self.transition(path, "task_send_failed", |e| {
            e.queue_len = e.queue_len.saturating_sub(1);
        })
    }

    /// Record that every result this agent has posted so far was included in
    /// a fired fan-in batch (the watcher stamps each batch member at drain
    /// time). This closes the agent's delivery gap: `pending_results` drops
    /// to zero until it posts again (e.g. a follow-up task's answer).
    ///
    /// Deliberately not a `transition`: the derived status does not move, so
    /// no lifecycle event is emitted — but the snapshot is republished so
    /// watch consumers see the delivery gap close promptly.
    ///
    /// Returns `true` if the agent was found (closed agents are a no-op:
    /// their entries are gone, and their pending reports left with them).
    pub fn note_batch_handed_over(&mut self, path: &AgentPath) -> bool {
        let Some(entry) = self.agents.get_mut(path) else {
            return false;
        };
        entry.results_handed_over = entry.results_posted;
        self.publish_snapshot();
        true
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

    /// Number of agents with work outstanding (executing or queued).
    ///
    /// The fan-in coordinator does not use counts — it uses
    /// [`Self::quiescent`] — but operators and tests still want the number.
    pub fn busy_count(&self) -> usize {
        self.agents
            .values()
            .filter(|e| e.in_flight || e.queue_len > 0)
            .count()
    }

    /// Fan-in quiescence predicate — may a pending batch fire now?
    ///
    /// All three clauses must hold:
    /// 1. nobody is working: `!in_flight && queue_len == 0` for every agent;
    /// 2. every registered agent has delivered ≥1 result (delivery
    ///    completeness — structurally seals the spawn→send window, where a
    ///    freshly spawned agent would otherwise derive `Done` and let a
    ///    sibling's result fire a batch that excludes it);
    /// 3. (caller side) the batch itself is non-empty.
    pub fn quiescent(&self) -> bool {
        self.agents
            .values()
            .all(|e| !e.in_flight && e.queue_len == 0 && e.results_posted >= 1)
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

    /// Return whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Check if an agent path is registered.
    pub fn contains(&self, path: &AgentPath) -> bool {
        self.agents.contains_key(path)
    }

    /// Build a point-in-time [`RegistrySnapshot`] (sorted by path).
    pub fn snapshot(&self) -> RegistrySnapshot {
        RegistrySnapshot {
            agents: self
                .list()
                .into_iter()
                .map(|e| AgentSnapshot {
                    path: e.path.to_string(),
                    status: e.status().name().to_string(),
                    running_secs: if e.status() == AgentStatus::Running {
                        e.running_since.map(|t| t.elapsed().as_secs())
                    } else {
                        None
                    },
                    last_activity_secs: e.last_activity.map(|t| t.elapsed().as_secs()),
                    tool_calls: e.tool_calls,
                    task: e.task.clone(),
                    pending_results: e.results_posted.saturating_sub(e.results_handed_over),
                })
                .collect(),
        }
    }

    /// Subscribe to snapshot updates (watch channel; carries the latest
    /// snapshot only — consumers want "now", not a replay).
    ///
    /// Republishes happen on derived-status **transitions** (and close) only.
    /// Metric fields (`tool_calls`, `last_activity_secs`, `running_secs`,
    /// `task`) are therefore transition-frozen between events — a UI that
    /// wants live metrics must poll `list_agents`, and should derive only
    /// status from this stream.
    pub fn subscribe(&self) -> watch::Receiver<Arc<RegistrySnapshot>> {
        self.snapshot_tx.subscribe()
    }

    /// Borrow the most recent snapshot without cloning.
    pub fn latest_snapshot(&self) -> Arc<RegistrySnapshot> {
        self.snapshot_tx.borrow().clone()
    }

    /// The last `max` lifecycle events, oldest first (diagnostics).
    pub fn recent_events(&self, max: usize) -> Vec<AgentLifecycleEvent> {
        let skip = self.events.len().saturating_sub(max);
        self.events.iter().skip(skip).cloned().collect()
    }

    fn push_event(&mut self, event: AgentLifecycleEvent) {
        if self.events.len() >= EVENT_RING_CAP {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    fn publish_snapshot(&mut self) {
        let snapshot = Arc::new(self.snapshot());
        // `send_replace`, not `send`: publishing must not depend on a
        // receiver existing (`send` errors — and leaves the old value —
        // once every receiver is dropped).
        self.snapshot_tx.send_replace(snapshot);
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

    /// Register a worker and hand back the registry.
    fn registered(config: MultiAgentConfig, name: &str) -> (AgentRegistry, AgentPath) {
        let mut reg = AgentRegistry::new(config);
        let path = test_path(name);
        reg.register(&path, 1).unwrap();
        (reg, path)
    }

    #[test]
    fn derive_status_is_a_pure_function() {
        use AgentStatus::*;
        // Unregistered always wins.
        assert_eq!(derive_status(false, true, 3), Closed);
        assert_eq!(derive_status(false, false, 0), Closed);
        // In flight dominates queued work.
        assert_eq!(derive_status(true, true, 0), Running);
        assert_eq!(derive_status(true, true, 5), Running);
        // Settled with nothing queued → Done.
        assert_eq!(derive_status(true, false, 0), Done);
        // Work waiting, nothing executing → Queued.
        assert_eq!(derive_status(true, false, 1), Queued);
        assert_eq!(derive_status(true, false, 9), Queued);
    }

    #[test]
    fn status_names_are_wire_stable() {
        assert_eq!(AgentStatus::Queued.name(), "queued");
        assert_eq!(AgentStatus::Running.name(), "running");
        assert_eq!(AgentStatus::Done.name(), "done");
        assert_eq!(AgentStatus::Closed.name(), "closed");
        assert_eq!(AgentStatus::Paused.name(), "paused");
    }

    #[test]
    fn register_and_close() {
        let mut reg = AgentRegistry::new(test_config());
        let path = test_path("worker");

        assert!(reg.register(&path, 1).is_ok());
        assert_eq!(reg.count(), 1);
        assert!(reg.contains(&path));

        let entry = reg.get(&path).unwrap();
        // No Idle in the fact model: a registered agent with no queued work
        // and nothing in flight derives Done (see module docs).
        assert_eq!(entry.status(), AgentStatus::Done);
        assert_eq!(entry.depth, 1);
        // Fresh agent: no facts accumulated yet.
        assert_eq!(entry.queue_len, 0);
        assert!(!entry.in_flight);
        assert_eq!(entry.results_posted, 0);
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

    /// The full fact-driven lifecycle, including "queued behind a running
    /// task" (the shape session 20260904_c6559510 got wrong with markers).
    #[test]
    fn facts_drive_the_whole_lifecycle() {
        let (mut reg, path) = registered(test_config(), "worker");

        // Freshly registered → Done (no Idle variant).
        assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Done);

        // Two tasks queued back-to-back: Queued, then still Queued.
        assert!(reg.note_enqueued(&path));
        assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Queued);
        assert!(reg.note_enqueued(&path));
        assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Queued);

        // First dequeue → Running; second task still queued.
        assert!(reg.note_dequeued(&path));
        assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Running);
        assert!(reg.get(&path).unwrap().running_since.is_some());
        assert_eq!(reg.get(&path).unwrap().queue_len, 1);

        // First result posted (second task queued) → Queued.
        assert!(reg.note_posted(&path));
        assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Queued);
        assert!(reg.get(&path).unwrap().running_since.is_none());
        assert_eq!(reg.get(&path).unwrap().results_posted, 1);

        // Second dequeue → Running.
        assert!(reg.note_dequeued(&path));
        assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Running);

        // Final post, queue empty → Done. This is the Done-before-post
        // invariant, now structural: the fact lands before post_result.
        assert!(reg.note_posted(&path));
        assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Done);
        assert_eq!(reg.get(&path).unwrap().results_posted, 2);

        // Done agents still count toward quota.
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
        let (mut reg, path) = registered(test_config(), "worker");

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
        let (mut reg, path) = registered(test_config(), "worker");

        assert!(reg.touch(&path));
        assert!(reg.get(&path).unwrap().last_activity.is_some());
        assert_eq!(reg.get(&path).unwrap().tool_calls, 0);
        assert!(!reg.touch(&test_path("ghost")));
    }

    #[test]
    fn busy_count_tracks_outstanding_work() {
        let (mut reg, a) = registered(test_config(), "a");
        let b = test_path("b");
        reg.register(&b, 1).unwrap();
        assert_eq!(reg.busy_count(), 0);

        // a dequeued (in flight), b queued → both busy.
        assert!(reg.note_enqueued(&a));
        assert!(reg.note_dequeued(&a));
        assert!(reg.note_enqueued(&b));
        assert_eq!(reg.busy_count(), 2);

        // a posts with nothing left queued → only b busy.
        assert!(reg.note_posted(&a));
        assert_eq!(reg.busy_count(), 1);

        // b dequeued and posts → nobody busy.
        assert!(reg.note_dequeued(&b));
        assert!(reg.note_posted(&b));
        assert_eq!(reg.busy_count(), 0);
    }

    #[test]
    fn quiescence_requires_delivery_from_every_registered_agent() {
        // Session 20260904_c6559510 follow-up: the spawn→send window. A
        // freshly registered agent (spawned, task not yet enqueued) derives
        // Done on the old facts — the missing delivery clause is what keeps
        // a sibling's result from firing a batch that excludes it.
        let (mut reg, a) = registered(test_config(), "a");
        let b = test_path("b");
        reg.register(&b, 1).unwrap();

        // a finished and delivered; b exists but was never tasked.
        assert!(reg.note_enqueued(&a));
        assert!(reg.note_dequeued(&a));
        assert!(reg.note_posted(&a));
        assert!(!reg.quiescent(), "b never delivered — spawn→send window");

        // b gets its task, executes, delivers → now the field settles.
        assert!(reg.note_enqueued(&b));
        assert!(reg.note_dequeued(&b));
        assert!(!reg.quiescent(), "b is in flight");
        assert!(reg.note_posted(&b));
        assert!(reg.quiescent(), "everyone idle and delivered");

        // A queued follow-up re-blocks quiescence even though b delivered.
        assert!(reg.note_enqueued(&b));
        assert!(!reg.quiescent(), "queued work blocks the batch");
    }

    #[test]
    fn batch_handover_closes_and_reopens_the_delivery_gap() {
        // Session 20260904_841ed65b: a mid-turn parent saw `done` children
        // with no reports and concluded "the system didn't deliver" — the
        // gap must be visible as a fact, and handover must close it.
        let (mut reg, path) = registered(test_config(), "worker");

        reg.note_enqueued(&path);
        reg.note_dequeued(&path);
        reg.note_posted(&path);
        let snap = |reg: &AgentRegistry| reg.snapshot().agents[0].pending_results;
        assert_eq!(snap(&reg), 1, "posted but no batch has fired yet");

        // The watcher stamps the batch member at drain time.
        assert!(reg.note_batch_handed_over(&path));
        assert_eq!(snap(&reg), 0, "handed over — no longer pending");

        // A follow-up post (nudge answer) reopens the gap.
        reg.note_enqueued(&path);
        reg.note_dequeued(&path);
        reg.note_posted(&path);
        assert_eq!(snap(&reg), 1, "new post is pending until the next batch");

        // Ghost paths stay a no-op, and handover never goes backwards.
        assert!(!reg.note_batch_handed_over(&test_path("ghost")));
        assert_eq!(reg.get(&path).unwrap().results_handed_over, 1);
    }

    #[test]
    fn note_on_unknown_path_is_a_noop() {
        let mut reg = AgentRegistry::new(test_config());
        assert!(!reg.note_enqueued(&test_path("ghost")));
        assert!(!reg.note_dequeued(&test_path("ghost")));
        assert!(!reg.note_posted(&test_path("ghost")));
        assert!(!reg.note_send_failed(&test_path("ghost")));
        // Ghost notes stay fully silent: no transitions, no snapshot churn.
        assert!(reg.recent_events(10).is_empty());
        assert!(reg.latest_snapshot().agents.is_empty());
    }

    #[test]
    fn send_failure_rolls_back_the_enqueue_fact() {
        // send_task notes the enqueue BEFORE the task is receivable; a failed
        // send must give the fact back. Regression for the lock-race review
        // finding: a late enqueue note after the child's dequeue used to
        // leave a phantom `queue_len` that blocked quiescence forever.
        let (mut reg, path) = registered(test_config(), "worker");
        assert!(reg.note_enqueued(&path));
        assert_eq!(reg.get(&path).unwrap().queue_len, 1);

        assert!(reg.note_send_failed(&path));
        assert_eq!(reg.get(&path).unwrap().queue_len, 0);
        assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Done);
        // The agent still has zero deliveries, so it keeps blocking
        // quiescence via the delivery clause — but through a true fact, not
        // a phantom queue.
        assert!(!reg.quiescent());

        // Rollback cannot drive the count negative (saturating).
        assert!(reg.note_send_failed(&path));
        assert_eq!(reg.get(&path).unwrap().queue_len, 0);

        // A delivered agent whose follow-up send failed does not wedge the
        // batch: the rollback restores quiescence.
        let (mut reg2, p2) = registered(test_config(), "solo");
        reg2.note_enqueued(&p2);
        reg2.note_dequeued(&p2);
        reg2.note_posted(&p2);
        reg2.note_enqueued(&p2);
        assert!(!reg2.quiescent());
        reg2.note_send_failed(&p2);
        assert!(
            reg2.quiescent(),
            "rolled-back phantom queue must not wedge the batch"
        );
    }

    #[test]
    fn disabled_config_bypasses_spawn_limits() {
        // Restored (was dropped in the fact-model rewrite): limits are only
        // enforced when enabled=true — spawning works either way, the 6
        // tools just aren't registered when disabled.
        let config = MultiAgentConfig {
            enabled: false,
            ..MultiAgentConfig::default()
        };
        let reg = AgentRegistry::new(config);
        assert!(reg.can_spawn(999).is_ok());
    }

    #[tokio::test]
    async fn running_secs_and_watch_receiver_track_running_state() {
        let (mut reg, path) = registered(test_config(), "worker");
        let mut rx = reg.subscribe();
        assert!(
            reg.snapshot().agents[0].running_secs.is_none(),
            "not running yet"
        );

        reg.note_enqueued(&path);
        reg.note_dequeued(&path);
        rx.changed()
            .await
            .expect("watch receiver must be woken by the transition");
        assert_eq!(rx.borrow().agents[0].running_secs, Some(0));

        reg.note_posted(&path);
        assert!(
            reg.snapshot().agents[0].running_secs.is_none(),
            "stale seconds must clear on post"
        );
    }

    #[test]
    fn events_and_snapshot_track_transitions() {
        let (mut reg, path) = registered(test_config(), "worker");

        // No receivers: publishing must not fail loudly.
        assert!(reg.note_enqueued(&path));
        assert!(reg.note_dequeued(&path));
        assert!(reg.note_posted(&path));

        // register→Done, enqueue→Queued, dequeue→Running, post→Done: three
        // transitions (Done→Done at register emits nothing).
        let events = reg.recent_events(10);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].reason, "task_enqueued");
        assert_eq!(events[0].from, AgentStatus::Done);
        assert_eq!(events[0].to, AgentStatus::Queued);
        assert_eq!(events[1].reason, "task_dequeued");
        assert_eq!(events[1].to, AgentStatus::Running);
        assert_eq!(events[2].reason, "result_posted");
        assert_eq!(events[2].to, AgentStatus::Done);

        // close emits the unregistered transition.
        reg.close(&path);
        let events = reg.recent_events(10);
        assert_eq!(events.last().unwrap().reason, "unregistered");
        assert_eq!(events.last().unwrap().to, AgentStatus::Closed);

        // The snapshot no longer lists the agent.
        assert!(reg.snapshot().agents.is_empty());
    }

    #[test]
    fn event_ring_is_bounded() {
        let (mut reg, path) = registered(test_config(), "worker");
        for _ in 0..(EVENT_RING_CAP + 100) {
            // Enqueue+dequeue+post cycles generate 3 transitions each.
            reg.note_enqueued(&path);
            reg.note_dequeued(&path);
            reg.note_posted(&path);
        }
        assert_eq!(reg.events.len(), EVENT_RING_CAP);
        assert_eq!(reg.recent_events(3).len(), 3);
    }

    #[test]
    fn snapshot_reflects_derived_state() {
        let (mut reg, path) = registered(test_config(), "worker");

        reg.note_enqueued(&path);
        reg.note_dequeued(&path);
        reg.set_task(&path, "do the thing".to_string());
        reg.record_tool_call(&path);

        let snap = reg.snapshot();
        assert_eq!(snap.agents.len(), 1);
        let row = &snap.agents[0];
        assert_eq!(row.path, "root/worker");
        assert_eq!(row.status, "running");
        assert!(row.running_secs.is_some());
        assert_eq!(row.tool_calls, 1);
        assert_eq!(row.task.as_deref(), Some("do the thing"));

        // The watch channel carries the same snapshot once published
        // (publish happens on transitions; dequeue just moved status).
        let latest = reg.latest_snapshot();
        assert_eq!(latest.agents[0].status, "running");
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
