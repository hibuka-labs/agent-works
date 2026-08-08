//! Asynchronous message passing between agents.
//!
//! The [`MailboxHub`] manages per-agent mailboxes and a global sequence number
//! (`tokio::sync::watch<u64>`) that notifies waiters when any result arrives.
//!
//! # Architecture
//!
//! ```text
//! Parent (LLM tools)                      Child (AgentRuntime task)
//!        │                                         │
//!        │  send_task() / send_message()           │
//!        ├────────────────────────────────────────►│ task_rx
//!        │                                         │
//!        │                      post_result()      │
//!        │◄────────────────────────────────────────┤ (via result_tx clone)
//!        │                                         │
//!        │  wait_for_result() watches seq_rx       │
//!        │  (blocks until seq changes)             │
//! ```
//!
//! Every `post_result()` increments the global sequence number, waking all
//! `wait_for_result()` callers.

use std::collections::HashMap;
use std::sync::Mutex;

use tokio::sync::{mpsc, watch};

use super::path::AgentPath;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A task sent from parent to child agent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MailboxTask {
    /// The task description / user input for the child agent.
    pub task: String,
    /// Whether to interrupt the child's current execution.
    pub interrupt: bool,
    /// Pending messages accumulated before this task (from `send_message`).
    pub pending_messages: Vec<String>,
}

/// Status of a result posted from child to parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MailboxStatus {
    /// Child agent completed its task successfully.
    Ok,
    /// Child agent encountered an error.
    Error,
    /// Child agent was closed.
    Closed,
}

/// A result posted from child agent to parent.
#[derive(Clone, Debug)]
pub struct MailboxResult {
    /// Which agent produced this result.
    pub agent_path: AgentPath,
    /// The status of the result.
    pub status: MailboxStatus,
    /// The result text (if any).
    pub result: Option<String>,
}

// ---------------------------------------------------------------------------
// Per-agent mailbox handle (child side)
// ---------------------------------------------------------------------------

/// The child-side handle to a mailbox.
///
/// Given to the spawned child agent task. The child reads tasks from `task_rx`
/// and posts results via `hub.post_result()` using the agent path.
#[derive(Debug)]
pub struct ChildMailbox {
    /// Receive tasks from parent.
    pub task_rx: mpsc::Receiver<MailboxTask>,
}

// ---------------------------------------------------------------------------
// Per-agent mailbox (internal)
// ---------------------------------------------------------------------------

struct MailboxEntry {
    /// Send tasks to child.
    task_tx: mpsc::Sender<MailboxTask>,
    /// Results received from child (or posted by runtime).
    results: Vec<MailboxResult>,
    /// Pending messages (from `send_message`, no execution trigger).
    pending: Vec<String>,
}

// ---------------------------------------------------------------------------
// MailboxHub
// ---------------------------------------------------------------------------

/// Central hub for inter-agent message passing.
///
/// Manages per-agent mailboxes and a global sequence number. The sequence
/// number increments every time a result is posted, allowing `wait_for_result`
/// to efficiently block until new data arrives.
///
/// All methods use internal `Mutex` — the hub is designed to be shared
/// via `Arc<MailboxHub>`.
pub struct MailboxHub {
    entries: Mutex<HashMap<AgentPath, MailboxEntry>>,
    seq_tx: watch::Sender<u64>,
    seq_rx: watch::Receiver<u64>,
}

impl MailboxHub {
    /// Create a new empty mailbox hub.
    pub fn new() -> Self {
        let (seq_tx, seq_rx) = watch::channel(0);
        Self {
            entries: Mutex::new(HashMap::new()),
            seq_tx,
            seq_rx,
        }
    }

    /// Register a new agent mailbox.
    ///
    /// Returns the child-side handle to be given to the spawned agent task.
    /// Returns `None` if the agent_path is already registered.
    pub fn register(&self, agent_path: &AgentPath) -> Option<ChildMailbox> {
        let mut entries = self.entries.lock().unwrap();
        if entries.contains_key(agent_path) {
            return None;
        }
        let (task_tx, task_rx) = mpsc::channel(32);
        entries.insert(
            agent_path.clone(),
            MailboxEntry {
                task_tx,
                results: Vec::new(),
                pending: Vec::new(),
            },
        );
        Some(ChildMailbox { task_rx })
    }

    /// Unregister an agent mailbox.
    ///
    /// Posts a `Closed` result first (to wake any waiters), then removes the entry.
    /// Returns `true` if the agent was registered.
    pub fn unregister(&self, agent_path: &AgentPath) -> bool {
        let mut entries = self.entries.lock().unwrap();
        if entries.get(agent_path).is_some() {
            // Wake waiters with sequence bump
            let current = *self.seq_rx.borrow();
            let _ = self.seq_tx.send(current.wrapping_add(1));
            entries.remove(agent_path);
            true
        } else {
            false
        }
    }

    /// Send a message to a sub-agent (no execution trigger).
    ///
    /// The message is appended to the agent's pending message buffer.
    /// Returns `true` if the message was queued, `false` if the agent is not registered.
    pub fn send_message(&self, agent_path: &AgentPath, message: String) -> bool {
        let mut entries = self.entries.lock().unwrap();
        match entries.get_mut(agent_path) {
            Some(entry) => {
                entry.pending.push(message);
                true
            }
            None => false,
        }
    }

    /// Send a task to a sub-agent (triggers execution).
    ///
    /// Drains pending messages and packages them with the task.
    /// Returns `true` if the task was sent, `false` if the agent is not registered
    /// or the channel is full.
    pub fn send_task(&self, agent_path: &AgentPath, task: String, interrupt: bool) -> bool {
        let mut entries = self.entries.lock().unwrap();
        match entries.get_mut(agent_path) {
            Some(entry) => {
                let pending = std::mem::take(&mut entry.pending);
                let mailbox_task = MailboxTask {
                    task,
                    interrupt,
                    pending_messages: pending,
                };
                entry.task_tx.try_send(mailbox_task).is_ok()
            }
            None => false,
        }
    }

    /// Check if an agent has pending (unread) messages.
    pub fn has_pending(&self, agent_path: &AgentPath) -> bool {
        let entries = self.entries.lock().unwrap();
        entries
            .get(agent_path)
            .map(|e| !e.pending.is_empty())
            .unwrap_or(false)
    }

    /// Post a result from a child agent.
    ///
    /// Increments the global sequence number, waking all `wait_for_result` callers.
    pub fn post_result(&self, result: MailboxResult) {
        let mut entries = self.entries.lock().unwrap();
        if let Some(entry) = entries.get_mut(&result.agent_path) {
            entry.results.push(result);
            // Notify waiters
            let current = *self.seq_rx.borrow();
            let _ = self.seq_tx.send(current.wrapping_add(1));
        }
    }

    /// Get a clone of the global sequence number receiver.
    ///
    /// Used by `wait_agent` to watch for changes before polling.
    pub fn subscribe_seq(&self) -> watch::Receiver<u64> {
        self.seq_rx.clone()
    }

    /// Try to receive a result for a specific agent (non-blocking).
    ///
    /// Returns the oldest unread result for the agent, or `None`.
    pub fn try_recv_result(&self, agent_path: &AgentPath) -> Option<MailboxResult> {
        let mut entries = self.entries.lock().unwrap();
        entries.get_mut(agent_path).and_then(|e| {
            if e.results.is_empty() {
                None
            } else {
                Some(e.results.remove(0))
            }
        })
    }

    /// Try to receive any result (non-blocking).
    ///
    /// Returns the first available result from any agent mailbox.
    pub fn try_recv_any(&self) -> Option<MailboxResult> {
        let mut entries = self.entries.lock().unwrap();
        for entry in entries.values_mut() {
            if !entry.results.is_empty() {
                return Some(entry.results.remove(0));
            }
        }
        None
    }

    /// Check if an agent has unread results.
    pub fn has_results(&self, agent_path: &AgentPath) -> bool {
        let entries = self.entries.lock().unwrap();
        entries
            .get(agent_path)
            .map(|e| !e.results.is_empty())
            .unwrap_or(false)
    }

    /// Return the total number of unread results across all agents.
    pub fn total_pending_results(&self) -> usize {
        let entries = self.entries.lock().unwrap();
        entries.values().map(|e| e.results.len()).sum()
    }

    /// Check if an agent is registered.
    pub fn contains(&self, agent_path: &AgentPath) -> bool {
        let entries = self.entries.lock().unwrap();
        entries.contains_key(agent_path)
    }

    /// Return the number of registered agents.
    pub fn len(&self) -> usize {
        let entries = self.entries.lock().unwrap();
        entries.len()
    }

    /// Return whether there are no registered agents.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return all registered agent paths.
    pub fn agent_paths(&self) -> Vec<AgentPath> {
        let entries = self.entries.lock().unwrap();
        entries.keys().cloned().collect()
    }
}

impl Default for MailboxHub {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn test_path(name: &str) -> AgentPath {
        AgentPath::root().join(name)
    }

    #[test]
    fn register_and_unregister() {
        let hub = MailboxHub::new();
        let path = test_path("test-agent");

        assert!(!hub.contains(&path));
        assert_eq!(hub.len(), 0);

        let child = hub.register(&path);
        assert!(child.is_some());
        assert!(hub.contains(&path));
        assert_eq!(hub.len(), 1);

        // Duplicate register fails
        assert!(hub.register(&path).is_none());

        assert!(hub.unregister(&path));
        assert!(!hub.contains(&path));
        assert_eq!(hub.len(), 0);

        // Double unregister is a no-op
        assert!(!hub.unregister(&path));
    }

    #[test]
    fn send_message_and_task() {
        let hub = MailboxHub::new();
        let path = test_path("worker");

        let mut child = hub.register(&path).unwrap();

        // Messages accumulate without triggering
        assert!(hub.send_message(&path, "hello".into()));
        assert!(hub.send_message(&path, "world".into()));
        assert!(hub.has_pending(&path));

        // Send to non-existent agent
        assert!(!hub.send_message(&test_path("ghost"), "nope".into()));

        // Task drains pending
        assert!(hub.send_task(&path, "do work".into(), true));
        assert!(!hub.has_pending(&path));

        // Child receives task with pending messages
        let received = child.task_rx.try_recv().unwrap();
        assert_eq!(received.task, "do work");
        assert!(received.interrupt);
        assert_eq!(received.pending_messages, vec!["hello", "world"]);
    }

    #[test]
    fn post_and_receive_result() {
        let hub = MailboxHub::new();
        let path = test_path("worker");

        hub.register(&path);

        hub.post_result(MailboxResult {
            agent_path: path.clone(),
            status: MailboxStatus::Ok,
            result: Some("done!".into()),
        });

        assert!(hub.has_results(&path));

        let received = hub.try_recv_result(&path);
        assert!(received.is_some());
        let r = received.unwrap();
        assert_eq!(r.agent_path, path);
        assert_eq!(r.status, MailboxStatus::Ok);
        assert_eq!(r.result.unwrap(), "done!");

        assert!(!hub.has_results(&path));
    }

    #[test]
    fn try_recv_any_returns_all() {
        let hub = MailboxHub::new();
        let a = test_path("a");
        let b = test_path("b");

        hub.register(&a);
        hub.register(&b);

        hub.post_result(MailboxResult {
            agent_path: a.clone(),
            status: MailboxStatus::Ok,
            result: Some("first".into()),
        });
        hub.post_result(MailboxResult {
            agent_path: b.clone(),
            status: MailboxStatus::Error,
            result: Some("second".into()),
        });

        // HashMap iteration order is non-deterministic, so just check we get both
        let r1 = hub.try_recv_any().unwrap();
        let r2 = hub.try_recv_any().unwrap();
        assert!(hub.try_recv_any().is_none());

        let mut paths = vec![r1.agent_path.to_string(), r2.agent_path.to_string()];
        paths.sort();
        assert_eq!(paths, vec!["root/a", "root/b"]);
    }

    #[test]
    fn sequence_number_changes_on_post() {
        let hub = MailboxHub::new();
        let path = test_path("worker");
        hub.register(&path);

        let seq = hub.subscribe_seq();
        let initial = *seq.borrow();

        hub.post_result(MailboxResult {
            agent_path: path.clone(),
            status: MailboxStatus::Ok,
            result: None,
        });

        assert!(seq.has_changed().unwrap());
        assert_ne!(*seq.borrow(), initial);
    }

    #[test]
    fn sequence_number_changes_on_unregister() {
        let hub = MailboxHub::new();
        let path = test_path("worker");
        hub.register(&path);

        let seq = hub.subscribe_seq();
        let initial = *seq.borrow();

        hub.unregister(&path);

        assert!(seq.has_changed().unwrap());
        assert_ne!(*seq.borrow(), initial);
    }

    #[test]
    fn agent_paths() {
        let hub = MailboxHub::new();
        hub.register(&test_path("a"));
        hub.register(&test_path("b"));

        let mut paths = hub.agent_paths();
        paths.sort();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn total_pending_results() {
        let hub = MailboxHub::new();
        let a = test_path("a");
        hub.register(&a);

        assert_eq!(hub.total_pending_results(), 0);

        hub.post_result(MailboxResult {
            agent_path: a.clone(),
            status: MailboxStatus::Ok,
            result: None,
        });
        assert_eq!(hub.total_pending_results(), 1);

        hub.post_result(MailboxResult {
            agent_path: a.clone(),
            status: MailboxStatus::Ok,
            result: None,
        });
        assert_eq!(hub.total_pending_results(), 2);

        hub.try_recv_any();
        assert_eq!(hub.total_pending_results(), 1);
    }

    #[tokio::test]
    async fn wait_for_result_pattern() {
        let hub = Arc::new(MailboxHub::new());
        let path = test_path("worker");
        hub.register(&path);

        let hub_clone = hub.clone();
        let path_clone = path.clone();

        // Spawn a task that posts a result after a short delay
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            hub_clone.post_result(MailboxResult {
                agent_path: path_clone,
                status: MailboxStatus::Ok,
                result: Some("async result".into()),
            });
        });

        // Wait for result using the seq pattern
        let mut seq = hub.subscribe_seq();
        loop {
            match hub.try_recv_any() {
                Some(r) => {
                    assert_eq!(r.status, MailboxStatus::Ok);
                    assert_eq!(r.result.unwrap(), "async result");
                    break;
                }
                None => {
                    let _ = seq.changed().await;
                }
            }
        }
    }
}
