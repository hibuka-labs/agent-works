//! Child-agent view types (design doc §5.2).
//!
//! [`ChildHandle`] is a **view, not an owner** — dropping it does not cancel
//! the child (the concurrency slot and registry entry live in the child task's
//! `ChildCleanup` drop-guard (runtime-private), released when the
//! task future ends on any of the three paths). To get "close on scope exit"
//! you must *explicitly* convert a handle into a [`ChildGuard`] via
//! [`ChildHandle::into_guard`] — the one-way direction is deliberate (review
//! D1: symmetric constructors are the source of misuse).

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use agent_base::AgentError;

use super::path::AgentPath;
use super::runtime::{MultiAgentRuntime, WaitResult};

// ---------------------------------------------------------------------------
// ChildHandle
// ---------------------------------------------------------------------------

/// A handle to a spawned child agent.
#[derive(Clone)]
pub struct ChildHandle {
    runtime: Arc<MultiAgentRuntime>,
    path: AgentPath,
    spawned_tools: BTreeSet<String>,
}

impl ChildHandle {
    pub(crate) fn new(
        runtime: Arc<MultiAgentRuntime>,
        path: AgentPath,
        spawned_tools: BTreeSet<String>,
    ) -> Self {
        Self {
            runtime,
            path,
            spawned_tools,
        }
    }

    /// The child's agent path (e.g. `"root/worker"`).
    pub fn agent_path(&self) -> String {
        self.path.to_string()
    }

    /// The tools the child was **actually** given (post-exclusion whitelist),
    /// echoed at spawn time so the parent never assumes a permission the child
    /// lacks (§5.4 review M-3).
    pub fn spawned_tools(&self) -> &BTreeSet<String> {
        &self.spawned_tools
    }

    /// Queue a message without triggering execution.
    ///
    /// `Ok(false)` means delivery failed (child not registered / queue full)
    /// — surfaced faithfully, not swallowed into `Ok(())` (review m-1). Only
    /// a malformed path is an `Err`.
    pub fn send(&self, message: impl Into<String>) -> Result<bool, AgentError> {
        self.runtime
            .send_message(&self.agent_path(), message.into())
            .map_err(AgentError::ConfigError)
    }

    /// Send a task and trigger execution (replaces `followup_task`). Tasks run
    /// **serially** inside the child; the dead `interrupt` parameter is not
    /// exposed (defect K2).
    pub fn task(&self, task: impl Into<String>) -> Result<bool, AgentError> {
        self.runtime
            .send_task(&self.agent_path(), task.into(), false)
            .map_err(AgentError::ConfigError)
    }

    /// Wait for the child's next result. A timeout is
    /// [`ChildOutcome::Timeout`], not an error.
    pub async fn wait(&self, timeout: Duration) -> ChildOutcome {
        let wr = self
            .runtime
            .wait_for_result(Some(&self.agent_path()), timeout.as_millis() as u64)
            .await;
        ChildOutcome::from_wait_result(wr)
    }

    /// Initiate close: synchronously cancels and reports whether the child
    /// existed; the actual teardown (registry / mailbox / slot) is deferred to
    /// the child task's drop-guard (§4). Idempotent; a `wait` after close
    /// yields [`ChildOutcome::Closed`].
    pub fn close(&self) -> Result<bool, AgentError> {
        self.runtime
            .close_agent(&self.agent_path())
            .map(|r| r.closed)
            .map_err(AgentError::ConfigError)
    }

    /// One-way conversion into an RAII guard.
    pub fn into_guard(self) -> ChildGuard {
        ChildGuard {
            handle: Some(self),
            disarmed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ChildOutcome
// ---------------------------------------------------------------------------

/// Typed wait result (maps the existing `WaitResult.status` string).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChildOutcome {
    /// Task completed; text and denial info attached.
    Ok {
        text: Option<String>,
        has_more: bool,
        denied_tools: Vec<String>,
    },
    /// Task failed (status `"error"`).
    Failed { text: Option<String> },
    /// Child is closed / gone — terminal, from a post-close `wait` (§5.2).
    Closed,
    /// The `wait` deadline elapsed — not an error (§5.2).
    Timeout,
}

impl ChildOutcome {
    /// Map a raw [`WaitResult`] onto the typed outcome.
    pub(crate) fn from_wait_result(wr: WaitResult) -> Self {
        match wr.status.as_str() {
            "ok" => Self::Ok {
                text: wr.result,
                has_more: wr.has_more,
                denied_tools: wr.denied_tools,
            },
            "error" => Self::Failed { text: wr.result },
            "closed" => Self::Closed,
            _ => Self::Timeout,
        }
    }
}

// ---------------------------------------------------------------------------
// ChildGuard
// ---------------------------------------------------------------------------

/// RAII guard: closes the child on drop (i.e. cancels it; cleanup still
/// converges in the child task's drop-guard).
pub struct ChildGuard {
    handle: Option<ChildHandle>,
    disarmed: bool,
}

impl ChildGuard {
    /// Borrow the inner handle (the guard still owns close-on-drop semantics).
    pub fn handle(&self) -> &ChildHandle {
        self.handle
            .as_ref()
            .expect("ChildGuard handle is present until into_handle")
    }

    /// Give up the guard, keeping the handle (no close on drop).
    pub fn into_handle(mut self) -> ChildHandle {
        // Disarm first, then move the handle out of the `Option` — this is
        // the leak-free way to consume a `Drop` type (a direct field move is
        // rejected by the borrow checker; `ManuallyDrop` would skip dropping
        // the inner `Arc`s).
        self.disarmed = true;
        self.handle
            .take()
            .expect("handle is present before Drop runs")
    }

    /// Disarm: drop the guard without closing the child.
    pub fn disarm(&mut self) {
        self.disarmed = true;
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.disarmed
            && let Some(handle) = &self.handle
        {
            let _ = handle.close();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use agent_base::llm_trait::LlmProvider;

    use super::*;
    use crate::multi_agent::child_config::ChildConfig;
    use crate::multi_agent::config::MultiAgentConfig;
    use crate::multi_agent::path::AgentPath;
    use crate::multi_agent::runtime::WaitResult;

    // A minimal echo provider so children complete a task without network.
    struct EchoLlm;

    #[async_trait::async_trait]
    impl LlmProvider for EchoLlm {
        async fn stream(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
            Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
                futures_util::stream::iter(vec![
                    Ok(agent_base::StreamChunk::Text("done".to_string())),
                    Ok(agent_base::StreamChunk::Stop {
                        finish_reason: Some("stop".to_string()),
                    }),
                ]),
            )))
        }
        async fn chat(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
            unreachable!("unused")
        }
        fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
            agent_base::llm_trait::Capabilities::default()
        }
        fn info(&self) -> agent_base::llm_trait::ProviderInfo {
            agent_base::llm_trait::ProviderInfo {
                name: "echo".into(),
                model: "echo".into(),
                version: None,
            }
        }
    }

    fn runtime() -> Arc<MultiAgentRuntime> {
        Arc::new(MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            Arc::new(EchoLlm),
            vec![],
            tokio_util::sync::CancellationToken::new(),
            None,
            agent_base::Language::En,
            None,
            None,
        ))
    }

    fn wr(status: &str) -> WaitResult {
        WaitResult {
            status: status.into(),
            result: Some("x".into()),
            agent_path: Some("root/w".into()),
            has_more: false,
            denied_tools: vec!["read_file".into()],
        }
    }

    #[test]
    fn outcome_maps_all_four_statuses() {
        assert_eq!(
            ChildOutcome::from_wait_result(wr("ok")),
            ChildOutcome::Ok {
                text: Some("x".into()),
                has_more: false,
                denied_tools: vec!["read_file".into()],
            }
        );
        assert_eq!(
            ChildOutcome::from_wait_result(wr("error")),
            ChildOutcome::Failed {
                text: Some("x".into())
            }
        );
        assert_eq!(
            ChildOutcome::from_wait_result(wr("closed")),
            ChildOutcome::Closed
        );
        assert_eq!(
            ChildOutcome::from_wait_result(wr("timeout")),
            ChildOutcome::Timeout
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn handle_send_task_wait_close_round_trip() {
        let ma = runtime();
        let spawned = ma
            .spawn_with_config(
                "w".to_string(),
                ChildConfig {
                    system_prompt: Some("prompt".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let handle = ChildHandle::new(
            Arc::clone(&ma),
            spawned.agent_path().clone(),
            spawned.spawned_tools().clone(),
        );

        assert_eq!(handle.agent_path(), "root/w");
        // send queues without triggering.
        assert!(handle.send("heads up").unwrap());
        // task triggers execution; child answers "done".
        assert!(handle.task("do it").unwrap());
        let outcome = handle.wait(Duration::from_secs(3)).await;
        assert!(
            matches!(outcome, ChildOutcome::Ok { text: Some(ref t), .. } if t == "done"),
            "got {outcome:?}"
        );

        assert!(handle.close().unwrap());
        assert!(!handle.close().unwrap(), "idempotent close");
        // close → deferred cleanup → wait synthesizes Closed (§5.2 contract).
        let outcome = handle.wait(Duration::from_millis(500)).await;
        assert_eq!(outcome, ChildOutcome::Closed);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn guard_closes_on_drop_unless_disarmed() {
        let ma = runtime();
        let spawned = ma
            .spawn_with_config(
                "g".to_string(),
                ChildConfig {
                    system_prompt: Some("prompt".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let handle = ChildHandle::new(
            Arc::clone(&ma),
            spawned.agent_path().clone(),
            spawned.spawned_tools().clone(),
        );

        // Guard drop closes the child.
        let guard = handle.clone().into_guard();
        drop(guard);
        // Deferred cleanup: registry empties eventually.
        for _ in 0..100 {
            if ma.registry().lock().unwrap().count() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(ma.registry().lock().unwrap().count(), 0);

        // Disarmed guard leaves the child alive.
        let spawned2 = ma
            .spawn_with_config(
                "h".to_string(),
                ChildConfig {
                    system_prompt: Some("prompt".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let handle2 = ChildHandle::new(
            Arc::clone(&ma),
            spawned2.agent_path().clone(),
            spawned2.spawned_tools().clone(),
        );
        let mut guard2 = handle2.clone().into_guard();
        guard2.disarm();
        drop(guard2);
        assert!(
            ma.registry()
                .lock()
                .unwrap()
                .contains(&AgentPath::root().join("h"))
        );
    }
}
