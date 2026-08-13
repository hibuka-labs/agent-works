use tokio::sync::mpsc;

use agent_base::{AgentRuntime, RuntimeEvent, SessionId};

/// Agent session handle — unified input/output/cancel interface
///
/// `AgentHandle` wraps a command queue, a Worker task, an event stream, and cancellation.
/// All callers (CLI / UI / HTTP API) interact with agent-base through it.
///
/// # Example
///
/// ```rust,no_run
/// use agent_works::AgentHandle;
/// use agent_base::AgentRuntime;
///
/// # async fn example(runtime: AgentRuntime) {
/// let mut handle = AgentHandle::new(runtime);
///
/// // Send user input
/// handle.send_input("check disk space").await.unwrap();
///
/// // Receive events
/// while let Some(event) = handle.recv_event().await {
///     // Handle event...
///     if matches!(event, agent_base::RuntimeEvent::RunFinished { .. }
///         | agent_base::RuntimeEvent::RunCancelled { .. }) {
///         break;
///     }
/// }
///
/// // Cancel current execution
/// handle.cancel();
/// # }
/// ```
pub struct AgentHandle {
    cmd_tx: mpsc::Sender<AgentCommand>,
    event_rx: mpsc::UnboundedReceiver<RuntimeEvent>,
    runtime: AgentRuntime,
    default_session_id: Option<SessionId>,
}

enum AgentCommand {
    RunTurn {
        session_id: SessionId,
        input: String,
    },
}

#[derive(Debug)]
pub enum SendError {
    ChannelClosed,
}

impl AgentHandle {
    /// Create a new AgentHandle, spawning a background Worker
    pub fn new(runtime: AgentRuntime) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(32);
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let rt = runtime.clone();

        // Worker task: processes user requests serially
        tokio::spawn(async move {
            let mut rx = cmd_rx;
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    AgentCommand::RunTurn { session_id, input } => {
                        let tx = event_tx.clone();
                        let sid = session_id.clone();
                        let result = rt
                            .run_turn(session_id, &input, move |event| {
                                let _ = tx.send(event);
                                Ok(())
                            })
                            .await;

                        match &result {
                            Ok(_) => {}
                            Err(e) if e.is_cancelled() => {}
                            Err(e) => {
                                tracing::error!(error = %e, "run_turn failed");
                                let _ = event_tx.send(RuntimeEvent::RunFinished {
                                    session_id: sid,
                                    agent_id: None,
                                    trace_id: None,
                                });
                            }
                        }
                    }
                }
            }
        });

        Self {
            cmd_tx,
            event_rx,
            runtime,
            default_session_id: None,
        }
    }

    /// Create an AgentHandle with a default session_id
    /// The session_id will be used for all send_input calls unless overridden
    pub fn with_session(runtime: AgentRuntime, session_id: SessionId) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(32);
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let rt = runtime.clone();

        // Worker task: processes user requests serially
        tokio::spawn(async move {
            let mut rx = cmd_rx;
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    AgentCommand::RunTurn { session_id, input } => {
                        let tx = event_tx.clone();
                        let sid = session_id.clone();
                        let result = rt
                            .run_turn(session_id, &input, move |event| {
                                let _ = tx.send(event);
                                Ok(())
                            })
                            .await;

                        // Handle errors: ensure caller always gets a terminal event
                        match &result {
                            Ok(_) => {
                                // run_turn already emitted RunFinished or RunCancelled
                            }
                            Err(e) if e.is_cancelled() => {
                                // RunCancelled already emitted inside run_turn
                            }
                            Err(e) => {
                                // Non-cancellation error: emit RunFinished so caller isn't stuck
                                tracing::error!(error = %e, "run_turn failed");
                                let _ = event_tx.send(RuntimeEvent::RunFinished {
                                    session_id: sid,
                                    agent_id: None,
                                    trace_id: None,
                                });
                            }
                        }
                    }
                }
            }
        });

        Self {
            cmd_tx,
            event_rx,
            runtime,
            default_session_id: Some(session_id),
        }
    }

    /// Send user input (async, with error return)
    /// Uses the default session_id if set via with_session(), otherwise creates a new session
    pub async fn send_input(&self, input: &str) -> Result<(), SendError> {
        let session_id = match &self.default_session_id {
            Some(id) => id.clone(),
            None => self.runtime.create_session().await,
        };
        self.cmd_tx
            .send(AgentCommand::RunTurn {
                session_id,
                input: input.to_string(),
            })
            .await
            .map_err(|_| SendError::ChannelClosed)
    }

    /// Send user input with a specified session_id
    pub async fn send_input_with_session(
        &self,
        input: &str,
        session_id: SessionId,
    ) -> Result<(), SendError> {
        self.cmd_tx
            .send(AgentCommand::RunTurn {
                session_id,
                input: input.to_string(),
            })
            .await
            .map_err(|_| SendError::ChannelClosed)
    }

    /// Receive the next event (blocking)
    pub async fn recv_event(&mut self) -> Option<RuntimeEvent> {
        self.event_rx.recv().await
    }

    /// Try to receive an event (non-blocking)
    pub fn try_recv_event(&mut self) -> Option<RuntimeEvent> {
        self.event_rx.try_recv().ok()
    }

    /// Cancel current execution — delegates to runtime, always cancels the latest token
    pub fn cancel(&self) {
        self.runtime.cancel();
    }

    /// Get a reference to the underlying runtime
    pub fn runtime(&self) -> &AgentRuntime {
        &self.runtime
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::time::Duration;

    use agent_base::{
        AgentBuilder, AgentResult, ChatMessage, LlmCapabilities, ReasoningConfig, ResponseFormat,
        StreamChunk, StreamClient,
    };
    use futures_core::Stream;
    use serde_json::Value;

    struct StubClient;

    #[async_trait::async_trait]
    impl StreamClient for StubClient {
        async fn stream(
            &self,
            _messages: &[ChatMessage],
            _tools: &[Value],
            _reasoning: Option<&ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>>> {
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::Text("hello".to_string())),
                Ok(StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ])))
        }

        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::default()
        }
    }

    fn runtime() -> AgentRuntime {
        AgentBuilder::new(Arc::new(StubClient)).build().unwrap()
    }

    async fn wait_for_terminal(handle: &mut AgentHandle) -> Option<RuntimeEvent> {
        let mut terminal = None;
        for _ in 0..100 {
            let ev = tokio::time::timeout(Duration::from_secs(5), handle.recv_event()).await;
            match ev {
                Ok(Some(e @ RuntimeEvent::RunFinished { .. }))
                | Ok(Some(e @ RuntimeEvent::RunCancelled { .. })) => {
                    terminal = Some(e);
                    break;
                }
                Ok(Some(_)) => continue,
                Ok(None) => break,
                Err(_) => break,
            }
        }
        terminal
    }

    #[tokio::test]
    async fn test_send_input_and_recv_terminal() {
        let mut handle = AgentHandle::new(runtime());
        handle.send_input("hello").await.unwrap();
        let terminal = wait_for_terminal(&mut handle).await;
        assert!(
            terminal.is_some(),
            "expected a terminal event, got {terminal:?}"
        );
    }

    #[tokio::test]
    async fn test_send_input_with_session() {
        let rt = runtime();
        let session_id = rt.create_session().await;
        let mut handle = AgentHandle::with_session(rt, session_id.clone());
        handle
            .send_input_with_session("hello", session_id)
            .await
            .unwrap();
        let terminal = wait_for_terminal(&mut handle).await;
        assert!(terminal.is_some());
    }

    #[tokio::test]
    async fn test_runtime_accessor_and_cancel() {
        let rt = runtime();
        let handle = AgentHandle::new(rt.clone());
        // Accessor returns a runtime that shares the same session store.
        let session_id = handle.runtime().create_session().await;
        assert!(rt.session(&session_id).await.is_some());
        // Cancel is a no-op when nothing is running, but must not panic.
        handle.cancel();
    }

    #[tokio::test]
    async fn test_try_recv_event_initially_empty() {
        let mut handle = AgentHandle::new(runtime());
        assert!(handle.try_recv_event().is_none());
    }
}
