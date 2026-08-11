//! Multi-agent runtime — coordinates sub-agent lifecycle, event bridging, and
//! cancellation.
//!
//! The [`MultiAgentRuntime`] is the central coordinator. It is created once during
//! builder setup and shared via `Arc` to all 6 multi-agent tools.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_base::{
    AgentBuilder, AgentResult, AgentRuntime, DenyAllApprovalHandler, Language, LlmClient,
    RunOutcome, RuntimeEvent, SessionId, Tool, UserEvent,
};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use super::config::MultiAgentConfig;
use super::mailbox::{ChildMailbox, MailboxHub, MailboxResult, MailboxStatus, MailboxTask};
use super::path::AgentPath;
use super::registry::{AgentRegistry, AgentStatus};

// ---------------------------------------------------------------------------
// MultiAgentRuntime
// ---------------------------------------------------------------------------

/// Coordinates sub-agent lifecycle, event bridging, and cancellation.
///
/// Created once during builder setup and shared via `Arc` to all 6 multi-agent
/// tools. Each tool calls methods on the runtime to spawn, communicate with, or
/// close sub-agents.
pub struct MultiAgentRuntime {
    /// Agent lifecycle registry (spawn/close/query).
    registry: Mutex<AgentRegistry>,

    /// Inter-agent message hub.
    mailbox: Arc<MailboxHub>,

    /// Shared LLM client (from parent agent).
    client: Arc<dyn LlmClient>,

    /// Business tools to register on child agents (NOT the 6 multi-agent tools).
    business_tools: Vec<Arc<dyn Tool>>,

    /// Channel to the bridge task that emits events on parent's event bus.
    event_tx: Mutex<Option<tokio::sync::mpsc::UnboundedSender<RuntimeEvent>>>,

    /// Root cancellation token (propagates to all children).
    root_cancel: CancellationToken,

    /// JoinSet tracking all child agent tasks.
    join_set: Mutex<JoinSet<()>>,

    /// Per-child cancellation tokens.
    child_cancels: Mutex<HashMap<AgentPath, CancellationToken>>,

    /// Error recovery strategy (inherited from parent).
    error_recovery: Option<Arc<dyn agent_base::ToolErrorRecovery>>,

    /// Language preference.
    language: Language,

    /// Parent session manager — for fork_history (child context inheritance).
    session_manager: Mutex<Option<Arc<agent_base::engine::SessionManager>>>,
}

impl MultiAgentRuntime {
    /// Create a new multi-agent runtime.
    ///
    /// This is called internally by the builder. Tools receive an `Arc<Self>`.
    pub fn new(
        config: MultiAgentConfig,
        client: Arc<dyn LlmClient>,
        business_tools: Vec<Arc<dyn Tool>>,
        root_cancel: CancellationToken,
        error_recovery: Option<Arc<dyn agent_base::ToolErrorRecovery>>,
        language: Language,
    ) -> Self {
        Self {
            registry: Mutex::new(AgentRegistry::new(config)),
            mailbox: Arc::new(MailboxHub::new()),
            client,
            business_tools,
            event_tx: Mutex::new(None),
            root_cancel,
            join_set: Mutex::new(JoinSet::new()),
            child_cancels: Mutex::new(HashMap::new()),
            error_recovery,
            language,
            session_manager: Mutex::new(None),
        }
    }

    /// Set the event sender for bridging child events to parent.
    ///
    /// Called by the builder after creating the bridge channel.
    pub fn set_event_sender(&self, tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>) {
        *self.event_tx.lock().unwrap() = Some(tx);
    }

    /// Set the parent session manager for fork_history support.
    ///
    /// Called by the builder after creating the runtime.
    pub fn set_session_manager(&self, session_manager: Arc<agent_base::engine::SessionManager>) {
        *self.session_manager.lock().unwrap() = Some(session_manager);
    }

    /// Spawn a child agent at the given path with a specific system prompt.
    ///
    /// This is called by the `spawn_agent` tool. It:
    /// 1. Checks spawn limits
    /// 2. Registers the agent in the registry
    /// 3. Creates a mailbox
    /// 4. Builds a child AgentRuntime
    /// 5. Spawns a tokio task for the child's event loop
    /// 6. Returns the AgentPath
    ///
    /// `parent_messages` optionally provides context from the parent session
    /// for fork_history support.
    ///
    /// # Errors
    ///
    /// Returns a string error message if spawning fails (limits exceeded, etc.).
    pub async fn spawn_child(
        &self,
        name: &str,
        system_prompt: String,
        depth: i32,
        tool_count: usize,
        parent_messages: Vec<agent_base::ChatMessage>,
    ) -> Result<String, String> {
        let path = AgentPath::root().join(name);

        // 1. Check limits and register
        {
            let mut registry = self.registry.lock().unwrap();
            registry.can_spawn(depth).map_err(|e| e.to_string())?;
            registry
                .register(&path, depth, tool_count)
                .map_err(|e| e.to_string())?;
        }

        // 2. Create mailbox
        let child_mailbox = self
            .mailbox
            .register(&path)
            .ok_or_else(|| "mailbox already exists".to_string())?;

        // 3. Build child AgentRuntime (roll back registry+mailbox on failure)
        let child_runtime = self.build_child_runtime(system_prompt).await.map_err(|e| {
            self.registry.lock().unwrap().close(&path);
            self.mailbox.unregister(&path);
            format!("failed to build child runtime: {}", e)
        })?;

        // 4. Create session for child and pre-fill with parent context
        let session_id = child_runtime.create_session().await;
        self.prefill_child_session(&child_runtime, &session_id, &parent_messages)
            .await
            .map_err(|e| {
                self.registry.lock().unwrap().close(&path);
                self.mailbox.unregister(&path);
                format!("failed to prefill child session: {}", e)
            })?;

        // 5. Create child cancellation token
        let child_cancel = self.root_cancel.child_token();
        {
            let mut cancels = self.child_cancels.lock().unwrap();
            cancels.insert(path.clone(), child_cancel.clone());
        }

        // 6. Spawn child agent event loop
        let agent_path = path.clone();
        let mailbox_for_task = self.mailbox.clone();
        let mailbox_for_close = self.mailbox.clone();
        let event_tx = self.event_tx.lock().unwrap().clone();
        let registry_agent_path = path.clone();

        self.join_set.lock().unwrap().spawn(async move {
            run_child_loop(
                child_mailbox,
                child_runtime,
                session_id,
                agent_path.clone(),
                mailbox_for_task,
                event_tx,
                child_cancel,
            )
            .await;

            // Post close notification when loop exits
            mailbox_for_close.post_result(MailboxResult {
                agent_path,
                status: MailboxStatus::Closed,
                result: None,
            });
        });

        self.registry
            .lock()
            .unwrap()
            .set_status(&registry_agent_path, AgentStatus::Idle);

        Ok(path.to_string())
    }

    /// Spawn a child agent with fork_history support.
    ///
    /// `fork_history`: "none" (default), "all", or a number N for last N turns.
    /// `parent_session_id`: the parent agent's session ID.
    pub async fn spawn_child_with_history(
        &self,
        name: &str,
        system_prompt: String,
        depth: i32,
        tool_count: usize,
        fork_history: Option<String>,
        parent_session_id: &SessionId,
    ) -> Result<String, String> {
        let parent_messages = self
            .resolve_fork_history(fork_history, parent_session_id)
            .await;
        self.spawn_child(name, system_prompt, depth, tool_count, parent_messages)
            .await
    }

    /// Resolve fork_history parameter into a list of parent ChatMessages.
    pub(crate) async fn resolve_fork_history(
        &self,
        fork_history: Option<String>,
        parent_session_id: &SessionId,
    ) -> Vec<agent_base::ChatMessage> {
        use agent_base::ChatMessage;
        let mode = match fork_history.as_deref() {
            None | Some("none") => return vec![],
            Some(s) => s,
        };

        let sm = match self.session_manager.lock().unwrap().as_ref() {
            Some(sm) => sm.clone(),
            None => {
                tracing::warn!("fork_history requested but no session_manager set");
                return vec![];
            }
        };

        // Get all messages from parent session
        let all_messages = match sm.session_or_err(parent_session_id).await {
            Ok(session) => session.chat_messages().to_vec(),
            Err(e) => {
                tracing::warn!(session_id = parent_session_id.id, error = %e, "failed to load parent session for fork_history");
                return vec![];
            }
        };

        if all_messages.is_empty() {
            return vec![];
        }

        // Filter out system messages (child has its own system prompt)
        let non_system: Vec<ChatMessage> = all_messages
            .into_iter()
            .filter(|m| !matches!(m, ChatMessage::System { .. }))
            .collect();

        match mode {
            "all" => non_system,
            n_str => {
                // Parse N: number of recent user/assistant message pairs (turns)
                let n: usize = match n_str.parse() {
                    Ok(n) if n > 0 => n,
                    _ => {
                        tracing::warn!(
                            fork_history = n_str,
                            "invalid fork_history value, treating as 'none'"
                        );
                        return vec![];
                    }
                };

                // Count turns from the end (each turn = user message followed by response)
                let mut turns = 0usize;
                let mut cutoff = non_system.len();
                for (i, msg) in non_system.iter().enumerate().rev() {
                    if matches!(msg, ChatMessage::User { .. }) {
                        turns += 1;
                        if turns >= n {
                            cutoff = i;
                            break;
                        }
                    }
                }
                non_system[cutoff..].to_vec()
            }
        }
    }

    /// Send a message to a child agent (no execution trigger).
    ///
    /// Called by `send_message` tool.
    pub fn send_message(&self, agent_path: &str, message: String) -> Result<bool, String> {
        let path = self.parse_path(agent_path)?;
        Ok(self.mailbox.send_message(&path, message))
    }

    /// Send a task to a child agent (triggers execution).
    ///
    /// Called by `followup_task` tool. Updates status to Running.
    pub fn send_task(
        &self,
        agent_path: &str,
        task: String,
        interrupt: bool,
    ) -> Result<bool, String> {
        let path = self.parse_path(agent_path)?;
        if !self.mailbox.contains(&path) {
            return Err("agent not found".to_string());
        }
        let sent = self.mailbox.send_task(&path, task, interrupt);
        if sent {
            self.registry
                .lock()
                .unwrap()
                .set_status(&path, AgentStatus::Running);
        }
        Ok(sent)
    }

    /// Wait for a result from any or a specific child agent.
    ///
    /// Called by `wait_agent` tool. Blocks until a result arrives or timeout.
    pub async fn wait_for_result(&self, agent_path: Option<&str>, timeout_ms: u64) -> WaitResult {
        let filter_path = match agent_path {
            Some(s) => match AgentPath::parse(s) {
                Some(p) => Some(p),
                None => {
                    return WaitResult {
                        status: "error".to_string(),
                        result: Some(format!("invalid agent path: {}", s)),
                        agent_path: None,
                        has_more: false,
                    };
                }
            },
            None => None,
        };

        let mut seq = self.mailbox.subscribe_seq();
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_millis(timeout_ms);

        loop {
            // Check for existing results
            let result = match &filter_path {
                Some(path) => self.mailbox.try_recv_result(path),
                None => self.mailbox.try_recv_any(),
            };

            if let Some(r) = result {
                let has_more = self.mailbox.total_pending_results() > 0;
                let (status_str, result_text) = match r.status {
                    MailboxStatus::Ok => ("ok".to_string(), r.result),
                    MailboxStatus::Error => ("error".to_string(), r.result),
                    MailboxStatus::Closed => ("closed".to_string(), r.result),
                };
                return WaitResult {
                    status: status_str,
                    result: result_text,
                    agent_path: Some(r.agent_path.to_string()),
                    has_more,
                };
            }

            // Wait for sequence number change or timeout
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return WaitResult {
                    status: "timeout".to_string(),
                    result: None,
                    agent_path: None,
                    has_more: false,
                };
            }

            let remaining = deadline - now;
            tokio::select! {
                _ = seq.changed() => {
                    // Sequence changed — loop back to check results
                    continue;
                }
                _ = tokio::time::sleep(remaining) => {
                    return WaitResult {
                        status: "timeout".to_string(),
                        result: None,
                        agent_path: None,
                        has_more: false,
                    };
                }
            }
        }
    }

    /// Close a child agent.
    ///
    /// Called by `close_agent` tool. Cancels the child's task, removes from
    /// registry, and posts a Closed result.
    pub fn close_agent(&self, agent_path: &str) -> Result<CloseResult, String> {
        let path = self.parse_path(agent_path)?;

        // Get previous status
        let previous_status = {
            let registry = self.registry.lock().unwrap();
            registry
                .get(&path)
                .map(|e| format!("{:?}", e.status).to_lowercase())
                .unwrap_or_else(|| "unknown".to_string())
        };

        // Cancel child token
        {
            let mut cancels = self.child_cancels.lock().unwrap();
            if let Some(token) = cancels.remove(&path) {
                token.cancel();
            }
        }

        // Close in registry
        let existed = { self.registry.lock().unwrap().close(&path).is_some() };

        // Unregister mailbox
        self.mailbox.unregister(&path);

        Ok(CloseResult {
            closed: existed,
            previous_status,
            message: if existed {
                "agent closed".to_string()
            } else {
                "agent not found".to_string()
            },
        })
    }

    /// List all active sub-agents.
    ///
    /// Called by `list_agents` tool.
    pub fn list_agents(&self) -> Vec<AgentInfo> {
        let registry = self.registry.lock().unwrap();
        registry
            .list()
            .into_iter()
            .map(|e| AgentInfo {
                agent_path: e.path.to_string(),
                status: format!("{:?}", e.status).to_lowercase(),
                tool_count: e.tool_count,
            })
            .collect()
    }

    /// Get the mailbox hub (for tools that need it directly).
    pub fn mailbox(&self) -> &Arc<MailboxHub> {
        &self.mailbox
    }

    /// Get reference to the registry.
    pub fn registry(&self) -> &Mutex<AgentRegistry> {
        &self.registry
    }

    /// Cancel all child agents.
    pub fn cancel_all(&self) {
        let mut cancels = self.child_cancels.lock().unwrap();
        for (_, token) in cancels.drain() {
            token.cancel();
        }
    }
}

impl Drop for MultiAgentRuntime {
    fn drop(&mut self) {
        self.cancel_all();
        // Drain any already-completed join handles to detect panics
        let mut js = self.join_set.lock().unwrap();
        while let Some(result) = js.try_join_next() {
            if let Err(e) = result
                && e.is_panic()
            {
                tracing::error!(
                    error = %e,
                    "child agent task panicked"
                );
            }
        }
    }
}

impl MultiAgentRuntime {
    fn parse_path(&self, s: &str) -> Result<AgentPath, String> {
        AgentPath::parse(s).ok_or_else(|| format!("invalid agent path: '{}'", s))
    }

    async fn build_child_runtime(&self, system_prompt: String) -> AgentResult<AgentRuntime> {
        let mut builder = AgentBuilder::new(self.client.clone())
            .system_prompt(system_prompt)
            .approval_handler(Arc::new(DenyAllApprovalHandler))
            .language(self.language.clone());

        // Register business tools (NOT multi-agent tools)
        for tool in &self.business_tools {
            builder = builder.register_tool_arc(tool.clone());
        }

        if let Some(ref recovery) = self.error_recovery {
            builder = builder.error_recovery(recovery.clone());
        }

        builder.build()
    }

    /// Pre-fill a child session with parent conversation context (fork_history).
    ///
    /// Skips system messages and tool-call-only assistant messages. Assistant text
    /// responses and tool results are stored as system messages with labels so the
    /// child sees the context without confusing role semantics.
    pub(crate) async fn prefill_child_session(
        &self,
        child_runtime: &AgentRuntime,
        session_id: &SessionId,
        parent_messages: &[agent_base::ChatMessage],
    ) -> AgentResult<()> {
        use agent_base::ChatMessage;

        for msg in parent_messages {
            match msg {
                ChatMessage::User { content, .. } => {
                    child_runtime.add_user_message(session_id, content).await?;
                }
                ChatMessage::Assistant {
                    content: Some(text),
                    ..
                } => {
                    child_runtime
                        .add_system_message(
                            session_id,
                            format!("[Parent assistant response]: {}", text),
                        )
                        .await?;
                }
                ChatMessage::Assistant { tool_calls, .. } if tool_calls.is_some() => {
                    // Skip tool-call-only messages — parent's tool decisions
                    // don't make sense in the child's context.
                }
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                } => {
                    child_runtime
                        .add_system_message(
                            session_id,
                            format!("[Parent tool result ({}): {}]", tool_call_id, content),
                        )
                        .await?;
                }
                _ => {} // Skip system messages and empty assistant
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result from `wait_for_result()`.
#[derive(Clone, Debug)]
pub struct WaitResult {
    pub status: String,
    pub result: Option<String>,
    pub agent_path: Option<String>,
    pub has_more: bool,
}

/// Result from `close_agent()`.
#[derive(Clone, Debug)]
pub struct CloseResult {
    pub closed: bool,
    pub previous_status: String,
    pub message: String,
}

/// Agent info for `list_agents()`.
#[derive(Clone, Debug, serde::Serialize)]
pub struct AgentInfo {
    pub agent_path: String,
    pub status: String,
    pub tool_count: usize,
}

// ---------------------------------------------------------------------------
// Child agent event loop
// ---------------------------------------------------------------------------

/// Run the child agent's main event loop.
///
/// This function runs inside a tokio task spawned by [`MultiAgentRuntime::spawn_child`].
/// It:
/// 1. Subscribes to child agent events and bridges them to parent
/// 2. Listens for tasks from the mailbox
/// 3. Executes each task via `run_turn`
/// 4. Posts results back via the mailbox
async fn run_child_loop(
    child_mailbox: ChildMailbox,
    child_runtime: AgentRuntime,
    session_id: SessionId,
    agent_path: AgentPath,
    mailbox: Arc<MailboxHub>,
    event_tx: Option<tokio::sync::mpsc::UnboundedSender<RuntimeEvent>>,
    child_cancel: CancellationToken,
) {
    let mut task_rx = child_mailbox.task_rx;

    // Spawn event bridging: forward child events to parent as SubAgentEvent
    if let Some(tx) = event_tx {
        let mut child_events = child_runtime.subscribe_runtime_events();
        let bridge_path = agent_path.to_string();
        let bridge_cancel = child_cancel.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = bridge_cancel.cancelled() => break,
                    event = child_events.recv() => {
                        match event {
                            Ok(event) => {
                                if matches!(event, RuntimeEvent::RunFinished { .. } | RuntimeEvent::RunCancelled { .. }) {
                                    continue;
                                }
                                let _ = tx.send(RuntimeEvent::UserEvent {
                                    session_id: SessionId::new(0),
                                    event: UserEvent::SubAgentEvent {
                                        subagent: bridge_path.clone(),
                                        event: Box::new(event),
                                    },
                                    agent_id: None,
                                    trace_id: None,
                                });
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                                tracing::warn!(
                                    subagent = %bridge_path,
                                    lagged = n,
                                    "child event bridge lagged"
                                );
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
        });
    }

    // Main task loop
    loop {
        tokio::select! {
            _ = child_cancel.cancelled() => {
                break;
            }
            task = task_rx.recv() => {
                match task {
                    Some(task) => {
                        let input = build_child_input(&task);
                        let result = child_runtime.run_turn_collect(
                            session_id.clone(),
                            &input,
                        ).await;

                        match result {
                            Ok((_events, outcome)) => {
                                let summary = summarize_outcome(&outcome);
                                mailbox.post_result(MailboxResult {
                                    agent_path: agent_path.clone(),
                                    status: MailboxStatus::Ok,
                                    result: Some(summary),
                                });
                            }
                            Err(e) => {
                                mailbox.post_result(MailboxResult {
                                    agent_path: agent_path.clone(),
                                    status: MailboxStatus::Error,
                                    result: Some(e.to_string()),
                                });
                            }
                        }
                    }
                    None => break, // task channel closed
                }
            }
        }
    }
}

/// Build the input text for a child agent from a mailbox task.
fn build_child_input(task: &MailboxTask) -> String {
    if task.pending_messages.is_empty() {
        task.task.clone()
    } else {
        let mut parts: Vec<String> = Vec::new();
        for msg in &task.pending_messages {
            parts.push(format!("[Message]: {}", msg));
        }
        parts.push(format!("[Task]: {}", task.task));
        parts.join("\n\n")
    }
}

/// Extract a human-readable summary from a run outcome.
fn summarize_outcome(outcome: &RunOutcome) -> String {
    match outcome {
        RunOutcome::Completed => "task completed".to_string(),
        RunOutcome::Failed { error } => format!("task failed: {}", error),
        RunOutcome::MaxTurnsExceeded { turns } => {
            format!("max turns exceeded ({} turns)", turns)
        }
        RunOutcome::Cancelled => "cancelled".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::RunOutcome;

    // ── summarize_outcome ──

    #[test]
    fn test_summarize_completed() {
        let s = summarize_outcome(&RunOutcome::Completed);
        assert_eq!(s, "task completed");
    }

    #[test]
    fn test_summarize_failed() {
        let outcome = RunOutcome::Failed {
            error: "connection refused".to_string(),
        };
        let s = summarize_outcome(&outcome);
        assert_eq!(s, "task failed: connection refused");
    }

    #[test]
    fn test_summarize_max_turns() {
        let outcome = RunOutcome::MaxTurnsExceeded { turns: 42 };
        let s = summarize_outcome(&outcome);
        assert!(s.contains("max turns exceeded"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_summarize_cancelled() {
        let s = summarize_outcome(&RunOutcome::Cancelled);
        assert_eq!(s, "cancelled");
    }

    // ── build_child_input ──

    #[test]
    fn test_build_child_input_task_only() {
        let task = MailboxTask {
            task: "do work".into(),
            interrupt: true,
            pending_messages: vec![],
        };
        let out = build_child_input(&task);
        assert_eq!(out, "do work");
    }

    #[test]
    fn test_build_child_input_with_pending_messages() {
        let task = MailboxTask {
            task: "do work".into(),
            interrupt: false,
            pending_messages: vec!["context 1".into(), "context 2".into()],
        };
        let out = build_child_input(&task);
        assert!(out.contains("[Message]: context 1"));
        assert!(out.contains("[Message]: context 2"));
        assert!(out.contains("[Task]: do work"));
        // Messages come before task
        let msg_pos = out.find("[Message]:").unwrap();
        let task_pos = out.find("[Task]:").unwrap();
        assert!(msg_pos < task_pos, "messages should precede task");
    }

    #[test]
    fn test_build_child_input_single_message() {
        let task = MailboxTask {
            task: "final task".into(),
            interrupt: true,
            pending_messages: vec!["hint".into()],
        };
        let out = build_child_input(&task);
        assert_eq!(out, "[Message]: hint\n\n[Task]: final task");
    }

    // ── fork_history: resolve_fork_history ──

    /// Mock LLM client for fork_history tests (minimal — never called).
    #[derive(Clone)]
    struct NoopLlmClient;

    #[async_trait::async_trait]
    impl agent_base::LlmClient for NoopLlmClient {
        async fn chat(
            &self,
            _messages: &[agent_base::ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&agent_base::ResponseFormat>,
        ) -> agent_base::AgentResult<serde_json::Value> {
            unimplemented!()
        }

        async fn chat_stream(
            &self,
            _messages: &[agent_base::ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&agent_base::ResponseFormat>,
        ) -> agent_base::AgentResult<
            std::pin::Pin<
                Box<
                    dyn futures_core::Stream<
                            Item = agent_base::AgentResult<agent_base::StreamChunk>,
                        > + Send,
                >,
            >,
        > {
            unimplemented!()
        }

        fn capabilities(&self) -> agent_base::LlmCapabilities {
            agent_base::LlmCapabilities {
                supports_streaming: true,
                supports_tools: false,
                supports_vision: false,
                supports_thinking: false,
                max_context_tokens: None,
                max_output_tokens: None,
            }
        }
    }

    /// Build a MultiAgentRuntime with a parent runtime that has a populated session.
    async fn setup_fork_history_test(
        parent_messages: Vec<agent_base::ChatMessage>,
    ) -> (Arc<MultiAgentRuntime>, agent_base::SessionId) {
        use tokio_util::sync::CancellationToken;

        let llm: Arc<dyn agent_base::LlmClient> = Arc::new(NoopLlmClient);
        let parent_runtime = agent_base::AgentBuilder::new(llm)
            .build()
            .expect("build parent runtime");
        let parent_sid = parent_runtime.create_session().await;

        // Push messages directly into the session's chat_messages vector so
        // we can use proper Assistant/Tool variants (not just System).
        parent_runtime
            .with_session_mut(&parent_sid, |session| {
                session.chat_messages_mut().extend(parent_messages.clone());
            })
            .await
            .unwrap();

        let session_manager = Arc::new(parent_runtime.session_manager().clone());

        let ma_runtime = Arc::new(MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            Arc::new(NoopLlmClient),
            vec![],
            CancellationToken::new(),
            None,
            agent_base::Language::En,
        ));
        ma_runtime.set_session_manager(session_manager);

        (ma_runtime, parent_sid)
    }

    #[tokio::test]
    async fn resolve_fork_history_none_returns_empty() {
        let messages = vec![agent_base::ChatMessage::User {
            content: "hello".into(),
            images: vec![],
            ephemeral: false,
        }];
        let (ma, parent_sid) = setup_fork_history_test(messages).await;

        // None
        let result = ma.resolve_fork_history(None, &parent_sid).await;
        assert!(result.is_empty());

        // Some("none")
        let result = ma
            .resolve_fork_history(Some("none".to_string()), &parent_sid)
            .await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn resolve_fork_history_all_returns_all_non_system() {
        let messages = vec![
            agent_base::ChatMessage::User {
                content: "question 1".into(),
                images: vec![],
                ephemeral: false,
            },
            agent_base::ChatMessage::Assistant {
                content: Some("answer 1".into()),
                reasoning_content: None,
                tool_calls: None,
            },
            agent_base::ChatMessage::User {
                content: "question 2".into(),
                images: vec![],
                ephemeral: false,
            },
            agent_base::ChatMessage::Assistant {
                content: Some("answer 2".into()),
                reasoning_content: None,
                tool_calls: None,
            },
        ];
        let (ma, parent_sid) = setup_fork_history_test(messages).await;

        let result = ma
            .resolve_fork_history(Some("all".to_string()), &parent_sid)
            .await;

        // Should have 4 messages (2 user + 2 assistant) — system messages are filtered out
        assert_eq!(result.len(), 4);
        assert!(matches!(result[0], agent_base::ChatMessage::User { .. }));
        assert!(matches!(
            result[1],
            agent_base::ChatMessage::Assistant { .. }
        ));
        assert!(matches!(result[2], agent_base::ChatMessage::User { .. }));
        assert!(matches!(
            result[3],
            agent_base::ChatMessage::Assistant { .. }
        ));
    }

    #[tokio::test]
    async fn resolve_fork_history_n_turns() {
        // 3 turns: 3 user messages, 3 assistant responses
        let messages = vec![
            agent_base::ChatMessage::User {
                content: "q1".into(),
                images: vec![],
                ephemeral: false,
            },
            agent_base::ChatMessage::Assistant {
                content: Some("a1".into()),
                reasoning_content: None,
                tool_calls: None,
            },
            agent_base::ChatMessage::User {
                content: "q2".into(),
                images: vec![],
                ephemeral: false,
            },
            agent_base::ChatMessage::Assistant {
                content: Some("a2".into()),
                reasoning_content: None,
                tool_calls: None,
            },
            agent_base::ChatMessage::User {
                content: "q3".into(),
                images: vec![],
                ephemeral: false,
            },
            agent_base::ChatMessage::Assistant {
                content: Some("a3".into()),
                reasoning_content: None,
                tool_calls: None,
            },
        ];
        let (ma, parent_sid) = setup_fork_history_test(messages).await;

        // Last 1 turn
        let result = ma
            .resolve_fork_history(Some("1".to_string()), &parent_sid)
            .await;
        assert_eq!(result.len(), 2, "1 turn = user q3 + assistant a3");
        assert!(matches!(result[0], agent_base::ChatMessage::User { .. }));
        assert_eq!(extract_user_content(&result[0]), "q3");

        // Last 2 turns
        let result = ma
            .resolve_fork_history(Some("2".to_string()), &parent_sid)
            .await;
        assert_eq!(result.len(), 4, "2 turns = q2,a2,q3,a3");
    }

    #[tokio::test]
    async fn resolve_fork_history_invalid_number_treats_as_none() {
        let messages = vec![agent_base::ChatMessage::User {
            content: "hello".into(),
            images: vec![],
            ephemeral: false,
        }];
        let (ma, parent_sid) = setup_fork_history_test(messages).await;

        // Invalid number → empty
        let result = ma
            .resolve_fork_history(Some("not-a-number".to_string()), &parent_sid)
            .await;
        assert!(result.is_empty());

        // Zero → empty
        let result = ma
            .resolve_fork_history(Some("0".to_string()), &parent_sid)
            .await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn resolve_fork_history_no_session_manager_returns_empty() {
        use tokio_util::sync::CancellationToken;

        let ma_runtime = MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            Arc::new(NoopLlmClient),
            vec![],
            CancellationToken::new(),
            None,
            agent_base::Language::En,
        );
        // session_manager is NOT set

        let sid = agent_base::SessionId::new(9999);
        let result = ma_runtime
            .resolve_fork_history(Some("all".to_string()), &sid)
            .await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn resolve_fork_history_empty_session_returns_empty() {
        let (ma, parent_sid) = setup_fork_history_test(vec![]).await;

        let result = ma
            .resolve_fork_history(Some("all".to_string()), &parent_sid)
            .await;
        assert!(result.is_empty());
    }

    // ── fork_history: prefill_child_session ──

    #[tokio::test]
    async fn prefill_child_session_user_and_assistant() {
        let llm: Arc<dyn agent_base::LlmClient> = Arc::new(NoopLlmClient);
        let child_runtime = agent_base::AgentBuilder::new(llm)
            .build()
            .expect("build child runtime");
        let child_sid = child_runtime.create_session().await;

        let parent_messages = vec![
            agent_base::ChatMessage::User {
                content: "user question".into(),
                images: vec![],
                ephemeral: false,
            },
            agent_base::ChatMessage::Assistant {
                content: Some("assistant reply".into()),
                reasoning_content: None,
                tool_calls: None,
            },
            agent_base::ChatMessage::Tool {
                tool_call_id: "call_123".into(),
                content: "tool output".into(),
            },
        ];

        // Create a minimal MultiAgentRuntime just to call prefill_child_session
        use tokio_util::sync::CancellationToken;
        let ma_runtime = MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            Arc::new(NoopLlmClient),
            vec![],
            CancellationToken::new(),
            None,
            agent_base::Language::En,
        );

        ma_runtime
            .prefill_child_session(&child_runtime, &child_sid, &parent_messages)
            .await
            .expect("prefill should succeed");

        // Verify the child session contains the pre-filled messages
        let session = child_runtime
            .session(&child_sid)
            .await
            .expect("session exists");
        let msgs = session.chat_messages().to_vec();

        // Should have: user msg + system msg (assistant) + system msg (tool)
        assert_eq!(msgs.len(), 3);
        assert!(matches!(msgs[0], agent_base::ChatMessage::User { .. }));
        assert!(matches!(msgs[1], agent_base::ChatMessage::System { .. }));
        assert!(matches!(msgs[2], agent_base::ChatMessage::System { .. }));
    }

    #[tokio::test]
    async fn prefill_child_session_tool_call_only_skipped() {
        let llm: Arc<dyn agent_base::LlmClient> = Arc::new(NoopLlmClient);
        let child_runtime = agent_base::AgentBuilder::new(llm)
            .build()
            .expect("build child runtime");
        let child_sid = child_runtime.create_session().await;

        // Assistant message with only tool_calls (no text content) should be skipped
        let parent_messages = vec![
            agent_base::ChatMessage::User {
                content: "do something".into(),
                images: vec![],
                ephemeral: false,
            },
            agent_base::ChatMessage::Assistant {
                content: None, // no text — tool call only
                reasoning_content: None,
                tool_calls: Some(vec![]),
            },
        ];

        use tokio_util::sync::CancellationToken;
        let ma_runtime = MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            Arc::new(NoopLlmClient),
            vec![],
            CancellationToken::new(),
            None,
            agent_base::Language::En,
        );

        ma_runtime
            .prefill_child_session(&child_runtime, &child_sid, &parent_messages)
            .await
            .expect("prefill should succeed");

        let session = child_runtime
            .session(&child_sid)
            .await
            .expect("session exists");
        let msgs = session.chat_messages().to_vec();

        // Only the user message — tool-call-only assistant should be skipped
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0], agent_base::ChatMessage::User { .. }));
    }

    #[tokio::test]
    async fn prefill_child_session_empty_vec_noop() {
        let llm: Arc<dyn agent_base::LlmClient> = Arc::new(NoopLlmClient);
        let child_runtime = agent_base::AgentBuilder::new(llm)
            .build()
            .expect("build child runtime");
        let child_sid = child_runtime.create_session().await;

        use tokio_util::sync::CancellationToken;
        let ma_runtime = MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            Arc::new(NoopLlmClient),
            vec![],
            CancellationToken::new(),
            None,
            agent_base::Language::En,
        );

        ma_runtime
            .prefill_child_session(&child_runtime, &child_sid, &[])
            .await
            .expect("prefill should succeed");

        let session = child_runtime
            .session(&child_sid)
            .await
            .expect("session exists");
        let msgs = session.chat_messages().to_vec();

        // System prompt is added but we don't assert exact count — just that no user/injected msgs
        assert!(msgs.is_empty() || matches!(msgs[0], agent_base::ChatMessage::System { .. }));
    }

    fn extract_user_content(msg: &agent_base::ChatMessage) -> &str {
        match msg {
            agent_base::ChatMessage::User { content, .. } => content.as_str(),
            _ => "",
        }
    }
}
