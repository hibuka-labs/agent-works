//! `fork_history` resolution: reading the parent session and pre-filling the
//! child session (A+C split from `runtime.rs`).
//!
//! Both halves are the per-spawn context bridge — they touch the optional
//! parent `SessionManager` set by the builder, nothing else.

use agent_base::{AgentResult, AgentRuntime, SessionId};

use super::MultiAgentRuntime;

impl MultiAgentRuntime {
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
                    ..
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
