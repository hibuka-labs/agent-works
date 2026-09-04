//! Pure result-formatting helpers for child runs (A+C split from `runtime.rs`).
//!
//! No access to [`MultiAgentRuntime`](super::MultiAgentRuntime) state — these
//! map a finished run (outcome + collected events) to the strings that travel
//! back to the parent through the mailbox.

use agent_base::{RunOutcome, RuntimeEvent};

use crate::multi_agent::mailbox::{MailboxResult, MailboxStatus, MailboxTask};
use super::watcher::ChildReport;

/// Build the input text for a child agent from a mailbox task.
pub(super) fn build_child_input(task: &MailboxTask) -> String {
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
pub(super) fn summarize_outcome(outcome: &RunOutcome) -> String {
    match outcome {
        RunOutcome::Completed => "task completed".to_string(),
        RunOutcome::Continuing => "continuing".to_string(),
        RunOutcome::Failed { error } => format!("task failed: {}", error),
        RunOutcome::MaxTurnsExceeded { turns } => {
            format!("max turns exceeded ({} turns)", turns)
        }
        RunOutcome::Cancelled => "cancelled".to_string(),
    }
}

/// Extract the child agent's final assistant message from its collected events.
///
/// Walks backwards from the end of the event list, collecting `TextDelta` text
/// until hitting a `ToolCallStarted` boundary. This isolates the child's last
/// reply (the "report") from intermediate tool-use conversation, keeping the
/// result small and focused.
///
/// Excludes sub-sub-agent text (events tagged with an `agent_id`), so the parent
/// only sees this child's direct answer.
pub(super) fn extract_last_assistant_message(events: &[RuntimeEvent]) -> String {
    // Find the last ToolCallStarted — everything after it is the final reply.
    let start = events
        .iter()
        .rposition(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. }))
        .map(|i| i + 1)
        .unwrap_or(0);

    let mut text = String::new();
    for event in &events[start..] {
        if let RuntimeEvent::TextDelta {
            text: delta,
            agent_id,
            ..
        } = event
            && agent_id.is_none()
        {
            text.push_str(delta);
        }
    }
    text
}

/// Collect the names of tools the child attempted but was denied permission to
/// call, from a run's collected events.
///
/// Filters to the child's own events (`agent_id == None`) so grandchild denials
/// — if children ever gain multi-agent tools — are not mis-attributed to the
/// child. Only the child's direct denials matter to the parent.
pub(super) fn collect_denied_tools(events: &[RuntimeEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match e {
            RuntimeEvent::ToolCallFinished {
                tool_name,
                denied: true,
                agent_id: None,
                ..
            } => Some(tool_name.clone()),
            _ => None,
        })
        .collect()
}

/// Build the result string posted back to the parent via the mailbox.
///
/// For a completed task this returns the child's actual final answer (so the
/// parent learns what the child concluded — e.g. "no permission" reports) rather
/// than a coarse "task completed". Other outcomes keep the coarse summary.
pub(super) fn build_child_result(outcome: &RunOutcome, events: &[RuntimeEvent]) -> String {
    match outcome {
        RunOutcome::Completed => {
            let text = extract_last_assistant_message(events);
            if text.trim().is_empty() {
                summarize_outcome(outcome)
            } else {
                text
            }
        }
        _ => summarize_outcome(outcome),
    }
}

/// Format a [`MailboxResult`] into a [`ChildReport`] for the watcher's batch.
///
/// Builds a human-readable message that can be injected into the parent agent's
/// context as a synthetic user message when the fan-in batch completes.
pub fn format_child_result(result: &MailboxResult) -> ChildReport {
    let status_str = match &result.status {
        MailboxStatus::Ok => "ok",
        MailboxStatus::Error => "error",
        MailboxStatus::Closed => "closed",
    };

    let result_text = result.result.as_deref().unwrap_or("");
    let path = result.agent_path.to_string();

    let message = match &result.status {
        MailboxStatus::Ok => {
            if result_text.is_empty() {
                format!("[子 agent {} 已完成]", path)
            } else {
                format!("[子 agent {} 已完成]\n{}", path, result_text)
            }
        }
        MailboxStatus::Error => {
            if result_text.is_empty() {
                format!("[子 agent {} 执行出错]", path)
            } else {
                format!("[子 agent {} 执行出错]\n{}", path, result_text)
            }
        }
        MailboxStatus::Closed => {
            format!("[子 agent {} 已关闭]", path)
        }
    };

    ChildReport {
        agent_path: path,
        status: status_str.to_string(),
        result: result.result.clone(),
        message,
    }
}
