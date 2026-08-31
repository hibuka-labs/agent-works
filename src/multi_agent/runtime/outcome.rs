//! Pure result-formatting helpers for child runs (A+C split from `runtime.rs`).
//!
//! No access to [`MultiAgentRuntime`](super::MultiAgentRuntime) state — these
//! map a finished run (outcome + collected events) to the strings that travel
//! back to the parent through the mailbox.

use agent_base::{RunOutcome, RuntimeEvent};

use crate::multi_agent::mailbox::MailboxTask;

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

/// Extract the child agent's own assistant text from its collected events.
///
/// Excludes sub-sub-agent text (events tagged with an `agent_id`), so the parent
/// only sees this child's direct answer.
pub(super) fn extract_assistant_text(events: &[RuntimeEvent]) -> String {
    let mut text = String::new();
    for event in events {
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
            let text = extract_assistant_text(events);
            if text.trim().is_empty() {
                summarize_outcome(outcome)
            } else {
                text
            }
        }
        _ => summarize_outcome(outcome),
    }
}
