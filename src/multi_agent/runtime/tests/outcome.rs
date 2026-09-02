use super::*;

use crate::multi_agent::mailbox::MailboxTask;
use crate::multi_agent::runtime::outcome::*;

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

// ── build_child_result / extract_last_assistant_message ──

fn text_delta(text: &str, agent_id: Option<&str>) -> agent_base::RuntimeEvent {
    agent_base::RuntimeEvent::TextDelta {
        session_id: agent_base::SessionId::new(1),
        text: text.to_string(),
        agent_id: agent_id.map(|s| s.to_string()),
        trace_id: None,
    }
}

fn tool_started(tool_name: &str) -> agent_base::RuntimeEvent {
    agent_base::RuntimeEvent::ToolCallStarted {
        session_id: agent_base::SessionId::new(1),
        tool_name: tool_name.to_string(),
        args_json: "{}".to_string(),
        agent_id: None,
        trace_id: None,
    }
}

#[test]
fn test_build_child_result_completed_returns_final_text() {
    let events = vec![text_delta("I couldn't ", None), text_delta("delete.", None)];
    assert_eq!(
        build_child_result(&RunOutcome::Completed, &events),
        "I couldn't delete."
    );
}

#[test]
fn test_build_child_result_completed_falls_back_when_no_text() {
    assert_eq!(
        build_child_result(&RunOutcome::Completed, &[]),
        "task completed"
    );
}

#[test]
fn test_extract_last_assistant_message_ignores_subagent_text() {
    let events = vec![
        tool_started("read_file"),
        text_delta("root answer", None),
        text_delta("grandchild", Some("root/child/grandchild")),
    ];
    assert_eq!(extract_last_assistant_message(&events), "root answer");
}

#[test]
fn test_extract_last_assistant_message_skips_earlier_turns() {
    // Simulate: assistant text → tool call → assistant text (final report)
    let events = vec![
        text_delta("thinking about this...", None),
        tool_started("read_file"),
        text_delta("Here is my final analysis.", None),
    ];
    assert_eq!(
        extract_last_assistant_message(&events),
        "Here is my final analysis."
    );
}

#[test]
fn test_extract_last_assistant_message_no_tool_calls() {
    // No tool calls → returns all text (single-turn agent)
    let events = vec![text_delta("I couldn't ", None), text_delta("delete.", None)];
    assert_eq!(
        extract_last_assistant_message(&events),
        "I couldn't delete."
    );
}

#[test]
fn test_build_child_result_failed_keeps_error() {
    let outcome = RunOutcome::Failed {
        error: "boom".to_string(),
    };
    assert_eq!(build_child_result(&outcome, &[]), "task failed: boom");
}

// ── collect_denied_tools ──

fn tool_finished(tool_name: &str, denied: bool) -> agent_base::RuntimeEvent {
    agent_base::RuntimeEvent::ToolCallFinished {
        session_id: agent_base::SessionId::new(1),
        tool_name: tool_name.to_string(),
        summary: "summary".to_string(),
        agent_id: None,
        trace_id: None,
        denied,
    }
}

#[test]
fn test_collect_denied_tools_filters_denied_only() {
    let events = vec![
        tool_finished("read_file", false),
        tool_finished("delete_file", true),
        tool_finished("shell", true),
    ];
    assert_eq!(
        collect_denied_tools(&events),
        vec!["delete_file".to_string(), "shell".to_string()]
    );
}

#[test]
fn test_collect_denied_tools_empty_when_no_denials() {
    let events = vec![
        tool_finished("read_file", false),
        text_delta("all good", None),
    ];
    assert!(collect_denied_tools(&events).is_empty());
}

#[test]
fn test_collect_denied_tools_excludes_grandchild_denials() {
    // A grandchild's denial carries an agent_id and must not be attributed to
    // the child.
    let events = vec![
        agent_base::RuntimeEvent::ToolCallFinished {
            session_id: agent_base::SessionId::new(1),
            tool_name: "grandchild_tool".to_string(),
            summary: "summary".to_string(),
            agent_id: Some("root/child/grandchild".to_string()),
            trace_id: None,
            denied: true,
        },
        tool_finished("child_tool", true),
    ];
    assert_eq!(
        collect_denied_tools(&events),
        vec!["child_tool".to_string()]
    );
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
