//! Filtering helpers for context compression.
//!
//! - [`is_summary_message`] — detect messages produced by a previous summarization pass.
//! - [`split_system_prompt`] — peel off leading system messages from a message list.

use agent_base::ChatMessage;

/// Prefix injected in front of every summary produced by the compressor.
///
/// The summarizer prepends this block to the summary text so that subsequent
/// compression passes can detect (and skip) old summaries, preventing
/// summary-of-summary degradation.
pub const SUMMARY_PREFIX: &str = "Another language model started to solve this problem \
and produced a summary of its thinking process. You also have access to the state \
of the tools that were used by that language model. Use this to build on the work \
that has already been done and avoid duplicating work. Here is the summary produced \
by the other language model, use the information in this summary to assist with \
your own analysis:";

/// Returns `true` if `msg` is a user message whose content starts with
/// [`SUMMARY_PREFIX`] — i.e. a summary injected by a previous compression pass.
///
/// This is used to filter old summaries out before re-summarising, which
/// prevents summary-of-summary drift.
pub fn is_summary_message(msg: &ChatMessage) -> bool {
    match msg {
        ChatMessage::User { content, .. } => content.starts_with(SUMMARY_PREFIX),
        _ => false,
    }
}

/// Split a message slice into `(system_messages, rest)`.
///
/// Takes all **leading** contiguous `System` messages (supporting multi-message
/// system prompts) and returns them separately from the remaining conversation
/// messages. This is the recommended way to peel off the system prompt before
/// compressing the conversation body.
///
/// # Examples
///
/// ```ignore
/// let (sys, body) = split_system_prompt(&messages);
/// // sys  = [System("You are ..."), System("Extra instructions")]
/// // body = [User("hi"), Assistant("hello"), ...]
/// ```
pub fn split_system_prompt(messages: &[ChatMessage]) -> (&[ChatMessage], &[ChatMessage]) {
    let mut end = 0;
    for msg in messages {
        if matches!(msg, ChatMessage::System { .. }) {
            end += 1;
        } else {
            break;
        }
    }
    messages.split_at(end)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_summary_message ───────────────────────────────────────────────────

    #[test]
    fn test_summary_message_detected() {
        let msg = ChatMessage::user(format!("{SUMMARY_PREFIX}\nThe user asked for X."));
        assert!(is_summary_message(&msg));
    }

    #[test]
    fn test_normal_user_message_not_detected() {
        let msg = ChatMessage::user("hello, can you help me?");
        assert!(!is_summary_message(&msg));
    }

    #[test]
    fn test_assistant_message_not_detected() {
        let msg = ChatMessage::assistant("sure, I can help.");
        assert!(!is_summary_message(&msg));
    }

    #[test]
    fn test_system_message_not_detected() {
        let msg = ChatMessage::system("You are a helpful assistant.");
        assert!(!is_summary_message(&msg));
    }

    #[test]
    fn test_tool_message_not_detected() {
        let msg = ChatMessage::tool("call_1", "result data");
        assert!(!is_summary_message(&msg));
    }

    #[test]
    fn test_partial_prefix_not_detected() {
        // Only a substring of the prefix — must NOT match.
        let msg = ChatMessage::user("Another language model started");
        assert!(!is_summary_message(&msg));
    }

    // ── split_system_prompt ──────────────────────────────────────────────────

    #[test]
    fn test_split_single_system() {
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        let (sys, body) = split_system_prompt(&msgs);
        assert_eq!(sys.len(), 1);
        assert_eq!(body.len(), 2);
    }

    #[test]
    fn test_split_multiple_consecutive_systems() {
        let msgs = vec![
            ChatMessage::system("sys1"),
            ChatMessage::system("sys2"),
            ChatMessage::system("sys3"),
            ChatMessage::user("hi"),
        ];
        let (sys, body) = split_system_prompt(&msgs);
        assert_eq!(sys.len(), 3);
        assert_eq!(body.len(), 1);
    }

    #[test]
    fn test_split_no_system() {
        let msgs = vec![ChatMessage::user("hi"), ChatMessage::assistant("hello")];
        let (sys, body) = split_system_prompt(&msgs);
        assert_eq!(sys.len(), 0);
        assert_eq!(body.len(), 2);
    }

    #[test]
    fn test_split_all_system() {
        let msgs = vec![ChatMessage::system("s1"), ChatMessage::system("s2")];
        let (sys, body) = split_system_prompt(&msgs);
        assert_eq!(sys.len(), 2);
        assert!(body.is_empty());
    }

    #[test]
    fn test_split_empty() {
        let msgs: Vec<ChatMessage> = vec![];
        let (sys, body) = split_system_prompt(&msgs);
        assert!(sys.is_empty());
        assert!(body.is_empty());
    }

    #[test]
    fn test_split_non_contiguous_system_not_merged() {
        // A system message after a non-system one must NOT be included.
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
            ChatMessage::system("late sys"),
        ];
        let (sys, body) = split_system_prompt(&msgs);
        assert_eq!(sys.len(), 1, "only the leading contiguous system block");
        assert_eq!(body.len(), 2);
    }
}
