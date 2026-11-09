use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use agent_base::ChatMessage;

/// Total character length of a message's text content.
///
/// Used for size-based comparison (before vs after compression) rather than
/// token estimation.
pub(super) fn message_content_len(msg: &ChatMessage) -> usize {
    match msg {
        ChatMessage::System { content, .. } => content.len(),
        ChatMessage::User { content, .. } => content.len(),
        ChatMessage::Assistant {
            content,
            reasoning_content,
            tool_calls,
            thinking_signature: _,
        } => {
            let mut len = content.as_deref().map(|c| c.len()).unwrap_or(0);
            if let Some(rc) = reasoning_content {
                len += rc.len();
            }
            if let Some(calls) = tool_calls {
                for c in calls {
                    len += c.id.len() + c.name.len() + c.arguments.len();
                }
            }
            len
        }
        ChatMessage::Tool {
            tool_call_id,
            content,
            ..
        } => tool_call_id.len() + content.len(),
        ChatMessage::Custom { role, data } => role.len() + data.to_string().len(),
    }
}

/// Walk `cut` backward until the boundary is tool-pairing safe.
///
/// A cut is *unsafe* when:
/// - the message just left of the cut is an `Assistant` with pending tool calls, or
/// - the message just right of the cut is a `Tool` result.
///
/// In either case we move the cut leftward until neither condition holds,
/// ensuring the resulting message list never has orphaned `Tool` messages or
/// `Assistant{tool_calls}` whose results are missing.
pub fn safe_cut_index(messages: &[ChatMessage], start: usize, mut cut: usize) -> usize {
    while cut > start {
        let left_is_tool_call = matches!(
            messages[cut - 1],
            ChatMessage::Assistant {
                tool_calls: Some(_),
                ..
            }
        );
        let right_is_tool = matches!(messages[cut], ChatMessage::Tool { .. });
        if !left_is_tool_call && !right_is_tool {
            break;
        }
        cut -= 1;
    }
    cut
}

/// CJK-aware token estimate for a single message.
///
/// Delegates to agent-base's `estimate_messages_tokens` — the single source
/// of truth for pricing (its per-message core, `message_tokens`, is
/// `pub(crate)`, but the list form is public and this is a one-element
/// list). A hand-mirrored copy used to live here; it priced pictures at a
/// flat 85 tokens (and tool pictures at zero), drifting from the engine's
/// own budgeting every time that formula changed.
pub(crate) fn estimate_message_tokens(msg: &ChatMessage) -> usize {
    agent_base::estimate_messages_tokens(std::slice::from_ref(msg))
}

/// Estimate total tokens across a message list.
#[allow(dead_code)]
fn estimate_total_tokens(messages: &[ChatMessage]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Hash a string into a `u64` using the default hasher.
pub(super) fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Serialise a block of messages into a compact human-readable transcript.
///
/// Each message is rendered as `[role] content` with individual fields
/// truncated.  The entire output is then truncated to `max_chars`.
pub fn serialize_block(messages: &[&ChatMessage], max_chars: usize) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(messages.len());
    for msg in messages {
        let line = match msg {
            ChatMessage::System { content, .. } => {
                format!("[system] {}", truncate_str(content, 400))
            }
            ChatMessage::User { content, .. } => format!("[user] {}", truncate_str(content, 400)),
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => match tool_calls {
                Some(calls) if !calls.is_empty() => {
                    let calls: Vec<String> = calls
                        .iter()
                        .map(|c| format!("{}({})", c.name, truncate_str(&c.arguments, 150)))
                        .collect();
                    format!("[assistant tool_call] {}", calls.join("; "))
                }
                _ => format!(
                    "[assistant] {}",
                    content
                        .as_deref()
                        .map(|c| truncate_str(c, 400))
                        .unwrap_or_default()
                ),
            },
            ChatMessage::Tool {
                tool_call_id,
                content,
                ..
            } => format!(
                "[tool:{}] {}",
                truncate_str(tool_call_id, 50),
                truncate_str(content, 300)
            ),
            ChatMessage::Custom { role, data } => {
                format!("[custom:{}] {}", role, truncate_str(&data.to_string(), 400))
            }
        };
        parts.push(line);
    }
    let joined = parts.join("\n");
    truncate_str(&joined, max_chars)
}

/// Truncate a string to `max_chars`, appending `…` if truncated.
pub fn truncate_str(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars).collect();
    format!("{head}…")
}

/// Serialise messages into a transcript and truncate to `max_chars`.
pub(super) fn truncate_to_chars(messages: &[&ChatMessage], max_chars: usize) -> String {
    serialize_block(messages, max_chars)
}

/// Truncate a list of user messages to fit within `max_tokens`.
///
/// Preserves messages from the *end* (most recent) and drops from the start
/// when the budget is exceeded.  This mirrors codex's approach: recent user
/// instructions are more valuable than older ones.
pub(super) fn truncate_user_messages<'a>(
    messages: &[&'a ChatMessage],
    max_tokens: usize,
) -> Vec<&'a ChatMessage> {
    if max_tokens == 0 {
        return Vec::new();
    }
    let mut result: Vec<&ChatMessage> = Vec::with_capacity(messages.len());
    let mut remaining = max_tokens;
    // Walk from end to start, keeping messages that fit.
    for msg in messages.iter().rev() {
        let tokens = estimate_message_tokens(msg);
        if tokens <= remaining {
            result.push(msg);
            remaining -= tokens;
        }
        // Skip oversized messages — don't break, earlier messages may still fit.
    }
    result.reverse();
    result
}

// ── ContextCompaction trait implementation ──────────────────────────────────
