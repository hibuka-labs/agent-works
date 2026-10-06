//! `memory_read`: read one memory's content by name, paginated.
//!
//! Paginated on purpose: the pipeline rejects tool output over
//! `max_tool_output_chars` instead of truncating it (design §6.5), so returning
//! a whole file turns any oversized memory into an unreadable hard error —
//! the model gets `ToolOutputTooLarge` and no way to page in. Caps run as low
//! as 4,000 chars (phi-agent's default; phimint sets 16,000), so every
//! response is bounded at [`DEFAULT_READ_LIMIT`] chars and carries the offset
//! to continue from — same shape as `history.read_item`.

use std::sync::Arc;

use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::memory::{MemoryStore, validate_memory_name};

/// Default page size. Stays under every family output cap (the smallest is
/// phi-agent's 4,000) so a single call can never be rejected as too large.
const DEFAULT_READ_LIMIT: usize = 3000;

/// Read the raw memory file (frontmatter + body) for `name`.
pub struct MemoryReadTool {
    store: Arc<MemoryStore>,
}

impl MemoryReadTool {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for MemoryReadTool {
    fn name(&self) -> &'static str {
        "memory_read"
    }

    fn description(&self) -> &'static str {
        "Read one memory's content (frontmatter + markdown body) by its name. Large memories are \
         paginated at 3000 characters — use offset_chars to continue reading. Use after spotting a \
         relevant entry in the memory index — recall is selective: read only the memories that \
         matter for the current task."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^[a-z0-9][a-z0-9-]*$",
                    "description": "memory name from the index (the link target without .md)"
                },
                "offset_chars": {
                    "type": "integer",
                    "description": "Character offset to start reading from. Default 0."
                },
                "limit_chars": {
                    "type": "integer",
                    "description": "Max characters to read. Default: 3000 (paginated). Cannot exceed 3000."
                }
            },
            "required": ["name"],
            "additionalProperties": false
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError::internal("memory_read requires a string `name` argument"))?;
        validate_memory_name(name)?;
        let raw = self.store.read_memory(name)?;

        let total_len = raw.chars().count();
        let offset_chars = args
            .get("offset_chars")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(0);
        // The caller may shrink the page but never grow it past the default:
        // a larger page would just recreate the `ToolOutputTooLarge` failure
        // this pagination exists to prevent.
        let limit_chars = args
            .get("limit_chars")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_READ_LIMIT)
            .clamp(1, DEFAULT_READ_LIMIT);

        let start = offset_chars.min(total_len);
        let end = start.saturating_add(limit_chars).min(total_len);
        let content = slice_chars(&raw, start, end);
        let header = if start > 0 || end < total_len {
            format!("[chars {}..{}/{}]\n", start, end, total_len)
        } else {
            String::new()
        };
        let continuation = if end < total_len {
            format!("\n[use offset_chars={} to continue reading]", end)
        } else {
            String::new()
        };
        Ok(vec![Content::text(format!(
            "{}{}{}",
            header, content, continuation
        ))])
    }
}

/// Slice `raw` by character range.
///
/// Characters, not bytes: memory bodies are largely CJK, and a byte range that
/// lands mid-sequence panics instead of returning a boundary-mangled string.
fn slice_chars(raw: &str, start: usize, end: usize) -> String {
    raw.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

// `AgentError` is used through the `internal` constructor in `call` above.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_store() -> (tempfile::TempDir, Arc<MemoryStore>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(MemoryStore::new(dir.path().join("memory"), "MEMORY.md"));
        (dir, store)
    }

    #[tokio::test]
    async fn reads_existing_memory_raw_content() {
        let (_dir, store) = temp_store();
        store
            .write_memory("my-note", "a note", "project", "remember this", None)
            .unwrap();

        let out = MemoryReadTool::new(Arc::clone(&store))
            .call(&json!({"name": "my-note"}), &ToolContext::for_test())
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.starts_with("---\n"), "raw file, not a summary: {text}");
        assert!(text.contains("name: my-note"));
        assert!(text.contains("remember this"));
    }

    #[tokio::test]
    async fn missing_memory_is_error() {
        let (_dir, store) = temp_store();
        let err = MemoryReadTool::new(store)
            .call(&json!({"name": "ghost"}), &ToolContext::for_test())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }

    #[tokio::test]
    async fn rejects_path_escape_name() {
        let (_dir, store) = temp_store();
        let err = MemoryReadTool::new(store)
            .call(
                &json!({"name": "../../etc/passwd"}),
                &ToolContext::for_test(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid memory name"), "{err}");
    }

    #[tokio::test]
    async fn missing_name_arg_is_error() {
        let (_dir, store) = temp_store();
        let err = MemoryReadTool::new(store)
            .call(&json!({}), &ToolContext::for_test())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("name"), "{err}");
    }

    #[tokio::test]
    async fn large_memory_is_paginated_not_rejected() {
        // Regression: an oversized memory used to come back as
        // `Tool 'memory_read' output exceeds the 16000-char limit` — the model
        // got an error instead of content and had no way to page in.
        let (_dir, store) = temp_store();
        store
            .write_memory("big", "big note", "project", &"x".repeat(20_000), None)
            .unwrap();

        let out = MemoryReadTool::new(Arc::clone(&store))
            .call(&json!({"name": "big"}), &ToolContext::for_test())
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(
            text.chars().count() < 16_000,
            "page must fit the smallest cap, got {} chars",
            text.chars().count()
        );
        assert!(text.starts_with("[chars 0..3000/"), "{text:.40}");
        assert!(
            text.contains("[use offset_chars=3000 to continue reading]"),
            "no continuation hint: {text:.40}"
        );
    }

    #[tokio::test]
    async fn pagination_continues_at_the_requested_offset() {
        let (_dir, store) = temp_store();
        store
            .write_memory("big", "big note", "project", &"x".repeat(20_000), None)
            .unwrap();

        let out = MemoryReadTool::new(Arc::clone(&store))
            .call(
                &json!({"name": "big", "offset_chars": 3000, "limit_chars": 100}),
                &ToolContext::for_test(),
            )
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.starts_with("[chars 3000..3100/"), "{text:.40}");
        assert!(text.contains(&"x".repeat(100)), "page body missing");
    }

    #[tokio::test]
    async fn offsets_are_characters_not_bytes() {
        // CJK is 3 bytes/char. Byte semantics would panic on a mid-sequence
        // offset and, where the offset happened to land on a boundary, return
        // 10 chars instead of 30 — either way the exact char count catches it.
        let (_dir, store) = temp_store();
        store
            .write_memory("cjk", "中文笔记", "project", &"忆".repeat(5000), None)
            .unwrap();

        // 500 sits inside the body (frontmatter is far shorter), so the whole
        // page is body content.
        let out = MemoryReadTool::new(Arc::clone(&store))
            .call(
                &json!({"name": "cjk", "offset_chars": 500, "limit_chars": 30}),
                &ToolContext::for_test(),
            )
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert_eq!(
            text.chars().filter(|c| *c == '忆').count(),
            30,
            "expected exactly 30 body chars: {text:.60}"
        );
    }

    #[tokio::test]
    async fn oversized_limit_is_clamped_to_the_default_page() {
        let (_dir, store) = temp_store();
        store
            .write_memory("big", "big note", "project", &"x".repeat(20_000), None)
            .unwrap();

        let out = MemoryReadTool::new(Arc::clone(&store))
            .call(
                &json!({"name": "big", "limit_chars": 100_000}),
                &ToolContext::for_test(),
            )
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.starts_with("[chars 0..3000/"), "{text:.40}");
        assert!(
            text.chars().count() < 16_000,
            "clamp must keep the page under the cap, got {}",
            text.chars().count()
        );
    }

    #[test]
    fn slice_chars_handles_multibyte_boundaries() {
        assert_eq!(slice_chars("记忆文件测试", 0, 2), "记忆");
        assert_eq!(slice_chars("记忆文件测试", 2, 5), "文件测");
        // Past the end clamps instead of panicking.
        assert_eq!(slice_chars("记忆", 5, 9), "");
    }
}
