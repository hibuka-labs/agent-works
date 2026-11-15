//! `memory_list`: return the current MEMORY.md index.
//!
//! The startup prompt carries an index *snapshot*; this tool is the refresh
//! path for memories written after startup (by this session or Claude Code).

use std::sync::Arc;

use agent_base::{AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::memory::MemoryStore;

/// Return the full index text so the LLM can pick relevant entries by
/// description.
pub struct MemoryListTool {
    store: Arc<MemoryStore>,
}

impl MemoryListTool {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for MemoryListTool {
    fn name(&self) -> &'static str {
        "memory_list"
    }

    fn description(&self) -> &'static str {
        "List the memories on file: one row per memory (name plus one-line description). \
         Nothing is injected into your prompt, so this is the only way to see what is \
         remembered. Pass `query` to narrow to the rows mentioning it — a plain \
         case-insensitive substring match on the row. The listing is trimmed to the tool \
         output budget; when rows are dropped the tail says how many, and narrowing with \
         `query` (or calling `memory_read` on a name) is how you see the rest."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Return only the rows mentioning this, matched case-insensitively against the row (name and description). Omit for the whole listing."
                }
            },
            "additionalProperties": false
        })
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let index = self.store.read_index();
        let trimmed = index.trim();
        if trimmed.is_empty() {
            return Ok(vec![Content::text(
                "No memories stored yet for this workspace. Use memory_write to create one.",
            )]);
        }

        let query = args
            .get("query")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let needle = query.map(str::to_lowercase);

        let rows: Vec<&str> = trimmed
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .filter(|l| match &needle {
                Some(q) => l.to_lowercase().contains(q),
                None => true,
            })
            .collect();

        if rows.is_empty() {
            return Ok(vec![Content::text(format!(
                "No memory matches {:?}. Call memory_list with no query to see every entry, \
                 or memory_write to create one.",
                query.unwrap_or("")
            ))]);
        }

        // Fill to the output budget at a whole-row boundary, reserving room for
        // the tail that says what was dropped. Without this the whole listing is
        // rejected above the cap rather than trimmed to it.
        const TAIL_RESERVE: usize = 220;
        let cap = ctx.max_output_chars.unwrap_or(usize::MAX);
        let room = cap.saturating_sub(TAIL_RESERVE);
        let mut body = String::new();
        let mut shown = 0usize;
        for row in &rows {
            let line = format!("{row}\n");
            if body.len() + line.len() > room {
                break;
            }
            body.push_str(&line);
            shown += 1;
        }

        let omitted = rows.len() - shown;
        if omitted == 0 {
            return Ok(vec![Content::text(body)]);
        }
        let tail = format!(
            "(… {} of {} rows omitted to fit the output budget — narrow with a `query`, or \
             call memory_read on a name.)",
            omitted,
            rows.len()
        );
        body.push_str(&tail);
        Ok(vec![Content::text(body)])
    }
}

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
    async fn returns_index_rows_after_writes() {
        let (_dir, store) = temp_store();
        store
            .write_memory("alpha", "first", "user", "b", None)
            .unwrap();
        store
            .write_memory("beta", "second", "project", "b", None)
            .unwrap();

        let out = MemoryListTool::new(Arc::clone(&store))
            .call(&json!({}), &ToolContext::for_test())
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.contains("[alpha](alpha.md) — first"), "{text}");
        assert!(text.contains("[beta](beta.md) — second"), "{text}");
        assert_eq!(text.lines().count(), 2, "{text}");
    }

    #[tokio::test]
    async fn empty_memory_dir_reports_no_memories() {
        let (_dir, store) = temp_store();
        let out = MemoryListTool::new(store)
            .call(&json!({}), &ToolContext::for_test())
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.contains("No memories"), "{text}");
    }

    #[tokio::test]
    async fn reflects_deletes() {
        let (_dir, store) = temp_store();
        store
            .write_memory("alpha", "first", "user", "b", None)
            .unwrap();
        store.delete_memory("alpha").unwrap();

        let out = MemoryListTool::new(store)
            .call(&json!({}), &ToolContext::for_test())
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.contains("No memories"), "{text}");
    }

    /// Value: protects=the listing fits the output budget instead of being
    /// rejected whole; fails_when=the cap check is dropped and an index over
    /// `max_output_chars` fails with "output exceeds the limit".
    /// why_new=the four tests above cover content, not size — none of them
    /// makes an index bigger than the budget.
    /// seam=none; no_claim=what the model does with a truncated listing.
    #[tokio::test]
    async fn the_listing_stops_at_the_output_budget_and_names_what_it_dropped() {
        let (_dir, store) = temp_store();
        for i in 0..40 {
            store
                .write_memory(
                    &format!("m{i:02}"),
                    &format!("description number {i:02}"),
                    "project",
                    "body",
                    None,
                )
                .unwrap();
        }
        let mut ctx = ToolContext::for_test();
        ctx.max_output_chars = Some(400);
        let out = MemoryListTool::new(Arc::clone(&store))
            .call(&json!({}), &ctx)
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.len() <= 400, "over budget: {} chars", text.len());
        assert!(text.contains("omitted"), "{text}");
        // Rows are whole: no entry cut mid-line.
        for line in text.lines() {
            assert!(
                line.contains(".md)") || line.contains("omitted"),
                "row cut mid-line: {line}"
            );
        }
    }

    /// Value: protects=the model can find one memory among many without
    /// reading all of them; fails_when=the query argument is dropped and the
    /// only way to see a memory is the truncated whole listing.
    /// why_new=no existing test narrows the listing.
    /// seam=none; no_claim=recall quality (substring match on the row only).
    #[tokio::test]
    async fn a_query_narrows_the_rows() {
        let (_dir, store) = temp_store();
        store
            .write_memory("alpha", "release script notes", "project", "b", None)
            .unwrap();
        store
            .write_memory("beta", "user prefers short answers", "user", "b", None)
            .unwrap();

        let out = MemoryListTool::new(Arc::clone(&store))
            .call(&json!({"query": "release"}), &ToolContext::for_test())
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.contains("alpha.md"), "{text}");
        assert!(!text.contains("beta.md"), "{text}");
    }

    #[tokio::test]
    async fn a_query_matching_nothing_says_so_and_offers_the_ways_out() {
        let (_dir, store) = temp_store();
        store
            .write_memory("alpha", "release script notes", "project", "b", None)
            .unwrap();
        let out = MemoryListTool::new(Arc::clone(&store))
            .call(&json!({"query": "zzz-nothing"}), &ToolContext::for_test())
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.contains("No memory"), "{text}");
        assert!(
            text.contains("memory_list"),
            "must name the unfiltered way out: {text}"
        );
        assert!(
            text.contains("memory_write"),
            "and the create-one way out: {text}"
        );
    }

    /// The tool's face used to say the index was "embedded in your system
    /// prompt" — untrue since the product stopped injecting it, and the model
    /// would look for a snapshot that is not there.
    #[test]
    fn the_description_does_not_promise_a_prompt_snapshot() {
        let d = MemoryListTool::new(Arc::new(MemoryStore::new(
            std::path::PathBuf::from("/tmp/unused"),
            "MEMORY.md",
        )))
        .description()
        .to_string();
        assert!(
            !d.contains("embedded in your system prompt"),
            "there is no prompt snapshot any more: {d}"
        );
        assert!(
            d.contains("query"),
            "the face must document the filter: {d}"
        );
    }

    #[tokio::test]
    async fn extra_args_are_ignored() {
        let (_dir, store) = temp_store();
        store
            .write_memory("alpha", "first", "user", "b", None)
            .unwrap();
        let out = MemoryListTool::new(store)
            .call(&json!({"unused": true}), &ToolContext::for_test())
            .await
            .unwrap();
        assert!(crate::tools::test_support::first_text(&out).contains("alpha.md"));
    }
}
