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
        "List all memories by returning the current MEMORY.md index (one row per memory: name \
         plus one-line description). The index embedded in your system prompt is a startup \
         snapshot — call this to refresh it when unsure whether a memory exists."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }

    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let index = self.store.read_index();
        let trimmed = index.trim();
        if trimmed.is_empty() {
            Ok(vec![Content::text(
                "No memories stored yet for this workspace. Use memory_write to create one.",
            )])
        } else {
            Ok(vec![Content::text(index)])
        }
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
        assert!(text.contains("[first](alpha.md) — first"), "{text}");
        assert!(text.contains("[second](beta.md) — second"), "{text}");
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
