//! `memory_delete`: remove a memory file and its index row.

use std::sync::Arc;

use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::memory::{MemoryStore, validate_memory_name};

/// Delete the memory `name`: remove `<name>.md`, then its MEMORY.md row.
pub struct MemoryDeleteTool {
    store: Arc<MemoryStore>,
}

impl MemoryDeleteTool {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for MemoryDeleteTool {
    fn name(&self) -> &'static str {
        "memory_delete"
    }

    fn description(&self) -> &'static str {
        "Delete one memory: remove its file and its MEMORY.md index row together.\n\n\
         Use it for something that is no longer true and cannot be salvaged — a project status \
         that has moved on, a rule the user has withdrawn, a memory of your own that turns out to \
         be wrong. The pair is the point: a file deleted without its row leaves a name pointing \
         at nothing, and a row deleted without its file leaves the memory unreadable and \
         unlisted.\n\n\
         Do NOT delete because the memory is out of date but still true — `memory_write` updates \
         it, and a deleted history is not a corrected one. Do NOT delete on a guess: this is \
         permanent and there is no undo. When the user is vague about which memory, list first \
         and say which you mean to remove.\n\n\
         Deleting the right thing at the wrong time is still a loss: a memory that looks stale \
         in one session may be exactly what a later one needs. Remove what is false, not what is \
         merely old."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^[a-z0-9][a-z0-9-]*$",
                    "description": "memory name from the index (the link target without .md)"
                }
            },
            "required": ["name"],
            "additionalProperties": false
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let name = args.get("name").and_then(Value::as_str).ok_or_else(|| {
            AgentError::internal("memory_delete requires a string `name` argument")
        })?;
        validate_memory_name(name)?;
        self.store.delete_memory(name)?;
        Ok(vec![Content::text(format!(
            "Memory `{name}` deleted; the index row was removed."
        ))])
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
    async fn deletes_file_and_index_row() {
        let (_dir, store) = temp_store();
        store
            .write_memory("gone", "d1", "user", "b1", None)
            .unwrap();
        store
            .write_memory("kept", "d2", "user", "b2", None)
            .unwrap();

        let out = MemoryDeleteTool::new(Arc::clone(&store))
            .call(&json!({"name": "gone"}), &ToolContext::for_test())
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        assert!(text.contains("deleted"), "{text}");

        assert!(!store.read_memory("gone").is_ok(), "file must be gone");
        let index = store.read_index();
        assert!(!index.contains("(gone.md)"), "{index}");
        assert!(index.contains("(kept.md)"), "{index}");
    }

    #[tokio::test]
    async fn deleting_missing_memory_is_error() {
        let (_dir, store) = temp_store();
        let err = MemoryDeleteTool::new(store)
            .call(&json!({"name": "ghost"}), &ToolContext::for_test())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
    }

    #[tokio::test]
    async fn rejects_path_escape_name() {
        let (_dir, store) = temp_store();
        let err = MemoryDeleteTool::new(store)
            .call(&json!({"name": "../escape"}), &ToolContext::for_test())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid memory name"), "{err}");
    }

    #[tokio::test]
    async fn missing_name_arg_is_error() {
        let (_dir, store) = temp_store();
        let err = MemoryDeleteTool::new(store)
            .call(&json!({}), &ToolContext::for_test())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("name"), "{err}");
    }
}
