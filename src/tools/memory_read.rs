//! `memory_read`: read one memory's full content by name.

use std::sync::Arc;

use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::memory::{MemoryStore, validate_memory_name};

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
        "Read one memory's full content (frontmatter + markdown body) by its name. Use after \
         spotting a relevant entry in the memory index — recall is selective: read only the \
         memories that matter for the current task."
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
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| AgentError::internal("memory_read requires a string `name` argument"))?;
        validate_memory_name(name)?;
        let raw = self.store.read_memory(name)?;
        Ok(vec![Content::text(raw)])
    }
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
}
