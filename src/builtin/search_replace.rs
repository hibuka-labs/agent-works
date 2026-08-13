use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::path_util::validate_path;
use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};

pub struct SearchReplaceTool {
    pub workspace: PathBuf,
}

#[async_trait]
impl Tool for SearchReplaceTool {
    fn name(&self) -> &'static str {
        "search_replace"
    }

    fn description(&self) -> &'static str {
        "Search and replace text in a file"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file"
                },
                "old_str": {
                    "type": "string",
                    "description": "Text to search for (first occurrence will be replaced)"
                },
                "new_str": {
                    "type": "string",
                    "description": "Text to replace with"
                }
            },
            "required": ["path", "old_str", "new_str"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| AgentError::internal("missing 'path' argument"))?;

        let old_str = args["old_str"]
            .as_str()
            .ok_or_else(|| AgentError::internal("missing 'old_str' argument"))?;

        let new_str = args["new_str"]
            .as_str()
            .ok_or_else(|| AgentError::internal("missing 'new_str' argument"))?;

        // Short-circuit when old and new are identical — no I/O needed
        if old_str == new_str {
            return Ok(vec![Content::text(format!(
                "No changes needed in {} (old_str == new_str)",
                path
            ))]);
        }

        let full_path = validate_path(&self.workspace, path)?;

        tracing::debug!(file = %path, "search replace start");
        let content = tokio::fs::read_to_string(&full_path).await.map_err(|e| {
            tracing::error!(file = %path, error = %e, "search replace failed");
            AgentError::internal(format!("failed to read {}: {e}", full_path.display()))
        })?;

        let replaced = content.replacen(old_str, new_str, 1);

        if replaced == content {
            return Ok(vec![Content::text(format!("Text not found in {}", path))]);
        }

        tokio::fs::write(&full_path, &replaced).await.map_err(|e| {
            AgentError::internal(format!("failed to write {}: {e}", full_path.display()))
        })?;

        Ok(vec![Content::text(format!(
            "Successfully replaced text in {}",
            path
        ))])
    }
}
