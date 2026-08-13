use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::path_util::validate_path;
use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};

pub struct ListDirectoryTool {
    pub workspace: PathBuf,
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn name(&self) -> &'static str {
        "list_directory"
    }

    fn description(&self) -> &'static str {
        "List the contents of a directory"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                }
            },
            "required": []
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path = args["path"].as_str().unwrap_or(".");
        let full_path = validate_path(&self.workspace, path)?;

        tracing::debug!(path = %path, "list directory start");
        let mut entries = tokio::fs::read_dir(&full_path).await.map_err(|e| {
            tracing::error!(path = %path, error = %e, "list directory failed");
            AgentError::internal(format!("failed to read dir {}: {e}", full_path.display()))
        })?;

        let mut names: Vec<String> = Vec::new();
        loop {
            let entry = entries.next_entry().await.map_err(|e| {
                tracing::error!(path = %path, error = %e, "list directory failed");
                AgentError::internal(format!("failed to read entry: {e}"))
            })?;
            let Some(entry) = entry else {
                break;
            };
            names.push(entry.file_name().to_string_lossy().into_owned());
        }

        let summary = if names.is_empty() {
            format!("Directory {} is empty", path)
        } else {
            names.join("\n")
        };

        Ok(vec![Content::text(summary)])
    }
}
