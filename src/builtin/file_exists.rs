use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::path_util::validate_path;
use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};

pub struct FileExistsTool {
    pub workspace: PathBuf,
}

#[async_trait]
impl Tool for FileExistsTool {
    fn name(&self) -> &'static str {
        "file_exists"
    }

    fn description(&self) -> &'static str {
        "Check if a file exists"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory to check"
                }
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| AgentError::internal("missing 'path' argument"))?;

        let full_path = validate_path(&self.workspace, path)?;

        let metadata = tokio::fs::metadata(&full_path).await;

        let exists = metadata.is_ok();

        tracing::trace!(path = %path, exists = exists, "file exists check");
        Ok(vec![Content::text(if exists {
            format!("{} exists", path)
        } else {
            format!("{} does not exist", path)
        })])
    }
}
