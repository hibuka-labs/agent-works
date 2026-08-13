use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::path_util::validate_path;
use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};

pub struct ReadFileTool {
    pub workspace: PathBuf,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read the contents of a file"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
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

        tracing::debug!(path = %path, "read file start");
        let content = tokio::fs::read_to_string(&full_path).await.map_err(|e| {
            tracing::error!(path = %path, error = %e, "read file failed");
            AgentError::internal(format!("failed to read {}: {e}", full_path.display()))
        })?;

        tracing::info!(path = %path, size = content.len(), "read file success");
        Ok(vec![Content::text(content)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;

    fn dummy_ctx() -> ToolContext {
        ToolContext::for_test()
    }

    #[test]
    fn test_name_and_schema() {
        let tool = ReadFileTool {
            workspace: PathBuf::from("/tmp"),
        };
        assert_eq!(tool.name(), "read_file");
        assert!(tool.description().contains("Read"));
        assert!(
            tool.schema()["required"]
                .as_array()
                .unwrap()
                .contains(&json!("path"))
        );
    }

    #[tokio::test]
    async fn test_call_reads_content() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world").unwrap();
        let tool = ReadFileTool {
            workspace: dir.path().to_path_buf(),
        };
        let out = tool
            .call(&json!({"path": "a.txt"}), &dummy_ctx())
            .await
            .unwrap();
        assert_eq!(content_text(&out), "hello world");
    }

    #[tokio::test]
    async fn test_call_missing_path() {
        let tool = ReadFileTool {
            workspace: PathBuf::from("/tmp"),
        };
        let err = tool.call(&json!({}), &dummy_ctx()).await.unwrap_err();
        assert!(format!("{err}").contains("missing 'path'"));
    }

    #[tokio::test]
    async fn test_call_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool {
            workspace: dir.path().to_path_buf(),
        };
        let err = tool
            .call(&json!({"path": "nope.txt"}), &dummy_ctx())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("failed to read"));
    }
}
