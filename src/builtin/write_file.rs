use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::path_util::validate_path;
use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};

pub struct WriteFileTool {
    pub workspace: PathBuf,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Write content to a file"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| AgentError::internal("missing 'path' argument"))?;

        let content = args["content"]
            .as_str()
            .ok_or_else(|| AgentError::internal("missing 'content' argument"))?;

        let full_path = validate_path(&self.workspace, path)?;

        tracing::debug!(path = %path, "write file start");

        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                tracing::error!(path = %path, error = %e, "write file failed");
                AgentError::internal(format!(
                    "failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        tokio::fs::write(&full_path, content).await.map_err(|e| {
            tracing::error!(path = %path, error = %e, "write file failed");
            AgentError::internal(format!("failed to write {}: {e}", full_path.display()))
        })?;

        tracing::info!(path = %path, "write file success");
        Ok(vec![Content::text(format!(
            "Successfully wrote to {}",
            path
        ))])
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
        let tool = WriteFileTool {
            workspace: PathBuf::from("/tmp"),
        };
        assert_eq!(tool.name(), "write_file");
        assert!(
            tool.schema()["required"]
                .as_array()
                .unwrap()
                .contains(&json!("content"))
        );
    }

    #[tokio::test]
    async fn test_call_writes_content() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFileTool {
            workspace: dir.path().to_path_buf(),
        };
        let out = tool
            .call(&json!({"path": "a.txt", "content": "hello"}), &dummy_ctx())
            .await
            .unwrap();
        assert!(content_text(&out).contains("Successfully wrote"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "hello"
        );
    }

    #[tokio::test]
    async fn test_call_writes_to_existing_subdir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub/deep")).unwrap();
        let tool = WriteFileTool {
            workspace: dir.path().to_path_buf(),
        };
        let out = tool
            .call(
                &json!({"path": "sub/deep/file.txt", "content": "x"}),
                &dummy_ctx(),
            )
            .await
            .unwrap();
        assert!(content_text(&out).contains("Successfully wrote"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("sub/deep/file.txt")).unwrap(),
            "x"
        );
    }

    #[tokio::test]
    async fn test_call_missing_content() {
        let tool = WriteFileTool {
            workspace: PathBuf::from("/tmp"),
        };
        let err = tool
            .call(&json!({"path": "a.txt"}), &dummy_ctx())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("missing 'content'"));
    }
}
