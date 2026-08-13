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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;

    fn dummy_ctx() -> ToolContext {
        ToolContext::for_test()
    }

    #[test]
    fn test_name_and_schema() {
        let tool = FileExistsTool {
            workspace: PathBuf::from("/tmp"),
        };
        assert_eq!(tool.name(), "file_exists");
        assert!(tool.description().contains("exists"));
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("path"))
        );
    }

    #[tokio::test]
    async fn test_call_exists() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        let tool = FileExistsTool {
            workspace: dir.path().to_path_buf(),
        };
        let out = tool
            .call(&json!({"path": "a.txt"}), &dummy_ctx())
            .await
            .unwrap();
        assert!(content_text(&out).contains("a.txt exists"));
    }

    #[tokio::test]
    async fn test_call_not_exists() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FileExistsTool {
            workspace: dir.path().to_path_buf(),
        };
        let out = tool
            .call(&json!({"path": "missing.txt"}), &dummy_ctx())
            .await
            .unwrap();
        assert!(content_text(&out).contains("missing.txt does not exist"));
    }

    #[tokio::test]
    async fn test_call_missing_path() {
        let tool = FileExistsTool {
            workspace: PathBuf::from("/tmp"),
        };
        let err = tool.call(&json!({}), &dummy_ctx()).await.unwrap_err();
        assert!(format!("{err}").contains("missing 'path'"));
    }
}
