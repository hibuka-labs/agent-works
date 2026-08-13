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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;

    fn dummy_ctx() -> ToolContext {
        ToolContext::for_test()
    }

    #[test]
    fn test_name_and_schema() {
        let tool = ListDirectoryTool {
            workspace: PathBuf::from("/tmp"),
        };
        assert_eq!(tool.name(), "list_directory");
        assert!(tool.description().contains("directory"));
        assert_eq!(tool.schema()["type"], "object");
    }

    #[tokio::test]
    async fn test_call_lists_entries() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        std::fs::write(dir.path().join("b.txt"), "y").unwrap();
        let tool = ListDirectoryTool {
            workspace: dir.path().to_path_buf(),
        };
        let out = tool
            .call(&json!({"path": "."}), &dummy_ctx())
            .await
            .unwrap();
        let text = content_text(&out);
        assert!(text.contains("a.txt"));
        assert!(text.contains("b.txt"));
    }

    #[tokio::test]
    async fn test_call_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ListDirectoryTool {
            workspace: dir.path().to_path_buf(),
        };
        let out = tool
            .call(&json!({"path": "."}), &dummy_ctx())
            .await
            .unwrap();
        assert!(content_text(&out).contains("is empty"));
    }

    #[tokio::test]
    async fn test_call_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ListDirectoryTool {
            workspace: dir.path().to_path_buf(),
        };
        let err = tool
            .call(&json!({"path": "nope"}), &dummy_ctx())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("failed to read dir"));
    }
}
