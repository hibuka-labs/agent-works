use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use super::types::{McpToolInfo, McpTransport};

struct StdioProcess {
    #[allow(dead_code)] // held for RAII: child process is killed on drop
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[allow(clippy::large_enum_variant)] // Http carries reqwest::Client; Stdio carries a child process
enum TransportInner {
    Http {
        url: String,
        client: reqwest::Client,
    },
    Stdio {
        process: Mutex<StdioProcess>,
    },
}

pub struct McpClient {
    transport: TransportInner,
    request_id: AtomicU64,
}

impl McpClient {
    pub async fn new(transport: McpTransport) -> AgentResult<Self> {
        match transport {
            McpTransport::Http { url } => Ok(Self {
                transport: TransportInner::Http {
                    url,
                    client: reqwest::Client::new(),
                },
                request_id: AtomicU64::new(1),
            }),
            McpTransport::Stdio { command, args } => {
                let mut child = Command::new(&command)
                    .args(&args)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::inherit())
                    .spawn()
                    .map_err(|e| AgentError::internal(format!("spawn MCP process: {e}")))?;

                let stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| AgentError::internal("no stdin"))?;
                let stdout = child
                    .stdout
                    .take()
                    .ok_or_else(|| AgentError::internal("no stdout"))?;

                Ok(Self {
                    transport: TransportInner::Stdio {
                        process: Mutex::new(StdioProcess {
                            child,
                            stdin,
                            stdout: BufReader::new(stdout),
                        }),
                    },
                    request_id: AtomicU64::new(1),
                })
            }
        }
    }

    async fn send_request(&self, method: &str, params: Value) -> AgentResult<Value> {
        match &self.transport {
            TransportInner::Http { url, client } => {
                self.send_request_http(url, client, method, params).await
            }
            TransportInner::Stdio { process } => {
                self.send_request_stdio(process, method, params).await
            }
        }
    }

    async fn send_request_http(
        &self,
        url: &str,
        client: &reqwest::Client,
        method: &str,
        params: Value,
    ) -> AgentResult<Value> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let response = client
            .post(url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::internal(format!("MCP request failed: {e}")))?;

        let res: Value = response
            .json()
            .await
            .map_err(|e| AgentError::json(format!("MCP response parse: {e}")))?;

        if let Some(error) = res.get("error") {
            return Err(AgentError::internal(format!("MCP error: {error}")));
        }

        Ok(res.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn send_request_stdio(
        &self,
        process: &Mutex<StdioProcess>,
        method: &str,
        params: Value,
    ) -> AgentResult<Value> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut request_line = serde_json::to_string(&request)
            .map_err(|e| AgentError::json(format!("serialize: {e}")))?;
        request_line.push('\n');

        let response_line = {
            let mut proc = process.lock().await;
            proc.stdin
                .write_all(request_line.as_bytes())
                .await
                .map_err(|e| AgentError::internal(format!("stdio write: {e}")))?;

            let mut line = String::new();
            proc.stdout
                .read_line(&mut line)
                .await
                .map_err(|e| AgentError::internal(format!("stdio read: {e}")))?;
            line
        };

        let res: Value = serde_json::from_str(&response_line)
            .map_err(|e| AgentError::json(format!("parse: {e}")))?;

        if let Some(error) = res.get("error") {
            return Err(AgentError::internal(format!("MCP error: {error}")));
        }

        Ok(res.get("result").cloned().unwrap_or(Value::Null))
    }

    pub async fn list_tools(&self) -> AgentResult<Vec<McpToolInfo>> {
        let result = self.send_request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| AgentError::internal("MCP: invalid tools/list response"))?;

        let mut infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let description = tool
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            infos.push(McpToolInfo {
                name,
                description,
                input_schema,
            });
        }
        Ok(infos)
    }

    pub async fn call_tool(&self, tool_name: &str, arguments: &Value) -> AgentResult<Value> {
        self.send_request(
            "tools/call",
            json!({
                "name": tool_name,
                "arguments": arguments,
            }),
        )
        .await
    }
}

/// Wraps an MCP server tool as an agent-base `Tool`.
///
/// Tool names are prefixed with `mcp.{server_name}.` to prevent collisions
/// between tools from different MCP servers.
pub struct McpToolAdapter {
    /// Prefixed name exposed to the LLM: `mcp.{server}.{tool}`
    name: &'static str,
    /// Original tool name used when calling the MCP server
    original_name: String,
    description: &'static str,
    input_schema: Value,
    mcp_client: Arc<McpClient>,
}

impl McpToolAdapter {
    pub fn new(info: McpToolInfo, mcp_client: Arc<McpClient>, server_name: &str) -> Self {
        let original_name = info.name;
        let canonical = format!("mcp.{}.{}", server_name, original_name);
        let static_name: &'static str = Box::leak(canonical.into_boxed_str());
        let static_description: &'static str = Box::leak(info.description.into_boxed_str());
        Self {
            name: static_name,
            original_name,
            description: static_description,
            input_schema: info.input_schema,
            mcp_client,
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn schema(&self) -> Value {
        self.input_schema.clone()
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let result = self.mcp_client.call_tool(&self.original_name, args).await?;
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| result.to_string());

        Ok(vec![Content::text(content)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::tool::content_text;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn tool_info() -> McpToolInfo {
        McpToolInfo {
            name: "search".to_string(),
            description: "search the web".to_string(),
            input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        }
    }

    async fn http_client(server: &MockServer) -> McpClient {
        McpClient::new(McpTransport::Http {
            url: format!("{}/mcp", server.uri()),
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_new_http_transport() {
        let server = MockServer::start().await;
        let client = http_client(&server).await;
        assert!(matches!(client.transport, TransportInner::Http { .. }));
    }

    #[tokio::test]
    async fn test_new_stdio_spawn_error() {
        let err = McpClient::new(McpTransport::Stdio {
            command: "definitely-not-a-real-command-xyz".to_string(),
            args: vec![],
        })
        .await
        .err()
        .unwrap();
        assert!(format!("{err}").contains("spawn MCP process"));
    }

    #[tokio::test]
    async fn test_send_request_http_returns_result() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"tools": [{"name": "run"}]}
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let res = client.send_request("tools/list", json!({})).await.unwrap();
        assert_eq!(res["tools"][0]["name"], "run");
    }

    #[tokio::test]
    async fn test_send_request_http_error_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "error": {"code": -32601, "message": "Method not found"}
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let err = client.send_request("nope", json!({})).await.unwrap_err();
        assert!(format!("{err}").contains("MCP error"));
    }

    #[tokio::test]
    async fn test_send_request_http_network_error() {
        // No mock mounted + pointing at an unbound port should fail at send().
        let client = McpClient::new(McpTransport::Http {
            url: "http://127.0.0.1:1/mcp".to_string(),
        })
        .await
        .unwrap();
        let err = client
            .send_request("tools/list", json!({}))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("MCP request failed"));
    }

    #[tokio::test]
    async fn test_send_request_stdio_returns_result() {
        // A shell that reads one line and echoes a fixed JSON-RPC response.
        let client = McpClient::new(McpTransport::Stdio {
            command: "sh".to_string(),
            args: vec![
                "-c".to_string(),
                r#"read _; echo '{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}'"#.to_string(),
            ],
        })
        .await
        .unwrap();

        let res = client.send_request("tools/list", json!({})).await.unwrap();
        assert!(res["tools"].is_array());
    }

    #[tokio::test]
    async fn test_list_tools_parses_http() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {
                    "tools": [
                        {"name": "search", "description": "search the web",
                         "inputSchema": {"type": "object"}},
                        {"name": "run", "description": ""}
                    ]
                }
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "search");
        assert_eq!(tools[0].description, "search the web");
        // Missing inputSchema falls back to {"type": "object"}
        assert_eq!(tools[1].input_schema, json!({"type": "object"}));
    }

    #[tokio::test]
    async fn test_list_tools_invalid_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1, "result": {"not_tools": []}
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let err = client.list_tools().await.unwrap_err();
        assert!(format!("{err}").contains("invalid tools/list response"));
    }

    #[tokio::test]
    async fn test_call_tool_http() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"content": [{"type": "text", "text": "done"}]}
            })))
            .mount(&server)
            .await;

        let client = http_client(&server).await;
        let result = client
            .call_tool("search", &json!({"q": "rust"}))
            .await
            .unwrap();
        assert_eq!(result["content"][0]["text"], "done");
    }

    #[tokio::test]
    async fn test_adapter_new_and_metadata() {
        let server = MockServer::start().await;
        let client = Arc::new(http_client(&server).await);
        let adapter = McpToolAdapter::new(tool_info(), client, "brave");

        assert_eq!(adapter.name(), "mcp.brave.search");
        assert_eq!(adapter.description(), "search the web");
        assert_eq!(adapter.schema()["properties"]["q"]["type"], "string");
    }

    #[tokio::test]
    async fn test_adapter_call_returns_text_content() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"content": [{"type": "text", "text": "alpha"},
                                       {"type": "text", "text": "beta"}]}
            })))
            .mount(&server)
            .await;

        let client = Arc::new(http_client(&server).await);
        let adapter = McpToolAdapter::new(tool_info(), client, "brave");
        let out = adapter
            .call(&json!({"q": "rust"}), &ToolContext::for_test())
            .await
            .unwrap();
        // Text items are joined by newline.
        assert_eq!(content_text(&out), "alpha\nbeta");
    }

    #[tokio::test]
    async fn test_adapter_call_without_content_falls_back_to_json() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1,
                "result": {"plain": "no content field"}
            })))
            .mount(&server)
            .await;

        let client = Arc::new(http_client(&server).await);
        let adapter = McpToolAdapter::new(tool_info(), client, "brave");
        let out = adapter
            .call(&json!({"q": "rust"}), &ToolContext::for_test())
            .await
            .unwrap();
        assert!(content_text(&out).contains("no content field"));
    }
}
