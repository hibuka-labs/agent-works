use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

use super::types::{McpToolInfo, McpTransport};

pub struct McpClient {
    transport: TransportInner,
    request_id: AtomicU64,
    /// Streamable HTTP session ID: captured from the `Mcp-Session-Id` response
    /// header of `initialize` and replayed on every subsequent request. Required
    /// by the spec (without it the server answers -32000 “no valid session ID
    /// provided”, as implementations like Taobao Desktop do). HTTP-only.
    session: Mutex<Option<String>>,
    /// Protocol handshake (`initialize` + `notifications/initialized`) happens
    /// only once for the whole process. When the server invalidates the session,
    /// this flag is cleared along with [`Self::session`] and everything starts over.
    handshaken: Mutex<bool>,
}

impl McpClient {
    pub async fn new(transport: McpTransport) -> AgentResult<Self> {
        match transport {
            McpTransport::Http { url, headers } => Ok(Self {
                transport: TransportInner::Http {
                    url,
                    headers,
                    client: reqwest::Client::new(),
                },
                request_id: AtomicU64::new(1),
                session: Mutex::new(None),
                handshaken: Mutex::new(false),
            }),
            McpTransport::Stdio { command, args, env } => {
                let (process, stderr) = spawn_stdio_process(&command, &args, &env)
                    .map_err(|e| AgentError::internal(format!("spawn MCP process: {e}")))?;
                tokio::spawn(drain_stderr(command.clone(), stderr));

                Ok(Self {
                    transport: TransportInner::Stdio {
                        process: Mutex::new(process),
                    },
                    request_id: AtomicU64::new(1),
                    session: Mutex::new(None),
                    handshaken: Mutex::new(false),
                })
            }
        }
    }

    /// MCP protocol handshake: `initialize` → record `Mcp-Session-Id` →
    /// `notifications/initialized`. Done only once per process (flag); on failure
    /// the flag is not set, so the next call retries.
    ///
    /// The spec-mandated first step — before any business request. Consequences
    /// of skipping it: lenient servers silently tolerate it, spec-compliant
    /// servers (Taobao Desktop MCP etc.) reject `tools/list` outright
    /// (-32000 "no valid session ID provided").
    async fn ensure_init(&self) -> AgentResult<()> {
        let mut handshaken = self.handshaken.lock().await;
        if *handshaken {
            return Ok(());
        }
        self.send_message(
            &json!({
                "jsonrpc": "2.0",
                "id": self.request_id.fetch_add(1, Ordering::SeqCst),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "phi-agent",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }
            }),
            true,
        )
        .await?;
        self.send_message(
            &json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            false,
        )
        .await?;
        *handshaken = true;
        Ok(())
    }

    /// Session invalidated (server restarted / expired): clear the session ID and
    /// the handshake flag together, and redo the whole flow on the next call.
    async fn forget_session(&self) {
        *self.session.lock().await = None;
        *self.handshaken.lock().await = false;
    }

    async fn send_request(&self, method: &str, params: Value) -> AgentResult<Value> {
        self.ensure_init().await?;
        match self.send_rpc(method, &params).await {
            // The server invalidated the session (bridge restart / expiry):
            // redo the full handshake once and retry.
            // **Only one retry** — tools/call is not idempotent, no endless retries.
            Err(e)
                if is_session_rejected(&e)
                    && matches!(self.transport, TransportInner::Http { .. }) =>
            {
                self.forget_session().await;
                self.ensure_init().await?;
                self.send_rpc(method, &params).await
            }
            other => other,
        }
    }

    /// One business request (fresh id + response).
    async fn send_rpc(&self, method: &str, params: &Value) -> AgentResult<Value> {
        self.send_message(
            &json!({
                "jsonrpc": "2.0",
                "id": self.request_id.fetch_add(1, Ordering::SeqCst),
                "method": method,
                "params": params,
            }),
            true,
        )
        .await
    }

    /// One round trip of a JSON-RPC envelope. `expect_response` = false means a
    /// notification (one-way, no response read — servers reply 202/empty body to
    /// notifications).
    async fn send_message(&self, body: &Value, expect_response: bool) -> AgentResult<Value> {
        match &self.transport {
            TransportInner::Http {
                url,
                headers,
                client,
            } => {
                self.send_http(url, headers, client, body, expect_response)
                    .await
            }
            TransportInner::Stdio { process } => {
                self.send_stdio(process, body, expect_response).await
            }
        }
    }

    async fn send_http(
        &self,
        url: &str,
        headers: &BTreeMap<String, String>,
        client: &reqwest::Client,
        body: &Value,
        expect_response: bool,
    ) -> AgentResult<Value> {
        let mut request = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }
        // Session replay: attach the `Mcp-Session-Id` obtained from the initialize
        // response header to every subsequent request.
        if let Some(sid) = self.session.lock().await.as_deref() {
            request = request.header("Mcp-Session-Id", sid);
        }

        let response = request
            .json(body)
            .send()
            .await
            .map_err(|e| transport_error(url, e))?;

        // The server may send the session header on any response; it must on
        // initialize (spec).
        if let Some(sid) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
        {
            *self.session.lock().await = Some(sid.to_string());
        }

        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let text = response
            .text()
            .await
            .map_err(|e| AgentError::json(format!("MCP response read: {e}")))?;

        if !expect_response {
            return Ok(Value::Null);
        }

        let res = parse_jsonrpc_body(&content_type, &text).map_err(|e| {
            AgentError::json(format!(
                "MCP response parse: {e} (HTTP {status}, content-type {content_type})"
            ))
        })?;
        if let Some(error) = res.get("error") {
            return Err(AgentError::internal(format!("MCP error: {error}")));
        }
        if !status.is_success() {
            return Err(AgentError::internal(format!(
                "MCP HTTP {status}: {}",
                text.chars().take(200).collect::<String>()
            )));
        }
        Ok(res.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn send_stdio(
        &self,
        process: &Mutex<StdioProcess>,
        body: &Value,
        expect_response: bool,
    ) -> AgentResult<Value> {
        let mut request_line =
            serde_json::to_string(body).map_err(|e| AgentError::json(format!("serialize: {e}")))?;
        request_line.push('\n');

        let response_line = {
            let mut proc = process.lock().await;
            proc.stdin
                .write_all(request_line.as_bytes())
                .await
                .map_err(|e| AgentError::internal(format!("stdio write: {e}")))?;
            if !expect_response {
                // Notification: do not read a response (the server does not answer).
                return Ok(Value::Null);
            }
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
        let canonical = format!("mcp.{}.{}", server_name, info.name);
        Self::with_identity_key(&canonical, info, mcp_client)
    }

    /// Constructor with an explicit identity key (MCP servers owned by a plugin
    /// use `plugin.{plugin}.{tool}`; keys are assigned at the product layer in
    /// [`crate::plugin::loader::assign_mcp_tool_keys`]).
    pub fn with_identity_key(
        identity_key: &str,
        info: McpToolInfo,
        mcp_client: Arc<McpClient>,
    ) -> Self {
        let original_name = info.name;
        let static_name: &'static str = Box::leak(identity_key.to_string().into_boxed_str());
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

mod transport;
use transport::{
    StdioProcess, TransportInner, drain_stderr, is_session_rejected, parse_jsonrpc_body,
    spawn_stdio_process, transport_error,
};

#[cfg(test)]
mod tests;
