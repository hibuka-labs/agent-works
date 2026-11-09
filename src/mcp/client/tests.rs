use super::*;
use agent_base::tool::content_text;
use tokio::io::BufReader;
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
    McpClient::new(McpTransport::http(format!("{}/mcp", server.uri())))
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
    let err = McpClient::new(McpTransport::stdio(
        "definitely-not-a-real-command-xyz",
        Vec::<String>::new(),
    ))
    .await
    .err()
    .unwrap();
    assert!(format!("{err}").contains("spawn MCP process"));
}

/// Regression (BUG 7): a stdio server's stderr must come back to us, not be
/// inherited into the user's terminal. Observed in practice: npm deprecation
/// warnings used to land as whole lines in the alt-screen.
#[tokio::test]
async fn test_stdio_child_stderr_is_piped_not_inherited() {
    let (mut process, stderr) = spawn_stdio_process(
        "sh",
        &["-c".to_string(), "echo NOISE-XYZZY >&2".to_string()],
        &BTreeMap::new(),
    )
    .unwrap();
    let mut lines = BufReader::new(stderr).lines();
    let line = lines.next_line().await.unwrap().unwrap();
    assert_eq!(line, "NOISE-XYZZY");
    let _ = process.child.wait().await;
}

/// Regression (BUG 7): the child must setsid away from the controlling
/// terminal, otherwise it can open /dev/tty and tcsetattr the user's tty
/// (setting termios back to cooked → keyboard silently dies). Criterion:
/// after setsid the child is the session leader, sid == pid.
#[cfg(unix)]
#[tokio::test]
async fn test_stdio_child_leaves_the_controlling_terminal() {
    let (mut process, _stderr) =
        spawn_stdio_process("sleep", &["5".to_string()], &BTreeMap::new()).unwrap();
    let pid = process.child.id().unwrap();
    // After setsid the child is its own session: from our side getsid(child)
    // returns either EPERM (different session, -1) or the child's own
    // sid == pid. Without setsid we would get this process's session id,
    // satisfying neither — the criterion would fail.
    let sid = unsafe { libc::getsid(pid as i32) };
    assert!(
        sid == -1 || sid == pid as i32,
        "child is still in the parent's session (sid={sid}, pid={pid})"
    );
    let _ = process.child.start_kill();
    let _ = process.child.wait().await;
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
    // The error surface must be human: say clearly "the server is gone",
    // don't just dump the raw reqwest text (UX⑥).
    let client = McpClient::new(McpTransport::http("http://127.0.0.1:1/mcp"))
        .await
        .unwrap();
    let err = client
        .send_request("tools/list", json!({}))
        .await
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("unreachable"), "{msg}");
    assert!(msg.contains("http://127.0.0.1:1/mcp"), "{msg}");
}

#[tokio::test]
async fn test_send_request_http_sends_headers() {
    // headers port: Authorization must appear on the request (wiremock 404s
    // if the header matcher doesn't hit).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .and(wiremock::matchers::header("authorization", "Bearer tok-9"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1, "result": {"ok": true}
        })))
        .mount(&server)
        .await;

    let client = McpClient::new(
        McpTransport::http(format!("{}/mcp", server.uri()))
            .with_header("Authorization", "Bearer tok-9"),
    )
    .await
    .unwrap();
    let res = client.send_request("tools/list", json!({})).await.unwrap();
    assert_eq!(res["ok"], true);
}

/// Replays Taobao Desktop's failure mode: `initialize` issues a session ID,
/// then subsequent requests that do **not** carry `Mcp-Session-Id` get
/// -32000 “no valid session ID provided”. The client must handshake and
/// replay it.
struct SessionEchoServer {
    initialized: std::sync::atomic::AtomicU64,
}

impl wiremock::Respond for SessionEchoServer {
    fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
        let body = std::str::from_utf8(&request.body).unwrap_or("");
        if body.contains("\"initialize\"") {
            self.initialized
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return ResponseTemplate::new(200)
                .insert_header("Mcp-Session-Id", "sess-77")
                .set_body_json(json!({
                    "jsonrpc": "2.0", "id": 1,
                    "result": {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}}
                }));
        }
        if body.contains("notifications/initialized") {
            return ResponseTemplate::new(202);
        }
        let has_session = request
            .headers
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            == Some("sess-77");
        if has_session {
            ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": 2,
                    "result": {"tools": [{"name": "search_products", "description": "d", "inputSchema": {}}]}
                }))
        } else {
            ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 2,
                "error": {"code": -32000, "message": "错误请求：未提供有效的会话ID"}
            }))
        }
    }
}

#[tokio::test]
async fn test_handshake_captures_session_id_and_replays_it() {
    let server = MockServer::start().await;
    let responder = SessionEchoServer {
        initialized: std::sync::atomic::AtomicU64::new(0),
    };
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .respond_with(responder)
        .mount(&server)
        .await;

    let client = http_client(&server).await;
    // Handshake (initialize + notification) happens automatically;
    // tools/list carries the session header → tools are returned.
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "search_products");
}

#[tokio::test]
async fn test_sse_response_is_parsed() {
    // A Streamable HTTP success response may be `text/event-stream`
    // (Taobao Desktop does exactly this).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[{\"name\":\"buy_now\",\"description\":\"d\",\"inputSchema\":{}}]}}\n\n",
                "text/event-stream",
            ))
            .mount(&server)
            .await;

    let client = http_client(&server).await;
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "buy_now");
}

#[tokio::test]
async fn test_handshake_happens_once_per_client() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let server = MockServer::start().await;
    let counter = Arc::new(AtomicU64::new(0));
    struct CountInit(Arc<AtomicU64>);
    impl wiremock::Respond for CountInit {
        fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
            let body = std::str::from_utf8(&request.body).unwrap_or("");
            if body.contains("\"initialize\"") {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
            ResponseTemplate::new(200).set_body_json(json!({
                "jsonrpc": "2.0", "id": 1, "result": {"tools": []}
            }))
        }
    }
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .respond_with(CountInit(Arc::clone(&counter)))
        .mount(&server)
        .await;

    let client = http_client(&server).await;
    let _ = client.list_tools().await.unwrap();
    let _ = client.list_tools().await.unwrap();
    let _ = client.call_tool("t", &json!({})).await.unwrap();
    assert_eq!(counter.load(Ordering::SeqCst), 1, "initialize exactly once");
}

/// Bridge-restart scenario: after the first business request `sess-1` is
/// invalidated (-32000); only a re-handshake yielding `sess-2` is accepted.
/// The client must automatically re-handshake and successfully replay the
/// original call, and **only once**.
#[tokio::test]
async fn test_invalid_session_rehandshakes_and_retries_once() {
    use std::sync::atomic::{AtomicU64, Ordering};
    struct RestartingBridge {
        inits: Arc<AtomicU64>,
        business: Arc<AtomicU64>,
    }
    impl wiremock::Respond for RestartingBridge {
        fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
            let body = std::str::from_utf8(&request.body).unwrap_or("");
            if body.contains("\"initialize\"") {
                let n = self.inits.fetch_add(1, Ordering::SeqCst) + 1;
                return ResponseTemplate::new(200)
                    .insert_header("Mcp-Session-Id", format!("sess-{n}"))
                    .set_body_json(json!({
                        "jsonrpc": "2.0", "id": 1,
                        "result": {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}}
                    }));
            }
            if body.contains("notifications/initialized") {
                return ResponseTemplate::new(202);
            }
            let sid = request
                .headers
                .get("mcp-session-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            let nth = self.business.fetch_add(1, Ordering::SeqCst);
            // The old session is invalidated after the first business request;
            // the new session (from the re-handshake) is accepted forever.
            if sid == "sess-1" && nth >= 1 {
                return ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": 2,
                    "error": {"code": -32000, "message": "错误请求：未提供有效的会话ID"}
                }));
            }
            ResponseTemplate::new(200).set_body_json(json!({
                    "jsonrpc": "2.0", "id": 2,
                    "result": {"tools": [{"name": "search_products", "description": "d", "inputSchema": {}}]}
                }))
        }
    }
    let server = MockServer::start().await;
    let inits = Arc::new(AtomicU64::new(0));
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .respond_with(RestartingBridge {
            inits: Arc::clone(&inits),
            business: Arc::new(AtomicU64::new(0)),
        })
        .mount(&server)
        .await;

    let client = http_client(&server).await;
    // First call: handshake + normal response.
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    // Second call: sess-1 is now invalid → automatic re-handshake (sess-2)
    // → replay succeeds.
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    // Afterwards sess-2 is used directly, no further handshake
    // (the recovery happens exactly once).
    let _ = client
        .call_tool("search_products", &json!({}))
        .await
        .unwrap();
    assert_eq!(inits.load(Ordering::SeqCst), 2, "exactly one re-handshake");
}

#[tokio::test]
async fn test_send_request_stdio_returns_result() {
    // Multi-turn shell: only reply to requests that carry an id
    // (notifications get no reply — the handshake includes one notification).
    let client = McpClient::new(McpTransport::stdio(
            "sh",
            [
                "-c",
                r#"while read line; do case "$line" in *'"id"'*) echo '{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}';; esac; done"#,
            ],
        ))
        .await
        .unwrap();

    let res = client.send_request("tools/list", json!({})).await.unwrap();
    assert!(res["tools"].is_array());
}

#[cfg(unix)]
// file-write probe goes through sh; Windows semantics are covered by a separate spawn-env test (see docs)
#[tokio::test]
async fn test_stdio_env_override_reaches_child() {
    // env port: takes effect at spawn (no protocol round trip needed) — the
    // child writes $MCP_TEST_TOKEN to a file.
    let dir = tempfile::tempdir().unwrap();
    let outfile = dir.path().join("env.txt");
    let script = format!("printf '%s' \"$MCP_TEST_TOKEN\" > {}", outfile.display());
    let _client = McpClient::new(
        McpTransport::stdio("sh", ["-c", &script]).with_env("MCP_TEST_TOKEN", "tok-from-env"),
    )
    .await
    .unwrap();

    let mut content = String::new();
    for _ in 0..100 {
        if let Ok(text) = std::fs::read_to_string(&outfile) {
            content = text;
            if !content.is_empty() {
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(content, "tok-from-env");
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
