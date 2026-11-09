use std::collections::BTreeMap;

use agent_base::AgentError;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

pub(super) struct StdioProcess {
    #[allow(dead_code)] // held for RAII: child process is killed on drop
    pub(super) child: Child,
    pub(super) stdin: ChildStdin,
    pub(super) stdout: BufReader<ChildStdout>,
}

/// Spawn a stdio MCP server under the one contract that keeps it away from
/// the user's terminal:
///
/// - stdin/stdout are the transport (piped);
/// - stderr is **piped** — never `inherit()`. An inherited stderr paints
///   straight onto the TUI's alternate screen (npm deprecation warnings were
///   observed landing mid-frame) and ratatui's diffing never repaints those
///   cells;
/// - the child `setsid`s out of the controlling terminal, so it can neither
///   `open("/dev/tty")` nor `tcsetattr` the user's tty (a clobbered termios
///   silently disables keyboard input — the "stuck TUI" incident).
///
/// Returns the stderr handle so the caller can drain it into the log — the
/// original intent of `inherit()` (server errors stay visible) without the
/// screen damage.
pub(super) fn spawn_stdio_process(
    command: &str,
    args: &[String],
    env: &BTreeMap<String, String>,
) -> std::io::Result<(StdioProcess, ChildStderr)> {
    let mut cmd = Command::new(command);
    cmd.args(args)
        .envs(env) // inherit process env + per-entry overrides (server token ports)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt as _;
        // From here on the child is its own session with no controlling terminal.
        // pre_exec runs async-signal-safe syscalls only; if setsid fails we refuse
        // to spawn (fail-closed — never quietly run with a tty attached).
        cmd.as_std_mut().pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = cmd.spawn()?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("no stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("no stderr"))?;
    Ok((
        StdioProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        },
        stderr,
    ))
}

/// Drain a MCP server's stderr into the log, one line per warning.
pub(super) async fn drain_stderr(command: String, stderr: ChildStderr) {
    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        tracing::warn!(server = %command, %line, "MCP server stderr");
    }
}

#[allow(clippy::large_enum_variant)] // Http carries reqwest::Client; Stdio carries a child process
pub(super) enum TransportInner {
    Http {
        url: String,
        /// Extra request headers (token etc.; `${ENV:}` has already been resolved
        /// to literals by the product layer).
        headers: BTreeMap<String, String>,
        client: reqwest::Client,
    },
    Stdio {
        process: Mutex<StdioProcess>,
    },
}

/// Response body → JSON-RPC envelope. Accepts both shapes: plain
/// `application/json`, and Streamable HTTP `text/event-stream` (`data:` frames —
/// servers like Taobao Desktop wrap successful responses in SSE).
pub(super) fn parse_jsonrpc_body(content_type: &str, body: &str) -> Result<Value, String> {
    let looks_sse = content_type.contains("text/event-stream")
        || body.trim_start().starts_with("event:")
        || body.lines().any(|l| l.starts_with("data:"));
    if !looks_sse {
        return serde_json::from_str(body).map_err(|e| e.to_string());
    }
    // SSE: multi-frame messages separated by blank lines; each frame's `data:`
    // lines are concatenated back into one JSON.
    let mut payload = String::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            if !payload.is_empty() {
                payload.push('\n');
            }
            payload.push_str(data.trim_start());
            continue;
        }
        if line.is_empty() && !payload.is_empty() {
            let parsed = sse_payload_value(&payload);
            payload.clear();
            if let Some(value) = parsed {
                return Ok(value);
            }
        }
    }
    if let Some(value) = sse_payload_value(&payload) {
        return Ok(value);
    }
    Err("no JSON-RPC payload in SSE stream".to_string())
}

/// Reassembled `data:` payload → JSON-RPC envelope (must be a JSON object;
/// empty/non-object doesn't count).
pub(super) fn sse_payload_value(payload: &str) -> Option<Value> {
    serde_json::from_str::<Value>(payload)
        .ok()
        .filter(Value::is_object)
}

/// A decent error for transport failures: say plainly "the server is gone /
/// timed out", don't just dump the raw reqwest text
/// (UX checklist ⑥ "server unreachable" error surface — the model relays this
/// sentence to the user).
pub(super) fn transport_error(url: &str, e: reqwest::Error) -> AgentError {
    let timed_out = e.is_timeout();
    let detail = e.without_url();
    if timed_out {
        AgentError::internal(format!(
            "MCP server at {url} timed out — it is up but did not answer ({detail})"
        ))
    } else {
        AgentError::internal(format!(
            "MCP server at {url} is unreachable — the app that provides it may have quit or its bridge died ({detail})"
        ))
    }
}

/// The server invalidated the session (-32000 invalid session ID / the spec
/// answers HTTP 404 for expired sessions). On a match, redo the full handshake
/// once (see the recovery path in [`McpClient::send_request`]).
pub(super) fn is_session_rejected(err: &AgentError) -> bool {
    let msg = err.to_string();
    // The JSON-RPC error object is embedded in the message verbatim:
    // `MCP error: {"code":-32000,...}`.
    msg.contains("\"code\":-32000") || msg.contains("MCP HTTP 404")
}
