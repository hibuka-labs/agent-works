use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use agent_base::{AgentError, AgentResult, Content, ToolContext};
use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use super::TRUNCATION_MARKER;
use super::law::script_file_index;
use super::subst::substitute_params;
use super::types::ScriptToolEntry;

/// Output truncation convention (copied from phi-kernel-tools `local_shell::truncate_output`):
/// first 1/3 + `...[output truncated]` + last 2/3. Plus one clamp the kernel
/// copy lacks: a budget too small to carry the marker returns a head slice,
/// because the contract is `result <= max_chars` — an over-budget result makes
/// the engine reject the whole call (`Tool '…' output exceeds the …-char
/// limit`) and the model loses the result entirely.
pub(crate) fn truncate_output(s: &str, max_chars: Option<usize>) -> String {
    let Some(max_chars) = max_chars else {
        return s.to_string();
    };
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }
    if max_chars <= TRUNCATION_MARKER.chars().count() + 2 {
        return s.chars().take(max_chars).collect();
    }
    let available = max_chars.saturating_sub(TRUNCATION_MARKER.chars().count() + 2);
    let head_chars = available / 3;
    let tail_chars = available - head_chars;
    let head: String = s.chars().take(head_chars).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}\n{TRUNCATION_MARKER}\n{tail}")
}

/// Kill the whole process tree (on timeout/cancellation).
///
/// - POSIX: the script starts as a process group leader (`process_group(0)`), `killpg` SIGKILL;
/// - Windows: `taskkill /T /F /PID` (Job Object comes later; this semantically equivalent fallback
///   covers us for now).
async fn kill_tree(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    #[cfg(unix)]
    {
        // process group = leader pid (process_group(0) at spawn).
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
}

/// Execute one script entry (shared by `ScriptProvider` / `ScriptTool` / the future plugin `script` form).
///
/// `tool_key` is the identity key (carried in errors); `args` MUST already have been validated by
/// the host against the envelope.
pub async fn execute_entry(
    tool_key: &str,
    entry: &ScriptToolEntry,
    args: &Value,
    ctx: &ToolContext,
) -> AgentResult<Vec<Content>> {
    let argv = substitute_params(tool_key, &entry.run, args)?;

    // Script gone → "script is gone" error, no silent file swap.
    let script_index = script_file_index(&entry.run).map_err(|e| AgentError::ToolExecution {
        name: tool_key.to_string(),
        source: Box::new(AgentError::Internal(e.to_string())),
    })?;
    let script_path = Path::new(&argv[script_index]);
    if !script_path.is_file() {
        return Err(AgentError::ToolExecution {
            name: tool_key.to_string(),
            source: Box::new(AgentError::Internal(format!(
                "script is gone: {}",
                script_path.display()
            ))),
        });
    }

    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .envs(&entry.env) // inherit process env + entry overrides
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(dir) = &entry.cwd {
        cmd.current_dir(dir);
    }
    #[cfg(unix)]
    cmd.process_group(0); // own process group, leader = this script → killpg the whole tree
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

    let mut child = cmd.spawn().map_err(|e| AgentError::ToolExecution {
        name: tool_key.to_string(),
        source: Box::new(AgentError::Internal(format!(
            "failed to spawn {}: {e}",
            argv[0]
        ))),
    })?;
    let pid = child.id();

    // Concurrent pipe reads (wait and the read ends don't exclude each other; leaked background
    // children holding the pipes get a fallback deadline).
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let stdout_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buf).await;
        }
        buf
    });
    let stderr_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buf).await;
        }
        buf
    });

    // Timeout / cancellation / normal exit. Bind before matching so the wait future is dropped
    // before `child` is finalized (a match scrutinee temporary lives until the match ends).
    enum WaitEnd {
        Exited(std::io::Result<std::process::ExitStatus>),
        Cancelled,
    }
    let timeout = Duration::from_millis(entry.timeout_ms());
    let wait_future = async {
        tokio::select! {
            _ = ctx.cancel_token.cancelled() => WaitEnd::Cancelled,
            status = child.wait() => WaitEnd::Exited(status),
        }
    };
    let ended = tokio::time::timeout(timeout, wait_future).await;

    let status = match ended {
        Err(_elapsed) => {
            // Timeout: kill the whole tree (the tool dies, not the host).
            kill_tree(pid).await;
            let _ = child.wait().await;
            return Err(AgentError::ToolTimeout);
        }
        Ok(WaitEnd::Cancelled) => {
            kill_tree(pid).await;
            let _ = child.wait().await;
            return Err(AgentError::Cancelled);
        }
        Ok(WaitEnd::Exited(Err(e))) => {
            return Err(AgentError::ToolExecution {
                name: tool_key.to_string(),
                source: Box::new(AgentError::Internal(format!("wait failed: {e}"))),
            });
        }
        Ok(WaitEnd::Exited(Ok(status))) => status,
    };

    let read_deadline = Duration::from_secs(5);
    let stdout = tokio::time::timeout(read_deadline, stdout_task)
        .await
        .ok()
        .and_then(|t| t.ok())
        .unwrap_or_default();
    let stderr = tokio::time::timeout(read_deadline, stderr_task)
        .await
        .ok()
        .and_then(|t| t.ok())
        .unwrap_or_default();
    let stdout = String::from_utf8_lossy(&stdout).into_owned();
    let stderr = String::from_utf8_lossy(&stderr).into_owned();

    if !status.success() {
        // exit ≠ 0 → tool error (with a stderr summary).
        let summary = if stderr.trim().is_empty() {
            stdout.clone()
        } else {
            stderr.clone()
        };
        let summary = truncate_output(summary.trim(), ctx.max_output_chars);
        return Err(AgentError::ToolExecution {
            name: tool_key.to_string(),
            source: Box::new(AgentError::Internal(format!(
                "script exited with {status}: {summary}"
            ))),
        });
    }

    let body = if stdout.trim().is_empty() {
        stderr
    } else {
        stdout
    };
    Ok(vec![Content::text(truncate_output(
        body.trim_end_matches('\n'),
        ctx.max_output_chars,
    ))])
}
