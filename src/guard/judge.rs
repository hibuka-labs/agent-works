use agent_base::llm_trait::{ChatRequest, LlmProvider};
use agent_base::types::ChatMessage;
use std::sync::Arc;
use std::time::Duration;

/// LLM judge result
#[derive(serde::Deserialize, Debug)]
pub(crate) struct JudgeResult {
    pub done: bool,
    pub reason: String,
}

/// Extract the first balanced JSON object from a raw response.
///
/// Models routinely wrap the requested JSON in a markdown fence or prose
/// (issue #31: "expected value at line 1 column 26"). This scanner finds the
/// first `{` and tracks brace depth, skipping over string literals (with
/// escape handling) so braces inside values don't break the balance.
fn extract_json_object(raw: &str) -> Option<&str> {
    let bytes = raw.as_bytes();
    let start = raw.find('{')?;
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(&raw[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse the judge's response: strict parse first (zero overhead for clean
/// JSON), then fall back to extracting the first balanced JSON object.
fn parse_judge_response(raw: &str) -> Result<JudgeResult, String> {
    if let Ok(result) = serde_json::from_str(raw) {
        return Ok(result);
    }
    let Some(candidate) = extract_json_object(raw) else {
        return Err(format!(
            "Failed to parse judge response: no JSON object found in: {}",
            truncate_for_log(raw, 500)
        ));
    };
    serde_json::from_str(candidate)
        .map_err(|e| format!("Failed to parse judge response: {} in: {}", e, candidate))
}

/// Truncate a string on char boundaries for log/error embedding.
pub(crate) fn truncate_for_log(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    let cut = s
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..cut]
}

/// Call LLM judge to determine if the task is complete.
///
/// Used when the model returns text-only after having called tools —
/// this is suspicious and needs verification.
///
/// Parse failures get exactly one strict retry (the re-ask demands raw JSON
/// only) before the configured fail-open/fail-closed behavior applies —
/// a model formatting quirk must not silently end a run (issue #31).
pub(crate) async fn call_completion_judge(
    client: Option<&Arc<dyn LlmProvider>>,
    user_input: &str,
    model_response: &str,
    all_user_inputs: &[String],
    judge_fail_open: bool,
    judge_timeout_secs: u64,
    recent_user_count: usize,
) -> Result<JudgeResult, String> {
    let Some(client) = client else {
        // No LLM client available — use configured behavior
        if judge_fail_open {
            return Ok(JudgeResult {
                done: true,
                reason: "no LLM client available for judge".to_string(),
            });
        } else {
            return Err("no LLM client available for judge".to_string());
        }
    };

    let system_prompt = "You are a task completion judge. \
        Given the user's conversation history and the agent's response, \
        determine if the agent has sufficiently answered the task. \
        Reply with JSON: {\"done\": true/false, \"reason\": \"brief explanation\"}";

    // Build context from recent user messages
    let user_context = if all_user_inputs.is_empty() {
        user_input.to_string()
    } else {
        let n = recent_user_count;
        let start = all_user_inputs.len().saturating_sub(n);
        let recent = &all_user_inputs[start..];
        if recent.len() <= 1 {
            // Only one message (the current one) — use as-is
            user_input.to_string()
        } else {
            // Multiple messages — show conversation history
            recent
                .iter()
                .enumerate()
                .map(|(i, msg)| format!("{}. {}", start + i + 1, msg))
                .collect::<Vec<_>>()
                .join("\n")
        }
    };

    let user_prompt = format!(
        "【User Messages】\n{}\n\n【Agent Response】\n{}",
        user_context, model_response
    );

    let messages = vec![
        ChatMessage::system(system_prompt.to_string()),
        ChatMessage::user(user_prompt.clone()),
    ];

    let timeout_duration = Duration::from_secs(judge_timeout_secs);

    let failure = match ask_judge_once(client, &messages, timeout_duration).await {
        Ok(result) => return Ok(result),
        // Transport failures (timeout, client error) are not retried —
        // a second call into the same broken transport just burns the budget.
        Err(failure @ JudgeFailure::Transport(_)) => failure,
        Err(JudgeFailure::Unparseable { .. }) => {
            // Parse failure — one strict retry demanding raw JSON only.
            tracing::warn!(
                "completion judge response unparseable, retrying once with raw-JSON prompt"
            );
            let retry_messages = vec![
                ChatMessage::system(format!(
                    "{} Respond with ONLY the raw JSON object — no markdown fences, no prose.",
                    system_prompt
                )),
                ChatMessage::user(user_prompt),
            ];
            match ask_judge_once(client, &retry_messages, timeout_duration).await {
                Ok(result) => return Ok(result),
                // Keep the FINAL attempt's failure — the WARN must carry the
                // raw text the fail-open decision was actually based on.
                Err(failure @ JudgeFailure::Unparseable { .. }) => failure,
                Err(failure @ JudgeFailure::Transport(_)) => failure,
            }
        }
    };

    // Flattened failure — keep the retry's raw text when the first attempt
    // was unparseable but the retry failed differently (transport error).
    let error_message = match &failure {
        JudgeFailure::Unparseable { raw } => format!(
            "Failed to parse judge response after retry; raw: {}",
            truncate_for_log(raw, 500)
        ),
        JudgeFailure::Transport(e) => e.clone(),
    };

    // All failures (timeout, client error, parse error) — use configured behavior
    tracing::warn!(
        error = %error_message,
        fail_open = judge_fail_open,
        "completion judge failed"
    );
    if judge_fail_open {
        // Fail-open: trust the model
        Ok(JudgeResult {
            done: true,
            reason: format!("judge failed ({}), trusting model", error_message),
        })
    } else {
        // Fail-closed: don't trust the model
        Err(error_message)
    }
}

/// Internal failure kinds for one judge call attempt.
///
/// Parse failures carry the raw response (for WARN detail); transport
/// failures (timeout, client error) carry the error string.
enum JudgeFailure {
    Unparseable { raw: String },
    Transport(String),
}

/// One judge LLM call attempt with its own timeout budget.
async fn ask_judge_once(
    client: &Arc<dyn LlmProvider>,
    messages: &[ChatMessage],
    timeout_duration: Duration,
) -> Result<JudgeResult, JudgeFailure> {
    let result = tokio::time::timeout(timeout_duration, async {
        let request = ChatRequest::new(messages.to_vec())
            .with_response_format(agent_base::llm_trait::request::ResponseFormat::JsonObject);
        let response = client
            .chat(request)
            .await
            .map_err(|e| JudgeFailure::Transport(format!("LLM judge call failed: {}", e)))?;

        parse_judge_response(&response.content).map_err(|_| JudgeFailure::Unparseable {
            raw: response.content.clone(),
        })
    })
    .await;

    // Flatten: unwrap the timeout layer, treating timeout as a transport error.
    result.unwrap_or_else(|_elapsed| {
        Err(JudgeFailure::Transport(format!(
            "judge timeout after {}s",
            timeout_duration.as_secs()
        )))
    })
}
