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

/// The judge's system prompt: a COMPLETENESS judge, not a fact-checker.
///
/// Incident 20261107_27dc727a: the old prompt asked whether the agent
/// "sufficiently answered" with zero ground truth, so the judge fabricated
/// one — a judge running as the boot model contradicted the agent's correct
/// "mimo-v2.6-pro" with its own identity ("the configured model is
/// MiniMax-M3.1-Flash") and misquoted the answer as "mimo-v2.5-pro". The
/// injected "Your answer is incomplete" correction then dragged a trivial
/// turn through two extra tool calls and three extra LLM turns. Measured on
/// the incident case (36 calls, live models): this prompt scored 36/36
/// (correct answer passes, dodging answer fails, wrong model name fails via
/// the ground-truth line) where the old prompt false-positived 4/6.
///
/// `session_model` is the model serving the session, taken from the same
/// live provider the judge runs on (`LlmProvider::info().model`). It is the
/// ONLY fact this prompt sanctions, scoped to model-name claims; an empty
/// name drops the line.
///
/// Wire contract: the first line must keep the phrase "completion judge" —
/// the phimint test/PTY harnesses classify judge side-calls by it and give
/// them canned answers, so rewording it silently desyncs their LLM scripts
/// (20261107_27dc727a). Pinned by `judge_prompt_is_completeness_only_with_scoped_ground_truth`.
pub(crate) fn judge_system_prompt(session_model: &str) -> String {
    let mut prompt = String::from(
        "You are a task-completion judge. You are given the user's messages and the agent's \
         final response in an ongoing conversation. Your ONLY question: does the response \
         address what the user asked?\n\
         Rules:\n\
         - Judge COMPLETENESS only. Do NOT fact-check the response: you have no source of \
         truth beyond the ground truth line and the messages in this prompt.\n\
         - Never state, assume, or introduce facts that are not written below (including \
         anything about which model is configured or how the session is set up).\n\
         - Do not paraphrase or \"correct\" values: quote names only exactly as they appear \
         below.\n\
         - Do not penalize a response as unverifiable, unreliable, or fabricated.\n\
         - Earlier user messages may already have been answered earlier in the conversation: \
         judge the response against the question(s) it addresses, and do not demand details \
         the user did not ask for. A concise answer to the question asked is complete.\n",
    );
    if !session_model.is_empty() {
        prompt.push_str(&format!(
            "- Ground truth: the model serving this session is `{session_model}`. Use this \
             ONLY to check a claim about which model is serving the session; a response with \
             no such claim is consistent.\n"
        ));
    }
    prompt.push_str("Reply with JSON: {\"done\": true/false, \"reason\": \"brief explanation\"}");
    prompt
}

/// Call LLM judge to determine if the task is complete.
///
/// Used when the model returns text-only after having called tools —
/// this is suspicious and needs verification.
///
/// Parse failures get exactly one strict retry (the re-ask demands raw JSON
/// only) before the configured fail-open/fail-closed behavior applies —
/// a model formatting quirk must not silently end a run (issue #31).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn call_completion_judge(
    client: Option<&Arc<dyn LlmProvider>>,
    user_input: &str,
    model_response: &str,
    all_user_inputs: &[String],
    judge_fail_open: bool,
    judge_timeout_secs: u64,
    recent_user_count: usize,
    notice: Option<&agent_base::NoticeHandle>,
) -> Result<JudgeResult, String> {
    let Some(client) = client else {
        // No LLM client available — use configured behavior
        if judge_fail_open {
            // Same honesty contract as the call-failure fail-open below: the
            // degradation must be visible in the UI, never silent.
            if let Some(handle) = notice {
                handle.send(
                    agent_base::types::NoticeKind::Warning,
                    "guard",
                    "guard judge unavailable — treating as complete",
                );
            }
            return Ok(JudgeResult {
                done: true,
                reason: "no LLM client available for judge".to_string(),
            });
        } else {
            return Err("no LLM client available for judge".to_string());
        }
    };

    let system_prompt = judge_system_prompt(&client.info().model);

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
        // Honest fail-open (issue #31): the decision is Complete, but the
        // user must see why — surface a TUI notice via the engine handle
        // (best-effort: no handle / closed channel = silently dropped).
        // The text names the actual mechanism: an unparseable response is
        // not a failed call.
        if let Some(handle) = notice {
            let text = match &failure {
                JudgeFailure::Unparseable { .. } => "guard judge unparsed — treating as complete",
                JudgeFailure::Transport(_) => "guard judge call failed — treating as complete",
            };
            handle.send(agent_base::types::NoticeKind::Warning, "guard", text);
        }
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
