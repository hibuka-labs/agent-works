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

/// Call LLM judge to determine if the task is complete.
///
/// Used when the model returns text-only after having called tools —
/// this is suspicious and needs verification.
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
        ChatMessage::user(user_prompt),
    ];

    let timeout_duration = Duration::from_secs(judge_timeout_secs);

    let result = tokio::time::timeout(timeout_duration, async {
        let request = ChatRequest::new(messages)
            .with_response_format(agent_base::llm_trait::request::ResponseFormat::JsonObject);
        let response = client
            .chat(request)
            .await
            .map_err(|e| format!("LLM judge call failed: {}", e))?;

        let result: JudgeResult = serde_json::from_str(&response.content)
            .map_err(|e| format!("Failed to parse judge response: {}", e))?;

        Ok(result)
    })
    .await;

    // Flatten: unwrap the timeout layer, treating timeout as an error.
    let result: Result<JudgeResult, String> = match result {
        Ok(inner) => inner, // inner is Result<JudgeResult, String>
        Err(_elapsed) => Err(format!("judge timeout after {}s", judge_timeout_secs)),
    };

    match result {
        Ok(judge_result) => Ok(judge_result),
        Err(e) => {
            // All failures (timeout, client error, parse error) — use configured behavior
            tracing::warn!(
                error = %e,
                fail_open = judge_fail_open,
                "completion judge failed"
            );
            if judge_fail_open {
                // Fail-open: trust the model
                Ok(JudgeResult {
                    done: true,
                    reason: format!("judge failed ({}), trusting model", e),
                })
            } else {
                // Fail-closed: don't trust the model
                Err(e)
            }
        }
    }
}
