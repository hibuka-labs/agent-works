use agent_base::engine::react_loop_guard::{GuardAction, GuardCtx, ReactLoopGuard};
use agent_base::llm::StreamClient;
use agent_base::types::{ChatMessage, ResponseFormat};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

/// Default guard configuration
pub struct DefaultGuardConfig {
    /// Maximum retries for reasoning-only responses
    pub reasoning_only_max_strikes: usize,
    /// Maximum retries for empty responses
    pub empty_response_max_strikes: usize,
    /// Nudge message for reasoning-only responses
    pub reasoning_only_nudge: String,
    /// Nudge message for empty responses
    pub empty_response_nudge: String,
    /// Whether to use LLM judge for text-only after tools
    pub use_llm_judge: bool,
    /// Timeout in seconds for the LLM judge call
    pub judge_timeout_secs: u64,
    /// Skip judge if response is longer than this (likely complete)
    pub judge_skip_threshold: usize,
    /// Whether to trust LLM when judge fails or times out.
    /// - true: fail-open (trust the model, end loop)
    /// - false: fail-closed (don't trust, force continue)
    pub judge_fail_open: bool,
    /// Enable short-response detection (merged from CompletionGateMiddleware).
    /// When the user input is long but the model response is very short,
    /// treat it as potentially incomplete and nudge/judge accordingly.
    pub detect_short_response: bool,
    /// Minimum user input character count to trigger short-response detection.
    pub short_response_min_input: usize,
    /// Maximum LLM output character count to be considered a short response.
    pub short_response_max_output: usize,
    /// Nudge message for short responses
    pub short_response_nudge: String,
}

impl Default for DefaultGuardConfig {
    fn default() -> Self {
        Self {
            reasoning_only_max_strikes: 3,
            empty_response_max_strikes: 3,
            reasoning_only_nudge: "You produced internal reasoning but no tool call \
                and no final answer. Make a decision now: call a tool to make progress, \
                or write your final answer as plain text."
                .to_string(),
            empty_response_nudge: "Your response was empty. Please provide a response \
                with either a tool call or your final answer."
                .to_string(),
            use_llm_judge: true,
            judge_timeout_secs: 10,
            judge_skip_threshold: 256,
            judge_fail_open: false, // Default: don't trust LLM on judge failure
            detect_short_response: true,
            short_response_min_input: 128,
            short_response_max_output: 64,
            short_response_nudge: "Your response may be incomplete — \
                you may need to continue."
                .to_string(),
        }
    }
}

/// Default guard implementation
///
/// Does not manage its own state; uses RunState information from GuardCtx.
pub struct DefaultGuard {
    config: DefaultGuardConfig,
    llm_client: Option<Arc<dyn StreamClient>>,
}

impl DefaultGuard {
    pub fn new(config: DefaultGuardConfig) -> Self {
        Self {
            config,
            llm_client: None,
        }
    }

    /// Create a new DefaultGuard with LLM client for judge functionality
    pub fn with_llm_client(config: DefaultGuardConfig, llm_client: Arc<dyn StreamClient>) -> Self {
        Self {
            config,
            llm_client: Some(llm_client),
        }
    }

    /// Call LLM judge to determine if the task is complete
    ///
    /// Used when the model returns text-only after having called tools —
    /// this is suspicious and needs verification.
    async fn call_completion_judge(
        &self,
        user_input: &str,
        model_response: &str,
    ) -> Result<JudgeResult, String> {
        let Some(client) = &self.llm_client else {
            // No LLM client available — use configured behavior
            if self.config.judge_fail_open {
                return Ok(JudgeResult {
                    done: true,
                    reason: "no LLM client available for judge".to_string(),
                });
            } else {
                return Err("no LLM client available for judge".to_string());
            }
        };

        let system_prompt = "You are a task completion judge. \
            Given the user's original question and the agent's response, \
            determine if the agent has sufficiently answered the task. \
            Reply with JSON: {\"done\": true/false, \"reason\": \"brief explanation\"}";

        let user_prompt = format!(
            "【User Question】\n{}\n\n【Agent Response】\n{}",
            user_input, model_response
        );

        let messages = vec![
            ChatMessage::system(system_prompt.to_string()),
            ChatMessage::user(user_prompt),
        ];

        let timeout_duration = Duration::from_secs(self.config.judge_timeout_secs);

        let result = tokio::time::timeout(timeout_duration, async {
            let raw_response = client
                .chat(&messages, &[], None, Some(&ResponseFormat::JsonObject))
                .await
                .map_err(|e| format!("LLM judge call failed: {}", e))?;

            let result: JudgeResult = serde_json::from_str(&raw_response)
                .map_err(|e| format!("Failed to parse judge response: {}", e))?;

            Ok(result)
        })
        .await;

        match result {
            Ok(judge_result) => judge_result,
            Err(_) => {
                // Timeout or failure — use configured behavior
                tracing::warn!(
                    timeout_secs = self.config.judge_timeout_secs,
                    fail_open = self.config.judge_fail_open,
                    "completion judge timeout or failure"
                );
                if self.config.judge_fail_open {
                    // Fail-open: trust the model
                    Ok(JudgeResult {
                        done: true,
                        reason: format!(
                            "judge timeout after {}s, trusting model",
                            self.config.judge_timeout_secs
                        ),
                    })
                } else {
                    // Fail-closed: don't trust the model
                    Err(format!(
                        "judge timeout after {}s, not trusting model",
                        self.config.judge_timeout_secs
                    ))
                }
            }
        }
    }
}

/// LLM judge result
#[derive(serde::Deserialize, Debug)]
struct JudgeResult {
    done: bool,
    reason: String,
}

#[async_trait]
impl ReactLoopGuard for DefaultGuard {
    async fn on_reasoning_only(&self, ctx: &GuardCtx) -> GuardAction {
        // Use RunState information from GuardCtx
        let strikes = ctx.reasoning_only_strikes;

        if strikes >= self.config.reasoning_only_max_strikes {
            return GuardAction::Fail(
                "model produced only reasoning across multiple turns".to_string(),
            );
        }

        GuardAction::Continue(self.config.reasoning_only_nudge.clone())
    }

    async fn on_empty_response(&self, ctx: &GuardCtx) -> GuardAction {
        // Use RunState information from GuardCtx
        let strikes = ctx.empty_response_strikes;

        if strikes >= self.config.empty_response_max_strikes {
            return GuardAction::Fail("model returned empty responses repeatedly".to_string());
        }

        GuardAction::Continue(self.config.empty_response_nudge.clone())
    }

    async fn on_text_only(&self, ctx: &GuardCtx) -> GuardAction {
        let input_len = ctx.user_input.chars().count();
        let output_len = ctx.model_response.chars().count();

        // Short-response detection: user asked a substantial question but the
        // model gave a very short answer — likely incomplete.
        let is_short_response = self.config.detect_short_response
            && input_len > self.config.short_response_min_input
            && output_len < self.config.short_response_max_output
            && input_len > output_len;

        if is_short_response {
            tracing::info!(
                input_chars = input_len,
                output_chars = output_len,
                min_input = self.config.short_response_min_input,
                max_output = self.config.short_response_max_output,
                run_has_tool_calls = ctx.run_has_tool_calls,
                "short response detected in text-only branch"
            );

            if ctx.run_has_tool_calls && self.config.use_llm_judge {
                // Short response after tools — call judge to verify completion
                match self
                    .call_completion_judge(&ctx.user_input, &ctx.model_response)
                    .await
                {
                    Ok(judge) => {
                        if judge.done {
                            GuardAction::Done
                        } else {
                            GuardAction::Continue(format!(
                                "Your answer is incomplete: {}. Continue working on the task.",
                                judge.reason
                            ))
                        }
                    }
                    Err(e) => {
                        // Judge failed — behavior depends on judge_fail_open config
                        tracing::warn!("completion judge failed: {}", e);
                        if self.config.judge_fail_open {
                            GuardAction::Done
                        } else {
                            GuardAction::Continue(
                                "Cannot verify task completion, please continue working."
                                    .to_string(),
                            )
                        }
                    }
                }
            } else {
                // Short response without tools or judge disabled — nudge
                GuardAction::Continue(self.config.short_response_nudge.clone())
            }
        } else if ctx.run_has_tool_calls && self.config.use_llm_judge {
            // Non-short response after tools — check skip threshold
            if output_len >= self.config.judge_skip_threshold {
                tracing::debug!(
                    response_chars = output_len,
                    threshold = self.config.judge_skip_threshold,
                    "text-only response long enough, skipping judge"
                );
                return GuardAction::Done;
            }

            tracing::info!(
                response_chars = output_len,
                threshold = self.config.judge_skip_threshold,
                "text-only response short, calling judge"
            );
            match self
                .call_completion_judge(&ctx.user_input, &ctx.model_response)
                .await
            {
                Ok(judge) => {
                    if judge.done {
                        GuardAction::Done
                    } else {
                        GuardAction::Continue(format!(
                            "Your answer is incomplete: {}. Continue working on the task.",
                            judge.reason
                        ))
                    }
                }
                Err(e) => {
                    // Judge failed — behavior depends on judge_fail_open config
                    tracing::warn!("completion judge failed: {}", e);
                    if self.config.judge_fail_open {
                        GuardAction::Done
                    } else {
                        GuardAction::Continue(
                            "Cannot verify task completion, please continue working."
                                .to_string(),
                        )
                    }
                }
            }
        } else {
            GuardAction::Done
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::types::{FinishReason, SessionId};

    fn make_ctx(
        reasoning_only_strikes: usize,
        empty_response_strikes: usize,
        run_has_tool_calls: bool,
    ) -> GuardCtx {
        GuardCtx {
            session_id: SessionId {
                id: 1,
                external_id: None,
            },
            turn_count: 1,
            user_input: "test".to_string(),
            model_response: "response".to_string(),
            finish_reason: FinishReason::Stop,
            available_tools: vec![],
            reasoning_only_strikes,
            empty_response_strikes,
            run_has_tool_calls,
        }
    }

    #[tokio::test]
    async fn test_reasoning_only_below_threshold() {
        let guard = DefaultGuard::new(DefaultGuardConfig::default());
        let ctx = make_ctx(1, 0, false);

        let action = guard.on_reasoning_only(&ctx).await;
        assert!(matches!(action, GuardAction::Continue(_)));
    }

    #[tokio::test]
    async fn test_reasoning_only_at_threshold() {
        let guard = DefaultGuard::new(DefaultGuardConfig::default());
        let ctx = make_ctx(3, 0, false);

        let action = guard.on_reasoning_only(&ctx).await;
        assert!(matches!(action, GuardAction::Fail(_)));
    }

    #[tokio::test]
    async fn test_empty_response_below_threshold() {
        let guard = DefaultGuard::new(DefaultGuardConfig::default());
        let ctx = make_ctx(0, 1, false);

        let action = guard.on_empty_response(&ctx).await;
        assert!(matches!(action, GuardAction::Continue(_)));
    }

    #[tokio::test]
    async fn test_empty_response_at_threshold() {
        let guard = DefaultGuard::new(DefaultGuardConfig::default());
        let ctx = make_ctx(0, 3, false);

        let action = guard.on_empty_response(&ctx).await;
        assert!(matches!(action, GuardAction::Fail(_)));
    }

    #[tokio::test]
    async fn test_text_only_without_tool_calls() {
        let guard = DefaultGuard::new(DefaultGuardConfig::default());
        let ctx = make_ctx(0, 0, false);

        let action = guard.on_text_only(&ctx).await;
        assert!(matches!(action, GuardAction::Done));
    }

    #[tokio::test]
    async fn test_text_only_with_tool_calls_no_llm_client() {
        let config = DefaultGuardConfig {
            use_llm_judge: true,
            ..DefaultGuardConfig::default()
        };
        let guard = DefaultGuard::new(config);
        let ctx = make_ctx(0, 0, true);

        // No LLM client, judge_fail_open=false (default), should force continue
        let action = guard.on_text_only(&ctx).await;
        assert!(matches!(action, GuardAction::Continue(_)));
    }

    #[tokio::test]
    async fn test_text_only_with_tool_calls_no_llm_client_fail_open() {
        let config = DefaultGuardConfig {
            use_llm_judge: true,
            judge_fail_open: true,
            ..DefaultGuardConfig::default()
        };
        let guard = DefaultGuard::new(config);
        let ctx = make_ctx(0, 0, true);

        // No LLM client, judge_fail_open=true, should trust model (Done)
        let action = guard.on_text_only(&ctx).await;
        assert!(matches!(action, GuardAction::Done));
    }

    #[tokio::test]
    async fn test_text_only_with_tool_calls_judge_disabled() {
        let config = DefaultGuardConfig {
            use_llm_judge: false,
            ..DefaultGuardConfig::default()
        };
        let guard = DefaultGuard::new(config);
        let ctx = make_ctx(0, 0, true);

        // LLM judge disabled, should Done
        let action = guard.on_text_only(&ctx).await;
        assert!(matches!(action, GuardAction::Done));
    }

    #[tokio::test]
    async fn test_custom_config() {
        let config = DefaultGuardConfig {
            reasoning_only_max_strikes: 5,
            empty_response_max_strikes: 2,
            ..DefaultGuardConfig::default()
        };
        let guard = DefaultGuard::new(config);

        // reasoning-only: 4 < 5, should continue
        let ctx = make_ctx(4, 0, false);
        let action = guard.on_reasoning_only(&ctx).await;
        assert!(matches!(action, GuardAction::Continue(_)));

        // empty response: 2 >= 2, should fail
        let ctx = make_ctx(0, 2, false);
        let action = guard.on_empty_response(&ctx).await;
        assert!(matches!(action, GuardAction::Fail(_)));
    }
}
