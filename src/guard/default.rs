use agent_base::engine::react_loop_guard::{GuardCtx, GuardDecision, ReactLoopGuard};
use agent_base::llm_trait::LlmProvider;
use async_trait::async_trait;
use std::sync::Arc;

use super::config::{DefaultGuardConfig, ReasoningOnlyAction};
use super::judge::call_completion_judge;

/// Default guard implementation
///
/// Does not manage its own state; uses RunState information from GuardCtx.
pub struct DefaultGuard {
    config: DefaultGuardConfig,
    llm_client: Option<Arc<dyn LlmProvider>>,
}

impl DefaultGuard {
    pub fn new(config: DefaultGuardConfig) -> Self {
        Self {
            config,
            llm_client: None,
        }
    }

    /// Create a new DefaultGuard with LLM client for judge functionality
    pub fn with_llm_client(config: DefaultGuardConfig, llm_client: Arc<dyn LlmProvider>) -> Self {
        Self {
            config,
            llm_client: Some(llm_client),
        }
    }

    // ── Scene handlers ──────────────────────────────────────────────────

    async fn handle_reasoning_only(&self, ctx: &GuardCtx) -> GuardDecision {
        let strikes = ctx.reasoning_only_strikes;

        match self.config.reasoning_only_action {
            ReasoningOnlyAction::Fail => {
                // Default behavior: fail after max strikes
                if strikes >= self.config.reasoning_only_max_strikes {
                    return GuardDecision::Fail {
                        error: "model produced only reasoning across multiple turns".to_string(),
                    };
                }

                GuardDecision::Continue {
                    nudge: Some(self.config.reasoning_only_nudge.clone()),
                }
            }
            ReasoningOnlyAction::DisableThinking => {
                // New behavior: disable thinking after max strikes
                if strikes >= self.config.reasoning_only_max_strikes {
                    // Check if thinking is already disabled
                    if ctx.thinking_disabled {
                        // Thinking is already disabled but still reasoning-only → fail
                        return GuardDecision::Fail {
                            error: "model produced only reasoning even after thinking was disabled"
                                .to_string(),
                        };
                    }

                    // Disable thinking and continue
                    return GuardDecision::DisableThinking {
                        nudge: self.config.disable_thinking_nudge.clone(),
                    };
                }

                GuardDecision::Continue {
                    nudge: Some(self.config.reasoning_only_nudge.clone()),
                }
            }
        }
    }

    async fn handle_empty_response(&self, ctx: &GuardCtx) -> GuardDecision {
        let strikes = ctx.empty_response_strikes;

        if strikes >= self.config.empty_response_max_strikes {
            return GuardDecision::Fail {
                error: "model returned empty responses repeatedly".to_string(),
            };
        }

        GuardDecision::Continue {
            nudge: Some(self.config.empty_response_nudge.clone()),
        }
    }

    async fn handle_text_only(&self, ctx: &GuardCtx) -> GuardDecision {
        // Session 20260904_efad759c: after the react-side truncation guard
        // rejected a spawn_agent call, the model replied with a text-only
        // "success" narrative and the completion judge — which sees only the
        // user inputs and that narrative — passed it, ending the run with
        // zero children spawned. A text-only turn that follows rejected tool
        // calls is definitionally not completion: the work the text describes
        // never executed. Skip the judge, push the model back to re-issuing.
        // (Bounded: the react truncation breaker fails the run at its strike
        // limit, and max_turns still applies.)
        if ctx.last_tool_calls_invalid {
            tracing::info!(
                session_id = ctx.session_id.id,
                turn = ctx.turn_count,
                "text-only response after rejected tool calls — not completion, re-issuing"
            );
            return GuardDecision::Continue {
                nudge: Some(
                    "Your previous tool call was NOT executed — its arguments \
                     were invalid or truncated. The report you just wrote \
                     describes work that never happened; do not narrate \
                     results. Re-issue the tool call with complete, valid \
                     JSON arguments."
                        .to_string(),
                ),
            };
        }

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
                // Skip LLM judge for very large inputs — judge would be too slow
                const INPUT_LEN_LIMIT: usize = 10_000;
                if input_len > INPUT_LEN_LIMIT {
                    tracing::info!(
                        input_chars = input_len,
                        input_limit = INPUT_LEN_LIMIT,
                        "skipping LLM judge — user input too large, trusting model"
                    );
                    return GuardDecision::Complete;
                }
                // Short response after tools — call judge to verify completion
                match call_completion_judge(
                    self.llm_client.as_ref(),
                    &ctx.user_input,
                    &ctx.model_response,
                    &ctx.all_user_inputs,
                    self.config.judge_fail_open,
                    self.config.judge_timeout_secs,
                    self.config.recent_user_count,
                )
                .await
                {
                    Ok(judge) => {
                        if judge.done {
                            GuardDecision::Complete
                        } else {
                            GuardDecision::Continue {
                                nudge: Some(format!(
                                    "Your answer is incomplete: {}. Continue working on the task.",
                                    judge.reason
                                )),
                            }
                        }
                    }
                    Err(e) => {
                        // Judge failed — behavior depends on judge_fail_open config
                        tracing::warn!("completion judge failed: {}", e);
                        if self.config.judge_fail_open {
                            GuardDecision::Complete
                        } else {
                            GuardDecision::Continue {
                                nudge: Some(
                                    "Cannot verify task completion, please continue working."
                                        .to_string(),
                                ),
                            }
                        }
                    }
                }
            } else {
                // Short response without tools or judge disabled — nudge
                GuardDecision::Continue {
                    nudge: Some(self.config.short_response_nudge.clone()),
                }
            }
        } else if ctx.run_has_tool_calls && self.config.use_llm_judge {
            // Non-short response after tools — check skip threshold
            if output_len >= self.config.judge_skip_threshold {
                tracing::debug!(
                    response_chars = output_len,
                    threshold = self.config.judge_skip_threshold,
                    "text-only response long enough, skipping judge"
                );
                return GuardDecision::Complete;
            }

            // Skip LLM judge for very large inputs — judge would be too slow
            const INPUT_LEN_LIMIT: usize = 10_000;
            if input_len > INPUT_LEN_LIMIT {
                tracing::info!(
                    input_chars = input_len,
                    input_limit = INPUT_LEN_LIMIT,
                    "skipping LLM judge — user input too large, trusting model"
                );
                return GuardDecision::Complete;
            }

            tracing::info!(
                response_chars = output_len,
                threshold = self.config.judge_skip_threshold,
                "text-only response short, calling judge"
            );
            match call_completion_judge(
                self.llm_client.as_ref(),
                &ctx.user_input,
                &ctx.model_response,
                &ctx.all_user_inputs,
                self.config.judge_fail_open,
                self.config.judge_timeout_secs,
                self.config.recent_user_count,
            )
            .await
            {
                Ok(judge) => {
                    if judge.done {
                        GuardDecision::Complete
                    } else {
                        GuardDecision::Continue {
                            nudge: Some(format!(
                                "Your answer is incomplete: {}. Continue working on the task.",
                                judge.reason
                            )),
                        }
                    }
                }
                Err(e) => {
                    // Judge failed — behavior depends on judge_fail_open config
                    tracing::warn!("completion judge failed: {}", e);
                    if self.config.judge_fail_open {
                        GuardDecision::Complete
                    } else {
                        GuardDecision::Continue {
                            nudge: Some(
                                "Cannot verify task completion, please continue working."
                                    .to_string(),
                            ),
                        }
                    }
                }
            }
        } else {
            GuardDecision::Complete
        }
    }
}

#[async_trait]
impl ReactLoopGuard for DefaultGuard {
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision {
        if ctx.is_reasoning_only {
            self.handle_reasoning_only(ctx).await
        } else if ctx.is_empty_response {
            self.handle_empty_response(ctx).await
        } else if ctx.is_text_only {
            self.handle_text_only(ctx).await
        } else {
            GuardDecision::Complete
        }
    }

    async fn on_tool_call(&self, ctx: &GuardCtx) -> GuardDecision {
        // Restore thinking when:
        // 1. Thinking is currently disabled (by guard)
        // 2. Original thinking was enabled (user wanted thinking)
        // 3. Model calls a tool (showing it's working again)
        if ctx.thinking_disabled && ctx.original_thinking_enabled {
            tracing::info!(
                session_id = ctx.session_id.id,
                turn = ctx.turn_count,
                "tool call detected while thinking disabled, restoring thinking"
            );
            return GuardDecision::RestoreThinking;
        }

        GuardDecision::Complete
    }
}
