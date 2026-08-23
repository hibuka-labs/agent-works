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
    /// Number of recent user messages to include in the judge prompt.
    /// Helps the judge understand context like "继续" after a multi-turn discussion.
    pub recent_user_count: usize,
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
            recent_user_count: 5,
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
        all_user_inputs: &[String],
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
            Given the user's conversation history and the agent's response, \
            determine if the agent has sufficiently answered the task. \
            Reply with JSON: {\"done\": true/false, \"reason\": \"brief explanation\"}";

        // Build context from recent user messages
        let user_context = if all_user_inputs.is_empty() {
            user_input.to_string()
        } else {
            let n = self.config.recent_user_count;
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
                // Skip LLM judge for very large inputs — judge would be too slow
                const INPUT_LEN_LIMIT: usize = 10_000;
                if input_len > INPUT_LEN_LIMIT {
                    tracing::info!(
                        input_chars = input_len,
                        input_limit = INPUT_LEN_LIMIT,
                        "skipping LLM judge — user input too large, trusting model"
                    );
                    return GuardAction::Done;
                }
                // Short response after tools — call judge to verify completion
                match self
                    .call_completion_judge(
                        &ctx.user_input,
                        &ctx.model_response,
                        &ctx.all_user_inputs,
                    )
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

            // Skip LLM judge for very large inputs — judge would be too slow
            const INPUT_LEN_LIMIT: usize = 10_000;
            if input_len > INPUT_LEN_LIMIT {
                tracing::info!(
                    input_chars = input_len,
                    input_limit = INPUT_LEN_LIMIT,
                    "skipping LLM judge — user input too large, trusting model"
                );
                return GuardAction::Done;
            }

            tracing::info!(
                response_chars = output_len,
                threshold = self.config.judge_skip_threshold,
                "text-only response short, calling judge"
            );
            match self
                .call_completion_judge(&ctx.user_input, &ctx.model_response, &ctx.all_user_inputs)
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
                            "Cannot verify task completion, please continue working.".to_string(),
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
    use agent_base::llm::{LlmCapabilities, StreamChunk};
    use agent_base::types::{AgentResult, FinishReason, SessionId};
    use futures_core::Stream;
    use serde_json::Value;
    use std::pin::Pin;
    use std::task::{Context, Poll};

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
            all_user_inputs: vec!["test".to_string()],
        }
    }

    /// Build a GuardCtx with custom input/response text (for length-sensitive tests).
    fn make_ctx_with_text(
        user_input: &str,
        model_response: &str,
        run_has_tool_calls: bool,
    ) -> GuardCtx {
        GuardCtx {
            session_id: SessionId {
                id: 1,
                external_id: None,
            },
            turn_count: 2,
            user_input: user_input.to_string(),
            model_response: model_response.to_string(),
            finish_reason: FinishReason::Stop,
            available_tools: vec!["echo".to_string()],
            reasoning_only_strikes: 0,
            empty_response_strikes: 0,
            run_has_tool_calls,
            all_user_inputs: vec![user_input.to_string()],
        }
    }

    // ── Mock LLM clients for judge testing ──

    /// Mock StreamClient that returns a fixed JSON response (for judge calls).
    struct MockJudgeClient {
        response: String,
    }

    impl MockJudgeClient {
        fn new(response: Value) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[async_trait]
    impl StreamClient for MockJudgeClient {
        async fn stream(
            &self,
            _messages: &[ChatMessage],
            _tools: &[Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>>> {
            let chunks = vec![
                Ok(StreamChunk::Text(self.response.clone())),
                Ok(StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ];
            Ok(Box::pin(futures_util::stream::iter(chunks)))
        }

        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::default()
        }
    }

    /// Mock StreamClient that never resolves (for timeout testing).
    struct MockTimeoutClient;

    #[async_trait]
    impl StreamClient for MockTimeoutClient {
        async fn stream(
            &self,
            _messages: &[ChatMessage],
            _tools: &[Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>>> {
            // Return a stream whose first item never resolves.
            struct HangingStream;
            impl Stream for HangingStream {
                type Item = AgentResult<StreamChunk>;
                fn poll_next(
                    self: Pin<&mut Self>,
                    _cx: &mut Context<'_>,
                ) -> Poll<Option<Self::Item>> {
                    Poll::Pending // never wakes
                }
            }
            Ok(Box::pin(HangingStream))
        }

        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::default()
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

    // ── Branch 4: text-only after tools — LLM judge tests ──

    #[tokio::test]
    async fn test_text_only_judge_says_done() {
        let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": true,
            "reason": "task is complete"
        })));
        let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
        let ctx = make_ctx_with_text(
            "what is 2+2?",
            "The answer is 4.",
            true, // run_has_tool_calls
        );

        let action = guard.on_text_only(&ctx).await;
        assert!(
            matches!(action, GuardAction::Done),
            "judge says done → Done, got: {:?}",
            action
        );
    }

    #[tokio::test]
    async fn test_text_only_judge_says_not_done() {
        let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": false,
            "reason": "only answered part of the question"
        })));
        let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
        let ctx = make_ctx_with_text(
            "list all files and explain each",
            "Here are the files:", // incomplete
            true,
        );

        let action = guard.on_text_only(&ctx).await;
        match &action {
            GuardAction::Continue(msg) => {
                assert!(
                    msg.contains("incomplete"),
                    "nudge should mention incomplete: {}",
                    msg
                );
                assert!(
                    msg.contains("only answered part"),
                    "nudge should include judge reason: {}",
                    msg
                );
            }
            other => panic!("expected Continue, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_text_only_short_response_detected_no_tools() {
        // Long input (>128 chars) + short output (<64 chars) + no tool calls → nudge
        let long_input = "a".repeat(200);
        let short_output = "done";
        let guard = DefaultGuard::new(DefaultGuardConfig::default());
        let ctx = make_ctx_with_text(&long_input, short_output, false);

        let action = guard.on_text_only(&ctx).await;
        match &action {
            GuardAction::Continue(msg) => {
                assert!(msg.contains("incomplete"), "short response nudge: {}", msg);
            }
            other => panic!("expected Continue for short response, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_text_only_short_response_with_judge_done() {
        // Short response + tools + judge says done → Done
        let long_input = "a".repeat(200);
        let short_output = "42";
        let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": true,
            "reason": "answer is correct"
        })));
        let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
        let ctx = make_ctx_with_text(&long_input, short_output, true);

        let action = guard.on_text_only(&ctx).await;
        assert!(
            matches!(action, GuardAction::Done),
            "short response + judge done → Done, got: {:?}",
            action
        );
    }

    #[tokio::test]
    async fn test_text_only_skip_threshold() {
        // Output >= 256 chars → skip judge entirely → Done
        let long_output = "x".repeat(300);
        let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": false,
            "reason": "would be incomplete but skipped"
        })));
        let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
        let ctx = make_ctx_with_text("query", &long_output, true);

        let action = guard.on_text_only(&ctx).await;
        assert!(
            matches!(action, GuardAction::Done),
            "response >= skip_threshold → Done without calling judge, got: {:?}",
            action
        );
    }

    #[tokio::test]
    async fn test_text_only_judge_timeout_fail_open() {
        let judge_client = Arc::new(MockTimeoutClient);
        let config = DefaultGuardConfig {
            judge_fail_open: true,
            judge_timeout_secs: 1, // 1 second timeout
            ..DefaultGuardConfig::default()
        };
        let guard = DefaultGuard::with_llm_client(config, judge_client);
        let ctx = make_ctx_with_text("query", "short answer", true);

        let action = guard.on_text_only(&ctx).await;
        assert!(
            matches!(action, GuardAction::Done),
            "timeout + fail_open → Done (trust model), got: {:?}",
            action
        );
    }

    #[tokio::test]
    async fn test_text_only_judge_timeout_fail_closed() {
        let judge_client = Arc::new(MockTimeoutClient);
        let config = DefaultGuardConfig {
            judge_fail_open: false,
            judge_timeout_secs: 1, // 1 second timeout
            ..DefaultGuardConfig::default()
        };
        let guard = DefaultGuard::with_llm_client(config, judge_client);
        let ctx = make_ctx_with_text("query", "short answer", true);

        let action = guard.on_text_only(&ctx).await;
        match &action {
            GuardAction::Continue(msg) => {
                assert!(
                    msg.contains("Cannot verify"),
                    "fail_closed timeout → Continue with verify message: {}",
                    msg
                );
            }
            other => panic!(
                "expected Continue for timeout + fail_closed, got: {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    async fn test_text_only_skip_judge_when_input_too_large() {
        // user_input > 10k chars → skip judge, trust model (Done)
        let long_input = "a".repeat(10_001);
        let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": false,
            "reason": "would be incomplete but skipped"
        })));
        let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
        let ctx = make_ctx_with_text(&long_input, "short answer", true);

        let action = guard.on_text_only(&ctx).await;
        assert!(
            matches!(action, GuardAction::Done),
            "input > 10k → skip judge → Done, got: {:?}",
            action
        );
    }

    #[tokio::test]
    async fn test_text_only_judge_with_multiple_user_messages() {
        // Simulates: user discusses a problem over 3 turns, then says "继续".
        // The judge should see all messages, not just "继续".
        let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": false,
            "reason": "agent has not finished the task yet"
        })));
        let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);

        let mut ctx = make_ctx_with_text("继续", "I'll continue working on it.", true);
        ctx.all_user_inputs = vec![
            "帮我分析一下这个 bug 的根因".to_string(),
            "好的，那你帮我修复一下".to_string(),
            "继续".to_string(),
        ];

        let action = guard.on_text_only(&ctx).await;
        match &action {
            GuardAction::Continue(msg) => {
                assert!(
                    msg.contains("incomplete"),
                    "judge should see history and say incomplete: {}",
                    msg
                );
            }
            other => panic!("expected Continue (judge says not done), got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_text_only_judge_respects_recent_user_count() {
        // recent_user_count=2, but there are 5 messages in all_user_inputs.
        // Judge should only see the last 2.
        let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": true,
            "reason": "task complete"
        })));
        let config = DefaultGuardConfig {
            recent_user_count: 2,
            ..DefaultGuardConfig::default()
        };
        let guard = DefaultGuard::with_llm_client(config, judge_client);

        let mut ctx = make_ctx_with_text("继续", "done", true);
        ctx.all_user_inputs = vec![
            "msg1".to_string(),
            "msg2".to_string(),
            "msg3".to_string(),
            "msg4".to_string(),
            "继续".to_string(),
        ];

        let action = guard.on_text_only(&ctx).await;
        assert!(
            matches!(action, GuardAction::Done),
            "judge says done → Done, got: {:?}",
            action
        );
    }
}
