use super::NoopGuard;
use super::config::{DefaultGuardConfig, ReasoningOnlyAction};
use super::default::DefaultGuard;
use agent_base::engine::react_loop_guard::{GuardCtx, GuardDecision, ReactLoopGuard};
use agent_base::llm_trait::response::{
    ChatResponse, ChatStream, FinishReason as LlmFinishReason, StreamChunk,
};
use agent_base::llm_trait::{Capabilities, ChatRequest, LlmError, LlmProvider};
use agent_base::types::{FinishReason, SessionId};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;

fn make_ctx(
    reasoning_only_strikes: usize,
    empty_response_strikes: usize,
    run_has_tool_calls: bool,
    is_reasoning_only: bool,
    is_empty_response: bool,
    is_text_only: bool,
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
        is_reasoning_only,
        is_empty_response,
        is_text_only,
        thinking_disabled: false,
        original_thinking_enabled: true,
    }
}

/// Build a GuardCtx with custom input/response text (for length-sensitive tests).
fn make_ctx_with_text(
    user_input: &str,
    model_response: &str,
    run_has_tool_calls: bool,
    is_text_only: bool,
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
        is_reasoning_only: false,
        is_empty_response: false,
        is_text_only,
        thinking_disabled: false,
        original_thinking_enabled: true,
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

#[async_trait::async_trait]
impl LlmProvider for MockJudgeClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chunks = vec![
            Ok(StreamChunk::Text(self.response.clone())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ];
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
            chunks,
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            content: self.response.clone(),
            tool_calls: vec![],
            usage: agent_base::UsageInfo::default(),
            finish_reason: LlmFinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "mock-judge".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

/// Mock StreamClient that never resolves (for timeout testing).
struct MockTimeoutClient;

#[async_trait::async_trait]
impl LlmProvider for MockTimeoutClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        // Return a stream whose first item never resolves.
        struct HangingStream;
        impl futures_core::Stream for HangingStream {
            type Item = Result<StreamChunk, LlmError>;
            fn poll_next(
                self: Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Self::Item>> {
                std::task::Poll::Pending // never wakes
            }
        }
        Ok(ChatStream::new(Box::pin(HangingStream)))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        // Never resolve — will be interrupted by timeout.
        std::future::pending().await
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "mock-timeout".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

/// Mock StreamClient that always returns an error (for judge error testing).
struct MockErrorClient;

#[async_trait::async_trait]
impl LlmProvider for MockErrorClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Err(LlmError::llm("simulated stream error"))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Err(LlmError::llm("simulated judge error"))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "mock-error".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

// ── Reasoning-only tests ──

#[tokio::test]
async fn test_reasoning_only_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(1, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(
        decision,
        GuardDecision::Continue { nudge: Some(_) }
    ));
}

#[tokio::test]
async fn test_reasoning_only_at_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(3, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Fail { .. }));
}

#[tokio::test]
async fn test_reasoning_only_with_tools_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = make_ctx(1, 0, false, true, false, false);
    ctx.available_tools = vec!["echo".to_string()];
    ctx.run_has_tool_calls = true;

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { nudge: Some(_) }),
        "reasoning_only with tools, below threshold → Continue, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_reasoning_only_with_tools_at_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = make_ctx(3, 0, false, true, false, false);
    ctx.available_tools = vec!["echo".to_string()];
    ctx.run_has_tool_calls = true;

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "reasoning_only with tools, at threshold → Fail, got: {:?}",
        decision
    );
}

// ── Empty response tests ──

#[tokio::test]
async fn test_empty_response_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 1, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(
        decision,
        GuardDecision::Continue { nudge: Some(_) }
    ));
}

#[tokio::test]
async fn test_empty_response_at_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 3, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Fail { .. }));
}

#[tokio::test]
async fn test_empty_response_with_tool_calls() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 1, true, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { nudge: Some(_) }),
        "empty response with tools → Continue (nudge), got: {:?}",
        decision
    );
}

// ── Text-only tests ──

#[tokio::test]
async fn test_text_only_without_tool_calls() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, false, false, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Complete));
}

#[tokio::test]
async fn test_text_only_with_tool_calls_no_llm_client() {
    let config = DefaultGuardConfig {
        use_llm_judge: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx(0, 0, true, false, false, true);

    // No LLM client, judge_fail_open=false (default), should force continue
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(
        decision,
        GuardDecision::Continue { nudge: Some(_) }
    ));
}

#[tokio::test]
async fn test_text_only_with_tool_calls_no_llm_client_fail_open() {
    let config = DefaultGuardConfig {
        use_llm_judge: true,
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx(0, 0, true, false, false, true);

    // No LLM client, judge_fail_open=true, should trust model (Complete)
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Complete));
}

#[tokio::test]
async fn test_text_only_with_tool_calls_judge_disabled() {
    let config = DefaultGuardConfig {
        use_llm_judge: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx(0, 0, true, false, false, true);

    // LLM judge disabled, should Complete
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Complete));
}

#[tokio::test]
async fn test_text_only_no_tools_judge_enabled_no_client() {
    let config = DefaultGuardConfig {
        use_llm_judge: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let long_output = "x".repeat(300);
    let ctx = make_ctx_with_text("query", &long_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "no tools + judge enabled + no client → Complete (skips judge), got: {:?}",
        decision
    );
}

// ── Custom config tests ──

#[tokio::test]
async fn test_custom_config() {
    let config = DefaultGuardConfig {
        reasoning_only_max_strikes: 5,
        empty_response_max_strikes: 2,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // reasoning-only: 4 < 5, should continue
    let ctx = make_ctx(4, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(
        decision,
        GuardDecision::Continue { nudge: Some(_) }
    ));

    // empty response: 2 >= 2, should fail
    let ctx = make_ctx(0, 2, false, false, true, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Fail { .. }));
}

// ── LLM judge tests ──

#[tokio::test]
async fn test_text_only_judge_says_done() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": true,
        "reason": "task is complete"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("what is 2+2?", "The answer is 4.", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "judge says done → Complete, got: {:?}",
        decision
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
        "Here are the files:",
        true,
        true,
    );

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
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
async fn test_text_only_judge_reason_propagated_to_nudge() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "missing file list and explanation"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("list files and explain", "here are the files:", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("missing file list and explanation"),
                "nudge should include judge reason: {}",
                msg
            );
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

// ── Short response detection tests ──

#[tokio::test]
async fn test_text_only_short_response_detected_no_tools() {
    let long_input = "a".repeat(200);
    let short_output = "done";
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&long_input, short_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(msg.contains("incomplete"), "short response nudge: {}", msg);
        }
        other => panic!("expected Continue for short response, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_text_only_short_response_with_judge_done() {
    let long_input = "a".repeat(200);
    let short_output = "42";
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": true,
        "reason": "answer is correct"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text(&long_input, short_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "short response + judge done → Complete, got: {:?}",
        decision
    );
}

// ── Skip threshold tests ──

#[tokio::test]
async fn test_text_only_skip_threshold() {
    let long_output = "x".repeat(300);
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "would be incomplete but skipped"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("query", &long_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "response >= skip_threshold → Complete without calling judge, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_text_only_skip_judge_when_input_too_large() {
    let long_input = "a".repeat(10_001);
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "would be incomplete but skipped"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text(&long_input, "short answer", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "input > 10k → skip judge → Complete, got: {:?}",
        decision
    );
}

// ── Judge timeout tests ──

#[tokio::test]
async fn test_text_only_judge_timeout_fail_open() {
    let judge_client = Arc::new(MockTimeoutClient);
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        judge_timeout_secs: 1,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short answer", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "timeout + fail_open → Complete (trust model), got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_text_only_judge_timeout_fail_closed() {
    let judge_client = Arc::new(MockTimeoutClient);
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        judge_timeout_secs: 1,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short answer", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
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

// ── Judge error tests ──

#[tokio::test]
async fn test_text_only_judge_error_fail_closed() {
    let judge_client = Arc::new(MockErrorClient);
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "fail_closed error → Continue with verify message: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for judge error + fail_closed, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_text_only_judge_error_fail_open() {
    let judge_client = Arc::new(MockErrorClient);
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "fail_open + judge error → Complete (trust model), got: {:?}",
        decision
    );
}

// ── Short response + judge combination tests ──

#[tokio::test]
async fn test_text_only_short_response_judge_timeout_fail_closed() {
    let judge_client = Arc::new(MockTimeoutClient);
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        judge_timeout_secs: 1,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let long_input = "a".repeat(200);
    let ctx = make_ctx_with_text(&long_input, "42", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "short + timeout + fail_closed → Continue: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for short response + timeout + fail_closed, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_text_only_short_response_judge_error_fail_closed() {
    let judge_client = Arc::new(MockErrorClient);
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let long_input = "a".repeat(200);
    let ctx = make_ctx_with_text(&long_input, "42", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "short + error + fail_closed → Continue: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for short response + judge error + fail_closed, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_text_only_short_response_judge_error_fail_open() {
    let judge_client = Arc::new(MockErrorClient);
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let long_input = "a".repeat(200);
    let ctx = make_ctx_with_text(&long_input, "42", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "short + error + fail_open → Complete, got: {:?}",
        decision
    );
}

// ── Multi-turn context tests ──

#[tokio::test]
async fn test_text_only_judge_with_multiple_user_messages() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "agent has not finished the task yet"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);

    let mut ctx = make_ctx_with_text("继续", "I'll continue working on it.", true, true);
    ctx.all_user_inputs = vec![
        "帮我分析一下这个 bug 的根因".to_string(),
        "好的，那你帮我修复一下".to_string(),
        "继续".to_string(),
    ];

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
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
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": true,
        "reason": "task complete"
    })));
    let config = DefaultGuardConfig {
        recent_user_count: 2,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);

    let mut ctx = make_ctx_with_text("继续", "done", true, true);
    ctx.all_user_inputs = vec![
        "msg1".to_string(),
        "msg2".to_string(),
        "msg3".to_string(),
        "msg4".to_string(),
        "继续".to_string(),
    ];

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "judge says done → Complete, got: {:?}",
        decision
    );
}

// ── Edge case: no scene flags ──

#[tokio::test]
async fn test_no_scene_flags_returns_complete() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, false, false, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "no scene flags → Complete, got: {:?}",
        decision
    );
}

// ═══════════════════════════════════════════════════════════════
// Phase 3: Boundary cases and coverage gaps
// ═══════════════════════════════════════════════════════════════

// ── Boundary: strikes = 0 (first occurrence) ──

#[tokio::test]
async fn test_reasoning_only_strikes_zero() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("reasoning") || msg.contains("tool call"),
                "nudge should mention the issue: {}",
                msg
            );
        }
        other => panic!("strikes=0 → Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_empty_response_strikes_zero() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("empty") || msg.contains("response"),
                "nudge should mention empty response: {}",
                msg
            );
        }
        other => panic!("strikes=0 → Continue, got: {:?}", other),
    }
}

// ── Boundary: strikes = threshold - 1 (just below) ──

#[tokio::test]
async fn test_reasoning_only_just_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    // default threshold = 3, so 2 should Continue
    let ctx = make_ctx(2, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { nudge: Some(_) }),
        "strikes=2, threshold=3 → Continue, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_empty_response_just_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 2, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { nudge: Some(_) }),
        "strikes=2, threshold=3 → Continue, got: {:?}",
        decision
    );
}

// ── Nudge message content verification ──

#[tokio::test]
async fn test_reasoning_only_nudge_content() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(1, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().reasoning_only_nudge);
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_empty_response_nudge_content() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 1, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().empty_response_nudge);
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_short_response_nudge_content() {
    let long_input = "a".repeat(200);
    let short_output = "ok";
    let config = DefaultGuardConfig {
        use_llm_judge: false, // disable judge to get nudge path
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx_with_text(&long_input, short_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().short_response_nudge);
        }
        other => panic!("expected Continue for short response, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_fail_error_message_reasoning_only() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(3, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Fail { error } => {
            assert!(
                error.contains("reasoning"),
                "fail error should mention reasoning: {}",
                error
            );
        }
        other => panic!("expected Fail, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_fail_error_message_empty_response() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 3, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Fail { error } => {
            assert!(
                error.contains("empty"),
                "fail error should mention empty: {}",
                error
            );
        }
        other => panic!("expected Fail, got: {:?}", other),
    }
}

// ── Config default values ──

#[test]
fn test_default_guard_config_values() {
    let config = DefaultGuardConfig::default();
    assert_eq!(config.reasoning_only_max_strikes, 3);
    assert_eq!(config.empty_response_max_strikes, 3);
    assert!(config.use_llm_judge);
    assert_eq!(config.judge_timeout_secs, 10);
    assert_eq!(config.judge_skip_threshold, 256);
    assert!(!config.judge_fail_open);
    assert!(config.detect_short_response);
    assert_eq!(config.short_response_min_input, 128);
    assert_eq!(config.short_response_max_output, 64);
    assert_eq!(config.recent_user_count, 5);
}

// ── Text-only: tools present + judge disabled ──

#[tokio::test]
async fn test_text_only_with_tools_judge_disabled_returns_complete() {
    let config = DefaultGuardConfig {
        use_llm_judge: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx(0, 0, true, false, false, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "tools + judge disabled → Complete, got: {:?}",
        decision
    );
}

// ── Text-only: short response without tools → nudge ──

#[tokio::test]
async fn test_text_only_short_response_without_tools_nudge() {
    let long_input = "a".repeat(200);
    let short_output = "done";
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&long_input, short_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            // Without tools, uses short_response_nudge
            assert!(
                msg.contains("incomplete"),
                "should use short_response_nudge: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for short response without tools, got: {:?}",
            other
        ),
    }
}

// ── Text-only: input shorter than output (not short response) ──

#[tokio::test]
async fn test_text_only_input_shorter_than_output_not_short() {
    let short_input = "hi";
    let long_output = "a".repeat(200);
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(short_input, &long_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // input_len (2) < output_len (200), and input_len (2) < min_input (128)
    // → not short response → no tools → Complete
    assert!(
        matches!(decision, GuardDecision::Complete),
        "short input + long output + no tools → Complete, got: {:?}",
        decision
    );
}

// ── Text-only: input just at short_response_min_input boundary ──

#[tokio::test]
async fn test_text_only_short_response_exact_min_input() {
    // Exactly at min_input (128) — uses strict `>`, so 128 > 128 is false → NOT short
    let input = "a".repeat(128);
    let output = "x"; // < max_output (64), and input_len > output_len
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&input, output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // input_len=128 is NOT > min_input=128 (strict), so NOT short response
    // no tools → Complete
    assert!(
        matches!(decision, GuardDecision::Complete),
        "exact min_input (128 > 128 is false) → not short → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_text_only_short_response_just_above_min_input() {
    // One char above min_input — should trigger short response
    let input = "a".repeat(129);
    let output = "x";
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&input, output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // input_len=129 > min_input=128, output=1 < max_output=64, 129 > 1
    // → short response → no tools → nudge
    match &decision {
        GuardDecision::Continue { nudge } => {
            assert!(nudge.is_some(), "just above min_input → short → nudge");
        }
        other => panic!(
            "expected Continue for just above boundary, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_text_only_not_short_response_just_below_min_input() {
    // One char below min_input — should NOT trigger short response
    let input = "a".repeat(127);
    let output = "x";
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&input, output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // input_len=127 < min_input=128 → not short response → no tools → Complete
    assert!(
        matches!(decision, GuardDecision::Complete),
        "below min_input → not short → Complete, got: {:?}",
        decision
    );
}

// ── Text-only: output exactly at max_output boundary ──

#[tokio::test]
async fn test_text_only_short_response_output_at_max() {
    let input = "a".repeat(200);
    let output = "a".repeat(64); // exactly at max_output
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&input, &output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // output_len=64 is NOT < max_output=64, so NOT short response
    // no tools → Complete
    assert!(
        matches!(decision, GuardDecision::Complete),
        "output at max_output → not short → Complete, got: {:?}",
        decision
    );
}

// ── Custom nudge messages ──

#[tokio::test]
async fn test_custom_nudge_messages() {
    let config = DefaultGuardConfig {
        reasoning_only_nudge: "custom reasoning nudge".to_string(),
        empty_response_nudge: "custom empty nudge".to_string(),
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // reasoning_only
    let ctx = make_ctx(1, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            assert_eq!(nudge.as_deref(), Some("custom reasoning nudge"));
        }
        other => panic!("expected Continue, got: {:?}", other),
    }

    // empty_response
    let ctx = make_ctx(0, 1, false, false, true, false);
    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            assert_eq!(nudge.as_deref(), Some("custom empty nudge"));
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

// ── Judge: malformed JSON response ──

#[tokio::test]
async fn test_judge_malformed_json_fail_closed() {
    // Mock that returns non-JSON
    struct MalformedJsonClient;
    #[async_trait::async_trait]
    impl LlmProvider for MalformedJsonClient {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            let chunks = vec![
                Ok(StreamChunk::Text("not json at all".to_string())),
                Ok(StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ];
            Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
                chunks,
            ))))
        }
        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                content: "not json at all".to_string(),
                tool_calls: vec![],
                usage: agent_base::UsageInfo::default(),
                finish_reason: LlmFinishReason::Stop,
                raw: None,
                reasoning_content: None,
                thinking_signature: None,
            })
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }
        fn info(&self) -> agent_base::llm_trait::ProviderInfo {
            agent_base::llm_trait::ProviderInfo {
                name: "malformed".to_string(),
                model: "malformed-model".to_string(),
                version: None,
            }
        }
    }

    let config = DefaultGuardConfig {
        judge_fail_open: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, Arc::new(MalformedJsonClient));
    let ctx = make_ctx_with_text("query", "short", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "malformed JSON + fail_closed → Continue with verify: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for malformed JSON + fail_closed, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_judge_malformed_json_fail_open() {
    struct MalformedJsonClient;
    #[async_trait::async_trait]
    impl LlmProvider for MalformedJsonClient {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            let chunks = vec![
                Ok(StreamChunk::Text("not json".to_string())),
                Ok(StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ];
            Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
                chunks,
            ))))
        }
        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                content: "not json".to_string(),
                tool_calls: vec![],
                usage: agent_base::UsageInfo::default(),
                finish_reason: LlmFinishReason::Stop,
                raw: None,
                reasoning_content: None,
                thinking_signature: None,
            })
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }
        fn info(&self) -> agent_base::llm_trait::ProviderInfo {
            agent_base::llm_trait::ProviderInfo {
                name: "malformed".to_string(),
                model: "malformed-model".to_string(),
                version: None,
            }
        }
    }

    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, Arc::new(MalformedJsonClient));
    let ctx = make_ctx_with_text("query", "short", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "malformed JSON + fail_open → Complete (trust model), got: {:?}",
        decision
    );
}

// ── Judge: partial JSON (missing fields) ──

#[tokio::test]
async fn test_judge_partial_json_missing_done() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "reason": "some reason"
        // missing "done" field
    })));
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short", true, true);

    let decision = guard.on_turn(&ctx).await;
    // serde will fail to deserialize because "done" is required
    // fail_closed → Continue
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "missing field + fail_closed → Continue: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for partial JSON + fail_closed, got: {:?}",
            other
        ),
    }
}

// ── call_completion_judge direct tests ──

mod judge_function_tests {
    use super::super::judge::call_completion_judge;
    use super::*;

    /// Helper to cast a concrete Arc to Arc<dyn LlmProvider>.
    fn to_dyn<T: LlmProvider + 'static>(client: Arc<T>) -> Arc<dyn LlmProvider> {
        client
    }

    #[tokio::test]
    async fn test_judge_no_client_fail_open() {
        let result = call_completion_judge(
            None,
            "input",
            "response",
            &["input".to_string()],
            true, // fail_open
            10,
            5,
        )
        .await;

        assert!(result.is_ok());
        let judge = result.unwrap();
        assert!(judge.done, "no client + fail_open → done=true");
    }

    #[tokio::test]
    async fn test_judge_no_client_fail_closed() {
        let result = call_completion_judge(
            None,
            "input",
            "response",
            &["input".to_string()],
            false, // fail_closed
            10,
            5,
        )
        .await;

        assert!(result.is_err(), "no client + fail_closed → error");
    }

    #[tokio::test]
    async fn test_judge_with_valid_response() {
        let client = to_dyn(Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": true,
            "reason": "all tasks complete"
        }))));

        let result = call_completion_judge(
            Some(&client),
            "do something",
            "done",
            &["do something".to_string()],
            false,
            10,
            5,
        )
        .await;

        assert!(result.is_ok());
        let judge = result.unwrap();
        assert!(judge.done);
        assert_eq!(judge.reason, "all tasks complete");
    }

    #[tokio::test]
    async fn test_judge_empty_user_inputs() {
        let client = to_dyn(Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": false,
            "reason": "not enough context"
        }))));

        let result = call_completion_judge(
            Some(&client),
            "single input",
            "response",
            &[], // empty all_user_inputs
            false,
            10,
            5,
        )
        .await;

        assert!(result.is_ok());
        let judge = result.unwrap();
        assert!(!judge.done);
    }

    #[tokio::test]
    async fn test_judge_single_user_input() {
        let client = to_dyn(Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": true,
            "reason": "complete"
        }))));

        let result = call_completion_judge(
            Some(&client),
            "only message",
            "response",
            &["only message".to_string()],
            false,
            10,
            5,
        )
        .await;

        assert!(result.is_ok());
        assert!(result.unwrap().done);
    }

    #[tokio::test]
    async fn test_judge_timeout_fail_open() {
        let client = to_dyn(Arc::new(MockTimeoutClient));

        let result = call_completion_judge(
            Some(&client),
            "input",
            "response",
            &["input".to_string()],
            true, // fail_open
            1,    // 1 second timeout
            5,
        )
        .await;

        assert!(result.is_ok(), "timeout + fail_open → Ok");
        assert!(result.unwrap().done);
    }

    #[tokio::test]
    async fn test_judge_timeout_fail_closed() {
        let client = to_dyn(Arc::new(MockTimeoutClient));

        let result = call_completion_judge(
            Some(&client),
            "input",
            "response",
            &["input".to_string()],
            false, // fail_closed
            1,
            5,
        )
        .await;

        assert!(result.is_err(), "timeout + fail_closed → Err");
    }

    #[tokio::test]
    async fn test_judge_error_fail_open() {
        let client = to_dyn(Arc::new(MockErrorClient));

        // fail_open=true → client errors also handled: trust model → Ok(done: true)
        let result = call_completion_judge(
            Some(&client),
            "input",
            "response",
            &["input".to_string()],
            true,
            10,
            5,
        )
        .await;

        assert!(result.is_ok(), "error + fail_open → Ok (trust model)");
        let judge = result.unwrap();
        assert!(judge.done, "fail_open → done=true");
        assert!(
            judge.reason.contains("trusting model"),
            "reason should mention trusting model: {}",
            judge.reason
        );
    }

    #[tokio::test]
    async fn test_judge_error_fail_closed() {
        let client = to_dyn(Arc::new(MockErrorClient));

        let result = call_completion_judge(
            Some(&client),
            "input",
            "response",
            &["input".to_string()],
            false,
            10,
            5,
        )
        .await;

        assert!(result.is_err(), "error + fail_closed → Err");
    }

    #[tokio::test]
    async fn test_judge_recent_user_count_truncation() {
        let client = to_dyn(Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": true,
            "reason": "complete"
        }))));

        // recent_user_count=2 but 5 inputs → only last 2 used
        let result = call_completion_judge(
            Some(&client),
            "msg5",
            "response",
            &[
                "msg1".to_string(),
                "msg2".to_string(),
                "msg3".to_string(),
                "msg4".to_string(),
                "msg5".to_string(),
            ],
            false,
            10,
            2, // recent_user_count
        )
        .await;

        assert!(result.is_ok());
    }
}

// ── GuardDecision pattern matching ──

#[test]
fn test_guard_decision_continue_with_nudge() {
    let d = GuardDecision::Continue {
        nudge: Some("test nudge".to_string()),
    };
    match &d {
        GuardDecision::Continue { nudge } => {
            assert_eq!(nudge.as_deref(), Some("test nudge"));
        }
        _ => panic!("expected Continue"),
    }
}

#[test]
fn test_guard_decision_continue_without_nudge() {
    let d = GuardDecision::Continue { nudge: None };
    match &d {
        GuardDecision::Continue { nudge } => {
            assert!(nudge.is_none());
        }
        _ => panic!("expected Continue"),
    }
}

#[test]
fn test_guard_decision_complete() {
    let d = GuardDecision::Complete;
    assert!(matches!(d, GuardDecision::Complete));
}

#[test]
fn test_guard_decision_fail() {
    let d = GuardDecision::Fail {
        error: "something went wrong".to_string(),
    };
    match &d {
        GuardDecision::Fail { error } => {
            assert_eq!(error, "something went wrong");
        }
        _ => panic!("expected Fail"),
    }
}

// ── NoopGuard tests ──
// Note: NoopGuard returns Fail for reasoning_only/empty_response (safety),
// and Complete for text_only and normal scenarios.

#[tokio::test]
async fn test_noop_guard_normal_complete() {
    let guard = NoopGuard;
    let ctx = make_ctx(0, 0, false, false, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "NoopGuard normal → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_noop_guard_text_only_complete() {
    let guard = NoopGuard;
    let ctx = make_ctx(0, 0, false, false, false, true);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "NoopGuard text_only → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_noop_guard_reasoning_only_fails() {
    let guard = NoopGuard;
    let ctx = make_ctx(0, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "NoopGuard reasoning_only → Fail (safety), got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_noop_guard_empty_response_fails() {
    let guard = NoopGuard;
    let ctx = make_ctx(0, 0, false, false, true, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "NoopGuard empty_response → Fail (safety), got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_noop_guard_all_flags_reasoning_priority() {
    let guard = NoopGuard;
    let ctx = make_ctx(3, 3, true, true, true, true);
    let decision = guard.on_turn(&ctx).await;
    // reasoning_only || empty_response → Fail
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "NoopGuard all flags → Fail (reasoning/empty), got: {:?}",
        decision
    );
}

// ── Multiple scene flags set simultaneously ──

#[tokio::test]
async fn test_reasoning_only_takes_priority_over_empty() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    // Both reasoning_only and empty_response flags set
    let ctx = make_ctx(1, 1, false, true, true, false);

    let decision = guard.on_turn(&ctx).await;
    // on_turn checks is_reasoning_only first → should handle as reasoning_only
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().reasoning_only_nudge);
        }
        other => panic!(
            "reasoning_only takes priority, expected Continue, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_reasoning_only_takes_priority_over_text_only() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(1, 0, false, true, false, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().reasoning_only_nudge);
        }
        other => panic!(
            "reasoning_only takes priority, expected Continue, got: {:?}",
            other
        ),
    }
}

// ── Short response + judge: non-short response fallback path ──

#[tokio::test]
async fn test_text_only_non_short_with_tools_judge_not_done() {
    // Response that's not short (> max_output=64) but < skip_threshold=256
    let medium_output = "a".repeat(100);
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "missing details"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("query", &medium_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("incomplete") && msg.contains("missing details"),
                "non-short + judge not done → Continue with reason: {}",
                msg
            );
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_text_only_non_short_with_tools_judge_done() {
    let medium_output = "a".repeat(100);
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": true,
        "reason": "task complete"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("query", &medium_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "non-short + judge done → Complete, got: {:?}",
        decision
    );
}

// ── Short response + skip_judge_when_input_too_large for short path ──

#[tokio::test]
async fn test_short_response_skip_judge_when_input_too_large() {
    let long_input = "a".repeat(10_001);
    let short_output = "42";
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "would be skipped"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text(&long_input, short_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "short response + input > 10k → skip judge → Complete, got: {:?}",
        decision
    );
}

// ── DisableThinking strategy tests ──────────────────────────────────────

#[tokio::test]
async fn test_disable_thinking_strategy_below_threshold() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        reasoning_only_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Below threshold → Continue with nudge
    let ctx = make_ctx(2, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { .. }),
        "below threshold → Continue, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_disable_thinking_strategy_at_threshold() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        reasoning_only_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // At threshold → DisableThinking
    let ctx = make_ctx(3, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::DisableThinking { .. }),
        "at threshold → DisableThinking, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_disable_thinking_strategy_already_disabled() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        reasoning_only_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Thinking already disabled but still reasoning-only → Fail
    let mut ctx = make_ctx(3, 0, false, true, false, false);
    ctx.thinking_disabled = true;
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "thinking disabled + reasoning-only → Fail, got: {:?}",
        decision
    );
}

// ── on_tool_call tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_on_tool_call_restores_thinking() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Thinking disabled + original enabled + tool call → RestoreThinking
    let mut ctx = make_ctx(0, 0, true, false, false, false);
    ctx.thinking_disabled = true;
    ctx.original_thinking_enabled = true;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::RestoreThinking),
        "thinking disabled + tool call → RestoreThinking, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_on_tool_call_no_restore_when_original_disabled() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Thinking disabled + original disabled + tool call → Complete
    let mut ctx = make_ctx(0, 0, true, false, false, false);
    ctx.thinking_disabled = true;
    ctx.original_thinking_enabled = false;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "original disabled + tool call → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_on_tool_call_no_restore_when_thinking_enabled() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Thinking enabled + tool call → Complete
    let mut ctx = make_ctx(0, 0, true, false, false, false);
    ctx.thinking_disabled = false;
    ctx.original_thinking_enabled = true;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "thinking enabled + tool call → Complete, got: {:?}",
        decision
    );
}

// ── Fail strategy tests (default behavior) ──────────────────────────────

#[tokio::test]
async fn test_fail_strategy_at_threshold() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::Fail,
        reasoning_only_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // At threshold → Fail
    let ctx = make_ctx(3, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "at threshold → Fail, got: {:?}",
        decision
    );
}

// ── Empty response tests with DisableThinking ───────────────────────────

#[tokio::test]
async fn test_empty_response_with_disable_thinking_strategy() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        empty_response_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Empty response at threshold → Fail (even with DisableThinking strategy)
    let ctx = make_ctx(0, 3, false, false, true, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "empty response at threshold → Fail, got: {:?}",
        decision
    );
}
