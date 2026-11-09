// ── Notice handle (Batch E, PR3) ────────────────────────────────────────────

use super::*;
use agent_base::types::NoticeKind;

/// Garbage judge client — every judge call fails to parse (issue #31
/// degradation scenario). No retry happens inside the guard: the judge
/// helper itself retries once, then fail-opens.
struct GarbageJudgeClient;
#[async_trait::async_trait]
impl LlmProvider for GarbageJudgeClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chunks = vec![
            Ok(StreamChunk::Text("total garbage !!!".to_string())),
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
            content: "total garbage !!!".to_string(),
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
            name: "garbage-judge".to_string(),
            model: "garbage".to_string(),
            version: None,
        }
    }
}

/// Exercises the **non-short-response** judge call site (default.rs's
/// `run_has_actions && use_llm_judge` branch): the input is 58 chars,
/// under `short_response_min_input` (128), so short-response detection
/// does not fire. See `judge_fail_open_short_response_path_emits_notice`
/// for the other call site.
fn judge_fail_open_ctx() -> GuardCtx {
    make_ctx_with_text(
        "a user input that is not long enough for short-response detection",
        "done!",
        true,
        true,
    )
}

/// The short-response judge call site (input > 128 chars, output < 64,
/// run had tool calls) must also emit the degradation notice — both
/// `call_completion_judge` sites take the handle.
fn judge_fail_open_short_response_ctx() -> GuardCtx {
    let long_input = "please analyze the repository structure in exhaustive detail ".repeat(3); // 177 chars — over short_response_min_input (128)
    make_ctx_with_text(&long_input, "done!", true, true)
}

fn fail_open_guard_with_judge() -> DefaultGuard {
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    DefaultGuard::with_llm_client(config, Arc::new(GarbageJudgeClient))
}

/// Design §9: degradation path with a handle — guard emits
/// Warning/"guard"/"guard judge unparsed — treating as complete".
#[tokio::test]
async fn judge_fail_open_emits_warning_notice_via_handle() {
    let (handle, mut rx) = agent_base::NoticeHandle::new_channel();
    let guard = fail_open_guard_with_judge();
    guard.set_notice(handle);

    let ctx = judge_fail_open_ctx();
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "fail-open → Complete, got: {:?}",
        decision
    );

    let notice = rx.try_recv().expect("notice must be sent on degradation");
    assert_eq!(notice.kind, NoticeKind::Warning);
    assert_eq!(notice.source, "guard");
    assert_eq!(notice.text, "guard judge unparsed — treating as complete");
}

/// The short-response judge call site (default.rs's `is_short_response`
/// + `run_has_actions && use_llm_judge` branch) must emit the same
/// degradation notice — both `call_completion_judge` sites plumb the handle.
#[tokio::test]
async fn judge_fail_open_short_response_path_emits_notice() {
    let (handle, mut rx) = agent_base::NoticeHandle::new_channel();
    let guard = fail_open_guard_with_judge();
    guard.set_notice(handle);

    let ctx = judge_fail_open_short_response_ctx();
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "fail-open → Complete, got: {:?}",
        decision
    );

    let notice = rx
        .try_recv()
        .expect("notice must be sent on the short-response degradation");
    assert_eq!(notice.kind, NoticeKind::Warning);
    assert_eq!(notice.source, "guard");
    assert_eq!(notice.text, "guard judge unparsed — treating as complete");
}

/// Design §9: no handle — behavior byte-identical to today (no send, no
/// panic, same decision).
#[tokio::test]
async fn judge_fail_open_without_handle_behaves_as_before() {
    let guard = fail_open_guard_with_judge();
    // No set_notice call — Option stays None.

    let ctx = judge_fail_open_ctx();
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "fail-open → Complete regardless of handle presence, got: {:?}",
        decision
    );
}

/// Success path must NOT emit any notice — only the degradation path does.
#[tokio::test]
async fn judge_success_emits_no_notice() {
    let (handle, mut rx) = agent_base::NoticeHandle::new_channel();
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(
        config,
        Arc::new(MockJudgeClient::new(serde_json::json!({
            "done": true,
            "reason": "answered"
        }))),
    );
    guard.set_notice(handle);

    let ctx = judge_fail_open_ctx();
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Complete));
    assert!(rx.try_recv().is_err(), "no notice on the success path");
}

/// Guard without any judge involvement (reasoning-only under threshold →
/// Continue) must not emit notices — the notice is specific to judge
/// fail-open degradation.
#[tokio::test]
async fn non_judge_paths_emit_no_notice() {
    let (handle, mut rx) = agent_base::NoticeHandle::new_channel();
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    guard.set_notice(handle);

    let ctx = make_ctx(0, 0, false, false, true, false); // empty response, under threshold
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Continue { .. }));
    assert!(rx.try_recv().is_err(), "no notice on non-judge paths");
}

/// A transport failure (client error) is a different mechanism than an
/// unparseable response — the notice must say "call failed", not lie
/// about parsing.
#[tokio::test]
async fn judge_transport_failure_emits_call_failed_notice() {
    let (handle, mut rx) = agent_base::NoticeHandle::new_channel();
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, Arc::new(MockErrorClient));
    guard.set_notice(handle);

    let ctx = judge_fail_open_ctx();
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "fail-open → Complete, got: {:?}",
        decision
    );

    let notice = rx
        .try_recv()
        .expect("notice must be sent on transport degradation");
    assert_eq!(notice.kind, NoticeKind::Warning);
    assert_eq!(notice.source, "guard");
    assert_eq!(
        notice.text,
        "guard judge call failed — treating as complete"
    );
}

/// A timeout is a transport failure too — same "call failed" notice,
/// never "unparsed".
#[tokio::test]
async fn judge_timeout_emits_call_failed_notice() {
    let (handle, mut rx) = agent_base::NoticeHandle::new_channel();
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        judge_timeout_secs: 1,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, Arc::new(MockTimeoutClient::new()));
    guard.set_notice(handle);

    let ctx = judge_fail_open_ctx();
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "fail-open → Complete, got: {:?}",
        decision
    );

    let notice = rx
        .try_recv()
        .expect("notice must be sent on timeout degradation");
    assert_eq!(notice.kind, NoticeKind::Warning);
    assert_eq!(notice.source, "guard");
    assert_eq!(
        notice.text,
        "guard judge call failed — treating as complete"
    );
}

/// No LLM client + fail-open degrades just as silently as a failed call
/// — the notice must name the missing judge ("unavailable"), not claim
/// an unparseable response.
#[tokio::test]
async fn no_client_fail_open_emits_unavailable_notice() {
    let (handle, mut rx) = agent_base::NoticeHandle::new_channel();
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config); // no client — the unavailable path
    guard.set_notice(handle);

    let ctx = judge_fail_open_ctx();
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "fail-open → Complete, got: {:?}",
        decision
    );

    let notice = rx
        .try_recv()
        .expect("no-client fail-open must also be visible");
    assert_eq!(notice.kind, NoticeKind::Warning);
    assert_eq!(notice.source, "guard");
    assert_eq!(
        notice.text,
        "guard judge unavailable — treating as complete"
    );
}
