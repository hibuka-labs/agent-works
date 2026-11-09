// ── call_completion_judge direct tests ──

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
        None,
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
        None,
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
        None,
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
        None,
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
        None,
    )
    .await;

    assert!(result.is_ok());
    assert!(result.unwrap().done);
}

#[tokio::test]
async fn test_judge_timeout_fail_open() {
    let client = to_dyn(Arc::new(MockTimeoutClient::new()));

    let result = call_completion_judge(
        Some(&client),
        "input",
        "response",
        &["input".to_string()],
        true, // fail_open
        1,    // 1 second timeout
        5,
        None,
    )
    .await;

    assert!(result.is_ok(), "timeout + fail_open → Ok");
    assert!(result.unwrap().done);
}

#[tokio::test]
async fn test_judge_timeout_fail_closed() {
    let client = to_dyn(Arc::new(MockTimeoutClient::new()));

    let result = call_completion_judge(
        Some(&client),
        "input",
        "response",
        &["input".to_string()],
        false, // fail_closed
        1,
        5,
        None,
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
        None,
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
        None,
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
        None,
    )
    .await;

    assert!(result.is_ok());
}

// ── Parse tolerance (issue #31): fences / prose around the JSON object ──

#[tokio::test]
async fn test_judge_fenced_json_parses() {
    let raw = "```json\n{\"done\": true, \"reason\": \"complete\"}\n```";
    let client = to_dyn(Arc::new(MockJudgeClient::raw(raw)));

    // fail_closed: a parse failure surfaces as Err, so Ok proves it parsed.
    let result = call_completion_judge(
        Some(&client),
        "do something",
        "done",
        &["do something".to_string()],
        false,
        10,
        5,
        None,
    )
    .await;

    let judge = result.expect("fenced JSON should parse (issue #31)");
    assert!(judge.done);
    assert_eq!(judge.reason, "complete");
}

#[tokio::test]
async fn test_judge_prose_wrapped_json_parses() {
    let raw = "Sure, here is my assessment: {\"done\": false, \"reason\": \"still incomplete\"} Let me know if you need more.";
    let client = to_dyn(Arc::new(MockJudgeClient::raw(raw)));

    let result = call_completion_judge(
        Some(&client),
        "do something",
        "response",
        &["do something".to_string()],
        false,
        10,
        5,
        None,
    )
    .await;

    let judge = result.expect("prose-wrapped JSON should parse (issue #31)");
    assert!(!judge.done);
    assert_eq!(judge.reason, "still incomplete");
}

#[tokio::test]
async fn test_judge_json_with_braces_inside_string_parses() {
    // Braces and escaped quotes inside the value must not break the
    // balanced-object scan.
    let raw = "Result: {\"done\": true, \"reason\": \"used {braces} and \\\"quotes\\\" inside\"}";
    let client = to_dyn(Arc::new(MockJudgeClient::raw(raw)));

    let result = call_completion_judge(
        Some(&client),
        "do something",
        "response",
        &["do something".to_string()],
        false,
        10,
        5,
        None,
    )
    .await;

    let judge = result.expect("JSON with braces inside strings should parse");
    assert!(judge.done);
    assert_eq!(judge.reason, "used {braces} and \"quotes\" inside");
}

// ── Strict single retry on parse failure (issue #31) ──

#[tokio::test]
async fn test_judge_garbage_retries_once_then_fails_open() {
    let client = Arc::new(SequenceJudgeClient::new(&[
        "total garbage",
        "still garbage",
    ]));

    let result = call_completion_judge(
        Some(&(Arc::clone(&client) as Arc<dyn LlmProvider>)),
        "do something",
        "response",
        &["do something".to_string()],
        true, // fail_open
        10,
        5,
        None,
    )
    .await;

    // Exactly one retry: two calls total, never a loop.
    assert_eq!(
        client.call_count(),
        2,
        "parse failure must retry exactly once"
    );
    // Honest fail-open: Ok(done=true) but reason carries the raw text.
    let judge = result.expect("fail_open must convert failure to done=true");
    assert!(judge.done);
    assert!(
        judge.reason.contains("still garbage"),
        "reason must carry the raw judge text (WARN detail): {}",
        judge.reason
    );
}

#[tokio::test]
async fn test_judge_garbage_then_valid_retry_succeeds() {
    let valid = r#"{"done": true, "reason": "complete"}"#;
    let client = Arc::new(SequenceJudgeClient::new(&["I cannot answer that", valid]));

    let result = call_completion_judge(
        Some(&(Arc::clone(&client) as Arc<dyn LlmProvider>)),
        "do something",
        "response",
        &["do something".to_string()],
        false,
        10,
        5,
        None,
    )
    .await;

    assert_eq!(
        client.call_count(),
        2,
        "garbage → valid retry uses both calls"
    );
    let judge = result.expect("retry with valid JSON should succeed");
    assert!(judge.done);
    assert_eq!(judge.reason, "complete");
}

#[tokio::test]
async fn test_judge_timeout_does_not_retry() {
    let client = Arc::new(MockTimeoutClient::new());

    let result = call_completion_judge(
        Some(&(Arc::clone(&client) as Arc<dyn LlmProvider>)),
        "input",
        "response",
        &["input".to_string()],
        false, // fail_closed: timeout surfaces as Err
        1,     // 1s timeout
        5,
        None,
    )
    .await;

    assert!(result.is_err(), "timeout + fail_closed → Err");
    assert_eq!(
        client.calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "transport failures must not be retried"
    );
}

#[tokio::test]
async fn test_judge_garbage_fail_closed_retries_then_errs() {
    let client = Arc::new(SequenceJudgeClient::new(&["garbage one", "garbage two"]));

    let result = call_completion_judge(
        Some(&(Arc::clone(&client) as Arc<dyn LlmProvider>)),
        "do something",
        "response",
        &["do something".to_string()],
        false, // fail_closed
        10,
        5,
        None,
    )
    .await;

    assert_eq!(
        client.call_count(),
        2,
        "fail_closed parse failure also retries once"
    );
    let err = result.expect_err("fail_closed + unparseable → Err");
    assert!(
        err.contains("garbage two"),
        "error must carry the raw judge text (WARN detail): {}",
        err
    );
}

#[tokio::test]
async fn test_truncate_respects_char_boundary() {
    // Chinese chars are multi-byte; a byte-slice cut would panic.
    let raw = "解析失败原文".repeat(200);
    let truncated = super::super::judge::truncate_for_log(&raw, 500);
    assert!(truncated.chars().count() <= 500);
    // Short input passes through untouched.
    assert_eq!(super::super::judge::truncate_for_log("short", 500), "short");
}

// ── Judge prompt contract + live client resolution (20261107_27dc727a) ──
//
// The incident: the completion judge fabricated ground truth ("the session's
// configured model is MiniMax-M3.1-Flash" — its own stale boot identity) and
// misquoted the answer it judged ("mimo-v2.5-pro" for "mimo-v2.6-pro"), then
// injected a false "answer is incomplete" correction. These tests pin both
// halves of the fix: the prompt is completeness-only with a scoped ground
// truth, and the judge resolves its client through the engine's live provider
// handle instead of its build-time clone.

use super::super::judge::judge_system_prompt;
use agent_base::ProviderHandle;
use agent_base::types::ChatMessage;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A mock provider that records the request of its last `chat()` call and
/// answers a fixed verdict — for prompt-contract and client-resolution tests.
struct RecordingClient {
    model: &'static str,
    verdict: &'static str,
    seen: std::sync::Mutex<Option<ChatRequest>>,
    calls: AtomicUsize,
}

impl RecordingClient {
    fn new(model: &'static str, verdict: &'static str) -> Arc<Self> {
        Arc::new(Self {
            model,
            verdict,
            seen: std::sync::Mutex::new(None),
            calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait::async_trait]
impl LlmProvider for RecordingClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chunks = vec![
            Ok(StreamChunk::Text(self.verdict.to_string())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ];
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
            chunks,
        ))))
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.seen.lock().unwrap() = Some(request);
        Ok(ChatResponse {
            content: self.verdict.to_string(),
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
            name: "recording".to_string(),
            model: self.model.to_string(),
            version: None,
        }
    }
}

#[test]
fn judge_prompt_is_completeness_only_with_scoped_ground_truth() {
    let prompt = judge_system_prompt("mimo-v2.6-pro");
    // The three contract elements: completion-only mandate, no-fabrication
    // rule (no invented facts, no paraphrased values, no "unverifiable"
    // penalties), and the single sanctioned fact — the session's model.
    for needle in [
        // The side-call classifier anchor: phimint's test and PTY harnesses
        // recognize judge requests by the phrase "completion judge" in the
        // prompt's first line. Keep this phrase stable (hyphenation of the
        // surrounding words is fine) — a silent rewording desyncs their LLM
        // scripts (20261107_27dc727a).
        "completion judge",
        "Judge COMPLETENESS only",
        "Do NOT fact-check",
        "Never state, assume, or introduce facts",
        "quote names only exactly as they appear",
        "Do not penalize a response as unverifiable",
        "A concise answer to the question asked is complete",
        "Ground truth: the model serving this session is `mimo-v2.6-pro`",
        "Reply with JSON",
    ] {
        assert!(
            prompt.contains(needle),
            "judge prompt missing {needle:?}:\n{prompt}"
        );
    }
    // No model name → no ground-truth line at all (never an empty fact).
    assert!(
        !judge_system_prompt("").contains("Ground truth"),
        "empty model must not produce a ground-truth line"
    );
}

#[tokio::test]
async fn judge_request_ground_truth_is_the_clients_model() {
    let client = RecordingClient::new("mimo-v2.6-pro", r#"{"done": true, "reason": "ok"}"#);
    let dyn_client: Arc<dyn LlmProvider> = client.clone();
    call_completion_judge(
        Some(&dyn_client),
        "现在呢",
        "现在是 mimo-v2.6-pro。",
        &["现在呢".to_string()],
        true,
        10,
        5,
        None,
    )
    .await
    .expect("judge call succeeds");

    let seen = client.seen.lock().unwrap();
    let request = seen.as_ref().expect("judge call recorded");
    let system = request
        .messages
        .iter()
        .find_map(|m| match m {
            ChatMessage::System { content, .. } => Some(content.clone()),
            _ => None,
        })
        .expect("judge request carries a system message");
    assert!(
        system.contains("the model serving this session is `mimo-v2.6-pro`"),
        "ground truth must be the client's live model name:\n{system}"
    );
}

#[tokio::test]
async fn guard_judges_on_the_provider_source_not_the_build_client() {
    let frozen = RecordingClient::new("boot-model", r#"{"done": true, "reason": "frozen"}"#);
    let live = RecordingClient::new("live-model", r#"{"done": false, "reason": "from live"}"#);
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), frozen.clone());
    guard.set_provider_source(ProviderHandle::new(live.clone()));
    let ctx = make_ctx_with_text("现在呢", "现在是 mimo-v2.6-pro。", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            assert!(
                nudge.as_ref().unwrap().contains("from live"),
                "nudge must carry the live judge's reason: {nudge:?}"
            );
        }
        other => panic!("live judge said done:false → Continue, got {other:?}"),
    }
    assert_eq!(
        live.calls.load(Ordering::SeqCst),
        1,
        "the judge must run on the provider source"
    );
    assert_eq!(
        frozen.calls.load(Ordering::SeqCst),
        0,
        "the build-time client must not be used once a source is set"
    );
}
