use super::*;

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
