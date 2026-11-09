use super::*;
use agent_base::ToolCallMessage;
use agent_base::llm_trait::response::FinishReason;
use agent_base::llm_trait::types::UsageInfo;
use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};

// ── Test helpers ──────────────────────────────────────────────────────

/// Minimal mock that returns a fixed string.
struct MockClient(&'static str);

#[async_trait::async_trait]
impl LlmProvider for MockClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let response = self.0.to_string();
        Ok(ChatStream::new(Box::pin(futures_util::stream::once(
            async move { Ok(agent_base::StreamChunk::Text(response)) },
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            content: self.0.to_string(),
            tool_calls: vec![],
            usage: UsageInfo::default(),
            finish_reason: FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

/// Mock that counts invocations.
struct CountingClient {
    response: &'static str,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl LlmProvider for CountingClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let response = self.response.to_string();
        Ok(ChatStream::new(Box::pin(futures_util::stream::once(
            async move { Ok(agent_base::StreamChunk::Text(response)) },
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(ChatResponse {
            content: self.response.to_string(),
            tool_calls: vec![],
            usage: UsageInfo::default(),
            finish_reason: FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

/// Mock that always returns an error.
struct FailingClient;

#[async_trait::async_trait]
impl LlmProvider for FailingClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Err(LlmError::llm("summarisation failed"))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Err(LlmError::llm("summarisation failed"))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

fn make_messages(count: usize) -> Vec<ChatMessage> {
    let mut msgs = vec![ChatMessage::system("You are a test agent.")];
    for i in 0..count {
        msgs.push(ChatMessage::user(format!("question {i}")));
        msgs.push(ChatMessage::assistant(format!(
            "answer {i} with some extra content to make it longer"
        )));
    }
    msgs
}

// ── safe_cut_index ────────────────────────────────────────────────────

#[test]
fn test_safe_cut_never_splits_tool_pair() {
    let msgs = vec![
        ChatMessage::user("hi"),
        ChatMessage::assistant(""),
        ChatMessage::assistant_tool_call("call_1", "bash", "{}"),
        ChatMessage::tool("call_1", "result"),
        ChatMessage::user("next"),
    ];
    // Cut at 3 (between assistant{tool_call} and tool result) → unsafe.
    // Should walk back past the assistant{tool_call} message.
    let cut = super::safe_cut_index(&msgs, 0, 3);
    assert!(
        cut <= 2,
        "cut should walk back before assistant{{tool_call}}"
    );
    assert!(
        !matches!(
            &msgs[cut - 1],
            ChatMessage::Assistant {
                tool_calls: Some(_),
                ..
            }
        ),
        "left boundary must not be assistant with pending tool calls"
    );
}

#[test]
fn test_safe_cut_prefers_given_boundary_when_safe() {
    let msgs = vec![
        ChatMessage::user("hi"),
        ChatMessage::assistant("hello"),
        ChatMessage::user("next"),
        ChatMessage::assistant("world"),
    ];
    // Cut at 2 is safe: left=Assistant(no tool_calls), right=User.
    assert_eq!(super::safe_cut_index(&msgs, 0, 2), 2);
}

#[test]
fn test_safe_cut_walks_back_from_orphan_tool() {
    let msgs = vec![
        ChatMessage::user("a"),
        ChatMessage::assistant("b"),
        ChatMessage::tool("c1", "r1"),
        ChatMessage::user("d"),
    ];
    // Cut=1 lands on assistant (no tool_calls), right=user → safe.
    assert_eq!(super::safe_cut_index(&msgs, 0, 1), 1);
}

// ── estimate_message_tokens ───────────────────────────────────────────

#[test]
fn test_estimate_tokens_cjk_and_tool() {
    let msg = ChatMessage::tool("t1", "你好世界");
    let tokens = super::estimate_message_tokens(&msg);
    // CJK: ~4 chars / 1.5 ≈ 3; id "t1": 2 chars / 4 = 1
    assert!((2..=10).contains(&tokens));
}

#[test]
fn test_estimate_tokens_user_with_images() {
    let msg = ChatMessage::user_with_images(
        "describe this",
        vec![agent_base::ImageAttachment::Url {
            url: "http://example.com/img.png".into(),
            detail: Some(agent_base::ImageDetail::Auto),
        }],
    );
    let tokens = super::estimate_message_tokens(&msg);
    // Should include the 85-token per-image overhead.
    assert!(tokens >= 85);
}

// ── serialize_block ───────────────────────────────────────────────────

#[test]
fn test_serialize_block_preserves_tool_calls() {
    let msgs = [
        ChatMessage::assistant(""),
        ChatMessage::tool("c1", "result"),
    ];
    let msg_refs: Vec<&ChatMessage> = msgs.iter().collect();
    let out = super::serialize_block(&msg_refs, 2000);
    assert!(out.contains("[tool:c1]"), "got: {out}");
    assert!(out.contains("result"), "got: {out}");
}

#[test]
fn test_serialize_block_truncates_oversized_fields() {
    let long_arg = "x".repeat(500);
    let msgs = [ChatMessage::Assistant {
        content: None,
        reasoning_content: None,
        tool_calls: Some(vec![ToolCallMessage {
            id: "t1".into(),
            name: "bash".into(),
            arguments: long_arg,
        }]),
        thinking_signature: None,
    }];
    let msg_refs: Vec<&ChatMessage> = msgs.iter().collect();
    let out = super::serialize_block(&msg_refs, 2000);
    // Should contain "bash(" and "…", but NOT all 500 chars.
    assert!(out.contains("bash("));
    assert!(out.contains("…"));
    assert!(out.len() < 400);
}

// ── compact ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_compact_noop_when_disabled() {
    let config = CompressionConfig::default().with_enabled(false);
    let client = std::sync::Arc::new(MockClient("summary"));
    let compactor = ContextCompactor::new(client, config);
    let msgs = make_messages(100);
    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_compact_noop_when_below_threshold() {
    let config = CompressionConfig::default().with_trigger_tokens(999_999);
    let client = std::sync::Arc::new(MockClient("summary"));
    let compactor = ContextCompactor::new(client, config);
    let msgs = make_messages(10);
    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_compact_produces_valid_output() {
    let config = CompressionConfig::default()
        .with_trigger_tokens(1) // Always trigger.
        .with_keep_recent_messages(4);
    let client = std::sync::Arc::new(MockClient("test summary text"));
    let compactor = ContextCompactor::new(client, config);
    let msgs = make_messages(20);
    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap()
        .unwrap();

    // First message should be system prompt.
    assert!(matches!(&result[0], ChatMessage::System { .. }));

    // Should contain a summary message and then recent messages.
    let has_summary = result.iter().any(|m| match m {
        ChatMessage::User { content, .. } => content.starts_with(SUMMARY_PREFIX),
        _ => false,
    });
    assert!(has_summary, "expected summary message in output");

    // Recent messages should be present.
    let last = result.last().unwrap();
    assert!(matches!(
        last,
        ChatMessage::User { .. } | ChatMessage::Assistant { .. }
    ));

    // No orphan Tool messages: every Tool must have index > 0.
    for (i, msg) in result.iter().enumerate() {
        assert!(
            !matches!(msg, ChatMessage::Tool { .. }) || i > 0,
            "orphan Tool at index 0"
        );
    }
}

// ── compact with failing client ───────────────────────────────────────

#[tokio::test]
async fn test_compact_fallback_on_summarisation_failure() {
    let config = CompressionConfig::default()
        .with_trigger_tokens(1)
        .with_keep_recent_messages(4);
    let client = std::sync::Arc::new(FailingClient);
    let compactor = ContextCompactor::new(client, config);
    let msgs = make_messages(20);
    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap()
        .unwrap();

    // Fallback: summary is empty, so we get system + recent only.
    assert!(matches!(&result[0], ChatMessage::System { .. }));
    let has_summary = result.iter().any(|m| match m {
        ChatMessage::User { content, .. } => content.starts_with(SUMMARY_PREFIX),
        _ => false,
    });
    assert!(!has_summary, "should have no summary on failure");
}

// ── compact cache ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_compact_caches_summary() {
    let config = CompressionConfig::default()
        .with_trigger_tokens(1)
        .with_keep_recent_messages(4);
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let client = std::sync::Arc::new(CountingClient {
        response: "cached summary",
        calls: calls.clone(),
    });
    let compactor = ContextCompactor::new(client, config);
    let msgs = make_messages(20);

    // First call — should invoke LLM once.
    let _ = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Second call with same old block — should hit cache (no new LLM call).
    let _ = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Third call after appending messages (old block grows) — should re-summarize.
    let mut grown = msgs.clone();
    grown.push(ChatMessage::user("new question"));
    grown.push(ChatMessage::assistant("new answer"));
    let _ = compactor
        .compact(1, &grown, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

// ── compact cache length guard ────────────────────────────────────────

#[tokio::test]
async fn test_compact_cache_length_guard_re_summarises_on_growth() {
    // Construct old block > 4096 chars (CACHE_PREFIX_CHARS) so the prefix
    // hash stays stable when we grow the tail.  The length guard
    // (cl == transcript.len()) should detect the growth and re-summarise.
    let config = CompressionConfig::default()
        .with_trigger_tokens(1)
        .with_keep_recent_messages(4)
        .with_max_transcript_chars(60_000); // High cap so we don't hit the freeze guard.
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let client = std::sync::Arc::new(CountingClient {
        response: "summary",
        calls: calls.clone(),
    });
    let compactor = ContextCompactor::new(client, config);

    // Each pair "question NNNN\nanswer NNNN" serialises to ~30 chars.
    // 200 pairs → ~6000 chars transcript (> 4096 prefix).
    let mut msgs = vec![ChatMessage::system("sys")];
    for i in 0..200 {
        msgs.push(ChatMessage::user(format!("question {i:04}")));
        msgs.push(ChatMessage::assistant(format!("answer {i:04}")));
    }

    // First call — cache miss, LLM called once.
    let _ = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Second call, identical — cache hit (same prefix + same length).
    let _ = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    // Grow the tail: prefix hash unchanged (first 4096 chars identical),
    // but transcript length increased → length guard should reject cache.
    msgs.push(ChatMessage::user("extra question"));
    msgs.push(ChatMessage::assistant("extra answer"));
    let _ = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "length guard should have triggered re-summarisation"
    );
}

// ── compact orphan Tool assertion ─────────────────────────────────────

#[tokio::test]
async fn test_compact_no_orphan_tool_messages() {
    let config = CompressionConfig::default()
        .with_trigger_tokens(1)
        .with_keep_recent_messages(4);
    let client = std::sync::Arc::new(MockClient("summary"));
    let compactor = ContextCompactor::new(client, config);

    // Build messages with tool_calls + tool results in the old block.
    let mut msgs = vec![ChatMessage::system("sys")];
    for i in 0..10 {
        msgs.push(ChatMessage::user(format!("q{i}")));
        msgs.push(ChatMessage::assistant_tool_call(
            format!("tc{i}"),
            "bash",
            format!("{{\"cmd\":\"echo {i} with some extra arguments to make it longer\"}}"),
        ));
        msgs.push(ChatMessage::tool(
            format!("tc{i}"),
            format!("result {i} with some extra output content"),
        ));
    }

    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap()
        .unwrap();

    // No orphan Tool: every Tool must be preceded by an Assistant whose
    // tool_calls reference the same tool_call_id.
    for (i, msg) in result.iter().enumerate() {
        if let ChatMessage::Tool { tool_call_id, .. } = msg {
            assert!(i > 0, "Tool message at index 0 has no preceding assistant");
            let prev = &result[i - 1];
            match prev {
                ChatMessage::Assistant {
                    tool_calls: Some(calls),
                    ..
                } => {
                    let ids: Vec<&str> = calls.iter().map(|c| c.id.as_str()).collect();
                    assert!(
                        ids.contains(&tool_call_id.as_str()),
                        "Tool {tool_call_id} not referenced by preceding assistant: {ids:?}"
                    );
                }
                other => panic!(
                    "Tool {tool_call_id} at index {i} not preceded by \
                         Assistant{{tool_calls}}, got: {other:?}"
                ),
            }
        }
    }
}

// ── compact boundary conditions ─────────────────────────────────────

#[tokio::test]
async fn test_compact_noop_empty_messages() {
    let config = CompressionConfig::default().with_trigger_tokens(1);
    let client = std::sync::Arc::new(MockClient("summary"));
    let compactor = ContextCompactor::new(client, config);
    let result = compactor
        .compact(1, &[], CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_compact_noop_system_only() {
    let config = CompressionConfig::default().with_trigger_tokens(1);
    let client = std::sync::Arc::new(MockClient("summary"));
    let compactor = ContextCompactor::new(client, config);
    let msgs = vec![ChatMessage::system("You are helpful.")];
    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_compact_noop_conversation_equals_keep() {
    // conversation.len() == keep_recent_messages → too few to split.
    let config = CompressionConfig::default()
        .with_trigger_tokens(1)
        .with_keep_recent_messages(100);
    let client = std::sync::Arc::new(MockClient("summary"));
    let compactor = ContextCompactor::new(client, config);
    let msgs = make_messages(3); // 1 system + 6 conversation = 7
    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_compact_filters_old_summary_in_old_block() {
    let config = CompressionConfig::default()
        .with_trigger_tokens(1)
        .with_keep_recent_messages(4);
    let client = std::sync::Arc::new(MockClient("new summary"));
    let compactor = ContextCompactor::new(client, config);

    // Build messages with an old summary in the old block.
    let mut msgs = vec![ChatMessage::system("sys")];
    // Old summary (should be filtered).
    msgs.push(ChatMessage::user(format!(
        "{SUMMARY_PREFIX}\nprevious summary text"
    )));
    for i in 0..10 {
        msgs.push(ChatMessage::user(format!("question {i}")));
        msgs.push(ChatMessage::assistant(format!(
            "answer {i} with some extra content to make the old block large enough"
        )));
    }

    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap()
        .unwrap();

    // The old summary should NOT appear in the output.
    let summary_count = result
        .iter()
        .filter(|m| match m {
            ChatMessage::User { content, .. } => content.starts_with(SUMMARY_PREFIX),
            _ => false,
        })
        .count();
    assert_eq!(
        summary_count, 1,
        "should have exactly 1 summary (the new one), old one must be filtered"
    );
}

#[tokio::test]
async fn test_compact_filters_old_summary_in_recent_block() {
    let config = CompressionConfig::default()
        .with_trigger_tokens(1)
        .with_keep_recent_messages(4);
    let client = std::sync::Arc::new(MockClient("new summary"));
    let compactor = ContextCompactor::new(client, config);

    // Build messages where old summary falls in the recent window.
    let mut msgs = vec![ChatMessage::system("sys")];
    for i in 0..5 {
        msgs.push(ChatMessage::user(format!("question {i}")));
        msgs.push(ChatMessage::assistant(format!(
            "answer {i} with some extra content to make the old block large enough"
        )));
    }
    // Old summary at the end of old block / start of recent.
    msgs.push(ChatMessage::user(format!(
        "{SUMMARY_PREFIX}\nold summary in recent"
    )));
    for i in 5..10 {
        msgs.push(ChatMessage::user(format!("question {i}")));
        msgs.push(ChatMessage::assistant(format!(
            "answer {i} with some extra content to make the old block large enough"
        )));
    }

    let result = compactor
        .compact(1, &msgs, CompressionTrigger::Auto, None)
        .await
        .unwrap()
        .unwrap();

    // Only the new summary should remain.
    let summary_count = result
        .iter()
        .filter(|m| match m {
            ChatMessage::User { content, .. } => content.starts_with(SUMMARY_PREFIX),
            _ => false,
        })
        .count();
    assert_eq!(
        summary_count, 1,
        "old summary in recent block must be filtered"
    );
}

#[tokio::test]
#[ignore] // Requires real API key: DEEPSEEK_API_KEY
async fn test_compact_with_real_deepseek_api() {
    // This test calls the real DeepSeek API to verify compact() works end-to-end.
    // Run with: cargo test test_compact_with_real_deepseek_api -- --nocapture

    let api_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        eprintln!("Skipping test: DEEPSEEK_API_KEY not set");
        return;
    }

    let _base_url = std::env::var("DEEPSEEK_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

    // Create a real DeepSeek provider.
    // TODO: Use llm-unified factory to create provider.
    // let client: std::sync::Arc<dyn agent_base::llm_trait::LlmProvider> = ...;
    return; // TODO: update to new LlmProvider API

    // The rest of this test needs to be updated for the new LlmProvider API.
    // It previously used agent_base::llm::adapt(OpenAiClient::new(...)).
    // TODO: Use llm-unified factory to create a real provider for integration testing.
}

// ── proptest: truncate_str / safe_cut_index ──

mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn truncate_str_chars_count_bounded(s in ".*", max in 0usize..500) {
            let result = truncate_str(&s, max);
            let char_count = result.chars().count();
            let original_count = s.chars().count();
            if original_count <= max {
                assert_eq!(char_count, original_count);
            } else {
                // +1 for the '…' character
                assert!(char_count <= max + 1,
                    "truncated {} chars > max {} + 1", char_count, max);
            }
        }

        #[test]
        fn truncate_str_short_string_unchanged(s in "[a-zA-Z\u{4e00}-\u{9fff}]{0,50}", max in 50usize..200) {
            let result = truncate_str(&s, max);
            assert_eq!(result, s, "short string should be unchanged");
        }

        #[test]
        fn truncate_str_cjk_char_boundary_safe(s in "[\u{4e00}-\u{9fff}]{1,200}", max in 1usize..100) {
            // Must not panic on char boundary
            let result = truncate_str(&s, max);
            assert!(result.chars().count() <= max + 1);
        }

        #[test]
        fn truncate_str_result_is_valid_utf8(s in ".*", max in 0usize..500) {
            let result = truncate_str(&s, max);
            // Verify the result is valid UTF-8 and doesn't panic on string operations
            assert!(std::str::from_utf8(result.as_bytes()).is_ok());
            let _ = result.len(); // byte length accessible
            let _ = result.chars().count(); // char count accessible
        }
    }

    /// Generate a simple message sequence for safe_cut_index testing.
    fn arb_messages() -> impl Strategy<Value = Vec<ChatMessage>> {
        prop::collection::vec(
            prop_oneof![
                "[a-z ]{0,50}".prop_map(|s| ChatMessage::user(&s)),
                "[a-z ]{0,50}".prop_map(|s| ChatMessage::assistant(&s)),
                ("[a-z]{1,10}", "[a-z ]{0,50}")
                    .prop_map(|(id, content)| ChatMessage::tool(&id, &content)),
            ],
            0..15,
        )
    }

    proptest! {
        #[test]
        fn safe_cut_index_never_panics(
            messages in arb_messages(),
            start in 0usize..15,
        ) {
            if messages.is_empty() || start >= messages.len() {
                return Ok(());
            }
            let cut = start + (messages.len() - start) / 2;
            if cut > messages.len() {
                return Ok(());
            }
            let _result = super::safe_cut_index(&messages, start, cut);
        }

        #[test]
        fn safe_cut_index_result_in_bounds(
            messages in arb_messages(),
            start in 0usize..15,
        ) {
            if messages.is_empty() || start >= messages.len() {
                return Ok(());
            }
            let cut = start + (messages.len() - start) / 2;
            if cut > messages.len() || cut < start {
                return Ok(());
            }
            let result = super::safe_cut_index(&messages, start, cut);
            assert!(result >= start, "result {} < start {}", result, start);
            assert!(result <= messages.len(), "result {} > len {}", result, messages.len());
        }

        #[test]
        fn safe_cut_index_never_ends_on_tool_call(
            messages in arb_messages(),
            start in 0usize..15,
        ) {
            if messages.is_empty() || start >= messages.len() {
                return Ok(());
            }
            let cut = start + (messages.len() - start) / 2;
            if cut > messages.len() || cut < start {
                return Ok(());
            }
            let result = super::safe_cut_index(&messages, start, cut);
            // The message left of the cut should not be Assistant with tool_calls
            if result > 0 && result < messages.len() {
                assert!(
                    !matches!(&messages[result - 1], ChatMessage::Assistant { tool_calls: Some(_), .. }),
                    "safe_cut left boundary should not be assistant with tool_calls"
                );
            }
        }
    }
}
