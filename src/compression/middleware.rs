//! Middleware integration for context compression.
//!
//! [`CompressionMiddleware`] implements the [`agent_base::Middleware`] trait,
//! compressing the per-LLM-call message list on every `on_pre_llm` invocation.
//!
//! The middleware delegates to [`ContextCompactor`] which handles the hybrid
//! retention strategy (system prompt + summary + recent N messages) and the
//! stable-prefix cache.  The session's stored history is **never** modified;
//! only the mutable message copy in [`PreLlmCtx`] is replaced.
//!
//! A [`CompressionPolicy`] controls whether compression proceeds once the token
//! threshold is crossed.  The default [`AutoCompressionPolicy`] always proceeds;
//! custom policies can add user confirmation, rate limiting, etc.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_base::{AgentResult, ChatMessage, Middleware, PreLlmCtx, StreamClient};

use crate::compression::compactor::{ContextCompactor, estimate_message_tokens};
use crate::compression::config::CompressionConfig;
use crate::compression::events::{CompressionEvent, CompressionTrigger};
use crate::compression::filter::is_summary_message;
use crate::compression::policy::{AutoCompressionPolicy, CompressionPolicy};

/// Write compression before/after log to `/tmp/phi-agent-compression/`.
fn write_compression_log(
    session_id: u64,
    before: &[ChatMessage],
    after: &[ChatMessage],
    tokens_before: usize,
    tokens_after: usize,
    reduction_pct: i32,
) {
    use std::io::Write;

    let dir = std::path::Path::new("/tmp/phi-agent-compression");
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!("Failed to create compression log dir: {e}");
        return;
    }

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = dir.join(format!("session_{session_id}_{ts}.json"));

    let log = serde_json::json!({
        "session_id": session_id,
        "timestamp": ts,
        "evaluation": {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction_pct": reduction_pct,
            "msg_count_before": before.len(),
            "msg_count_after": after.len(),
        },
        "before": before.iter().map(format_message).collect::<Vec<_>>(),
        "after": after.iter().map(format_message).collect::<Vec<_>>(),
    });

    match std::fs::File::create(&filename) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(serde_json::to_string_pretty(&log).unwrap().as_bytes()) {
                tracing::warn!("Failed to write compression log: {e}");
            } else {
                tracing::info!("Compression log written to {}", filename.display());
            }
        }
        Err(e) => tracing::warn!("Failed to create compression log file: {e}"),
    }
}

/// Format a ChatMessage into a readable JSON value for logging.
fn format_message(msg: &ChatMessage) -> serde_json::Value {
    match msg {
        ChatMessage::System { content, .. } => serde_json::json!({
            "role": "system",
            "content": content,
        }),
        ChatMessage::User { content, .. } => serde_json::json!({
            "role": "user",
            "content": content,
        }),
        ChatMessage::Assistant {
            content,
            tool_calls,
            ..
        } => {
            let mut m = serde_json::json!({
                "role": "assistant",
                "content": content,
            });
            if let Some(tc) = tool_calls {
                m["tool_calls"] = serde_json::to_value(tc).unwrap_or_default();
            }
            m
        }
        ChatMessage::Tool {
            content,
            tool_call_id,
            ..
        } => serde_json::json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }),
        ChatMessage::Custom { role, data } => serde_json::json!({
            "role": role,
            "data": data,
        }),
    }
}

/// Middleware that compresses conversation history before each LLM call.
///
/// Wraps a [`ContextCompactor`] and forwards `on_pre_llm` to its [`compact`]
/// method.  When the estimated token count is below [`CompressionConfig::trigger_tokens`],
/// the middleware is a no-op and the original messages pass through unchanged.
///
/// A [`CompressionPolicy`] controls whether compression actually proceeds once
/// the threshold is crossed.  Use [`with_policy`](Self::with_policy) to supply
/// a custom policy (e.g. user confirmation, rate limiting).
///
/// # Usage
///
/// ```ignore
/// use agent_works::compression::{CompressionConfig, CompressionMiddleware};
///
/// let mw = CompressionMiddleware::new(config, client);
/// builder.middleware(mw);
/// ```
///
/// [`compact`]: ContextCompactor::compact
pub struct CompressionMiddleware {
    compactor: ContextCompactor,
    policy: Box<dyn CompressionPolicy>,
    /// Message count after the last successful compression.
    /// Used to calculate only the *new* old-block tokens for threshold checks,
    /// preventing re-triggering caused by recent messages rolling into the old block.
    last_compressed_msg_count: AtomicUsize,
}

#[allow(missing_docs)]
impl CompressionMiddleware {
    /// Create a new middleware with the given config and LLM client.
    ///
    /// Uses [`AutoCompressionPolicy`] (always compress when threshold is hit).
    pub fn new(config: CompressionConfig, client: Arc<dyn StreamClient>) -> Self {
        Self {
            compactor: ContextCompactor::new(client, config),
            policy: Box::new(AutoCompressionPolicy),
            last_compressed_msg_count: AtomicUsize::new(0),
        }
    }

    /// Create a new middleware from an existing [`ContextCompactor`].
    ///
    /// Uses [`AutoCompressionPolicy`].
    pub fn from_compactor(compactor: ContextCompactor) -> Self {
        Self {
            compactor,
            policy: Box::new(AutoCompressionPolicy),
            last_compressed_msg_count: AtomicUsize::new(0),
        }
    }

    /// Create a new middleware with a custom [`CompressionPolicy`].
    pub fn with_policy(
        config: CompressionConfig,
        client: Arc<dyn StreamClient>,
        policy: Box<dyn CompressionPolicy>,
    ) -> Self {
        Self {
            compactor: ContextCompactor::new(client, config),
            policy,
            last_compressed_msg_count: AtomicUsize::new(0),
        }
    }

    /// Access the inner compactor (e.g. for `/compact` or cache clearing).
    pub fn compactor(&self) -> &ContextCompactor {
        &self.compactor
    }

    /// Create a cloned handle of the inner compactor.
    ///
    /// The clone shares the same cache — clearing it through either handle
    /// affects both.  Useful for storing a separate handle outside the
    /// middleware (e.g. in `PhiAgent` for `/compact` access).
    pub fn clone_compactor(&self) -> ContextCompactor {
        self.compactor.clone_handle()
    }

    /// Access the compression config.
    pub fn config(&self) -> &CompressionConfig {
        self.compactor.config()
    }
}

#[async_trait::async_trait]
impl Middleware for CompressionMiddleware {
    async fn on_pre_llm(&self, ctx: &mut PreLlmCtx) -> AgentResult<()> {
        let t0 = std::time::Instant::now();
        let msg_count = ctx.messages.len();

        // Quick check — below minimum message count, skip entirely.
        // Need at least system + keep_recent + 1 to have anything to compress.
        let keep = self.config().keep_recent_messages;
        if msg_count <= keep + 1 {
            return Ok(());
        }

        // Only count tokens added SINCE the last compression (new old-block
        // content).  This prevents re-triggering caused by recent messages
        // rolling into the old block after compression.
        //
        // Also skip system and existing summary messages.
        let last_compressed = self.last_compressed_msg_count.load(Ordering::Relaxed);
        let tokens_before: usize = ctx
            .messages
            .iter()
            .skip(last_compressed)
            .filter(|m| !matches!(m, ChatMessage::System { .. }))
            .filter(|m| !is_summary_message(m))
            .map(estimate_message_tokens)
            .sum();
        tracing::info!(
            tokens_before,
            msg_count,
            trigger = self.config().trigger_tokens,
            "[compression-timing] threshold check"
        );

        // Quick check — below threshold, skip entirely.
        if tokens_before <= self.config().trigger_tokens {
            return Ok(());
        }

        tracing::info!(
            elapsed_ms = t0.elapsed().as_millis() as u64,
            "[compression-timing] threshold passed"
        );

        // Policy check — ask policy whether to proceed.
        if !self.policy.should_compress(tokens_before, msg_count).await {
            return Ok(());
        }

        tracing::info!(
            elapsed_ms = t0.elapsed().as_millis() as u64,
            "[compression-timing] policy passed, entering compact"
        );

        // Determine trigger type (manual if /compact, auto otherwise).
        // For now, middleware is always auto; /compact goes through a different path.
        let trigger = CompressionTrigger::Auto;
        let sid = ctx.session_id.id;

        // Discard old summary before re-compressing.
        // compact() handles preserving user messages and summarizing assistant/tool.
        // We only need to remove the old summary so it doesn't get re-summarized.
        let filtered: Vec<ChatMessage> = ctx
            .messages
            .iter()
            .filter(|m| !is_summary_message(m))
            .cloned()
            .collect();

        // After filtering, check if there's enough content to compress.
        // Need more than keep_recent messages to have an old block at all.
        let keep = self.config().keep_recent_messages;
        if filtered.len() <= keep + 1 {
            self.last_compressed_msg_count
                .store(msg_count, Ordering::Relaxed);
            return Ok(());
        }

        // Save messages before compression for logging.
        let messages_before = ctx.messages.clone();
        let t_compact = std::time::Instant::now();

        match self
            .compactor
            .compact(
                sid,
                &filtered,
                trigger.clone(),
                Some(&|ev| ctx.emit(ev.into_user_event())),
            )
            .await?
        {
            Some(compressed) => {
                tracing::info!(
                    elapsed_ms = t_compact.elapsed().as_millis() as u64,
                    "[compression-timing] compact() returned Some"
                );
                ctx.messages = compressed;
                // Record the pre-compression message count so the next threshold
                // check only counts new content added since this compression.
                // Using pre-compression count (not compressed length) because
                // skip() operates on the full message array structure.
                self.last_compressed_msg_count
                    .store(msg_count, Ordering::Relaxed);
                // Count replacement tokens (same scope as tokens_before: old block only).
                let keep = self.config().keep_recent_messages;
                let old_end = ctx.messages.len().saturating_sub(keep);
                let tokens_after: usize = ctx
                    .messages
                    .iter()
                    .take(old_end)
                    .filter(|m| !matches!(m, ChatMessage::System { .. }))
                    .map(estimate_message_tokens)
                    .sum();
                let reduction_pct = if tokens_before > 0 {
                    ((tokens_before as f64 - tokens_after as f64) / tokens_before as f64 * 100.0)
                        .round() as i32
                } else {
                    0
                };

                // Send Completed event.
                ctx.emit(
                    CompressionEvent::Completed {
                        session_id: sid,
                        tokens_before,
                        tokens_after,
                        reduction_pct,
                        msg_count_before: msg_count,
                        msg_count_after: ctx.messages.len(),
                        trigger,
                    }
                    .into_user_event(),
                );

                // Write before/after log to /tmp/phi-agent-compression/.
                write_compression_log(
                    ctx.session_id.id,
                    &messages_before,
                    &ctx.messages,
                    tokens_before,
                    tokens_after,
                    reduction_pct,
                );
            }
            None => {
                // Compression skipped (threshold not reached, disabled, or too few messages).
                // This is a normal no-op — do NOT send Failed event.
                // compact() is pure and never modified ctx.messages, so no restore needed.
                //
                // Still update the checkpoint so the next threshold check only
                // counts truly new content, preventing a re-trigger loop.
                self.last_compressed_msg_count
                    .store(msg_count, Ordering::Relaxed);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::events::CompressionEvent;
    use crate::compression::policy::{CompressionPolicy, RateLimitPolicy};
    use agent_base::{
        AgentResult, ChatMessage, LlmCapabilities, Middleware, PreLlmCtx, ResponseFormat,
        SessionId, StreamClient,
    };

    // ── Test helpers ──────────────────────────────────────────────────────

    /// Minimal mock that returns a fixed string and counts calls.
    struct MockClient {
        response: &'static str,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl StreamClient for MockClient {
        async fn stream(
            &self,
            _messages: &[ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<
            std::pin::Pin<
                Box<dyn futures_core::Stream<Item = AgentResult<agent_base::StreamChunk>> + Send>,
            >,
        > {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let response = self.response.to_string();
            Ok(Box::pin(futures_util::stream::once(async move {
                Ok(agent_base::StreamChunk::Text(response))
            })))
        }

        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(self.response.to_string())
        }

        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::default()
        }
    }

    /// Mock that always returns an error.
    struct FailingClient;

    #[async_trait::async_trait]
    impl StreamClient for FailingClient {
        async fn stream(
            &self,
            _messages: &[ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<
            std::pin::Pin<
                Box<dyn futures_core::Stream<Item = AgentResult<agent_base::StreamChunk>> + Send>,
            >,
        > {
            Err(agent_base::AgentError::llm("summarisation failed"))
        }

        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<String> {
            Err(agent_base::AgentError::llm("summarisation failed"))
        }

        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::default()
        }
    }

    /// Policy that records calls and returns a fixed value.
    struct SpyPolicy {
        /// `(tokens_before, msg_count)` for each call.
        observed: std::sync::Arc<std::sync::Mutex<Vec<(usize, usize)>>>,
        result: bool,
    }

    impl SpyPolicy {
        #[allow(clippy::type_complexity)]
        fn new(result: bool) -> (Self, std::sync::Arc<std::sync::Mutex<Vec<(usize, usize)>>>) {
            let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            (
                Self {
                    observed: observed.clone(),
                    result,
                },
                observed,
            )
        }
    }

    #[async_trait::async_trait]
    impl CompressionPolicy for SpyPolicy {
        async fn should_compress(&self, tokens_before: usize, msg_count: usize) -> bool {
            self.observed
                .lock()
                .unwrap()
                .push((tokens_before, msg_count));
            self.result
        }
    }

    fn make_ctx(messages: Vec<ChatMessage>) -> PreLlmCtx {
        PreLlmCtx {
            session_id: SessionId::new(1),
            messages,
            tools: vec![],
            emit_fn: None,
        }
    }

    fn make_ctx_with_events(
        messages: Vec<ChatMessage>,
        events: std::sync::Arc<std::sync::Mutex<Vec<CompressionEvent>>>,
    ) -> PreLlmCtx {
        let events_clone = events.clone();
        PreLlmCtx {
            session_id: SessionId::new(1),
            messages,
            tools: vec![],
            emit_fn: Some(Box::new(move |event: agent_base::UserEvent| {
                if let Some(ev) = CompressionEvent::from_user_event(&event) {
                    events_clone.lock().unwrap().push(ev);
                }
            })),
        }
    }

    fn make_messages(count: usize) -> Vec<ChatMessage> {
        let mut msgs = vec![ChatMessage::system("You are a test agent.")];
        for i in 0..count {
            msgs.push(ChatMessage::user(format!("question {i}")));
            msgs.push(ChatMessage::assistant(format!(
                "answer {i} with some extra content to make the old block large enough"
            )));
        }
        msgs
    }

    // ── CompressionMiddleware::on_pre_llm ─────────────────────────────────

    #[tokio::test]
    async fn test_middleware_noop_when_below_threshold() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default().with_trigger_tokens(999_999);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(5);
        let original_len = msgs.len();
        let mut ctx = make_ctx(msgs);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        assert_eq!(ctx.messages.len(), original_len);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_middleware_compresses_when_above_threshold() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "compressed summary of earlier work",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);
        let mut ctx = make_ctx(msgs);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        assert!(
            ctx.messages.len() < 41,
            "expected compression, got {} messages",
            ctx.messages.len()
        );
        assert!(matches!(&ctx.messages[0], ChatMessage::System { .. }));

        use crate::compression::SUMMARY_PREFIX;
        let has_summary = ctx.messages.iter().any(|m| match m {
            ChatMessage::User { content, .. } => content.starts_with(SUMMARY_PREFIX),
            _ => false,
        });
        assert!(has_summary, "expected summary in compressed output");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_middleware_uses_cache_on_repeated_calls() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "cached summary",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);

        let mut ctx1 = make_ctx(msgs.clone());
        mw.on_pre_llm(&mut ctx1).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        let mut ctx2 = make_ctx(msgs);
        mw.on_pre_llm(&mut ctx2).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_middleware_fallback_on_summarisation_failure() {
        let client = std::sync::Arc::new(FailingClient);
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);
        let mut ctx = make_ctx(msgs);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        assert!(matches!(&ctx.messages[0], ChatMessage::System { .. }));

        use crate::compression::SUMMARY_PREFIX;
        let has_summary = ctx.messages.iter().any(|m| match m {
            ChatMessage::User { content, .. } => content.starts_with(SUMMARY_PREFIX),
            _ => false,
        });
        assert!(!has_summary, "should have no summary on failure");
    }

    #[tokio::test]
    async fn test_middleware_noop_when_disabled() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default().with_enabled(false);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);
        let original_len = msgs.len();
        let mut ctx = make_ctx(msgs);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        assert_eq!(ctx.messages.len(), original_len);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_middleware_preserves_system_prompt() {
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let mut msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::system("Extra instructions."),
        ];
        for i in 0..20 {
            msgs.push(ChatMessage::user(format!("q{i}")));
            msgs.push(ChatMessage::assistant(format!("a{i}")));
        }

        let mut ctx = make_ctx(msgs);
        mw.on_pre_llm(&mut ctx).await.unwrap();

        assert!(matches!(&ctx.messages[0], ChatMessage::System { .. }));
        assert!(matches!(&ctx.messages[1], ChatMessage::System { .. }));
        match &ctx.messages[0] {
            ChatMessage::System { content, .. } => assert_eq!(content, "You are helpful."),
            _ => unreachable!(),
        }
        match &ctx.messages[1] {
            ChatMessage::System { content, .. } => assert_eq!(content, "Extra instructions."),
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn test_from_compactor_and_accessor() {
        let client = std::sync::Arc::new(MockClient {
            response: "s",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default().with_trigger_tokens(42);
        let compactor = ContextCompactor::new(client, config.clone());
        let mw = CompressionMiddleware::from_compactor(compactor);

        assert_eq!(mw.config().trigger_tokens, 42);
    }

    // ── Policy integration ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_policy_called_with_correct_args() {
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let (policy, observed) = SpyPolicy::new(true);
        let mw = CompressionMiddleware::with_policy(config, client, Box::new(policy));

        let msgs = make_messages(10);
        let msg_count = msgs.len();
        let tokens_est: usize = msgs.iter().map(estimate_message_tokens).sum();
        let mut ctx = make_ctx(msgs);
        mw.on_pre_llm(&mut ctx).await.unwrap();

        // Policy should have been called once.
        let spy_calls = observed.lock().unwrap();
        assert_eq!(spy_calls.len(), 1);
        assert_eq!(spy_calls[0].1, msg_count);
        // tokens_before should be > trigger_tokens (which is 1).
        assert!(spy_calls[0].0 > 1);
        // Approximate match — spy should receive roughly the same token estimate.
        assert!(
            (spy_calls[0].0 as i64 - tokens_est as i64).unsigned_abs() < 100,
            "token estimate mismatch: spy={}, computed={}",
            spy_calls[0].0,
            tokens_est
        );
    }

    #[tokio::test]
    async fn test_policy_deny_skips_compression() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let (policy, _observed) = SpyPolicy::new(false);
        let mw = CompressionMiddleware::with_policy(config, client, Box::new(policy));

        let msgs = make_messages(20);
        let original_len = msgs.len();
        let mut ctx = make_ctx(msgs);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        // Messages should be unchanged — policy denied.
        assert_eq!(ctx.messages.len(), original_len);
        // No LLM call should have been made.
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_policy_not_called_when_below_threshold() {
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default().with_trigger_tokens(999_999);
        let (policy, observed) = SpyPolicy::new(true);
        let mw = CompressionMiddleware::with_policy(config, client, Box::new(policy));

        let msgs = make_messages(5);
        let mut ctx = make_ctx(msgs);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        // Policy should NOT have been called — below threshold.
        let spy_calls = observed.lock().unwrap();
        assert_eq!(spy_calls.len(), 0);
    }

    #[tokio::test]
    async fn test_rate_limit_policy_blocks_repeated_compression() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        // 60-second rate limit — second call should be blocked.
        let policy = Box::new(RateLimitPolicy::new(std::time::Duration::from_secs(60)));
        let mw = CompressionMiddleware::with_policy(config, client, policy);

        let msgs = make_messages(20);

        // First call — should compress.
        let mut ctx1 = make_ctx(msgs.clone());
        mw.on_pre_llm(&mut ctx1).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Second call immediately — rate limited, no LLM call.
        let mut ctx2 = make_ctx(msgs);
        mw.on_pre_llm(&mut ctx2).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    // ── Typed event emission ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_emits_preparing_and_completed_events() {
        let client = std::sync::Arc::new(MockClient {
            response: "compressed summary text",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<CompressionEvent>::new()));
        let mut ctx = make_ctx_with_events(msgs, events.clone());

        mw.on_pre_llm(&mut ctx).await.unwrap();

        let events = events.lock().unwrap();
        // Should have: Preparing (from compact), Started (from compact), Progress, Completed.
        assert!(
            events.len() >= 3,
            "expected at least 3 events, got {}",
            events.len()
        );

        // First event should be Preparing.
        assert!(
            matches!(&events[0], CompressionEvent::Preparing { .. }),
            "first event should be Preparing, got {:?}",
            events[0]
        );

        // Second event should be Started.
        assert!(
            matches!(&events[1], CompressionEvent::Started { .. }),
            "second event should be Started, got {:?}",
            events[1]
        );

        // Last event should be Completed.
        assert!(
            matches!(events.last().unwrap(), CompressionEvent::Completed { .. }),
            "last event should be Completed, got {:?}",
            events.last().unwrap()
        );

        // Verify Preparing has trigger = Auto.
        if let CompressionEvent::Preparing { trigger, .. } = &events[0] {
            assert_eq!(*trigger, CompressionTrigger::Auto);
        }
    }

    #[tokio::test]
    async fn test_no_events_when_below_threshold() {
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default().with_trigger_tokens(999_999);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(5);
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<CompressionEvent>::new()));
        let mut ctx = make_ctx_with_events(msgs, events.clone());

        mw.on_pre_llm(&mut ctx).await.unwrap();

        let events = events.lock().unwrap();
        assert!(events.is_empty(), "no events expected below threshold");
    }

    #[tokio::test]
    async fn test_no_events_when_policy_denies() {
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let (policy, _observed) = SpyPolicy::new(false);
        let mw = CompressionMiddleware::with_policy(config, client, Box::new(policy));

        let msgs = make_messages(20);
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<CompressionEvent>::new()));
        let mut ctx = make_ctx_with_events(msgs, events.clone());

        mw.on_pre_llm(&mut ctx).await.unwrap();

        let events = events.lock().unwrap();
        assert!(events.is_empty(), "no events expected when policy denies");
    }

    #[tokio::test]
    async fn test_messages_unchanged_when_compact_returns_none() {
        // compact() returns None when disabled or too few messages —
        // compact() is pure so ctx.messages is never mutated.
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "summary",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4)
            .with_enabled(false); // disabled → compact() returns None
        let (policy, _observed) = SpyPolicy::new(true);
        let mw = CompressionMiddleware::with_policy(config, client, Box::new(policy));

        let msgs = make_messages(20);
        let original_len = msgs.len();
        let mut ctx = make_ctx(msgs);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        // Messages unchanged — compact() never modified them.
        assert_eq!(ctx.messages.len(), original_len);
    }

    #[tokio::test]
    async fn test_exact_event_sequence_on_cache_miss() {
        let client = std::sync::Arc::new(MockClient {
            response: "compressed summary text",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<CompressionEvent>::new()));
        let mut ctx = make_ctx_with_events(msgs, events.clone());

        mw.on_pre_llm(&mut ctx).await.unwrap();

        let events = events.lock().unwrap();
        // Exact sequence: Preparing → Started → Progress(0) → Progress(N) → Completed
        // Preparing = compact emits Preparing first, then Started,
        // Progress(0) = "Connecting to LLM", Progress(N) = streaming chars from summarizer.
        assert_eq!(
            events.len(),
            5,
            "expected 5 events, got {}: {:?}",
            events.len(),
            *events
        );
        assert!(matches!(&events[0], CompressionEvent::Preparing { .. }));
        assert!(matches!(&events[1], CompressionEvent::Started { .. }));
        assert!(matches!(
            &events[2],
            CompressionEvent::Progress { chars: 0, .. }
        ));
        // events[3] is Progress with actual char count from the mock response.
        assert!(matches!(&events[3], CompressionEvent::Progress { chars, .. } if *chars > 0));
        assert!(matches!(&events[4], CompressionEvent::Completed { .. }));
    }

    #[tokio::test]
    async fn test_event_sequence_on_no_retrigger() {
        // After compression, calling with the same messages should NOT re-trigger
        // because last_compressed_msg_count was set to the pre-compression count.
        let client = std::sync::Arc::new(MockClient {
            response: "cached summary",
            calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<CompressionEvent>::new()));

        // First call — should compress and emit events.
        let mut ctx1 = make_ctx_with_events(msgs.clone(), events.clone());
        mw.on_pre_llm(&mut ctx1).await.unwrap();
        assert!(events.lock().unwrap().len() >= 3, "first call should emit events");
        events.lock().unwrap().clear();

        // Second call with the same messages — should NOT re-trigger.
        // last_compressed=20, skip(20) → 0 new messages → below threshold.
        let mut ctx2 = make_ctx_with_events(msgs, events.clone());
        mw.on_pre_llm(&mut ctx2).await.unwrap();

        let events = events.lock().unwrap();
        assert_eq!(
            events.len(),
            0,
            "same messages: expected 0 events (no re-trigger), got {}",
            events.len()
        );
    }

    #[tokio::test]
    async fn test_no_retrigger_after_compression() {
        // After successful compression, the next call with the compressed messages
        // should NOT re-trigger compression because last_compressed_msg_count
        // was set to the pre-compression message count, so skip() returns 0 new messages.
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "short",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(1)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);

        // First call — should compress (20 messages > trigger_tokens=1).
        let mut ctx1 = make_ctx(msgs.clone());
        mw.on_pre_llm(&mut ctx1).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let compressed_len = ctx1.messages.len();
        assert!(compressed_len < msgs.len(), "should have compressed");

        // Second call with the compressed messages (realistic: next turn reloads
        // compressed state from disk). last_compressed=20 (pre-compression count),
        // skip(20) on compressed_len messages → 0 messages → no re-trigger.
        let mut ctx2 = make_ctx(ctx1.messages.clone());
        mw.on_pre_llm(&mut ctx2).await.unwrap();
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "should NOT have re-triggered compression"
        );
    }

    /// End-to-end simulation: 30 turns of long conversation.
    /// Verifies compression triggers, no re-trigger, and old block stability.
    #[tokio::test]
    async fn test_e2e_simulation_30_turns() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let client = std::sync::Arc::new(MockClient {
            response: "compressed summary of the conversation so far",
            calls: calls.clone(),
        });
        let config = CompressionConfig::default()
            .with_trigger_tokens(2000)
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let mut messages = vec![ChatMessage::system("You are a helpful assistant.")];
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<CompressionEvent>::new()));
        let mut compression_count = 0;
        let mut compression_turns = Vec::new();
        let mut old_block_sizes = Vec::new();

        for turn in 0..30 {
            messages.push(ChatMessage::user(format!(
                "请继续讲故事，这是第{}轮对话。我想听一个关于古代英雄的故事，要有曲折的情节和深刻的寓意，最好能让人有所启发和思考。",
                turn + 1
            )));
            // Longer assistant response (~500 chars) to accumulate tokens faster.
            messages.push(ChatMessage::assistant(format!(
                "好的，让我继续讲第{}轮的故事。从前有座山，山里有座庙，庙里有个老和尚在讲故事。\
                 这个故事讲的是从前有座山，山里有座庙，庙里有个老和尚在讲故事。\
                 故事的内容是关于一个勇敢的冒险者，他走遍了千山万水，经历了无数磨难。\
                 他遇到了各种各样的人，有善良的农夫，有狡猾的商人，有智慧的老者。\
                 每个人都给了他不同的启示，让他对人生有了更深的理解。\
                 他学会了坚韧不拔，学会了与人为善，学会了在困境中寻找希望。\
                 最终，他回到了家乡，成为了一个受人尊敬的长者，把自己的故事讲给后人听。\
                 这个故事告诉我们，人生就是一场旅行，重要的不是目的地，而是沿途的风景。",
                turn + 1
            )));

            let mut ctx = make_ctx_with_events(messages.clone(), events.clone());
            mw.on_pre_llm(&mut ctx).await.unwrap();
            messages = ctx.messages.clone();

            let new_compressions = events
                .lock()
                .unwrap()
                .iter()
                .filter(|e| matches!(e, CompressionEvent::Completed { .. }))
                .count();
            let turn_compressions = new_compressions - compression_count;
            compression_count = new_compressions;

            if turn_compressions > 0 {
                compression_turns.push(turn + 1);
                // Record old block size from the compression event.
                if let CompressionEvent::Completed {
                    msg_count_before, msg_count_after, ..
                } = events.lock().unwrap().last().unwrap()
                {
                    old_block_sizes.push((*msg_count_before, *msg_count_after));
                }
                println!(
                    "[Turn {:2}] COMPRESSED | msgs={:2} → {:2} | llm_calls={}",
                    turn + 1,
                    old_block_sizes.last().unwrap().0,
                    old_block_sizes.last().unwrap().1,
                    calls.load(std::sync::atomic::Ordering::SeqCst)
                );
            } else {
                println!(
                    "[Turn {:2}] ok         | msgs={:2}",
                    turn + 1,
                    messages.len(),
                );
            }
        }

        println!("\n=== Simulation Summary ===");
        println!("Compression turns: {:?}", compression_turns);
        println!("Total compressions: {}", compression_count);
        println!("LLM calls: {}", calls.load(std::sync::atomic::Ordering::SeqCst));
        println!("Final messages: {}", messages.len());
        println!("Old block before/after: {:?}", old_block_sizes);

        // Verify: at least 2 compressions in 30 turns.
        assert!(
            compression_count >= 2,
            "expected at least 2 compressions, got {}",
            compression_count
        );

        // Verify: LLM calls == compression count (no wasted calls).
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            compression_count,
        );

        // Verify: no two compressions on consecutive turns.
        for window in compression_turns.windows(2) {
            assert!(
                window[1] - window[0] >= 2,
                "consecutive compressions at turns {:?}",
                window
            );
        }

        // Verify: final message count is reasonable.
        // With "preserve user messages" strategy, users accumulate, so the count
        // grows over time.  But it should be less than uncompressed (61 msgs).
        assert!(
            messages.len() < 61,
            "final messages should be < 61 (uncompressed), got {}",
            messages.len()
        );
    }
}
