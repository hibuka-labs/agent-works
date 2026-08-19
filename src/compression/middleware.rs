//! Middleware integration for context compression.
//!
//! [`CompressionMiddleware`] implements the [`agent_base::Middleware`] trait,
//! compressing the per-LLM-call message list on every `on_pre_llm` invocation.
//!
//! The middleware delegates to [`ContextCompactor`] which handles the hybrid
//! retention strategy (system prompt + summary + recent N messages) and the
//! stable-prefix cache.  The session's stored history is **never** modified;
//! only the mutable message copy in [`PreLlmCtx`] is replaced.

use std::sync::Arc;

use agent_base::{AgentResult, Middleware, PreLlmCtx, StreamClient};

use crate::compression::compactor::ContextCompactor;
use crate::compression::config::CompressionConfig;

/// Middleware that compresses conversation history before each LLM call.
///
/// Wraps a [`ContextCompactor`] and forwards `on_pre_llm` to its [`compact`]
/// method.  When the estimated token count is below [`CompressionConfig::trigger_tokens`],
/// the middleware is a no-op and the original messages pass through unchanged.
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
}

#[allow(missing_docs)]
impl CompressionMiddleware {
    /// Create a new middleware with the given config and LLM client.
    ///
    /// The client is used for summary generation (only on cache miss).
    pub fn new(config: CompressionConfig, client: Arc<dyn StreamClient>) -> Self {
        Self {
            compactor: ContextCompactor::new(client, config),
        }
    }

    /// Create a new middleware from an existing [`ContextCompactor`].
    ///
    /// Useful when the caller already has a shared compactor (e.g. for the
    /// `/compact` command that reuses the same instance).
    pub fn from_compactor(compactor: ContextCompactor) -> Self {
        Self { compactor }
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
        match self
            .compactor
            .compact(ctx.session_id.id, &ctx.messages)
            .await?
        {
            Some(compressed) => {
                ctx.messages = compressed;
            }
            None => {
                // Compression skipped (below threshold, disabled, or too few messages).
                // Original messages pass through unchanged.
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            unreachable!()
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
            unreachable!()
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

    fn make_ctx(messages: Vec<ChatMessage>) -> PreLlmCtx {
        PreLlmCtx {
            session_id: SessionId::new(1),
            messages,
            tools: vec![],
        }
    }

    fn make_messages(count: usize) -> Vec<ChatMessage> {
        let mut msgs = vec![ChatMessage::system("You are a test agent.")];
        for i in 0..count {
            msgs.push(ChatMessage::user(format!("question {i}")));
            msgs.push(ChatMessage::assistant(format!("answer {i}")));
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

        // Messages should be unchanged.
        assert_eq!(ctx.messages.len(), original_len);
        // No LLM call should have been made.
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
            .with_trigger_tokens(1) // Always trigger.
            .with_keep_recent_messages(4);
        let mw = CompressionMiddleware::new(config, client);

        let msgs = make_messages(20);
        let mut ctx = make_ctx(msgs);

        mw.on_pre_llm(&mut ctx).await.unwrap();

        // Should have fewer messages than original.
        assert!(
            ctx.messages.len() < 41,
            "expected compression, got {} messages",
            ctx.messages.len()
        );

        // First message should still be system prompt.
        assert!(matches!(&ctx.messages[0], ChatMessage::System { .. }));

        // Should have a summary message.
        use crate::compression::SUMMARY_PREFIX;
        let has_summary = ctx.messages.iter().any(|m| match m {
            ChatMessage::User { content, .. } => content.starts_with(SUMMARY_PREFIX),
            _ => false,
        });
        assert!(has_summary, "expected summary in compressed output");

        // LLM should have been called once (cache miss).
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

        // First call.
        let mut ctx1 = make_ctx(msgs.clone());
        mw.on_pre_llm(&mut ctx1).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Second call with same messages — should hit cache.
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

        // Should not error — falls back to dropping old block.
        mw.on_pre_llm(&mut ctx).await.unwrap();

        // Should have system + recent only (no summary).
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

        // Multi-system prompt scenario.
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

        // Both system messages should be preserved.
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
}
