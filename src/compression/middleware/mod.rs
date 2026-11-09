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

use agent_base::{AgentResult, ChatMessage, Middleware, PreLlmCtx};

use crate::compression::compactor::{ContextCompactor, estimate_message_tokens};
use std::sync::Once;

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
    pub fn new(
        config: CompressionConfig,
        client: Arc<dyn agent_base::llm_trait::LlmProvider>,
    ) -> Self {
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
        client: Arc<dyn agent_base::llm_trait::LlmProvider>,
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

        // Defensive: warn once if trigger_tokens >= 1M.
        // The caller should ensure trigger_tokens < context_window.
        static TRIGGER_WARN: Once = Once::new();
        if self.config().trigger_tokens >= 1_000_000 {
            TRIGGER_WARN.call_once(|| {
                tracing::warn!(
                    trigger_tokens = self.config().trigger_tokens,
                    "trigger_tokens >= 1M — likely exceeds context_window; compression may never fire efficiently"
                );
            });
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
                // Count tokens after compression (same scope as tokens_before).
                // tokens_before counts all non-system, non-summary messages from last_compressed.
                // tokens_after should count the same scope after compression.
                // Since compression replaces the old block with a summary, we count
                // all non-system messages in the compressed result (including the new summary).
                let tokens_after: usize = ctx
                    .messages
                    .iter()
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
mod tests;
