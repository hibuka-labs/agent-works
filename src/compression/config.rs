//! Compression configuration.

use tracing;

/// Tuning knobs for context compression.
///
/// Controls when and how the conversation history is compressed to fit within
/// the model's context window. The compressor keeps the most recent N messages
/// intact (including tool results) and summarises older history via LLM.
///
/// # Defaults
///
/// | Field | Default | Purpose |
/// |-------|---------|---------|
/// | `enabled` | `true` | Master switch |
/// | `trigger_tokens` | 256 000 | Skip compression when estimated tokens are below this |
/// | `keep_recent_messages` | 16 | Number of most-recent messages kept verbatim |
/// | `max_summary_chars` | 10 240 (10 KB) | Hard cap on the generated summary length |
/// | `max_transcript_chars` | 20 480 (20 KB) | Max chars of old messages sent to the summarizer |
#[derive(Clone, Debug)]
pub struct CompressionConfig {
    /// Master switch. When `false`, compression is a no-op.
    pub enabled: bool,
    /// Compress when the estimated token count of the message list exceeds this.
    /// Set lower than `context_window` so compression fires before the window
    /// manager's blunt trim kicks in.
    pub trigger_tokens: usize,
    /// Always keep the most recent N messages intact (the agent's working
    /// context, including tool results). Older messages are summarised.
    pub keep_recent_messages: usize,
    /// Hard cap (in chars) on the summary the summarizer LLM may produce.
    /// Output exceeding this is truncated (front 80 % + rear 20 %).
    pub max_summary_chars: usize,
    /// Max chars of the old-message transcript handed to the summarizer.
    pub max_transcript_chars: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_tokens: 256_000,
            keep_recent_messages: 16,
            max_summary_chars: 10 * 1024,    // 10 KB
            max_transcript_chars: 20 * 1024, // 20 KB
        }
    }
}

// ── Builder methods ──────────────────────────────────────────────────────────

impl CompressionConfig {
    /// Create a config with everything at default, then customise via builder.
    pub fn builder() -> Self {
        Self::default()
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_trigger_tokens(mut self, tokens: usize) -> Self {
        self.trigger_tokens = tokens;
        self
    }

    pub fn with_keep_recent_messages(mut self, n: usize) -> Self {
        self.keep_recent_messages = n;
        self
    }

    pub fn with_max_summary_chars(mut self, chars: usize) -> Self {
        self.max_summary_chars = chars;
        self
    }

    pub fn with_max_transcript_chars(mut self, chars: usize) -> Self {
        self.max_transcript_chars = chars;
        self
    }

    /// Validate that `trigger_tokens` is less than the model's context window.
    /// Logs a warning and clamps if violated. Returns the (possibly clamped) config.
    pub fn validate_context_window(mut self, context_window: usize) -> Self {
        if self.trigger_tokens >= context_window {
            tracing::warn!(
                trigger_tokens = self.trigger_tokens,
                context_window,
                "trigger_tokens >= context_window, clamping to 50% of context_window"
            );
            self.trigger_tokens = context_window / 2;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let cfg = CompressionConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.trigger_tokens, 256_000);
        assert_eq!(cfg.keep_recent_messages, 16);
        assert_eq!(cfg.max_summary_chars, 10 * 1024);
        assert_eq!(cfg.max_transcript_chars, 20 * 1024);
    }

    #[test]
    fn test_builder_chaining() {
        let cfg = CompressionConfig::builder()
            .with_enabled(false)
            .with_trigger_tokens(50_000)
            .with_keep_recent_messages(20)
            .with_max_summary_chars(8_000)
            .with_max_transcript_chars(16_000);

        assert!(!cfg.enabled);
        assert_eq!(cfg.trigger_tokens, 50_000);
        assert_eq!(cfg.keep_recent_messages, 20);
        assert_eq!(cfg.max_summary_chars, 8_000);
        assert_eq!(cfg.max_transcript_chars, 16_000);
    }
}
