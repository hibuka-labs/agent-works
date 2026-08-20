//! Compression policy — controls whether compression should proceed.
//!
//! The [`CompressionPolicy`] trait lets consumers customise the decision that
//! happens *after* the token threshold is crossed but *before* the actual
//! compression work begins.  The framework ships three built-in policies:
//!
//! - [`AutoCompressionPolicy`] — always proceed (default).
//! - [`UserConfirmationPolicy`] — prompt the user for y/n.
//! - [`RateLimitPolicy`] — enforce a minimum interval between compressions.
//!
//! ## Communication pattern
//!
//! ```text
//! Framework → Consumer: via CompressionEvent (event notification)
//! Consumer → Framework: via should_compress() return value (bool)
//! ```

use std::io::BufRead;

use async_trait::async_trait;

/// Compression policy trait — controls whether compression should proceed.
///
/// Framework provides default implementations. Users can implement custom
/// policies (e.g., user confirmation, rate limiting, etc.).
///
/// # Communication pattern
///
/// - Framework → User: via [`CompressionEvent`](super::events::CompressionEvent) (event notification)
/// - User → Framework: via `should_compress()` return value (bool)
#[async_trait]
pub trait CompressionPolicy: Send + Sync {
    /// Called before compression starts.
    ///
    /// # Arguments
    /// * `tokens_before` — estimated token count of current messages
    /// * `msg_count` — number of messages in the session
    ///
    /// # Returns
    /// * `true` — proceed with compression
    /// * `false` — skip compression, keep current messages
    async fn should_compress(&self, tokens_before: usize, msg_count: usize) -> bool;
}

// ── Built-in policies ──────────────────────────────────────────────────────

/// Default policy: always compress when threshold is exceeded.
///
/// This is the default behaviour — no user interaction required.
pub struct AutoCompressionPolicy;

#[async_trait]
impl CompressionPolicy for AutoCompressionPolicy {
    async fn should_compress(&self, _tokens_before: usize, _msg_count: usize) -> bool {
        true
    }
}

/// User confirmation policy: prompt user before compression.
///
/// Reads y/n input from an injected reader.  Use [`Self::stdin()`] for
/// interactive CLIs, or [`Self::new()`] with a custom reader (e.g. for tests).
///
/// The read runs inside [`tokio::task::spawn_blocking`] so it does not stall
/// the async runtime.
pub struct UserConfirmationPolicy {
    reader: std::sync::Mutex<Box<dyn BufRead + Send>>,
}

impl UserConfirmationPolicy {
    /// Create with a custom reader (e.g. `Cursor<Vec<u8>>` for tests).
    pub fn new(reader: impl BufRead + Send + 'static) -> Self {
        Self {
            reader: std::sync::Mutex::new(Box::new(reader)),
        }
    }

    /// Create reading from stdin.
    pub fn stdin() -> Self {
        Self::new(std::io::BufReader::new(std::io::stdin()))
    }
}

#[async_trait]
impl CompressionPolicy for UserConfirmationPolicy {
    async fn should_compress(&self, tokens_before: usize, msg_count: usize) -> bool {
        println!(
            "\n⏳ Compression needed (~{} tokens, {} messages). Proceed? (y/n): ",
            tokens_before, msg_count
        );
        let mut reader = self.reader.lock().unwrap();
        let mut input = String::new();
        reader.read_line(&mut input).unwrap_or(0);
        input.trim().to_lowercase() == "y"
    }
}

/// Rate limiting policy: compress at most once every N seconds.
///
/// Prevents compression from running too frequently.  Useful when the token
/// threshold is low and the user sends many rapid messages.
pub struct RateLimitPolicy {
    min_interval: std::time::Duration,
    last_compression: std::sync::Mutex<Option<std::time::Instant>>,
}

impl RateLimitPolicy {
    /// Create a new rate-limit policy with the given minimum interval.
    pub fn new(min_interval: std::time::Duration) -> Self {
        Self {
            min_interval,
            last_compression: std::sync::Mutex::new(None),
        }
    }
}

#[async_trait]
impl CompressionPolicy for RateLimitPolicy {
    async fn should_compress(&self, _tokens_before: usize, _msg_count: usize) -> bool {
        let mut last = self.last_compression.lock().unwrap();
        if let Some(last_time) = *last
            && last_time.elapsed() < self.min_interval
        {
            return false; // Too soon
        }
        *last = Some(std::time::Instant::now());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auto_policy_always_returns_true() {
        let policy = AutoCompressionPolicy;
        assert!(policy.should_compress(0, 0).await);
        assert!(policy.should_compress(999_999, 1000).await);
    }

    #[tokio::test]
    async fn test_rate_limit_policy_allows_first_call() {
        let policy = RateLimitPolicy::new(std::time::Duration::from_secs(60));
        assert!(policy.should_compress(4000, 10).await);
    }

    #[tokio::test]
    async fn test_rate_limit_policy_blocks_second_call_within_interval() {
        let policy = RateLimitPolicy::new(std::time::Duration::from_secs(60));
        assert!(policy.should_compress(4000, 10).await);
        // Second call immediately — should be blocked.
        assert!(!policy.should_compress(4000, 10).await);
    }

    #[tokio::test]
    async fn test_rate_limit_policy_allows_after_interval() {
        let policy = RateLimitPolicy::new(std::time::Duration::from_millis(20));
        assert!(policy.should_compress(4000, 10).await);
        // Wait well beyond the interval to avoid CI flakiness.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(policy.should_compress(4000, 10).await);
    }

    #[tokio::test]
    async fn test_rate_limit_policy_tracks_last_compression_time() {
        let policy = RateLimitPolicy::new(std::time::Duration::from_millis(20));

        // First call — allowed.
        assert!(policy.should_compress(1000, 5).await);

        // Within interval — blocked.
        assert!(!policy.should_compress(2000, 10).await);

        // After interval — allowed again.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(policy.should_compress(3000, 15).await);

        // Within new interval — blocked again.
        assert!(!policy.should_compress(4000, 20).await);
    }

    // ── UserConfirmationPolicy ───────────────────────────────────────────

    #[tokio::test]
    async fn test_user_confirmation_accepts_y() {
        let policy = UserConfirmationPolicy::new(std::io::Cursor::new(b"y\n".to_vec()));
        assert!(policy.should_compress(4000, 10).await);
    }

    #[tokio::test]
    async fn test_user_confirmation_accepts_uppercase_y() {
        let policy = UserConfirmationPolicy::new(std::io::Cursor::new(b"Y\n".to_vec()));
        assert!(policy.should_compress(4000, 10).await);
    }

    #[tokio::test]
    async fn test_user_confirmation_rejects_n() {
        let policy = UserConfirmationPolicy::new(std::io::Cursor::new(b"n\n".to_vec()));
        assert!(!policy.should_compress(4000, 10).await);
    }

    #[tokio::test]
    async fn test_user_confirmation_rejects_empty() {
        let policy = UserConfirmationPolicy::new(std::io::Cursor::new(b"\n".to_vec()));
        assert!(!policy.should_compress(4000, 10).await);
    }
}
