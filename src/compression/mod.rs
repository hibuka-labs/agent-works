//! Context compression for long agent conversations.
//!
//! Long tool-heavy conversations balloon the message list sent to the LLM on
//! every turn, slowing each call down and eventually exceeding the context
//! window. This module provides a `CompressionMiddleware` (coming in Phase 4) that
//! observes the per-LLM-call message list via
//! [`Middleware::on_pre_llm`](agent_base::Middleware::on_pre_llm) and, once
//! estimated tokens exceed a configurable threshold, summarises the *earlier*
//! portion into a compact handoff message.
//!
//! # Strategy
//!
//! The compressor uses a **hybrid retention** approach:
//!
//! 1. **System prompt** — always preserved verbatim.
//! 2. **Recent N messages** — kept verbatim (including tool results), placed at
//!    the end to leverage the model's recency bias.
//! 3. **Older history** — condensed into a single handoff summary by an LLM
//!    call, cached via a stable-prefix hash to avoid redundant re-summarisation.
//!
//! This balances information preservation (recent tool data stays intact) with
//! token efficiency (older context is compressed).
//!
//! # Architecture
//!
//! ```text
//! session.chat_messages().to_vec()          ← full history clone
//!     ↓
//! CompressionMiddleware::on_pre_llm()       ← gate → split → cache check → summarise → assemble
//!     ↓
//! [system prompt] + [summary] + [recent N]  ← compressed copy for this LLM call only
//! ```
//!
//! The session's stored history is **never** modified by automatic compression;
//! only the per-call message copy is trimmed. The `/compact` CLI command can
//! optionally write the compressed form back to the session.
//!
//! # Feature gate
//!
//! This module is behind the `compression` Cargo feature.

mod config;
mod filter;

pub use config::CompressionConfig;
pub use filter::{SUMMARY_PREFIX, is_summary_message, split_system_prompt};
