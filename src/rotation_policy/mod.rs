//! Token-budget context management: window rotation instead of summarization.
//!
//! The sibling strategy to [`crate::compression`]. Where compression rewrites
//! the *per-call* message copy via LLM summarization, this strategy lets the
//! model self-manage state through private `notes`/`history` tools and, when
//! the budget is exhausted, rotates to a **fresh context window**: the old
//! messages are archived by the application, and a new window starts from a
//! fixed base (system prompt + window info + guidance + seed).
//!
//! # Work-room semantics
//!
//! The configured budget is **work room** — conversation space ABOVE the
//! window's fixed base — not the total window size. The reset point is
//! `base + work_budget + buffer`, so any positive budget produces a window
//! that fits. This makes the old "budget below system-prompt overhead"
//! failure class structurally impossible: a reset can never loop, because
//! the fresh window always starts below the threshold that triggers resets.
//!
//! # Phases (per window)
//!
//! ```text
//! work used: 0 ──────────── work_budget×0.8 ── work_budget ── +buffer
//!              normal         ① reminder       ② fallback     ③ reset
//!                             (save to notes)  (last turn)    (archive + fresh window)
//! ```
//!
//! # Layering
//!
//! This module is a **pure strategy**: no LLM calls, no runtime, no storage
//! I/O. [`TokenBudgetCore`] owns the decision state machine and message
//! assembly; the application implements
//! [`ContextCompaction`](agent_base::ContextCompaction) on a thin shell that
//! performs the I/O (archiving, notes lookup) at the [`TokenBudgetAction::Reset`]
//! point. agent-base stays strategy-free: contract + primitives only.
//!
//! # Contract invariants (enforced by agent-base, satisfied here)
//!
//! The engine validates that a compaction's output can be sent to an LLM:
//! at least one non-System/non-Custom message must remain (System maps to
//! the top-level `system` parameter — an all-System window is an HTTP 400).
//! This strategy satisfies it via [`TokenBudgetConfig::seed_message`], a
//! user turn that also gives the model something to act on when
//! reconstructing state after the rotation.

mod config;
mod core;

pub use config::{
    DEFAULT_SEED_MESSAGE, TokenBudgetConfig, TokenBudgetState, build_context_window_info,
    build_fallback_message, build_reminder_message, token_budget_base_overhead,
};
pub use core::{TokenBudgetAction, TokenBudgetCore};
