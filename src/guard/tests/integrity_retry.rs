//! Turn-as-transaction POLICY tests (works layer = judgment; base executes).
//!
//! The kernel supplies facts (integrity, visibility, retry count) and the
//! `RetryTurn` execution seam; THIS guard decides when to spend the silent
//! retry. Design: phimint notes/turn-transaction-design.md §6.

use super::*;
use agent_base::StreamIntegrity;

fn dirty_ctx() -> GuardCtx {
    let mut ctx = make_ctx(0, 0, false, false, false, false);
    ctx.integrity = StreamIntegrity {
        invalid_args: 1,
        orphan_args_bytes: 0,
    };
    ctx
}

/// Dirty + silent + budget left → RetryTurn.
#[tokio::test]
async fn dirty_silent_round_requests_retry() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let decision = guard.on_tool_call(&dirty_ctx()).await;
    assert!(
        matches!(decision, GuardDecision::RetryTurn),
        "expected RetryTurn, got {decision:?}"
    );
}

/// Text already reached the user → never discard, never retry.
#[tokio::test]
async fn visible_text_blocks_silent_retry() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = dirty_ctx();
    ctx.has_visible_text = true;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "expected Complete, got {decision:?}"
    );
}

/// Healthy stream → no retry.
#[tokio::test]
async fn clean_stream_never_retries() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, false, false, false);
    let decision = guard.on_tool_call(&ctx).await;
    assert!(matches!(decision, GuardDecision::Complete));
}

/// Budget (default 1 per round) exhausted → fall through to the ladder.
#[tokio::test]
async fn budget_exhausted_falls_through() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = dirty_ctx();
    ctx.silent_retries_in_round = 1;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "expected Complete past budget, got {decision:?}"
    );
}

/// `max_silent_retries: 0` disables the whole gate.
#[tokio::test]
async fn zero_budget_disables_gate() {
    let cfg = DefaultGuardConfig {
        max_silent_retries: 0,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(cfg);
    let decision = guard.on_tool_call(&dirty_ctx()).await;
    assert!(matches!(decision, GuardDecision::Complete));
}

/// Thinking is tolerated as "silent" by default; the knob tightens it.
#[tokio::test]
async fn thinking_visibility_knob() {
    let mut ctx = dirty_ctx();
    ctx.has_thinking = true;

    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::RetryTurn),
        "default tolerates thinking, expected RetryTurn, got {decision:?}"
    );

    let cfg = DefaultGuardConfig {
        thinking_is_visible: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(cfg);
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "thinking_is_visible must block discard, got {decision:?}"
    );
}

/// Integrity retry wins over RestoreThinking: discarding the corrupt batch
/// beats executing it, whatever the thinking state.
#[tokio::test]
async fn retry_wins_over_restore_thinking() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = dirty_ctx();
    ctx.thinking_disabled = true;
    ctx.original_thinking_enabled = true;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::RetryTurn),
        "expected RetryTurn over RestoreThinking, got {decision:?}"
    );
}
