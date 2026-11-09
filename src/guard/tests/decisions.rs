use super::*;

// ── GuardDecision pattern matching ──

#[test]
fn test_guard_decision_continue_with_nudge() {
    let d = GuardDecision::Continue {
        nudge: Some("test nudge".to_string()),
    };
    match &d {
        GuardDecision::Continue { nudge } => {
            assert_eq!(nudge.as_deref(), Some("test nudge"));
        }
        _ => panic!("expected Continue"),
    }
}

#[test]
fn test_guard_decision_continue_without_nudge() {
    let d = GuardDecision::Continue { nudge: None };
    match &d {
        GuardDecision::Continue { nudge } => {
            assert!(nudge.is_none());
        }
        _ => panic!("expected Continue"),
    }
}

#[test]
fn test_guard_decision_complete() {
    let d = GuardDecision::Complete;
    assert!(matches!(d, GuardDecision::Complete));
}

#[test]
fn test_guard_decision_fail() {
    let d = GuardDecision::Fail {
        error: "something went wrong".to_string(),
    };
    match &d {
        GuardDecision::Fail { error } => {
            assert_eq!(error, "something went wrong");
        }
        _ => panic!("expected Fail"),
    }
}

// ── NoopGuard tests ──
// Note: NoopGuard returns Fail for reasoning_only/empty_response (safety),
// and Complete for text_only and normal scenarios.

#[tokio::test]
async fn test_noop_guard_normal_complete() {
    let guard = NoopGuard;
    let ctx = make_ctx(0, 0, false, false, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "NoopGuard normal → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_noop_guard_text_only_complete() {
    let guard = NoopGuard;
    let ctx = make_ctx(0, 0, false, false, false, true);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "NoopGuard text_only → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_noop_guard_reasoning_only_fails() {
    let guard = NoopGuard;
    let ctx = make_ctx(0, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "NoopGuard reasoning_only → Fail (safety), got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_noop_guard_empty_response_fails() {
    let guard = NoopGuard;
    let ctx = make_ctx(0, 0, false, false, true, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "NoopGuard empty_response → Fail (safety), got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_noop_guard_all_flags_reasoning_priority() {
    let guard = NoopGuard;
    let ctx = make_ctx(3, 3, true, true, true, true);
    let decision = guard.on_turn(&ctx).await;
    // reasoning_only || empty_response → Fail
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "NoopGuard all flags → Fail (reasoning/empty), got: {:?}",
        decision
    );
}

// ── Multiple scene flags set simultaneously ──

#[tokio::test]
async fn test_reasoning_only_takes_priority_over_empty() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    // Both reasoning_only and empty_response flags set
    let ctx = make_ctx(1, 1, false, true, true, false);

    let decision = guard.on_turn(&ctx).await;
    // on_turn checks is_reasoning_only first → should handle as reasoning_only
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().reasoning_only_nudge);
        }
        other => panic!(
            "reasoning_only takes priority, expected Continue, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_reasoning_only_takes_priority_over_text_only() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(1, 0, false, true, false, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().reasoning_only_nudge);
        }
        other => panic!(
            "reasoning_only takes priority, expected Continue, got: {:?}",
            other
        ),
    }
}

// ── Short response + judge: non-short response fallback path ──

#[tokio::test]
async fn test_text_only_non_short_with_tools_judge_not_done() {
    // Response that's not short (> max_output=64) but < skip_threshold=256
    let medium_output = "a".repeat(100);
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "missing details"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("query", &medium_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("incomplete") && msg.contains("missing details"),
                "non-short + judge not done → Continue with reason: {}",
                msg
            );
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_text_only_non_short_with_tools_judge_done() {
    let medium_output = "a".repeat(100);
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": true,
        "reason": "task complete"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("query", &medium_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "non-short + judge done → Complete, got: {:?}",
        decision
    );
}

// ── Short response + skip_judge_when_input_too_large for short path ──

#[tokio::test]
async fn test_short_response_skip_judge_when_input_too_large() {
    let long_input = "a".repeat(10_001);
    let short_output = "42";
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "would be skipped"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text(&long_input, short_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "short response + input > 10k → skip judge → Complete, got: {:?}",
        decision
    );
}

// ── DisableThinking strategy tests ──────────────────────────────────────

#[tokio::test]
async fn test_disable_thinking_strategy_below_threshold() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        reasoning_only_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Below threshold → Continue with nudge
    let ctx = make_ctx(2, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { .. }),
        "below threshold → Continue, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_disable_thinking_strategy_at_threshold() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        reasoning_only_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // At threshold → DisableThinking
    let ctx = make_ctx(3, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::DisableThinking { .. }),
        "at threshold → DisableThinking, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_disable_thinking_strategy_already_disabled() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        reasoning_only_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Thinking already disabled but still reasoning-only → Fail
    let mut ctx = make_ctx(3, 0, false, true, false, false);
    ctx.thinking_disabled = true;
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "thinking disabled + reasoning-only → Fail, got: {:?}",
        decision
    );
}

// ── on_tool_call tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_on_tool_call_restores_thinking() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Thinking disabled + original enabled + tool call → RestoreThinking
    let mut ctx = make_ctx(0, 0, true, false, false, false);
    ctx.thinking_disabled = true;
    ctx.original_thinking_enabled = true;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::RestoreThinking),
        "thinking disabled + tool call → RestoreThinking, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_on_tool_call_no_restore_when_original_disabled() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Thinking disabled + original disabled + tool call → Complete
    let mut ctx = make_ctx(0, 0, true, false, false, false);
    ctx.thinking_disabled = true;
    ctx.original_thinking_enabled = false;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "original disabled + tool call → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_on_tool_call_no_restore_when_thinking_enabled() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Thinking enabled + tool call → Complete
    let mut ctx = make_ctx(0, 0, true, false, false, false);
    ctx.thinking_disabled = false;
    ctx.original_thinking_enabled = true;
    let decision = guard.on_tool_call(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "thinking enabled + tool call → Complete, got: {:?}",
        decision
    );
}

// ── Fail strategy tests (default behavior) ──────────────────────────────

#[tokio::test]
async fn test_fail_strategy_at_threshold() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::Fail,
        reasoning_only_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // At threshold → Fail
    let ctx = make_ctx(3, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "at threshold → Fail, got: {:?}",
        decision
    );
}

// ── Empty response tests with DisableThinking ───────────────────────────

#[tokio::test]
async fn test_empty_response_with_disable_thinking_strategy() {
    let config = DefaultGuardConfig {
        reasoning_only_action: ReasoningOnlyAction::DisableThinking,
        empty_response_max_strikes: 3,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // Empty response at threshold → Fail (even with DisableThinking strategy)
    let ctx = make_ctx(0, 3, false, false, true, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "empty response at threshold → Fail, got: {:?}",
        decision
    );
}
