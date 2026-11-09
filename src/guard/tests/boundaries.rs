use super::*;

// ── Edge case: no scene flags ──

#[tokio::test]
async fn test_no_scene_flags_returns_complete() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, false, false, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "no scene flags → Complete, got: {:?}",
        decision
    );
}

// ═══════════════════════════════════════════════════════════════
// Phase 3: Boundary cases and coverage gaps
// ═══════════════════════════════════════════════════════════════

// ── Boundary: strikes = 0 (first occurrence) ──

#[tokio::test]
async fn test_reasoning_only_strikes_zero() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("reasoning") || msg.contains("tool call"),
                "nudge should mention the issue: {}",
                msg
            );
        }
        other => panic!("strikes=0 → Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_empty_response_strikes_zero() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("empty") || msg.contains("response"),
                "nudge should mention empty response: {}",
                msg
            );
        }
        other => panic!("strikes=0 → Continue, got: {:?}", other),
    }
}

// ── Boundary: strikes = threshold - 1 (just below) ──

#[tokio::test]
async fn test_reasoning_only_just_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    // default threshold = 3, so 2 should Continue
    let ctx = make_ctx(2, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { nudge: Some(_) }),
        "strikes=2, threshold=3 → Continue, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_empty_response_just_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 2, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { nudge: Some(_) }),
        "strikes=2, threshold=3 → Continue, got: {:?}",
        decision
    );
}

// ── Nudge message content verification ──

#[tokio::test]
async fn test_reasoning_only_nudge_content() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(1, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().reasoning_only_nudge);
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_empty_response_nudge_content() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 1, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().empty_response_nudge);
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_short_response_nudge_content() {
    let long_input = "a".repeat(200);
    let short_output = "ok";
    let config = DefaultGuardConfig {
        use_llm_judge: false, // disable judge to get nudge path
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx_with_text(&long_input, short_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert_eq!(msg, &DefaultGuardConfig::default().short_response_nudge);
        }
        other => panic!("expected Continue for short response, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_fail_error_message_reasoning_only() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(3, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Fail { error } => {
            assert!(
                error.contains("reasoning"),
                "fail error should mention reasoning: {}",
                error
            );
        }
        other => panic!("expected Fail, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_fail_error_message_empty_response() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 3, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Fail { error } => {
            assert!(
                error.contains("empty"),
                "fail error should mention empty: {}",
                error
            );
        }
        other => panic!("expected Fail, got: {:?}", other),
    }
}
