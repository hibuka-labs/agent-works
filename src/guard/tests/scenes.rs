use super::*;

// ── Reasoning-only tests ──

#[tokio::test]
async fn test_reasoning_only_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(1, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(
        decision,
        GuardDecision::Continue { nudge: Some(_) }
    ));
}

#[tokio::test]
async fn test_reasoning_only_at_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(3, 0, false, true, false, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Fail { .. }));
}

#[tokio::test]
async fn test_reasoning_only_with_tools_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = make_ctx(1, 0, false, true, false, false);
    ctx.available_tools = vec!["echo".to_string()];
    ctx.run_has_actions = true;

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { nudge: Some(_) }),
        "reasoning_only with tools, below threshold → Continue, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_reasoning_only_with_tools_at_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = make_ctx(3, 0, false, true, false, false);
    ctx.available_tools = vec!["echo".to_string()];
    ctx.run_has_actions = true;

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Fail { .. }),
        "reasoning_only with tools, at threshold → Fail, got: {:?}",
        decision
    );
}

// ── Empty response tests ──

#[tokio::test]
async fn test_empty_response_below_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 1, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(
        decision,
        GuardDecision::Continue { nudge: Some(_) }
    ));
}

#[tokio::test]
async fn test_empty_response_at_threshold() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 3, false, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Fail { .. }));
}

#[tokio::test]
async fn test_empty_response_with_tool_calls() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 1, true, false, true, false);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Continue { nudge: Some(_) }),
        "empty response with tools → Continue (nudge), got: {:?}",
        decision
    );
}

// ── Text-only tests ──

#[tokio::test]
async fn test_text_only_without_tool_calls() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx(0, 0, false, false, false, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Complete));
}

#[tokio::test]
async fn test_text_only_with_tool_calls_no_llm_client() {
    let config = DefaultGuardConfig {
        use_llm_judge: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx(0, 0, true, false, false, true);

    // No LLM client, judge_fail_open=false (default), should force continue
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(
        decision,
        GuardDecision::Continue { nudge: Some(_) }
    ));
}

#[tokio::test]
async fn test_text_only_with_tool_calls_no_llm_client_fail_open() {
    let config = DefaultGuardConfig {
        use_llm_judge: true,
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx(0, 0, true, false, false, true);

    // No LLM client, judge_fail_open=true, should trust model (Complete)
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Complete));
}

#[tokio::test]
async fn test_text_only_with_tool_calls_judge_disabled() {
    let config = DefaultGuardConfig {
        use_llm_judge: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx(0, 0, true, false, false, true);

    // LLM judge disabled, should Complete
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Complete));
}

#[tokio::test]
async fn test_text_only_no_tools_judge_enabled_no_client() {
    let config = DefaultGuardConfig {
        use_llm_judge: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let long_output = "x".repeat(300);
    let ctx = make_ctx_with_text("query", &long_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "no tools + judge enabled + no client → Complete (skips judge), got: {:?}",
        decision
    );
}

// ── Custom config tests ──

#[tokio::test]
async fn test_custom_config() {
    let config = DefaultGuardConfig {
        reasoning_only_max_strikes: 5,
        empty_response_max_strikes: 2,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // reasoning-only: 4 < 5, should continue
    let ctx = make_ctx(4, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(
        decision,
        GuardDecision::Continue { nudge: Some(_) }
    ));

    // empty response: 2 >= 2, should fail
    let ctx = make_ctx(0, 2, false, false, true, false);
    let decision = guard.on_turn(&ctx).await;
    assert!(matches!(decision, GuardDecision::Fail { .. }));
}
