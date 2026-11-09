use super::*;

// ── Config default values ──

#[test]
fn test_default_guard_config_values() {
    let config = DefaultGuardConfig::default();
    assert_eq!(config.reasoning_only_max_strikes, 3);
    assert_eq!(config.empty_response_max_strikes, 3);
    assert!(config.use_llm_judge);
    assert_eq!(config.judge_timeout_secs, 10);
    assert_eq!(config.judge_skip_threshold, 256);
    assert!(!config.judge_fail_open);
    assert!(config.detect_short_response);
    assert_eq!(config.short_response_min_input, 128);
    assert_eq!(config.short_response_max_output, 64);
    assert_eq!(config.recent_user_count, 5);
}

// ── Text-only: tools present + judge disabled ──

#[tokio::test]
async fn test_text_only_with_tools_judge_disabled_returns_complete() {
    let config = DefaultGuardConfig {
        use_llm_judge: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);
    let ctx = make_ctx(0, 0, true, false, false, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "tools + judge disabled → Complete, got: {:?}",
        decision
    );
}

// ── Text-only: short response without tools → nudge ──

#[tokio::test]
async fn test_text_only_short_response_without_tools_nudge() {
    let long_input = "a".repeat(200);
    let short_output = "done";
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&long_input, short_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            // Without tools, uses short_response_nudge
            assert!(
                msg.contains("incomplete"),
                "should use short_response_nudge: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for short response without tools, got: {:?}",
            other
        ),
    }
}

// ── Text-only: input shorter than output (not short response) ──

#[tokio::test]
async fn test_text_only_input_shorter_than_output_not_short() {
    let short_input = "hi";
    let long_output = "a".repeat(200);
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(short_input, &long_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // input_len (2) < output_len (200), and input_len (2) < min_input (128)
    // → not short response → no tools → Complete
    assert!(
        matches!(decision, GuardDecision::Complete),
        "short input + long output + no tools → Complete, got: {:?}",
        decision
    );
}

// ── Text-only: input just at short_response_min_input boundary ──

#[tokio::test]
async fn test_text_only_short_response_exact_min_input() {
    // Exactly at min_input (128) — uses strict `>`, so 128 > 128 is false → NOT short
    let input = "a".repeat(128);
    let output = "x"; // < max_output (64), and input_len > output_len
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&input, output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // input_len=128 is NOT > min_input=128 (strict), so NOT short response
    // no tools → Complete
    assert!(
        matches!(decision, GuardDecision::Complete),
        "exact min_input (128 > 128 is false) → not short → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_text_only_short_response_just_above_min_input() {
    // One char above min_input — should trigger short response
    let input = "a".repeat(129);
    let output = "x";
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&input, output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // input_len=129 > min_input=128, output=1 < max_output=64, 129 > 1
    // → short response → no tools → nudge
    match &decision {
        GuardDecision::Continue { nudge } => {
            assert!(nudge.is_some(), "just above min_input → short → nudge");
        }
        other => panic!(
            "expected Continue for just above boundary, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_text_only_not_short_response_just_below_min_input() {
    // One char below min_input — should NOT trigger short response
    let input = "a".repeat(127);
    let output = "x";
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&input, output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // input_len=127 < min_input=128 → not short response → no tools → Complete
    assert!(
        matches!(decision, GuardDecision::Complete),
        "below min_input → not short → Complete, got: {:?}",
        decision
    );
}

// ── Text-only: output exactly at max_output boundary ──

#[tokio::test]
async fn test_text_only_short_response_output_at_max() {
    let input = "a".repeat(200);
    let output = "a".repeat(64); // exactly at max_output
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&input, &output, false, true);

    let decision = guard.on_turn(&ctx).await;
    // output_len=64 is NOT < max_output=64, so NOT short response
    // no tools → Complete
    assert!(
        matches!(decision, GuardDecision::Complete),
        "output at max_output → not short → Complete, got: {:?}",
        decision
    );
}

// ── Custom nudge messages ──

#[tokio::test]
async fn test_custom_nudge_messages() {
    let config = DefaultGuardConfig {
        reasoning_only_nudge: "custom reasoning nudge".to_string(),
        empty_response_nudge: "custom empty nudge".to_string(),
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::new(config);

    // reasoning_only
    let ctx = make_ctx(1, 0, false, true, false, false);
    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            assert_eq!(nudge.as_deref(), Some("custom reasoning nudge"));
        }
        other => panic!("expected Continue, got: {:?}", other),
    }

    // empty_response
    let ctx = make_ctx(0, 1, false, false, true, false);
    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            assert_eq!(nudge.as_deref(), Some("custom empty nudge"));
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}
