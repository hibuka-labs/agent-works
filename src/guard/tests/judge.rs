use super::*;

// ── LLM judge tests ──

#[tokio::test]
async fn test_text_only_judge_says_done() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": true,
        "reason": "task is complete"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("what is 2+2?", "The answer is 4.", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "judge says done → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_text_only_judge_says_not_done() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "only answered part of the question"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text(
        "list all files and explain each",
        "Here are the files:",
        true,
        true,
    );

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("incomplete"),
                "nudge should mention incomplete: {}",
                msg
            );
            assert!(
                msg.contains("only answered part"),
                "nudge should include judge reason: {}",
                msg
            );
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_text_only_judge_reason_propagated_to_nudge() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "missing file list and explanation"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("list files and explain", "here are the files:", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("missing file list and explanation"),
                "nudge should include judge reason: {}",
                msg
            );
        }
        other => panic!("expected Continue, got: {:?}", other),
    }
}

// ── Awaiting-human park: a run parked on a human answer is waiting, not incomplete ──

/// Session 20261022_bb8c2dc3: `canvas_push` parked on a pick, the completion
/// judge saw the whole multi-step task and nudged "continue", and the model
/// decided the user's design choices for them. A parked run must complete
/// without the judge being consulted — a judge that says "not done" here would
/// reproduce the steamroll exactly.
#[tokio::test]
async fn awaiting_human_park_completes_even_when_judge_would_nudge() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "four design steps are still unanswered"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let mut ctx = make_ctx_with_text(
        "design a dashboard and let me pick each option",
        "The screen is up — click an option on the canvas and I'll continue.",
        true,
        true,
    );
    ctx.awaiting_human = true;

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "parked-on-human must Complete even when the judge would nudge, got: {:?}",
        decision
    );
}

/// Degenerate turns still fail while a pick is pending — a reasoning-only
/// response is broken regardless of the park.
#[tokio::test]
async fn awaiting_human_does_not_mask_degenerate_turns() {
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = make_ctx(1, 0, true, true, false, false);
    ctx.awaiting_human = true;

    let decision = guard.on_turn(&ctx).await;
    assert!(
        !matches!(decision, GuardDecision::Complete),
        "reasoning-only must not Complete just because a pick is pending, got: {:?}",
        decision
    );
}

// ── Short response detection tests ──

#[tokio::test]
async fn test_text_only_short_response_detected_no_tools() {
    let long_input = "a".repeat(200);
    let short_output = "done";
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let ctx = make_ctx_with_text(&long_input, short_output, false, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(msg.contains("incomplete"), "short response nudge: {}", msg);
        }
        other => panic!("expected Continue for short response, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_text_only_short_response_with_judge_done() {
    let long_input = "a".repeat(200);
    let short_output = "42";
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": true,
        "reason": "answer is correct"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text(&long_input, short_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "short response + judge done → Complete, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_text_only_after_rejected_tool_calls_never_judges_complete() {
    // Session 20260904_efad759c: the truncation guard rejected a spawn_agent
    // call, the model answered with a fabricated text-only success narrative,
    // and the completion judge — blind to the rejected tool_result — passed
    // it, ending the run with zero children spawned. last_tool_calls_invalid
    // must short-circuit the judge: no client here, so a Complete decision
    // could only come from the old (buggy) judge path failing open.
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let mut ctx = make_ctx_with_text(
        "analyze 3 projects with sub-agents",
        "已启动 4 个子 agent 分析，稍后汇总。",
        true,
        true,
    );
    ctx.last_tool_calls_invalid = true;

    let decision = guard.on_turn(&ctx).await;
    match decision {
        GuardDecision::Continue { nudge: Some(msg) } => {
            assert!(
                msg.contains("NOT executed"),
                "nudge must tell the model the tool call never executed, got: {msg}"
            );
        }
        other => panic!(
            "text-only after rejected tool calls must Continue with a re-issue \
             nudge, got: {other:?}"
        ),
    }
}

// ── Skip threshold tests ──

#[tokio::test]
async fn test_text_only_skip_threshold() {
    let long_output = "x".repeat(300);
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "would be incomplete but skipped"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text("query", &long_output, true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "response >= skip_threshold → Complete without calling judge, got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_text_only_skip_judge_when_input_too_large() {
    let long_input = "a".repeat(10_001);
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "would be incomplete but skipped"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);
    let ctx = make_ctx_with_text(&long_input, "short answer", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "input > 10k → skip judge → Complete, got: {:?}",
        decision
    );
}

// ── Judge timeout tests ──

#[tokio::test]
async fn test_text_only_judge_timeout_fail_open() {
    let judge_client = Arc::new(MockTimeoutClient::new());
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        judge_timeout_secs: 1,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short answer", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "timeout + fail_open → Complete (trust model), got: {:?}",
        decision
    );
}

#[tokio::test]
async fn test_text_only_judge_timeout_fail_closed() {
    let judge_client = Arc::new(MockTimeoutClient::new());
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        judge_timeout_secs: 1,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short answer", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "fail_closed timeout → Continue with verify message: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for timeout + fail_closed, got: {:?}",
            other
        ),
    }
}

// ── Judge error tests ──

#[tokio::test]
async fn test_text_only_judge_error_fail_closed() {
    let judge_client = Arc::new(MockErrorClient);
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "fail_closed error → Continue with verify message: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for judge error + fail_closed, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_text_only_judge_error_fail_open() {
    let judge_client = Arc::new(MockErrorClient);
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let ctx = make_ctx_with_text("query", "short", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "fail_open + judge error → Complete (trust model), got: {:?}",
        decision
    );
}

// ── Short response + judge combination tests ──

#[tokio::test]
async fn test_text_only_short_response_judge_timeout_fail_closed() {
    let judge_client = Arc::new(MockTimeoutClient::new());
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        judge_timeout_secs: 1,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let long_input = "a".repeat(200);
    let ctx = make_ctx_with_text(&long_input, "42", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "short + timeout + fail_closed → Continue: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for short response + timeout + fail_closed, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_text_only_short_response_judge_error_fail_closed() {
    let judge_client = Arc::new(MockErrorClient);
    let config = DefaultGuardConfig {
        judge_fail_open: false,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let long_input = "a".repeat(200);
    let ctx = make_ctx_with_text(&long_input, "42", true, true);

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("Cannot verify"),
                "short + error + fail_closed → Continue: {}",
                msg
            );
        }
        other => panic!(
            "expected Continue for short response + judge error + fail_closed, got: {:?}",
            other
        ),
    }
}

#[tokio::test]
async fn test_text_only_short_response_judge_error_fail_open() {
    let judge_client = Arc::new(MockErrorClient);
    let config = DefaultGuardConfig {
        judge_fail_open: true,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);
    let long_input = "a".repeat(200);
    let ctx = make_ctx_with_text(&long_input, "42", true, true);

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "short + error + fail_open → Complete, got: {:?}",
        decision
    );
}

// ── Multi-turn context tests ──

#[tokio::test]
async fn test_text_only_judge_with_multiple_user_messages() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": false,
        "reason": "agent has not finished the task yet"
    })));
    let guard = DefaultGuard::with_llm_client(DefaultGuardConfig::default(), judge_client);

    let mut ctx = make_ctx_with_text("继续", "I'll continue working on it.", true, true);
    ctx.all_user_inputs = vec![
        "帮我分析一下这个 bug 的根因".to_string(),
        "好的，那你帮我修复一下".to_string(),
        "继续".to_string(),
    ];

    let decision = guard.on_turn(&ctx).await;
    match &decision {
        GuardDecision::Continue { nudge } => {
            let msg = nudge.as_ref().unwrap();
            assert!(
                msg.contains("incomplete"),
                "judge should see history and say incomplete: {}",
                msg
            );
        }
        other => panic!("expected Continue (judge says not done), got: {:?}", other),
    }
}

#[tokio::test]
async fn test_text_only_judge_respects_recent_user_count() {
    let judge_client = Arc::new(MockJudgeClient::new(serde_json::json!({
        "done": true,
        "reason": "task complete"
    })));
    let config = DefaultGuardConfig {
        recent_user_count: 2,
        ..DefaultGuardConfig::default()
    };
    let guard = DefaultGuard::with_llm_client(config, judge_client);

    let mut ctx = make_ctx_with_text("继续", "done", true, true);
    ctx.all_user_inputs = vec![
        "msg1".to_string(),
        "msg2".to_string(),
        "msg3".to_string(),
        "msg4".to_string(),
        "继续".to_string(),
    ];

    let decision = guard.on_turn(&ctx).await;
    assert!(
        matches!(decision, GuardDecision::Complete),
        "judge says done → Complete, got: {:?}",
        decision
    );
}
