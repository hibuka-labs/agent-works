use super::*;

// ── lifecycle: spawn / send / wait / close ──

#[tokio::test(flavor = "multi_thread")]
async fn test_spawn_send_task_wait_close_lifecycle() {
    let ma = make_ma_runtime();

    let path = ma
        .spawn_child(
            "worker",
            "child system prompt".to_string(),
            0,
            0,
            false,
            vec![],
        )
        .await
        .expect("spawn child");
    assert_eq!(path, "root/worker");

    let agents = ma.list_agents();
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0].agent_path, "root/worker");

    // send_message posts a pending message (no execution trigger).
    assert!(
        ma.send_message("root/worker", "heads up".to_string())
            .unwrap()
    );

    // send_task triggers execution; the child completes via the stub stream.
    assert!(
        ma.send_task("root/worker", "do the thing".to_string(), false)
            .unwrap()
    );

    let result = ma.wait_for_result(Some("root/worker"), 2000).await;
    assert_eq!(result.status, "ok");
    assert_eq!(result.result.as_deref(), Some("child ok"));

    let close = ma.close_agent("root/worker").unwrap();
    assert!(close.closed);
    assert_eq!(close.message, "agent closed");

    // Closing a second time reports not found.
    let close2 = ma.close_agent("root/worker").unwrap();
    assert!(!close2.closed);
    assert_eq!(close2.message, "agent not found");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_spawn_child_with_history_defaults_to_none() {
    let ma = make_ma_runtime();
    // No session manager set → fork_history resolves to empty; spawn still succeeds.
    let path = ma
        .spawn_child_with_history(
            "w2",
            "prompt".to_string(),
            0,
            0,
            false,
            None,
            &agent_base::SessionId::new(0),
        )
        .await
        .expect("spawn with history");
    assert_eq!(path, "root/w2");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_error_paths() {
    let ma = make_ma_runtime();

    // Valid path, unknown agent → "agent not found".
    assert_eq!(
        ma.send_task("root/ghost", "x".to_string(), false)
            .unwrap_err(),
        "agent not found"
    );

    // Invalid paths (must start with "root", no empty segments).
    assert!(ma.send_message("worker", "x".to_string()).is_err());
    assert!(ma.send_message("", "x".to_string()).is_err());

    // wait_for_result with invalid path.
    let r = ma.wait_for_result(Some("worker"), 10).await;
    assert_eq!(r.status, "error");

    // wait_for_result with no results times out.
    let r2 = ma.wait_for_result(None, 50).await;
    assert_eq!(r2.status, "timeout");

    // cancel_all is a no-op when no children are running.
    ma.cancel_all();
}
