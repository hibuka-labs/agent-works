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
            false,
            vec![],
        )
        .await
        .expect("spawn child");
    assert_eq!(path, "root/worker");

    let agents = ma.list_agents();
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0].agent_path, "root/worker");
    // No task sent yet → task is None.
    assert_eq!(agents[0].task, None);

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

    // The assigned task is recorded and surfaced by list_agents.
    let agents = ma.list_agents();
    assert_eq!(agents[0].task.as_deref(), Some("do the thing"));

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
            false,
            None,
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

// ── try_wait (non-blocking) ──

#[tokio::test(flavor = "multi_thread")]
async fn test_try_wait_pending_when_no_result() {
    let ma = make_ma_runtime();

    let _path = ma
        .spawn_child(
            "worker",
            "child system prompt".to_string(),
            0,
            false,
            vec![],
        )
        .await
        .expect("spawn child");

    // No task sent yet → try_wait should return "pending"
    let result = ma.try_wait(Some("root/worker"));
    assert_eq!(result.status, "pending");
    assert!(result.result.is_none());
    assert_eq!(result.agent_path.as_deref(), Some("root/worker"));

    // Clean up
    ma.cancel_all();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_try_wait_returns_result_after_post() {
    use crate::multi_agent::mailbox::{MailboxResult, MailboxStatus};

    let ma = make_ma_runtime();

    let path = ma
        .spawn_child(
            "worker",
            "child system prompt".to_string(),
            0,
            false,
            vec![],
        )
        .await
        .expect("spawn child");

    // Manually post a result to the mailbox (simulating child completion)
    ma.mailbox().post_result(MailboxResult {
        agent_path: crate::multi_agent::path::AgentPath::parse("root/worker").unwrap(),
        status: MailboxStatus::Ok,
        result: Some("done!".to_string()),
        denied_tools: vec![],
    });

    // try_wait should return the result immediately
    let result = ma.try_wait(Some("root/worker"));
    assert_eq!(result.status, "ok");
    assert_eq!(result.result.as_deref(), Some("done!"));
    assert_eq!(result.agent_path.as_deref(), Some("root/worker"));
    assert!(!result.has_more);

    // Second call should return "pending" (result was consumed)
    let result2 = ma.try_wait(Some("root/worker"));
    assert_eq!(result2.status, "pending");

    ma.cancel_all();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_try_wait_any_returns_pending_when_empty() {
    let ma = make_ma_runtime();

    // No children at all → try_wait(None) should return "pending"
    let result = ma.try_wait(None);
    assert_eq!(result.status, "pending");
    assert!(result.result.is_none());
    assert!(result.agent_path.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_try_wait_invalid_path_returns_error() {
    let ma = make_ma_runtime();

    let result = ma.try_wait(Some("worker"));
    assert_eq!(result.status, "error");
    assert!(
        result
            .result
            .as_deref()
            .unwrap()
            .contains("invalid agent path")
    );
}

// ── try_wait: closed after agent gone ──

#[tokio::test(flavor = "multi_thread")]
async fn test_try_wait_closed_after_agent_gone() {
    let ma = make_ma_runtime();

    let path = ma
        .spawn_child("worker", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn");

    // Send task, wait for completion, then close
    ma.send_task(&path, "do it".to_string(), false).unwrap();
    let result = ma.wait_for_result(Some("root/worker"), 2000).await;
    assert_eq!(result.status, "ok");

    ma.close_agent("root/worker").unwrap();

    // Wait for the child task to finish and clean up (ChildCleanup::drop
    // removes the registry entry). Without this, the agent is still in the
    // registry and try_wait returns "pending".
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // After close + cleanup + result consumed, try_wait should see the agent
    // is gone (mailbox empty + registry empty → "closed")
    let result = ma.try_wait(Some("root/worker"));
    assert_eq!(result.status, "closed");
}

// ── try_wait: any with multiple children ──

#[tokio::test(flavor = "multi_thread")]
async fn test_try_wait_any_returns_first_available() {
    use crate::multi_agent::mailbox::{MailboxResult, MailboxStatus};

    let ma = make_ma_runtime();

    ma.spawn_child("a", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn a");
    ma.spawn_child("b", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn b");

    // Post result for "b" only
    ma.mailbox().post_result(MailboxResult {
        agent_path: crate::multi_agent::path::AgentPath::parse("root/b").unwrap(),
        status: MailboxStatus::Ok,
        result: Some("b done".to_string()),
        denied_tools: vec![],
    });

    // try_wait(None) should return b's result
    let result = ma.try_wait(None);
    assert_eq!(result.status, "ok");
    assert_eq!(result.agent_path.as_deref(), Some("root/b"));
    assert_eq!(result.result.as_deref(), Some("b done"));

    // Second call should return pending (a has no result yet)
    let result2 = ma.try_wait(None);
    assert_eq!(result2.status, "pending");

    ma.cancel_all();
}

// ── blocking wait_for_result ──

#[tokio::test(flavor = "multi_thread")]
async fn test_wait_for_result_basic() {
    let ma = make_ma_runtime();

    ma.spawn_child("w", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn");

    ma.send_task("root/w", "go".to_string(), false).unwrap();

    // Blocking wait should return the result from the stub stream
    let result = ma.wait_for_result(Some("root/w"), 5000).await;
    assert_eq!(result.status, "ok");
    assert_eq!(result.result.as_deref(), Some("child ok"));
    assert_eq!(result.agent_path.as_deref(), Some("root/w"));
    assert!(!result.has_more);

    ma.cancel_all();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_wait_for_result_any() {
    let ma = make_ma_runtime();

    ma.spawn_child("x", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn");

    ma.send_task("root/x", "go".to_string(), false).unwrap();

    // Wait for ANY agent (no filter)
    let result = ma.wait_for_result(None, 5000).await;
    assert_eq!(result.status, "ok");
    assert_eq!(result.agent_path.as_deref(), Some("root/x"));

    ma.cancel_all();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_wait_for_result_timeout() {
    let ma = make_ma_runtime();

    ma.spawn_child("slow", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn");

    // Don't send any task — wait should timeout
    let result = ma.wait_for_result(Some("root/slow"), 100).await;
    assert_eq!(result.status, "timeout");
    assert!(result.result.is_none());

    ma.cancel_all();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_wait_for_result_has_more() {
    use crate::multi_agent::mailbox::{MailboxResult, MailboxStatus};

    let ma = make_ma_runtime();

    ma.spawn_child("a", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn a");
    ma.spawn_child("b", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn b");

    // Post results for both
    ma.mailbox().post_result(MailboxResult {
        agent_path: crate::multi_agent::path::AgentPath::parse("root/a").unwrap(),
        status: MailboxStatus::Ok,
        result: Some("a done".to_string()),
        denied_tools: vec![],
    });
    ma.mailbox().post_result(MailboxResult {
        agent_path: crate::multi_agent::path::AgentPath::parse("root/b").unwrap(),
        status: MailboxStatus::Ok,
        result: Some("b done".to_string()),
        denied_tools: vec![],
    });

    // Wait for specific — has_more should be true (one result still pending)
    let result = ma.wait_for_result(Some("root/a"), 2000).await;
    assert_eq!(result.status, "ok");
    assert!(result.has_more, "should report more results pending");

    // Consume the second
    let result2 = ma.wait_for_result(Some("root/b"), 2000).await;
    assert_eq!(result2.status, "ok");
    assert!(!result2.has_more);

    ma.cancel_all();
}

// ── delivery guarantees: status truth-telling + close notifications ──
// Regression tests for session 20260903_2438d139 (pi's report was silently
// lost and list_agents reported "running" for a child that had delivered).

/// After a child delivers its result the registry must report `done`, not
/// `running`. list_agents is the parent's only window into child progress;
/// a stale "running" made the parent conclude the children were stuck, do
/// the work itself, and close them mid-flight.
#[tokio::test(flavor = "multi_thread")]
async fn test_list_agents_shows_done_after_result_delivered() {
    let ma = make_ma_runtime();
    ma.spawn_child("w", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn");
    ma.send_task("root/w", "task".to_string(), false).unwrap();

    let result = ma.wait_for_result(Some("root/w"), 2000).await;
    assert_eq!(result.status, "ok");

    let agents = ma.list_agents();
    assert_eq!(
        agents[0].status, "done",
        "a child that delivered its result must not report running"
    );
    ma.cancel_all();
}

/// Closing an idle child must surface a `closed` event on the watcher
/// channel. ChildCleanup posts Closed and then unregisters in the same
/// synchronous poll — if unregister discarded the queued result, the watcher
/// could never observe a close (all 4 close notifications were lost in the
/// regression session).
#[tokio::test(flavor = "multi_thread")]
async fn test_close_idle_child_delivers_closed_event() {
    let ma = make_ma_runtime();
    ma.spawn_child("w", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn");
    ma.send_task("root/w", "task".to_string(), false).unwrap();
    let result = ma.wait_for_result(Some("root/w"), 2000).await;
    assert_eq!(result.status, "ok");

    // Watcher starts after the Ok was drained, so the first event it can
    // possibly deliver is the close notification — as Progress (a lone
    // Closed never wakes the parent with a batch).
    let (_handle, mut rx) = ma.start_watcher();

    let close = ma.close_agent("root/w").unwrap();
    assert!(close.closed);

    let cr = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for closed event")
        .expect("watcher channel closed");
    match cr {
        ChildResultEvent::Progress {
            agent_path, status, ..
        } => {
            assert_eq!(agent_path, "root/w");
            assert_eq!(status, "closed");
        }
        other => panic!("expected Progress(closed), got {other:?}"),
    }
}

/// Session 20260904_c6559510 regression: with a second task QUEUED, the
/// first task's post must not expose a phantom-idle registry state. The old
/// protocol (Done at every post, nothing at dequeue) let the watcher's
/// quiescence check pass mid-queue, firing the batch early and delivering
/// the remaining results as one wake-up per straggler. Now the batch fires
/// exactly once, after the LAST task's post, carrying every report.
#[tokio::test(flavor = "multi_thread")]
async fn test_queued_task_does_not_fire_premature_batch() {
    let ma = make_ma_runtime();
    let (_handle, mut rx) = ma.start_watcher();

    ma.spawn_child("w", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn");
    // Queue two tasks back-to-back: task two sits in the channel while
    // task one executes.
    ma.send_task("root/w", "task one".to_string(), false)
        .unwrap();
    ma.send_task("root/w", "task two".to_string(), false)
        .unwrap();

    // Collect events until the batch arrives. Progress ordering is free
    // (detached summaries), but the FIRST batch must already carry both
    // reports — a 1-report batch means quiescence fired while task two
    // was still pending.
    let mut batches = Vec::new();
    loop {
        let cr = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out waiting for watcher events")
            .expect("watcher channel closed");
        if let ChildResultEvent::Batch { reports } = cr {
            batches.push(reports);
            break;
        }
    }
    assert_eq!(batches.len(), 1);
    assert_eq!(
        batches[0].len(),
        2,
        "first batch must carry both queued-task reports"
    );

    // Nothing may follow but stray Progress events — no second batch.
    while let Ok(Some(cr)) =
        tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await
    {
        assert!(
            !matches!(cr, ChildResultEvent::Batch { .. }),
            "straggler batch after the flush: {:?}",
            cr
        );
    }
    ma.cancel_all();
}

/// A child closed mid-task still finishes its in-flight work (cancel is only
/// honored between tasks). Its result must survive the close — pi's report
/// was lost exactly this way: task completed after close, result and Closed
/// both eaten by the unregister discard.
#[tokio::test(flavor = "multi_thread")]
async fn test_close_running_child_result_still_delivered() {
    let ma = make_ma_runtime_with(Arc::new(DelayedStub));
    let (_handle, mut rx) = ma.start_watcher();

    ma.spawn_child("w", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("spawn");
    ma.send_task("root/w", "task".to_string(), false).unwrap();

    // Let the child claim the task and enter the delayed LLM call.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let close = ma.close_agent("root/w").unwrap();
    assert!(close.closed, "child was mid-task, close must land");

    // The late task result (the report pi lost) must be delivered twice
    // over: once as Progress and once inside the fan-in batch. The summary
    // no longer gates the batch (session 20260903_d8fc41dc — detached
    // summaries), so the two events race; accept either order.
    let mut saw_progress_ok = false;
    let mut saw_batch = false;
    for _ in 0..2 {
        let cr = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for late result")
            .expect("watcher channel closed");
        match cr {
            ChildResultEvent::Progress {
                agent_path, status, ..
            } => {
                assert_eq!(agent_path, "root/w");
                assert_eq!(status, "ok");
                saw_progress_ok = true;
            }
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1, "redundant Closed must not ride along");
                assert_eq!(reports[0].status, "ok");
                assert_eq!(reports[0].result.as_deref(), Some("late ok"));
                saw_batch = true;
            }
        }
    }
    assert!(saw_progress_ok, "late result must surface as Progress");
    assert!(saw_batch, "late result must surface in the batch");

    // If the Closed landed after the flush it may still surface as a lone
    // Closed Progress (if it landed in the same drain it was deduped) —
    // either way nothing may follow but that.
    if let Ok(Some(extra)) =
        tokio::time::timeout(std::time::Duration::from_millis(300), rx.recv()).await
    {
        match extra {
            ChildResultEvent::Progress { status, .. } => {
                assert_eq!(status, "closed", "only a lone Closed may follow the flush");
            }
            other => panic!("unexpected extra event {other:?}"),
        }
    }
}
