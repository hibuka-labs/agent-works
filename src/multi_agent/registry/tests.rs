use super::*;

fn test_config() -> MultiAgentConfig {
    MultiAgentConfig::enabled()
}

fn test_path(name: &str) -> AgentPath {
    AgentPath::root().join(name)
}

/// Register a worker and hand back the registry.
fn registered(config: MultiAgentConfig, name: &str) -> (AgentRegistry, AgentPath) {
    let mut reg = AgentRegistry::new(config);
    let path = test_path(name);
    reg.register(&path, 1).unwrap();
    (reg, path)
}

#[test]
fn derive_status_is_a_pure_function() {
    use AgentStatus::*;
    use ResultKind as R;
    let failed = R::Failed("boom".to_string());
    // Unregistered always wins.
    assert_eq!(derive_status(false, true, 3, None), Closed);
    assert_eq!(derive_status(false, false, 0, Some(&failed)), Closed);
    // In flight dominates queued work and the previous result.
    assert_eq!(derive_status(true, true, 0, Some(&failed)), Running);
    assert_eq!(derive_status(true, true, 5, None), Running);
    // Settled with nothing queued → the last result speaks.
    assert_eq!(derive_status(true, false, 0, None), Done);
    assert_eq!(derive_status(true, false, 0, Some(&R::Success)), Done);
    assert_eq!(derive_status(true, false, 0, Some(&failed)), Failed);
    assert_eq!(
        derive_status(true, false, 0, Some(&R::Cancelled)),
        Cancelled
    );
    // Work waiting, nothing executing → Queued (fresh work outranks a
    // stale failure — the failure is not sticky).
    assert_eq!(derive_status(true, false, 1, None), Queued);
    assert_eq!(derive_status(true, false, 9, Some(&failed)), Queued);
}

#[test]
fn status_names_are_wire_stable() {
    assert_eq!(AgentStatus::Queued.name(), "queued");
    assert_eq!(AgentStatus::Running.name(), "running");
    assert_eq!(AgentStatus::Done.name(), "done");
    assert_eq!(AgentStatus::Failed.name(), "failed");
    assert_eq!(AgentStatus::Cancelled.name(), "cancelled");
    assert_eq!(AgentStatus::Closed.name(), "closed");
    assert_eq!(AgentStatus::Paused.name(), "paused");
    assert_eq!(AgentStatus::Unknown.name(), "unknown");
}

#[test]
fn name_from_name_round_trips() {
    // The wire vocabulary is hand-synced in exactly one place: a TUI (or
    // any consumer) parses with `from_name`, so a new word degrades to
    // Unknown instead of mis-classifying, and known words never drift.
    for status in [
        AgentStatus::Queued,
        AgentStatus::Running,
        AgentStatus::Done,
        AgentStatus::Failed,
        AgentStatus::Cancelled,
        AgentStatus::Closed,
        AgentStatus::Paused,
        AgentStatus::Unknown,
    ] {
        assert_eq!(AgentStatus::from_name(status.name()), status);
    }
    // Unknown wire names are the conservative fallback: not terminal,
    // not alarming.
    assert_eq!(AgentStatus::from_name("bogus"), AgentStatus::Unknown);
    assert_eq!(AgentStatus::from_name(""), AgentStatus::Unknown);
}

#[test]
fn register_and_close() {
    let mut reg = AgentRegistry::new(test_config());
    let path = test_path("worker");

    assert!(reg.register(&path, 1).is_ok());
    assert_eq!(reg.count(), 1);
    assert!(reg.contains(&path));

    let entry = reg.get(&path).unwrap();
    // No Idle in the fact model: a registered agent with no queued work
    // and nothing in flight derives Done (see module docs).
    assert_eq!(entry.status(), AgentStatus::Done);
    assert_eq!(entry.depth, 1);
    // Fresh agent: no facts accumulated yet.
    assert_eq!(entry.queue_len, 0);
    assert!(!entry.in_flight);
    assert_eq!(entry.results_posted, 0);
    assert_eq!(entry.tool_calls, 0);
    assert!(entry.last_activity.is_none());

    let closed = reg.close(&path).unwrap();
    assert_eq!(closed.path, path);
    assert_eq!(reg.count(), 0);
    assert!(!reg.contains(&path));
}

#[test]
fn duplicate_register_fails() {
    let mut reg = AgentRegistry::new(test_config());
    let path = test_path("worker");

    assert!(reg.register(&path, 1).is_ok());
    assert_eq!(
        reg.register(&path, 1).unwrap_err(),
        SpawnError::AlreadyExists
    );
}

#[test]
fn max_agents_limit() {
    let config = MultiAgentConfig::with_limits(2, 1);
    let mut reg = AgentRegistry::new(config);

    assert!(reg.register(&test_path("a"), 1).is_ok());
    assert!(reg.register(&test_path("b"), 1).is_ok());
    assert_eq!(
        reg.register(&test_path("c"), 1).unwrap_err(),
        SpawnError::MaxAgentsReached { max: 2 }
    );
}

#[test]
fn depth_limit() {
    let config = MultiAgentConfig::with_limits(8, 1);
    let reg = AgentRegistry::new(config);

    // Depth 1 is allowed
    assert!(reg.can_spawn(1).is_ok());

    // Depth 2 exceeds limit
    assert_eq!(
        reg.can_spawn(2).unwrap_err(),
        SpawnError::DepthLimitReached {
            max: 1,
            attempted: 2
        }
    );
}

/// The full fact-driven lifecycle, including "queued behind a running
/// task" (the shape session 20260904_c6559510 got wrong with markers).
#[test]
fn facts_drive_the_whole_lifecycle() {
    let (mut reg, path) = registered(test_config(), "worker");

    // Freshly registered → Done (no Idle variant).
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Done);

    // Two tasks queued back-to-back: Queued, then still Queued.
    assert!(reg.note_enqueued(&path));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Queued);
    assert!(reg.note_enqueued(&path));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Queued);

    // First dequeue → Running; second task still queued.
    assert!(reg.note_dequeued(&path));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Running);
    assert!(reg.get(&path).unwrap().running_since.is_some());
    assert_eq!(reg.get(&path).unwrap().queue_len, 1);

    // First result posted (second task queued) → Queued.
    assert!(reg.note_posted(&path, ResultKind::Success));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Queued);
    assert!(reg.get(&path).unwrap().running_since.is_none());
    assert_eq!(reg.get(&path).unwrap().results_posted, 1);

    // Second dequeue → Running.
    assert!(reg.note_dequeued(&path));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Running);

    // Final post, queue empty → Done. This is the Done-before-post
    // invariant, now structural: the fact lands before post_result.
    assert!(reg.note_posted(&path, ResultKind::Success));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Done);
    assert_eq!(reg.get(&path).unwrap().results_posted, 2);

    // Done agents still count toward quota.
    assert_eq!(reg.count(), 1);
}

/// The failure half of the lifecycle: a failed result derives `Failed`
/// (never `Done` — that laundering is what panels showed green), its
/// error text rides the snapshot for ledger lines, and the outcome is
/// not sticky: the next task flips the agent back to `Running`.
#[test]
fn failed_result_derives_failed_until_the_next_result() {
    let (mut reg, path) = registered(test_config(), "worker");

    reg.note_enqueued(&path);
    reg.note_dequeued(&path);
    assert!(reg.note_posted(&path, ResultKind::Failed("exit 1".into())));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Failed);

    // The error text travels to snapshot consumers (ledger lines).
    let row = &reg.snapshot().agents[0];
    assert_eq!(row.status, "failed");
    assert_eq!(row.last_error.as_deref(), Some("exit 1"));

    // Re-task: the failure is not sticky, the new run speaks.
    reg.note_enqueued(&path);
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Queued);
    reg.note_dequeued(&path);
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Running);
    assert!(reg.note_posted(&path, ResultKind::Success));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Done);
    let row = &reg.snapshot().agents[0];
    assert_eq!(row.status, "done");
    assert!(row.last_error.is_none(), "a later success clears the error");
}

/// Cancelled is its own derived terminal — "not successful" is not the
/// same fact as "failed" (alarms must not mix causes).
#[test]
fn cancelled_result_derives_cancelled() {
    let (mut reg, path) = registered(test_config(), "worker");

    reg.note_enqueued(&path);
    reg.note_dequeued(&path);
    assert!(reg.note_posted(&path, ResultKind::Cancelled));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Cancelled);
    assert_eq!(reg.snapshot().agents[0].status, "cancelled");

    // A later success restores Done.
    reg.note_enqueued(&path);
    reg.note_dequeued(&path);
    assert!(reg.note_posted(&path, ResultKind::Success));
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Done);
}

#[test]
fn close_frees_quota() {
    let config = MultiAgentConfig::with_limits(1, 1);
    let mut reg = AgentRegistry::new(config);

    let path = test_path("worker");
    reg.register(&path, 1).unwrap();
    assert_eq!(reg.count(), 1);

    // Can't spawn another — quota full
    assert!(reg.can_spawn(1).is_err());

    // Close frees the slot
    reg.close(&path);
    assert_eq!(reg.count(), 0);
    assert!(reg.can_spawn(1).is_ok());
}

#[test]
fn list_sorted() {
    let mut reg = AgentRegistry::new(test_config());
    reg.register(&test_path("b"), 1).unwrap();
    reg.register(&test_path("a"), 1).unwrap();

    let list = reg.list();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].path.name(), "a");
    assert_eq!(list[1].path.name(), "b");
}

#[test]
fn tool_call_counter_is_monotonic() {
    // Session 20260903_0cf95e79 regression: list_agents showed a frozen
    // `tool_count: 9` (static inventory size) while the children actually
    // executed 100+ calls — the parent read it as "stuck". The executed
    // count must start at zero and only grow.
    let (mut reg, path) = registered(test_config(), "worker");

    for _ in 0..3 {
        assert!(reg.record_tool_call(&path));
    }
    assert_eq!(reg.get(&path).unwrap().tool_calls, 3);
    assert!(reg.get(&path).unwrap().last_activity.is_some());

    assert!(reg.record_tool_call(&path));
    assert_eq!(reg.get(&path).unwrap().tool_calls, 4);

    // Unknown paths are a no-op (event bridge may race cleanup).
    assert!(!reg.record_tool_call(&test_path("ghost")));
}

#[test]
fn touch_marks_activity_without_counting() {
    let (mut reg, path) = registered(test_config(), "worker");

    assert!(reg.touch(&path));
    assert!(reg.get(&path).unwrap().last_activity.is_some());
    assert_eq!(reg.get(&path).unwrap().tool_calls, 0);
    assert!(!reg.touch(&test_path("ghost")));
}

#[test]
fn busy_count_tracks_outstanding_work() {
    let (mut reg, a) = registered(test_config(), "a");
    let b = test_path("b");
    reg.register(&b, 1).unwrap();
    assert_eq!(reg.busy_count(), 0);

    // a dequeued (in flight), b queued → both busy.
    assert!(reg.note_enqueued(&a));
    assert!(reg.note_dequeued(&a));
    assert!(reg.note_enqueued(&b));
    assert_eq!(reg.busy_count(), 2);

    // a posts with nothing left queued → only b busy.
    assert!(reg.note_posted(&a, ResultKind::Success));
    assert_eq!(reg.busy_count(), 1);

    // b dequeued and posts → nobody busy.
    assert!(reg.note_dequeued(&b));
    assert!(reg.note_posted(&b, ResultKind::Success));
    assert_eq!(reg.busy_count(), 0);
}

#[test]
fn quiescence_requires_delivery_from_every_registered_agent() {
    // Session 20260904_c6559510 follow-up: the spawn→send window. A
    // freshly registered agent (spawned, task not yet enqueued) derives
    // Done on the old facts — the missing delivery clause is what keeps
    // a sibling's result from firing a batch that excludes it.
    let (mut reg, a) = registered(test_config(), "a");
    let b = test_path("b");
    reg.register(&b, 1).unwrap();

    // a finished and delivered; b exists but was never tasked.
    assert!(reg.note_enqueued(&a));
    assert!(reg.note_dequeued(&a));
    assert!(reg.note_posted(&a, ResultKind::Success));
    assert!(!reg.quiescent(), "b never delivered — spawn→send window");

    // b gets its task, executes, delivers → now the field settles.
    assert!(reg.note_enqueued(&b));
    assert!(reg.note_dequeued(&b));
    assert!(!reg.quiescent(), "b is in flight");
    assert!(reg.note_posted(&b, ResultKind::Success));
    assert!(reg.quiescent(), "everyone idle and delivered");

    // A queued follow-up re-blocks quiescence even though b delivered.
    assert!(reg.note_enqueued(&b));
    assert!(!reg.quiescent(), "queued work blocks the batch");
}

#[test]
fn a_killed_child_settles_quiescence() {
    // Force-kill regression (20260914_50cf809d report loss, survey §5.1):
    // cancel must land a terminal fact (`note_closing`) instead of leaving
    // the entry `in_flight=true` forever. Before that, one force-kill made
    // quiescence structurally unreachable and the watcher could never
    // hand the held batch over — the sibling's report died with it.
    let (mut reg, a) = registered(test_config(), "a");
    let b = test_path("b");
    reg.register(&b, 1).unwrap();

    // a delivered; b is mid-run when root kills it.
    assert!(reg.note_enqueued(&a));
    assert!(reg.note_dequeued(&a));
    assert!(reg.note_posted(&a, ResultKind::Success));
    assert!(reg.note_enqueued(&b));
    assert!(reg.note_dequeued(&b));
    assert!(!reg.quiescent(), "b is in flight");

    // The cancel lands the terminal fact: b never delivered, yet it no
    // longer blocks the batch.
    assert!(reg.note_closing(&b));
    assert!(
        reg.quiescent(),
        "a closing child counts as settled — its cancel is the terminal fact"
    );
}

#[test]
fn batch_handover_closes_and_reopens_the_delivery_gap() {
    // Session 20260904_841ed65b: a mid-turn parent saw `done` children
    // with no reports and concluded "the system didn't deliver" — the
    // gap must be visible as a fact, and handover must close it.
    let (mut reg, path) = registered(test_config(), "worker");

    reg.note_enqueued(&path);
    reg.note_dequeued(&path);
    reg.note_posted(&path, ResultKind::Success);
    let snap = |reg: &AgentRegistry| reg.snapshot().agents[0].pending_results;
    assert_eq!(snap(&reg), 1, "posted but no batch has fired yet");

    // The watcher stamps the batch member at drain time.
    assert!(reg.note_batch_handed_over(&path));
    assert_eq!(snap(&reg), 0, "handed over — no longer pending");

    // A follow-up post (nudge answer) reopens the gap.
    reg.note_enqueued(&path);
    reg.note_dequeued(&path);
    reg.note_posted(&path, ResultKind::Success);
    assert_eq!(snap(&reg), 1, "new post is pending until the next batch");

    // Ghost paths stay a no-op, and handover never goes backwards.
    assert!(!reg.note_batch_handed_over(&test_path("ghost")));
    assert_eq!(reg.get(&path).unwrap().results_handed_over, 1);
}

#[test]
fn note_on_unknown_path_is_a_noop() {
    let mut reg = AgentRegistry::new(test_config());
    assert!(!reg.note_enqueued(&test_path("ghost")));
    assert!(!reg.note_dequeued(&test_path("ghost")));
    assert!(!reg.note_posted(&test_path("ghost"), ResultKind::Success));
    assert!(!reg.note_send_failed(&test_path("ghost")));
    // Ghost notes stay fully silent: no transitions, no snapshot churn.
    assert!(reg.recent_events(10).is_empty());
    assert!(reg.latest_snapshot().agents.is_empty());
}

#[test]
fn send_failure_rolls_back_the_enqueue_fact() {
    // send_task notes the enqueue BEFORE the task is receivable; a failed
    // send must give the fact back. Regression for the lock-race review
    // finding: a late enqueue note after the child's dequeue used to
    // leave a phantom `queue_len` that blocked quiescence forever.
    let (mut reg, path) = registered(test_config(), "worker");
    assert!(reg.note_enqueued(&path));
    assert_eq!(reg.get(&path).unwrap().queue_len, 1);

    assert!(reg.note_send_failed(&path));
    assert_eq!(reg.get(&path).unwrap().queue_len, 0);
    assert_eq!(reg.get(&path).unwrap().status(), AgentStatus::Done);
    // The agent still has zero deliveries, so it keeps blocking
    // quiescence via the delivery clause — but through a true fact, not
    // a phantom queue.
    assert!(!reg.quiescent());

    // Rollback cannot drive the count negative (saturating).
    assert!(reg.note_send_failed(&path));
    assert_eq!(reg.get(&path).unwrap().queue_len, 0);

    // A delivered agent whose follow-up send failed does not wedge the
    // batch: the rollback restores quiescence.
    let (mut reg2, p2) = registered(test_config(), "solo");
    reg2.note_enqueued(&p2);
    reg2.note_dequeued(&p2);
    reg2.note_posted(&p2, ResultKind::Success);
    reg2.note_enqueued(&p2);
    assert!(!reg2.quiescent());
    reg2.note_send_failed(&p2);
    assert!(
        reg2.quiescent(),
        "rolled-back phantom queue must not wedge the batch"
    );
}

#[test]
fn disabled_config_bypasses_spawn_limits() {
    // Restored (was dropped in the fact-model rewrite): limits are only
    // enforced when enabled=true — spawning works either way, the 6
    // tools just aren't registered when disabled.
    let config = MultiAgentConfig {
        enabled: false,
        ..MultiAgentConfig::default()
    };
    let reg = AgentRegistry::new(config);
    assert!(reg.can_spawn(999).is_ok());
}

#[tokio::test]
async fn running_secs_and_watch_receiver_track_running_state() {
    let (mut reg, path) = registered(test_config(), "worker");
    let mut rx = reg.subscribe();
    assert!(
        reg.snapshot().agents[0].running_secs.is_none(),
        "not running yet"
    );

    reg.note_enqueued(&path);
    reg.note_dequeued(&path);
    rx.changed()
        .await
        .expect("watch receiver must be woken by the transition");
    assert_eq!(rx.borrow().agents[0].running_secs, Some(0));

    reg.note_posted(&path, ResultKind::Success);
    assert!(
        reg.snapshot().agents[0].running_secs.is_none(),
        "stale seconds must clear on post"
    );
}

#[test]
fn events_and_snapshot_track_transitions() {
    let (mut reg, path) = registered(test_config(), "worker");

    // No receivers: publishing must not fail loudly.
    assert!(reg.note_enqueued(&path));
    assert!(reg.note_dequeued(&path));
    assert!(reg.note_posted(&path, ResultKind::Success));

    // register→Done, enqueue→Queued, dequeue→Running, post→Done: three
    // transitions (Done→Done at register emits nothing).
    let events = reg.recent_events(10);
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].reason, "task_enqueued");
    assert_eq!(events[0].from, AgentStatus::Done);
    assert_eq!(events[0].to, AgentStatus::Queued);
    assert_eq!(events[1].reason, "task_dequeued");
    assert_eq!(events[1].to, AgentStatus::Running);
    assert_eq!(events[2].reason, "result_posted");
    assert_eq!(events[2].to, AgentStatus::Done);

    // close emits the unregistered transition.
    reg.close(&path);
    let events = reg.recent_events(10);
    assert_eq!(events.last().unwrap().reason, "unregistered");
    assert_eq!(events.last().unwrap().to, AgentStatus::Closed);

    // The snapshot no longer lists the agent.
    assert!(reg.snapshot().agents.is_empty());
}

#[test]
fn event_ring_is_bounded() {
    let (mut reg, path) = registered(test_config(), "worker");
    for _ in 0..(EVENT_RING_CAP + 100) {
        // Enqueue+dequeue+post cycles generate 3 transitions each.
        reg.note_enqueued(&path);
        reg.note_dequeued(&path);
        reg.note_posted(&path, ResultKind::Success);
    }
    assert_eq!(reg.events.len(), EVENT_RING_CAP);
    assert_eq!(reg.recent_events(3).len(), 3);
}

#[test]
fn snapshot_reflects_derived_state() {
    let (mut reg, path) = registered(test_config(), "worker");

    reg.note_enqueued(&path);
    reg.note_dequeued(&path);
    reg.set_task(&path, "do the thing".to_string());
    reg.record_tool_call(&path);

    let snap = reg.snapshot();
    assert_eq!(snap.agents.len(), 1);
    let row = &snap.agents[0];
    assert_eq!(row.path, "root/worker");
    assert_eq!(row.status, "running");
    assert!(row.running_secs.is_some());
    assert_eq!(row.tool_calls, 1);
    assert_eq!(row.task.as_deref(), Some("do the thing"));

    // The watch channel carries the same snapshot once published
    // (publish happens on transitions; dequeue just moved status).
    let latest = reg.latest_snapshot();
    assert_eq!(latest.agents[0].status, "running");
}

#[test]
fn spawn_error_display() {
    assert_eq!(
        SpawnError::MaxAgentsReached { max: 8 }.to_string(),
        "max agents reached (limit: 8)"
    );
    assert_eq!(
        SpawnError::DepthLimitReached {
            max: 1,
            attempted: 2
        }
        .to_string(),
        "agent depth limit reached (max: 1, attempted: 2)"
    );
    assert_eq!(
        SpawnError::AlreadyExists.to_string(),
        "agent with this path already exists"
    );
}
