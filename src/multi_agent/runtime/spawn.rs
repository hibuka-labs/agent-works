//! Spawn paths and the child task (B split from `runtime.rs`).
//!
//! Everything between "a spawn is requested" and "the child's event loop is
//! running": the gate-ordered preparation (`spawn_inner`, §7.2 ticket in
//! first-out order), the tracked launch (`spawn_ready`) with its
//! [`ChildCleanup`] drop-guard, and the child event loop itself. All spawn
//! entry points — legacy positional, `ChildConfig`-driven, and the
//! [`ChildBuilder`](super::super::ChildBuilder) fluent front — funnel through
//! the same two steps, so the gate order and the cleanup credential exist
//! exactly once.

use super::*;

impl MultiAgentRuntime {
    /// Spawn a child agent at the given path with a specific system prompt.
    ///
    /// This is called by the `spawn_agent` tool. It:
    /// 1. Checks spawn limits
    /// 2. Registers the agent in the registry
    /// 3. Creates a mailbox
    /// 4. Builds a child AgentRuntime
    /// 5. Spawns a tokio task for the child's event loop
    /// 6. Returns the AgentPath
    ///
    /// `parent_messages` optionally provides context from the parent session
    /// for fork_history support.
    ///
    /// # Errors
    ///
    /// Returns a string error message if spawning fails (limits exceeded, etc.).
    pub async fn spawn_child(
        &self,
        name: &str,
        system_prompt: String,
        depth: i32,
        full_permission: bool,
        parent_messages: Vec<agent_base::ChatMessage>,
    ) -> Result<String, String> {
        let config = ChildConfig {
            system_prompt: Some(system_prompt),
            full_permission: Some(full_permission),
            ..Default::default()
        };
        let prepared = self
            .spawn_inner(name, depth, &config, parent_messages)
            .await
            // Legacy error strings preserved byte-for-byte: `ConfigError`
            // wraps the plain SpawnError / "mailbox already exists" /
            // "failed to …" messages, so unwrap them back to the historical
            // text. A typed `ToolNotFound` has no legacy equivalent here
            // (`tool_names` is always `None` on this path), so it falls
            // through to its Display.
            .map_err(|e| match e {
                AgentError::ConfigError(s) => s,
                other => other.to_string(),
            })?;
        let path = prepared.path.clone();
        self.spawn_ready(prepared);
        Ok(path.to_string())
    }

    /// Steps 1–7 of spawning: budget reserve → registry check → limiter
    /// acquire → mailbox register → build child runtime → session + prefill
    /// → child cancellation token.
    ///
    /// Pure extraction from [`spawn_child`](Self::spawn_child) (no behaviour
    /// change); the tokio task launch is [`spawn_ready`](Self::spawn_ready).
    /// Failure paths roll back registry + mailbox exactly as before; the
    /// budget reservation rolls itself back via [`SpawnTicket::Drop`].
    #[allow(clippy::too_many_arguments)]
    async fn spawn_inner(
        &self,
        name: &str,
        depth: i32,
        config: &ChildConfig,
        parent_messages: Vec<agent_base::ChatMessage>,
    ) -> Result<PreparedChild, AgentError> {
        let path = AgentPath::root().join(name);

        // 1. Rollout budget gate (§5.4 gate 1 / §7.2): taken ahead of
        // everything else so a rejection has nothing to roll back — the
        // ticket *is* the reservation. It rides in `PreparedChild` and is
        // committed by `spawn_ready` once the child task actually launches;
        // any failure between here and there returns it via `Drop`.
        let ticket = self
            .control
            .budget()
            .try_reserve_spawn()
            .map_err(|e| AgentError::ConfigError(e.to_string()))?;

        // 2. Check limits and register
        {
            let mut registry = self.registry.lock().unwrap();
            registry
                .can_spawn(depth)
                .map_err(|e| AgentError::ConfigError(e.to_string()))?;
            registry
                .register(&path, depth)
                .map_err(|e| AgentError::ConfigError(e.to_string()))?;
        }

        // 3. Live-concurrency gate (design §3.3 gate 3): acquired right
        // after the registry gate, before the build step. Failure rolls back
        // the registry entry (逆序归还: gate 3 failed → undo gate 2); any
        // later `Err` in this function returns the slot via the `?` path
        // (local drop), and the ticket with it.
        let slot = match self.limiter.try_acquire() {
            Ok(slot) => slot,
            Err(e) => {
                self.registry.lock().unwrap().close(&path);
                return Err(AgentError::ConfigError(e.to_string()));
            }
        };

        // 4. Create mailbox
        let child_mailbox = self
            .mailbox
            .register(&path)
            .ok_or_else(|| AgentError::ConfigError("mailbox already exists".to_string()))?;

        // 5. Build child AgentRuntime (roll back registry+mailbox on failure).
        // A `ToolNotFound` from whitelist validation propagates typed (it is
        // the caller's contract, §5.4); every other build error is wrapped
        // with the historical "failed to build child runtime:" prefix so the
        // legacy spawn_child string output is unchanged.
        let (child_runtime, spawned_tools) = self
            .build_child_runtime_with_config(config, self.spawn_permission(config.full_permission))
            .await
            .map_err(|e| {
                self.registry.lock().unwrap().close(&path);
                self.mailbox.unregister(&path);
                match e {
                    AgentError::ToolNotFound { .. } => e,
                    other => {
                        AgentError::ConfigError(format!("failed to build child runtime: {other}"))
                    }
                }
            })?;

        // 6. Create session for child and pre-fill with parent context
        let session_id = child_runtime.create_session().await;
        self.prefill_child_session(&child_runtime, &session_id, &parent_messages)
            .await
            .map_err(|e| {
                self.registry.lock().unwrap().close(&path);
                self.mailbox.unregister(&path);
                AgentError::ConfigError(format!("failed to prefill child session: {e}"))
            })?;

        // 7. Create child cancellation token
        let child_cancel = self.root_cancel.child_token();
        {
            let mut cancels = self.child_cancels.lock().unwrap();
            cancels.insert(path.clone(), child_cancel.clone());
        }

        Ok(PreparedChild {
            path,
            child_mailbox,
            child_runtime,
            session_id,
            child_cancel,
            slot,
            ticket,
            spawned_tools,
        })
    }

    /// Step 6 of spawning: run the child's event loop inside a tracked tokio
    /// task, wrapped in the [`ChildCleanup`] drop-guard.
    ///
    /// Whichever way the task closure leaves — normal return, panic unwind,
    /// or JoinSet abort (parent [`Drop`](MultiAgentRuntime)) — the future is
    /// dropped and the guard's teardown runs exactly once (design §4, review
    /// B-2: one path, panic safe; the v3 tail-code / `child_slots` map designs
    /// are superseded).
    fn spawn_ready(&self, prepared: PreparedChild) {
        let PreparedChild {
            path,
            child_mailbox,
            child_runtime,
            session_id,
            child_cancel,
            slot,
            ticket,
            spawned_tools: _spawned_tools,
        } = prepared;

        let task_timeout = self.task_timeout;
        let agent_path = path.clone();
        let mailbox_for_task = self.mailbox.clone();
        let registry_for_task = self.registry.clone();
        let event_tx = self.event_tx.lock().unwrap().clone();
        let registry_agent_path = path.clone();

        let cleanup = ChildCleanup {
            _slot: slot,
            mailbox: self.mailbox.clone(),
            registry: self.registry.clone(),
            child_cancels: self.child_cancels.clone(),
            path: agent_path.clone(),
        };

        self.join_set.lock().unwrap().spawn(async move {
            let _cleanup = cleanup;
            run_child_loop(
                child_mailbox,
                child_runtime,
                session_id,
                agent_path,
                mailbox_for_task,
                registry_for_task,
                event_tx,
                child_cancel,
                task_timeout,
            )
            .await;
        });

        // The budget reservation becomes permanent only once the child task
        // is actually launched (§7.2): every failure point of the spawn chain
        // — gates, build, session — happened before this line, and on those
        // paths the ticket's `Drop` already returned the count. Committing
        // after `spawn` (not before) also mirrors the slot: if the launch
        // itself panics, neither credential is spent.
        ticket.commit();

        self.registry
            .lock()
            .unwrap()
            .set_status(&registry_agent_path, AgentStatus::Idle);
    }

    /// Spawn a child agent with fork_history support.
    ///
    /// `fork_history`: "none" (default), "all", or a number N for last N turns.
    /// `parent_session_id`: the parent agent's session ID.
    /// `model`: requested model override, carried into `ChildConfig.model`.
    /// TODO(layer-3): inert today — see `ChildConfig::model`.
    #[allow(clippy::too_many_arguments)] // spawn config is naturally positional
    pub async fn spawn_child_with_history(
        &self,
        name: &str,
        system_prompt: String,
        full_permission: bool,
        fork_history: Option<String>,
        model: Option<String>,
        parent_session_id: &SessionId,
    ) -> Result<String, String> {
        let parent_messages = self
            .resolve_fork_history(fork_history, parent_session_id)
            .await;
        let config = ChildConfig {
            system_prompt: Some(system_prompt),
            full_permission: Some(full_permission),
            model,
            ..Default::default()
        };
        self.spawn_with_config_forked(name.to_string(), config, parent_messages)
            .await
            // Legacy error strings preserved byte-for-byte (see spawn_child).
            .map_err(|e| match e {
                AgentError::ConfigError(s) => s,
                other => other.to_string(),
            })
            .map(|spawned| spawned.path.to_string())
    }

    /// Fluent entry point for the new spawn API (§5.3). Takes `&Arc<Self>`
    /// because the resulting [`ChildHandle`](super::super::ChildHandle) must hold an
    /// owned runtime reference; the `Arc` requirement is also what stops this
    /// being called on a stack temporary.
    pub fn child(self: &Arc<Self>) -> ChildBuilder {
        ChildBuilder::new(Arc::clone(self))
    }

    /// Test-facing convenience over [`spawn_with_config_forked`] with no
    /// pre-fill (the §5.4 two-arg signature). Production callers go through
    /// [`ChildBuilder`](super::super::ChildBuilder), which always uses the forked
    /// variant (with an empty message list when `fork_history` is unset).
    #[cfg(test)]
    pub(crate) async fn spawn_with_config(
        &self,
        name: String,
        config: ChildConfig,
    ) -> Result<SpawnedChild, AgentError> {
        self.spawn_with_config_forked(name, config, Vec::new())
            .await
    }

    /// New spawn entry point driven by [`ChildConfig`] (design §5.4), plus
    /// legacy `fork_history` support: `parent_messages` pre-fills the child
    /// session exactly like
    /// [`spawn_child_with_history`](Self::spawn_child_with_history) (which
    /// resolves its own messages via [`resolve_fork_history`]). The
    /// [`ChildBuilder::fork_history`](super::super::ChildBuilder::fork_history)
    /// setter is the new entry to that route.
    ///
    /// The legacy positional [`spawn_child`](Self::spawn_child) /
    /// [`spawn_child_with_history`](Self::spawn_child_with_history) keep
    /// their signatures; both funnel into `spawn_inner` (which also this
    /// method uses), so the seven preparation steps, the gate order, and the
    /// `ChildCleanup` drop-guard exist exactly once.
    ///
    /// Errors: `ConfigError` when `system_prompt` is empty (the fail-fast
    /// required-field check, §5.1); a typed `ToolNotFound` when a whitelisted
    /// tool resolves nowhere (§5.4); `ConfigError` for gate rejections.
    pub(crate) async fn spawn_with_config_forked(
        &self,
        name: String,
        config: ChildConfig,
        parent_messages: Vec<agent_base::ChatMessage>,
    ) -> Result<SpawnedChild, AgentError> {
        // Required-field check (runtime, fail-fast — §5.1).
        if config.system_prompt.as_deref().unwrap_or("").is_empty() {
            return Err(AgentError::ConfigError(
                "ChildConfig.system_prompt is required (set it directly or use a preset)".into(),
            ));
        }

        // Nesting is structurally absent (K5 / §10.1 B4): a config child is
        // always a direct child of root. The *actual* registered tool set is
        // echoed in `SpawnedChild::spawned_tools`.
        let prepared = self
            .spawn_inner(&name, 1, &config, parent_messages)
            .await?;

        let path = prepared.path.clone();
        let spawned_tools = prepared.spawned_tools.clone();
        self.spawn_ready(prepared);
        Ok(SpawnedChild {
            path,
            spawned_tools,
        })
    }
}

// ---------------------------------------------------------------------------
// Prepared child
// ---------------------------------------------------------------------------

/// Everything [`spawn_inner`](MultiAgentRuntime::spawn_inner) prepares before
/// the child's event-loop task is launched by
/// [`spawn_ready`](MultiAgentRuntime::spawn_ready).
///
/// The tokio task closure consumes this struct; keeping the pieces bundled lets
/// future spawn paths (drop-guard cleanup, `ChildConfig`) wrap the same launch
/// step without re-doing the six preparation steps.
struct PreparedChild {
    path: AgentPath,
    child_mailbox: ChildMailbox,
    child_runtime: AgentRuntime,
    session_id: SessionId,
    child_cancel: CancellationToken,
    slot: ExecutionSlot,
    /// Budget reservation for this spawn (§7.2), committed by
    /// [`spawn_ready`](MultiAgentRuntime::spawn_ready) after the task
    /// launches — dropped (rolled back) on every path that never gets there.
    ticket: SpawnTicket,
    /// The tools actually registered on the child (post-exclusion,
    /// post-whitelist). Echoed to the caller so the parent can see the
    /// child's real capability set (§5.4). Empty for the legacy path's
    /// internal bookkeeping is unused, but always accurate.
    spawned_tools: BTreeSet<String>,
}

/// Result of [`MultiAgentRuntime::spawn_with_config`] (design §5.4).
///
/// The child's path plus the tools it **actually** received (the "echo" the
/// spawn output surfaces so the parent sees the child's real capability set,
/// §5.4 review M-3). `pub(crate)`: the public view is `ChildHandle` (§5.2),
/// which wraps this once the builder lands.
pub(crate) struct SpawnedChild {
    path: AgentPath,
    spawned_tools: BTreeSet<String>,
}

impl SpawnedChild {
    /// Path of the spawned child.
    pub(crate) fn agent_path(&self) -> &AgentPath {
        &self.path
    }
    /// The tools the child was actually given (post-exclusion whitelist).
    pub(crate) fn spawned_tools(&self) -> &BTreeSet<String> {
        &self.spawned_tools
    }
}

/// The child task's cleanup credential (design doc §5.4, review B-2/M-2/M-7).
///
/// Lives as a local inside the task closure created by
/// [`spawn_ready`](MultiAgentRuntime::spawn_ready), so **every** way the task
/// can leave — normal return, panic unwind, JoinSet abort — runs its `Drop`.
/// There is deliberately no `child_slots` map and no `mem::forget`: the slot
/// and the teardown belong to the task itself, not to a side table that could
/// leak or race (the insert-after-spawn race of the earlier design closes with
/// this).
///
/// Holds `Arc` clones of the components it must reach (mailbox, registry,
/// child tokens) rather than an `Arc<MultiAgentRuntime>`: a task-side strong
/// ref to the runtime would form a reference cycle (runtime → join_set →
/// task → guard → runtime) that makes `Drop(MultiAgentRuntime)` — the very
/// `cancel_all` + abort mechanism §4 relies on — unreachable.
struct ChildCleanup {
    /// Held for RAII only: dropping the guard releases the live-concurrency
    /// slot (design §3.2: 额度凭证活在子任务闭包里).
    _slot: ExecutionSlot,
    mailbox: Arc<MailboxHub>,
    registry: Arc<Mutex<AgentRegistry>>,
    child_cancels: Arc<Mutex<HashMap<AgentPath, CancellationToken>>>,
    path: AgentPath,
}

impl Drop for ChildCleanup {
    fn drop(&mut self) {
        // Order is load-bearing. §5.4 lists post → unregister → close; we
        // close the registry BEFORE unregistering the mailbox — the K1 fix
        // (Closed posted while the entry still exists) is preserved, and it
        // makes `wait_for_result`'s terminal "closed" synthesis race-free:
        // once the mailbox entry is gone (the waiter's last wake point), the
        // registry slot is provably already released.
        //
        // 1. post Closed — wake waiters while the entry still exists (K1 fix).
        self.mailbox.post_result(MailboxResult {
            agent_path: self.path.clone(),
            status: MailboxStatus::Closed,
            result: None,
            denied_tools: vec![],
        });
        // 2. release the registry slot (terminal state per §9.2: entry gone).
        self.registry.lock().unwrap().close(&self.path);
        // 3. remove the mailbox entry (bumps seq again — the waiter's
        //    deterministic close→wait wake point).
        self.mailbox.unregister(&self.path);
        // 4. drop the cancellation token if `close_agent` left it in place
        //    (`close_agent` cancels but no longer removes; this is the single
        //    removal point).
        self.child_cancels.lock().unwrap().remove(&self.path);
        // `_slot` drops at the end of this function → live concurrency − 1.
    }
}

// ---------------------------------------------------------------------------
// Child agent event loop
// ---------------------------------------------------------------------------

/// Run one dequeued child task to completion and build its mailbox result.
///
/// §9.2: `Some(dur)` puts a hard wall on one task. On elapse the turn future
/// is dropped (execution stops at its next await point), `cancel()` tells the
/// engine to abandon detached work, and the parent learns via the mailbox
/// Error. The child itself survives: agent-base resets the cancel token
/// before every `run_turn`, so later tasks run normally.
async fn execute_child_task(
    child_runtime: &AgentRuntime,
    session_id: &SessionId,
    task: &crate::multi_agent::mailbox::MailboxTask,
    task_timeout: Option<Duration>,
) -> (MailboxStatus, Option<String>, Vec<String>) {
    let input = outcome::build_child_input(task);
    let run = child_runtime.run_turn_collect(session_id.clone(), &input);
    let result = if let Some(dur) = task_timeout {
        match tokio::time::timeout(dur, run).await {
            Ok(r) => r,
            Err(_elapsed) => {
                child_runtime.cancel();
                return (
                    MailboxStatus::Error,
                    Some(format!("task timed out after {dur:?}")),
                    vec![],
                );
            }
        }
    } else {
        run.await
    };
    match result {
        Ok((events, outcome)) => (
            MailboxStatus::Ok,
            Some(outcome::build_child_result(&outcome, &events)),
            outcome::collect_denied_tools(&events),
        ),
        Err(e) => (MailboxStatus::Error, Some(e.to_string()), vec![]),
    }
}

/// Run the child agent's main event loop.
///
/// This function runs inside a tokio task spawned by [`MultiAgentRuntime::spawn_child`].
/// It:
/// 1. Subscribes to child agent events and bridges them to parent
/// 2. Listens for tasks from the mailbox
/// 3. Executes each task via `run_turn`
/// 4. Posts results back via the mailbox
///
/// `task_timeout` (§9.2): `Some(dur)` hard-stops a single task's `run_turn`
/// after the duration elapses — the child is *not* closed, it reports the
/// timeout and keeps serving later tasks.
#[allow(clippy::too_many_arguments)] // the loop's whole world is positional inputs
async fn run_child_loop(
    child_mailbox: ChildMailbox,
    child_runtime: AgentRuntime,
    session_id: SessionId,
    agent_path: AgentPath,
    mailbox: Arc<MailboxHub>,
    registry: Arc<Mutex<AgentRegistry>>,
    event_tx: Option<tokio::sync::mpsc::UnboundedSender<RuntimeEvent>>,
    child_cancel: CancellationToken,
    task_timeout: Option<Duration>,
) {
    let mut task_rx = child_mailbox.task_rx;

    // Spawn event bridging: forward child events to parent, tagging agent_id.
    // Tool-call starts are also recorded in the registry (monotonic counter +
    // activity timestamp) so `list_agents` shows real progress, not a frozen
    // inventory size (session 20260903_0cf95e79 regression).
    if let Some(tx) = event_tx {
        let mut child_events = child_runtime.subscribe_runtime_events();
        let bridge_path = agent_path.to_string();
        let bridge_cancel = child_cancel.clone();
        let bridge_registry = registry.clone();
        let bridge_agent_path = agent_path.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = bridge_cancel.cancelled() => break,
                    event = child_events.recv() => {
                        match event {
                            Ok(event) => {
                                if matches!(event, RuntimeEvent::ToolCallStarted { .. }) {
                                    bridge_registry
                                        .lock()
                                        .unwrap()
                                        .record_tool_call(&bridge_agent_path);
                                }
                                if matches!(event,
                                    RuntimeEvent::RunFinished { .. }
                                    | RuntimeEvent::RunCancelled { .. }
                                    | RuntimeEvent::AwaitingApproval { .. }) {
                                    continue;
                                }
                                let _ = tx.send(event.with_agent_id(bridge_path.as_str()));
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                                tracing::warn!(
                                    subagent = %bridge_path,
                                    lagged = n,
                                    "child event bridge lagged"
                                );
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
        });
    }

    // Main task loop
    loop {
        tokio::select! {
            _ = child_cancel.cancelled() => {
                break;
            }
            task = task_rx.recv() => {
                match task {
                    Some(mut task) => {
                        // Serial task-queue loop. Status protocol (session
                        // 20260904_c6559510): `send_task` marks Running at
                        // QUEUE time, but only this loop knows whether more
                        // tasks remain. The next task is peeked via
                        // `try_recv` BEFORE posting the current result, and
                        // the status is set accordingly — so the watcher's
                        // quiescence check (`running_count() == 0`) never
                        // sees a phantom-idle child whose queue still holds
                        // work. The old protocol (Done at every post, no
                        // Running at dequeue) made the watcher fire the
                        // batch early and fragment the remaining results
                        // into one wake-up per straggler.
                        loop {
                            if child_cancel.is_cancelled() {
                                break; // close is honored between tasks
                            }
                            let (status, result_text, denied_tools) =
                                execute_child_task(
                                    &child_runtime,
                                    &session_id,
                                    &task,
                                    task_timeout,
                                )
                                .await;

                            // Peek the next queued task BEFORE posting: the
                            // status the watcher sees when it wakes on this
                            // post's seq bump must reflect the queue. With a
                            // queued task → Running (a later post will re-wake
                            // it, so holding this result cannot hang the
                            // batch). Queue empty → Done (Done-before-post,
                            // the fan-in invariant: a bare `set_status` never
                            // bumps the seq, so posting in Done state is what
                            // makes "result drained ⇒ producer quiescent"
                            // hold for the final task).
                            let next = task_rx.try_recv().ok();
                            let new_status = if next.is_some() {
                                AgentStatus::Running
                            } else {
                                AgentStatus::Done
                            };
                            registry
                                .lock()
                                .unwrap()
                                .set_status(&agent_path, new_status);
                            mailbox.post_result(MailboxResult {
                                agent_path: agent_path.clone(),
                                status,
                                result: result_text,
                                denied_tools,
                            });
                            match next {
                                Some(t) => task = t,
                                None => break,
                            }
                        }
                    }
                    None => break, // task channel closed
                }
            }
        }
    }
}
