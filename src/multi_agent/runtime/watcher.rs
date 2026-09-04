//! Background watcher task — the fan-in coordinator for child results.
//!
//! Session 20260903_0cf95e79 redesign: instead of waking the parent for
//! every single result (which pushed the "are we done yet?" decision into
//! the parent's LLM — it misread a frozen inventory `tool_count` as a stall
//! and started doing the children's work itself), the watcher now owns the
//! coordination:
//!
//! - **Progress** — each result is announced for the *user* immediately
//!   (Focus summary once wired, plain notice otherwise). Progress never
//!   wakes the parent agent.
//! - **Batch** — once nothing is executing anymore
//!   (`registry.running_count() == 0`) and at least one non-Closed report
//!   is pending, the watcher emits one `Batch` carrying every full report.
//!   The parent wakes exactly once per generation of children and runs one
//!   synthesis turn.
//!
//! Rules encoded here:
//! - A redundant `Closed` for an agent whose `Ok`/`Error` is already in the
//!   batch is dropped — close-after-complete is bookkeeping noise.
//! - A batch that contains only `Closed` results never wakes the parent
//!   (the user closed the children; there is nothing to synthesize). They
//!   were surfaced as Progress and the batch resets so stale notifications
//!   cannot leak into a later generation's batch.
//! - The Focus summary never gates the wake (session 20260903_d8fc41dc:
//!   awaited summaries serialized their timeouts in front of the batch).
//!   Ok/Error get a plain Progress notice synchronously, then the summary
//!   runs detached and follows as a second Progress event whenever Focus
//!   answers (or not at all on failure). A Progress event may therefore
//!   arrive after the Batch event, and the same agent may Progress twice —
//!   consumers must be idempotent (phimint's UI is: mark-finished is a
//!   no-op on an already-finished entry). Closed results need no summary
//!   and are announced synchronously.
//! - Two ordering invariants keep every quiescence check well-timed:
//!   the child loop sets `Done` *before* posting its FINAL task's result
//!   (a bare `set_status` never bumps the mailbox seq — see `spawn.rs`;
//!   while a task remains queued it instead posts in `Running` state, so a
//!   phantom-idle child can never make the batch fire early — session
//!   20260904_c6559510), and `unregister` bumps the seq after releasing
//!   the registry slot (the close path).

use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::outcome::format_child_result;
use crate::focus::ProgressSummarizer;
use crate::multi_agent::mailbox::{MailboxHub, MailboxResult, MailboxStatus};
use crate::multi_agent::registry::AgentRegistry;

/// One child's final report — the full conclusion, never the child's
/// working history (extraction happens in `outcome.rs` at post time).
#[derive(Clone, Debug)]
pub struct ChildReport {
    /// The child agent's path (e.g. "root/worker").
    pub agent_path: String,
    /// Status: "ok", "error", or "closed".
    pub status: String,
    /// The result text (if any).
    pub result: Option<String>,
    /// Formatted message suitable for injection into the parent's context.
    pub message: String,
}

/// An event delivered by the watcher task.
#[derive(Clone, Debug)]
pub enum ChildResultEvent {
    /// One child returned. User-facing progress only — never wakes the
    /// parent agent. Emitted **twice** for Ok/Error results when a
    /// summarizer is wired: first synchronously with `summary: None` (the
    /// plain "已返回" notice, the moment the child returns), then once more
    /// with the Focus summary when it lands (no second event on Focus
    /// failure). Consumers must treat repeated Progress for the same
    /// agent as idempotent.
    Progress {
        /// The child agent's path.
        agent_path: String,
        /// Status: "ok", "error", or "closed".
        status: String,
        /// Focus-generated summary shown as a follow-up line. `None` → the
        /// consumer shows a plain notice.
        summary: Option<String>,
    },
    /// Every child has returned — wake the parent once with all reports,
    /// including any `Closed` siblings of the same generation.
    Batch {
        /// Full reports, one per returned child, in arrival order.
        reports: Vec<ChildReport>,
    },
}

impl ChildResultEvent {
    /// The producing agent's path, for logging.
    pub fn agent_path(&self) -> &str {
        match self {
            Self::Progress { agent_path, .. } => agent_path,
            Self::Batch { reports } => reports
                .first()
                .map(|r| r.agent_path.as_str())
                .unwrap_or_default(),
        }
    }
}

/// Spawn the background watcher (fan-in coordinator) task.
///
/// Returns a `JoinHandle` that the caller can store (or ignore — the task
/// exits when the `cancel` token is fired or the channel sender is dropped).
pub fn spawn_watcher(
    mailbox: Arc<MailboxHub>,
    registry: Arc<Mutex<AgentRegistry>>,
    summarizer: Option<Arc<ProgressSummarizer>>,
    child_result_tx: Option<mpsc::UnboundedSender<ChildResultEvent>>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let mut seq_rx = mailbox.subscribe_seq();

    tokio::spawn(async move {
        // Results held until the whole generation has returned.
        let mut batch: Vec<MailboxResult> = Vec::new();

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    break;
                }
                result = seq_rx.changed() => {
                    if result.is_err() {
                        break;
                    }

                    // Drain everything pending into the batch.
                    while let Some(r) = mailbox.try_recv_any() {
                        if matches!(r.status, MailboxStatus::Closed) {
                            // Redundant close notification: this agent's real
                            // result is already in the batch (or already
                            // flushed) — bookkeeping noise, drop it.
                            let already_accounted = batch.iter().any(|b| {
                                b.agent_path == r.agent_path
                                    && !matches!(b.status, MailboxStatus::Closed)
                            });
                            if already_accounted {
                                tracing::debug!(
                                    agent = %r.agent_path,
                                    "dropping redundant Closed notification"
                                );
                                continue;
                            }
                        }

                        if let Some(tx) = &child_result_tx {
                            let status = status_str(&r.status);
                            // Focus summarizes for the user, on the main
                            // agent's behalf. Closed needs no LLM call — a
                            // plain "已关闭" notice is enough. Any Focus
                            // failure degrades to `None` (plain notice).
                            //
                            // Session 20260903_d8fc41dc: the summarize call
                            // used to be awaited right here, which put every
                            // timeout (30 s) on the wake path — the batch
                            // could not be evaluated until all pending
                            // summaries had run, serializing N×timeout in
                            // front of the parent's wake. The summary only
                            // ever serves the user, so it now runs detached:
                            // the result enters the batch immediately and the
                            // Progress event follows whenever Focus answers.
                            // Consequence: a Progress event may arrive after
                            // the Batch event — consumers must be idempotent
                            // (phimint's UI is: mark-finished is a no-op on
                            // an already-finished entry).
                            match (&summarizer, &r.status) {
                                (Some(s), MailboxStatus::Ok | MailboxStatus::Error) => {
                                    // Plain notice FIRST, synchronously: the
                                    // user learns the child returned the
                                    // moment it does (session
                                    // 20260904_e6612477: one summary's 30 s
                                    // Focus timeout delayed that child's only
                                    // notice by 30 s). The Focus summary —
                                    // when it lands — follows as a second
                                    // Progress event; on Focus failure no
                                    // second event is sent (the plain notice
                                    // already covers it).
                                    let _ = tx.send(ChildResultEvent::Progress {
                                        agent_path: r.agent_path.to_string(),
                                        status: status.to_string(),
                                        summary: None,
                                    });
                                    let task = registry
                                        .lock()
                                        .unwrap()
                                        .get(&r.agent_path)
                                        .and_then(|e| e.task.clone());
                                    let agent_name = r.agent_path.name().to_string();
                                    let agent_path = r.agent_path.to_string();
                                    let status = status.to_string();
                                    let result_text = r.result.clone();
                                    let summarizer = Arc::clone(s);
                                    let tx = tx.clone();
                                    tokio::spawn(async move {
                                        let summary = match summarizer
                                            .summarize(
                                                &agent_name,
                                                &status,
                                                task.as_deref(),
                                                result_text.as_deref(),
                                            )
                                            .await
                                        {
                                            Some(text) => Some(text),
                                            None => {
                                                tracing::debug!(
                                                    agent = %agent_path,
                                                    "progress summary unavailable — plain notice already sent"
                                                );
                                                return;
                                            }
                                        };
                                        let _ = tx.send(ChildResultEvent::Progress {
                                            agent_path,
                                            status,
                                            summary,
                                        });
                                    });
                                }
                                _ => {
                                    let _ = tx.send(ChildResultEvent::Progress {
                                        agent_path: r.agent_path.to_string(),
                                        status: status.to_string(),
                                        summary: None,
                                    });
                                }
                            }
                        }
                        batch.push(r);
                    }

                    // Quiescence: nothing is executing anymore and at least
                    // one real (non-Closed) report is pending → wake the
                    // parent once with everything.
                    if batch.is_empty() || registry.lock().unwrap().running_count() != 0 {
                        continue;
                    }
                    let any_real = batch
                        .iter()
                        .any(|b| !matches!(b.status, MailboxStatus::Closed));
                    if !any_real {
                        // Only close notifications — surfaced as Progress
                        // above; the parent has nothing to synthesize. Reset
                        // so they cannot leak into a later generation.
                        batch.clear();
                        continue;
                    }
                    let reports = batch
                        .drain(..)
                        .map(|r| format_child_result(&r))
                        .collect();
                    if let Some(tx) = &child_result_tx {
                        let _ = tx.send(ChildResultEvent::Batch { reports });
                    }
                }
            }
        }
    })
}

fn status_str(status: &MailboxStatus) -> &'static str {
    match status {
        MailboxStatus::Ok => "ok",
        MailboxStatus::Error => "error",
        MailboxStatus::Closed => "closed",
    }
}

/// Spawn a watchdog-wrapped watcher task.
///
/// The watchdog monitors the inner watcher task and restarts it if it panics.
/// This ensures the child result delivery mechanism remains operational even
/// in the face of unexpected panics. A panic counter is logged for diagnostics.
///
/// The watchdog itself exits when the `cancel` token is fired.
pub fn spawn_watcher_with_watchdog(
    mailbox: Arc<MailboxHub>,
    registry: Arc<Mutex<AgentRegistry>>,
    summarizer: Option<Arc<ProgressSummarizer>>,
    child_result_tx: Option<mpsc::UnboundedSender<ChildResultEvent>>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut restart_count: u32 = 0;

        loop {
            if cancel.is_cancelled() {
                break;
            }

            let handle = spawn_watcher(
                mailbox.clone(),
                registry.clone(),
                summarizer.clone(),
                child_result_tx.clone(),
                cancel.clone(),
            );

            // Wait for the watcher to finish.
            match handle.await {
                Ok(()) => {
                    // Normal exit (cancel or channel closed). No restart needed.
                    break;
                }
                Err(join_err) if join_err.is_panic() => {
                    restart_count += 1;
                    let panic_info = join_err
                        .try_into_panic()
                        .ok()
                        .and_then(|p| {
                            p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| {
                                p.downcast_ref::<String>().map(|s| s.clone())
                            })
                        })
                        .unwrap_or_else(|| "unknown panic".to_string());

                    tracing::warn!(
                        restart_count,
                        panic_info = %panic_info,
                        "watcher task panicked, restarting"
                    );

                    // Brief backoff to avoid tight restart loops on
                    // persistent panics (e.g. a bug in format_child_result).
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
                Err(join_err) => {
                    // Cancelled or other JoinError — exit.
                    tracing::debug!(error = %join_err, "watcher task join error, exiting watchdog");
                    break;
                }
            }
        }

        if restart_count > 0 {
            tracing::info!(
                restart_count,
                "watcher watchdog exiting after restarts"
            );
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_agent::config::MultiAgentConfig;
    use crate::multi_agent::mailbox::MailboxResult;
    use crate::multi_agent::path::AgentPath;
    use crate::multi_agent::registry::AgentStatus;

    struct Fixture {
        mailbox: Arc<MailboxHub>,
        registry: Arc<Mutex<AgentRegistry>>,
        rx: mpsc::UnboundedReceiver<ChildResultEvent>,
        cancel: CancellationToken,
        #[allow(dead_code)]
        handle: tokio::task::JoinHandle<()>,
    }

    /// Watcher + registry fixture. Children start `Idle`; tests move them
    /// through the same Done-before-post order the child loop uses.
    fn fixture() -> Fixture {
        fixture_with_summarizer(None)
    }

    fn fixture_with_summarizer(summarizer: Option<Arc<ProgressSummarizer>>) -> Fixture {
        let mailbox = Arc::new(MailboxHub::new());
        let registry = Arc::new(Mutex::new(AgentRegistry::new(MultiAgentConfig::enabled())));
        let (tx, rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let handle = spawn_watcher_with_watchdog(
            mailbox.clone(),
            registry.clone(),
            summarizer,
            Some(tx),
            cancel.clone(),
        );
        Fixture {
            mailbox,
            registry,
            rx,
            cancel,
            handle,
        }
    }

    /// Register a child in both the mailbox and the registry, marking it
    /// `Running` (the state `send_task` produces).
    fn spawn_running(fx: &Fixture, name: &str) {
        let path = AgentPath::root().join(name);
        fx.mailbox.register(&path);
        fx.registry.lock().unwrap().register(&path, 1).unwrap();
        fx.registry
            .lock()
            .unwrap()
            .set_status(&path, AgentStatus::Running);
    }

    /// The child loop's terminal order: publish `Done` *before* posting the
    /// result (see the fan-in note in `spawn.rs`).
    fn finish_and_post(fx: &Fixture, name: &str, status: MailboxStatus, text: Option<&str>) {
        let path = AgentPath::root().join(name);
        fx.registry
            .lock()
            .unwrap()
            .set_status(&path, AgentStatus::Done);
        fx.mailbox.register(&path);
        fx.mailbox.post_result(MailboxResult {
            agent_path: path,
            status,
            result: text.map(|s| s.to_string()),
            denied_tools: vec![],
        });
    }

    async fn next_event(fx: &mut Fixture) -> ChildResultEvent {
        tokio::time::timeout(std::time::Duration::from_secs(2), fx.rx.recv())
            .await
            .expect("event timeout")
            .expect("channel closed")
    }

    async fn assert_no_event(fx: &mut Fixture) {
        let got = tokio::time::timeout(std::time::Duration::from_millis(200), fx.rx.recv()).await;
        assert!(
            got.is_err(),
            "expected no event, got {:?}",
            got.ok().flatten()
        );
    }

    #[tokio::test]
    async fn lone_result_progress_then_batch() {
        let mut fx = fixture();
        finish_and_post(&fx, "worker", MailboxStatus::Ok, Some("done!"));

        match next_event(&mut fx).await {
            ChildResultEvent::Progress {
                agent_path, status, ..
            } => {
                assert_eq!(agent_path, "root/worker");
                assert_eq!(status, "ok");
            }
            other => panic!("expected Progress, got {other:?}"),
        }
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1);
                assert_eq!(reports[0].agent_path, "root/worker");
                assert_eq!(reports[0].status, "ok");
                assert_eq!(reports[0].result.as_deref(), Some("done!"));
                assert!(reports[0].message.contains("done!"));
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn batch_held_until_last_child_finishes() {
        // Two children; A finishes while B is still Running → only A's
        // Progress. When B finishes, one Batch carries both reports.
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { agent_path, .. } => {
                assert_eq!(agent_path, "root/a");
            }
            other => panic!("expected Progress for a, got {other:?}"),
        }
        assert_no_event(&mut fx).await; // b still Running — no Batch

        finish_and_post(&fx, "b", MailboxStatus::Ok, Some("b done"));
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { agent_path, .. } => {
                assert_eq!(agent_path, "root/b");
            }
            other => panic!("expected Progress for b, got {other:?}"),
        }
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                let paths: Vec<&str> = reports.iter().map(|r| r.agent_path.as_str()).collect();
                assert_eq!(paths, vec!["root/a", "root/b"]);
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn closed_sibling_rides_along_in_batch() {
        // A completes Ok; B was closed (never delivered a real result).
        // The parent must learn B is gone in the same wake.
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress a
        assert_no_event(&mut fx).await;

        finish_and_post(&fx, "b", MailboxStatus::Closed, None);
        let _ = next_event(&mut fx).await; // Progress b
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 2);
                assert_eq!(reports[0].status, "ok");
                assert_eq!(reports[1].status, "closed");
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn closed_only_batch_never_wakes_parent() {
        // The user closed every child; no real result ever arrived.
        // Progress is fine, but no Batch may fire.
        let mut fx = fixture();
        spawn_running(&fx, "w");

        finish_and_post(&fx, "w", MailboxStatus::Closed, None);
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { status, .. } => assert_eq!(status, "closed"),
            other => panic!("expected Progress, got {other:?}"),
        }
        assert_no_event(&mut fx).await; // no Batch
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn redundant_closed_is_dropped_while_held() {
        // A completes, then the user closes it before B finishes. The close
        // notification must not duplicate A in the eventual batch.
        let mut fx = fixture();
        for name in ["a", "b"] {
            spawn_running(&fx, name);
        }

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress a

        // Close B's sibling A again: post a redundant Closed (the registry
        // slot is still held by B, so the batch stays pending).
        fx.mailbox.post_result(MailboxResult {
            agent_path: AgentPath::root().join("a"),
            status: MailboxStatus::Closed,
            result: None,
            denied_tools: vec![],
        });
        assert_no_event(&mut fx).await; // dropped — no duplicate Progress

        finish_and_post(&fx, "b", MailboxStatus::Ok, Some("b done"));
        let _ = next_event(&mut fx).await; // Progress b
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 2, "a + b, no duplicate of either");
                assert!(
                    reports.iter().all(|r| r.status == "ok"),
                    "redundant Closed must not ride along"
                );
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn stale_closed_after_flush_does_not_wake() {
        // A's Ok batch flushed; a late Closed for A arrives (cleanup race).
        // Lone Closed → Progress only, never a Batch.
        let mut fx = fixture();
        fx.mailbox.register(&AgentPath::root().join("a"));

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
        let _ = next_event(&mut fx).await; // Progress
        let _ = next_event(&mut fx).await; // Batch

        fx.mailbox.post_result(MailboxResult {
            agent_path: AgentPath::root().join("a"),
            status: MailboxStatus::Closed,
            result: None,
            denied_tools: vec![],
        });
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { .. } => {}
            other => panic!("expected Progress, got {other:?}"),
        }
        assert_no_event(&mut fx).await; // lone Closed → no Batch
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn batch_flushed_results_are_not_repeated() {
        // After a flush, the batch is empty: a second generation of results
        // produces its own batch, without stale reports from the first.
        let mut fx = fixture();
        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("gen1"));
        let _ = next_event(&mut fx).await;
        let _ = next_event(&mut fx).await; // Batch gen1

        finish_and_post(&fx, "b", MailboxStatus::Ok, Some("gen2"));
        let _ = next_event(&mut fx).await;
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1);
                assert!(reports[0].message.contains("gen2"));
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn watcher_exits_on_cancel() {
        let mailbox = Arc::new(MailboxHub::new());
        let registry = Arc::new(Mutex::new(AgentRegistry::new(MultiAgentConfig::enabled())));
        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let handle =
            spawn_watcher(mailbox.clone(), registry, None, Some(tx), cancel.clone());
        cancel.cancel();

        let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
        assert!(result.is_ok(), "watcher should exit on cancel");
    }

    #[tokio::test]
    async fn watchdog_exits_on_cancel() {
        let mailbox = Arc::new(MailboxHub::new());
        let registry = Arc::new(Mutex::new(AgentRegistry::new(MultiAgentConfig::enabled())));
        let (tx, _rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let handle =
            spawn_watcher_with_watchdog(mailbox.clone(), registry, None, Some(tx), cancel.clone());
        cancel.cancel();

        let result = tokio::time::timeout(std::time::Duration::from_secs(3), handle).await;
        assert!(result.is_ok(), "watchdog should exit on cancel");
    }

    #[tokio::test]
    async fn concurrent_results_all_land_in_one_batch() {
        // Two results posted before the watcher processes — both Progress
        // events fire, and one Batch carries both reports.
        let mut fx = fixture();
        fx.mailbox.register(&AgentPath::root().join("a"));
        fx.mailbox.register(&AgentPath::root().join("b"));

        fx.mailbox.post_result(MailboxResult {
            agent_path: AgentPath::root().join("a"),
            status: MailboxStatus::Ok,
            result: Some("a done".to_string()),
            denied_tools: vec![],
        });
        fx.mailbox.post_result(MailboxResult {
            agent_path: AgentPath::root().join("b"),
            status: MailboxStatus::Error,
            result: Some("b failed".to_string()),
            denied_tools: vec![],
        });

        let mut statuses = Vec::new();
        for _ in 0..2 {
            match next_event(&mut fx).await {
                ChildResultEvent::Progress { agent_path, status, .. } => {
                    assert_ne!(agent_path, "root/none");
                    statuses.push(status);
                }
                other => panic!("expected Progress, got {other:?}"),
            }
        }
        assert!(statuses.contains(&"ok".to_string()));
        assert!(statuses.contains(&"error".to_string()));

        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 2);
                assert!(reports.iter().any(|r| r.status == "ok"));
                assert!(reports.iter().any(|r| r.status == "error"));
            }
            other => panic!("expected Batch, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    // ── Focus summarizer integration ──

    /// Minimal mock provider: `chat()` sleeps `delay`, then answers a
    /// canned summary JSON. The delay models slow providers (mimo took
    /// 21-30 s per one-sentence summary in session 20260903_d8fc41dc),
    /// scaled down for tests.
    struct SummaryStub {
        delay: std::time::Duration,
    }

    #[async_trait::async_trait]
    impl agent_base::llm_trait::LlmProvider for SummaryStub {
        async fn stream(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
            Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
                futures_util::stream::empty(),
            )))
        }

        async fn chat(
            &self,
            _request: agent_base::llm_trait::ChatRequest,
        ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            Ok(agent_base::llm_trait::ChatResponse {
                content: r#"{"summary": "mock 摘要"}"#.to_string(),
                tool_calls: vec![],
                usage: Default::default(),
                finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
                raw: None,
                reasoning_content: None,
                thinking_signature: None,
            })
        }

        fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
            Default::default()
        }

        fn info(&self) -> agent_base::llm_trait::ProviderInfo {
            agent_base::llm_trait::ProviderInfo {
                name: "summary-stub".to_string(),
                model: "stub".to_string(),
                version: None,
            }
        }
    }

    #[tokio::test]
    async fn progress_summary_flows_from_summarizer() {
        let summarizer = Arc::new(ProgressSummarizer::new(
            Arc::new(SummaryStub {
                delay: std::time::Duration::ZERO,
            }),
            std::time::Duration::from_secs(5),
        ));
        let mut fx = fixture_with_summarizer(Some(summarizer));

        spawn_running(&fx, "a");
        fx.registry
            .lock()
            .unwrap()
            .set_task(&AgentPath::root().join("a"), "分析任务".to_string());

        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("report text"));

        // Three events, order-tolerant except for the plain-first rule: a
        // synchronous plain Progress (summary None) precedes everything, the
        // detached summary Progress and the Batch race each other, and each
        // must arrive with its payload.
        let mut saw_plain = false;
        let mut saw_summary = false;
        let mut batch_ok = false;
        for _ in 0..3 {
            match next_event(&mut fx).await {
                ChildResultEvent::Progress { status, summary, .. } => {
                    assert_eq!(status, "ok");
                    match summary.as_deref() {
                        None => saw_plain = true,
                        Some("mock 摘要") => saw_summary = true,
                        other => panic!("unexpected summary {other:?}"),
                    }
                }
                ChildResultEvent::Batch { reports } => {
                    // The batch still carries the full report — Focus does
                    // not compress.
                    assert_eq!(reports.len(), 1);
                    assert_eq!(reports[0].result.as_deref(), Some("report text"));
                    batch_ok = true;
                }
            }
        }
        assert!(saw_plain, "synchronous plain Progress must arrive first-class");
        assert!(saw_summary, "Progress with summary must arrive");
        assert!(batch_ok, "Batch with full report must arrive");
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn slow_summary_does_not_delay_batch() {
        // The regression for session 20260903_d8fc41dc: a slow summary used
        // to be awaited on the wake path, adding its full timeout to every
        // batch. Now the Batch must beat a slow summary by a wide margin.
        let summarizer = Arc::new(ProgressSummarizer::new(
            Arc::new(SummaryStub {
                delay: std::time::Duration::from_millis(500),
            }),
            std::time::Duration::from_secs(5),
        ));
        let mut fx = fixture_with_summarizer(Some(summarizer));

        spawn_running(&fx, "a");
        let started = std::time::Instant::now();
        finish_and_post(&fx, "a", MailboxStatus::Ok, Some("report text"));

        // First: the synchronous plain Progress (summary None) — the child's
        // return is announced immediately, never held hostage by Focus.
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { summary, .. } => {
                assert!(summary.is_none(), "first notice must be the plain one");
            }
            other => panic!("expected plain Progress first, got {other:?}"),
        }

        // Second: the Batch, well under the summary's 500 ms delay.
        match next_event(&mut fx).await {
            ChildResultEvent::Batch { reports } => {
                assert_eq!(reports.len(), 1);
            }
            ChildResultEvent::Progress {
                summary: Some(_), ..
            } => panic!("summary raced ahead of Batch"),
            ChildResultEvent::Progress { summary: None, .. } => {
                panic!("only one plain Progress expected")
            }
        }
        assert!(
            started.elapsed() < std::time::Duration::from_millis(400),
            "Batch must not wait for the summary (took {:?})",
            started.elapsed()
        );

        // The summary still lands afterwards for the user.
        match next_event(&mut fx).await {
            ChildResultEvent::Progress { summary, .. } => {
                assert_eq!(summary.as_deref(), Some("mock 摘要"));
            }
            other => panic!("expected late Progress, got {other:?}"),
        }
        fx.cancel.cancel();
    }

    #[tokio::test]
    async fn closed_progress_skips_summarizer() {
        let summarizer = Arc::new(ProgressSummarizer::new(
            Arc::new(SummaryStub {
                delay: std::time::Duration::ZERO,
            }),
            std::time::Duration::from_secs(5),
        ));
        let mut fx = fixture_with_summarizer(Some(summarizer));

        spawn_running(&fx, "w");
        finish_and_post(&fx, "w", MailboxStatus::Closed, None);

        match next_event(&mut fx).await {
            ChildResultEvent::Progress { status, summary, .. } => {
                assert_eq!(status, "closed");
                assert!(summary.is_none(), "closed needs no LLM call");
            }
            other => panic!("expected Progress, got {other:?}"),
        }
        assert_no_event(&mut fx).await; // lone Closed → no Batch
        fx.cancel.cancel();
    }
}
