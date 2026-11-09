use super::*;
use crate::multi_agent::config::MultiAgentConfig;
use crate::multi_agent::mailbox::MailboxResult;
use crate::multi_agent::path::AgentPath;
use crate::multi_agent::registry::ResultKind;

struct Fixture {
    mailbox: Arc<MailboxHub>,
    registry: Arc<Mutex<AgentRegistry>>,
    rx: mpsc::UnboundedReceiver<ChildResultEvent>,
    cancel: CancellationToken,
    #[allow(dead_code)]
    handle: tokio::task::JoinHandle<()>,
}

/// Watcher + registry fixture. Children move through the same fact
/// points (`note_enqueued` / `note_dequeued` / `note_posted`) the child
/// loop uses, in the same order.
fn fixture() -> Fixture {
    fixture_with_summarizer(None)
}

fn fixture_with_summarizer(summarizer: Option<Arc<ProgressSummarizer>>) -> Fixture {
    fixture_with(Config {
        summarizer,
        held_reap_after: std::time::Duration::from_secs(1),
    })
}

/// Fixture knobs that must differ per test (production uses
/// `HELD_BATCH_REAP_AFTER`; tests keep the reap window above the
/// `assert_no_event` windows except where the reaper is the subject).
struct Config {
    summarizer: Option<Arc<ProgressSummarizer>>,
    held_reap_after: std::time::Duration,
}

fn fixture_with(cfg: Config) -> Fixture {
    let mailbox = Arc::new(MailboxHub::new());
    let registry = Arc::new(Mutex::new(AgentRegistry::new(MultiAgentConfig::enabled())));
    let (tx, rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let handle = spawn_watcher_with_watchdog(
        mailbox.clone(),
        registry.clone(),
        cfg.summarizer,
        Some(tx),
        cancel.clone(),
        cfg.held_reap_after,
    );
    Fixture {
        mailbox,
        registry,
        rx,
        cancel,
        handle,
    }
}

/// Register a child in both the mailbox and the registry with a task
/// dequeued and executing — the facts `send_task` + the child loop's
/// dequeue produce.
fn spawn_running(fx: &Fixture, name: &str) {
    let path = AgentPath::root().join(name);
    fx.mailbox.register(&path);
    fx.registry.lock().unwrap().register(&path, 1).unwrap();
    fx.registry.lock().unwrap().note_enqueued(&path);
    fx.registry.lock().unwrap().note_dequeued(&path);
}

/// The child loop's terminal order: record the delivery fact *before*
/// posting the result (see `note_posted` in `spawn.rs`). The registry
/// result kind mirrors production's mapping (a delivered error is
/// `Failed`, a closed delivery is `Cancelled`).
fn finish_and_post(fx: &Fixture, name: &str, status: MailboxStatus, text: Option<&str>) {
    let path = AgentPath::root().join(name);
    let kind = match &status {
        MailboxStatus::Ok => ResultKind::Success,
        MailboxStatus::Error => ResultKind::Failed(text.unwrap_or("error").to_string()),
        MailboxStatus::Closed => ResultKind::Cancelled,
    };
    fx.registry.lock().unwrap().note_posted(&path, kind);
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
async fn batch_fire_clears_pending_results() {
    // Session 20260904_841ed65b fix: the moment a batch fires, its
    // members are "handed over" — list_agents must read pending_results
    // == 0 from then on, or a mid-turn parent would keep seeing a
    // delivery gap that has already been closed. The stamp happens
    // BEFORE tx.send, so reading the registry after receiving the Batch
    // is deterministic.
    let mut fx = fixture();
    for name in ["a", "b"] {
        spawn_running(&fx, name);
    }
    let pending = |fx: &Fixture, name: &str| {
        fx.registry
            .lock()
            .unwrap()
            .snapshot()
            .agents
            .into_iter()
            .find(|a| a.path == format!("root/{name}"))
            .expect("agent registered")
            .pending_results
    };

    finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
    finish_and_post(&fx, "b", MailboxStatus::Ok, Some("b done"));
    // Both Progress notices come first. (The pre-batch pending==1 state
    // is asserted in registry.rs — here the watcher may legitimately
    // have stamped+sent the batch before we get scheduled, so only the
    // post-Batch state is deterministic.)
    next_event(&mut fx).await;
    next_event(&mut fx).await;

    match next_event(&mut fx).await {
        ChildResultEvent::Batch { reports } => assert_eq!(reports.len(), 2),
        other => panic!("expected Batch, got {other:?}"),
    }
    assert_eq!(pending(&fx, "a"), 0, "handed over with the batch");
    assert_eq!(pending(&fx, "b"), 0, "handed over with the batch");
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

/// Variant-B regression (session 20260914_50cf809d): a held report must
/// fire once the force-closed sibling is marked closing — even though it
/// will never deliver a real result. Before the `closing` fact, the
/// killed child's `in_flight=true, results_posted=0` blocked quiescence
/// forever and the report was lost (the parent's final review was then
/// written blind, claiming "both specialists completed").
#[tokio::test]
async fn force_closed_child_does_not_block_batch() {
    let mut fx = fixture();
    for name in ["a", "b"] {
        spawn_running(&fx, name);
    }

    // a finishes; held because b is in-flight with no result.
    finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
    let _ = next_event(&mut fx).await; // Progress a
    assert_no_event(&mut fx).await;

    // The parent force-closes b: terminal fact lands at cancel time,
    // then the child's unwind eventually posts Closed (the wake).
    fx.registry
        .lock()
        .unwrap()
        .note_closing(&AgentPath::root().join("b"));
    finish_and_post(&fx, "b", MailboxStatus::Closed, None);
    let _ = next_event(&mut fx).await; // Progress b (closed)

    match next_event(&mut fx).await {
        ChildResultEvent::Batch { reports } => {
            let paths: Vec<&str> = reports.iter().map(|r| r.agent_path.as_str()).collect();
            assert!(
                paths.contains(&"root/a"),
                "a's held report must be delivered, got {paths:?}"
            );
        }
        other => panic!("expected Batch after close, got {other:?}"),
    }
    fx.cancel.cancel();
}

/// A watcher's death must never bury held reports: on cancel the batch
/// is flushed to the parent (session 20260914_50cf809d — the stranded
/// batch died with the process).
#[tokio::test]
async fn watcher_shutdown_flush_delivers_held_batch() {
    let mut fx = fixture();
    for name in ["a", "b"] {
        spawn_running(&fx, name);
    }

    // a's report held: b is in-flight and (in this test) never settles —
    // no closing fact, no Closed post, no wake. Only the shutdown flush
    // can deliver it.
    finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
    let _ = next_event(&mut fx).await; // Progress a
    assert_no_event(&mut fx).await;

    fx.cancel.cancel();
    match next_event(&mut fx).await {
        ChildResultEvent::Batch { reports } => {
            let paths: Vec<&str> = reports.iter().map(|r| r.agent_path.as_str()).collect();
            assert_eq!(paths, vec!["root/a"], "held report flushed on shutdown");
        }
        other => panic!("expected Batch on shutdown flush, got {other:?}"),
    }
}

/// The reaper backstop: a batch held past the reap window while no agent
/// can still post is handed over even without any wake (here: b was
/// closed and its unwind never posts anything).
#[tokio::test]
async fn held_batch_reaper_fires_when_nothing_can_post() {
    let mut fx = fixture_with(Config {
        summarizer: None,
        held_reap_after: std::time::Duration::from_millis(100),
    });
    for name in ["a", "b"] {
        spawn_running(&fx, name);
    }

    finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
    let _ = next_event(&mut fx).await; // Progress a
    assert_no_event(&mut fx).await; // held — b in-flight

    // b force-closed; its Closed post NEVER arrives (unwind never lands).
    fx.registry
        .lock()
        .unwrap()
        .note_closing(&AgentPath::root().join("b"));

    match next_event(&mut fx).await {
        ChildResultEvent::Batch { reports } => {
            let paths: Vec<&str> = reports.iter().map(|r| r.agent_path.as_str()).collect();
            assert_eq!(paths, vec!["root/a"], "reaper must deliver the held report");
        }
        other => panic!("expected reaper Batch, got {other:?}"),
    }
    fx.cancel.cancel();
}

#[tokio::test]
async fn watcher_exits_on_cancel() {
    let mailbox = Arc::new(MailboxHub::new());
    let registry = Arc::new(Mutex::new(AgentRegistry::new(MultiAgentConfig::enabled())));
    let (tx, _rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();

    let handle = spawn_watcher(
        mailbox.clone(),
        registry,
        None,
        Some(tx),
        cancel.clone(),
        HELD_BATCH_REAP_AFTER,
    );
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

    let handle = spawn_watcher_with_watchdog(
        mailbox.clone(),
        registry,
        None,
        Some(tx),
        cancel.clone(),
        HELD_BATCH_REAP_AFTER,
    );
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
            ChildResultEvent::Progress {
                agent_path, status, ..
            } => {
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
            ChildResultEvent::Progress {
                status, summary, ..
            } => {
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
    assert!(
        saw_plain,
        "synchronous plain Progress must arrive first-class"
    );
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
        ChildResultEvent::Progress {
            status, summary, ..
        } => {
            assert_eq!(status, "closed");
            assert!(summary.is_none(), "closed needs no LLM call");
        }
        other => panic!("expected Progress, got {other:?}"),
    }
    assert_no_event(&mut fx).await; // lone Closed → no Batch
    fx.cancel.cancel();
}

#[tokio::test]
async fn error_progress_carries_raw_reason_not_a_paraphrase() {
    // Session 20260906_0f6d4341: an Error child used to surface as a
    // terse "execution error" plus a detached Focus paraphrase, while the raw
    // error ("LLM call failed: HTTP request failed: …") went only to
    // the parent. Now the synchronous Progress must carry the raw
    // first line itself, and Focus must not be called for errors —
    // exactly one Progress, no follow-up.
    let summarizer = Arc::new(ProgressSummarizer::new(
        Arc::new(SummaryStub {
            delay: std::time::Duration::ZERO,
        }),
        std::time::Duration::from_secs(5),
    ));
    let mut fx = fixture_with_summarizer(Some(summarizer));

    spawn_running(&fx, "e");
    finish_and_post(
        &fx,
        "e",
        MailboxStatus::Error,
        Some(
            "LLM call failed: HTTP request failed: error sending request for url (https://example.invalid/v1/messages)",
        ),
    );

    match next_event(&mut fx).await {
        ChildResultEvent::Progress {
            status, summary, ..
        } => {
            assert_eq!(status, "error");
            let reason = summary.expect("error Progress must carry the raw reason");
            assert!(reason.starts_with("LLM call failed"), "{reason}");
            assert!(reason.contains("error sending request"), "{reason}");
            assert!(!reason.contains("mock 摘要"), "paraphrase leaked: {reason}");
        }
        other => panic!("expected Progress, got {other:?}"),
    }
    // The lone Error still wakes the parent (unlike lone Closed), with
    // the full raw error in the report — then nothing further (no
    // paraphrase Progress must follow).
    match next_event(&mut fx).await {
        ChildResultEvent::Batch { reports } => {
            assert_eq!(reports.len(), 1);
            assert_eq!(
                reports[0].result.as_deref(),
                Some(
                    "LLM call failed: HTTP request failed: error sending request for url (https://example.invalid/v1/messages)"
                )
            );
        }
        other => panic!("expected Batch, got {other:?}"),
    }
    assert_no_event(&mut fx).await;
    fx.cancel.cancel();
}

#[tokio::test]
async fn error_reason_truncates_and_skips_blank_lines() {
    let long = format!("{}…tail", "x".repeat(300));
    // Blank lines are skipped; the FIRST content line wins over later
    // ones (real errors are multi-line: "LLM call failed:\n  caused by").
    assert_eq!(
        error_reason(Some("\n  \nboom: first line is the meat")),
        Some("boom: first line is the meat".to_string())
    );
    assert_eq!(
        error_reason(Some("first line\nsecond line\nthird")),
        Some("first line".to_string())
    );
    // Exact boundary: 160 chars passes through untouched, 161 gains the
    // ellipsis (pins off-by-one at MAX_ERROR_REASON_CHARS).
    let at = "x".repeat(MAX_ERROR_REASON_CHARS);
    let r = error_reason(Some(&at)).unwrap();
    assert_eq!(r.chars().count(), MAX_ERROR_REASON_CHARS);
    assert!(!r.ends_with('…'));
    let over = "x".repeat(MAX_ERROR_REASON_CHARS + 1);
    let r = error_reason(Some(&over)).unwrap();
    assert_eq!(r.chars().count(), MAX_ERROR_REASON_CHARS + 1);
    assert!(r.ends_with('…'));
    // Far over the limit: 300 chars → 160 + ellipsis, tail dropped.
    let truncated = error_reason(Some(&long)).unwrap();
    assert_eq!(truncated.chars().count(), 161, "160 chars + ellipsis");
    assert!(truncated.ends_with('…'));
    assert_eq!(error_reason(Some("  \n\n  ")), None);
    assert_eq!(error_reason(None), None);
}

#[tokio::test]
async fn error_reason_neutralizes_control_chars_and_ansi() {
    // Provider error bodies are gateway-controlled bytes headed for the
    // TUI: escape sequences must not paint terminal state (or leave
    // "[31m" debris behind a dropped ESC), CR must not splice text, and
    // a pure-control first line must not shadow the real reason.
    assert_eq!(
        error_reason(Some("\x1b[31mError: boom\x1b[0m")),
        Some("Error: boom".to_string())
    );
    // CR mid-line becomes a space instead of a silent splice (A\rB → AB).
    assert_eq!(error_reason(Some("A\rB")), Some("A B".to_string()));
    // Tab stays readable rather than gluing words together.
    assert_eq!(
        error_reason(Some("error:\tboom")),
        Some("error: boom".to_string())
    );
    // A control-garbage first line is skipped, not shown invisibly.
    assert_eq!(
        error_reason(Some("\x07\nreal reason")),
        Some("real reason".to_string())
    );
    // Sanitization composes with truncation (still bounded at 160+1).
    let dirty = format!("\x1b[1m{}", "y".repeat(300));
    let r = error_reason(Some(&dirty)).unwrap();
    assert_eq!(r.chars().count(), 161);
    assert!(r.ends_with('…'));
}

#[tokio::test]
async fn error_progress_with_none_result_falls_back_to_plain_notice() {
    // Error with no result text: error_reason(None) → the Progress
    // carries summary: None and the consumer shows its terse fallback
    // notice. Focus still must not run, and the parent still gets the
    // batch with the None report.
    let summarizer = Arc::new(ProgressSummarizer::new(
        Arc::new(SummaryStub {
            delay: std::time::Duration::ZERO,
        }),
        std::time::Duration::from_secs(5),
    ));
    let mut fx = fixture_with_summarizer(Some(summarizer));

    spawn_running(&fx, "e");
    finish_and_post(&fx, "e", MailboxStatus::Error, None);

    match next_event(&mut fx).await {
        ChildResultEvent::Progress {
            status, summary, ..
        } => {
            assert_eq!(status, "error");
            assert!(
                summary.is_none(),
                "None error text must fall back to a plain notice"
            );
        }
        other => panic!("expected Progress, got {other:?}"),
    }
    match next_event(&mut fx).await {
        ChildResultEvent::Batch { reports } => {
            assert_eq!(reports.len(), 1);
            assert_eq!(reports[0].result, None);
        }
        other => panic!("expected Batch, got {other:?}"),
    }
    assert_no_event(&mut fx).await;
    fx.cancel.cancel();
}

#[tokio::test]
async fn fresh_spawn_without_delivery_blocks_the_batch() {
    // The spawn→send window: b is registered (spawn returned) but its
    // send_task is still in flight — zero deliveries. a's result must
    // Progress but NOT fire a batch that excludes b. Pins the delivery
    // clause of `quiescent()` at the layer that consumes it.
    let mut fx = fixture();
    spawn_running(&fx, "a");
    let b = AgentPath::root().join("b");
    fx.mailbox.register(&b);
    fx.registry.lock().unwrap().register(&b, 1).unwrap();

    finish_and_post(&fx, "a", MailboxStatus::Ok, Some("a done"));
    let _ = next_event(&mut fx).await; // Progress a
    assert_no_event(&mut fx).await; // b never delivered — batch held

    // b's task lands and the child loop dequeues it.
    {
        let mut reg = fx.registry.lock().unwrap();
        reg.note_enqueued(&b);
        reg.note_dequeued(&b);
    }
    finish_and_post(&fx, "b", MailboxStatus::Ok, Some("b done"));
    let _ = next_event(&mut fx).await; // Progress b

    match next_event(&mut fx).await {
        ChildResultEvent::Batch { reports } => assert_eq!(
            reports
                .iter()
                .map(|r| r.agent_path.as_str())
                .collect::<Vec<_>>(),
            vec!["root/a", "root/b"]
        ),
        other => panic!("expected Batch, got {other:?}"),
    }
    fx.cancel.cancel();
}

#[tokio::test]
async fn close_during_queued_task_drops_phantom_queue_from_quiescence() {
    // Close lands while a task sits queued (sent, never dequeued). The
    // registry entry — and its queue_len — must vanish with the close,
    // and the late notes the child loop may still issue must be no-ops.
    let mut fx = fixture();
    let b = AgentPath::root().join("b");
    fx.mailbox.register(&b);
    {
        let mut reg = fx.registry.lock().unwrap();
        reg.register(&b, 1).unwrap();
        reg.note_enqueued(&b);
    }
    assert!(!fx.registry.lock().unwrap().quiescent());

    fx.registry.lock().unwrap().close(&b); // close wins the race
    assert!(
        fx.registry.lock().unwrap().quiescent(),
        "phantom queue must not outlive the entry"
    );

    // Cleanup's Closed delivery; the child loop's note_posted is a no-op
    // on the removed entry.
    finish_and_post(&fx, "b", MailboxStatus::Closed, None);
    match next_event(&mut fx).await {
        ChildResultEvent::Progress { status, .. } => assert_eq!(status, "closed"),
        other => panic!("expected Progress(closed), got {other:?}"),
    }
    assert_no_event(&mut fx).await;
    fx.cancel.cancel();
}
