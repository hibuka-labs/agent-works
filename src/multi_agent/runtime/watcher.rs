//! Background watcher task — the fan-in coordinator for child results.
//!
//! Session 20260903_0cf95e79 redesign: instead of waking the parent for
//! every single result (which pushed the "are we done yet?" decision into
//! the parent's LLM — it misread a frozen inventory `tool_count` as a stall
//! and started doing the children's work itself), the watcher now owns the
//! coordination:
//!
//! - **Progress** — each result is announced for the *user* immediately
//!   (Ok: Focus summary once wired, plain notice otherwise; Error: the
//!   error's first line, ANSI/control sanitized; Closed: plain notice).
//!   Progress never
//!   wakes the parent agent.
//! - **Batch** — once the registry is quiescent (see
//!   [`AgentRegistry::quiescent`]: nobody executing, nothing queued, and
//!   every registered agent has delivered ≥1 result) and at least one
//!   non-Closed report is pending, the watcher emits one `Batch` carrying
//!   every full report. The parent wakes exactly once per generation of
//!   children and runs one synthesis turn.
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
//!   Ok results get a plain Progress notice synchronously, then the summary
//!   runs detached and follows as a second Progress event whenever Focus
//!   answers (or not at all on failure). A Progress event may therefore
//!   arrive after the Batch event, and the same agent may Progress twice —
//!   consumers must be idempotent (phimint's UI is: mark-finished is a
//!   no-op on an already-finished entry). Closed results need no summary
//!   and are announced synchronously. Error results skip Focus entirely:
//!   one synchronous Progress carrying the raw error's first line (session
//!   20260906_0f6d4341 — the user wants the real reason, not a paraphrase).
//! - Quiescence timing is **derived, not marked** (session
//!   20260904_c6559510): the child loop records facts at dequeue and before
//!   each post (`note_posted` precedes the post's seq bump), so "result
//!   drained ⇒ producer settled" holds structurally — no Done-before-post
//!   comment contract for callers to violate, and no phantom-idle window
//!   while a sibling task is still queued (`queue_len > 0` blocks the
//!   predicate directly). The delivery clause additionally seals the
//!   spawn→send window: a freshly spawned agent has zero deliveries, so a
//!   sibling's result cannot fire a batch that would exclude it.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::outcome::format_child_result;
use crate::focus::ProgressSummarizer;
use crate::multi_agent::mailbox::{MailboxHub, MailboxResult, MailboxStatus};
use crate::multi_agent::path::AgentPath;
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
    /// parent agent. Ok results with a wired summarizer are emitted
    /// **twice**: first synchronously with `summary: None` (the plain
    /// "returned" notice, the moment the child returns), then once more with
    /// the Focus summary when it lands (no second event on Focus failure).
    /// **Error results are emitted once**, synchronously, with `summary`
    /// carrying the raw first line of the child's error text — the user
    /// wants the real reason, not an LLM paraphrase of it (session
    /// 20260906_0f6d4341). Consumers must treat repeated Progress for the
    /// same agent as idempotent.
    Progress {
        /// The child agent's path.
        agent_path: String,
        /// Status: "ok", "error", or "closed".
        status: String,
        /// For Ok: the Focus-generated summary shown as a follow-up line.
        /// For Error: the error's first line, sanitized (ANSI stripped,
        /// control characters blanked) and truncated at 160 chars.
        /// `None` → the consumer shows a plain notice.
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

/// Cap on the raw error reason surfaced in a Progress notice (fits a TUI
/// line or two; the full text still reaches the parent via the batch).
const MAX_ERROR_REASON_CHARS: usize = 160;

/// Strip ANSI escape sequences (CSI: ESC `[` params final byte). Provider
/// error bodies are gateway-controlled bytes (llm-providers embeds the whole
/// non-2xx response body in `LlmError::api`), and this reason line is headed
/// for the TUI transcript — never let it paint terminal state or leave
/// `[31m` debris behind a dropped ESC.
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            while let Some(&n) = chars.peek() {
                chars.next();
                if ('@'..='~').contains(&n) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Extract the user-visible reason from a child's error text: the first
/// line with visible content — ANSI stripped, control characters (CR
/// splices, BEL, stray ESC…) blanked — truncated with a bounded check so a
/// multi-megabyte single-line body cannot stall the watcher drain loop.
/// `None` → the consumer's terse fallback notice.
fn error_reason(result_text: Option<&str>) -> Option<String> {
    let line = result_text
        .map(strip_ansi)
        .as_deref()?
        .lines()
        .map(|l| {
            l.chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect::<String>()
        })
        .map(|l| l.trim().to_string())
        .find(|l| !l.is_empty())?;
    let mut reason: String = line.chars().take(MAX_ERROR_REASON_CHARS).collect();
    if line.chars().nth(MAX_ERROR_REASON_CHARS).is_some() {
        reason.push('…');
    }
    Some(reason)
}

/// Spawn the background watcher (fan-in coordinator) task.
///
/// Returns a `JoinHandle` that the caller can store (or ignore — the task
/// Production held-batch reap delay: a batch held this long while no agent
/// can still post means quiescence failed on facts that will never settle
/// (force-closed child, unwind that never lands) — hand the batch over.
pub(crate) const HELD_BATCH_REAP_AFTER: Duration = Duration::from_secs(90);

/// exits when the `cancel` token is fired or the channel sender is dropped).
pub fn spawn_watcher(
    mailbox: Arc<MailboxHub>,
    registry: Arc<Mutex<AgentRegistry>>,
    summarizer: Option<Arc<ProgressSummarizer>>,
    child_result_tx: Option<mpsc::UnboundedSender<ChildResultEvent>>,
    cancel: CancellationToken,
    held_reap_after: Duration,
) -> tokio::task::JoinHandle<()> {
    let mut seq_rx = mailbox.subscribe_seq();

    tokio::spawn(async move {
        // Results held until the whole generation has returned.
        let mut batch: Vec<MailboxResult> = Vec::new();
        // When the oldest held report arrived — drives the held-batch reaper.
        let mut batch_first_at: Option<Instant> = None;
        let mut reap_tick = tokio::time::interval(held_reap_after);
        // Consume the interval's immediate first tick so reaping runs on a
        // steady cadence from now on, not at spawn time.
        reap_tick.tick().await;

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    // Shutdown flush: reports already held must reach the
                    // parent — a watcher's death must never bury them.
                    force_handover(&mut batch, &registry, &child_result_tx);
                    break;
                }
                _ = reap_tick.tick() => {
                    // Held-batch reaper (liveness backstop): a batch held past
                    // `held_reap_after` while no agent can still post means
                    // quiescence failed on facts that will never settle —
                    // hand over anyway rather than strand the reports.
                    let held = batch_first_at.is_some_and(|t| t.elapsed() >= held_reap_after);
                    if !held {
                        continue;
                    }
                    let any_working = registry
                        .lock()
                        .unwrap()
                        .list()
                        .iter()
                        .any(|e| !e.closing && (e.in_flight || e.queue_len > 0));
                    if any_working {
                        continue;
                    }
                    tracing::info!(
                        batch_len = batch.len(),
                        held_for_secs = batch_first_at.map(|t| t.elapsed().as_secs()),
                        "watcher: held-batch reaper firing — quiescence unachievable, forcing handover"
                    );
                    force_handover(&mut batch, &registry, &child_result_tx);
                    batch_first_at = None;
                }
                result = seq_rx.changed() => {
                    if result.is_err() {
                        // Hub dropped — same contract as cancel: flush what
                        // the parent would otherwise never see.
                        force_handover(&mut batch, &registry, &child_result_tx);
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
                            // plain "closed" notice is enough. Any Focus
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
                                (_, MailboxStatus::Error) => {
                                    // Errors skip Focus entirely: the user
                                    // wants the real reason (e.g. "LLM call
                                    // failed: HTTP request failed: …"), and
                                    // an LLM paraphrase of an error adds a
                                    // second delayed notice plus a 30 s
                                    // timeout risk on a call that analyzes a
                                    // failure (session 20260906_0f6d4341:
                                    // both Error children surfaced as a
                                    // terse "execution error" followed by a vague
                                    // paraphrase while the raw text went
                                    // only to the parent). The raw first
                                    // line goes out synchronously instead.
                                    let _ = tx.send(ChildResultEvent::Progress {
                                        agent_path: r.agent_path.to_string(),
                                        status: status.to_string(),
                                        summary: error_reason(r.result.as_deref()),
                                    });
                                }
                                (Some(s), MailboxStatus::Ok) => {
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
                        if batch_first_at.is_none() {
                            batch_first_at = Some(Instant::now());
                        }
                        batch.push(r);
                    }

                    // Quiescence: nobody working, nothing queued, and every
                    // registered agent has delivered ≥1 result (delivery
                    // completeness seals the spawn→send window) — plus at
                    // least one real (non-Closed) report pending → wake the
                    // parent once with everything.
                    {
                        let reg = registry.lock().unwrap();
                        let q = reg.quiescent();
                        if !q {
                            let snapshot = reg.snapshot();
                            tracing::info!(
                                batch_len = batch.len(),
                                agent_count = snapshot.agents.len(),
                                agents = ?snapshot.agents.iter().map(|a| {
                                    format!("{}: status={}, tool_calls={}, pending={}", a.path, a.status, a.tool_calls, a.pending_results)
                                }).collect::<Vec<_>>(),
                                "watcher: quiescent=false, holding batch"
                            );
                        }
                    }
                    if batch.is_empty() || !registry.lock().unwrap().quiescent() {
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
                        batch_first_at = None;
                        continue;
                    }
                    // Stamp the delivery fact BEFORE draining: everything in
                    // this batch is now handed over, so `pending_results` on
                    // list_agents drops to zero the moment the batch fires.
                    let batch_paths: Vec<AgentPath> = {
                        let mut seen = std::collections::BTreeSet::new();
                        batch
                            .iter()
                            .filter(|r| !matches!(r.status, MailboxStatus::Closed))
                            .filter(|r| seen.insert(r.agent_path.to_string()))
                            .map(|r| r.agent_path.clone())
                            .collect()
                    };
                    {
                        let mut reg = registry.lock().unwrap();
                        for path in &batch_paths {
                            reg.note_batch_handed_over(path);
                        }
                    }
                    let reports = batch
                        .drain(..)
                        .map(|r| format_child_result(&r))
                        .collect();
                    batch_first_at = None;
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

/// Fire a held batch unconditionally — the liveness backstop shared by the
/// shutdown flush and the held-batch reaper.
///
/// The normal handover waits for registry quiescence, but registry facts can
/// settle into a state quiescence can never accept (a force-closed child
/// whose unwind never lands, an entry left mid-flight by an abnormal exit):
/// without this, held reports are stranded until process death and the
/// parent's final report is written blind (session 20260914_50cf809d — two
/// specialist reports lost). Semantics mirror the main handover path: drop
/// Closed-only batches, stamp `note_batch_handed_over` first, then drain.
fn force_handover(
    batch: &mut Vec<MailboxResult>,
    registry: &Arc<Mutex<AgentRegistry>>,
    child_result_tx: &Option<mpsc::UnboundedSender<ChildResultEvent>>,
) {
    if batch.is_empty() {
        return;
    }
    let any_real = batch
        .iter()
        .any(|b| !matches!(b.status, MailboxStatus::Closed));
    if !any_real {
        // Only close notifications — nothing to synthesize.
        batch.clear();
        return;
    }
    let batch_paths: Vec<AgentPath> = {
        let mut seen = std::collections::BTreeSet::new();
        batch
            .iter()
            .filter(|r| !matches!(r.status, MailboxStatus::Closed))
            .filter(|r| seen.insert(r.agent_path.to_string()))
            .map(|r| r.agent_path.clone())
            .collect()
    };
    {
        let mut reg = registry.lock().unwrap();
        for path in &batch_paths {
            reg.note_batch_handed_over(path);
        }
    }
    let reports = batch.drain(..).map(|r| format_child_result(&r)).collect();
    if let Some(tx) = child_result_tx {
        let _ = tx.send(ChildResultEvent::Batch { reports });
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
    held_reap_after: Duration,
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
                held_reap_after,
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
                            p.downcast_ref::<&str>()
                                .map(|s| s.to_string())
                                .or_else(|| p.downcast_ref::<String>().cloned())
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
            tracing::info!(restart_count, "watcher watchdog exiting after restarts");
        }
    })
}

#[cfg(test)]
mod tests;
