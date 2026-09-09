//! Child-result fan-in delivery policy — the UI-agnostic half of the TUI's
//! former `child_results` module (phimint session 20260903_0cf95e79 redesign).
//!
//! The watcher (see [`super::runtime::watcher`]) is the fan-in coordinator and
//! emits two kinds of events, so delivery splits in two:
//!
//! - **Progress** — one child returned. Display-only, never injected into
//!   the parent's context (the parent is deliberately not woken).
//! - **Batch** — every child has returned. This is what wakes the parent:
//!   delivered immediately when the agent is idle, held and flushed right
//!   after the turn ends when it is running.
//!
//! This module owns only the *decision*: pure state plus routing. The routes
//! carry data, not prose — the caller renders the words (UI copy stays with
//! the product) and executes the side effects (transcript line, status-bar
//! notice, synthetic run).

use super::runtime::{ChildReport, ChildResultEvent};

/// What the caller should do with a watcher event.
#[derive(Clone, Debug)]
pub enum ChildResultRoute {
    /// Display-only: a progress event for the transcript (never injected).
    Progress {
        /// The child agent's path (e.g. `"root/analyze-pi"`).
        agent_path: String,
        /// Status: "ok", "error", or "closed".
        status: String,
        /// Focus-generated summary, when one landed.
        summary: Option<String>,
    },
    /// The parent is mid-turn: the batch was held until the turn ends.
    Held {
        /// Total reports now held (including this batch).
        held: usize,
    },
    /// Start a synthetic run carrying the batch reports.
    Batch {
        /// Full reports, in arrival order.
        reports: Vec<ChildReport>,
    },
}

/// Owns batch reports held during a turn and decides their delivery timing.
#[derive(Debug, Default)]
pub struct ChildResultRouter {
    pending_reports: Vec<ChildReport>,
}

impl ChildResultRouter {
    /// Per-report injection cap, in characters. The whole batch must stay well
    /// under the session's `max_message_tokens` safety valve — phimint
    /// session 20260904_c6559510: a 212,996-char / 53,276-token batch (extra
    /// rounds created by the parent's own nudges) exceeded the valve and was
    /// silently popped; the parent then synthesized from memory. ~4
    /// chars/token for mixed CJK/EN, so 24,000 chars ≈ 6k tokens/report; even
    /// 8 reports land near 50k tokens, far under a 120k valve.
    pub const MAX_REPORT_CHARS: usize = 24_000;

    pub fn new() -> Self {
        Self {
            pending_reports: Vec::new(),
        }
    }

    /// Route one freshly delivered watcher event.
    pub fn on_event(&mut self, agent_running: bool, event: ChildResultEvent) -> ChildResultRoute {
        match event {
            ChildResultEvent::Progress {
                agent_path,
                status,
                summary,
            } => ChildResultRoute::Progress {
                agent_path,
                status,
                summary,
            },
            ChildResultEvent::Batch { reports } => {
                if agent_running {
                    self.pending_reports.extend(reports);
                    ChildResultRoute::Held {
                        held: self.pending_reports.len(),
                    }
                } else {
                    ChildResultRoute::Batch { reports }
                }
            }
        }
    }

    /// Drain reports held during the turn that just ended, as one batch.
    /// Returns `None` while nothing is pending.
    pub fn flush_when_idle(&mut self) -> Option<Vec<ChildReport>> {
        if self.pending_reports.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.pending_reports))
    }

    /// Clamp one report to [`Self::MAX_REPORT_CHARS`] characters. Returns
    /// `(kept, total)` — `kept` is the (possibly truncated) text to inject,
    /// `total` the original character count so the caller can label the cut.
    ///
    /// The truncation must be visible to the model (so it knows to ask the
    /// child for details); the wording of that marker is the caller's.
    pub fn clamp_report(message: &str) -> (String, usize) {
        let total = message.chars().count();
        if total <= Self::MAX_REPORT_CHARS {
            (message.to_string(), total)
        } else {
            (message.chars().take(Self::MAX_REPORT_CHARS).collect(), total)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(path: &str, message: &str) -> ChildReport {
        ChildReport {
            agent_path: path.to_string(),
            status: "ok".to_string(),
            result: Some(message.to_string()),
            message: format!("[子 agent {path} 已完成]\n{message}"),
        }
    }

    fn progress(path: &str) -> ChildResultEvent {
        ChildResultEvent::Progress {
            agent_path: path.to_string(),
            status: "ok".to_string(),
            summary: None,
        }
    }

    fn batch(reports: Vec<ChildReport>) -> ChildResultEvent {
        ChildResultEvent::Batch { reports }
    }

    #[test]
    fn progress_is_passed_through_as_display_data() {
        let mut router = ChildResultRouter::new();

        match router.on_event(true, progress("root/analyze-pi")) {
            ChildResultRoute::Progress {
                agent_path, status, ..
            } => {
                assert_eq!(agent_path, "root/analyze-pi");
                assert_eq!(status, "ok");
            }
            other => panic!("Progress must pass through, got {other:?}"),
        }

        // Progress never injects, never accumulates.
        assert!(router.flush_when_idle().is_none());
    }

    #[test]
    fn batch_when_idle_is_delivered_now() {
        let mut router = ChildResultRouter::new();

        match router.on_event(
            false,
            batch(vec![report("root/a", "report a"), report("root/b", "report b")]),
        ) {
            ChildResultRoute::Batch { reports } => {
                assert_eq!(reports.len(), 2);
                assert_eq!(reports[0].agent_path, "root/a");
                assert_eq!(reports[1].agent_path, "root/b");
            }
            other => panic!("idle agent must deliver now, got {other:?}"),
        }

        // Nothing was held.
        assert!(router.flush_when_idle().is_none());
    }

    #[test]
    fn batch_when_running_holds_and_counts() {
        let mut router = ChildResultRouter::new();

        match router.on_event(true, batch(vec![report("root/a", "one")])) {
            ChildResultRoute::Held { held } => assert_eq!(held, 1),
            other => panic!("running agent must hold, got {other:?}"),
        }
        match router.on_event(true, batch(vec![report("root/b", "two")])) {
            ChildResultRoute::Held { held } => assert_eq!(held, 2, "count accumulates"),
            other => panic!("running agent must hold, got {other:?}"),
        }

        // The turn ends → held reports flush as one batch, in arrival order.
        let flushed = router.flush_when_idle().expect("held reports must flush");
        assert_eq!(flushed.len(), 2);
        assert_eq!(flushed[0].message, "[子 agent root/a 已完成]\none");
        assert_eq!(flushed[1].message, "[子 agent root/b 已完成]\ntwo");

        // Drained — a second flush is a no-op.
        assert!(router.flush_when_idle().is_none());
    }

    #[test]
    fn progress_between_batches_does_not_break_flush() {
        let mut router = ChildResultRouter::new();

        assert!(matches!(
            router.on_event(true, batch(vec![report("root/a", "one")])),
            ChildResultRoute::Held { .. }
        ));
        // A lone Progress lands while a batch is held — display only.
        assert!(matches!(
            router.on_event(true, progress("root/x")),
            ChildResultRoute::Progress { .. }
        ));
        let flushed = router.flush_when_idle().expect("held batch must flush");
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].agent_path, "root/a", "Progress must not be held");
    }

    #[test]
    fn after_flush_new_batch_is_delivered_not_stale() {
        let mut router = ChildResultRouter::new();

        assert!(matches!(
            router.on_event(true, batch(vec![report("root/first", "one")])),
            ChildResultRoute::Held { .. }
        ));
        assert!(router.flush_when_idle().is_some());

        // Next generation arrives while idle → deliver now, never
        // accumulating into a stale batch.
        match router.on_event(false, batch(vec![report("root/second", "two")])) {
            ChildResultRoute::Batch { reports } => assert_eq!(reports[0].agent_path, "root/second"),
            other => panic!("expected immediate delivery, got {other:?}"),
        }
        assert!(router.flush_when_idle().is_none());
    }

    #[test]
    fn flush_without_pending_is_a_noop() {
        let mut router = ChildResultRouter::new();
        assert!(router.flush_when_idle().is_none());
    }

    /// Session 20260904_c6559510 regression: per-report clamping must keep
    /// every injection far under the session's token valve.
    #[test]
    fn clamp_report_caps_oversized_messages() {
        let long = "x".repeat(ChildResultRouter::MAX_REPORT_CHARS + 5_000);
        let (kept, total) = ChildResultRouter::clamp_report(&long);
        assert_eq!(kept.chars().count(), ChildResultRouter::MAX_REPORT_CHARS);
        assert_eq!(total, ChildResultRouter::MAX_REPORT_CHARS + 5_000);
    }

    #[test]
    fn clamp_report_passes_short_messages_verbatim() {
        let (kept, total) = ChildResultRouter::clamp_report("report a");
        assert_eq!(kept, "report a");
        assert_eq!(total, 8);
    }
}
