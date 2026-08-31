//! Live-concurrency gate (`AgentExecutionLimiter`, design doc §7.3).
//!
//! ## Semantics — stated honestly
//!
//! A slot covers a child agent's **whole live period** (spawn success →
//! close/cancel/abort), *not* the time slices in which it is executing a task:
//! `run_child_loop` returns to `task_rx.recv()` after each task and stays
//! alive. So this counts "live, not-yet-closed children".
//!
//! ## Overlap with `max_sub_agents`
//!
//! Both the registry gate and this limiter count live children. The registry
//! gate (`max_sub_agents`) shares its checkpoint with the depth check and stays
//! the primary limit; this limiter is an **independent session-level knob**
//! for temporarily tightening concurrency without changing deployment config.
//! It is **disabled by default** (`ControlConfig::max_concurrency = None` →
//! an unlimited limiter that still tracks `current` for observability).
//!
//! If you don't need the second knob, dropping `max_concurrency` loses no
//! functionality (review M-1).

use std::sync::Arc;

// The live-concurrency counter is loom's standard model subject (§12 stage 4):
// under the `loom-check` feature the CAS loop below is checked exhaustively
// (see the `loom_*` tests); a normal build uses plain std atomics.
#[cfg(feature = "loom-check")]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(feature = "loom-check"))]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Errors from [`AgentExecutionLimiter::try_acquire`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LimiterError {
    /// The live-concurrency cap is reached.
    ConcurrencyLimit { max: usize },
}

impl std::fmt::Display for LimiterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConcurrencyLimit { max } => {
                write!(f, "max concurrency reached (limit: {})", max)
            }
        }
    }
}

impl std::error::Error for LimiterError {}

/// Atomic live-concurrency counter with CAS reservation.
///
/// Reservation and release are single-primitive: [`try_acquire`](Self::try_acquire)
/// CAS-increments, and the returned [`ExecutionSlot`] drops to decrement.
/// (v2's two-phase check-then-act was merged into one step — see review notes
/// in design doc §7.3.)
pub struct AgentExecutionLimiter {
    max_concurrency: usize,
    current: AtomicUsize,
}

impl std::fmt::Debug for AgentExecutionLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentExecutionLimiter")
            .field("max_concurrency", &self.max_concurrency)
            .field("current", &self.current.load(Ordering::Relaxed))
            .finish()
    }
}

impl AgentExecutionLimiter {
    /// Create a limiter capped at `max_concurrency` live children.
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            max_concurrency,
            current: AtomicUsize::new(0),
        }
    }

    /// Create a limiter that never rejects (the gate is "not enabled"), while
    /// still tracking the live count for observability. This is the default
    /// (`ControlConfig::max_concurrency = None`).
    pub fn unlimited() -> Self {
        Self::new(usize::MAX)
    }

    /// The configured cap (`usize::MAX` when the gate is disabled).
    pub fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    /// CAS-reserve one concurrency slot.
    ///
    /// Returns [`LimiterError::ConcurrencyLimit`] when the cap is reached.
    /// On success the caller owns an [`ExecutionSlot`] whose drop releases it —
    /// there is no explicit `release()`, so the release path is unique (§7.3).
    pub fn try_acquire(self: &Arc<Self>) -> Result<ExecutionSlot, LimiterError> {
        let mut cur = self.current.load(Ordering::Acquire);
        loop {
            if cur >= self.max_concurrency {
                return Err(LimiterError::ConcurrencyLimit {
                    max: self.max_concurrency,
                });
            }
            match self.current.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(ExecutionSlot {
                        limiter: Arc::clone(self),
                    });
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Number of slots currently held (live children when wired into spawn).
    pub fn current(&self) -> usize {
        self.current.load(Ordering::Acquire)
    }
}

impl Default for AgentExecutionLimiter {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// RAII concurrency-slot token: drop releases the slot (current − 1).
///
/// Lives inside the child task's closure (`ChildCleanup`), so normal exit,
/// panic unwind, and JoinSet abort all release it — the three cleanup paths
/// converge on one mechanism (design doc §4, review B-2).
pub struct ExecutionSlot {
    limiter: Arc<AgentExecutionLimiter>,
}

impl std::fmt::Debug for ExecutionSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionSlot")
            .field("limiter", &self.limiter)
            .finish()
    }
}

impl ExecutionSlot {
    /// The path that will release this slot (for diagnostics).
    pub fn limiter(&self) -> &Arc<AgentExecutionLimiter> {
        &self.limiter
    }
}

impl Drop for ExecutionSlot {
    fn drop(&mut self) {
        self.limiter.current.fetch_sub(1, Ordering::AcqRel);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_release_round_trip() {
        let limiter = Arc::new(AgentExecutionLimiter::new(2));
        assert_eq!(limiter.current(), 0);

        let a = limiter.try_acquire().expect("first slot");
        assert_eq!(limiter.current(), 1);
        let b = limiter.try_acquire().expect("second slot");
        assert_eq!(limiter.current(), 2);

        // Cap reached: third acquire fails without touching the counter.
        assert_eq!(
            limiter.try_acquire().unwrap_err(),
            LimiterError::ConcurrencyLimit { max: 2 }
        );
        assert_eq!(limiter.current(), 2);

        drop(a);
        assert_eq!(limiter.current(), 1);
        let c = limiter.try_acquire().expect("slot after release");
        assert_eq!(limiter.current(), 2);
        drop(c);
        drop(b);
        assert_eq!(limiter.current(), 0);
    }

    #[test]
    fn unlimited_never_rejects_but_tracks() {
        let limiter = Arc::new(AgentExecutionLimiter::unlimited());
        let a = limiter.try_acquire().expect("always ok");
        let b = limiter.try_acquire().expect("always ok");
        assert_eq!(limiter.current(), 2);
        drop(a);
        drop(b);
        assert_eq!(limiter.current(), 0);
    }

    #[test]
    fn zero_cap_rejects_immediately() {
        let limiter = Arc::new(AgentExecutionLimiter::new(0));
        assert_eq!(
            limiter.try_acquire().unwrap_err(),
            LimiterError::ConcurrencyLimit { max: 0 }
        );
        assert_eq!(limiter.current(), 0);
    }

    #[test]
    fn concurrent_acquire_release_conservation() {
        // N threads hammer reserve/release; the counter must end exactly 0
        // and never exceed the cap while running.
        const THREADS: usize = 8;
        const ITERS: usize = 500;
        let limiter = Arc::new(AgentExecutionLimiter::new(THREADS)); // never blocks in practice
        let max_seen = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let limiter = Arc::clone(&limiter);
                let max_seen = Arc::clone(&max_seen);
                std::thread::spawn(move || {
                    for _ in 0..ITERS {
                        let slot = limiter.try_acquire().expect("cap == threads, must succeed");
                        let cur = limiter.current();
                        max_seen.fetch_max(cur, Ordering::AcqRel);
                        drop(slot);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(limiter.current(), 0, "counter conserved to exactly 0");
        assert!(max_seen.load(Ordering::Acquire) <= THREADS);
    }

    /// §12 stage-4: exhaustive interleaving check of the CAS reservation
    /// loop (run with the `loom-check` feature). Cap 1, two acquirers each
    /// holding-then-retrying once. The safety property is *exclusivity*:
    /// while a thread holds the only slot, `current()` must read exactly 1
    /// (a double booking would read 2). Total bookings are 2–4 depending on
    /// how the rounds serialize — releases re-admit, which is the point of
    /// the RAII slot — so the count is bounded, not fixed.
    #[test]
    #[cfg(feature = "loom-check")]
    fn loom_cap_one_exclusivity_across_serialized_rounds() {
        loom::model(|| {
            let limiter = Arc::new(AgentExecutionLimiter::new(1));
            let wins = Arc::new(loom::sync::atomic::AtomicUsize::new(0));
            let mut handles = Vec::new();
            for _ in 0..2 {
                let l = Arc::clone(&limiter);
                let w = Arc::clone(&wins);
                handles.push(loom::thread::spawn(move || {
                    if let Ok(slot) = l.try_acquire() {
                        // Round 1: this thread alone holds the only slot.
                        w.fetch_add(1, Ordering::Relaxed);
                        // Exclusivity: cap 1 + no concurrent co-holder means
                        // the counter reads exactly 1 while we hold.
                        assert_eq!(l.current(), 1, "the holder is the only booking");
                        drop(slot);
                        // Round 2: after release, the same thread retries —
                        // the other may or may not have observed its own
                        // rejection by now, but the cap still admits one.
                        if let Ok(_s) = l.try_acquire() {
                            w.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            let total = wins.load(Ordering::Relaxed);
            // ≥ 2: T-a's first win plus some later re-admission (retry or the
            // other thread after the release); ≤ 4: two rounds per thread.
            assert!((2..=4).contains(&total), "bookings bounded: {total}");
            assert_eq!(limiter.current(), 0, "every slot released");
        });
    }
}
