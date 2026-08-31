//! Session-level control plane (`AgentControl`, design doc §7.1).
//!
//! Bundles the two resource gates — the rollout [`RolloutBudget`] (§7.2) and
//! the live-concurrency [`AgentExecutionLimiter`] (§7.3) — behind one object
//! the runtime owns as a single `Arc` and hands to whoever needs a gate
//! (spawn chain, tool layer via [`control`](super::MultiAgentRuntime::control),
//! operator status reads).
//!
//! Deliberately **not** here (review, §7.1): no `registry` (the existing
//! [`AgentRegistry`](super::registry::AgentRegistry) is reused, not wrapped), and no
//! `clone_for_child` (one `Arc`, injected at construction).
//!
//! NOTE (doc fact-check): §7.1's pseudocode carries a `session_id: SessionId`
//! field, but `AgentControl::new(&ControlConfig)` has no session at
//! construction time — the runtime learns its parent session later
//! (`set_session_manager`). The field is omitted rather than half-filled;
//! if a real consumer appears it should go where the session is known.

use std::sync::Arc;

use super::budget::RolloutBudget;
use super::config::ControlConfig;
use super::limiter::AgentExecutionLimiter;

/// The two spawn-resource gates plus their config, as one `Arc`.
pub struct AgentControl {
    budget: Arc<RolloutBudget>,
    limiter: Arc<AgentExecutionLimiter>,
}

impl std::fmt::Debug for AgentControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentControl")
            .field("budget", &self.budget)
            .field("limiter", &self.limiter)
            .finish()
    }
}

impl AgentControl {
    /// Build both gates from [`ControlConfig`]. `None` knobs → unlimited
    /// gates that still count for observability (§7.2 / §7.3).
    pub fn new(config: &ControlConfig) -> Self {
        Self {
            budget: Arc::new(RolloutBudget::new(
                config.child_max_tokens,
                config.max_spawns,
            )),
            limiter: Arc::new(match config.max_concurrency {
                Some(max) => AgentExecutionLimiter::new(max),
                None => AgentExecutionLimiter::unlimited(),
            }),
        }
    }

    /// The rollout budget (spawn count + child tokens).
    pub fn budget(&self) -> &Arc<RolloutBudget> {
        &self.budget
    }

    /// The live-concurrency gate.
    pub fn limiter(&self) -> &Arc<AgentExecutionLimiter> {
        &self.limiter
    }

    /// One snapshot of the three control readings (§7.1): budget spend,
    /// cumulative spawns, live concurrency.
    pub fn status(&self) -> ControlStatus {
        ControlStatus {
            used_tokens: self.budget.used_tokens(),
            child_max_tokens: self.budget.child_max_tokens(),
            spawn_count: self.budget.spawn_count(),
            max_spawns: self.budget.max_spawns(),
            live_children: self.limiter.current(),
            max_concurrency: self.limiter.max_concurrency(),
        }
    }
}

/// Read-only snapshot of [`AgentControl`] (for status surfaces / tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlStatus {
    /// Cumulative child tokens metered by hook A.
    pub used_tokens: u64,
    /// Child-token budget cap (`u64::MAX` = disabled).
    pub child_max_tokens: u64,
    /// Committed + uncommitted-rolled-back spawn count.
    pub spawn_count: usize,
    /// Cumulative spawn cap (`usize::MAX` = disabled).
    pub max_spawns: usize,
    /// Children currently holding a slot (live, not yet cleaned up).
    pub live_children: usize,
    /// Live-concurrency cap (`usize::MAX` = disabled).
    pub max_concurrency: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_none_config_yields_observable_unlimited_gates() {
        let c = AgentControl::new(&ControlConfig::default());
        let slot = c.limiter().try_acquire().expect("unlimited");
        let ticket = c.budget().try_reserve_spawn().expect("unlimited");
        assert_eq!(
            c.status(),
            ControlStatus {
                used_tokens: 0,
                child_max_tokens: u64::MAX,
                spawn_count: 1,
                max_spawns: usize::MAX,
                live_children: 1,
                max_concurrency: usize::MAX,
            }
        );
        ticket.commit();
        drop(slot);
        assert_eq!(c.status().live_children, 0);
    }

    #[test]
    fn config_knobs_reach_the_gates() {
        let cfg = ControlConfig {
            child_max_tokens: Some(500),
            max_spawns: Some(3),
            max_concurrency: Some(2),
            ..Default::default()
        };
        let c = AgentControl::new(&cfg);
        let s = c.status();
        assert_eq!(
            (s.child_max_tokens, s.max_spawns, s.max_concurrency),
            (500, 3, 2)
        );
    }
}
