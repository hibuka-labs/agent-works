//! CodeAct action backend (M3 T10) — the [`ActionExecutor`] adapter over the
//! codexec engine.
//!
//! OV-8 keeps codexec free of agent-base: the engine knows nothing about LLM
//! turns or history protocols, and this module is the only bridge. The action
//! wire is the engine's own `<intent>…</intent><code>…</code>` vocabulary
//! (codexec-design §4.4) — `parse` maps the engine parser's verdicts onto the
//! four-state contract (D4+OV-7), and `execute` renders one engine step into
//! an [`Observation`] whose `history_messages` use the default shape A (user
//! message re-injection, action-executor-design §7.0): the assistant message
//! is the batch's intent/code blocks re-rendered in the engine wire format —
//! never a faked tool call — and the observation follows as a plain
//! `[observation]` user message.
//!
//! Routing precondition (the mount's job, M3 T10c): tool-envelope turns go to
//! the tool-call backend; this parse only claims text turns. An envelope turn
//! arriving here is a routing bug and fails loudly rather than nudging.
//!
//! Per-action fates (D9) are reconstructed from the engine's step-level
//! verdict: the fold reports the first failure with its block index, which is
//! exactly the boundary between "ran", "failed", and "never ran" — except on
//! timeout/worker loss, where the round is gone and every fate honestly says
//! so ([`BatchStatus::EnvDied`]).
//!
//! Cell interactions (T10b, OV-6) cross the same seam: a policy `Ask`
//! suspends at the engine's dispatch throat and `ask.user` suspends in its own
//! host call, both answered through the run's [`ExecCtx`] — the engine speaks
//! JSON and knows nothing about the product's human. This module carries the
//! bridge ([`ExecCtxBridge`]); the suspension discipline (answer / cancel /
//! watchdog) stays in `ExecCtx`'s own methods.

use std::sync::Arc;

use agent_base::{
    Action, ActionBackend, ActionExecutor, ActionFormat, ActionOutcome, ChatMessage, ExecCtx,
    ExecError, LlmTurnResult, Observation, ParseError, ParseOutcome, SessionError, SessionHandle,
};
// BatchStatus is referenced only by the [`BatchStatus::EnvDied`] doc link above;
// rustc's unused lint does not count doc links.
#[allow(unused_imports)]
use agent_base::BatchStatus;
use async_trait::async_trait;
use codexec::{CodeCellExecutor, Worker};

/// The model-face cap for one observation (action-executor-design §6.1): the
/// family default, overridable at assembly — a big observation spills to the
/// artifact store and keeps summary + reference instead of flooding.
pub(crate) const DEFAULT_OBSERVATION_CAP: usize = 16_000;

/// The CodeAct action backend: one resident engine behind the seam, plus the
/// interaction bridge — the engine holds the same handle from assembly
/// (`ExecutorBuilder::with_interaction(Arc::clone(&bridge))`), so a cell's
/// consent/ask reaches the run's [`ExecCtx`] through it.
pub struct CodeActExecutor<W: Worker> {
    engine: CodeCellExecutor<W>,
    interact: Arc<ExecCtxBridge>,
    /// Model-face cap for one observation ([`DEFAULT_OBSERVATION_CAP`] unless
    /// the assembly says otherwise).
    observation_cap: usize,
}

impl<W: Worker> CodeActExecutor<W> {
    /// Wrap a built engine together with the interaction bridge mounted into
    /// it. The two ends must be the same handle: one that was never mounted
    /// finds the engine answering holds itself, and a bridge the engine never
    /// saw fails its suspensions closed ("no run context armed").
    pub fn new(engine: CodeCellExecutor<W>, interact: Arc<ExecCtxBridge>) -> Self {
        Self {
            engine,
            interact,
            observation_cap: DEFAULT_OBSERVATION_CAP,
        }
    }

    /// Override the observation cap (assembly). Bodies over it spill to the
    /// engine's artifact store and keep summary + reference (§6.1).
    pub fn with_observation_cap(mut self, cap: usize) -> Self {
        self.observation_cap = cap;
        self
    }

    /// The engine, for hosts that talk to it directly (journal path, packs).
    pub fn engine(&self) -> &CodeCellExecutor<W> {
        &self.engine
    }

    /// Unwrap back into the engine.
    pub fn into_engine(self) -> CodeCellExecutor<W> {
        self.engine
    }
}

#[async_trait]
impl<W: Worker + Send + Sync + 'static> ActionExecutor for CodeActExecutor<W> {
    fn parse(&self, turn: &LlmTurnResult) -> Result<ParseOutcome, ParseError> {
        // Routing precondition: envelope turns belong to the tool backend.
        if turn.is_tool_call {
            return Err(ParseError::Unparseable(
                "tool-envelope turn routed to the CodeAct backend".into(),
            ));
        }
        Ok(classify_text(&turn.full_text))
    }

    async fn execute(
        &mut self,
        actions: Vec<Action>,
        ctx: &ExecCtx,
    ) -> Result<Observation, ExecError> {
        if actions.is_empty() {
            return Err(ExecError::NothingRan {
                reason: "empty batch".into(),
            });
        }
        // D2: the whole batch must decode before anything runs.
        let cells = decode_batch(&actions).map_err(|reason| ExecError::NothingRan { reason })?;
        // T10b: interactions from inside a cell suspend on the run's `ExecCtx`
        // (answer / cancel / watchdog live there) — arm the bridge for the
        // step; interactions only exist inside one.
        self.interact.arm(ctx);
        let obs = self.engine.execute(cells).await;
        self.interact.disarm();
        let hub = self.engine.registry().artifacts();
        Ok(fold(&actions, obs, self.observation_cap, hub.as_deref()))
    }
}

#[async_trait]
impl<W: Worker + Send + Sync + 'static> SessionHandle for CodeActExecutor<W> {
    fn is_alive(&self) -> bool {
        self.engine.is_session_alive()
    }

    async fn rebuild(&mut self) -> Result<(), SessionError> {
        // The engine owns its lifecycle (codexec-design §7): the next step
        // respawns the worker and reports `session_reset` to the model — that
        // report is this backend's rebuild protocol, so there is nothing to
        // force here (D10 stays "one rebuild attempt, no retry loop": the
        // step either runs on a fresh worker or reports the reset).
        Ok(())
    }
}

#[async_trait]
impl<W: Worker + Send + Sync + 'static> ActionBackend for CodeActExecutor<W> {
    fn format(&self) -> ActionFormat {
        ActionFormat::CodeAct
    }

    /// Shape A for a batch with no observation (resume synthesis OV-4, Err
    /// fates D15): the same assistant/user pair as [`render_history`], with
    /// the fate summaries standing in for the engine's text. The fates carry
    /// the kernel's vocabulary (e.g. [`agent_base::INTERRUPTED_SUMMARY`]);
    /// the shape is this backend's own.
    fn render_fates(&self, actions: &[Action], fates: &[ActionOutcome]) -> Vec<ChatMessage> {
        let mut out = String::from("[observation]\n");
        for (i, (a, f)) in actions.iter().zip(fates.iter()).enumerate() {
            let (intent, _) = block_of(a);
            out.push_str(&format!("#{} {}: {}\n", i + 1, intent, f.summary));
        }
        render_pair(actions, out)
    }
}

mod adapt;
mod bridge;
mod render;

use adapt::{classify_text, decode_batch, fold};
use render::{block_of, render_pair};

pub use bridge::ExecCtxBridge;

#[cfg(test)]
mod tests;
