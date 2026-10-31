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

use std::sync::{Arc, Mutex};

use agent_base::{
    Action, ActionBackend, ActionExecutor, ActionFormat, ActionOutcome, ApprovalDecision,
    ApprovalRequest, AskAnswer, AskOption, BatchStatus, ChatMessage, ExecCtx, ExecError,
    LlmTurnResult, Observation, ObservationEvent, ParseError, ParseOutcome, RiskLevel,
    SessionError, SessionHandle,
};
use async_trait::async_trait;
use codexec::observation::CellEvent;
use codexec::{
    Action as EngineAction, ArtifactHub, ArtifactRef, CodeCellExecutor, ConsentRequest,
    ExecError as EngineError, InteractionBridge, Observation as EngineObservation, PolicyDecision,
    Worker, artifact_line,
};
use phi_cap::Capability;
use serde_json::Value;

use crate::script_tool::truncate_output;

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

/// The cell-interaction bridge (T10b, OV-6): codexec's [`InteractionBridge`]
/// over the run's [`ExecCtx`]. The bridge outlives any one step (the engine
/// holds it from assembly) while the context arrives with `execute` — so it
/// keeps an owned handle for the step and rebuilds the context per call, which
/// keeps the suspension discipline (answer / cancel / watchdog) in exactly one
/// place: `ExecCtx`'s own methods.
#[derive(Default)]
pub struct ExecCtxBridge {
    armed: Mutex<Option<ExecCtx>>,
}

impl ExecCtxBridge {
    /// An un-armed bridge: nothing suspends until [`Self::arm`] runs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Arm with the step's run context (a handle clone — the step owns the
    /// original).
    pub fn arm(&self, ctx: &ExecCtx) {
        *self.armed.lock().unwrap() = Some(clone_ctx(ctx));
    }

    /// Disarm at the end of the step.
    pub fn disarm(&self) {
        *self.armed.lock().unwrap() = None;
    }

    /// The armed handle — or why the suspension cannot happen (which fails
    /// closed on the engine side).
    fn current(&self) -> Result<ExecCtx, String> {
        self.armed
            .lock()
            .unwrap()
            .as_ref()
            .map(clone_ctx)
            .ok_or_else(|| "no run context armed (interaction outside a step)".to_string())
    }
}

/// `ExecCtx` is a handle (token + channel + deadline), so cloning the handle
/// is as good as the original. (Struct literal: those fields are the type.)
fn clone_ctx(ctx: &ExecCtx) -> ExecCtx {
    ExecCtx {
        cancel: ctx.cancel.clone(),
        interact: Arc::clone(&ctx.interact),
        interact_timeout: ctx.interact_timeout,
    }
}

#[async_trait]
impl InteractionBridge for ExecCtxBridge {
    async fn consent(&self, req: &ConsentRequest<'_>) -> Result<PolicyDecision, String> {
        let ctx = self.current()?;
        let request = ApprovalRequest {
            title: req.fn_name.to_string(),
            message: consent_message(req),
            action_key: Some(req.fn_name.to_string()),
            risk_level: risk_of(req.cap),
            raw: req.args.cloned(),
            source: None,
        };
        match ctx.request_consent(request).await {
            Ok(ApprovalDecision::AllowOnce) | Ok(ApprovalDecision::AllowAlways) => {
                Ok(PolicyDecision::Allow)
            }
            Ok(ApprovalDecision::Deny) => Ok(PolicyDecision::Deny),
            // Cancel / watchdog / a dead channel: fail closed, and the reason
            // ("interaction cancelled") rides the hold the model sees.
            Err(e) => Err(e.to_string()),
        }
    }

    async fn ask(&self, question: &str, options: &[Value]) -> Result<Value, String> {
        let ctx = self.current()?;
        let options = options
            .iter()
            .enumerate()
            .map(|(i, o)| ask_option(o, i))
            .collect::<Result<Vec<_>, _>>()?;
        match ctx.ask_user(question, &options).await {
            Ok(AskAnswer::Picked { id, label }) => {
                Ok(serde_json::json!({"picked": {"id": id, "label": label}}))
            }
            Ok(AskAnswer::Skipped) => Ok(serde_json::json!({"skipped": true})),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// What the human sees on the consent dialog: the call and its payload (the
/// digest only when the chokepoint holds nothing else — the escape hatch).
fn consent_message(req: &ConsentRequest<'_>) -> String {
    match req.args {
        Some(wire) => format!("{} {wire}", req.fn_name),
        None => format!("{} (args digest {})", req.fn_name, req.args_digest),
    }
}

/// Risk for the approval dialog: reads are safe, anything that writes or
/// leaves the box is sensitive — a cell is code, so sensitive and never
/// destructive (the product's approval face owns the final wording).
fn risk_of(cap: Capability) -> RiskLevel {
    match cap {
        Capability::Pure | Capability::ReadFs => RiskLevel::Safe,
        _ => RiskLevel::Sensitive,
    }
}

/// Wire `{"id", "label", "note"?}` → the ask vocabulary. A bad shape fails
/// the interaction instead of guessing.
fn ask_option(wire: &Value, i: usize) -> Result<AskOption, String> {
    let obj = wire
        .as_object()
        .ok_or_else(|| format!("ask.user option {i} must be an object"))?;
    let field = |k: &str| {
        obj.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("ask.user option {i} needs a string `{k}`"))
    };
    Ok(AskOption {
        id: field("id")?,
        label: field("label")?,
        note: obj.get("note").and_then(Value::as_str).map(str::to_string),
    })
}

/// Text turn → four-state verdict (D4+OV-7). The engine parser is strict
/// (an unclosed tag fails the whole step — D2), so its three verdicts map
/// onto the loop's branches like this:
/// - blocks parsed → [`ParseOutcome::Actions`] (materialized);
/// - no blocks at all → [`ParseOutcome::Empty`] (the model's prose answer —
///   empty text first, same trap as the tool backend);
/// - action intent that never materialized (a tag cut off mid-stream, an
///   `<intent>` with no `<code>`) → [`ParseOutcome::Incomplete`] (light
///   retry);
/// - structurally complete but wrong (an empty `<code>` block) →
///   [`ParseOutcome::Malformed`] (anomaly branch).
fn classify_text(text: &str) -> ParseOutcome {
    match codexec::parse_actions(text) {
        Ok(cells) => ParseOutcome::Actions(
            cells
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    Action::from_code_cell(
                        format!("cell_{i}"),
                        c.intent().to_string(),
                        c.code().to_string(),
                    )
                })
                .collect(),
        ),
        Err(codexec::ParseError::NoBlocks) => ParseOutcome::Empty,
        Err(codexec::ParseError::Malformed { detail, .. }) => {
            if detail.starts_with("unclosed") || detail.contains("without a following") {
                ParseOutcome::Incomplete
            } else {
                ParseOutcome::Malformed
            }
        }
    }
}

/// D2: the full batch decodes before anything runs. Every action must carry
/// the engine block shape `{"intent", "code"}` — this adapter is
/// single-format ([`ActionFormat::CodeAct`]); mixed routing is the mount's
/// job.
fn decode_batch(actions: &[Action]) -> Result<Vec<EngineAction>, String> {
    let mut cells = Vec::with_capacity(actions.len());
    for a in actions {
        if a.format != ActionFormat::CodeAct {
            return Err(format!("action {} is not a CodeAct cell", a.id));
        }
        let Some(obj) = a.args.as_object() else {
            return Err(format!("action {} does not carry the block shape", a.id));
        };
        let field = |k: &str| {
            obj.get(k)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("action {} is missing `{k}`", a.id))
        };
        let intent = field("intent")?;
        let code = field("code")?;
        if code.trim().is_empty() {
            return Err(format!("action {} has empty code", a.id));
        }
        cells.push(EngineAction::CodeCell { intent, code });
    }
    Ok(cells)
}

/// One engine step → the seam's observation (D15: the step's verdict is
/// always an observation; the engine's errors are model-visible content).
fn fold(
    actions: &[Action],
    obs: EngineObservation,
    cap: usize,
    hub: Option<&ArtifactHub>,
) -> Observation {
    let batch_status = match &obs.error {
        // The round died with its worker (or never started on a lost
        // session): batch health says the environment, not the actions.
        Some(EngineError::Timeout { .. } | EngineError::Worker(_) | EngineError::SessionReset) => {
            BatchStatus::EnvDied
        }
        _ => BatchStatus::Completed,
    };
    let (pane, spill) = observation_text(&obs, cap, hub);
    let mut events = observation_events(&obs);
    // A render-level spill is an observation artifact like any other (T13):
    // same typed face, same line in the pane above.
    if let Some(r) = spill {
        events.push(ObservationEvent::Artifact {
            capture_id: r.capture_id,
            kind: r.kind,
            summary: r.summary,
        });
    }
    Observation {
        outcomes: fates(actions, &obs),
        events,
        history_messages: render_pair(actions, pane),
        batch_status,
    }
}

/// The typed channel (T13): engine cell events → seam events. Image bytes
/// stay out (the typed channel has no image face; the `alt` line is in the
/// observation text), and a spilled body's `bytes` are line metadata only —
/// the frozen [`ObservationEvent::Artifact`] shape carries id/kind/summary.
fn observation_events(obs: &EngineObservation) -> Vec<ObservationEvent> {
    obs.events
        .iter()
        .filter_map(|ev| match ev {
            CellEvent::TodoChanged { id, status } => Some(ObservationEvent::Text {
                text: format!("todo {id}: {status}"),
            }),
            CellEvent::Notice { text } => Some(ObservationEvent::Text { text: text.clone() }),
            CellEvent::Image(_) => None,
            CellEvent::Artifact {
                capture_id,
                kind,
                summary,
                ..
            } => Some(ObservationEvent::Artifact {
                capture_id: capture_id.clone(),
                kind: kind.clone(),
                summary: summary.clone(),
            }),
        })
        .collect()
}

/// Per-action fates from the step verdict (D9: a refusal is a fate, never a
/// silent drop). The fold reports the first failure with its block index;
/// blocks before it ran, the failing one failed, blocks after it never ran.
fn fates(actions: &[Action], obs: &EngineObservation) -> Vec<ActionOutcome> {
    let n = actions.len();
    let fate = |i: usize, ok: bool, denied: bool, summary: String| ActionOutcome {
        action_id: actions[i].id.clone(),
        name: actions[i].name.clone(),
        ok,
        denied,
        summary,
        details: None,
    };
    match &obs.error {
        None => (0..n)
            .map(|i| fate(i, true, false, "completed".into()))
            .collect(),
        Some(EngineError::CompileError { index, message }) => (0..n)
            .map(|i| {
                if i == *index {
                    fate(i, false, false, format!("compile_error: {message}"))
                } else {
                    // D2: a compile error stops the round before any block
                    // runs — even the blocks before the bad one did not.
                    fate(
                        i,
                        false,
                        false,
                        format!("not run (compile_error in block #{index})"),
                    )
                }
            })
            .collect(),
        Some(EngineError::CellError {
            index,
            kind,
            message,
        }) => (0..n)
            .map(|i| {
                if i < *index {
                    fate(i, true, false, "completed".into())
                } else if i == *index {
                    fate(i, false, false, format!("{kind}: {message}"))
                } else {
                    fate(
                        i,
                        false,
                        false,
                        format!("not run (step halted at block #{index})"),
                    )
                }
            })
            .collect(),
        // The hold's Display carries the reason ("consent refused",
        // "interaction cancelled") — the model must see why the cell stopped.
        Some(err @ EngineError::PolicyHold { cell_index, .. }) => (0..n)
            .map(|i| {
                if i < *cell_index {
                    fate(i, true, false, "completed".into())
                } else if i == *cell_index {
                    fate(i, false, true, err.to_string())
                } else {
                    fate(
                        i,
                        false,
                        false,
                        format!("not run (step halted on policy hold at block #{cell_index})"),
                    )
                }
            })
            .collect(),
        // The reset report arrives before the step runs (codexec-design §7):
        // nothing ran, and the model must re-observe.
        Some(EngineError::SessionReset) => (0..n)
            .map(|i| fate(i, false, false, "not run (session_reset)".into()))
            .collect(),
        // Timeout / channel loss took the round with it: which cells
        // completed is not knowable from the fold, so every fate says so
        // rather than lie.
        Some(err @ (EngineError::Timeout { .. } | EngineError::Worker(_))) => (0..n)
            .map(|i| fate(i, false, false, format!("fate not reported ({err})")))
            .collect(),
        // Journal failure aborts fail-closed before the worker runs.
        Some(EngineError::Journal(msg)) => (0..n)
            .map(|i| {
                fate(
                    i,
                    false,
                    false,
                    format!("not run (journal failed: {msg}; fail-closed)"),
                )
            })
            .collect(),
    }
}

/// The shape-A pair: the batch's blocks as one assistant message, the
/// observation pane as the user message that follows. Both live renderings
/// (engine text) and synthesized ones (fate summaries) share it.
fn render_pair(actions: &[Action], observation: String) -> Vec<ChatMessage> {
    let blocks: Vec<String> = actions
        .iter()
        .map(|a| {
            let (intent, code) = block_of(a);
            format!("<intent>{intent}</intent>\n<code>{code}</code>")
        })
        .collect();
    vec![
        ChatMessage::assistant(blocks.join("\n")),
        ChatMessage::user(observation),
    ]
}

/// The `{"intent", "code"}` pair of a cell action (empty strings when the
/// shape is wrong — the model still sees the block markers).
fn block_of(action: &Action) -> (String, String) {
    let obj = action.args.as_object();
    let field = |k: &str| {
        obj.and_then(|o| o.get(k))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    (field("intent"), field("code"))
}

/// The observation half of shape A: `[observation]` plus the engine's text
/// stream (stdout carries one `namespace: …` snapshot line per cell; event
/// texts ride as lines; image alts already landed in stdout). Returns the
/// pane plus the reference of anything this render itself spilled (M3c).
fn observation_text(
    obs: &EngineObservation,
    cap: usize,
    hub: Option<&ArtifactHub>,
) -> (String, Option<ArtifactRef>) {
    let mut out = String::from("[observation]\n");
    let mut any = false;
    let line = |text: String, out: &mut String, any: &mut bool| {
        if *any && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&text);
        *any = true;
    };
    if !obs.stdout.is_empty() {
        line(obs.stdout.clone(), &mut out, &mut any);
    }
    if !obs.stderr.is_empty() {
        line(format!("stderr:\n{}", obs.stderr), &mut out, &mut any);
    }
    if let Some(err) = &obs.error {
        line(format!("error: {err}"), &mut out, &mut any);
    }
    for ev in &obs.events {
        let text = match ev {
            CellEvent::TodoChanged { id, status } => format!("todo {id}: {status}"),
            CellEvent::Notice { text } => format!("notice: {text}"),
            // A spilled body's standing reference (T13): the model face is
            // this line — summary + one fetch instruction.
            CellEvent::Artifact {
                capture_id,
                kind,
                bytes,
                summary,
            } => artifact_line(capture_id, kind, *bytes, summary),
            // The alt line is already in stdout (sdk-assembly §4.2 rule 3);
            // the bytes ride the typed channel (see `observation_events`).
            CellEvent::Image(_) => continue,
        };
        line(text, &mut out, &mut any);
    }
    if !any {
        out.push_str("(no output)");
    }
    fit_to_cap(out, cap, hub)
}

/// Cap the observation pane (M3c, action-executor-design §6.1): an over-cap
/// body spills **whole** to the artifact store and the pane keeps the family
/// head/tail preview plus the summary+reference line, fitted inside the cap —
/// the line counts against the budget, so nothing here can trip a hard
/// reject. The store write goes through the store directly (not `hub.spill`):
/// the hub's event queue belongs to the engine's round, and this render
/// happens after the fold drained it — a queued event here would surface a
/// step late. Without a mounted store the family truncation stands and the
/// body is lost as before — marked either way, never silent.
fn fit_to_cap(
    text: String,
    cap: usize,
    hub: Option<&ArtifactHub>,
) -> (String, Option<ArtifactRef>) {
    if text.chars().count() <= cap {
        return (text, None);
    }
    let Some(r) = hub.and_then(|h| h.store().spill("observation", &text).ok()) else {
        return (truncate_output(&text, Some(cap)), None);
    };
    let line = artifact_line(&r.capture_id, &r.kind, r.bytes, &r.summary);
    if line.chars().count() + 1 >= cap {
        // The reference line cannot fit — plain truncation wins (and no typed
        // event points at a reference the model cannot see).
        return (truncate_output(&text, Some(cap)), None);
    }
    let budget = cap - line.chars().count() - 1;
    let text = format!("{}\n{}", truncate_output(&text, Some(budget)), line);
    (text, Some(r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::llm_trait::{
        self, Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider,
        ProviderInfo,
    };
    use agent_base::{
        AgentBuilder, AgentRuntime, CheckpointData, CheckpointStep, InteractionChannel,
        InteractionError, RunOutcome, RunState, StreamChunk,
    };
    use codexec::{
        AllowAll, ClassPolicy, ExecutorBuilder, JournalHandle, Limits, MemJournal, PolicyDecision,
        PythonWorker,
    };
    use phi_ns::builtin::PhiNsShell;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    /// The canonical assembly: one [`ExecCtxBridge`] mounted into the engine
    /// (`with_interaction`) and held by the adapter — the same handle on both
    /// ends, as [`CodeActExecutor::new`] documents.
    async fn assemble(
        policy: impl codexec::Policy + 'static,
        limits: Limits,
    ) -> CodeActExecutor<PythonWorker> {
        let bridge = Arc::new(ExecCtxBridge::new());
        let engine = ExecutorBuilder::new(JournalHandle::new(Box::new(MemJournal::default())))
            .with_policy(policy)
            .with_limits(limits)
            .with_ns(PhiNsShell::builtin())
            .with_interaction(bridge.clone())
            .build(move |registry| PythonWorker::spawn(registry, limits))
            .await
            .expect("engine assembles");
        CodeActExecutor::new(engine, bridge)
    }

    async fn stage() -> CodeActExecutor<PythonWorker> {
        assemble(AllowAll, Limits::default()).await
    }

    fn turn(text: &str) -> LlmTurnResult {
        LlmTurnResult {
            full_text: text.into(),
            reasoning_text: String::new(),
            is_tool_call: false,
            tool_calls: vec![],
            usage: None,
            finish_reason: Some("stop".into()),
            ttft_ms: 1,
            llm_duration_ms: 2,
            reasoning_only: false,
            thinking_signature: None,
        }
    }

    fn cell(i: usize, intent: &str, code: &str) -> Action {
        Action::from_code_cell(format!("cell_{i}"), intent, code)
    }

    fn exec_ctx() -> ExecCtx {
        struct Noop;
        #[async_trait]
        impl agent_base::InteractionChannel for Noop {
            async fn request_consent(
                &self,
                _request: agent_base::ApprovalRequest,
                _cancel: &tokio_util::sync::CancellationToken,
            ) -> Result<agent_base::ApprovalDecision, agent_base::InteractionError> {
                unreachable!("no consent in these tests")
            }
            async fn ask_user(
                &self,
                _question: &str,
                _options: &[agent_base::AskOption],
                _cancel: &tokio_util::sync::CancellationToken,
            ) -> Result<agent_base::AskAnswer, agent_base::InteractionError> {
                unreachable!("no ask in these tests")
            }
        }
        ExecCtx::new(
            tokio_util::sync::CancellationToken::new(),
            std::sync::Arc::new(Noop),
        )
    }

    /// A run whose product layer answers by script (human time is unbounded —
    /// the run's cancel token is the guard).
    fn ctx_on(cancel: CancellationToken, channel: Arc<Scripted>) -> ExecCtx {
        ExecCtx::new(cancel, channel).with_interact_timeout(None)
    }

    /// The consent shape: `sh.run` (Process) asks before it runs.
    fn ask_policy() -> impl codexec::Policy + 'static {
        ClassPolicy::new(PolicyDecision::Allow).with(Capability::Process, PolicyDecision::Ask)
    }

    /// What the product's human does when a cell suspends on consent.
    #[derive(Debug)]
    enum ConsentScript {
        Allow,
        Deny,
        /// Never answers by itself — only the run's cancel token releases it.
        Suspend,
    }

    /// What the product's picker does when a cell calls `ask.user`.
    #[derive(Debug)]
    enum AskScript {
        Pick(String, String),
        Skip,
    }

    /// The product layer behind the channel: answers by script and records
    /// what the dialog was actually shown.
    struct Scripted {
        consent: ConsentScript,
        ask: AskScript,
        /// Consent requests as the product saw them (the payload must reach
        /// the human, not just a digest).
        seen: Mutex<Vec<ApprovalRequest>>,
        /// `(question, options)` as the picker saw them.
        asked: Mutex<Vec<(String, Vec<AskOption>)>>,
    }

    impl Scripted {
        fn new(consent: ConsentScript, ask: AskScript) -> Self {
            Self {
                consent,
                ask,
                seen: Mutex::new(Vec::new()),
                asked: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl InteractionChannel for Scripted {
        async fn request_consent(
            &self,
            request: ApprovalRequest,
            cancel: &CancellationToken,
        ) -> Result<ApprovalDecision, InteractionError> {
            self.seen.lock().unwrap().push(request);
            match &self.consent {
                ConsentScript::Allow => Ok(ApprovalDecision::AllowOnce),
                ConsentScript::Deny => Ok(ApprovalDecision::Deny),
                ConsentScript::Suspend => {
                    cancel.cancelled().await;
                    Err(InteractionError::Cancelled)
                }
            }
        }

        async fn ask_user(
            &self,
            question: &str,
            options: &[AskOption],
            _cancel: &CancellationToken,
        ) -> Result<AskAnswer, InteractionError> {
            self.asked
                .lock()
                .unwrap()
                .push((question.to_string(), options.to_vec()));
            match &self.ask {
                AskScript::Pick(id, label) => Ok(AskAnswer::Picked {
                    id: id.clone(),
                    label: label.clone(),
                }),
                AskScript::Skip => Ok(AskAnswer::Skipped),
            }
        }
    }

    fn text_after(obs: &Observation) -> String {
        obs.history_messages
            .iter()
            .filter_map(|m| match m {
                ChatMessage::User { content, .. } => Some(content.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn parse_maps_the_engine_verdicts_onto_four_states() {
        // Blocks materialize as CodeAct actions with the engine block shape.
        match classify_text(
            "<intent>a</intent>\n<code>x = 1</code>\n<intent>b</intent>\n<code>print(x)</code>",
        ) {
            ParseOutcome::Actions(actions) => {
                assert_eq!(actions.len(), 2);
                assert_eq!(actions[0], cell(0, "a", "x = 1"));
                assert_eq!(actions[1], cell(1, "b", "print(x)"));
            }
            other => panic!("expected Actions, got {other:?}"),
        }

        // Prose with no blocks is the empty branch (empty text first).
        assert_eq!(classify_text("all done."), ParseOutcome::Empty);
        assert_eq!(classify_text(""), ParseOutcome::Empty);

        // Action intent that never materialized — the light-retry branch.
        assert_eq!(
            classify_text("<intent>a</intent>\n<code>x = 1"),
            ParseOutcome::Incomplete
        );
        assert_eq!(
            classify_text("<intent>a</intent>"),
            ParseOutcome::Incomplete
        );

        // Structurally complete but wrong — the anomaly branch.
        assert_eq!(
            classify_text("<intent>a</intent>\n<code>  </code>"),
            ParseOutcome::Malformed
        );
    }

    #[tokio::test]
    async fn parse_rejects_an_envelope_turn_loudly() {
        let s = stage().await;
        let mut t = turn("");
        t.is_tool_call = true;
        t.tool_calls =
            vec![serde_json::json!({"id": "c1", "function": {"name": "x", "arguments": "{}"}})];
        assert!(
            matches!(s.parse(&t), Err(ParseError::Unparseable(_))),
            "a routing bug must fail loudly, not nudge"
        );
    }

    #[tokio::test]
    async fn execute_runs_the_batch_and_renders_shape_a() {
        let mut s = stage().await;
        let actions = vec![cell(0, "bind", "x = 41"), cell(1, "print", "print(x + 1)")];
        let obs = s.execute(actions.clone(), &exec_ctx()).await.unwrap();

        assert_eq!(obs.outcomes.len(), 2);
        assert!(obs.outcomes.iter().all(|o| o.ok), "{:?}", obs.outcomes);
        assert_eq!(obs.batch_status, BatchStatus::Completed);
        // Shape A pair: the assistant re-render plus a [observation] message.
        assert_eq!(obs.history_messages.len(), 2);
        assert!(text_after(&obs).contains("[observation]"));
        assert!(text_after(&obs).contains("42"));
    }

    #[tokio::test]
    async fn a_failing_cell_reports_fates_and_halts_the_step() {
        let mut s = stage().await;
        let actions = vec![
            cell(0, "bind", "x = 1"),
            cell(1, "boom", "raise ValueError('nope')"),
            cell(2, "never", "print('late')"),
        ];
        let obs = s.execute(actions, &exec_ctx()).await.unwrap();

        assert!(obs.outcomes[0].ok);
        assert!(!obs.outcomes[1].ok);
        assert!(obs.outcomes[1].summary.contains("ValueError"));
        assert!(obs.outcomes[2].summary.contains("not run"));
        assert!(
            text_after(&obs).contains("error:"),
            "the model sees the raise"
        );
    }

    #[tokio::test]
    async fn a_compile_error_means_nothing_ran() {
        let mut s = stage().await;
        let actions = vec![cell(0, "bind", "x = 1"), cell(1, "broken", "def (")];
        let obs = s.execute(actions, &exec_ctx()).await.unwrap();

        // D2: the whole round stops before any block runs.
        assert!(obs.outcomes.iter().all(|o| !o.ok), "{:?}", obs.outcomes);
        assert!(obs.outcomes[0].summary.contains("not run"));
        assert!(obs.outcomes[1].summary.contains("compile_error"));
    }

    #[tokio::test]
    async fn the_batch_validates_before_anything_runs() {
        let mut s = stage().await;
        // One poisoned action (no block shape) must stop the whole batch (D2).
        let bad = Action {
            id: "cell_1".into(),
            name: "code_cell".into(),
            args: serde_json::json!({"code": "print(1)"}),
            identity: "print(1)".into(),
            format: ActionFormat::CodeAct,
        };
        let err = s
            .execute(vec![cell(0, "bind", "x = 1"), bad], &exec_ctx())
            .await
            .unwrap_err();
        assert!(matches!(err, ExecError::NothingRan { .. }), "{err}");
    }

    #[tokio::test]
    async fn resident_state_survives_across_steps() {
        let mut s = stage().await;
        let first = s
            .execute(vec![cell(0, "bind", "x = 41")], &exec_ctx())
            .await
            .unwrap();
        assert!(first.outcomes[0].ok);

        let second = s
            .execute(vec![cell(0, "print", "print(x + 1)")], &exec_ctx())
            .await
            .unwrap();
        assert!(
            text_after(&second).contains("42"),
            "the session remembers x"
        );
    }

    #[tokio::test]
    async fn a_lost_round_reports_unreported_fates() {
        // A tiny budget and a slow cell: the round dies on timeout and no
        // per-cell fate may be invented.
        let limits = Limits {
            timeout_ms: 200,
            ..Limits::default()
        };
        let mut s = assemble(AllowAll, limits).await;

        let obs = s
            .execute(
                vec![cell(0, "sleep", "import time; time.sleep(5)")],
                &exec_ctx(),
            )
            .await
            .unwrap();
        assert_eq!(obs.batch_status, BatchStatus::EnvDied);
        assert!(obs.outcomes[0].summary.contains("fate not reported"));
    }

    // ---- T13/M3c: the observation cap, the spill, and the reference line ----

    /// Over the observation cap with a store mounted: the pane keeps the
    /// family preview plus the summary+reference line (bounded), and the
    /// typed channel marks the artifact (M3c, action-executor-design §6.1).
    #[tokio::test]
    async fn an_oversized_observation_spills_and_keeps_the_reference_on_the_pane() {
        let tmp = tempfile::tempdir().unwrap();
        let bridge = Arc::new(ExecCtxBridge::new());
        let limits = Limits::default();
        let engine = ExecutorBuilder::new(JournalHandle::new(Box::new(MemJournal::default())))
            .with_policy(AllowAll)
            .with_limits(limits)
            .with_ns(PhiNsShell::builtin())
            .with_interaction(bridge.clone())
            .with_artifacts(tmp.path().join("captures"))
            .build(move |registry| PythonWorker::spawn(registry, limits))
            .await
            .expect("engine assembles");
        let mut s = CodeActExecutor::new(engine, bridge).with_observation_cap(2_000);

        let obs = s
            .execute(vec![cell(0, "print", "print('x' * 5000)")], &exec_ctx())
            .await
            .unwrap();
        assert!(obs.outcomes[0].ok, "{:?}", obs.outcomes);
        let text = text_after(&obs);
        assert!(
            text.chars().count() <= 2_000,
            "the pane must fit the cap, got {} chars",
            text.chars().count()
        );
        assert!(
            text.contains("obs.fetch('art_0001') for the full body"),
            "the way back must be on the pane: {text:.200}"
        );
        assert!(
            obs.events
                .iter()
                .any(|e| matches!(e, ObservationEvent::Artifact { capture_id, .. } if capture_id == "art_0001")),
            "the typed channel marks the spill: {:?}",
            obs.events
        );
    }

    /// A spilled body's event renders its standing reference on the pane and
    /// its typed face on the channel — one event, two faces, no duplication.
    #[test]
    fn a_spilled_reference_event_renders_on_the_pane_and_the_typed_channel() {
        let obs = EngineObservation {
            events: vec![CellEvent::Artifact {
                capture_id: "art_0001".to_string(),
                kind: "stdout".to_string(),
                bytes: 11,
                summary: "first body".to_string(),
            }],
            ..EngineObservation::default()
        };
        let (pane, spill) = observation_text(&obs, 16_000, None);
        assert!(spill.is_none());
        assert!(
            pane.contains(
                "artifact art_0001 (stdout, 11 bytes): first body — obs.fetch('art_0001') for the full body"
            ),
            "{pane}"
        );
        let events = observation_events(&obs);
        assert!(
            matches!(
                &events[0],
                ObservationEvent::Artifact { capture_id, kind, summary }
                    if capture_id == "art_0001" && kind == "stdout" && summary == "first body"
            ),
            "{events:?}"
        );
    }

    /// No store: the pane truncates with the family marker — bounded either
    /// way, and no typed event points at a reference that does not exist.
    #[test]
    fn an_over_cap_pane_without_a_store_truncates_marked() {
        let obs = EngineObservation {
            stdout: "x".repeat(5_000),
            ..EngineObservation::default()
        };
        let (pane, spill) = observation_text(&obs, 1_000, None);
        assert!(spill.is_none());
        assert!(pane.chars().count() <= 1_000, "{}", pane.chars().count());
        assert!(pane.contains("...[output truncated]"), "{pane:.80}");
    }

    // ---- T8: the three cell-interaction scenarios, on real cells ----

    #[tokio::test]
    async fn a_consented_call_runs_with_the_payload_on_the_dialog() {
        let mut s = assemble(ask_policy(), Limits::default()).await;
        let channel = Arc::new(Scripted::new(ConsentScript::Allow, AskScript::Skip));
        let obs = s
            .execute(
                vec![cell(0, "run", "sh.run(\"echo hi\")")],
                &ctx_on(CancellationToken::new(), Arc::clone(&channel)),
            )
            .await
            .unwrap();

        assert!(obs.outcomes[0].ok, "{:?}", obs.outcomes);
        assert!(text_after(&obs).contains("hi"), "{obs:?}");

        // The dialog saw the call's payload and its risk — not just a digest.
        let seen = channel.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1, "{seen:?}");
        assert_eq!(seen[0].action_key.as_deref(), Some("sh.run"), "{seen:?}");
        assert!(seen[0].message.contains("echo hi"), "{seen:?}");
        assert_eq!(
            seen[0].raw,
            Some(serde_json::json!(["echo hi"])),
            "{seen:?}"
        );
        assert!(
            matches!(seen[0].risk_level, RiskLevel::Sensitive),
            "{seen:?}"
        );
    }

    #[tokio::test]
    async fn a_refused_consent_halts_the_step_with_the_reason_on_the_fate() {
        let mut s = assemble(ask_policy(), Limits::default()).await;
        let channel = Arc::new(Scripted::new(ConsentScript::Deny, AskScript::Skip));
        let obs = s
            .execute(
                vec![
                    cell(0, "run", "sh.run(\"echo hi\")"),
                    cell(1, "late", "print('late')"),
                ],
                &ctx_on(CancellationToken::new(), channel),
            )
            .await
            .unwrap();

        assert!(obs.outcomes[0].denied, "{:?}", obs.outcomes);
        assert!(
            obs.outcomes[0].summary.contains("consent refused"),
            "{:?}",
            obs.outcomes
        );
        assert!(
            obs.outcomes[1].summary.contains("not run"),
            "{:?}",
            obs.outcomes
        );
        assert!(
            !text_after(&obs).contains("hi"),
            "the effect never ran: {obs:?}"
        );
        // The reason rides the model-visible error, not just the fate.
        assert!(text_after(&obs).contains("consent refused"), "{obs:?}");
    }

    #[tokio::test]
    async fn an_ask_from_the_cell_reaches_the_human_and_returns_the_pick() {
        let mut s = assemble(AllowAll, Limits::default()).await;
        let channel = Arc::new(Scripted::new(
            ConsentScript::Allow,
            AskScript::Pick("a".into(), "Alpha".into()),
        ));
        let obs = s
            .execute(
                vec![cell(
                    0,
                    "ask",
                    "print(ask.user(\"which one?\", [{\"id\": \"a\", \"label\": \"Alpha\"}]))",
                )],
                &ctx_on(CancellationToken::new(), Arc::clone(&channel)),
            )
            .await
            .unwrap();

        assert!(obs.outcomes[0].ok, "{:?}", obs.outcomes);
        assert!(
            text_after(&obs).contains("Alpha"),
            "the pick came back: {obs:?}"
        );
        // The picker saw the parsed ask vocabulary, not raw wire.
        let asked = channel.asked.lock().unwrap().clone();
        assert_eq!(asked.len(), 1, "{asked:?}");
        assert_eq!(asked[0].0, "which one?", "{asked:?}");
        assert_eq!(
            asked[0].1,
            vec![AskOption {
                id: "a".into(),
                label: "Alpha".into(),
                note: None
            }],
            "{asked:?}"
        );
    }

    #[tokio::test]
    async fn a_skipped_ask_returns_skipped_to_the_cell() {
        let mut s = assemble(AllowAll, Limits::default()).await;
        let channel = Arc::new(Scripted::new(ConsentScript::Allow, AskScript::Skip));
        let obs = s
            .execute(
                vec![cell(
                    0,
                    "ask",
                    "print(ask.user(\"which one?\", [{\"id\": \"a\", \"label\": \"Alpha\"}]))",
                )],
                &ctx_on(CancellationToken::new(), channel),
            )
            .await
            .unwrap();

        assert!(obs.outcomes[0].ok, "{:?}", obs.outcomes);
        assert!(text_after(&obs).contains("skipped"), "{obs:?}");
    }

    #[tokio::test]
    async fn interrupt_releases_a_suspended_cell_with_the_reason() {
        let mut s = assemble(ask_policy(), Limits::default()).await;
        let channel = Arc::new(Scripted::new(ConsentScript::Suspend, AskScript::Skip));
        let cancel = CancellationToken::new();
        let ctx = ctx_on(cancel.clone(), Arc::clone(&channel));

        let run = tokio::spawn(async move {
            s.execute(
                vec![
                    cell(0, "run", "sh.run(\"echo hi\")"),
                    cell(1, "late", "print('late')"),
                ],
                &ctx,
            )
            .await
        });
        // Interrupt only after the suspension really reached the product.
        for _ in 0..500 {
            if !channel.seen.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            !channel.seen.lock().unwrap().is_empty(),
            "the cell reached the consent suspension"
        );
        cancel.cancel();

        let obs = run.await.unwrap().unwrap();
        assert!(obs.outcomes[0].denied, "{:?}", obs.outcomes);
        assert!(
            obs.outcomes[0].summary.contains("interaction cancelled"),
            "{:?}",
            obs.outcomes
        );
        assert!(
            obs.outcomes[1].summary.contains("not run"),
            "{:?}",
            obs.outcomes
        );
        assert!(
            !text_after(&obs).contains("hi"),
            "the effect never ran: {obs:?}"
        );
    }

    // ---- T10c: the real backend through the kernel's mount face ----

    /// The execution sentinel: the cell appends one line per run, so the line
    /// count is the number of times it actually ran.
    fn sentinel(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "phimint_code_act_t10c_{}_{}.txt",
            std::process::id(),
            tag
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn sentinel_runs(p: &std::path::Path) -> usize {
        std::fs::read_to_string(p)
            .map(|s| s.lines().count())
            .unwrap_or(0)
    }

    /// The cell body that touches the sentinel.
    fn appender(p: &std::path::Path) -> String {
        format!("open({:?}, \"a\").write(\"x\\n\")", p.display().to_string())
    }

    /// The chat history flattened to one string for readable assertions.
    fn flat(msgs: &[ChatMessage]) -> String {
        msgs.iter()
            .map(|m| match m {
                ChatMessage::Assistant {
                    content: Some(c), ..
                } => c.clone(),
                ChatMessage::User { content, .. } => content.clone(),
                other => format!("{other:?}"),
            })
            .collect::<Vec<_>>()
            .join("\n---\n")
    }

    /// A scripted LLM double (multi-turn): each `stream` call plays the next
    /// turn's chunks.
    struct LlmScript {
        script: Mutex<std::vec::IntoIter<Vec<StreamChunk>>>,
    }

    impl LlmScript {
        fn new(turns: Vec<Vec<StreamChunk>>) -> Arc<Self> {
            Arc::new(Self {
                script: Mutex::new(turns.into_iter()),
            })
        }

        fn says(text: &str) -> Vec<StreamChunk> {
            vec![
                StreamChunk::Text(text.to_string()),
                StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                },
            ]
        }
    }

    #[async_trait]
    impl LlmProvider for LlmScript {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            let chunks: Vec<Result<StreamChunk, LlmError>> = self
                .script
                .lock()
                .unwrap()
                .next()
                .unwrap_or_default()
                .into_iter()
                .map(Ok)
                .collect();
            Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
                chunks,
            ))))
        }

        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                content: String::new(),
                reasoning_content: None,
                tool_calls: vec![],
                usage: Default::default(),
                finish_reason: llm_trait::response::FinishReason::Stop,
                raw: None,
                thinking_signature: None,
            })
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }

        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                name: "test".to_string(),
                model: "test".to_string(),
                version: None,
            }
        }
    }

    /// Assemble the kernel the way a product does (T10c): one runtime, the
    /// real CodeAct executor mounted as its action backend.
    async fn mounted_runtime(turns: Vec<Vec<StreamChunk>>) -> AgentRuntime {
        AgentBuilder::new(LlmScript::new(turns))
            .system_prompt("test")
            .action_backend(Box::new(stage().await))
            .build()
            .expect("build runtime")
    }

    #[tokio::test]
    async fn action_backend_face_is_the_code_act_protocol() {
        let backend = stage().await;
        assert_eq!(backend.format(), ActionFormat::CodeAct);

        let actions = vec![cell(0, "first", "print(1)"), cell(1, "second", "print(2)")];
        let fates = vec![
            ActionOutcome {
                action_id: "cell_0".to_string(),
                name: "code_cell".to_string(),
                ok: true,
                denied: false,
                summary: "completed".to_string(),
                details: None,
            },
            ActionOutcome {
                action_id: "cell_1".to_string(),
                name: "code_cell".to_string(),
                ok: false,
                denied: false,
                summary: "已中断/未执行".to_string(),
                details: None,
            },
        ];
        let msgs = backend.render_fates(&actions, &fates);
        // Shape A: the batch verbatim as the assistant half, the fates in the
        // `[observation]` pane that follows — same pair as a live step.
        match &msgs[..] {
            [
                ChatMessage::Assistant {
                    content: Some(blocks),
                    ..
                },
                ChatMessage::User { content, .. },
            ] => {
                assert!(
                    blocks.contains("<intent>first</intent>\n<code>print(1)</code>"),
                    "{blocks}"
                );
                assert!(
                    blocks.contains("<intent>second</intent>\n<code>print(2)</code>"),
                    "{blocks}"
                );
                assert!(content.starts_with("[observation]\n"), "{content}");
                assert!(content.contains("#1 first: completed"), "{content}");
                assert!(content.contains("#2 second: 已中断/未执行"), "{content}");
            }
            other => panic!("shape A pair expected, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn resume_synthesizes_a_dangling_batch_without_executing_it() {
        // D3 twin on the real backend: the crash-window batch never re-runs,
        // and the model is told so in the backend's own shape A.
        let p = sentinel("dangling");
        let runtime = mounted_runtime(vec![LlmScript::says("resumed answer")]).await;
        let sid = runtime.create_session().await;

        let checkpoint = CheckpointData::new(
            sid.clone(),
            "resume",
            CheckpointStep::BeforeActions {
                actions: vec![cell(0, "write", &appender(&p))],
            },
            0,
            ActionFormat::CodeAct,
            RunState::default(),
        );

        let result = runtime.resume_from_checkpoint(checkpoint, |_| Ok(())).await;
        assert!(
            matches!(result, Ok(RunOutcome::Completed)),
            "resume should complete: {:?}",
            result.err()
        );
        assert_eq!(sentinel_runs(&p), 0, "a dangling batch must never execute");

        let session = runtime.session(&sid).await.expect("session exists");
        let text = flat(session.chat_messages());
        assert!(
            text.contains("已中断/未执行"),
            "the model must be told plainly: {text}"
        );
        assert!(
            text.contains("<intent>write</intent>"),
            "the batch blocks are replayed in shape A: {text}"
        );
        assert!(
            text.contains("[observation]"),
            "the fate pane is shape A: {text}"
        );
    }

    #[tokio::test]
    async fn resume_replays_a_recorded_observation_verbatim() {
        // D3 twin on the real backend: an AfterActions line carries its
        // observation — resume rebuilds history from it instead of re-running
        // the batch.
        let p = sentinel("recorded");
        let runtime = mounted_runtime(vec![LlmScript::says("resumed answer")]).await;
        let sid = runtime.create_session().await;

        let observation = Observation {
            outcomes: vec![ActionOutcome {
                action_id: "cell_0".to_string(),
                name: "code_cell".to_string(),
                ok: true,
                denied: false,
                summary: "completed".to_string(),
                details: None,
            }],
            events: vec![],
            history_messages: vec![
                ChatMessage::assistant("<intent>write</intent>\n<code>…</code>"),
                ChatMessage::user("[observation]\nstored result"),
            ],
            batch_status: BatchStatus::Completed,
        };
        let checkpoint = CheckpointData::new(
            sid.clone(),
            "resume",
            CheckpointStep::AfterActions {
                actions: vec![cell(0, "write", &appender(&p))],
                observation: Some(observation),
            },
            1,
            ActionFormat::CodeAct,
            RunState::default(),
        );

        let result = runtime.resume_from_checkpoint(checkpoint, |_| Ok(())).await;
        assert!(
            matches!(result, Ok(RunOutcome::Completed)),
            "resume should complete: {:?}",
            result.err()
        );
        assert_eq!(
            sentinel_runs(&p),
            0,
            "the recorded observation is replayed, not re-executed"
        );

        let session = runtime.session(&sid).await.expect("session exists");
        let text = flat(session.chat_messages());
        assert!(text.contains("stored result"), "verbatim replay: {text}");
        assert!(
            !text.contains("已中断/未执行"),
            "the recorded fate is not synthesized over: {text}"
        );
    }

    #[tokio::test]
    async fn a_live_text_turn_runs_one_real_cell_through_the_mounted_backend() {
        let p = sentinel("live");
        let runtime = mounted_runtime(vec![
            LlmScript::says(&format!(
                "<intent>append</intent>\n<code>{}</code>",
                appender(&p)
            )),
            LlmScript::says("done"),
        ])
        .await;
        let sid = runtime.create_session().await;

        let result = runtime
            .run_turn(sid.clone(), "run the cell", |_| Ok(()))
            .await;
        assert!(
            matches!(result, Ok(RunOutcome::Completed)),
            "run should complete: {:?}",
            result.err()
        );
        assert_eq!(sentinel_runs(&p), 1, "the cell ran exactly once");

        let session = runtime.session(&sid).await.expect("session exists");
        let text = flat(session.chat_messages());
        assert!(
            text.contains("<intent>append</intent>\n<code>"),
            "the batch blocks re-render in shape A: {text}"
        );
        assert!(
            text.contains("[observation]"),
            "the observation pane landed: {text}"
        );
    }
}
