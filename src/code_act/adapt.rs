use agent_base::{
    Action, ActionFormat, ActionOutcome, BatchStatus, Observation, ObservationEvent, ParseOutcome,
};
use codexec::observation::CellEvent;
use codexec::{
    Action as EngineAction, ArtifactHub, ExecError as EngineError, Observation as EngineObservation,
};
use serde_json::Value;

use super::render::{observation_text, render_pair};

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
pub(super) fn classify_text(text: &str) -> ParseOutcome {
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
pub(super) fn decode_batch(actions: &[Action]) -> Result<Vec<EngineAction>, String> {
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
pub(super) fn fold(
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
pub(super) fn observation_events(obs: &EngineObservation) -> Vec<ObservationEvent> {
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
pub(super) fn fates(actions: &[Action], obs: &EngineObservation) -> Vec<ActionOutcome> {
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
