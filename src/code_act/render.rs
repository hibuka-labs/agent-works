use agent_base::{Action, ChatMessage};
use codexec::observation::CellEvent;
use codexec::{ArtifactHub, ArtifactRef, Observation as EngineObservation, artifact_line};
use serde_json::Value;

use crate::script_tool::truncate_output;

/// The shape-A pair: the batch's blocks as one assistant message, the
/// observation pane as the user message that follows. Both live renderings
/// (engine text) and synthesized ones (fate summaries) share it.
pub(super) fn render_pair(actions: &[Action], observation: String) -> Vec<ChatMessage> {
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
pub(super) fn block_of(action: &Action) -> (String, String) {
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
pub(super) fn observation_text(
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
