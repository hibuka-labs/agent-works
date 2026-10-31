//! `code_cell` — the CodeAct main road as one agent tool (codexec-design §11 M2).
//!
//! One tool call = one engine step: a list of cells run against **one resident
//! interpreter**, where variables, imports and functions survive across cells
//! *and across calls*. Side effects run through the engine's single host-call
//! seam — policy-gated, journaled — so this module never sees a domain rule; it
//! only bridges the agent [`Tool`] face to [`CodeCellExecutor`].
//!
//! Assembly stays outside (sdk-assembly §3.4): the host lines up policy,
//! journal and capability packs on the [`ExecutorBuilder`] face and hands the
//! built executor to [`CodeCellTool::new`]. Several sessions in one process —
//! a host and its sub-agents — line up one executor per session through
//! [`SessionFactory`] instead, and [`CodeCellTool::per_session`] keeps each
//! resident: which session gets which policy and journal is the host's
//! decision, the pool only keys by session. The prompt face — what the model
//! reads about the installed namespaces — is derived from the very same
//! installed packs (`methods()` → `api_doc()`), never hand-written, and lands
//! in the tool description so vocabulary and schema arrive together. When a
//! face outgrows a description (dozens of methods), move it to a system-prompt
//! region swap the way `phimint/src/browser_face.rs` does — same source either
//! way.
//!
//! Boundary with `script_tool` (sdk-assembly §9.5): `script.{name}` spawns an
//! external process and forgets everything when it exits; a code cell lives in
//! the resident session. Both stay — neither replaces the other.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_base::tool::Tool;
use agent_base::types::SessionId;
use agent_base::{AgentError, AgentResult, Content, ToolContext, ToolMetadata};
use async_trait::async_trait;
use base64::Engine as _;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::script_tool::truncate_output;

pub use codexec::observation::{CellEvent, ExecError, Observation};
pub use codexec::worker::Worker;
pub use codexec::{
    Action, AllowAll, CallCtx, ClassPolicy, CodeCellExecutor, ConsentRequest, DenyAll,
    ExecutorBuilder, FileJournal, InteractionBridge, Journal, JournalHandle, Limits, MemJournal,
    MethodPolicy, Policy, PolicyDecision, PythonWorker,
};
use codexec::{ArtifactHub, artifact_line};
pub use phi_cap::Capability;
pub use phi_ns::builtin::PhiNsShell;
pub use phi_ns::{ImageRef, ImageSource, NamespaceProvider};

/// Tool name — bare, so it counts as a built-in face (tool-contract.md).
pub const TOOL_NAME: &str = "code_cell";

/// Slack added to the engine's round timeout for the framework watchdog.
const WATCHDOG_SLACK_MS: u64 = 3_000;

/// How many sessions a pooled tool keeps resident. Dropping an entry kills its
/// worker (`kill_on_drop`), so a burst of sub-agents cannot leave one Python
/// process per session behind.
const MAX_RESIDENT_SESSIONS: usize = 16;

/// One session's resident executor — the host's assembly seam (sdk-assembly
/// §3.4) for the many-sessions case: policy, journal and packs stay
/// application decisions, and the pool only keeps what [`SessionFactory::open`]
/// returns resident, keyed by session. The pool never interprets the key;
/// which session gets which policy is the host's business.
pub trait SessionFactory<W: Worker>: Send + Sync {
    /// The prompt face of what every session this factory opens installs —
    /// `ExecutorBuilder::prompt_face()` over exactly the packs `open` builds
    /// with (same-source rule: the prompt can never advertise a name the
    /// worker won't have).
    fn prompt_face(&self) -> String;

    /// The round quotas every session runs under; the framework watchdog
    /// budget derives from `timeout_ms`.
    fn limits(&self) -> Limits;

    /// Build the resident executor for one session. Called once per session
    /// key, lazily, on that session's first call.
    fn open(&self, session: &SessionId) -> SessionFuture<W>;

    /// Sessions the pool must never evict — the host's own long-lived ones.
    fn pinned(&self, session: &SessionId) -> bool {
        let _ = session;
        false
    }
}

/// What [`SessionFactory::open`] returns.
pub type SessionFuture<W> = Pin<Box<dyn Future<Output = AgentResult<CodeCellExecutor<W>>> + Send>>;

/// Static description head; the installed packs' `api_doc()` is appended when
/// the tool is built (see [`CodeCellTool::new`]).
const DESCRIPTION: &str = "Run code cells in a resident Python session (CodeAct). \
One call is one step: its cells run in order against one shared session whose \
state — variables, imports, functions, working directory — survives across \
cells AND across calls, so build on what earlier steps established instead of \
re-deriving it. Prefer one cell that does the whole mechanical job over many \
separate tool calls. Namespace calls with side effects are policy-gated and \
journaled; a held call halts the step. Below are the namespaces installed in \
this session and their methods — each namespace is already bound as a global \
in the cell, so call `ns.method(...)` directly; there is nothing to import.";

/// One agent tool call = one engine step of code cells.
pub struct CodeCellTool<W: Worker> {
    name: &'static str,
    description: &'static str,
    face: String,
    schema: Value,
    sessions: Sessions<W>,
    call_budget_ms: u64,
}

/// Where calls run: one resident executor for every caller session, or one
/// shared by all.
enum Sessions<W: Worker> {
    /// The single-executor shape [`CodeCellTool::new`] builds.
    One(Arc<Mutex<CodeCellExecutor<W>>>),
    /// One resident executor per caller session, opened through the factory.
    PerSession {
        factory: Arc<dyn SessionFactory<W>>,
        map: Mutex<HashMap<SessionId, Resident<W>>>,
        /// Last-use ticks — a counter, not a clock, so LRU order is total.
        tick: AtomicU64,
    },
}

/// A pooled session's resident executor and its last-use stamp.
struct Resident<W: Worker> {
    executor: Arc<Mutex<CodeCellExecutor<W>>>,
    used: u64,
}

impl<W: Worker + Send + 'static> CodeCellTool<W> {
    /// Wrap a built executor under the bare name `code_cell`.
    pub fn new(executor: CodeCellExecutor<W>) -> Self {
        Self::with_name(TOOL_NAME, executor)
    }

    /// Wrap a built executor under an explicit tool name.
    pub fn with_name(name: &'static str, executor: CodeCellExecutor<W>) -> Self {
        let face = prompt_face_of(&executor);
        let call_budget_ms = watchdog_budget(executor.limits().timeout_ms);
        Self::assemble(
            name,
            face,
            call_budget_ms,
            Sessions::One(Arc::new(Mutex::new(executor))),
        )
    }

    /// One resident executor per caller session: the pool asks the factory on
    /// a session's first call and keeps the executor resident, keyed by the
    /// session in the [`ToolContext`]. Past [`MAX_RESIDENT_SESSIONS`] it drops
    /// the least-recently-used session the factory does not pin — the worker
    /// dies with its executor, so eviction is clean.
    pub fn per_session(name: &'static str, factory: Arc<dyn SessionFactory<W>>) -> Self {
        let face = factory.prompt_face();
        let call_budget_ms = watchdog_budget(factory.limits().timeout_ms);
        let sessions = Sessions::PerSession {
            factory,
            map: Mutex::new(HashMap::new()),
            tick: AtomicU64::new(0),
        };
        Self::assemble(name, face, call_budget_ms, sessions)
    }

    fn assemble(
        name: &'static str,
        face: String,
        call_budget_ms: u64,
        sessions: Sessions<W>,
    ) -> Self {
        let description: &'static str = Box::leak(
            format!("{DESCRIPTION}\n\n## Installed namespaces\n\n{face}").into_boxed_str(),
        );
        Self {
            name,
            description,
            face,
            schema: input_schema(),
            sessions,
            call_budget_ms,
        }
    }

    /// The installed packs' prompt face (same-source rule, sdk-assembly §3.4) —
    /// for hosts that keep vocabulary in a system-prompt section instead.
    pub fn prompt_face(&self) -> &str {
        &self.face
    }
}

/// The framework watchdog budget: the engine's round timeout plus slack, so
/// the watchdog can never fire first and strand a mid-round worker.
fn watchdog_budget(round_timeout_ms: u64) -> u64 {
    round_timeout_ms.saturating_add(WATCHDOG_SLACK_MS)
}

impl<W: Worker + Send + 'static> Sessions<W> {
    /// The caller session's resident executor — the shared one, or a session
    /// of its own opened through the factory on first use.
    async fn resident(&self, session: &SessionId) -> AgentResult<Arc<Mutex<CodeCellExecutor<W>>>> {
        match self {
            Sessions::One(executor) => Ok(Arc::clone(executor)),
            Sessions::PerSession { factory, map, tick } => {
                let my_tick = tick.fetch_add(1, Ordering::Relaxed);
                {
                    let mut map = map.lock().await;
                    if let Some(resident) = map.get_mut(session) {
                        resident.used = my_tick;
                        return Ok(Arc::clone(&resident.executor));
                    }
                }
                // Open outside the lock (it spawns a worker): two calls racing
                // on one session's first step may both build, and the map keeps
                // whichever lands first — the other executor drops and its
                // worker dies with it.
                let opened = Arc::new(Mutex::new(factory.open(session).await?));
                let mut map = map.lock().await;
                let resident = map.entry(session.clone()).or_insert_with(|| Resident {
                    executor: opened,
                    used: my_tick,
                });
                resident.used = my_tick;
                let executor = Arc::clone(&resident.executor);
                while map.len() > MAX_RESIDENT_SESSIONS {
                    let Some(victim) = eviction_candidate(&map, factory.as_ref()) else {
                        break;
                    };
                    map.remove(&victim);
                }
                Ok(executor)
            }
        }
    }
}

/// The least-recently-used session the host does not pin — the one to drop
/// when the pool is full. `None` when everything is pinned (the pool then just
/// keeps them all, bounded by the pinned count).
fn eviction_candidate<W: Worker>(
    map: &HashMap<SessionId, Resident<W>>,
    factory: &dyn SessionFactory<W>,
) -> Option<SessionId> {
    map.iter()
        .filter(|(session, _)| !factory.pinned(session))
        .min_by_key(|(_, resident)| resident.used)
        .map(|(session, _)| session.clone())
}

#[async_trait]
impl<W: Worker + Send + 'static> Tool for CodeCellTool<W> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let actions = parse_cells(args, self.name)?;
        // Steps serialize per resident session; different sessions run on
        // their own executors.
        let executor = self.sessions.resident(&ctx.session_id).await?;
        let (observation, hub) = {
            let mut guard = executor.lock().await;
            let hub = guard.registry().artifacts();
            (guard.execute(actions).await, hub)
        };
        Ok(render(observation, ctx.max_output_chars, hub.as_deref()))
    }

    fn timeout_ms(&self) -> Option<u64> {
        // The engine owns the round budget — on expiry it recycles the worker
        // cleanly. The framework watchdog gets round + slack so it can never
        // fire first and strand a mid-round worker. No per-call `timeout_ms`
        // argument on purpose: per-call budgets need engine-side quota work.
        Some(self.call_budget_ms)
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name.to_string(),
            description: self.description.to_string(),
            origin: "code-exec".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            requirements: vec![],
        }
    }
}

/// `{"cells": [{"intent", "code"}]}` → engine actions. Malformed arguments are
/// tool errors (the caller learns it asked wrong); everything the engine says
/// is an observation and rides back as content.
///
/// One shape of "wrong" gets a deliberate second look: under nested-quoting
/// pressure (code holding shell double quotes and regex escapes) the model
/// occasionally emits `cells` as a JSON-encoded *string* of the array — all
/// four parse failures in live session 20261029_fff23e77 were this fumble,
/// doubly encoded with the inner escaping half-done. A string whose contents
/// parse as the step array is that fumble recovered; anything else is refused
/// with the shape spelled out and the blob head, because replaying the whole
/// doubly-encoded blob back at the model teaches it nothing.
fn parse_cells(args: &Value, name: &str) -> AgentResult<Vec<Action>> {
    let bad = |raw: &Value| AgentError::ToolArgsInvalid {
        name: name.to_string(),
        raw: raw.to_string(),
    };
    let guidance = |reason: &str, got: &str| AgentError::ToolArgsInvalid {
        name: name.to_string(),
        raw: format!(
            "cells must be a JSON array of {{\"intent\": \"…\", \"code\": \"…\"}} objects, \
             passed directly — {reason}. Do not encode the array into a string, and prefer \
             single quotes inside `code` so the tool-call JSON never needs nested escaping. \
             Received (head): {}",
            blob_head(got)
        ),
    };
    let Some(cells_arg) = args.get("cells") else {
        return Err(bad(args));
    };
    let decoded;
    let cells: &[Value] = match cells_arg {
        Value::Array(cells) => cells.as_slice(),
        Value::String(encoded) => match serde_json::from_str::<Vec<Value>>(encoded) {
            Ok(cells) if !cells.is_empty() => {
                decoded = cells;
                decoded.as_slice()
            }
            _ => return Err(guidance("it arrived as a JSON-encoded string", encoded)),
        },
        other => {
            return Err(guidance(
                "it arrived as a non-array value",
                &other.to_string(),
            ));
        }
    };
    let mut actions = Vec::with_capacity(cells.len());
    for cell in cells {
        let intent = cell
            .get("intent")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(cell))?;
        let code = cell
            .get("code")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(cell))?;
        actions.push(Action::CodeCell {
            intent: intent.to_string(),
            code: code.to_string(),
        });
    }
    if actions.is_empty() {
        return Err(bad(args));
    }
    Ok(actions)
}

/// First 160 chars of a blob for an error echo, ellipsized so a doubly-encoded
/// step can't flood the model's context with its own garbage.
fn blob_head(s: &str) -> String {
    const N: usize = 160;
    if s.chars().count() <= N {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(N).collect::<String>())
    }
}

/// Observation → tool content. The text stream — with its per-cell
/// `namespace:` snapshot lines and image `alt` floors — is what the model
/// reads; errors and events ride as their own lines at the tail so nothing
/// hides in the noise. Engine failures are content, not tool errors: the model
/// must see them and act (re-observe, rebuild references, try again).
///
/// One text block, self-truncated to `max_chars` with the family convention
/// (`truncate_output`: head 1/3 + `...[output truncated]` + tail 2/3 — error
/// and event lines land on the tail side and survive). The engine's hard
/// reject (`Tool 'code_cell' output exceeds the …-char limit`) must never fire
/// for a big print: the code already ran, and its result is the model's only
/// way forward. Images ride their own content and don't count against the cap.
/// Text is joined the same way `content_text` flattens tool output, so the
/// model-visible bytes are the multi-block shape's, cut when over budget.
///
/// M3c (action-executor-design §6.1): a body cut here does not die — the whole
/// joined body spills to the engine's artifact store and the block keeps the
/// preview plus the summary+reference line (both fitted inside `max_chars`).
/// Spilled bodies also get a typed [`Content::Artifact`] face alongside the
/// text (the line is the model face; the typed item is for UI/logs).
fn render(obs: Observation, max_chars: Option<usize>, hub: Option<&ArtifactHub>) -> Vec<Content> {
    let mut blocks: Vec<String> = Vec::new();
    let mut artifacts: Vec<Content> = Vec::new();
    if !obs.stdout.is_empty() {
        blocks.push(obs.stdout);
    }
    if !obs.stderr.is_empty() {
        blocks.push(format!("stderr:\n{}", obs.stderr));
    }
    if let Some(err) = &obs.error {
        blocks.push(format!("error: {err}"));
    }
    let mut images = Vec::new();
    for event in &obs.events {
        match event {
            CellEvent::Image(image) => images.push(image_content(image)),
            CellEvent::TodoChanged { id, status } => {
                blocks.push(format!("todo {id}: {status}"));
            }
            CellEvent::Notice { text } => blocks.push(format!("notice: {text}")),
            // A spilled body's standing reference (T13): summary + fetch
            // instruction in the text stream, typed face alongside.
            CellEvent::Artifact {
                capture_id,
                kind,
                bytes,
                summary,
            } => {
                blocks.push(artifact_line(capture_id, kind, *bytes, summary));
                artifacts.push(Content::artifact(
                    capture_id.clone(),
                    kind.clone(),
                    summary.clone(),
                ));
            }
        }
    }
    let mut out = Vec::new();
    if blocks.is_empty() {
        if images.is_empty() {
            out.push(Content::text("(no output)"));
        }
    } else {
        out.push(Content::text(fit_to_cap(
            &blocks.join("\n"),
            max_chars,
            hub,
            &mut artifacts,
        )));
    }
    out.extend(artifacts);
    out.extend(images);
    out
}

/// Cap one text block: over budget, spill the whole body to the store (M3c)
/// and keep the family preview plus the summary+reference line, fitted inside
/// `max_chars` — the line counts against the budget so the hard reject can
/// never fire on a spill. The spill writes through the store directly (not
/// `hub.spill`): the hub's event queue belongs to the engine's round and this
/// render runs after the fold drained it. The typed face of a spill lands in
/// `artifacts`. No store (or a store failure): the family truncation stands,
/// body lost as before — marked, never silent.
fn fit_to_cap(
    body: &str,
    max_chars: Option<usize>,
    hub: Option<&ArtifactHub>,
    artifacts: &mut Vec<Content>,
) -> String {
    let Some(max_chars) = max_chars else {
        return body.to_string();
    };
    if body.chars().count() <= max_chars {
        return body.to_string();
    }
    let Some(r) = hub.and_then(|h| h.store().spill("observation", body).ok()) else {
        return truncate_output(body, Some(max_chars));
    };
    let line = artifact_line(&r.capture_id, &r.kind, r.bytes, &r.summary);
    if line.chars().count() + 1 >= max_chars {
        // The reference line cannot fit — the cap is the hard contract (the
        // engine rejects the whole call over it), so plain truncation wins
        // and no typed face points at a reference the model cannot see.
        return truncate_output(body, Some(max_chars));
    }
    artifacts.push(Content::artifact(
        r.capture_id.clone(),
        r.kind.clone(),
        r.summary.clone(),
    ));
    let budget = max_chars - line.chars().count() - 1;
    format!("{}\n{}", truncate_output(body, Some(budget)), line)
}

/// One image event → one `Content::Image`. Bytes that cannot be produced are
/// reported as text — the model must never lose the sentence (the `alt` floor
/// is already in `stdout`; this covers the bytes themselves).
fn image_content(image: &ImageRef) -> Content {
    let bytes = match &image.source {
        ImageSource::Inline(bytes) => Ok(bytes.clone()),
        ImageSource::Path(path) => std::fs::read(path),
    };
    match bytes {
        Ok(bytes) => {
            let (bytes, mime) = wire_image_bytes(bytes, image.mime);
            Content::image(
                base64::engine::general_purpose::STANDARD.encode(bytes),
                mime,
            )
        }
        Err(e) => Content::text(format!("image \"{}\": bytes unavailable ({e})", image.alt)),
    }
}

/// Raw PNG screenshots are ~800 KB each and base64 inflates them by a third;
/// a request carrying a few of them tripped the gateway's body limit (HTTP
/// 413, session 20261030_862ddf5c). Large pictures are re-encoded as JPEG
/// **at unchanged pixel dimensions** — the model does coordinate math off the
/// image's geometry, so only the codec may change, never the layout. The
/// on-disk artifact keeps full fidelity; only the model-facing copy is
/// budgeted. Anything that fails to decode, or that re-encodes no smaller,
/// goes out byte-for-byte as it came in.
const IMAGE_TRANSCODE_OVER_BYTES: usize = 100 * 1024;
const IMAGE_JPEG_QUALITY: u8 = 85;

/// Re-encode an oversized picture for the wire; returns the bytes and the
/// mime that truthfully describes them.
fn wire_image_bytes(bytes: Vec<u8>, mime: &'static str) -> (Vec<u8>, &'static str) {
    if bytes.len() <= IMAGE_TRANSCODE_OVER_BYTES {
        return (bytes, mime);
    }
    let Ok(img) = image::load_from_memory(&bytes) else {
        return (bytes, mime);
    };
    // JPEG carries no alpha; a screenshot's transparency is meaningless to
    // the model, and RGB is what every screenshot decodes to anyway.
    let mut out = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, IMAGE_JPEG_QUALITY);
    if img.to_rgb8().write_with_encoder(encoder).is_err() || out.len() >= bytes.len() {
        return (bytes, mime);
    }
    (out, "image/jpeg")
}

/// Input schema: one step = `cells`, each `{intent, code}` — the same shape the
/// codexec host protocol calls `blocks`. One vocabulary, one meaning.
fn input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "cells": {
                "type": "array",
                "minItems": 1,
                "description": "the step's cells, run in order against one shared resident session — a real JSON array here, never JSON-encoded into a string",
                "items": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "description": "one line: why this cell runs"
                        },
                        "code": {
                            "type": "string",
                            "description": "Python source for the cell"
                        }
                    },
                    "required": ["intent", "code"]
                }
            }
        },
        "required": ["cells"]
    })
}

/// The installed packs' prompt face — one `api_doc()` per pack, derived from
/// the same `methods()` list the engine enforces (same-source rule).
fn prompt_face_of<W: Worker>(executor: &CodeCellExecutor<W>) -> String {
    let docs: Vec<String> = executor
        .registry()
        .packs()
        .iter()
        .map(|pack| pack.api_doc())
        .filter(|doc| !doc.trim().is_empty())
        .collect();
    if docs.is_empty() {
        "(no capability packs installed)".to_string()
    } else {
        docs.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script_tool::TRUNCATION_MARKER;

    #[test]
    fn parse_cells_reads_the_step_shape() {
        let actions = parse_cells(
            &json!({"cells": [{"intent": "bind", "code": "x = 1"}]}),
            "code_cell",
        )
        .expect("well-formed step parses");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].intent(), "bind");
        assert_eq!(actions[0].code(), "x = 1");
    }

    #[test]
    fn parse_cells_rejects_what_is_not_a_step() {
        for bad in [
            json!({}),
            json!({"cells": []}),
            json!({"cells": [{"intent": "no code"}]}),
            json!({"cells": [{"code": "1"}]}),
            json!({"cells": "x = 1"}),
        ] {
            assert!(
                parse_cells(&bad, "code_cell").is_err(),
                "must reject {bad} as tool args"
            );
        }
    }

    #[test]
    fn parse_cells_recovers_a_cleanly_stringified_step() {
        // Live session 20261029_fff23e77: the model double-encoded the array
        // as a JSON string (four parse failures in one run). A clean encoding
        // carries the same intent — recover it instead of burning a turn.
        let actions = parse_cells(
            &json!({"cells": "[{\"intent\": \"bind\", \"code\": \"x = 1\"}]"}),
            "code_cell",
        )
        .expect("a stringified step is the fumble, recover it");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].intent(), "bind");
        assert_eq!(actions[0].code(), "x = 1");
    }

    #[test]
    fn parse_cells_names_the_fumble_and_the_shape() {
        // The live failure shape (20261029_fff23e77): doubly encoded AND the
        // inner escaping half-done — some quotes escaped, some raw — so even
        // the inner JSON won't parse. The error must teach (what went wrong /
        // expected shape / how to fix) and echo only a blob head, not replay
        // the whole blob back at the model.
        let blob = r#"[{"code": "out = subprocess.run(['bash','-lc','echo "== unwrap =="; grep -E \"\\.unwrap\\(\\)\" .'])", "intent": "sweep"}]"#;
        let err = parse_cells(&json!({"cells": blob}), "code_cell")
            .expect_err("an unparseable stringified blob is still a tool error");
        let msg = err.to_string();
        assert!(
            msg.contains("JSON-encoded string"),
            "what went wrong: {msg}"
        );
        assert!(
            msg.contains(r#"{"intent": "…", "code": "…"}"#),
            "expected shape: {msg}"
        );
        assert!(msg.contains("single quotes"), "how to fix: {msg}");
        assert!(
            msg.len() < 600,
            "the blob echo must be truncated, not replayed: {msg}"
        );
    }

    #[test]
    fn inline_image_bytes_ride_content() {
        let image = ImageRef::inline("image/png", vec![1, 2, 3], "three bytes");
        match image_content(&image) {
            Content::Image { data, mime_type } => {
                assert_eq!(mime_type, "image/png");
                assert_eq!(
                    data,
                    base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3])
                );
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }

    #[test]
    fn a_failed_image_read_becomes_an_honest_sentence() {
        let image = ImageRef::from_path("image/png", "/nonexistent/never.png", "a shot of nothing");
        match image_content(&image) {
            Content::Text { text } => {
                assert!(
                    text.contains("a shot of nothing"),
                    "alt must survive: {text}"
                );
                assert!(text.contains("bytes unavailable"), "{text}");
            }
            other => panic!("expected honest text, got {other:?}"),
        }
    }

    /// A deterministic noisy picture: a flat colour compresses to almost
    /// nothing, so the fixture needs entropy to land over the transcode
    /// threshold the way a real screenshot does.
    fn noisy_png(width: u32, height: u32) -> Vec<u8> {
        let mut rgb = image::RgbImage::new(width, height);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for p in rgb.pixels_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *p = image::Rgb([
                (state >> 33) as u8,
                (state >> 41) as u8,
                (state >> 49) as u8,
            ]);
        }
        let mut buf = Vec::new();
        rgb.write_with_encoder(image::codecs::png::PngEncoder::new(&mut buf))
            .unwrap();
        buf
    }

    #[test]
    fn a_large_png_goes_on_the_wire_as_a_smaller_jpeg_of_the_same_size() {
        let png = noisy_png(900, 600);
        assert!(
            png.len() > IMAGE_TRANSCODE_OVER_BYTES,
            "fixture must exceed the threshold, got {} bytes",
            png.len()
        );
        let image = ImageRef::inline("image/png", png.clone(), "a noisy screenshot");
        match image_content(&image) {
            Content::Image { data, mime_type } => {
                assert_eq!(mime_type, "image/jpeg", "the wire copy is a JPEG");
                let jpeg = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .unwrap();
                assert!(
                    jpeg.len() < png.len(),
                    "transcode must shrink: {} vs {} original bytes",
                    jpeg.len(),
                    png.len()
                );
                let decoded = image::load_from_memory(&jpeg).unwrap();
                assert_eq!(
                    (decoded.width(), decoded.height()),
                    (900, 600),
                    "geometry is what the model's coordinate math relies on — only the codec may change"
                );
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }

    #[test]
    fn a_small_picture_keeps_its_exact_bytes() {
        let mut buf = Vec::new();
        image::RgbImage::new(16, 16)
            .write_with_encoder(image::codecs::png::PngEncoder::new(&mut buf))
            .unwrap();
        assert!(buf.len() <= IMAGE_TRANSCODE_OVER_BYTES);
        let image = ImageRef::inline("image/png", buf.clone(), "a tiny icon");
        match image_content(&image) {
            Content::Image { data, mime_type } => {
                assert_eq!(mime_type, "image/png", "under the budget: untouched");
                assert_eq!(data, base64::engine::general_purpose::STANDARD.encode(buf));
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }

    #[test]
    fn undecodable_oversized_bytes_still_reach_the_model() {
        let garbage = vec![0xABu8; IMAGE_TRANSCODE_OVER_BYTES + 1];
        let image = ImageRef::inline("image/png", garbage.clone(), "a corrupt capture");
        match image_content(&image) {
            Content::Image { data, mime_type } => {
                assert_eq!(
                    mime_type, "image/png",
                    "no silent re-label when nothing was re-encoded"
                );
                assert_eq!(
                    data,
                    base64::engine::general_purpose::STANDARD.encode(garbage),
                    "a picture the codec cannot parse is still better than a silent loss"
                );
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_observation_says_so() {
        let out = render(Observation::empty(), None, None);
        assert_eq!(out.len(), 1);
        assert_eq!(content_text(&out), "(no output)");
    }

    #[test]
    fn an_oversized_observation_is_truncated_not_rejected() {
        // Regression: a big print used to come back as
        // `Tool 'code_cell' output exceeds the 16000-char limit` — the engine
        // rejected the whole call, so the model lost the result of code that
        // had already run (session 20261029_30f097df).
        let obs = Observation {
            stdout: format!("HEAD\n{}\nTAIL", "x".repeat(30_000)),
            stderr: "a warning from the tail side".to_string(),
            error: Some(ExecError::CellError {
                index: 0,
                kind: "ValueError".to_string(),
                message: "boom".to_string(),
            }),
            events: vec![CellEvent::TodoChanged {
                id: "1".to_string(),
                status: "done".to_string(),
            }],
        };
        let out = render(obs, Some(16_000), None);
        let text = content_text(&out);
        assert!(
            text.chars().count() <= 16_000,
            "must fit the cap, got {} chars",
            text.chars().count()
        );
        assert!(text.contains(TRUNCATION_MARKER), "{text:.60}");
        assert!(text.starts_with("HEAD"), "{text:.60}");
        for tail_piece in [
            "TAIL",
            "a warning from the tail side",
            "error in block #0 (ValueError): boom",
            "todo 1: done",
        ] {
            assert!(text.contains(tail_piece), "lost {tail_piece:?}: {text:.60}");
        }
    }

    #[test]
    fn a_small_observation_rides_whole() {
        let obs = Observation {
            stdout: "42".to_string(),
            ..Observation::empty()
        };
        let out = render(obs, Some(16_000), None);
        assert_eq!(content_text(&out), "42");
    }

    #[test]
    fn images_survive_the_truncation() {
        let obs = Observation {
            stdout: "x".repeat(30_000),
            events: vec![CellEvent::Image(ImageRef::inline(
                "image/png",
                vec![1, 2, 3],
                "three bytes",
            ))],
            ..Observation::empty()
        };
        let out = render(obs, Some(500), None);
        assert!(
            out.iter().any(|c| matches!(c, Content::Image { .. })),
            "the typed channel must keep the image"
        );
        assert!(content_text(&out).chars().count() <= 500);
    }

    /// M3c (action-executor-design §6.1): over cap with a store mounted — the
    /// body is fetchable byte for byte, and the pane keeps the family preview
    /// plus the summary+reference line, fitted inside the cap (the hard
    /// reject must never fire) with a typed face alongside.
    #[test]
    fn an_oversized_observation_spills_and_keeps_a_reference_line() {
        let tmp = tempfile::TempDir::new().unwrap();
        let hub = ArtifactHub::new(tmp.path().join("captures"));
        let body = format!("HEAD\n{}\nTAIL", "x".repeat(30_000));
        let obs = Observation {
            stdout: body.clone(),
            ..Observation::empty()
        };
        let out = render(obs, Some(16_000), Some(&hub));
        let text = content_text(&out);
        assert!(
            text.chars().count() <= 16_000,
            "must fit the cap, got {} chars",
            text.chars().count()
        );
        assert!(text.starts_with("HEAD"), "{text:.80}");
        assert!(
            text.contains("TAIL"),
            "the tail preview must survive: {text:.120}"
        );
        assert!(
            text.contains("obs.fetch('art_0001') for the full body"),
            "the way back must be on the pane: {text:.200}"
        );
        // 账-物对账: the whole joined body landed in the store, byte for byte.
        assert_eq!(
            hub.store().fetch("art_0001").as_deref(),
            Some(body.as_str()),
            "the spill must be the full body"
        );
        assert!(
            out.iter().any(
                |c| matches!(c, Content::Artifact { capture_id, .. } if capture_id == "art_0001")
            ),
            "the typed face rides alongside: {out:?}"
        );
    }

    /// A spilled body's event renders its standing reference — the line is the
    /// model face, the typed item marks it for UI/logs (content_text skips it).
    #[test]
    fn a_spilled_reference_event_renders_line_and_typed_face() {
        let obs = Observation {
            events: vec![CellEvent::Artifact {
                capture_id: "art_0001".to_string(),
                kind: "stdout".to_string(),
                bytes: 11,
                summary: "first body".to_string(),
            }],
            ..Observation::empty()
        };
        let out = render(obs, None, None);
        let text = content_text(&out);
        assert!(
            text.contains(
                "artifact art_0001 (stdout, 11 bytes): first body — obs.fetch('art_0001') for the full body"
            ),
            "{text}"
        );
        assert!(
            out.iter().any(|c| matches!(c, Content::Artifact { .. })),
            "{out:?}"
        );
        // The typed item must not double the line in the model face.
        assert_eq!(text.matches("first body").count(), 1, "{text}");
    }

    /// A worker that echoes its own sequence number instead of running code —
    /// enough to see *which* resident session served a call, no Python needed.
    struct StubWorker {
        id: usize,
    }

    #[async_trait]
    impl phi_ns::WorkerSink for StubWorker {
        fn register_fn(
            &mut self,
            _name: &str,
            _call: Box<dyn phi_ns::HostCall>,
        ) -> Result<(), phi_ns::NsError> {
            Ok(())
        }

        async fn eval(&mut self, _code: &str) -> Result<(), phi_ns::NsError> {
            Ok(())
        }

        async fn namespace_entries(&mut self) -> Result<Vec<phi_ns::NsObject>, phi_ns::NsError> {
            Ok(Vec::new())
        }
    }

    #[async_trait]
    impl Worker for StubWorker {
        async fn exec(
            &mut self,
            blocks: Vec<codexec::BlockSpec>,
        ) -> Result<codexec::worker::RoundResult, ExecError> {
            Ok(codexec::worker::RoundResult {
                blocks: blocks
                    .into_iter()
                    .enumerate()
                    .map(|(index, _)| codexec::worker::BlockOutcome {
                        index,
                        stdout: format!("worker {}", self.id),
                        ..codexec::worker::BlockOutcome::default()
                    })
                    .collect(),
                elapsed_ms: 0,
                compile_error: None,
                compile_error_index: None,
                stopped_at: None,
            })
        }

        fn is_alive(&self) -> bool {
            true
        }

        fn kill(&mut self) {}

        async fn shutdown(&mut self) -> Result<(), ExecError> {
            Ok(())
        }
    }

    /// Opens stub executors, counting how many sessions got one. `pin` marks
    /// the sessions the host would never evict.
    #[derive(Default)]
    struct StubFactory {
        opens: AtomicU64,
        pin: Option<SessionId>,
    }

    impl StubFactory {
        fn pinning(pin: SessionId) -> Self {
            Self {
                opens: AtomicU64::new(0),
                pin: Some(pin),
            }
        }

        fn opens(&self) -> u64 {
            self.opens.load(Ordering::Relaxed)
        }
    }

    impl SessionFactory<StubWorker> for StubFactory {
        fn prompt_face(&self) -> String {
            "stub.pack — stub()".to_string()
        }

        fn limits(&self) -> Limits {
            Limits::default()
        }

        fn open(&self, _session: &SessionId) -> SessionFuture<StubWorker> {
            let id = self.opens.fetch_add(1, Ordering::Relaxed) as usize + 1;
            Box::pin(async move {
                ExecutorBuilder::new(JournalHandle::new(Box::new(MemJournal::default())))
                    .with_policy(AllowAll)
                    .with_limits(Limits::default())
                    .build(move |_registry| Ok(StubWorker { id }))
                    .await
                    .map_err(|e| AgentError::internal(format!("stub engine: {e}")))
            })
        }

        fn pinned(&self, session: &SessionId) -> bool {
            self.pin.as_ref() == Some(session)
        }
    }

    fn cells_call() -> Value {
        json!({"cells": [{"intent": "who", "code": "print(1)"}]})
    }

    fn ctx_for(session: u64) -> ToolContext {
        let mut ctx = ToolContext::for_test();
        ctx.session_id = SessionId::new(session);
        ctx
    }

    fn text_of(contents: &[Content]) -> String {
        contents
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn a_pool_opens_one_executor_per_session_and_keeps_it_resident() {
        let factory = Arc::new(StubFactory::default());
        let pooled: Arc<dyn SessionFactory<StubWorker>> = factory.clone();
        let tool = CodeCellTool::per_session(TOOL_NAME, pooled);

        let first = text_of(
            &tool
                .call(&cells_call(), &ctx_for(1))
                .await
                .expect("the call runs"),
        );
        let second = text_of(
            &tool
                .call(&cells_call(), &ctx_for(1))
                .await
                .expect("the call runs"),
        );
        assert_eq!(factory.opens(), 1, "one session, one resident executor");
        assert!(first.contains("worker 1"), "{first}");
        assert!(second.contains("worker 1"), "{second}");

        let other = text_of(
            &tool
                .call(&cells_call(), &ctx_for(2))
                .await
                .expect("the call runs"),
        );
        assert_eq!(factory.opens(), 2, "a second session opens its own");
        assert!(
            other.contains("worker 2"),
            "session 2 ran its own resident: {other}"
        );
    }

    #[tokio::test]
    async fn the_pool_evicts_the_least_recently_used_unpinned_session() {
        // Session 0 is the host's own (pinned); the rest come and go.
        let factory = Arc::new(StubFactory::pinning(SessionId::new(0)));
        let pooled: Arc<dyn SessionFactory<StubWorker>> = factory.clone();
        let tool = CodeCellTool::per_session(TOOL_NAME, pooled);

        for session in 0..=MAX_RESIDENT_SESSIONS as u64 {
            tool.call(&cells_call(), &ctx_for(session))
                .await
                .expect("the call runs");
        }
        assert_eq!(factory.opens(), MAX_RESIDENT_SESSIONS as u64 + 1);

        // Session 1 is the oldest unpinned one — the pool dropped it to make
        // room, so revisiting it costs an open. Session 0 is pinned and older
        // still: it must never have been dropped.
        tool.call(&cells_call(), &ctx_for(1))
            .await
            .expect("the call runs");
        assert_eq!(
            factory.opens(),
            MAX_RESIDENT_SESSIONS as u64 + 2,
            "the evicted session opened a fresh executor"
        );
        tool.call(&cells_call(), &ctx_for(0))
            .await
            .expect("the call runs");
        assert_eq!(
            factory.opens(),
            MAX_RESIDENT_SESSIONS as u64 + 2,
            "the pinned session never left the pool"
        );
    }

    #[tokio::test]
    async fn a_factory_failure_is_a_tool_error_not_a_panic() {
        struct BrokenFactory;

        impl SessionFactory<StubWorker> for BrokenFactory {
            fn prompt_face(&self) -> String {
                String::new()
            }

            fn limits(&self) -> Limits {
                Limits::default()
            }

            fn open(&self, _session: &SessionId) -> SessionFuture<StubWorker> {
                Box::pin(async { Err(AgentError::internal("no worker available")) })
            }
        }

        let tool = CodeCellTool::per_session(TOOL_NAME, Arc::new(BrokenFactory));
        let err = tool.call(&cells_call(), &ctx_for(1)).await;
        assert!(
            err.is_err(),
            "an unopenable session fails the call honestly"
        );
    }

    fn content_text(contents: &[Content]) -> String {
        contents
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
