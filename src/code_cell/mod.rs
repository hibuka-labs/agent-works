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
use serde_json::{Value, json};
use tokio::sync::Mutex;

mod render;
use render::render;

pub use codexec::observation::{CellEffects, CellEvent, ExecError, Observation};
pub use codexec::worker::Worker;
// `ArtifactHub` joins the vocabulary list for the shell-output spill adapter
// (hosts mount the engine's store as a `SpillSink` — M3c: adapters use
// store-level `spill`, no CellEvent).
pub use codexec::{
    Action, AllowAll, ArtifactHub, CallCtx, ClassPolicy, CodeCellExecutor, ConsentRequest, DenyAll,
    ExecutorBuilder, FileJournal, InteractionBridge, Journal, JournalHandle, Limits, MemJournal,
    MethodPolicy, Policy, PolicyDecision, PythonWorker,
};
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
journaled; a held call halts the step. Results return only when the whole call \
ends: output, error, the effect tally and the progress trail are one snapshot \
at the end — one call is one observation, so split long staged work across \
several calls to observe between stages. Below are the namespaces installed in \
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
/// One principle for the shapes of "wrong": a fumble whose step survives
/// losslessly is *recovered*, not refused — replaying a parse error back at the
/// model teaches it nothing, and the round-trip costs more than the fumble.
/// Two live fumbles qualify: `cells` as a JSON-encoded *string* of the array
/// (all four parse failures in live session 20261029_fff23e77 were this, under
/// nested-quoting pressure), and the `cells` wrapper dropped entirely — one
/// flat cell at top level, or a single cell object where the array belongs
/// (live session 20261101_43d6c579, deepseek-flash, again at the most
/// quote-heavy payloads). Anything that loses fields is refused with the shape
/// spelled out and the blob head.
fn parse_cells(args: &Value, name: &str) -> AgentResult<Vec<Action>> {
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
    let flat_step;
    let cells_arg: &Value = match args.get("cells") {
        Some(cells_arg) => cells_arg,
        // The wrapper dropped, the cell itself intact: wrap it back.
        None if args.get("code").is_some() => {
            flat_step = Value::Array(vec![args.clone()]);
            &flat_step
        }
        None => {
            return Err(guidance(
                "there is no top-level \"cells\" key",
                &args.to_string(),
            ));
        }
    };
    let decoded;
    let decoded_cell;
    let cells: &[Value] = match cells_arg {
        Value::Array(cells) => cells.as_slice(),
        Value::String(encoded) => match serde_json::from_str::<Vec<Value>>(encoded) {
            Ok(cells) if !cells.is_empty() => {
                decoded = cells;
                decoded.as_slice()
            }
            _ => return Err(guidance("it arrived as a JSON-encoded string", encoded)),
        },
        // One cell object where the step array belongs: same lossless wrap.
        Value::Object(_) if cells_arg.get("code").is_some() => {
            decoded_cell = vec![cells_arg.clone()];
            decoded_cell.as_slice()
        }
        other => {
            return Err(guidance(
                "it arrived as a non-array value",
                &other.to_string(),
            ));
        }
    };
    let mut actions = Vec::with_capacity(cells.len());
    for (idx, cell) in cells.iter().enumerate() {
        // A missing field names the cell's position and the field, so the model
        // fixes the call instead of re-reading a bare cell echo (live session
        // 20261101_9c6b6bc7: DeepSeek dropped `intent` on one of two cells and
        // the bare echo looked like the whole call was `{"code": …}`).
        let intent = cell.get("intent").and_then(Value::as_str).ok_or_else(|| {
            guidance(
                &format!("cell {} is missing a string \"intent\" label", idx + 1),
                &cell.to_string(),
            )
        })?;
        let code = cell.get("code").and_then(Value::as_str).ok_or_else(|| {
            guidance(
                &format!("cell {} is missing a string \"code\" body", idx + 1),
                &cell.to_string(),
            )
        })?;
        actions.push(Action::CodeCell {
            intent: intent.to_string(),
            code: code.to_string(),
        });
    }
    if actions.is_empty() {
        return Err(guidance(
            "the \"cells\" array is empty",
            &cells_arg.to_string(),
        ));
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
    fn parse_cells_names_a_missing_cell_field_and_its_position() {
        // Live session 20261101_9c6b6bc7 (DeepSeek): a two-cell step where one
        // cell dropped `intent`. The bare cell echo read like the whole call was
        // `{"code": …}`; the error must name the field, the cell's position and
        // the expected shape so the model can repair the call in one shot.
        let err = parse_cells(
            &json!({"cells": [
                {"code": "out = sh.run('ls -la')"},
                {"code": "print(2)", "intent": "probe"}
            ]}),
            "code_cell",
        )
        .expect_err("a cell missing its intent is still a tool error");
        let msg = err.to_string();
        assert!(msg.contains("cell 1"), "which cell: {msg}");
        assert!(msg.contains("\"intent\""), "which field: {msg}");
        assert!(
            msg.contains(r#"{"intent": "…", "code": "…"}"#),
            "expected shape: {msg}"
        );
        assert!(
            msg.contains(r#"{"code":"out = sh.run('ls -la')"}"#),
            "echo the cell, not the whole step: {msg}"
        );
    }

    #[test]
    fn parse_cells_wraps_a_flat_single_cell_step() {
        // Live session 20261101_43d6c579 (deepseek-flash): at the most
        // quote-heavy payload of the run the step arrived as one flat cell at
        // top level — `code` and `intent`, no `cells` wrapper. Lossless: run
        // it as the one-cell step it obviously is.
        let actions = parse_cells(
            &json!({"intent": "probe", "code": "probe = r'''\nprint(1)\n'''"}),
            "code_cell",
        )
        .expect("a flat single cell is the fumble, recover it");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].intent(), "probe");
        assert_eq!(actions[0].code(), "probe = r'''\nprint(1)\n'''");
    }

    #[test]
    fn parse_cells_wraps_a_single_cell_object() {
        let actions = parse_cells(
            &json!({"cells": {"intent": "probe", "code": "print(1)"}}),
            "code_cell",
        )
        .expect("a single cell object is the fumble, recover it");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].intent(), "probe");
    }

    #[test]
    fn parse_cells_still_refuses_a_flat_cell_that_loses_fields() {
        // A flat cell without `intent` is not lossless — refuse with the shape
        // spelled out rather than inventing an audit label.
        let err = parse_cells(&json!({"code": "print(1)"}), "code_cell")
            .expect_err("a flat cell missing intent still loses the audit label");
        let msg = err.to_string();
        assert!(
            msg.contains(r#"{"intent": "…", "code": "…"}"#),
            "expected shape: {msg}"
        );
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
                progress: Vec::new(),
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
}
