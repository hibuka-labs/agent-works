use super::adapt::observation_events;
use super::render::observation_text;
use super::*;
use agent_base::llm_trait::{
    self, Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};
use agent_base::{
    AgentBuilder, AgentRuntime, ApprovalDecision, ApprovalRequest, AskAnswer, AskOption,
    CheckpointData, CheckpointStep, InteractionChannel, InteractionError, ObservationEvent,
    RiskLevel, RunOutcome, RunState, StreamChunk,
};
use codexec::observation::CellEvent;
use codexec::{
    AllowAll, ClassPolicy, ExecutorBuilder, InteractionBridge, JournalHandle, Limits, MemJournal,
    Observation as EngineObservation, PolicyDecision, PythonWorker,
};
use phi_cap::Capability;
use phi_ns::builtin::PhiNsShell;
use std::sync::Mutex;
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
    /// The user typed a question instead of picking ("Chat about this").
    Clarify(String),
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
            AskScript::Clarify(text) => Ok(AskAnswer::Clarify { text: text.clone() }),
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

/// protects: the host-call answer wire — every `ask_user` exit serializes to
/// its documented JSON shape (pick/skip byte-stable; a "Chat about this"
/// clarification rides `{"clarify": <text>}` verbatim).
/// fails_when: any arm's bytes drift, or a clarification is dropped/renamed.
/// no_claim: the product's picker face or the tool-result wording (phimint
/// cards) — only what a cell's `ask.user` returns.
#[tokio::test]
async fn the_bridge_answer_wire_is_byte_stable_for_every_exit() {
    async fn bridge_answer(script: AskScript) -> String {
        let bridge = ExecCtxBridge::new();
        let channel = Arc::new(Scripted::new(ConsentScript::Allow, script));
        bridge.arm(&ctx_on(CancellationToken::new(), channel));
        let wire = vec![serde_json::json!({"id": "a", "label": "Alpha"})];
        bridge.ask("which one?", &wire).await.unwrap().to_string()
    }

    assert_eq!(
        bridge_answer(AskScript::Pick("a".into(), "Alpha".into())).await,
        r#"{"picked":{"id":"a","label":"Alpha"}}"#
    );
    assert_eq!(bridge_answer(AskScript::Skip).await, r#"{"skipped":true}"#);
    assert_eq!(
        bridge_answer(AskScript::Clarify("what scale?".into())).await,
        r#"{"clarify":"what scale?"}"#
    );
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
