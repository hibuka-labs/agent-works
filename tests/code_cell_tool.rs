//! M2 acceptance through the agent `Tool` seam (codexec-design §11): one call
//! = one step of cells, and the resident session survives calls — the CodeAct
//! property (codexec-design §10) proven through `Tool::call`, not just the
//! engine face.
//!
//! Runs with `--features code-exec` (the target declares `required-features`);
//! the plain `cargo test` gate skips it.

use std::sync::Arc;

use agent_base::types::SessionId;
use agent_base::{AgentError, Content, Tool, ToolContext};
use serde_json::json;

use agent_works::code_cell::{
    AllowAll, Capability, ClassPolicy, CodeCellTool, ExecutorBuilder, FileJournal, JournalHandle,
    Limits, MemJournal, PolicyDecision, PythonWorker, SessionFactory, SessionFuture,
};
use phi_ns::builtin::PhiNsShell;

struct Stage {
    _dir: tempfile::TempDir,
    journal: std::path::PathBuf,
    tool: CodeCellTool<PythonWorker>,
}

async fn stage() -> Stage {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal = dir.path().join("journal.jsonl");
    let limits = Limits::default();
    let file = FileJournal::open(&journal).expect("journal opens");
    let executor = ExecutorBuilder::new(JournalHandle::new(Box::new(file)))
        .with_policy(AllowAll)
        .with_limits(limits)
        .with_ns(PhiNsShell::builtin())
        .build(move |registry| PythonWorker::spawn(registry, limits))
        .await
        .expect("engine assembles");
    Stage {
        _dir: dir,
        journal,
        tool: CodeCellTool::new(executor),
    }
}

/// One tool call's worth of cells.
fn cells(items: &[(&str, &str)]) -> serde_json::Value {
    let cells: Vec<serde_json::Value> = items
        .iter()
        .map(|(intent, code)| json!({"intent": intent, "code": code}))
        .collect();
    json!({ "cells": cells })
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

fn journal_rows(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

#[tokio::test]
async fn one_call_runs_the_step_and_the_session_survives() {
    let s = stage().await;
    let ctx = ToolContext::for_test();

    let out = s
        .tool
        .call(
            &cells(&[("bind", "x = 41"), ("print", "print(x + 1)")]),
            &ctx,
        )
        .await
        .expect("the step runs");
    let text = text_of(&out);
    assert!(
        text.contains("42"),
        "cell 2 sees cell 1's variable:\n{text}"
    );
    assert!(
        text.contains("namespace:"),
        "the observation carries its snapshot:\n{text}"
    );

    // The CodeAct property across *calls*: x is still there.
    let out = s
        .tool
        .call(&cells(&[("reuse", "print(x * 2)")]), &ctx)
        .await
        .expect("the second step runs");
    let text = text_of(&out);
    assert!(text.contains("82"), "state survives tool calls:\n{text}");
}

#[tokio::test]
async fn the_output_cap_truncates_the_observation_instead_of_failing_the_call() {
    // Regression (session 20261029_30f097df): a big print used to blow the
    // engine's tool-output cap, so the call came back as
    // `Tool 'code_cell' output exceeds the 16000-char limit` — the model lost
    // the result of code that had already run. The observation self-truncates
    // (kernel convention: head + `...[output truncated]` + tail) instead.
    let s = stage().await;
    let mut ctx = ToolContext::for_test();
    ctx.max_output_chars = Some(16_000);

    let out = s
        .tool
        .call(&cells(&[("flood", "print('x' * 30000)")]), &ctx)
        .await
        .expect("the call must succeed — the code ran");
    let text = text_of(&out);
    assert!(
        text.chars().count() <= 16_000,
        "observation must fit the cap, got {} chars",
        text.chars().count()
    );
    assert!(
        text.contains("...[output truncated]"),
        "the model must see the cut:\n{text:.60}"
    );
}

#[tokio::test]
async fn the_description_carries_the_installed_namespaces() {
    let s = stage().await;
    assert_eq!(s.tool.name(), "code_cell");
    assert!(
        s.tool.description().contains("sh."),
        "the prompt face derives from the installed packs"
    );
    assert!(s.tool.prompt_face().contains("sh."), "same source, twice");
}

#[tokio::test]
async fn bad_arguments_never_reach_the_engine() {
    let s = stage().await;
    let ctx = ToolContext::for_test();
    for bad in [
        json!({}),
        json!({"cells": []}),
        json!({"cells": [{"intent": "no code"}]}),
        json!({"cells": [{"code": "1"}]}),
        json!({"cells": "x = 1"}),
    ] {
        assert!(
            s.tool.call(&bad, &ctx).await.is_err(),
            "must reject {bad} as tool args"
        );
    }
}

#[tokio::test]
async fn a_cell_failure_is_an_observation_the_model_acts_on() {
    let s = stage().await;
    let ctx = ToolContext::for_test();
    let out = s
        .tool
        .call(&cells(&[("boom", "1 / 0")]), &ctx)
        .await
        .expect("tool args were fine — the call itself succeeds");
    let text = text_of(&out);
    assert!(
        text.contains("error:"),
        "the failure rides as content:\n{text}"
    );
    assert!(
        text.contains("ZeroDivisionError"),
        "the model sees the exception:\n{text}"
    );
}

#[tokio::test]
async fn side_effects_are_journaled_one_row_per_call() {
    let s = stage().await;
    let ctx = ToolContext::for_test();
    s.tool
        .call(&cells(&[("run", "sh.run(\"echo hi\")")]), &ctx)
        .await
        .expect("the step runs");
    let rows = journal_rows(&s.journal);
    let calls: Vec<&String> = rows
        .iter()
        .filter(|r| r.contains("\"kind\":\"call\""))
        .collect();
    assert_eq!(calls.len(), 1, "one side effect, one call row: {rows:?}");
    assert!(calls[0].contains("\"fn_name\":\"sh.run\""), "{}", calls[0]);
    assert!(calls[0].contains("\"decision\":\"allow\""), "{}", calls[0]);
    let cell_rows: Vec<&String> = rows
        .iter()
        .filter(|r| r.contains("\"kind\":\"cell\""))
        .collect();
    assert_eq!(
        cell_rows.len(),
        1,
        "one submitted cell, one cell row: {rows:?}"
    );
}

#[tokio::test]
async fn the_watchdog_budget_outlives_the_engine_round() {
    let s = stage().await;
    let budget = s.tool.timeout_ms().expect("a budget is set");
    assert!(
        budget > Limits::default().timeout_ms,
        "the engine must always time out before the framework watchdog (got {budget})"
    );
}

/// Per-session assembly — the sub-agent shape: one resident executor per
/// session, each journaling to its own file. `read_only` swaps AllowAll for
/// the audit-sealed read-only posture (reads pass, writes are refused at the
/// seam).
struct PoolFactory {
    dir: tempfile::TempDir,
    read_only: bool,
    face: String,
}

impl PoolFactory {
    fn new(read_only: bool) -> Self {
        let face = ExecutorBuilder::new(JournalHandle::new(Box::new(MemJournal::default())))
            .with_ns(PhiNsShell::builtin())
            .prompt_face();
        Self {
            dir: tempfile::tempdir().expect("tempdir"),
            read_only,
            face,
        }
    }
}

impl SessionFactory<PythonWorker> for PoolFactory {
    fn prompt_face(&self) -> String {
        self.face.clone()
    }

    fn limits(&self) -> Limits {
        Limits::default()
    }

    fn open(&self, session: &SessionId) -> SessionFuture<PythonWorker> {
        let journal = self
            .dir
            .path()
            .join(format!("code_journal-{}.jsonl", session.id));
        let read_only = self.read_only;
        Box::pin(async move {
            let file = FileJournal::open(&journal).expect("journal opens");
            let limits = Limits::default();
            let builder = ExecutorBuilder::new(JournalHandle::new(Box::new(file)))
                .with_limits(limits)
                .with_ns(PhiNsShell::builtin());
            let builder = if read_only {
                builder.with_policy(
                    ClassPolicy::new(PolicyDecision::Deny)
                        .with(Capability::ReadFs, PolicyDecision::Allow),
                )
            } else {
                builder.with_policy(AllowAll)
            };
            builder
                .build(move |registry| PythonWorker::spawn(registry, limits))
                .await
                .map_err(|e| AgentError::internal(format!("engine assembles: {e}")))
        })
    }
}

fn ctx_for(session: u64) -> ToolContext {
    let mut ctx = ToolContext::for_test();
    ctx.session_id = SessionId::new(session);
    ctx
}

#[tokio::test]
async fn sessions_in_one_process_do_not_share_state() {
    // The sub-agent shape: one tool, several sessions, one resident
    // interpreter each. What one session binds must not leak into another's —
    // and each keeps its own across calls.
    let tool = CodeCellTool::per_session("code_cell", Arc::new(PoolFactory::new(false)));

    tool.call(&cells(&[("bind", "x = 41")]), &ctx_for(1))
        .await
        .expect("session 1 runs");

    let out = tool
        .call(&cells(&[("probe", "print(x)")]), &ctx_for(2))
        .await
        .expect("session 2 runs");
    let text = text_of(&out);
    assert!(
        text.contains("NameError"),
        "session 2 must not see session 1's bindings:\n{text}"
    );

    let out = tool
        .call(&cells(&[("reuse", "print(x + 1)")]), &ctx_for(1))
        .await
        .expect("session 1 runs again");
    let text = text_of(&out);
    assert!(text.contains("42"), "session 1's state survived:\n{text}");
}

#[tokio::test]
async fn a_read_only_session_can_read_but_not_write() {
    // The host's per-session policy split: a sub-agent's cells read the disk,
    // but a write is refused at the audit seam (PEP 578 mediates `open` into
    // ReadFs / WriteFs — fail-closed).
    let tool = CodeCellTool::per_session("code_cell", Arc::new(PoolFactory::new(true)));
    let dir = tempfile::tempdir().expect("tempdir");
    let seed = dir.path().join("seed.txt");
    std::fs::write(&seed, "hello from disk").expect("seed file");
    let sneaky = dir.path().join("sneaky.txt");

    let ctx = ctx_for(1);
    let read = format!(
        "print(open({:?}, encoding='utf-8').read())",
        seed.display().to_string()
    );
    let out = tool
        .call(&cells(&[("read", read.as_str())]), &ctx)
        .await
        .expect("the call runs");
    let text = text_of(&out);
    assert!(text.contains("hello from disk"), "reads pass:\n{text}");

    let write = format!(
        "open({:?}, 'w').write('sneaky')",
        sneaky.display().to_string()
    );
    let out = tool
        .call(&cells(&[("write", write.as_str())]), &ctx)
        .await
        .expect("tool args were fine — the refusal rides as content");
    let text = text_of(&out);
    assert!(
        text.contains("policy held") && text.contains("write_fs"),
        "the write is refused at the seam:\n{text}"
    );
    assert!(
        !sneaky.exists(),
        "nothing may land on disk after a denied write"
    );
}
