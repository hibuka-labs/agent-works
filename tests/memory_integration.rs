//! Phase 9b integration tests: the full auto-memory flow through the public
//! API — write → read → list → delete with index consistency checks, first-use
//! auto-init, and builder wiring.
//!
//! These tests are feature-gated: run with
//! `cargo test -p agent-works --features memory`.

#![cfg(feature = "memory")]

use std::sync::Arc;

use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider,
};
use agent_base::{Content, Tool, ToolContext};
use agent_works::{AgentBuilder, MemoryConfig, MemoryStore};
use async_trait::async_trait;
use serde_json::{Value, json};

/// Minimal LLM stub (no turns are executed; the builder only needs a provider).
struct StubProvider;

#[async_trait]
impl LlmProvider for StubProvider {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chunks = vec![
            Ok(agent_base::StreamChunk::Text("ok".to_string())),
            Ok(agent_base::StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ];
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
            chunks,
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            content: "ok".to_string(),
            tool_calls: vec![],
            usage: agent_base::UsageInfo::default(),
            finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tools: true,
            ..Default::default()
        }
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

/// Fixture: the three memory tools bound to a fresh temp memory directory.
///
/// Reading is deliberately absent — the contract is that the index row names
/// the file and `read_file` opens it, so these tests read it from disk.
struct Fixture {
    _dir: tempfile::TempDir,
    memory_dir: std::path::PathBuf,
    store: Arc<MemoryStore>,
    write: Arc<dyn Tool>,
    list: Arc<dyn Tool>,
    delete: Arc<dyn Tool>,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let memory_dir = dir.path().join("memory");
        let store = Arc::new(MemoryStore::new(memory_dir.clone(), "MEMORY.md"));
        let mut tools = agent_works::create_memory_tools(Arc::clone(&store));
        let delete = tools.pop().unwrap();
        let list = tools.pop().unwrap();
        let write = tools.pop().unwrap();
        Self {
            _dir: dir,
            memory_dir,
            store,
            write,
            list,
            delete,
        }
    }

    fn root(&self) -> std::path::PathBuf {
        self.memory_dir.clone()
    }

    fn index(&self) -> String {
        std::fs::read_to_string(self.root().join("MEMORY.md")).unwrap_or_default()
    }

    /// Read a memory the way the contract says the model does: the index row
    /// names the file, and the file is plain markdown on disk.
    fn read_memory(&self, name: &str) -> String {
        std::fs::read_to_string(self.root().join(format!("{name}.md"))).unwrap_or_default()
    }
}

fn first_text(out: &[Content]) -> String {
    match out.first() {
        Some(Content::Text { text }) => text.clone(),
        _ => String::new(),
    }
}

async fn call(tool: &dyn Tool, args: Value) -> String {
    let out = tool.call(&args, &ToolContext::for_test()).await.unwrap();
    first_text(&out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Full flow: write → read → list → delete → index consistent
// ─────────────────────────────────────────────────────

#[tokio::test]
async fn full_write_read_list_delete_flow_keeps_index_consistent() {
    let fx = Fixture::new();

    // WRITE: two memories in different types.
    let out = call(
        fx.write.as_ref(),
        json!({
            "name": "cargo-commit-discipline",
            "description": "绝不主动 commit",
            "type": "feedback",
            "body": "0. 绝不主动 commit\n1. commit 永不带 Cargo 文件"
        }),
    )
    .await;
    assert!(out.contains("created"), "{out}");

    call(
        fx.write.as_ref(),
        json!({
            "name": "working-principles",
            "description": "架构高内聚低耦合",
            "type": "user",
            "body": "原则一：高内聚低耦合"
        }),
    )
    .await;

    // Disk: two memory files + index with exactly two rows.
    assert!(fx.root().join("cargo-commit-discipline.md").exists());
    assert!(fx.root().join("working-principles.md").exists());
    let index = fx.index();
    assert_eq!(index.lines().count(), 2, "{index}");
    assert!(index.contains("(cargo-commit-discipline.md)"), "{index}");
    assert!(index.contains("(working-principles.md)"), "{index}");

    // READ: the file the index row names round-trips as plain markdown.
    let raw = fx.read_memory("cargo-commit-discipline");
    assert!(raw.starts_with("---\n"), "{raw}");
    assert!(raw.contains("name: cargo-commit-discipline"));
    assert!(raw.contains("type: feedback"));
    assert!(raw.contains("originSessionId"));
    assert!(raw.contains("commit 永不带 Cargo 文件"));

    // LIST: full index returned.
    let listed = call(fx.list.as_ref(), json!({})).await;
    assert_eq!(listed, index, "memory_list must return the live index");

    // UPDATE same name: still two files, one row, second content wins.
    call(
        fx.write.as_ref(),
        json!({
            "name": "working-principles",
            "description": "架构高内聚低耦合（更新版）",
            "type": "user",
            "body": "原则一：高内聚低耦合；原则二：工作严谨"
        }),
    )
    .await;
    let files: Vec<_> = std::fs::read_dir(fx.root())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".md") && n != "MEMORY.md")
        .collect();
    assert_eq!(files.len(), 2, "no duplicates: {files:?}");
    let index = fx.index();
    assert_eq!(index.lines().count(), 2, "{index}");
    assert!(index.contains("更新版"), "{index}");

    // DELETE: file gone, index row gone, the other memory untouched.
    let out = call(
        fx.delete.as_ref(),
        json!({"name": "cargo-commit-discipline"}),
    )
    .await;
    assert!(out.contains("deleted"), "{out}");
    assert!(!fx.root().join("cargo-commit-discipline.md").exists());
    let index = fx.index();
    assert_eq!(index.lines().count(), 1, "{index}");
    assert!(!index.contains("cargo-commit-discipline"), "{index}");
    assert!(index.contains("working-principles"), "{index}");

    // The deleted memory is gone from disk too — with no read tool, "the file
    // does not exist" is what the model sees when it follows an index row that
    // is no longer there.
    assert!(
        !fx.root().join("cargo-commit-discipline.md").exists(),
        "delete must remove the file as well as the index row"
    );
    assert_eq!(
        fx.read_memory("cargo-commit-discipline"),
        "",
        "reading a deleted memory yields nothing"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// First use: directory absent → write auto-creates everything
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn first_use_auto_creates_dir_and_index() {
    let fx = Fixture::new();
    assert!(!fx.root().exists(), "fixture must start absent");
    assert!(
        call(fx.list.as_ref(), json!({}))
            .await
            .contains("No memories")
    );

    call(
        fx.write.as_ref(),
        json!({"name": "first-memory", "description": "初体验", "type": "project", "body": "hello"}),
    )
    .await;

    assert!(fx.root().is_dir(), "memory dir auto-created");
    assert!(fx.root().join("first-memory.md").exists());
    assert!(fx.root().join("MEMORY.md").exists());

    // Subsequent read/list work normally.
    let raw = fx.read_memory("first-memory");
    assert!(raw.contains("hello"), "{raw}");
    let listed = call(fx.list.as_ref(), json!({})).await;
    assert!(listed.contains("初体验"), "{listed}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Fault tolerance: corrupt memory file doesn't break list; rebuild repairs index
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn corrupt_files_are_skipped_and_rebuild_repairs_index() {
    let fx = Fixture::new();
    call(
        fx.write.as_ref(),
        json!({"name": "good", "description": "fine", "type": "reference", "body": "ok"}),
    )
    .await;
    // Claude Code writes a bad file directly on disk.
    std::fs::write(fx.root().join("broken.md"), "no frontmatter here\n").unwrap();

    // Listing skips the corrupt file but keeps the good one.
    let entries = fx.store.list_memories();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "good");

    // Delete the index to simulate corruption, then rebuild from files.
    std::fs::remove_file(fx.root().join("MEMORY.md")).unwrap();
    agent_works::rebuild_index(&fx.root(), "MEMORY.md").unwrap();
    let index = fx.index();
    assert_eq!(index, "- [good](good.md) — fine\n", "{index}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-writer compatibility: Claude Code and phimint share one memory
// directory. Neither may destroy what the other wrote (format, unknown
// frontmatter keys, human index titles).
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn claude_code_written_memory_survives_phimint_update() {
    let fx = Fixture::new();
    // Claude Code writes a memory file in its own documented format —
    // `metadata` carries only `type` (no node_type/stamps) — plus a frontmatter
    // key neither writer owns, and an index row with a human title.
    std::fs::create_dir_all(fx.root()).unwrap();
    std::fs::write(
        fx.root().join("cc-note.md"),
        "---\nname: cc-note\ndescription: old desc\ntags: [a, b]\nmetadata:\n  type: feedback\n---\n\nold body\n",
    )
    .unwrap();
    std::fs::write(
        fx.root().join("MEMORY.md"),
        "- [人类标题](cc-note.md) — old desc\n",
    )
    .unwrap();

    // phimint sees Claude Code's memory (list) and can read it.
    let listed = call(fx.list.as_ref(), json!({})).await;
    assert!(listed.contains("(cc-note.md)"), "{listed}");

    // phimint updates the same memory through its tool.
    call(
        fx.write.as_ref(),
        json!({"name": "cc-note", "description": "new desc", "type": "feedback", "body": "new body"}),
    )
    .await;

    // Unknown frontmatter keys survive; the known fields are refreshed.
    let raw = std::fs::read_to_string(fx.root().join("cc-note.md")).unwrap();
    assert!(raw.contains("tags:"), "unknown key dropped:\n{raw}");
    assert!(raw.contains("type: feedback"), "type dropped:\n{raw}");
    assert!(raw.contains("new body"), "{raw}");
    // The human title in the index row survives; only the description refreshes.
    let index = fx.index();
    assert_eq!(
        index, "- [人类标题](cc-note.md) — new desc\n",
        "update must not clobber the human title: {index}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// In-process concurrency: parallel tokio tasks must not lose index rows
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_tokio_tasks_keep_index_complete() {
    let fx = Fixture::new();
    let mut handles = Vec::new();
    for i in 0..16 {
        let write = Arc::clone(&fx.write);
        handles.push(tokio::spawn(async move {
            write
                .call(
                    &json!({
                        "name": format!("mem-{i:02}"),
                        "description": format!("desc {i}"),
                        "type": "project",
                        "body": format!("body {i}")
                    }),
                    &ToolContext::for_test(),
                )
                .await
                .unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let index = fx.index();
    assert_eq!(index.lines().count(), 16, "no rows may be lost:\n{index}");
    for i in 0..16 {
        assert!(
            index.contains(&format!("(mem-{i:02}.md)")),
            "missing mem-{i:02}:\n{index}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder wiring: memory_config registers tools and injects the index snapshot
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn builder_memory_config_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig::custom(
        dir.path().join("memory"),
        agent_works::CLAUDE_COMPATIBLE_TEMPLATE.to_string(),
    );

    // Pre-seed a memory that must appear in the injected prompt.
    MemoryStore::new(config.memory_root.clone(), "MEMORY.md")
        .write_memory("alpha", "first memory", "user", "body", None)
        .unwrap();

    let runtime = AgentBuilder::new(Arc::new(StubProvider))
        .system_prompt("base")
        .memory_config(config)
        .build()
        .unwrap();

    let names: Vec<String> = tokio::task::block_in_place(|| {
        let registry = runtime.tools_mut();
        let guard = registry.blocking_read();
        guard.metadatas().into_iter().map(|m| m.name).collect()
    });
    for expected in ["memory_write", "memory_list", "memory_delete"] {
        assert!(names.contains(&expected.to_string()), "{names:?}");
    }
    assert!(
        !names.contains(&"memory_read".to_string()),
        "reading must not be a tool: {names:?}"
    );

    let prompt = tokio::task::block_in_place(|| runtime.config().system_prompt.clone().unwrap());
    assert!(prompt.starts_with("base"), "{prompt}");
    assert!(prompt.contains("## Memory"), "{prompt}");
    assert!(
        prompt.contains("- [alpha](alpha.md) — first memory"),
        "{prompt}"
    );

    // Claude-Code-compatible path helper sanity.
    let cc = MemoryConfig::claude_compatible(std::path::Path::new("/Users/xxx/project"));
    assert!(cc.memory_root.ends_with(std::path::Path::new(
        ".claude/projects/-Users-xxx-project/memory"
    )));
    assert_eq!(cc.index_filename, "MEMORY.md");
}

#[tokio::test(flavor = "multi_thread")]
async fn builder_without_memory_config_stays_clean() {
    let runtime = AgentBuilder::new(Arc::new(StubProvider)).build().unwrap();
    let names: Vec<String> = tokio::task::block_in_place(|| {
        let registry = runtime.tools_mut();
        let guard = registry.blocking_read();
        guard.metadatas().into_iter().map(|m| m.name).collect()
    });
    assert!(!names.iter().any(|n| n.starts_with("memory_")), "{names:?}");
    let prompt = tokio::task::block_in_place(|| runtime.config().system_prompt.clone());
    assert!(prompt.is_none());
}
