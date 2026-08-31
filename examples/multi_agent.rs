//! Demonstrates the multi-agent layer: fan out child agents, collect their
//! results, and tear down — end to end, fully offline (stub LLM + stub tools).
//!
//! Run with:
//!   cargo run --example multi_agent --features multi_agent
//!
//! This drives [`MultiAgentRuntime`] directly (the layer beneath the LLM-facing
//! tools). In a real deployment the same primitives are reached by the parent
//! LLM through the 5 multi-agent tools (`spawn_agent` / `send_message` /
//! `wait_agent` / `list_agents` / `close_agent`, provided by
//! `phi-kernel-tools`); the orchestration shape is identical:
//! spawn → task → wait → (optional message + wait) → close.

use std::sync::Arc;
use std::time::Duration;

use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};
use agent_base::{Content, Language, StreamChunk, Tool, ToolContext};
use agent_works::multi_agent::{ChildOutcome, MultiAgentConfig, MultiAgentRuntime};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

// ── Stub LLM: every answer is a fixed text stream ───────────────────

struct StubLlm;

#[async_trait]
impl LlmProvider for StubLlm {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(vec![
            Ok(StreamChunk::Text("all clear".to_string())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ]))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        unreachable!("the react loop streams")
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "stub".into(),
            model: "stub".into(),
            version: None,
        }
    }
}

// ── Stub business tools: what children are allowed to touch ─────────

struct StubTool(&'static str);

#[async_trait]
impl Tool for StubTool {
    fn name(&self) -> &'static str {
        self.0
    }
    fn description(&self) -> &'static str {
        "stub"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &ToolContext,
    ) -> agent_base::AgentResult<Vec<Content>> {
        Ok(vec![Content::text(format!("{}: ok", self.0))])
    }
}

fn business_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(StubTool("read_file")) as Arc<dyn Tool>,
        Arc::new(StubTool("list_files")) as Arc<dyn Tool>,
    ]
}

// ── Demo ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let runtime = Arc::new(MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        Arc::new(StubLlm),
        business_tools(),
        CancellationToken::new(),
        None,
        Language::En,
        None,
        None,
    ));

    // 1. Fan out: three children, each with its own identity prompt.
    //    `runtime.child()` is the builder façade — spawn failures are Err.
    let mut children = Vec::new();
    for name in ["scout", "analyst", "scribe"] {
        let child = runtime
            .child()
            .system_prompt(format!("You are the {name} sub-agent. Answer briefly."))
            .spawn(name)
            .await
            .expect("spawn ok");
        println!(
            "spawned {} (tools: {:?})",
            child.agent_path(),
            child.spawned_tools()
        );
        children.push(child);
    }

    // 2. Hand each child its work order (delivers a task; the child runs it).
    for (i, child) in children.iter().enumerate() {
        child
            .task(format!("investigate module #{i} and report one line"))
            .expect("task delivered");
    }

    // 3. Collect results — `wait` returns Ok{text} with the child's answer,
    //    Failed, or Timeout (deadline elapsed is not an error).
    for child in &children {
        match child.wait(Duration::from_secs(5)).await {
            ChildOutcome::Ok { text, .. } => {
                println!("{} -> {:?}", child.agent_path(), text.unwrap_or_default())
            }
            other => println!("{} -> {:?}", child.agent_path(), other),
        }
    }

    // 4. A follow-up round trip: send_message is mailbox-only (no trigger);
    //    task delivers new work. (The LLM tool layer maps trigger=true to a
    //    send_task — same primitive as step 2.)
    let scout = &children[0];
    scout.send("also summarize your finding in five words").ok();
    scout.task("summarize").expect("follow-up task delivered");
    if let ChildOutcome::Ok { text, .. } = scout.wait(Duration::from_secs(5)).await {
        println!(
            "follow-up {} -> {:?}",
            scout.agent_path(),
            text.unwrap_or_default()
        );
    }

    // 5. Observe and tear down. `list_agents` is the parent's audit view;
    //    closing returns concurrency slots and unregisters the mailbox.
    for info in runtime.list_agents() {
        println!("live: {} [{}]", info.agent_path, info.status);
    }
    for child in &children {
        child.close().ok();
    }
    // Teardown is async: the child task's `ChildCleanup` releases the registry
    // entry (and the concurrency slot) when its future ends — poll to observe.
    for _ in 0..100 {
        if runtime.list_agents().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    println!("closed; {} still registered", runtime.list_agents().len());

    // Dropping the runtime (end of main) cancels anything still live.
}
