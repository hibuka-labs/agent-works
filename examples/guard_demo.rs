//! Demonstrates the Guard system — DefaultGuard, NoopGuard, and custom guards.
//!
//! Run with:
//!   cargo run --example guard_demo

use std::sync::Arc;
use std::sync::Mutex;

use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};
use agent_base::{AgentResult, Content, StreamChunk, Tool, ToolContext};
use agent_works::AgentBuilder;
use agent_works::guard::{
    DefaultGuard, DefaultGuardConfig, GuardCtx, GuardDecision, ReactLoopGuard,
};
use async_trait::async_trait;
use serde_json::{Value, json};

// ── Mock LLM ────────────────────────────────────────────────────────

struct MockLlmProvider {
    responses: Mutex<std::vec::IntoIter<Vec<StreamChunk>>>,
}

impl MockLlmProvider {
    fn new(responses: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter()),
        }
    }
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chunks: Vec<Result<StreamChunk, LlmError>> = self
            .responses
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
        unimplemented!()
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_streaming: true,
            supports_tools: true,
            ..Default::default()
        }
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "mock".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

// ── Example tool ────────────────────────────────────────────────────

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn description(&self) -> &'static str {
        "Echo back the message"
    }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "message": { "type": "string" } },
            "required": ["message"]
        })
    }
    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let msg = args["message"].as_str().unwrap_or("");
        Ok(vec![Content::text(format!("echo: {msg}"))])
    }
}

// ── Custom guard example ────────────────────────────────────────────

/// A strict guard that fails if no tool calls were made in the run.
struct StrictGuard;

#[async_trait]
impl ReactLoopGuard for StrictGuard {
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision {
        if ctx.is_reasoning_only || ctx.is_empty_response {
            return GuardDecision::Fail {
                error: "no tool calls or text produced".into(),
            };
        }
        if ctx.is_text_only && !ctx.run_has_tool_calls {
            return GuardDecision::Fail {
                error: "model responded with text only — no tool calls were made".into(),
            };
        }
        GuardDecision::Complete
    }
}

// ── Main ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> AgentResult<()> {
    println!("=== agent-works Guard Demo ===\n");

    let llm: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new(vec![]));

    // ── 1. No guard (default NoopGuard) ──
    println!("[1] Default — no guard configured");
    let _runtime = AgentBuilder::new(llm.clone())
        .system_prompt("You are a helpful assistant.")
        .register_tool(EchoTool)
        .build()
        .unwrap();
    println!("    → NoopGuard injected automatically (no intervention)\n");

    // ── 2. DefaultGuard with default config ──
    println!("[2] DefaultGuard with default config");
    let guard = DefaultGuard::new(DefaultGuardConfig::default());
    let _runtime = AgentBuilder::new(llm.clone())
        .system_prompt("You are a helpful assistant.")
        .register_tool(EchoTool)
        .guard(guard)
        .build()
        .unwrap();
    println!("    → DefaultGuard handles reasoning_only, empty_response, text_only\n");

    // ── 3. DefaultGuard with LLM judge ──
    println!("[3] DefaultGuard with LLM judge for completion verification");
    let config = DefaultGuardConfig {
        use_llm_judge: true,
        judge_fail_open: true, // trust model if judge fails
        ..Default::default()
    };
    let guard = DefaultGuard::with_llm_client(config, llm.clone());
    let _runtime = AgentBuilder::new(llm.clone())
        .system_prompt("You are a helpful assistant.")
        .register_tool(EchoTool)
        .guard(guard)
        .build()
        .unwrap();
    println!("    → LLM judge verifies task completion on text-only responses\n");

    // ── 4. Custom guard ──
    println!("[4] Custom guard — StrictGuard");
    let _runtime = AgentBuilder::new(llm.clone())
        .system_prompt("You are a helpful assistant.")
        .register_tool(EchoTool)
        .guard(StrictGuard)
        .build()
        .unwrap();
    println!("    → StrictGuard fails if no tool calls were made\n");

    // ── 5. Direct guard usage (without runtime) ──
    println!("[5] Direct guard usage — calling on_turn() manually");
    let guard = DefaultGuard::new(DefaultGuardConfig::default());

    // Normal response — guard returns Complete
    let ctx = make_test_ctx(false, false, false, 0, 0, true);
    let decision = guard.on_turn(&ctx).await;
    println!("    Normal response → {:?}", decision);

    // Reasoning only — guard returns Continue with nudge
    let ctx = make_test_ctx(true, false, false, 0, 0, false);
    let decision = guard.on_turn(&ctx).await;
    println!("    Reasoning only  → {:?}", decision);

    // Reasoning only, max strikes — guard returns Fail
    let ctx = make_test_ctx(true, false, false, 3, 0, false);
    let decision = guard.on_turn(&ctx).await;
    println!("    Reasoning (max) → {:?}", decision);

    // Empty response — guard returns Continue with nudge
    let ctx = make_test_ctx(false, true, false, 0, 0, false);
    let decision = guard.on_turn(&ctx).await;
    println!("    Empty response  → {:?}", decision);

    println!("\nDemo complete.");
    Ok(())
}

/// Helper to build a GuardCtx for testing.
fn make_test_ctx(
    is_reasoning_only: bool,
    is_empty_response: bool,
    is_text_only: bool,
    reasoning_strikes: usize,
    empty_strikes: usize,
    run_has_tool_calls: bool,
) -> GuardCtx {
    GuardCtx {
        session_id: agent_base::SessionId::new(1),
        turn_count: 1,
        user_input: "test input".into(),
        model_response: "test response".into(),
        finish_reason: agent_base::FinishReason::Stop,
        available_tools: vec!["echo".into()],
        reasoning_only_strikes: reasoning_strikes,
        empty_response_strikes: empty_strikes,
        run_has_tool_calls,
        all_user_inputs: vec!["test input".into()],
        is_reasoning_only,
        is_empty_response,
        is_text_only,
        thinking_disabled: false,
    }
}
