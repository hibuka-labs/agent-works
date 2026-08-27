use std::sync::Arc;
use std::sync::Mutex;

use agent_base::llm_trait::response::FinishReason;
use agent_base::llm_trait::types::UsageInfo;
use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};
use agent_base::{
    AgentResult, ChatMessage, Content, RuntimeEvent, SessionId, StreamChunk, Tool, ToolContext,
};
use agent_works::{
    AgentBuilder,
    cli::{CliEventPrinter, CliRepl},
};
use async_trait::async_trait;
use serde_json::{Value, json};

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
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let msg = args["message"].as_str().unwrap_or("");
        Ok(vec![Content::text(format!("echo: {msg}"))])
    }
}

#[tokio::main]
async fn main() -> AgentResult<()> {
    println!("=== agent-works CLI Demo ===\n");

    let llm: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new(vec![]));

    let runtime = AgentBuilder::new(llm)
        .system_prompt("You are a helpful assistant.")
        .register_tool(EchoTool)
        .build()
        .unwrap();

    println!("[1] AgentRuntime created with AgentBuilder");
    println!("    - MockLlmProvider (for demo purposes)");
    println!("    - EchoTool registered");
    println!();

    let mut repl = CliRepl::new(runtime);
    println!("[2] CliRepl created from AgentRuntime");

    // Note: repl.run() is not called here since this is just a demo
    // showing how to construct the components.
    println!(
        "\nDemo complete. In a real app, call repl.run().await to start the interactive loop."
    );
    Ok(())
}
