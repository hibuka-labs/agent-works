use std::sync::Arc;
use std::sync::Mutex;

use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};
use agent_base::{AgentResult, Content, StreamChunk, Tool, ToolContext};
use agent_works::{AgentBuilder, skill::Skill};
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

struct AddTool;

#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &'static str {
        "add"
    }

    fn description(&self) -> &'static str {
        "Calculate the sum of two integers"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First addend" },
                "b": { "type": "integer", "description": "Second addend" }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        Ok(vec![Content::text(format!("{a} + {b} = {}", a + b))])
    }
}

struct SubtractTool;

#[async_trait]
impl Tool for SubtractTool {
    fn name(&self) -> &'static str {
        "subtract"
    }

    fn description(&self) -> &'static str {
        "Calculate the difference of two integers (a - b)"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "Minuend" },
                "b": { "type": "integer", "description": "Subtrahend" }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        Ok(vec![Content::text(format!("{a} - {b} = {}", a - b))])
    }
}

struct MathSkill;

impl Skill for MathSkill {
    fn name(&self) -> &'static str {
        "math"
    }

    fn brief_description(&self) -> String {
        "Math: supports addition and subtraction".to_string()
    }

    fn detailed_description(&self) -> String {
        "- **add**: Calculate the sum of two integers\n\
         - **subtract**: Calculate the difference of two integers"
            .to_string()
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![Arc::new(AddTool), Arc::new(SubtractTool)]
    }
}

#[tokio::main]
async fn main() -> AgentResult<()> {
    println!("=== agent-works Skill Demo ===\n");

    let llm: Arc<dyn LlmProvider> = Arc::new(MockLlmProvider::new(vec![
        vec![
            StreamChunk::ToolCall(json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "get_skill_detail",
                            "arguments": "{\"name\": \"math\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::ToolCall(json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_2",
                        "function": {
                            "name": "add",
                            "arguments": "{\"a\": 123, \"b\": 456}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("123 + 456 = 579".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ]));

    let _runtime = AgentBuilder::new(llm)
        .system_prompt("You are a helpful assistant. Use skills when needed.")
        .register_skill(MathSkill)
        .build()
        .unwrap();

    println!("[1] Registered skill with 'register_skill()' on agent-works AgentBuilder");
    println!("    - Skill tools (add, subtract) auto-registered");
    println!("    - LazySkillPrompter injected into system prompt");

    // Note: In a real app, you would run the agent here:
    // let session_id = runtime.create_session().await;
    // let result = runtime.run_turn_collect(session_id, "What is 2 + 3?").await;
    println!("\nDemo complete.");
    Ok(())
}
