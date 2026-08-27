//! Benchmarks: AgentBuilder construction in agent-works.

use agent_base::llm_trait::backend::LlmBackend;
use agent_base::llm_trait::response::FinishReason;
use agent_base::llm_trait::types::UsageInfo;
use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};
use agent_base::{AgentResult, Content, Tool, ToolContext, ToolMetadata};
use agent_works::AgentBuilder;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;

/// Mock LLM provider.
struct BenchLlmProvider;
#[async_trait::async_trait]
impl LlmProvider for BenchLlmProvider {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Ok(ChatStream::new(Box::pin(futures_util::stream::empty())))
    }
    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            content: String::new(),
            tool_calls: vec![],
            usage: UsageInfo::default(),
            finish_reason: FinishReason::Stop,
            raw: None,
        })
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_thinking: false,
            supports_streaming: false,
            supports_tools: true,
            supports_vision: false,
            max_context_tokens: Some(4096),
            max_output_tokens: Some(4096),
        }
    }
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "bench".to_string(),
            model: "bench-model".to_string(),
            backend: LlmBackend::Custom("bench".to_string()),
            version: None,
        }
    }
}

/// No-op tool.
#[derive(Clone)]
struct NoopTool;
#[async_trait::async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &'static str {
        "noop"
    }
    fn description(&self) -> &'static str {
        "no-op"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn call(&self, _: &serde_json::Value, _: &ToolContext) -> AgentResult<Vec<Content>> {
        Ok(vec![Content::text("ok")])
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "noop".into(),
            description: "no-op".into(),
            origin: "bench".into(),
            version: "0.0.0".into(),
            requirements: vec![],
        }
    }
}

fn bench_build_empty(c: &mut Criterion) {
    let client: Arc<dyn LlmProvider> = Arc::new(BenchLlmProvider);

    c.bench_function("agent_works/build_empty", |b| {
        b.iter(|| {
            let builder = AgentBuilder::new(client.clone());
            let agent = builder.build().unwrap();
            black_box(agent);
        });
    });
}

fn bench_build_with_prompt(c: &mut Criterion) {
    let client: Arc<dyn LlmProvider> = Arc::new(BenchLlmProvider);
    let prompt = "You are a helpful assistant.".repeat(50);

    c.bench_function("agent_works/build_with_prompt", |b| {
        b.iter(|| {
            let builder = AgentBuilder::new(client.clone()).system_prompt(prompt.clone());
            let agent = builder.build().unwrap();
            black_box(agent);
        });
    });
}

fn bench_build_with_tools(c: &mut Criterion) {
    let client: Arc<dyn LlmProvider> = Arc::new(BenchLlmProvider);

    for n in [10, 50, 100] {
        let client = client.clone();
        c.bench_function(&format!("agent_works/build_{}_tools", n), move |b| {
            b.iter(|| {
                let mut builder = AgentBuilder::new(client.clone());
                for _ in 0..n {
                    builder = builder.register_tool(NoopTool);
                }
                let agent = builder.build().unwrap();
                black_box(agent);
            });
        });
    }
}

criterion_group! {
    name = agent_builder_benches;
    config = Criterion::default().sample_size(200);
    targets = bench_build_empty, bench_build_with_prompt, bench_build_with_tools
}
criterion_main!(agent_builder_benches);
