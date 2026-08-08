//! Benchmarks: AgentBuilder construction in agent-works.

use std::pin::Pin;
use std::sync::Arc;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use futures_core::Stream;
use agent_base::{
    AgentResult, ChatMessage, LlmCapabilities, LlmClient, ReasoningConfig, ResponseFormat,
    StreamChunk, Tool, ToolContext, ToolControlFlow, ToolMetadata, ToolOutput,
};
use agent_works::AgentBuilder;

/// Mock LLM client.
struct BenchLlmClient;
#[async_trait::async_trait]
impl LlmClient for BenchLlmClient {
    async fn chat(&self, _: &[ChatMessage], _: &[serde_json::Value], _: Option<&ReasoningConfig>, _: Option<&ResponseFormat>) -> AgentResult<serde_json::Value> {
        Ok(serde_json::json!({}))
    }
    async fn chat_stream(&self, _: &[ChatMessage], _: &[serde_json::Value], _: Option<&ReasoningConfig>, _: Option<&ResponseFormat>) -> AgentResult<Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>>> {
        struct EmptyStream;
        impl Stream for EmptyStream {
            type Item = AgentResult<StreamChunk>;
            fn poll_next(self: Pin<&mut Self>, _: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
                std::task::Poll::Ready(None)
            }
        }
        Ok(Box::pin(EmptyStream))
    }
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities {
            supports_thinking: false, supports_streaming: false,
            supports_tools: true, supports_vision: false,
            max_context_tokens: Some(4096), max_output_tokens: Some(4096),
        }
    }
}

/// No-op tool.
#[derive(Clone)]
struct NoopTool;
#[async_trait::async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &'static str { "noop" }
    fn definition(&self) -> serde_json::Value {
        serde_json::json!({"function": {"name": "noop", "description": "no-op", "parameters": {}}})
    }
    async fn call(&self, _: &serde_json::Value, _: &ToolContext) -> AgentResult<ToolOutput> {
        Ok(ToolOutput {
            summary: "ok".into(), raw: None,
            control_flow: ToolControlFlow::Continue, truncation: None,
        })
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "noop".into(), description: "no-op".into(),
            origin: "bench".into(), version: "0.0.0".into(), requirements: vec![],
        }
    }
}

fn bench_build_empty(c: &mut Criterion) {
    let client: Arc<dyn LlmClient> = Arc::new(BenchLlmClient);

    c.bench_function("agent_works/build_empty", |b| {
        b.iter(|| {
            let builder = AgentBuilder::new(client.clone());
            let agent = builder.build().unwrap();
            black_box(agent);
        });
    });
}

fn bench_build_with_prompt(c: &mut Criterion) {
    let client: Arc<dyn LlmClient> = Arc::new(BenchLlmClient);
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
    let client: Arc<dyn LlmClient> = Arc::new(BenchLlmClient);

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
