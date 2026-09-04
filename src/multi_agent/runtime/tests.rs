use super::*;
use agent_base::RunOutcome;

use crate::multi_agent::config::ControlConfig;

// ---------------------------------------------------------------------------
// Shared fixtures (used by more than one scenario module)
// ---------------------------------------------------------------------------

struct StreamingStub;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for StreamingStub {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
            futures_util::stream::iter(vec![
                Ok(agent_base::StreamChunk::Text("child ok".to_string())),
                Ok(agent_base::StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ]),
        )))
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        Ok(agent_base::llm_trait::ChatResponse {
            content: "child ok".to_string(),
            tool_calls: vec![],
            usage: agent_base::llm_trait::types::UsageInfo::default(),
            finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

fn make_ma_runtime_with(client: Arc<dyn agent_base::llm_trait::LlmProvider>) -> Arc<MultiAgentRuntime> {
    Arc::new(MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        client,
        vec![],
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ))
}

fn make_ma_runtime() -> Arc<MultiAgentRuntime> {
    make_ma_runtime_with(Arc::new(StreamingStub))
}

struct NoopReadFileTool;

#[async_trait::async_trait]
impl Tool for NoopReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read a file's contents"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        })
    }

    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &agent_base::ToolContext,
    ) -> agent_base::AgentResult<Vec<agent_base::Content>> {
        Ok(vec![agent_base::Content::text("contents")])
    }
}

/// Poll a teardown condition (cleanup runs on a worker thread, so the
/// assertions after `close_agent` / panic / `drop` are eventually-true).
async fn poll_until(what: &str, done: impl Fn() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("teardown did not complete within 2s: {what}");
}

/// Provider whose `stream` never returns — keeps the child inside
/// `run_turn` so parent Drop can only end it via JoinSet abort.
struct HangingLlm;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for HangingLlm {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        unreachable!("the child task is aborted long before this returns")
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        unreachable!("chat is not called by the child loop")
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "hanging".to_string(),
            model: "hanging".to_string(),
            version: None,
        }
    }
}

/// Provider whose stream completes after a short delay — keeps the child
/// inside `run_turn` long enough for a mid-task `close_agent`, then lets the
/// task finish (close is only honored between tasks, so the late result must
/// still be delivered).
struct DelayedStub;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for DelayedStub {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
            futures_util::stream::iter(vec![
                Ok(agent_base::StreamChunk::Text("late ok".to_string())),
                Ok(agent_base::StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ]),
        )))
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        Ok(agent_base::llm_trait::ChatResponse {
            content: "late ok".to_string(),
            tool_calls: vec![],
            usage: agent_base::llm_trait::types::UsageInfo::default(),
            finish_reason: agent_base::llm_trait::response::FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "delayed".to_string(),
            model: "delayed".to_string(),
            version: None,
        }
    }
}

// scenario modules (split from the former inline `mod tests`)
mod autonomy;
mod cleanup;
mod control;
mod denied;
mod fork;
mod lifecycle;
mod outcome;
mod permission;
mod whitelist;
