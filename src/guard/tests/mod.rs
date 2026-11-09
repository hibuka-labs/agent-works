use super::NoopGuard;
use super::config::{DefaultGuardConfig, ReasoningOnlyAction};
use super::default::DefaultGuard;
use agent_base::engine::react_loop_guard::{GuardCtx, GuardDecision, ReactLoopGuard};
use agent_base::llm_trait::response::{
    ChatResponse, ChatStream, FinishReason as LlmFinishReason, StreamChunk,
};
use agent_base::llm_trait::{Capabilities, ChatRequest, LlmError, LlmProvider};
use agent_base::types::{FinishReason, SessionId};
use serde_json::Value;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

fn make_ctx(
    reasoning_only_strikes: usize,
    empty_response_strikes: usize,
    run_has_actions: bool,
    is_reasoning_only: bool,
    is_empty_response: bool,
    is_text_only: bool,
) -> GuardCtx {
    GuardCtx {
        session_id: SessionId {
            id: 1,
            external_id: None,
        },
        turn_count: 1,
        user_input: "test".to_string(),
        model_response: "response".to_string(),
        finish_reason: FinishReason::Stop,
        available_tools: vec![],
        reasoning_only_strikes,
        empty_response_strikes,
        run_has_actions,
        last_tool_calls_invalid: false,
        all_user_inputs: vec!["test".to_string()],
        is_reasoning_only,
        is_empty_response,
        is_text_only,
        awaiting_human: false,
        thinking_disabled: false,
        original_thinking_enabled: true,
        remaining_turns: 50,
    }
}

/// Build a GuardCtx with custom input/response text (for length-sensitive tests).
fn make_ctx_with_text(
    user_input: &str,
    model_response: &str,
    run_has_actions: bool,
    is_text_only: bool,
) -> GuardCtx {
    GuardCtx {
        session_id: SessionId {
            id: 1,
            external_id: None,
        },
        turn_count: 2,
        user_input: user_input.to_string(),
        model_response: model_response.to_string(),
        finish_reason: FinishReason::Stop,
        available_tools: vec!["echo".to_string()],
        reasoning_only_strikes: 0,
        empty_response_strikes: 0,
        run_has_actions,
        last_tool_calls_invalid: false,
        all_user_inputs: vec![user_input.to_string()],
        is_reasoning_only: false,
        is_empty_response: false,
        is_text_only,
        awaiting_human: false,
        thinking_disabled: false,
        original_thinking_enabled: true,
        remaining_turns: 50,
    }
}

// ── Mock LLM clients for judge testing ──

/// Mock StreamClient that returns a fixed JSON response (for judge calls).
struct MockJudgeClient {
    response: String,
}

impl MockJudgeClient {
    fn new(response: Value) -> Self {
        Self {
            response: response.to_string(),
        }
    }

    /// Fixed verbatim response — for fence/prose-wrapped JSON tests (issue #31).
    fn raw(response: &str) -> Self {
        Self {
            response: response.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for MockJudgeClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chunks = vec![
            Ok(StreamChunk::Text(self.response.clone())),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ];
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
            chunks,
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            content: self.response.clone(),
            tool_calls: vec![],
            usage: agent_base::UsageInfo::default(),
            finish_reason: LlmFinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "mock-judge".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

/// Mock StreamClient that never resolves (for timeout testing).
///
/// `calls` counts `chat()` invocations — issue #31 asserts transport failures
/// are NOT retried.
struct MockTimeoutClient {
    calls: std::sync::atomic::AtomicUsize,
}

impl MockTimeoutClient {
    fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl LlmProvider for MockTimeoutClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        // Return a stream whose first item never resolves.
        struct HangingStream;
        impl futures_core::Stream for HangingStream {
            type Item = Result<StreamChunk, LlmError>;
            fn poll_next(
                self: Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Self::Item>> {
                std::task::Poll::Pending // never wakes
            }
        }
        Ok(ChatStream::new(Box::pin(HangingStream)))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Never resolve — will be interrupted by timeout.
        std::future::pending().await
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "mock-timeout".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

/// Mock client that serves queued responses in order and counts chat() calls.
///
/// Issue #31: asserts the judge retries parse failures exactly once.
struct SequenceJudgeClient {
    responses: std::sync::Mutex<VecDeque<String>>,
    calls: std::sync::atomic::AtomicUsize,
}

impl SequenceJudgeClient {
    fn new(responses: &[&str]) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses.iter().map(|s| s.to_string()).collect()),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl LlmProvider for SequenceJudgeClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let content = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_default();
        let chunks = vec![
            Ok(StreamChunk::Text(content)),
            Ok(StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ];
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
            chunks,
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let content = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_default();
        Ok(ChatResponse {
            content,
            tool_calls: vec![],
            usage: agent_base::UsageInfo::default(),
            finish_reason: LlmFinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "mock-sequence".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

/// Mock StreamClient that always returns an error (for judge error testing).
struct MockErrorClient;

#[async_trait::async_trait]
impl LlmProvider for MockErrorClient {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Err(LlmError::llm("simulated stream error"))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Err(LlmError::llm("simulated judge error"))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "mock-error".to_string(),
            model: "mock-model".to_string(),
            version: None,
        }
    }
}

mod boundaries;
mod decisions;
mod judge;
mod judge_function_tests;
mod judge_parse;
mod notice_tests;
mod scenes;
mod short_response;
