use super::*;

use crate::multi_agent::ChildConfig;

// ── child turn-budget hygiene (2026-11-21 incident, session 20261121_0eae5ba5) ──
//
// Two reviewer children burned 33 turns on exploration, died at the wall with
// zero findings delivered, and the parent only saw "max turns exceeded (33
// turns)". Two contracts prevent a repeat:
//   1. near the cap the child gets a wrap-up nudge (stop exploring, write up);
//   2. a capped child still delivers its salvaged report to the parent.

/// Distinctive substring of the wrap-up nudge the child builder mounts.
const NUDGE: &str = "nearly exhausted your turn budget";

/// Scripted provider that records the message text of every request:
/// two tool-call turns, then a plain final answer.
struct TwoToolsThenText {
    requests: std::sync::Mutex<Vec<String>>,
    turns: std::sync::atomic::AtomicUsize,
}

impl TwoToolsThenText {
    fn new() -> Self {
        Self {
            requests: std::sync::Mutex::new(Vec::new()),
            turns: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn request_texts(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

fn request_text(request: &agent_base::llm_trait::ChatRequest) -> String {
    request
        .messages
        .iter()
        .map(|m| match m {
            agent_base::ChatMessage::System { content, .. } => content.as_str(),
            agent_base::ChatMessage::User { content, .. } => content.as_str(),
            agent_base::ChatMessage::Assistant { content, .. } => content.as_deref().unwrap_or(""),
            agent_base::ChatMessage::Tool { content, .. } => content.as_str(),
            _ => "",
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for TwoToolsThenText {
    async fn stream(
        &self,
        request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        self.requests.lock().unwrap().push(request_text(&request));
        let turn = self.turns.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let chunks: Vec<Result<agent_base::StreamChunk, agent_base::llm_trait::LlmError>> =
            if turn < 2 {
                vec![
                    Ok(agent_base::StreamChunk::ToolCall(serde_json::json!({
                        "delta": { "tool_calls": [
                            { "index": 0, "id": format!("call_{turn}"),
                              "function": { "name": "read_file", "arguments": "{\"path\":\"x\"}" } }
                        ] }
                    }))),
                    Ok(agent_base::StreamChunk::Stop {
                        finish_reason: Some("tool_calls".to_string()),
                    }),
                ]
            } else {
                vec![
                    Ok(agent_base::StreamChunk::Text("stub done".to_string())),
                    Ok(agent_base::StreamChunk::Stop {
                        finish_reason: Some("stop".to_string()),
                    }),
                ]
            };
        Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
            futures_util::stream::iter(chunks),
        )))
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        Ok(agent_base::llm_trait::ChatResponse {
            content: "stub done".to_string(),
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
            name: "two-tools-then-text".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn child_gets_a_wrap_up_nudge_near_the_turn_cap() {
    // max_turns = 4 with a 3-turn script: the early turns must NOT carry the
    // wrap-up nudge (the child is mid-exploration), the closing turn must.
    let stub = std::sync::Arc::new(TwoToolsThenText::new());
    let ma = make_ma_runtime_with_tools(
        stub.clone(),
        vec![std::sync::Arc::new(NoopReadFileTool) as std::sync::Arc<dyn Tool>],
    );
    let config = ChildConfig {
        system_prompt: Some("child".into()),
        max_turns: Some(4),
        ..Default::default()
    };
    let (child, _tools, _res) = ma
        .build_child_runtime_with_config(&config, true, None, "root/test")
        .await
        .expect("build child runtime");

    let sid = child.create_session().await;
    let (_events, outcome) = child
        .run_turn_collect(sid, "do the thing")
        .await
        .expect("run child turn");
    assert!(
        matches!(outcome, RunOutcome::Completed),
        "scripted run should complete: {outcome:?}"
    );

    let texts = stub.request_texts();
    assert_eq!(
        texts.len(),
        3,
        "expected two tool turns then the final answer, got: {texts:?}"
    );
    assert!(
        !texts[0].contains(NUDGE),
        "turn 1 (mid-exploration) must not carry the wrap-up nudge"
    );
    assert!(
        !texts[1].contains(NUDGE),
        "turn 2 (mid-exploration) must not carry the wrap-up nudge"
    );
    assert!(
        texts[2].contains(NUDGE),
        "the closing turn must carry the wrap-up nudge"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn capped_child_still_delivers_its_findings() {
    // The incident's exact shape: a child whose first turn is a tool call
    // trips the cap. The grace turn's text must reach the parent's mailbox
    // beside the honest "max turns exceeded" label.
    let stub = std::sync::Arc::new(ToolCallOnceStub::new("read_file", "{\"path\":\"x\"}"));
    let ma = make_ma_runtime_with_tools(
        stub,
        vec![std::sync::Arc::new(NoopReadFileTool) as std::sync::Arc<dyn Tool>],
    );
    let _spawned = ma
        .spawn_with_config(
            "worker".to_string(),
            ChildConfig {
                system_prompt: Some("reviewer".into()),
                max_turns: Some(1),
                ..Default::default()
            },
        )
        .await
        .expect("spawn child");
    ma.send_task("root/worker", "do the review".to_string(), false)
        .expect("send task");

    let result = ma.wait_for_result(Some("root/worker"), 5000).await;
    assert_eq!(result.status, "ok");
    let text = result.result.expect("child result text");
    assert!(
        text.contains("max turns exceeded"),
        "the honest cap label stays, got: {text}"
    );
    assert!(
        text.contains("stub done"),
        "the salvaged report must ride along, got: {text}"
    );
}
