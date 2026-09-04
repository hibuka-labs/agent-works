use super::*;

// ── end-to-end: denied_tools flows child → parent ──

/// Scripted client: the first turn requests the `read_file` tool (which the
/// child is denied); any later turn emits a plain text answer.
struct DenialScriptedClient {
    turn: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for DenialScriptedClient {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        let n = self.turn.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let chunks: Vec<Result<agent_base::StreamChunk, agent_base::llm_trait::LlmError>> =
            if n == 0 {
                vec![
                    Ok(agent_base::StreamChunk::ToolCall(serde_json::json!({
                        "delta": {
                            "tool_calls": [{
                                "id": "call_1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\":\"/etc/passwd\"}"
                                }
                            }]
                        }
                    }))),
                    Ok(agent_base::StreamChunk::Stop {
                        finish_reason: Some("tool_calls".to_string()),
                    }),
                ]
            } else {
                vec![
                    Ok(agent_base::StreamChunk::Text(
                        "I lack permission.".to_string(),
                    )),
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
            content: "I lack permission.".to_string(),
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

#[tokio::test(flavor = "multi_thread")]
async fn test_child_denied_tool_reaches_parent_via_wait() {
    // `None` mode + no parent policy → child carries DenyAllToolPolicy, so
    // every tool it attempts is denied. Assert the denied tool name flows
    // end-to-end: collect_denied_tools → mailbox → wait_for_result.
    let config = MultiAgentConfig {
        child_permission_mode: ChildPermissionMode::None,
        ..MultiAgentConfig::enabled()
    };
    let ma = Arc::new(MultiAgentRuntime::new(
        config,
        Arc::new(DenialScriptedClient {
            turn: std::sync::atomic::AtomicUsize::new(0),
        }),
        vec![Arc::new(NoopReadFileTool) as Arc<dyn Tool>],
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ));

    let path = ma
        .spawn_child(
            "worker",
            "child system prompt".to_string(),
            0,
            false,
            vec![],
        )
        .await
        .expect("spawn child");
    assert_eq!(path, "root/worker");

    ma.send_task("root/worker", "read the file".to_string(), false)
        .unwrap();

    let result = ma.wait_for_result(Some("root/worker"), 3000).await;
    assert_eq!(result.status, "ok");
    assert_eq!(result.denied_tools, vec!["read_file".to_string()]);
}
