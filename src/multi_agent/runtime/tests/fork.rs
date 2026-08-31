use super::*;

// ── fork_history: resolve_fork_history ──

/// Mock LLM provider for fork_history tests (minimal — never called).
#[derive(Clone)]
struct NoopLlmProvider;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for NoopLlmProvider {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        unimplemented!()
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        unimplemented!()
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities {
            supports_streaming: true,
            ..Default::default()
        }
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "noop".to_string(),
            model: "noop-model".to_string(),
            version: None,
        }
    }
}

/// Build a MultiAgentRuntime with a parent runtime that has a populated session.
async fn setup_fork_history_test(
    parent_messages: Vec<agent_base::ChatMessage>,
) -> (Arc<MultiAgentRuntime>, agent_base::SessionId) {
    use tokio_util::sync::CancellationToken;

    let llm = Arc::new(NoopLlmProvider);
    let parent_runtime = agent_base::AgentBuilder::new(llm)
        .build()
        .expect("build parent runtime");
    let parent_sid = parent_runtime.create_session().await;

    // Push messages directly into the session's chat_messages vector so
    // we can use proper Assistant/Tool variants (not just System).
    parent_runtime
        .with_session_mut(&parent_sid, |session| {
            session.chat_messages_mut().extend(parent_messages.clone());
        })
        .await
        .unwrap();

    let session_manager = Arc::new(parent_runtime.session_manager().clone());

    let ma_runtime = Arc::new(MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        Arc::new(NoopLlmProvider),
        vec![],
        CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ));
    ma_runtime.set_session_manager(session_manager);

    (ma_runtime, parent_sid)
}

#[tokio::test]
async fn resolve_fork_history_none_returns_empty() {
    let messages = vec![agent_base::ChatMessage::User {
        content: "hello".into(),
        images: vec![],
        ephemeral: false,
    }];
    let (ma, parent_sid) = setup_fork_history_test(messages).await;

    // None
    let result = ma.resolve_fork_history(None, &parent_sid).await;
    assert!(result.is_empty());

    // Some("none")
    let result = ma
        .resolve_fork_history(Some("none".to_string()), &parent_sid)
        .await;
    assert!(result.is_empty());
}

#[tokio::test]
async fn resolve_fork_history_all_returns_all_non_system() {
    let messages = vec![
        agent_base::ChatMessage::User {
            content: "question 1".into(),
            images: vec![],
            ephemeral: false,
        },
        agent_base::ChatMessage::Assistant {
            content: Some("answer 1".into()),
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: None,
        },
        agent_base::ChatMessage::User {
            content: "question 2".into(),
            images: vec![],
            ephemeral: false,
        },
        agent_base::ChatMessage::Assistant {
            content: Some("answer 2".into()),
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: None,
        },
    ];
    let (ma, parent_sid) = setup_fork_history_test(messages).await;

    let result = ma
        .resolve_fork_history(Some("all".to_string()), &parent_sid)
        .await;

    // Should have 4 messages (2 user + 2 assistant) — system messages are filtered out
    assert_eq!(result.len(), 4);
    assert!(matches!(result[0], agent_base::ChatMessage::User { .. }));
    assert!(matches!(
        result[1],
        agent_base::ChatMessage::Assistant { .. }
    ));
    assert!(matches!(result[2], agent_base::ChatMessage::User { .. }));
    assert!(matches!(
        result[3],
        agent_base::ChatMessage::Assistant { .. }
    ));
}

#[tokio::test]
async fn resolve_fork_history_n_turns() {
    // 3 turns: 3 user messages, 3 assistant responses
    let messages = vec![
        agent_base::ChatMessage::User {
            content: "q1".into(),
            images: vec![],
            ephemeral: false,
        },
        agent_base::ChatMessage::Assistant {
            content: Some("a1".into()),
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: None,
        },
        agent_base::ChatMessage::User {
            content: "q2".into(),
            images: vec![],
            ephemeral: false,
        },
        agent_base::ChatMessage::Assistant {
            content: Some("a2".into()),
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: None,
        },
        agent_base::ChatMessage::User {
            content: "q3".into(),
            images: vec![],
            ephemeral: false,
        },
        agent_base::ChatMessage::Assistant {
            content: Some("a3".into()),
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: None,
        },
    ];
    let (ma, parent_sid) = setup_fork_history_test(messages).await;

    // Last 1 turn
    let result = ma
        .resolve_fork_history(Some("1".to_string()), &parent_sid)
        .await;
    assert_eq!(result.len(), 2, "1 turn = user q3 + assistant a3");
    assert!(matches!(result[0], agent_base::ChatMessage::User { .. }));
    assert_eq!(extract_user_content(&result[0]), "q3");

    // Last 2 turns
    let result = ma
        .resolve_fork_history(Some("2".to_string()), &parent_sid)
        .await;
    assert_eq!(result.len(), 4, "2 turns = q2,a2,q3,a3");
}

#[tokio::test]
async fn resolve_fork_history_invalid_number_treats_as_none() {
    let messages = vec![agent_base::ChatMessage::User {
        content: "hello".into(),
        images: vec![],
        ephemeral: false,
    }];
    let (ma, parent_sid) = setup_fork_history_test(messages).await;

    // Invalid number → empty
    let result = ma
        .resolve_fork_history(Some("not-a-number".to_string()), &parent_sid)
        .await;
    assert!(result.is_empty());

    // Zero → empty
    let result = ma
        .resolve_fork_history(Some("0".to_string()), &parent_sid)
        .await;
    assert!(result.is_empty());
}

#[tokio::test]
async fn resolve_fork_history_no_session_manager_returns_empty() {
    use tokio_util::sync::CancellationToken;

    let ma_runtime = MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        Arc::new(NoopLlmProvider),
        vec![],
        CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    );
    // session_manager is NOT set

    let sid = agent_base::SessionId::new(9999);
    let result = ma_runtime
        .resolve_fork_history(Some("all".to_string()), &sid)
        .await;
    assert!(result.is_empty());
}

#[tokio::test]
async fn resolve_fork_history_empty_session_returns_empty() {
    let (ma, parent_sid) = setup_fork_history_test(vec![]).await;

    let result = ma
        .resolve_fork_history(Some("all".to_string()), &parent_sid)
        .await;
    assert!(result.is_empty());
}

// ── fork_history: prefill_child_session ──

#[tokio::test]
async fn prefill_child_session_user_and_assistant() {
    let llm = Arc::new(NoopLlmProvider);
    let child_runtime = agent_base::AgentBuilder::new(llm)
        .build()
        .expect("build child runtime");
    let child_sid = child_runtime.create_session().await;

    let parent_messages = vec![
        agent_base::ChatMessage::User {
            content: "user question".into(),
            images: vec![],
            ephemeral: false,
        },
        agent_base::ChatMessage::Assistant {
            content: Some("assistant reply".into()),
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: None,
        },
        agent_base::ChatMessage::Tool {
            tool_call_id: "call_123".into(),
            name: None,
            content: "tool output".into(),
        },
    ];

    // Create a minimal MultiAgentRuntime just to call prefill_child_session
    use tokio_util::sync::CancellationToken;
    let ma_runtime = MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        Arc::new(NoopLlmProvider),
        vec![],
        CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    );

    ma_runtime
        .prefill_child_session(&child_runtime, &child_sid, &parent_messages)
        .await
        .expect("prefill should succeed");

    // Verify the child session contains the pre-filled messages
    let session = child_runtime
        .session(&child_sid)
        .await
        .expect("session exists");
    let msgs = session.chat_messages().to_vec();

    // Should have: user msg + system msg (assistant) + system msg (tool)
    assert_eq!(msgs.len(), 3);
    assert!(matches!(msgs[0], agent_base::ChatMessage::User { .. }));
    assert!(matches!(msgs[1], agent_base::ChatMessage::System { .. }));
    assert!(matches!(msgs[2], agent_base::ChatMessage::System { .. }));
}

#[tokio::test]
async fn prefill_child_session_tool_call_only_skipped() {
    let llm = Arc::new(NoopLlmProvider);
    let child_runtime = agent_base::AgentBuilder::new(llm)
        .build()
        .expect("build child runtime");
    let child_sid = child_runtime.create_session().await;

    // Assistant message with only tool_calls (no text content) should be skipped
    let parent_messages = vec![
        agent_base::ChatMessage::User {
            content: "do something".into(),
            images: vec![],
            ephemeral: false,
        },
        agent_base::ChatMessage::Assistant {
            content: None, // no text — tool call only
            reasoning_content: None,
            thinking_signature: None,
            tool_calls: Some(vec![]),
        },
    ];

    use tokio_util::sync::CancellationToken;
    let ma_runtime = MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        Arc::new(NoopLlmProvider),
        vec![],
        CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    );

    ma_runtime
        .prefill_child_session(&child_runtime, &child_sid, &parent_messages)
        .await
        .expect("prefill should succeed");

    let session = child_runtime
        .session(&child_sid)
        .await
        .expect("session exists");
    let msgs = session.chat_messages().to_vec();

    // Only the user message — tool-call-only assistant should be skipped
    assert_eq!(msgs.len(), 1);
    assert!(matches!(msgs[0], agent_base::ChatMessage::User { .. }));
}

#[tokio::test]
async fn prefill_child_session_empty_vec_noop() {
    let llm = Arc::new(NoopLlmProvider);
    let child_runtime = agent_base::AgentBuilder::new(llm)
        .build()
        .expect("build child runtime");
    let child_sid = child_runtime.create_session().await;

    use tokio_util::sync::CancellationToken;
    let ma_runtime = MultiAgentRuntime::new(
        MultiAgentConfig::enabled(),
        Arc::new(NoopLlmProvider),
        vec![],
        CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    );

    ma_runtime
        .prefill_child_session(&child_runtime, &child_sid, &[])
        .await
        .expect("prefill should succeed");

    let session = child_runtime
        .session(&child_sid)
        .await
        .expect("session exists");
    let msgs = session.chat_messages().to_vec();

    // System prompt is added but we don't assert exact count — just that no user/injected msgs
    assert!(msgs.is_empty() || matches!(msgs[0], agent_base::ChatMessage::System { .. }));
}

fn extract_user_content(msg: &agent_base::ChatMessage) -> &str {
    match msg {
        agent_base::ChatMessage::User { content, .. } => content.as_str(),
        _ => "",
    }
}
