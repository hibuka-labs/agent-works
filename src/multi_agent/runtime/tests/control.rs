use super::*;

// ── stage 2: control plane (§7.1–§7.5, §9.2) ──

/// Runtime with an explicit control plane and provider.
fn make_runtime_controlled(
    business_tools: Vec<Arc<dyn Tool>>,
    control: ControlConfig,
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
) -> Arc<MultiAgentRuntime> {
    let config = MultiAgentConfig {
        control,
        ..MultiAgentConfig::enabled()
    };
    Arc::new(MultiAgentRuntime::new(
        config,
        client,
        business_tools,
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ))
}

/// `StreamingStub` plus a usage chunk — exercises hook A metering.
struct UsageStreamingStub;

#[async_trait::async_trait]
impl agent_base::llm_trait::LlmProvider for UsageStreamingStub {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        Ok(agent_base::llm_trait::ChatStream::new(Box::pin(
            futures_util::stream::iter(vec![
                Ok(agent_base::StreamChunk::Text("child ok".to_string())),
                Ok(agent_base::StreamChunk::Usage(agent_base::UsageInfo {
                    prompt_tokens: Some(60),
                    completion_tokens: Some(20),
                    total_tokens: Some(80),
                    reasoning_tokens: None,
                })),
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
        unreachable!("chat is not called by the child loop")
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities::default()
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "usage-stub".to_string(),
            model: "usage-stub".to_string(),
            version: None,
        }
    }
}

struct NoopWriteFileTool;

#[async_trait::async_trait]
impl Tool for NoopWriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "Write a file"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" }, "content": { "type": "string" } }
        })
    }
    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &agent_base::ToolContext,
    ) -> agent_base::AgentResult<Vec<agent_base::Content>> {
        Ok(vec![agent_base::Content::text("wrote")])
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn budget_cumulative_cap_survives_close() {
    // §7.2 acceptance: `max_spawns` counts spawns *cumulatively* — a
    // closed child still spent its spawn. Also proves the budget gate is
    // on the legacy `spawn_child` path (the phimint route).
    let ma = make_runtime_controlled(
        vec![],
        ControlConfig {
            max_spawns: Some(1),
            ..Default::default()
        },
        Arc::new(StreamingStub),
    );
    ma.spawn_child("a", "prompt".to_string(), 0, false, vec![])
        .await
        .expect("first spawn allowed");
    assert_eq!(ma.control().budget().spawn_count(), 1);

    // Full teardown of A — so a second rejection below is provably the
    // budget gate (gate 1), not the live-count registry gate.
    ma.close_agent("root/a").unwrap();
    poll_until("child a cleaned up", || {
        ma.registry.lock().unwrap().count() == 0 && ma.limiter.current() == 0
    })
    .await;

    let err = ma
        .spawn_child("b", "prompt".to_string(), 0, false, vec![])
        .await
        .unwrap_err();
    // Legacy string shape: the budget error surfaces as plain text.
    assert_eq!(err, "max spawn count reached (limit: 1)");
    let s = ma.control().status();
    assert_eq!((s.spawn_count, s.max_spawns, s.live_children), (1, 1, 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn budget_ticket_rolls_back_on_failed_spawn() {
    // §7.2 ticket discipline end-to-end: a spawn that passes gate 1 but
    // fails later (whitelist `ToolNotFound` at build) returns its
    // reservation via `Drop` — the cap stays spendable.
    let ma = make_runtime_controlled(
        vec![Arc::new(NoopReadFileTool)],
        ControlConfig {
            max_spawns: Some(1),
            ..Default::default()
        },
        Arc::new(StreamingStub),
    );
    let err = ma
        .spawn_with_config(
            "ghost".to_string(),
            ChildConfig {
                system_prompt: Some("p".to_string()),
                tool_names: Some(BTreeSet::from(["nope".to_string()])),
                ..Default::default()
            },
        )
        .await
        .map(|_| ())
        .expect_err("unknown tool must fail");
    assert!(matches!(err, AgentError::ToolNotFound { .. }));
    assert_eq!(
        ma.control().budget().spawn_count(),
        0,
        "failed spawn returned its reservation"
    );

    ma.spawn_with_config(
        "w".to_string(),
        ChildConfig {
            system_prompt: Some("p".to_string()),
            tool_names: Some(BTreeSet::from(["read_file".to_string()])),
            ..Default::default()
        },
    )
    .await
    .expect("cap still spendable after a rolled-back failure");
    assert_eq!(ma.control().budget().spawn_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn token_budget_metered_by_turn_end_hook() {
    // Hook A (§5.4 / M-4) integration: child usage arrives only through
    // `on_turn_end` — after two 80-token turns the 100-token budget is
    // spent and the next spawn is rejected on the token dimension.
    let ma = make_runtime_controlled(
        vec![],
        ControlConfig {
            child_max_tokens: Some(100),
            ..Default::default()
        },
        Arc::new(UsageStreamingStub),
    );
    ma.spawn_child("a", "prompt".to_string(), 0, false, vec![])
        .await
        .unwrap();
    for task in ["t1", "t2"] {
        ma.send_task("root/a", task.to_string(), false).unwrap();
        let r = ma.wait_for_result(Some("root/a"), 2000).await;
        assert_eq!(r.status, "ok", "task {task}");
    }
    // The hook fired inside `run_turn`, before the result was posted.
    assert_eq!(ma.control().budget().used_tokens(), 160);

    let err = ma
        .spawn_child("b", "prompt".to_string(), 0, false, vec![])
        .await
        .unwrap_err();
    assert_eq!(err, "child token budget exhausted (160 / 100)");
    // Rejection rolled its spawn increment back — only child "a" spent one.
    assert_eq!(ma.control().budget().spawn_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn task_timeout_reports_error_and_child_survives() {
    // §9.2: timeout is a per-Task wall, not a close — the parent gets an
    // Error result, the child stays live for later tasks.
    let ma = make_runtime_controlled(
        vec![],
        ControlConfig {
            task_timeout: Some(Duration::from_millis(50)),
            ..Default::default()
        },
        Arc::new(HangingLlm),
    );
    ma.spawn_child("slow", "prompt".to_string(), 0, false, vec![])
        .await
        .unwrap();
    ma.send_task("root/slow", "hang".to_string(), false)
        .unwrap();
    let r = ma.wait_for_result(Some("root/slow"), 2000).await;
    assert_eq!(r.status, "error");
    assert!(
        r.result
            .as_deref()
            .is_some_and(|s| s.contains("task timed out")),
        "got {:?}",
        r.result
    );
    assert_eq!(
        ma.registry.lock().unwrap().count(),
        1,
        "timeout must not close the child"
    );

    ma.close_agent("root/slow").unwrap();
    poll_until("closed after timeout", || {
        ma.registry.lock().unwrap().count() == 0
    })
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn manual_mode_hard_excludes_write_tools() {
    // §7.5 layer ①: `write_tools` joins the exclusion set *before* the
    // whitelist, so a child asking for `write_file` gets the reduced set
    // (the §5.4 case-2 warn path), never an error, never the tool.
    let ma = make_runtime_controlled(
        vec![Arc::new(NoopReadFileTool), Arc::new(NoopWriteFileTool)],
        ControlConfig {
            autonomy: AgentAutonomy::Manual,
            ..Default::default()
        },
        Arc::new(StreamingStub),
    );
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("p".to_string()),
                tool_names: Some(BTreeSet::from([
                    "read_file".to_string(),
                    "write_file".to_string(),
                ])),
                ..Default::default()
            },
        )
        .await
        .expect("Manual tightens, does not error");
    assert_eq!(
        spawned.spawned_tools(),
        &BTreeSet::from(["read_file".to_string()])
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_mode_leaves_write_tools_alone() {
    // The `Auto` default is byte-identical to stage 1: `write_tools` is
    // configured but unused, both whitelisted tools register.
    let ma = make_runtime_controlled(
        vec![Arc::new(NoopReadFileTool), Arc::new(NoopWriteFileTool)],
        ControlConfig::default(),
        Arc::new(StreamingStub),
    );
    let spawned = ma
        .spawn_with_config(
            "w".to_string(),
            ChildConfig {
                system_prompt: Some("p".to_string()),
                tool_names: Some(BTreeSet::from([
                    "read_file".to_string(),
                    "write_file".to_string(),
                ])),
                ..Default::default()
            },
        )
        .await
        .expect("Auto keeps the full whitelist");
    assert_eq!(spawned.spawned_tools().len(), 2);
}
