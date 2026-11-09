use agent_base::{AgentResult, AgentRuntime};

use super::*;
use agent_base::Content;
use agent_base::llm_trait::response::FinishReason;
use agent_base::llm_trait::types::UsageInfo;
use agent_base::llm_trait::{
    Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider, ProviderInfo,
};

// ── Stub LLM provider ──

struct StubProvider;

#[async_trait::async_trait]
impl LlmProvider for StubProvider {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let chunks = vec![
            Ok(agent_base::StreamChunk::Text("ok".to_string())),
            Ok(agent_base::StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }),
        ];
        Ok(ChatStream::new(Box::pin(futures_util::stream::iter(
            chunks,
        ))))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
        Ok(ChatResponse {
            content: "ok".to_string(),
            tool_calls: vec![],
            usage: UsageInfo::default(),
            finish_reason: FinishReason::Stop,
            raw: None,
            reasoning_content: None,
            thinking_signature: None,
        })
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
            name: "stub".to_string(),
            model: "stub-model".to_string(),
            version: None,
        }
    }
}

fn make_client() -> Arc<dyn LlmProvider> {
    Arc::new(StubProvider)
}

// ── setup_multi_agent tests ──

#[cfg(feature = "multi_agent")]
#[tokio::test(flavor = "multi_thread")]
async fn test_setup_multi_agent_without_factory_registers_no_tools() {
    let client = make_client();
    let runtime = agent_base::AgentBuilder::new(client.clone())
        .build()
        .unwrap();
    let config = MultiAgentConfig::enabled();

    let result = setup_multi_agent(
        &runtime,
        config,
        agent_base::Language::En,
        vec![],
        None,
        &HashSet::new(),
        None, // no factory
    );
    assert!(result.is_ok());
    let ma_runtime = result.unwrap();
    // Verify no tools were registered (the 5 multi-agent tools are absent)
    let agents = ma_runtime.list_agents();
    assert!(agents.is_empty());
}

#[cfg(feature = "multi_agent")]
#[tokio::test(flavor = "multi_thread")]
async fn test_setup_multi_agent_with_factory_registers_tools() {
    let client = make_client();
    let runtime = agent_base::AgentBuilder::new(client.clone())
        .build()
        .unwrap();
    let config = MultiAgentConfig::enabled();

    let factory: MultiAgentToolFactory = Arc::new(|_rt| {
        // Minimal factory returning a single fake tool
        struct FakeTool;
        #[async_trait::async_trait]
        impl Tool for FakeTool {
            fn name(&self) -> &'static str {
                "fake_tool"
            }
            fn description(&self) -> &'static str {
                ""
            }
            fn schema(&self) -> serde_json::Value {
                serde_json::json!({})
            }
            async fn call(
                &self,
                _args: &serde_json::Value,
                _ctx: &agent_base::ToolContext,
            ) -> AgentResult<Vec<Content>> {
                Ok(vec![Content::text("ok")])
            }
        }
        vec![Arc::new(FakeTool)]
    });

    let result = setup_multi_agent(
        &runtime,
        config,
        agent_base::Language::En,
        vec![],
        None,
        &HashSet::new(),
        Some(factory),
    );
    assert!(result.is_ok());

    // Check the tool was registered on the runtime
    let tools: Vec<String> = tokio::task::block_in_place(|| {
        let tools = runtime.tools_mut();
        let guard = tools.blocking_read();
        guard.metadatas().into_iter().map(|m| m.name).collect()
    });
    assert!(tools.contains(&"fake_tool".to_string()));
}

#[cfg(feature = "multi_agent")]
#[tokio::test(flavor = "multi_thread")]
async fn test_setup_multi_agent_skips_duplicate_tool_names() {
    let client = make_client();
    let runtime = agent_base::AgentBuilder::new(client.clone())
        .build()
        .unwrap();

    // Pre-register a tool with a conflicting name
    struct DupTool;
    #[async_trait::async_trait]
    impl Tool for DupTool {
        fn name(&self) -> &'static str {
            "dup_tool"
        }
        fn description(&self) -> &'static str {
            ""
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn call(
            &self,
            _args: &serde_json::Value,
            _ctx: &agent_base::ToolContext,
        ) -> AgentResult<Vec<Content>> {
            Ok(vec![Content::text("ok")])
        }
    }
    {
        let tools = runtime.tools_mut();
        let mut reg = tokio::task::block_in_place(|| tools.blocking_write());
        reg.register(DupTool);
    }

    let factory: MultiAgentToolFactory = Arc::new(|_rt| {
        struct FakeTool;
        #[async_trait::async_trait]
        impl Tool for FakeTool {
            fn name(&self) -> &'static str {
                "dup_tool"
            }
            fn description(&self) -> &'static str {
                ""
            }
            fn schema(&self) -> serde_json::Value {
                serde_json::json!({})
            }
            async fn call(
                &self,
                _args: &serde_json::Value,
                _ctx: &agent_base::ToolContext,
            ) -> AgentResult<Vec<Content>> {
                Ok(vec![Content::text("ok")])
            }
        }
        vec![Arc::new(FakeTool)]
    });

    let mut existing = HashSet::new();
    existing.insert("dup_tool".to_string());

    let result = setup_multi_agent(
        &runtime,
        MultiAgentConfig::enabled(),
        agent_base::Language::En,
        vec![],
        None,
        &existing,
        Some(factory),
    );
    assert!(result.is_ok());
    // dup_tool should NOT have been registered twice
    let tools = tokio::task::block_in_place(|| {
        let tools = runtime.tools_mut();
        let guard = tools.blocking_read();
        guard
            .metadatas()
            .into_iter()
            .map(|m| m.name)
            .collect::<Vec<String>>()
    });
    let count = tools.iter().filter(|n| n.as_str() == "dup_tool").count();
    assert_eq!(count, 1);
}

// ── AgentBuilder factory methods ──

#[cfg(feature = "multi_agent")]
#[tokio::test(flavor = "multi_thread")]
async fn test_builder_with_multi_agent_without_factory_builds_ok() {
    let client = make_client();
    let runtime = AgentBuilder::new(client)
        .with_multi_agent(MultiAgentConfig::enabled())
        .build()
        .unwrap();
    // Should succeed even without a factory (no tools registered)
    let tools = tokio::task::block_in_place(|| {
        let tools = runtime.tools_mut();
        let guard = tools.blocking_read();
        guard
            .metadatas()
            .into_iter()
            .map(|m| m.name)
            .collect::<Vec<String>>()
    });
    // No multi-agent tools registered
    assert!(!tools.contains(&"spawn_agent".to_string()));
}

#[cfg(feature = "multi_agent")]
#[tokio::test(flavor = "multi_thread")]
async fn test_builder_with_factory_registers_tools() {
    let client = make_client();
    // Create a simple factory that registers one recognizable tool
    let factory: MultiAgentToolFactory = Arc::new(|_rt| {
        struct TestTool;
        #[async_trait::async_trait]
        impl Tool for TestTool {
            fn name(&self) -> &'static str {
                "factory_test_tool"
            }
            fn description(&self) -> &'static str {
                ""
            }
            fn schema(&self) -> serde_json::Value {
                serde_json::json!({})
            }
            async fn call(
                &self,
                _args: &serde_json::Value,
                _ctx: &agent_base::ToolContext,
            ) -> AgentResult<Vec<Content>> {
                Ok(vec![Content::text("ok")])
            }
        }
        vec![Arc::new(TestTool)]
    });

    let runtime = AgentBuilder::new(client)
        .with_multi_agent(MultiAgentConfig::enabled())
        .with_multi_agent_tool_factory(factory)
        .build()
        .unwrap();

    let tools = tokio::task::block_in_place(|| {
        let tools = runtime.tools_mut();
        let guard = tools.blocking_read();
        guard
            .metadatas()
            .into_iter()
            .map(|m| m.name)
            .collect::<Vec<String>>()
    });
    assert!(tools.contains(&"factory_test_tool".to_string()));
}

#[cfg(feature = "multi_agent")]
#[test]
fn test_builder_disabled_multi_agent_skips_factory() {
    let client = make_client();
    let factory: MultiAgentToolFactory = Arc::new(|_rt| {
        panic!("factory should not be called when multi-agent is not configured");
    });

    let runtime = AgentBuilder::new(client)
        .with_multi_agent_tool_factory(factory)
        // Don't enable multi-agent — default (None) means disabled
        .build()
        .unwrap();

    let tools = tokio::task::block_in_place(|| {
        let tools = runtime.tools_mut();
        let guard = tools.blocking_read();
        guard
            .metadatas()
            .into_iter()
            .map(|m| m.name)
            .collect::<Vec<String>>()
    });
    assert!(!tools.contains(&"spawn_agent".to_string()));
}

// ── build_multi_agent_system_prompt ──

#[cfg(feature = "multi_agent")]
#[test]
fn test_system_prompt_contains_tool_names() {
    let prompt = build_multi_agent_system_prompt();
    assert!(prompt.contains("spawn_agent"));
    assert!(prompt.contains("send_message"));
    assert!(prompt.contains("trigger"));
    assert!(prompt.contains("list_agents"));
    assert!(prompt.contains("close_agent"));
    // followup_task is gone (§8.3) — guidance must not advertise a tool
    // the factory no longer registers.
    assert!(!prompt.contains("followup_task"));
    // wait_agent is gone too — results are pushed, the parent waits by
    // ending its turn.
    assert!(!prompt.contains("wait_agent"));
    assert!(prompt.contains("pushed into your context automatically"));
}

#[cfg(feature = "multi_agent")]
#[test]
fn test_system_prompt_contains_guidance() {
    let prompt = build_multi_agent_system_prompt();
    assert!(prompt.contains("When to Spawn"));
    assert!(prompt.contains("When NOT to Spawn"));
    assert!(prompt.contains("How Results Reach You"));
}

// ── without_multi_agent ──

#[cfg(feature = "multi_agent")]
#[tokio::test(flavor = "multi_thread")]
async fn test_without_multi_agent_clears_config_and_factory() {
    let client = make_client();

    // Set up a factory that would panic if called — without_multi_agent should prevent it
    let factory: MultiAgentToolFactory = Arc::new(|_rt| {
        panic!("factory should not be called when multi-agent is cleared");
    });

    let runtime = AgentBuilder::new(client)
        .with_multi_agent(MultiAgentConfig::enabled())
        .with_multi_agent_tool_factory(factory)
        .without_multi_agent() // clear both
        .build()
        .unwrap();

    let tools = tokio::task::block_in_place(|| {
        let tools = runtime.tools_mut();
        let guard = tools.blocking_read();
        guard
            .metadatas()
            .into_iter()
            .map(|m| m.name)
            .collect::<Vec<String>>()
    });
    assert!(!tools.contains(&"spawn_agent".to_string()));
}

// ── apply_if ──

#[test]
fn test_apply_if_some_applies_transformation() {
    let client = make_client();
    let builder = AgentBuilder::new(client)
        .apply_if(Some("custom prompt"), |b, prompt| b.system_prompt(prompt));
    // system_prompt is stored in self.system_prompt; verify it was set
    assert!(builder.system_prompt.unwrap().contains("custom prompt"));
}

#[test]
fn test_apply_if_none_passes_through() {
    let client = make_client();
    let builder = AgentBuilder::new(client).apply_if(None as Option<&str>, |_b, _prompt| {
        panic!("should not be called when value is None");
    });
    assert!(builder.system_prompt.is_none());
}

// ── build_memory_system_prompt ──

#[test]
fn test_build_memory_system_prompt_non_empty() {
    let prompt = build_memory_system_prompt();
    assert!(!prompt.is_empty());
    assert!(prompt.contains("Memory"));
    assert!(prompt.contains("MEMORY.md"));
    assert!(prompt.contains("read_file"));
    assert!(prompt.contains("write_file"));
}

// ── build_memory_system_prompt_with_config (Phase 9b) ──

#[cfg(feature = "memory")]
mod memory_tests {
    use super::*;
    use crate::memory::{CLAUDE_COMPATIBLE_TEMPLATE, MemoryConfig, MemoryStore};

    fn temp_config() -> (tempfile::TempDir, MemoryConfig) {
        let dir = tempfile::tempdir().unwrap();
        let config = MemoryConfig::custom(
            dir.path().join("memory"),
            CLAUDE_COMPATIBLE_TEMPLATE.to_string(),
        );
        (dir, config)
    }

    #[test]
    fn prompt_with_config_substitutes_all_placeholders() {
        let (_dir, config) = temp_config();
        let prompt =
            build_memory_system_prompt_with_config(&config, "- [alpha](alpha.md) — first\n");
        assert!(
            prompt.contains(config.memory_root.to_string_lossy().as_ref()),
            "{prompt}"
        );
        assert!(prompt.contains("- [alpha](alpha.md) — first"), "{prompt}");
        assert!(prompt.contains("memory_write"), "{prompt}");
        assert!(prompt.contains("memory_delete"), "{prompt}");
        assert!(!prompt.contains("{memory_root}"), "{prompt}");
        assert!(!prompt.contains("{index_content}"), "{prompt}");
        assert!(!prompt.contains("{tools_description}"), "{prompt}");
    }

    #[test]
    fn prompt_with_config_handles_empty_index() {
        let (_dir, config) = temp_config();
        let prompt = build_memory_system_prompt_with_config(&config, "");
        assert!(prompt.contains("no memories yet"), "{prompt}");
    }

    #[test]
    fn memory_disabled_by_default_registers_nothing() {
        let client = make_client();
        let runtime = AgentBuilder::new(client).build().unwrap();
        let names = runtime_tool_names(&runtime);
        assert!(!names.iter().any(|n| n.starts_with("memory_")), "{names:?}");
        assert!(runtime.config().system_prompt.is_none());
    }

    #[test]
    fn without_memory_config_clears_previous_config() {
        let (_dir, config) = temp_config();
        let client = make_client();
        let builder = AgentBuilder::new(client)
            .memory_config(config)
            .without_memory_config();
        assert!(builder.memory_config.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn memory_config_registers_tools_and_injects_prompt() {
        let (dir, config) = temp_config();

        // Pre-seed one memory so the injected prompt carries a real row.
        let store = MemoryStore::new(config.memory_root.clone(), "MEMORY.md");
        store
            .write_memory("seed-note", "seeded before build", "project", "body", None)
            .unwrap();

        let client = make_client();
        let runtime = AgentBuilder::new(client)
            .system_prompt("base prompt")
            .memory_config(config)
            .build()
            .unwrap();

        // All four tools registered.
        let names = runtime_tool_names(&runtime);
        for expected in [
            "memory_write",
            "memory_read",
            "memory_list",
            "memory_delete",
        ] {
            assert!(names.contains(&expected.to_string()), "{names:?}");
        }

        // Prompt = base + memory section with the seeded index row.
        let prompt =
            tokio::task::block_in_place(|| runtime.config().system_prompt.clone().unwrap());
        assert!(prompt.starts_with("base prompt"), "{prompt}");
        assert!(
            prompt.contains("- [seed-note](seed-note.md) — seeded before build"),
            "{prompt}"
        );
        assert!(
            prompt.contains(dir.path().join("memory").to_string_lossy().as_ref()),
            "{prompt}"
        );
    }
}

// ── Named tool for register_tool / skill tests ──

struct NamedTool(&'static str);

#[async_trait::async_trait]
impl Tool for NamedTool {
    fn name(&self) -> &'static str {
        self.0
    }

    fn description(&self) -> &'static str {
        ""
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &agent_base::ToolContext,
    ) -> AgentResult<Vec<Content>> {
        Ok(vec![Content::text("ok")])
    }
}

fn runtime_tool_names(runtime: &AgentRuntime) -> Vec<String> {
    tokio::task::block_in_place(|| {
        let tools = runtime.tools_mut();
        let guard = tools.blocking_read();
        guard.metadatas().into_iter().map(|m| m.name).collect()
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn test_builder_scalar_passthrough_methods() {
    let client = make_client();
    let runtime = AgentBuilder::new(client)
        .enable_thought(true)
        .reasoning(agent_base::ReasoningConfig::default())
        .enable_thinking(false)
        .thinking_budget(1000)
        .tool_timeout(5000)
        .max_tool_output_chars(4000)
        .max_sessions(16)
        .max_turns_per_session(20)
        .execution_max_turns(10)
        .max_message_tokens(8000)
        .context_window(64_000)
        .context_window_manager(agent_base::ContextWindowManager::new(64_000))
        .response_format(agent_base::ResponseFormat::JsonObject)
        .llm_retry(agent_base::RetryConfig::default())
        .tool_error_retry_prompt("please retry")
        .language(agent_base::Language::En)
        .event_bus_capacity(256)
        .build()
        .unwrap();

    assert!(runtime.provider().capabilities().supports_streaming);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_register_tool_variants() {
    let client = make_client();
    let runtime = AgentBuilder::new(client)
        .register_tool(NamedTool("tool_by_value"))
        .register_tool_arc(Arc::new(NamedTool("tool_by_arc")))
        .build()
        .unwrap();

    let names = runtime_tool_names(&runtime);
    assert!(names.contains(&"tool_by_value".to_string()));
    assert!(names.contains(&"tool_by_arc".to_string()));
}

#[cfg(feature = "skill")]
mod skill_tests {
    use super::*;
    use crate::skill::Skill;

    struct TestSkill;

    impl Skill for TestSkill {
        fn name(&self) -> &'static str {
            "test_skill"
        }

        fn brief_description(&self) -> String {
            "a test skill".to_string()
        }

        fn detailed_description(&self) -> String {
            "detailed test skill".to_string()
        }

        fn tools(&self) -> Vec<Arc<dyn Tool>> {
            vec![]
        }
    }

    struct ToolSkill;

    impl Skill for ToolSkill {
        fn name(&self) -> &'static str {
            "tool_skill"
        }

        fn brief_description(&self) -> String {
            "skill with a tool".to_string()
        }

        fn detailed_description(&self) -> String {
            "skill that provides a tool".to_string()
        }

        fn tools(&self) -> Vec<Arc<dyn Tool>> {
            vec![Arc::new(NamedTool("skill_provided_tool"))]
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_register_skill_builds_ok() {
        let client = make_client();
        let runtime = AgentBuilder::new(client)
            .register_skill(TestSkill)
            .build()
            .unwrap();
        // Prompt injection is applied during build; no tools provided.
        assert!(runtime_tool_names(&runtime).is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_register_skill_with_tool_registers_tool() {
        let client = make_client();
        let runtime = AgentBuilder::new(client)
            .register_skill(ToolSkill)
            .build()
            .unwrap();
        assert!(runtime_tool_names(&runtime).contains(&"skill_provided_tool".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_register_skill_tool_name_conflict() {
        let client = make_client();
        let result = AgentBuilder::new(client)
            .register_tool(NamedTool("skill_provided_tool"))
            .register_skill(ToolSkill)
            .build();
        let err = result.err().unwrap();
        assert!(format!("{err}").contains("Tool name conflict"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_skill_detail_tool_factory_registers_tool() {
        let client = make_client();
        let factory: SkillDetailToolFactory = Arc::new(|_skills, name| {
            assert_eq!(name, "get_skill_detail");
            Arc::new(NamedTool("detail_tool"))
        });
        let runtime = AgentBuilder::new(client)
            .register_skill(TestSkill)
            .with_skill_detail_tool_factory(factory)
            .build()
            .unwrap();
        assert!(runtime_tool_names(&runtime).contains(&"detail_tool".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_list_skills_tool_factory_registers_tool() {
        let client = make_client();
        let factory: ListSkillsToolFactory =
            Arc::new(|_registry| Arc::new(NamedTool("list_skills_tool")));
        let runtime = AgentBuilder::new(client)
            .register_skill(TestSkill)
            .with_list_skills_tool_factory(factory)
            .build()
            .unwrap();
        assert!(runtime_tool_names(&runtime).contains(&"list_skills_tool".to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_disable_skill_prompt_injection_builds() {
        let client = make_client();
        let runtime = AgentBuilder::new(client)
            .register_skill(ToolSkill)
            .disable_skill_prompt_injection()
            .build()
            .unwrap();
        // Tool still registered; prompt injection skipped.
        assert!(runtime_tool_names(&runtime).contains(&"skill_provided_tool".to_string()));
    }
}

// ── Agent instructions tests ──

#[test]
fn test_build_agent_instructions_prompt_reads_files() {
    let tmp = tempfile::tempdir().unwrap();
    let file1 = tmp.path().join("CLAUDE.md");
    let file2 = tmp.path().join("INSTRUCTIONS.md");
    std::fs::write(&file1, "# User instructions\nBe helpful.").unwrap();
    std::fs::write(&file2, "# Project instructions\nFollow conventions.").unwrap();

    let prompt = build_agent_instructions_prompt(&[file1, file2]);
    assert!(prompt.is_some());
    let prompt = prompt.unwrap();
    // Higher priority (later entries) appear first
    assert!(prompt.starts_with("# Project instructions"));
    assert!(prompt.contains("# User instructions"));
    assert!(prompt.contains("Be helpful."));
    assert!(prompt.contains("Follow conventions."));
}

#[test]
fn test_build_agent_instructions_prompt_skips_missing_files() {
    let prompt = build_agent_instructions_prompt(&[
        PathBuf::from("/nonexistent/CLAUDE.md"),
        PathBuf::from("/nonexistent/INSTRUCTIONS.md"),
    ]);
    assert!(prompt.is_none());
}

#[test]
fn test_build_agent_instructions_prompt_skips_empty_files() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("empty.md");
    std::fs::write(&file, "   \n  ").unwrap();

    let prompt = build_agent_instructions_prompt(&[file]);
    assert!(prompt.is_none());
}
