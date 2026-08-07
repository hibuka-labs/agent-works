use std::collections::HashSet;
use std::sync::Arc;

use agent_base::{AgentResult, AgentRuntime, LlmClient, Tool};

use crate::multi_agent::{MultiAgentConfig, MultiAgentRuntime};

#[cfg(feature = "skill")]
use crate::skill::{LazySkillPrompter, Skill, SkillPrompter};

/// Factory type for creating multi-agent tools from a MultiAgentRuntime.
pub type MultiAgentToolFactory =
    Arc<dyn Fn(Arc<MultiAgentRuntime>) -> Vec<Arc<dyn Tool>> + Send + Sync>;

/// Factory type for creating a skill detail tool from skills and a tool name.
#[cfg(feature = "skill")]
pub type SkillDetailToolFactory =
    Arc<dyn Fn(Vec<Arc<dyn Skill>>, String) -> Arc<dyn Tool> + Send + Sync>;

pub struct AgentBuilder {
    inner: agent_base::AgentBuilder,
    system_prompt: Option<String>,
    tool_names: HashSet<String>,
    /// Business tools to pass to child agents (all registered tools).
    business_tools: Vec<Arc<dyn Tool>>,
    /// Multi-agent configuration (None = disabled).
    multi_agent_config: Option<MultiAgentConfig>,
    /// Factory to create multi-agent tools (injected by phi-kernel-tools).
    multi_agent_tool_factory: Option<MultiAgentToolFactory>,
    /// Error recovery (stored for multi-agent child inheritance).
    error_recovery: Option<Arc<dyn agent_base::ToolErrorRecovery>>,
    /// Language preference.
    language: Option<agent_base::Language>,
    #[cfg(feature = "skill")]
    skills: Vec<Arc<dyn Skill>>,
    #[cfg(feature = "skill")]
    skill_prompter: Option<Arc<dyn SkillPrompter>>,
    #[cfg(feature = "skill")]
    skill_detail_tool_name: String,
    #[cfg(feature = "skill")]
    skill_detail_tool_factory: Option<SkillDetailToolFactory>,
    #[cfg(feature = "skill")]
    disable_skill_prompt_injection: bool,
}

impl AgentBuilder {
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        Self {
            inner: agent_base::AgentBuilder::new(client),
            system_prompt: None,
            tool_names: HashSet::new(),
            business_tools: Vec::new(),
            multi_agent_config: None,
            multi_agent_tool_factory: None,
            error_recovery: None,
            language: None,
            #[cfg(feature = "skill")]
            skills: Vec::new(),
            #[cfg(feature = "skill")]
            skill_prompter: None,
            #[cfg(feature = "skill")]
            skill_detail_tool_name: "get_skill_detail".to_string(),
            #[cfg(feature = "skill")]
            skill_detail_tool_factory: None,
            #[cfg(feature = "skill")]
            disable_skill_prompt_injection: false,
        }
    }

    /// Enable multi-agent support with the given configuration.
    ///
    /// Also sets the tool factory to create the 6 multi-agent tools.
    /// Callers should use `phi_kernel_tools::multi_agent::create_all_tools` as the factory.
    pub fn with_multi_agent(mut self, config: MultiAgentConfig) -> Self {
        self.multi_agent_config = Some(config);
        self
    }

    /// Set a custom factory for creating multi-agent tools.
    ///
    /// The factory receives the `MultiAgentRuntime` and returns the tools to register.
    /// If not set but multi-agent is enabled, no tools are registered (caller must
    /// set this for multi-agent to work).
    pub fn with_multi_agent_tool_factory(mut self, factory: MultiAgentToolFactory) -> Self {
        self.multi_agent_tool_factory = Some(factory);
        self
    }

    /// Set a custom factory for creating the skill detail tool.
    ///
    /// The factory receives the skill list and tool name, and returns the tool.
    /// If not set but skills are registered, no detail tool is added.
    #[cfg(feature = "skill")]
    pub fn with_skill_detail_tool_factory(mut self, factory: SkillDetailToolFactory) -> Self {
        self.skill_detail_tool_factory = Some(factory);
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        self.inner = self.inner.system_prompt(prompt.clone());
        self.system_prompt = Some(prompt);
        self
    }

    pub fn enable_thought(self, enable: bool) -> Self {
        Self {
            inner: self.inner.enable_thought(enable),
            ..self
        }
    }

    pub fn reasoning(self, config: agent_base::ReasoningConfig) -> Self {
        Self {
            inner: self.inner.reasoning(config),
            ..self
        }
    }

    pub fn enable_thinking(self, enable: bool) -> Self {
        Self {
            inner: self.inner.enable_thinking(enable),
            ..self
        }
    }

    pub fn thinking_budget(self, budget: u64) -> Self {
        Self {
            inner: self.inner.thinking_budget(budget),
            ..self
        }
    }

    pub fn tool_timeout(self, timeout_ms: u64) -> Self {
        Self {
            inner: self.inner.tool_timeout(timeout_ms),
            ..self
        }
    }

    pub fn max_tool_output_chars(self, max_chars: usize) -> Self {
        Self {
            inner: self.inner.max_tool_output_chars(max_chars),
            ..self
        }
    }

    pub fn register_tool(mut self, tool: impl Tool + 'static) -> Self {
        let tool_arc: Arc<dyn Tool> = Arc::new(tool);
        self.tool_names.insert(tool_arc.name().to_string());
        self.business_tools.push(tool_arc.clone());
        self.inner = self.inner.register_tool_arc(tool_arc);
        self
    }

    pub fn register_tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tool_names.insert(tool.name().to_string());
        self.business_tools.push(tool.clone());
        self.inner = self.inner.register_tool_arc(tool);
        self
    }

    pub fn approval_handler(self, handler: Arc<dyn agent_base::ApprovalHandler>) -> Self {
        Self {
            inner: self.inner.approval_handler(handler),
            ..self
        }
    }

    pub fn tool_policy(self, policy: Arc<dyn agent_base::ToolPolicy>) -> Self {
        Self {
            inner: self.inner.tool_policy(policy),
            ..self
        }
    }

    pub fn middleware(self, mw: impl agent_base::Middleware + 'static) -> Self {
        Self {
            inner: self.inner.middleware(mw),
            ..self
        }
    }

    pub fn context_window(self, max_tokens: usize) -> Self {
        Self {
            inner: self.inner.context_window(max_tokens),
            ..self
        }
    }

    pub fn context_window_manager(self, manager: agent_base::ContextWindowManager) -> Self {
        Self {
            inner: self.inner.context_window_manager(manager),
            ..self
        }
    }

    pub fn response_format(self, format: agent_base::ResponseFormat) -> Self {
        Self {
            inner: self.inner.response_format(format),
            ..self
        }
    }

    pub fn llm_retry(self, retry: agent_base::RetryConfig) -> Self {
        Self {
            inner: self.inner.llm_retry(retry),
            ..self
        }
    }

    pub fn session_store(self, store: Arc<dyn agent_base::SessionStore>) -> Self {
        Self {
            inner: self.inner.session_store(store),
            ..self
        }
    }

    pub fn error_recovery(mut self, recovery: Arc<dyn agent_base::ToolErrorRecovery>) -> Self {
        self.error_recovery = Some(recovery.clone());
        self.inner = self.inner.error_recovery(recovery);
        self
    }

    pub fn tool_error_retry_prompt(self, prompt: impl Into<String>) -> Self {
        Self {
            inner: self.inner.tool_error_retry_prompt(prompt),
            ..self
        }
    }

    pub fn language(mut self, language: agent_base::Language) -> Self {
        self.language = Some(language.clone());
        self.inner = self.inner.language(language);
        self
    }

    pub fn event_bus_capacity(self, capacity: usize) -> Self {
        Self {
            inner: self.inner.event_bus_capacity(capacity),
            ..self
        }
    }

    pub fn session_id_generator(
        self,
        generator: Arc<dyn agent_base::types::SessionIdGenerator>,
    ) -> Self {
        Self {
            inner: self.inner.session_id_generator(generator),
            ..self
        }
    }

    #[cfg(feature = "skill")]
    pub fn register_skill(mut self, skill: impl Skill + 'static) -> Self {
        self.skills.push(Arc::new(skill));
        self
    }

    #[cfg(feature = "skill")]
    pub fn register_skills(mut self, skills: Vec<Arc<dyn Skill>>) -> Self {
        self.skills.extend(skills);
        self
    }

    #[cfg(feature = "skill")]
    pub fn skill_prompter(mut self, prompter: Arc<dyn SkillPrompter>) -> Self {
        self.skill_prompter = Some(prompter);
        self
    }

    #[cfg(feature = "skill")]
    pub fn disable_skill_prompt_injection(mut self) -> Self {
        self.disable_skill_prompt_injection = true;
        self
    }

    #[cfg(feature = "skill")]
    pub fn skill_detail_tool_name(mut self, name: impl Into<String>) -> Self {
        self.skill_detail_tool_name = name.into();
        self
    }

    // ── Build ──

    pub fn build(self) -> AgentResult<AgentRuntime> {
        #[cfg(feature = "skill")]
        {
            self.build_with_skills()
        }
        #[cfg(not(feature = "skill"))]
        {
            self.build_inner()
        }
    }

    fn build_inner(mut self) -> AgentResult<AgentRuntime> {
        let lang = self.language.clone().unwrap_or_default();
        let ma_config = self.multi_agent_config.clone();
        let ma_tool_factory = self.multi_agent_tool_factory.take();
        let business_tools = std::mem::take(&mut self.business_tools);
        let error_recovery = self.error_recovery.clone();
        let tool_names = self.tool_names.clone();

        // Inject multi-agent prompt before build
        if ma_config.as_ref().map(|c| c.enabled).unwrap_or(false) {
            let ma_prompt = build_multi_agent_system_prompt();
            let new_prompt = match self.system_prompt.take() {
                Some(existing) => format!("{}\n\n---\n\n{}", existing, ma_prompt),
                None => ma_prompt,
            };
            self.inner = self.inner.system_prompt(new_prompt);
        }

        let runtime = self.inner.build()?;

        // Post-build: register multi-agent tools if enabled and factory is set
        if let Some(config) = ma_config
            && config.enabled
        {
            setup_multi_agent(
                &runtime,
                config,
                lang,
                business_tools,
                error_recovery,
                &tool_names,
                ma_tool_factory,
            )?;
        }

        Ok(runtime)
    }

    #[cfg(feature = "skill")]
    fn build_with_skills(mut self) -> AgentResult<AgentRuntime> {
        let mut ab = self.inner;
        let lang = self.language.clone().unwrap_or_default();
        let ma_config = self.multi_agent_config.clone();
        let ma_tool_factory = self.multi_agent_tool_factory.take();
        let business_tools = std::mem::take(&mut self.business_tools);
        let error_recovery = self.error_recovery.clone();
        let tool_names = self.tool_names.clone();

        // Process skills
        if !self.skills.is_empty() {
            let prompter: Arc<dyn SkillPrompter> = self
                .skill_prompter
                .take()
                .unwrap_or_else(|| Arc::new(LazySkillPrompter::new()));

            let mut skill_refs: Vec<Arc<dyn Skill>> = Vec::new();

            for skill in self.skills {
                for tool in skill.tools() {
                    let tool_name = tool.name().to_string();
                    if self.tool_names.contains(&tool_name) {
                        return Err(agent_base::AgentError::internal(format!(
                            "Tool name conflict: `{}` (Skill `{}`)",
                            tool_name,
                            skill.name()
                        )));
                    }
                    self.tool_names.insert(tool_name);
                    ab = ab.register_tool_arc(tool);
                }
                skill_refs.push(skill);
            }

            if !self.disable_skill_prompt_injection {
                let skill_prompt =
                    prompter.build_prompt(&skill_refs, &self.skill_detail_tool_name);
                if !skill_prompt.is_empty() {
                    let new_prompt = match self.system_prompt.take() {
                        Some(existing) => format!("{}\n\n---\n\n{}", existing, skill_prompt),
                        None => skill_prompt,
                    };
                    self.system_prompt = Some(new_prompt.clone());
                    ab = ab.system_prompt(new_prompt);
                }
            }

            // Use injected factory if available, otherwise skip creating detail tool
            if let Some(factory) = self.skill_detail_tool_factory.take() {
                let detail_tool = factory(skill_refs, self.skill_detail_tool_name);
                ab = ab.register_tool_arc(detail_tool);
            }
        }

        // Inject multi-agent prompt
        if ma_config.as_ref().map(|c| c.enabled).unwrap_or(false) {
            let ma_prompt = build_multi_agent_system_prompt();
            let new_prompt = match self.system_prompt.take() {
                Some(existing) => format!("{}\n\n---\n\n{}", existing, ma_prompt),
                None => ma_prompt,
            };
            ab = ab.system_prompt(new_prompt);
        }

        let runtime = ab.build()?;

        // Post-build: register multi-agent tools
        if let Some(config) = ma_config
            && config.enabled
        {
            setup_multi_agent(
                &runtime,
                config,
                lang,
                business_tools,
                error_recovery,
                &tool_names,
                ma_tool_factory,
            )?;
        }

        Ok(runtime)
    }
}

/// Set up the MultiAgentRuntime, event bridge, and register tools on an already-built runtime.
///
/// # Safety / Runtime Requirement
///
/// This function uses [`tokio::task::block_in_place`] to register tools synchronously.
/// It **requires** a multi-threaded tokio runtime. Calling it on a
/// `#[tokio::main]` single-threaded (`current_thread`) runtime will panic.
///
/// The phi-agent CLI and all examples use the default multi-threaded runtime,
/// so this is safe in practice.
pub fn setup_multi_agent(
    runtime: &AgentRuntime,
    config: MultiAgentConfig,
    lang: agent_base::Language,
    business_tools: Vec<Arc<dyn Tool>>,
    error_recovery: Option<Arc<dyn agent_base::ToolErrorRecovery>>,
    existing_tool_names: &HashSet<String>,
    tool_factory: Option<MultiAgentToolFactory>,
) -> AgentResult<Arc<MultiAgentRuntime>> {
    let client = runtime.client();
    let cancel_token = runtime.cancel_token();

    let ma_runtime = Arc::new(MultiAgentRuntime::new(
        config.clone(),
        client,
        business_tools,
        cancel_token,
        error_recovery,
        lang,
    ));

    // Set up event bridge: child events → parent event bus
    let (event_tx, mut event_rx) =
        tokio::sync::mpsc::unbounded_channel::<agent_base::RuntimeEvent>();
    ma_runtime.set_event_sender(event_tx);
    let parent_runtime = runtime.clone();
    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            parent_runtime.emit_event(event);
        }
    });

    // Register multi-agent tools if a factory is provided
    if let Some(factory) = tool_factory {
        let tools = factory(ma_runtime.clone());
        let registry = runtime.tools_mut();
        let mut reg = tokio::task::block_in_place(|| registry.blocking_write());
        for tool in tools {
            let tool_name = tool.name().to_string();
            if !existing_tool_names.contains(&tool_name) {
                reg.register_arc(tool);
            }
        }
        drop(reg);
    }

    Ok(ma_runtime)
}

/// Build the multi-agent system prompt guidance for the main agent.
pub fn build_multi_agent_system_prompt() -> String {
    r#"## Multi-Agent Capabilities

You have the ability to spawn sub-agents to execute tasks concurrently. Use these tools to delegate work:

- `spawn_agent`: Create a new sub-agent with a specific role. The agent runs independently.
- `send_message`: Send a message to a sub-agent without triggering execution.
- `followup_task`: Assign a task to a sub-agent and trigger its execution. Returns immediately.
- `wait_agent`: Wait for a sub-agent's result. Blocks until the agent completes or timeout.
- `list_agents`: List all active sub-agents and their status.
- `close_agent`: Close a sub-agent and release its resources.

### When to Spawn

- Tasks that can run independently and in parallel (e.g., "research X and Y simultaneously")
- Long-running tasks where you want to check intermediate results
- Decomposing complex tasks into sub-tasks for focused execution

### When NOT to Spawn

- Simple lookups or single-tool calls (just use the tool directly)
- Sequential dependencies where the next step requires the previous result
- Tasks that need your full context or reasoning

### Communication Pattern

1. `spawn_agent` → create the sub-agent
2. `followup_task` → assign work (can call multiple times)
3. `wait_agent` → collect results
4. `close_agent` → clean up when done"#
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::ToolControlFlow;
    use std::pin::Pin;

    // ── Stub LLM client ──

    struct StubClient;

    #[async_trait::async_trait]
    impl LlmClient for StubClient {
        async fn chat(
            &self,
            _messages: &[agent_base::ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&agent_base::ResponseFormat>,
        ) -> AgentResult<serde_json::Value> {
            Ok(serde_json::json!({"choices": [{"message": {"content": "ok"}}]}))
        }

        async fn chat_stream(
            &self,
            _messages: &[agent_base::ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&agent_base::ResponseFormat>,
        ) -> AgentResult<
            Pin<Box<dyn futures_core::Stream<Item = AgentResult<agent_base::StreamChunk>> + Send>>,
        > {
            let chunks: Vec<AgentResult<agent_base::StreamChunk>> = vec![
                Ok(agent_base::StreamChunk::Text("ok".to_string())),
                Ok(agent_base::StreamChunk::Stop),
            ];
            Ok(Box::pin(futures_util::stream::iter(chunks)))
        }

        fn capabilities(&self) -> agent_base::LlmCapabilities {
            agent_base::LlmCapabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_vision: false,
                supports_thinking: false,
                max_context_tokens: None,
                max_output_tokens: None,
            }
        }
    }

    fn make_client() -> Arc<dyn LlmClient> {
        Arc::new(StubClient)
    }

    // ── setup_multi_agent tests ──

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
        // Verify no tools were registered (the 6 multi-agent tools are absent)
        let agents = ma_runtime.list_agents();
        assert!(agents.is_empty());
    }

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
                fn definition(&self) -> serde_json::Value {
                    serde_json::json!({"type": "function", "function": {"name": "fake_tool"}})
                }
                async fn call(
                    &self,
                    _args: &serde_json::Value,
                    _ctx: &agent_base::ToolContext,
                ) -> AgentResult<agent_base::ToolOutput> {
                    Ok(agent_base::ToolOutput {
                        summary: "ok".into(),
                        raw: None,
                        control_flow: ToolControlFlow::Continue,
                        truncation: None,
                    })
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
            fn definition(&self) -> serde_json::Value {
                serde_json::json!({"type": "function", "function": {"name": "dup_tool"}})
            }
            async fn call(
                &self,
                _args: &serde_json::Value,
                _ctx: &agent_base::ToolContext,
            ) -> AgentResult<agent_base::ToolOutput> {
                Ok(agent_base::ToolOutput {
                    summary: "ok".into(),
                    raw: None,
                    control_flow: ToolControlFlow::Continue,
                    truncation: None,
                })
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
                fn definition(&self) -> serde_json::Value {
                    serde_json::json!({"type": "function", "function": {"name": "dup_tool"}})
                }
                async fn call(
                    &self,
                    _args: &serde_json::Value,
                    _ctx: &agent_base::ToolContext,
                ) -> AgentResult<agent_base::ToolOutput> {
                    Ok(agent_base::ToolOutput {
                        summary: "ok".into(),
                        raw: None,
                        control_flow: ToolControlFlow::Continue,
                        truncation: None,
                    })
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
            guard.metadatas().into_iter().map(|m| m.name).collect::<Vec<String>>()
        });
        let count = tools.iter().filter(|n| n.as_str() == "dup_tool").count();
        assert_eq!(count, 1);
    }

    // ── AgentBuilder factory methods ──

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
            guard.metadatas().into_iter().map(|m| m.name).collect::<Vec<String>>()
        });
        // No multi-agent tools registered
        assert!(!tools.contains(&"spawn_agent".to_string()));
    }

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
                fn definition(&self) -> serde_json::Value {
                    serde_json::json!({"type": "function", "function": {"name": "factory_test_tool"}})
                }
                async fn call(
                    &self,
                    _args: &serde_json::Value,
                    _ctx: &agent_base::ToolContext,
                ) -> AgentResult<agent_base::ToolOutput> {
                    Ok(agent_base::ToolOutput {
                        summary: "ok".into(),
                        raw: None,
                        control_flow: ToolControlFlow::Continue,
                        truncation: None,
                    })
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
            guard.metadatas().into_iter().map(|m| m.name).collect::<Vec<String>>()
        });
        assert!(tools.contains(&"factory_test_tool".to_string()));
    }

    #[test]
    fn test_builder_disabled_multi_agent_skips_factory() {
        let client = make_client();
        let factory: MultiAgentToolFactory = Arc::new(|_rt| {
            panic!("factory should not be called when multi-agent is disabled");
        });

        let runtime = AgentBuilder::new(client)
            .with_multi_agent_tool_factory(factory)
            // Don't enable multi-agent — leave default (disabled)
            .build()
            .unwrap();

        let tools = tokio::task::block_in_place(|| {
            let tools = runtime.tools_mut();
            let guard = tools.blocking_read();
            guard.metadatas().into_iter().map(|m| m.name).collect::<Vec<String>>()
        });
        assert!(!tools.contains(&"spawn_agent".to_string()));
    }

    // ── build_multi_agent_system_prompt ──

    #[test]
    fn test_system_prompt_contains_tool_names() {
        let prompt = build_multi_agent_system_prompt();
        assert!(prompt.contains("spawn_agent"));
        assert!(prompt.contains("send_message"));
        assert!(prompt.contains("followup_task"));
        assert!(prompt.contains("wait_agent"));
        assert!(prompt.contains("list_agents"));
        assert!(prompt.contains("close_agent"));
    }

    #[test]
    fn test_system_prompt_contains_guidance() {
        let prompt = build_multi_agent_system_prompt();
        assert!(prompt.contains("When to Spawn"));
        assert!(prompt.contains("When NOT to Spawn"));
        assert!(prompt.contains("Communication Pattern"));
    }
}
