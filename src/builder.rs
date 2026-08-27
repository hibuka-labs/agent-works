use std::collections::HashSet;
use std::sync::Arc;

use agent_base::{AgentResult, AgentRuntime, Tool};

#[cfg(feature = "multi_agent")]
use crate::multi_agent::{MultiAgentConfig, MultiAgentRuntime};

#[cfg(feature = "skill")]
use crate::skill::{LazySkillPrompter, Skill, SkillPrompter};

/// Factory type for creating multi-agent tools from a MultiAgentRuntime.
#[cfg(feature = "multi_agent")]
pub type MultiAgentToolFactory =
    Arc<dyn Fn(Arc<MultiAgentRuntime>) -> Vec<Arc<dyn Tool>> + Send + Sync>;

/// Factory type for creating a skill detail tool from skills and a tool name.
#[cfg(feature = "skill")]
pub type SkillDetailToolFactory =
    Arc<dyn Fn(Vec<Arc<dyn Skill>>, String) -> Arc<dyn Tool> + Send + Sync>;

/// Factory type for creating a list-skills tool from a SkillRegistry.
#[cfg(feature = "skill")]
pub type ListSkillsToolFactory =
    Arc<dyn Fn(Arc<crate::skill::SkillRegistry>) -> Arc<dyn Tool> + Send + Sync>;

pub struct AgentBuilder {
    inner: agent_base::AgentBuilder,
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
    system_prompt: Option<String>,
    tool_names: HashSet<String>,
    /// Business tools to pass to child agents (all registered tools).
    business_tools: Vec<Arc<dyn Tool>>,
    /// Multi-agent configuration (None = disabled).
    #[cfg(feature = "multi_agent")]
    multi_agent_config: Option<MultiAgentConfig>,
    /// Factory to create multi-agent tools (injected by phi-kernel-tools).
    #[cfg(feature = "multi_agent")]
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
    /// Optional: inject a custom skill-detail tool (old tool-based mode).
    /// In default prompt-injection mode, the LLM reads `SKILL.md` via
    /// `read_file` — no dedicated detail tool is needed.
    #[cfg(feature = "skill")]
    skill_detail_tool_factory: Option<SkillDetailToolFactory>,
    #[cfg(feature = "skill")]
    list_skills_tool_factory: Option<ListSkillsToolFactory>,
    #[cfg(feature = "skill")]
    disable_skill_prompt_injection: bool,
}

impl AgentBuilder {
    pub fn new(client: Arc<dyn agent_base::llm_trait::LlmProvider>) -> Self {
        Self {
            inner: agent_base::AgentBuilder::new(client.clone()),
            client,
            system_prompt: None,
            tool_names: HashSet::new(),
            business_tools: Vec::new(),
            #[cfg(feature = "multi_agent")]
            multi_agent_config: None,
            #[cfg(feature = "multi_agent")]
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
            list_skills_tool_factory: None,
            #[cfg(feature = "skill")]
            disable_skill_prompt_injection: false,
        }
    }

    /// Enable multi-agent support with the given configuration.
    ///
    /// Also sets the tool factory to create the 6 multi-agent tools.
    /// Callers should use `phi_kernel_tools::multi_agent::create_all_tools` as the factory.
    #[cfg(feature = "multi_agent")]
    pub fn with_multi_agent(mut self, config: MultiAgentConfig) -> Self {
        self.multi_agent_config = Some(config);
        self
    }

    /// Disable multi-agent support.
    ///
    /// Removes any previously set multi-agent configuration. No multi-agent tools
    /// will be registered and the system prompt will not mention multi-agent capabilities.
    #[cfg(feature = "multi_agent")]
    pub fn without_multi_agent(mut self) -> Self {
        self.multi_agent_config = None;
        self.multi_agent_tool_factory = None;
        self
    }

    /// Set a custom factory for creating multi-agent tools.
    ///
    /// The factory receives the `MultiAgentRuntime` and returns the tools to register.
    /// If not set but multi-agent is enabled, no tools are registered (caller must
    /// set this for multi-agent to work).
    #[cfg(feature = "multi_agent")]
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

    /// Set a custom factory for creating the list-skills tool.
    ///
    /// The factory receives the SkillRegistry and returns the tool.
    #[cfg(feature = "skill")]
    pub fn with_list_skills_tool_factory(mut self, factory: ListSkillsToolFactory) -> Self {
        self.list_skills_tool_factory = Some(factory);
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

    pub fn max_sessions(self, max: usize) -> Self {
        Self {
            inner: self.inner.max_sessions(max),
            ..self
        }
    }

    pub fn max_turns_per_session(self, max: usize) -> Self {
        Self {
            inner: self.inner.max_turns_per_session(max),
            ..self
        }
    }

    pub fn execution_max_turns(self, max: u32) -> Self {
        Self {
            inner: self.inner.execution_max_turns(max),
            ..self
        }
    }

    pub fn max_message_tokens(self, max: usize) -> Self {
        Self {
            inner: self.inner.max_message_tokens(max),
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

    pub fn guard(self, guard: impl agent_base::ReactLoopGuard + 'static) -> Self {
        Self {
            inner: self.inner.guard(guard),
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

    /// Conditionally apply a transformation when `value` is `Some`.
    ///
    /// This is a convenience for option-chaining builder patterns:
    ///
    /// ```ignore
    /// builder.apply_if(args.thinking_budget, |b, budget| b.thinking_budget(budget))
    /// ```
    pub fn apply_if<T>(self, value: Option<T>, f: impl FnOnce(Self, T) -> Self) -> Self {
        match value {
            Some(v) => f(self, v),
            None => self,
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

    #[allow(dead_code, unused_mut)]
    fn build_inner(mut self) -> AgentResult<AgentRuntime> {
        // Inject default guard if none was set by the consumer.
        // Uses the same LLM client for the judge (enabled by default).
        if self.inner.get_guard().is_none() {
            self.inner = self
                .inner
                .guard(crate::guard::DefaultGuard::with_llm_client(
                    crate::guard::DefaultGuardConfig::default(),
                    self.client.clone(),
                ));
        }

        #[cfg(feature = "multi_agent")]
        let lang = self.language.clone().unwrap_or_default();
        #[cfg(feature = "multi_agent")]
        let business_tools = std::mem::take(&mut self.business_tools);
        #[cfg(feature = "multi_agent")]
        let error_recovery = self.error_recovery.clone();
        #[cfg(feature = "multi_agent")]
        let tool_names = self.tool_names.clone();

        #[cfg(feature = "multi_agent")]
        let ma_config = self.multi_agent_config.clone();
        #[cfg(feature = "multi_agent")]
        let ma_tool_factory = self.multi_agent_tool_factory.take();

        // Inject multi-agent prompt before build
        #[cfg(feature = "multi_agent")]
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
        #[cfg(feature = "multi_agent")]
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

    /// Build the runtime with skill support.
    ///
    /// # Runtime requirement
    ///
    /// This method uses [`tokio::task::block_in_place`] to populate the skill
    /// registry from a synchronous context. It **requires** a multi-threaded
    /// tokio runtime. Calling it on a `#[tokio::main]` single-threaded
    /// (`current_thread`) runtime will panic.
    ///
    /// The phi-agent CLI and all examples use the default multi-threaded runtime,
    /// so this is safe in practice.
    #[cfg(feature = "skill")]
    fn build_with_skills(mut self) -> AgentResult<AgentRuntime> {
        // Inject default guard if none was set by the consumer.
        // Uses the same LLM client for the judge (enabled by default).
        if self.inner.get_guard().is_none() {
            self.inner = self
                .inner
                .guard(crate::guard::DefaultGuard::with_llm_client(
                    crate::guard::DefaultGuardConfig::default(),
                    self.client.clone(),
                ));
        }

        let mut ab = self.inner;
        #[cfg(feature = "multi_agent")]
        let lang = self.language.clone().unwrap_or_default();
        #[cfg(feature = "multi_agent")]
        let business_tools = std::mem::take(&mut self.business_tools);
        #[cfg(feature = "multi_agent")]
        let error_recovery = self.error_recovery.clone();
        #[cfg(feature = "multi_agent")]
        let tool_names = self.tool_names.clone();

        #[cfg(feature = "multi_agent")]
        let ma_config = self.multi_agent_config.clone();
        #[cfg(feature = "multi_agent")]
        let ma_tool_factory = self.multi_agent_tool_factory.take();

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
                let skill_prompt = prompter.build_prompt(&skill_refs, &self.skill_detail_tool_name);
                if !skill_prompt.is_empty() {
                    let new_prompt = match self.system_prompt.take() {
                        Some(existing) => format!("{}\n\n---\n\n{}", existing, skill_prompt),
                        None => skill_prompt,
                    };
                    self.system_prompt = Some(new_prompt.clone());
                    ab = ab.system_prompt(new_prompt);
                }
            }

            // Use injected factory if available, otherwise skip — prompt-injection
            // mode uses read_file instead of a dedicated detail tool.
            if let Some(factory) = self.skill_detail_tool_factory.take() {
                let detail_tool = factory(skill_refs.clone(), self.skill_detail_tool_name);
                ab = ab.register_tool_arc(detail_tool);
            }

            // Create SkillRegistry and populate it for the list-skills tool
            if let Some(factory) = self.list_skills_tool_factory.take() {
                let registry = Arc::new(crate::skill::SkillRegistry::new());
                for skill in &skill_refs {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            registry.register(skill.clone()).await;
                        })
                    });
                }
                let list_tool = factory(registry);
                ab = ab.register_tool_arc(list_tool);
            }
        }

        // Inject multi-agent prompt
        #[cfg(feature = "multi_agent")]
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
        #[cfg(feature = "multi_agent")]
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
#[cfg(feature = "multi_agent")]
pub fn setup_multi_agent(
    runtime: &AgentRuntime,
    config: MultiAgentConfig,
    lang: agent_base::Language,
    business_tools: Vec<Arc<dyn Tool>>,
    error_recovery: Option<Arc<dyn agent_base::ToolErrorRecovery>>,
    existing_tool_names: &HashSet<String>,
    tool_factory: Option<MultiAgentToolFactory>,
) -> AgentResult<Arc<MultiAgentRuntime>> {
    let client = runtime.provider();
    let cancel_token = runtime.cancel_token();
    let tool_policy = runtime.tool_policy().cloned();
    let approval_handler = runtime.approval_handler().cloned();

    let ma_runtime = Arc::new(MultiAgentRuntime::new(
        config.clone(),
        client,
        business_tools,
        cancel_token,
        error_recovery,
        lang,
        tool_policy,
        approval_handler,
    ));

    // Set parent session manager for fork_history support
    ma_runtime.set_session_manager(Arc::new(runtime.session_manager().clone()));

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
#[cfg(feature = "multi_agent")]
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

/// Build the memory system prompt guidance.
///
/// Tells the LLM how to use the file-based persistent memory system.
/// Memory is stored as markdown files — the LLM uses `read_file` / `write_file`
/// to manage them, following the same convention as Claude Code Memory.
///
/// This is prompt-injection only — no dedicated memory tools are registered.
/// The LLM uses the general-purpose file tools to read/write memory files.
pub fn build_memory_system_prompt() -> String {
    r#"## Memory

You have a persistent file-based memory at `.phi/memory/`. Use `read_file` and `write_file` to manage it — there are no dedicated memory tools.

### How Memory Works

- `MEMORY.md` is the index — it lists all memories with one-line descriptions. Read it first when you need to recall something.
- Each memory is a separate `.md` file with YAML frontmatter:
  ```yaml
  ---
  name: <short-kebab-case-slug>
  description: <one-line summary — used to decide relevance during recall>
  metadata:
    node_type: memory
    type: user | feedback | project | reference
  ---

  <the fact or instruction>
  ```
- The `description` field is the key for recall — write it so you can tell at a glance whether this memory is relevant to the current task.
- Link related memories with `[[memory-name]]` in the body.
- `user` type = who the user is (role, expertise, preferences).
- `feedback` type = guidance the user has given on how you should work.
- `project` type = ongoing work, goals, or constraints.
- `reference` type = pointers to external resources (URLs, dashboards, tickets).

### When to Use Memory

- The user explicitly asks you to remember something ("remember this", "save that")
- You learn something important about the user's preferences or workflow
- After completing a significant task, save context that would help in future sessions
- The user gives you feedback on how to work — save it as `feedback` type

### When NOT to Use Memory

- For transient information that won't be useful beyond this session
- For facts already recorded in the codebase (code structure, git history, config files)
- For items that only matter to the current conversation

### Pro Tips

- When creating your first memory of a new type, you can read template files for format reference (check `.phi/templates/memory/` if available).
- Keep the MEMORY.md index concise — it's loaded into context every session.
- Before writing a new memory, check if an existing file already covers it — update instead of duplicating.

### Workflow

**To recall:** read `MEMORY.md` → find relevant entries by description → read the specific `.md` files you need.
**To remember:** create a new `.md` file with proper frontmatter → update `MEMORY.md` with a new entry.
**To update:** edit the existing `.md` file (don't create a duplicate).
**To forget:** delete the `.md` file → remove its entry from `MEMORY.md`."#
        .to_string()
}

#[cfg(test)]
mod tests {
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
        // Verify no tools were registered (the 6 multi-agent tools are absent)
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
        assert!(prompt.contains("followup_task"));
        assert!(prompt.contains("wait_agent"));
        assert!(prompt.contains("list_agents"));
        assert!(prompt.contains("close_agent"));
    }

    #[cfg(feature = "multi_agent")]
    #[test]
    fn test_system_prompt_contains_guidance() {
        let prompt = build_multi_agent_system_prompt();
        assert!(prompt.contains("When to Spawn"));
        assert!(prompt.contains("When NOT to Spawn"));
        assert!(prompt.contains("Communication Pattern"));
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
}
