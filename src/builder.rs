use std::collections::HashSet;
use std::sync::Arc;

use agent_base::{AgentResult, AgentRuntime, LlmClient, Tool};

use crate::multi_agent::{MultiAgentConfig, MultiAgentRuntime};

#[cfg(feature = "skill")]
use crate::skill::{LazySkillPrompter, Skill, SkillDetailTool, SkillPrompter};

pub struct AgentBuilder {
    inner: agent_base::AgentBuilder,
    system_prompt: Option<String>,
    tool_names: HashSet<String>,
    /// Business tools to pass to child agents (all registered tools).
    business_tools: Vec<Arc<dyn Tool>>,
    /// Multi-agent configuration (None = disabled).
    multi_agent_config: Option<MultiAgentConfig>,
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
            error_recovery: None,
            language: None,
            #[cfg(feature = "skill")]
            skills: Vec::new(),
            #[cfg(feature = "skill")]
            skill_prompter: None,
            #[cfg(feature = "skill")]
            skill_detail_tool_name: "get_skill_detail".to_string(),
            #[cfg(feature = "skill")]
            disable_skill_prompt_injection: false,
        }
    }

    /// Enable multi-agent support with the given configuration.
    pub fn with_multi_agent(mut self, config: MultiAgentConfig) -> Self {
        self.multi_agent_config = Some(config);
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

        // Post-build: register multi-agent tools if enabled
        if let Some(config) = ma_config && config.enabled {
            setup_multi_agent(
                &runtime, config, lang, business_tools, error_recovery, &tool_names,
            )?;
        }

        Ok(runtime)
    }

    #[cfg(feature = "skill")]
    fn build_with_skills(mut self) -> AgentResult<AgentRuntime> {
        let mut ab = self.inner;
        let lang = self.language.clone().unwrap_or_default();
        let ma_config = self.multi_agent_config.clone();
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

            let detail_tool = SkillDetailTool::new(skill_refs, self.skill_detail_tool_name);
            ab = ab.register_tool(detail_tool);
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
        if let Some(config) = ma_config && config.enabled {
            setup_multi_agent(
                &runtime, config, lang, business_tools, error_recovery, &tool_names,
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
fn setup_multi_agent(
    runtime: &AgentRuntime,
    config: MultiAgentConfig,
    lang: agent_base::Language,
    business_tools: Vec<Arc<dyn Tool>>,
    error_recovery: Option<Arc<dyn agent_base::ToolErrorRecovery>>,
    existing_tool_names: &HashSet<String>,
) -> AgentResult<()> {
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

    // Register 6 multi-agent tools
    let tools = crate::multi_agent::tools::create_all_tools(ma_runtime);
    let registry = runtime.tools_mut();
    let mut reg = tokio::task::block_in_place(|| registry.blocking_write());
    for tool in tools {
        let tool_name = tool.name().to_string();
        if !existing_tool_names.contains(&tool_name) {
            reg.register_arc(tool);
        }
    }
    drop(reg);

    Ok(())
}

/// Build the multi-agent system prompt guidance for the main agent.
fn build_multi_agent_system_prompt() -> String {
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
