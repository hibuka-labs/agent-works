use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use agent_base::Tool;

#[cfg(feature = "multi_agent")]
use crate::multi_agent::{MultiAgentConfig, MultiAgentRuntime};

#[cfg(feature = "memory")]
use crate::memory::MemoryConfig;

#[cfg(feature = "skill")]
use crate::skill::{Skill, SkillPrompter};

/// Read agent instruction files and build a combined prompt section.
///
/// Files are read in order; later entries have higher priority and appear
/// first in the output. Missing files are silently skipped.
fn build_agent_instructions_prompt(paths: &[PathBuf]) -> Option<String> {
    let mut sections = Vec::new();

    for path in paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                sections.push(trimmed.to_string());
            }
        }
    }

    if sections.is_empty() {
        return None;
    }

    // Later entries (higher priority) appear first
    sections.reverse();
    Some(sections.join("\n\n---\n\n"))
}

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
    /// Optional inline context compactor for the react loop.
    context_compactor: Option<Arc<dyn agent_base::ContextCompaction>>,
    /// Paths to agent instruction files (e.g., CLAUDE.md, INSTRUCTIONS.md).
    /// These files are read at build time and appended to the system prompt.
    /// Supports multiple paths (user-level + project-level); later entries
    /// have higher priority and appear first in the prompt.
    agent_instructions_paths: Vec<PathBuf>,
    /// Memory configuration (Phase 9b). `None` = memory disabled: no memory
    /// tools registered, no memory prompt injected (opt-in by design).
    #[cfg(feature = "memory")]
    memory_config: Option<MemoryConfig>,
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
            context_compactor: None,
            agent_instructions_paths: Vec::new(),
            #[cfg(feature = "memory")]
            memory_config: None,
        }
    }

    /// Enable multi-agent support with the given configuration.
    ///
    /// Also sets the tool factory to create the 5 multi-agent tools.
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

    /// Mount this runtime's action backend (M3 T10c): text turns route to it,
    /// tool-envelope turns keep the tool-call backend. One backend per
    /// runtime — build one runtime per session when the backend holds a
    /// resident engine (the product's assembly-time choice).
    pub fn action_backend(self, backend: Box<dyn agent_base::ActionBackend>) -> Self {
        Self {
            inner: self.inner.action_backend(backend),
            ..self
        }
    }

    /// The product's interaction surface for the action backend's `ExecCtx`
    /// (OV-6): consent and ask_user suspend through it. Without one, every
    /// interaction fails closed — a cell reaching for the human refuses, it
    /// never hangs or silently allows.
    pub fn interaction(self, channel: Arc<dyn agent_base::InteractionChannel>) -> Self {
        Self {
            inner: self.inner.interaction(channel),
            ..self
        }
    }

    /// Per-interaction deadline for the action backend's suspensions (`None`
    /// = unbounded human time, the run's cancel token is the only guard).
    /// Unset = the kernel's default watchdog.
    pub fn interaction_timeout(self, interact_timeout: Option<std::time::Duration>) -> Self {
        Self {
            inner: self.inner.interaction_timeout(interact_timeout),
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

    /// Set an inline context compactor for the react loop.
    ///
    /// When set, the react loop will check token count after each tool turn
    /// and compact the session if it exceeds the configured threshold.
    pub fn context_compactor(mut self, compactor: Arc<dyn agent_base::ContextCompaction>) -> Self {
        self.context_compactor = Some(compactor);
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

    /// Set paths to agent instruction files (e.g., CLAUDE.md, INSTRUCTIONS.md).
    ///
    /// These files are read at build time and appended to the system prompt.
    /// Supports multiple paths (user-level + project-level); later entries
    /// have higher priority and appear first in the prompt.
    ///
    /// Example:
    /// ```rust,ignore
    /// use std::path::PathBuf;
    /// use agent_works::AgentBuilder;
    /// let builder = AgentBuilder::new();
    /// let builder = builder.agent_instructions_paths(vec![
    ///     PathBuf::from("~/.claude/CLAUDE.md"),  // user-level
    ///     PathBuf::from(".claude/CLAUDE.md"),     // project-level
    /// ]);
    /// ```
    pub fn agent_instructions_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.agent_instructions_paths = paths;
        self
    }

    #[cfg(feature = "skill")]
    pub fn skill_detail_tool_name(mut self, name: impl Into<String>) -> Self {
        self.skill_detail_tool_name = name.into();
        self
    }

    /// Enable persistent auto-memory (Phase 9b) with the given configuration.
    ///
    /// Registers the four `memory_*` tools and appends the memory prompt
    /// (MEMORY.md index snapshot + tool guidance) to the system prompt at
    /// build time. Opt-in: without this call no memory tools exist and the
    /// prompt is untouched.
    #[cfg(feature = "memory")]
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.memory_config = Some(config);
        self
    }

    /// Disable auto-memory (removes any previously set configuration).
    #[cfg(feature = "memory")]
    pub fn without_memory_config(mut self) -> Self {
        self.memory_config = None;
        self
    }
}

mod build;
mod prompts;

pub use prompts::build_memory_system_prompt;
#[cfg(feature = "memory")]
pub use prompts::build_memory_system_prompt_with_config;
#[cfg(feature = "multi_agent")]
pub use prompts::{build_multi_agent_system_prompt, setup_multi_agent};

#[cfg(test)]
mod tests;
