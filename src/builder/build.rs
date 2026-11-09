// `Arc` is needed by every path that hands out a shared runtime, registry or
// prompter: the memory store, the skill registry/prompter, and the
// multi-agent runtime. Gate on all three, not on a subset — a feature set
// with `multi_agent` alone must still compile.
#[cfg(any(feature = "memory", feature = "skill", feature = "multi_agent"))]
use std::sync::Arc;

use agent_base::{AgentResult, AgentRuntime};

use super::AgentBuilder;
use super::build_agent_instructions_prompt;
#[cfg(feature = "memory")]
use super::build_memory_system_prompt_with_config;
#[cfg(feature = "multi_agent")]
use super::build_multi_agent_system_prompt;
#[cfg(feature = "multi_agent")]
use super::setup_multi_agent;

#[cfg(feature = "memory")]
use crate::memory::MemoryStore;
#[cfg(feature = "multi_agent")]
use crate::multi_agent::MultiAgentRuntime;
#[cfg(feature = "skill")]
use crate::skill::{LazySkillPrompter, Skill, SkillPrompter};

impl AgentBuilder {
    /// Register the four memory tools and append the memory prompt (index
    /// snapshot + tool guidance) to the system prompt. No-op when no memory
    /// configuration was set. Called from every `build_*` path right after
    /// the agent-instructions (CLAUDE.md) injection.
    #[cfg(feature = "memory")]
    fn apply_memory_config(mut self) -> Self {
        let Some(config) = self.memory_config.take() else {
            return self;
        };

        let store = Arc::new(MemoryStore::new(
            config.memory_root.clone(),
            config.index_filename.clone(),
        ));

        // Register the memory tools unless the consumer already took the name.
        for tool in crate::tools::create_memory_tools(Arc::clone(&store)) {
            let tool_name = tool.name().to_string();
            if self.tool_names.contains(&tool_name) {
                tracing::warn!(tool = %tool_name, "memory tool name conflict; skipping registration");
                continue;
            }
            self.tool_names.insert(tool_name);
            self.inner = self.inner.register_tool_arc(tool);
        }

        // Inject the index snapshot + tool guidance into the system prompt.
        let index = store.read_index();
        if index.len() > 10_240 {
            tracing::warn!(
                index_bytes = index.len(),
                root = %config.memory_root.display(),
                "memory index is unusually large (>10KB); consider cleaning up stale memories"
            );
        }
        let mem_prompt = build_memory_system_prompt_with_config(&config, &index);
        tracing::info!(
            memory_root = %config.memory_root.display(),
            index_bytes = index.len(),
            tools = 4,
            "injecting memory prompt + tools into system prompt"
        );
        let new_prompt = match self.system_prompt.take() {
            Some(existing) => format!("{existing}\n\n---\n\n{mem_prompt}"),
            None => mem_prompt,
        };
        tracing::info!(
            prompt_len = new_prompt.len(),
            "system prompt after memory injection (full text logged at DEBUG level)"
        );
        tracing::debug!(system_prompt = %new_prompt, "full system prompt after memory injection");
        self.system_prompt = Some(new_prompt.clone());
        self.inner = self.inner.system_prompt(new_prompt);
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

    /// Build the runtime and return the multi-agent runtime (if enabled).
    ///
    /// Same as [`build`](Self::build) but also returns the `MultiAgentRuntime`
    /// created during setup. Callers that need to start the watcher task or
    /// access the mailbox should use this method.
    #[cfg(feature = "multi_agent")]
    pub fn build_with_ma(self) -> AgentResult<(AgentRuntime, Option<Arc<MultiAgentRuntime>>)> {
        #[cfg(feature = "skill")]
        {
            self.build_with_skills_ma()
        }
        #[cfg(not(feature = "skill"))]
        {
            self.build_inner_ma()
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

        // Inject agent instruction files (e.g., CLAUDE.md) into system prompt
        if !self.agent_instructions_paths.is_empty()
            && let Some(instructions) =
                build_agent_instructions_prompt(&self.agent_instructions_paths)
        {
            tracing::info!(
                paths = ?self.agent_instructions_paths,
                instructions_len = instructions.len(),
                "injecting agent instructions into system prompt"
            );
            let new_prompt = match self.system_prompt.take() {
                Some(existing) => format!("{}\n\n---\n\n{}", existing, instructions),
                None => instructions,
            };
            self.system_prompt = Some(new_prompt.clone());
            self.inner = self.inner.system_prompt(new_prompt);
        }

        // Register memory tools + inject memory prompt (Phase 9b)
        #[cfg(feature = "memory")]
        {
            self = self.apply_memory_config();
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

        // Inject inline context compactor if set
        if let Some(compactor) = self.context_compactor.take() {
            self.inner = self.inner.context_compactor(compactor);
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

    /// Like `build_inner` but also returns the `MultiAgentRuntime`.
    #[cfg(feature = "multi_agent")]
    #[allow(dead_code, unused_mut)]
    fn build_inner_ma(mut self) -> AgentResult<(AgentRuntime, Option<Arc<MultiAgentRuntime>>)> {
        if self.inner.get_guard().is_none() {
            self.inner = self
                .inner
                .guard(crate::guard::DefaultGuard::with_llm_client(
                    crate::guard::DefaultGuardConfig::default(),
                    self.client.clone(),
                ));
        }

        // Register memory tools + inject memory prompt (Phase 9b)
        #[cfg(feature = "memory")]
        {
            self = self.apply_memory_config();
        }

        let lang = self.language.clone().unwrap_or_default();
        let business_tools = std::mem::take(&mut self.business_tools);
        let error_recovery = self.error_recovery.clone();
        let tool_names = self.tool_names.clone();
        let ma_config = self.multi_agent_config.clone();
        let ma_tool_factory = self.multi_agent_tool_factory.take();

        if ma_config.as_ref().map(|c| c.enabled).unwrap_or(false) {
            let ma_prompt = build_multi_agent_system_prompt();
            let new_prompt = match self.system_prompt.take() {
                Some(existing) => format!("{}\n\n---\n\n{}", existing, ma_prompt),
                None => ma_prompt,
            };
            self.inner = self.inner.system_prompt(new_prompt);
        }

        if let Some(compactor) = self.context_compactor.take() {
            self.inner = self.inner.context_compactor(compactor);
        }

        let runtime = self.inner.build()?;

        let ma_runtime_out = if let Some(config) = ma_config
            && config.enabled
        {
            Some(setup_multi_agent(
                &runtime,
                config,
                lang,
                business_tools,
                error_recovery,
                &tool_names,
                ma_tool_factory,
            )?)
        } else {
            None
        };

        Ok((runtime, ma_runtime_out))
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

        // Inject agent instruction files (e.g., CLAUDE.md) into system prompt
        if !self.agent_instructions_paths.is_empty()
            && let Some(instructions) =
                build_agent_instructions_prompt(&self.agent_instructions_paths)
        {
            tracing::info!(
                paths = ?self.agent_instructions_paths,
                instructions_len = instructions.len(),
                "injecting agent instructions into system prompt"
            );
            let new_prompt = match self.system_prompt.take() {
                Some(existing) => format!("{}\n\n---\n\n{}", existing, instructions),
                None => instructions,
            };
            self.system_prompt = Some(new_prompt.clone());
            self.inner = self.inner.system_prompt(new_prompt);
        }

        // Register memory tools + inject memory prompt (Phase 9b)
        #[cfg(feature = "memory")]
        {
            self = self.apply_memory_config();
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

        // Inject inline context compactor if set
        if let Some(compactor) = self.context_compactor.take() {
            ab = ab.context_compactor(compactor);
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

    /// Like `build_with_skills` but also returns the `MultiAgentRuntime`.
    #[cfg(all(feature = "skill", feature = "multi_agent"))]
    fn build_with_skills_ma(
        mut self,
    ) -> AgentResult<(AgentRuntime, Option<Arc<MultiAgentRuntime>>)> {
        if self.inner.get_guard().is_none() {
            self.inner = self
                .inner
                .guard(crate::guard::DefaultGuard::with_llm_client(
                    crate::guard::DefaultGuardConfig::default(),
                    self.client.clone(),
                ));
        }

        // Inject agent instruction files (e.g., CLAUDE.md) into system prompt
        if !self.agent_instructions_paths.is_empty()
            && let Some(instructions) =
                build_agent_instructions_prompt(&self.agent_instructions_paths)
        {
            tracing::info!(
                paths = ?self.agent_instructions_paths,
                instructions_len = instructions.len(),
                "injecting agent instructions into system prompt"
            );
            let new_prompt = match self.system_prompt.take() {
                Some(existing) => format!("{}\n\n---\n\n{}", existing, instructions),
                None => instructions,
            };
            self.system_prompt = Some(new_prompt.clone());
            self.inner = self.inner.system_prompt(new_prompt);
        }

        // Register memory tools + inject memory prompt (Phase 9b)
        #[cfg(feature = "memory")]
        {
            self = self.apply_memory_config();
        }

        let mut ab = self.inner;
        let lang = self.language.clone().unwrap_or_default();
        let business_tools = std::mem::take(&mut self.business_tools);
        let error_recovery = self.error_recovery.clone();
        let tool_names = self.tool_names.clone();
        let ma_config = self.multi_agent_config.clone();
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

            if let Some(factory) = self.skill_detail_tool_factory.take() {
                let detail_tool = factory(skill_refs.clone(), self.skill_detail_tool_name);
                ab = ab.register_tool_arc(detail_tool);
            }

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
        if ma_config.as_ref().map(|c| c.enabled).unwrap_or(false) {
            let ma_prompt = build_multi_agent_system_prompt();
            let new_prompt = match self.system_prompt.take() {
                Some(existing) => format!("{}\n\n---\n\n{}", existing, ma_prompt),
                None => ma_prompt,
            };
            ab = ab.system_prompt(new_prompt);
        }

        if let Some(compactor) = self.context_compactor.take() {
            ab = ab.context_compactor(compactor);
        }

        let runtime = ab.build()?;

        let ma_runtime_out = if let Some(config) = ma_config
            && config.enabled
        {
            Some(setup_multi_agent(
                &runtime,
                config,
                lang,
                business_tools,
                error_recovery,
                &tool_names,
                ma_tool_factory,
            )?)
        } else {
            None
        };

        Ok((runtime, ma_runtime_out))
    }
}
