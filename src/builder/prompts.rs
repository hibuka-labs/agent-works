#[cfg(feature = "multi_agent")]
use std::collections::HashSet;
#[cfg(feature = "multi_agent")]
use std::sync::Arc;

#[cfg(feature = "multi_agent")]
use agent_base::{AgentResult, AgentRuntime, Tool};

#[cfg(feature = "multi_agent")]
use crate::multi_agent::{MultiAgentConfig, MultiAgentRuntime};

#[cfg(feature = "multi_agent")]
use super::MultiAgentToolFactory;

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

You can spawn sub-agents to work on tasks concurrently. Each sub-agent is an independent assistant with its own tools (repo_map, read_file, list_files, execute_command, etc.) and can reason through problems autonomously.

### Tools

- `spawn_agent`: Create a sub-agent. Use the `task` field to describe what you want done — the sub-agent will figure out the steps. Example: `spawn_agent({task_name: "analyze-codex", task: "analyze the codex project structure and output a structured report"})`.
- `send_message`: Send follow-up context or tasks to a sub-agent (`trigger: true` starts execution).
- `list_agents`: List active sub-agents with their assigned task and status — use it to answer "what is agent X doing?".
- `close_agent`: Close a sub-agent when done.

### How Results Reach You

Sub-agent results are **pushed into your context automatically** when they complete — there is no wait tool. To wait, simply end your turn: you will be woken with the result as a new message. Keep working on other things instead of polling.

### How Sub-Agents Work

Sub-agents are capable — they can explore codebases, read files, run commands, and produce structured analysis. You don't need to plan their steps or specify which tools to use. Just describe the goal in `task`, and the sub-agent handles the rest.

### When to Spawn

- Parallel tasks: analyze multiple codebases, compare implementations, etc.
- Tasks you want to delegate while you continue other work

### When NOT to Spawn

- Simple lookups (just use read_file or repo_map directly)
- Tasks that need your current conversation context"#
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

/// Build the memory system prompt from a [`MemoryConfig`]
/// (Phase 9b).
///
/// Renders `config.prompt_template`, substituting the placeholders:
///
/// - `{memory_root}` — the memory directory path
/// - `{index_content}` — the `MEMORY.md` index snapshot (startup state); an
///   empty index renders as a friendly "no memories yet" line
/// - `{tools_description}` — fixed description of the four `memory_*` tools
///
/// The legacy no-arg [`build_memory_system_prompt`] is kept for callers that
/// don't register memory tools (prompt-injection-only mode).
#[cfg(feature = "memory")]
pub fn build_memory_system_prompt_with_config(
    config: &crate::memory::MemoryConfig,
    index_content: &str,
) -> String {
    let index = if index_content.trim().is_empty() {
        "(no memories yet — memory_write creates the first one)".to_string()
    } else {
        index_content.to_string()
    };
    config
        .prompt_template
        .replace("{memory_root}", &config.memory_root.to_string_lossy())
        .replace("{index_content}", &index)
        .replace(
            "{tools_description}",
            crate::tools::MEMORY_TOOLS_DESCRIPTION,
        )
}
