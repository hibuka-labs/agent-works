# agent-works

[![crates.io](https://img.shields.io/crates/v/agent-works.svg)](https://crates.io/crates/agent-works)
[![Documentation](https://docs.rs/agent-works/badge.svg)](https://docs.rs/agent-works)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![codecov](https://codecov.io/gh/hibuka-labs/agent-works/branch/master/graph/badge.svg)](https://codecov.io/gh/hibuka-labs/agent-works)

**Batteries-included Agent toolbox built on [agent-base](https://github.com/hibuka-labs/agent-base).**

`agent-works` adds production-ready capabilities on top of the `agent-base` runtime kernel: loop guards for model misbehavior, MCP multi-server management, Skills with progressive disclosure, a Focus module for structured LLM extraction, multi-agent orchestration with fork_history, and a CLI REPL loop — all behind feature flags. Pick what you need.

## Relationship with agent-base

```
agent-base         Pure runtime kernel (~12 deps, trait interfaces only)
    ↑
agent-works        Batteries-included toolbox (wraps agent-base + enhancements)
```

- **Use `agent-base` alone** when you only need the runtime (LLM + tools + middleware).
- **Use `agent-works`** when you want MCP, Skills, Focus, multi-agent, and CLI — and still get everything from agent-base through re-exports.
- Switching from `agent-base` to `agent-works` is a one-line import change.

## Installation

```toml
[dependencies]
agent-works = { version = "0.1.7", features = ["full"] }
```

Or pick specific features:

```toml
agent-works = { version = "0.1.7", features = ["mcp", "skill"] }
```

## Feature Flags

| Feature | Description | Extra deps |
|---------|-------------|------------|
| `mcp` | `McpHUb` — multi-server MCP with HTTP + stdio transport | — |
| `skill` | `Skill` trait + `LazySkillPrompter` / `FullDetailPrompter` + `SkillDetailTool` + `SkillLoader` | — |
| `prompt_skill` | `PromptSkill` — skill definitions from prompt files | `serde_yaml` |
| `yaml_skill` | `YamlSkill` — skill definitions from YAML files | `serde_yaml` |
| `hot-reload` | Hot-reload skill definitions on file change | `notify`, `prompt_skill` |
| `cli` | `CliRepl` (generic REPL loop) + `CliEventPrinter` (terminal event output) | — |
| `full` | All of the above | — |

All types from `agent-base` are re-exported (`AgentBuilder`, `AgentRuntime`, `Tool`, `Middleware`, ...), so you only need to depend on `agent-works`.

## Quick Start

### Skills

Skills package tools + descriptions into reusable units with progressive disclosure:

```rust
use std::sync::Arc;
use agent_base::{AgentResult, Content, Tool, ToolContext};
use agent_works::{
    AgentBuilder,
    skill::{Skill, LazySkillPrompter},
};
use async_trait::async_trait;
use serde_json::{json, Value};

// 1. Define tools
struct AddTool;
#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &'static str { "add" }

    fn description(&self) -> &'static str {
        "Calculate the sum of two integers"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "integer" }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        Ok(vec![Content::text(format!("{a} + {b} = {}", a + b))])
    }
}

// 2. Pack into a Skill
struct MathSkill;
impl Skill for MathSkill {
    fn name(&self) -> &'static str { "math" }
    fn brief_description(&self) -> String {
        "Math: supports addition".to_string()
    }
    fn detailed_description(&self) -> String {
        "## Math Skill\n\n- **add**: Calculate the sum of two integers".to_string()
    }
    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![Arc::new(AddTool)]
    }
}

// 3. Build with agent-works AgentBuilder
let runtime = AgentBuilder::new(llm)
    .system_prompt("You are a helpful assistant.")
    .register_skill(MathSkill)  // auto-registers tools, injects prompt, adds detail tool
    .build()?;
```

The builder automatically:
- Registers skill tools and detects name conflicts
- Injects skill brief descriptions into the system prompt (via `LazySkillPrompter`)
- Registers `SkillDetailTool` for on-demand detailed prompt loading

### Focus — Structured LLM Extraction

`Focus` provides a clean API for extracting structured data from LLM responses:

```rust
use std::sync::Arc;
use std::time::Duration;
use agent_works::focus::Focus;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct TaskStatus { status: String, priority: u8 }

let focus = Focus::new(
    client,  // Arc<dyn StreamClient>
    "You are a task classifier. Output valid JSON matching the schema.",
);

let output = focus
    .ask::<TaskStatus>("Classify: 'deploy hotfix to production'", Duration::from_secs(5))
    .await?;

println!("Status: {}, Priority: {}", output.result.status, output.result.priority);
```

### Multi-Agent with fork_history

Spawn child agents that inherit conversation context:

```rust
use agent_works::{MultiAgentRuntime, MultiAgentConfig};

let mut runtime = MultiAgentRuntime::new(client, MultiAgentConfig::enabled());

// Spawn a child agent with full parent history
let child_id = runtime
    .spawn_child_with_history(
        "math-expert",
        "gpt-4o",
        "You are a math expert.",
        "all",  // fork_history: "none" | "all" | N (last N turns)
        None,   // reasoning_effort
        None,   // agent_type
    )
    .await?;

// Send a message and collect the result
let events = runtime.send_input(child_id, "What is 2+2?").await?;
```

`AgentHandle` provides a higher-level wrapper for agent lifecycle management:

```rust
use agent_works::AgentHandle;

let handle = AgentHandle::spawn(runtime, "researcher", "gpt-4o", "You are a researcher.")?;
handle.send("Research the history of Rust.").await?;
// Events stream from handle.events()
```

### MCP Multi-Server

```rust
use agent_works::mcp::*;

let mut hub = McpHUb::new();
hub.add_server(McpServerConfig {
    name: "filesystem".into(),
    transport: McpTransport::Stdio {
        command: "npx".into(),
        args: vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into()],
    },
    auto_reconnect: true,
});
hub.connect_all().await?;

// Discover tools from all servers
let all_tools = hub.discover_all().await?;

// Register into the agent runtime
let mut tools = runtime.tools_mut();
hub.register_all(&mut tools);
```

### CLI REPL

```rust
use agent_works::cli::{CliRepl, CliEventPrinter};

// Default (stdout)
let mut printer = CliEventPrinter::new();

// Or capture output for testing
let mut printer = CliEventPrinter::with_writer(Vec::new());

let mut repl = CliRepl::new(runtime);

// Register custom shell commands
repl.register_shell_command("time", Box::new(|_| {
    println!(">>> {}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
    true
}));

repl.run().await?;
```

### Tool Enforcement

The `ToolEnforcementMiddleware` (inherited from agent-base) nudges the LLM to actually call tools instead of just describing what it would do:

```rust
use agent_works::ToolEnforcementMiddleware;
use agent_works::ToolEnforcementConfig;

let runtime = AgentBuilder::new(llm)
    .register_tool(MyTool)
    .middleware(ToolEnforcementMiddleware::new(ToolEnforcementConfig::default()))
    .build()?;
```

### Guard — Loop Protection

Guards protect the agent loop from model misbehavior (reasoning-only responses, empty responses, incomplete answers). Without a guard the runtime still works; with one it's smarter.

```rust
use agent_works::guard::{DefaultGuard, DefaultGuardConfig};

// No guard — NoopGuard injected automatically, no intervention
let runtime = AgentBuilder::new(llm).build()?;

// DefaultGuard with defaults — handles reasoning_only, empty_response, text_only
let runtime = AgentBuilder::new(llm)
    .guard(DefaultGuard::new(DefaultGuardConfig::default()))
    .build()?;

// DefaultGuard with LLM judge — verifies task completion on text-only responses
let config = DefaultGuardConfig {
    use_llm_judge: true,
    judge_fail_open: true,  // trust model if judge fails
    ..Default::default()
};
let runtime = AgentBuilder::new(llm.clone())
    .guard(DefaultGuard::with_llm_client(config, llm))
    .build()?;
```

Custom guards implement the `ReactLoopGuard` trait:

```rust
use agent_works::guard::{GuardCtx, GuardDecision, ReactLoopGuard};

struct StrictGuard;

#[async_trait]
impl ReactLoopGuard for StrictGuard {
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision {
        if !ctx.run_has_tool_calls {
            return GuardDecision::Fail { error: "no tool calls".into() };
        }
        GuardDecision::Complete
    }
}
```

## Examples

```bash
# Guard system — DefaultGuard, NoopGuard, custom guards
cargo run --example guard_demo

# Skills with progressive disclosure
cargo run --example skill_demo --features skill

# MCP multi-server connection
cargo run --example mcp_demo --features mcp

# CLI REPL + event printer
cargo run --example cli_demo --features cli
```

## Module Structure

```
src/
├── lib.rs              # Re-exports agent-base + feature-gated modules
├── builder.rs          # AgentBuilder wrapper with skill integration
├── handle.rs           # AgentHandle — high-level agent lifecycle
├── guard/              # DefaultGuard + ReactLoopGuard trait
├── mcp/                # McpHUb + McpClient (HTTP + stdio transport)
├── skill/              # Skill trait + prompter strategies + detail tool
├── focus/              # Focus — structured LLM extraction
├── multi_agent/        # MultiAgentRuntime + fork_history support
└── cli/                # CliRepl + CliEventPrinter<W>
```

## License

MIT
