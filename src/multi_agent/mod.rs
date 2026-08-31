//! Multi-Agent support for agent-works.
//!
//! This module provides infrastructure for LLM-driven dynamic sub-agent spawning,
//! inter-agent communication, and lifecycle management.
//!
//! # Architecture
//!
//! ```text
//! multi_agent/
//! ├── child.rs          ChildHandle / ChildGuard / ChildOutcome (view layer)
//! ├── child_builder.rs  ChildBuilder (fluent spawn, §5.3 merge rule)
//! ├── child_config.rs   ChildConfig (per-child spawn config)
//! ├── preset.rs         ChildPreset + tool::* name constants (§6.1)
//! ├── config.rs         MultiAgentConfig / ControlConfig / AgentAutonomy
//! ├── registry.rs       AgentRegistry
//! ├── mailbox.rs        Mailbox
//! ├── path.rs           AgentPath
//! ├── budget.rs         RolloutBudget + SpawnTicket (§7.2)
//! ├── limiter.rs        AgentExecutionLimiter (live-concurrency gate, §7.3)
//! ├── control.rs        AgentControl (budget + limiter bundle, §7.1)
//! └── runtime.rs        MultiAgentRuntime
//! ```
//!
//! The 6 LLM tools that use this infrastructure live in `phi-kernel-tools`.

pub mod budget;
pub mod child;
pub mod child_builder;
pub mod child_config;
pub mod config;
pub mod control;
pub mod limiter;
pub mod mailbox;
pub mod path;
pub mod preset;
pub mod registry;
pub mod runtime;

pub use budget::{BudgetError, RolloutBudget, SpawnTicket, usage_total};
pub use child::{ChildGuard, ChildHandle, ChildOutcome};
pub use child_builder::ChildBuilder;
pub use child_config::ChildConfig;
pub use config::{AgentAutonomy, ChildPermissionMode, ControlConfig, MultiAgentConfig};
pub use control::{AgentControl, ControlStatus};
pub use limiter::{AgentExecutionLimiter, ExecutionSlot, LimiterError};
pub use path::AgentPath;
pub use preset::ChildPreset;
pub use runtime::MultiAgentRuntime;
