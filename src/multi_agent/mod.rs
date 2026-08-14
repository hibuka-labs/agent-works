//! Multi-Agent support for agent-works.
//!
//! This module provides infrastructure for LLM-driven dynamic sub-agent spawning,
//! inter-agent communication, and lifecycle management.
//!
//! # Architecture
//!
//! ```text
//! multi_agent/
//! ├── config.rs       MultiAgentConfig
//! ├── registry.rs     AgentRegistry
//! ├── mailbox.rs      Mailbox
//! ├── path.rs         AgentPath
//! └── runtime.rs      MultiAgentRuntime
//! ```
//!
//! The 6 LLM tools that use this infrastructure live in `phi-kernel-tools`.

pub mod config;
pub mod mailbox;
pub mod path;
pub mod registry;
pub mod runtime;

pub use config::{ChildPermissionMode, MultiAgentConfig};
pub use path::AgentPath;
pub use runtime::MultiAgentRuntime;
