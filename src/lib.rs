#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "skill")]
pub mod skill;

#[cfg(feature = "builtin-tools")]
pub mod builtin;

#[cfg(feature = "cli")]
pub mod cli;

mod builder;
pub mod handle;

pub use crate::builder::AgentBuilder;
pub use crate::handle::{AgentHandle, SendError};

#[cfg(feature = "skill")]
pub use skill::{
    ApplySkillTool, Skill, SkillParam, SkillParamType, SkillRegistry, SkillSummary,
};
