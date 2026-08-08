#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "skill")]
pub mod skill;

#[cfg(feature = "builtin-tools")]
pub mod builtin;

#[cfg(feature = "cli")]
pub mod cli;

pub mod focus;
pub mod multi_agent;

mod builder;
pub mod handle;

pub use crate::builder::{
    AgentBuilder, MultiAgentToolFactory, build_memory_system_prompt,
    build_multi_agent_system_prompt, setup_multi_agent,
};
#[cfg(feature = "skill")]
pub use crate::builder::SkillDetailToolFactory;
pub use crate::handle::{AgentHandle, SendError};

#[cfg(feature = "skill")]
pub use skill::{
    FullDetailPrompter, LazySkillPrompter, Skill, SkillParam, SkillParamType, SkillRegistry,
    SkillSummary,
};

#[cfg(feature = "prompt_skill")]
pub use skill::prompt_skill::PromptSkill;

#[cfg(feature = "yaml_skill")]
pub use skill::yaml_skill::YamlSkill;
