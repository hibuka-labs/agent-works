#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "skill")]
pub mod skill;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "focus")]
pub mod focus;

#[cfg(feature = "multi_agent")]
pub mod multi_agent;

#[cfg(feature = "compression")]
pub mod compression;

mod builder;
pub mod handle;

#[cfg(feature = "skill")]
pub use crate::builder::SkillDetailToolFactory;
pub use crate::builder::{
    AgentBuilder, build_memory_system_prompt,
};
#[cfg(feature = "multi_agent")]
pub use crate::builder::{
    MultiAgentToolFactory, build_multi_agent_system_prompt, setup_multi_agent,
};
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
