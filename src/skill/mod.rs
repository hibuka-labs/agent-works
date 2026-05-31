use std::sync::Arc;

use agent_base::Tool;

pub mod detail_tool;
pub mod prompter;

pub use detail_tool::SkillDetailTool;
pub use prompter::{FullDetailPrompter, LazySkillPrompter};

pub trait Skill: Send + Sync {
    fn name(&self) -> &'static str;
    fn brief_description(&self) -> String;
    fn detailed_description(&self) -> String;
    fn tools(&self) -> Vec<Arc<dyn Tool>>;

    fn version(&self) -> &'static str {
        "0.1.0"
    }

    fn tags(&self) -> &[&'static str] {
        &[]
    }

    fn author(&self) -> &'static str {
        ""
    }
}

pub trait SkillPrompter: Send + Sync {
    /// Build the system prompt snippet for the given skills.
    /// `detail_tool_name` is the name of the tool the LLM can call to get
    /// detailed skill descriptions (used by `LazySkillPrompter`).
    fn build_prompt(&self, skills: &[Arc<dyn Skill>], detail_tool_name: &str) -> String;
}
