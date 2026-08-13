use std::sync::Arc;

use super::{Skill, SkillPrompter};

pub struct LazySkillPrompter {
    title: String,
    instruction_template: String,
    item_prefix: String,
}

impl Default for LazySkillPrompter {
    fn default() -> Self {
        Self {
            title: "## Available Skills".to_string(),
            instruction_template:
                "> Use `read_file` with the file path to read the full skill instructions."
                    .to_string(),
            item_prefix: "- **".to_string(),
        }
    }
}

impl LazySkillPrompter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set a custom instruction line, emitted verbatim at the end of the prompt.
    ///
    /// In read_file (prompt-injection) mode the default tells the LLM to read
    /// the skill's `SKILL.md` via `read_file`. The instruction is not
    /// placeholder-expanded.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction_template = instruction.into();
        self
    }

    pub fn item_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.item_prefix = prefix.into();
        self
    }
}

impl SkillPrompter for LazySkillPrompter {
    fn build_prompt(&self, skills: &[Arc<dyn Skill>], _detail_tool_name: &str) -> String {
        if skills.is_empty() {
            return String::new();
        }

        let mut prompt = String::new();
        prompt.push_str(&self.title);
        prompt.push('\n');

        for skill in skills {
            let file_hint = if let Some(path) = skill.source_path() {
                format!(" → `{}`", path.display())
            } else {
                String::new()
            };
            prompt.push_str(&format!(
                "{}{}**{}**: {}{}\n",
                self.item_prefix,
                "", // placeholder prefix for future categorization
                skill.name(),
                skill.brief_description(),
                file_hint,
            ));
        }

        prompt.push('\n');
        prompt.push_str(&self.instruction_template);

        prompt
    }
}

pub struct FullDetailPrompter;

impl SkillPrompter for FullDetailPrompter {
    fn build_prompt(&self, skills: &[Arc<dyn Skill>], _detail_tool_name: &str) -> String {
        tracing::debug!("skill prompt generated");
        if skills.is_empty() {
            return String::new();
        }

        let mut prompt = String::from("## Available Skills\n\n");
        for skill in skills {
            prompt.push_str(&format!("### {}\n\n", skill.name()));
            prompt.push_str(&skill.brief_description());
            prompt.push_str("\n\n");
            prompt.push_str(&skill.detailed_description());
            prompt.push_str("\n\n---\n\n");
        }
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use agent_base::Tool;

    struct TestSkill {
        name: &'static str,
        source: Option<PathBuf>,
    }

    impl TestSkill {
        fn new(name: &'static str) -> Self {
            Self { name, source: None }
        }

        fn with_source(mut self, path: &str) -> Self {
            self.source = Some(PathBuf::from(path));
            self
        }
    }

    impl Skill for TestSkill {
        fn name(&self) -> &'static str {
            self.name
        }

        fn brief_description(&self) -> String {
            format!("{} brief", self.name)
        }

        fn detailed_description(&self) -> String {
            format!("{} detailed", self.name)
        }

        fn tools(&self) -> Vec<Arc<dyn Tool>> {
            vec![]
        }

        fn source_path(&self) -> Option<&std::path::Path> {
            self.source.as_deref()
        }
    }

    fn skill(name: &'static str) -> Arc<dyn Skill> {
        Arc::new(TestSkill::new(name))
    }

    #[test]
    fn test_lazy_prompter_empty() {
        let p = LazySkillPrompter::new();
        assert_eq!(p.build_prompt(&[], "read_file"), "");
    }

    #[test]
    fn test_lazy_prompter_builds_items() {
        let p = LazySkillPrompter::new();
        let skills: Vec<Arc<dyn Skill>> = vec![skill("deploy"), skill("commit")];
        let out = p.build_prompt(&skills, "read_file");
        assert!(out.contains("## Available Skills"));
        assert!(out.contains("**deploy**"));
        assert!(out.contains("**commit**"));
        assert!(out.contains("read_file"));
    }

    #[test]
    fn test_lazy_prompter_source_path_hint() {
        let p = LazySkillPrompter::new();
        let s: Arc<dyn Skill> = Arc::new(TestSkill::new("deploy").with_source("deploy/SKILL.md"));
        let out = p.build_prompt(&[s], "read_file");
        assert!(out.contains("deploy/SKILL.md"));
    }

    #[test]
    fn test_lazy_prompter_builders() {
        let p = LazySkillPrompter::new()
            .title("## Custom")
            .instruction("read custom")
            .item_prefix("* ");
        let out = p.build_prompt(&[skill("x")], "read_file");
        assert!(out.contains("## Custom"));
        assert!(out.contains("* **x**"));
        assert!(out.contains("read custom"));
    }

    #[test]
    fn test_full_detail_prompter_empty() {
        let p = FullDetailPrompter;
        assert_eq!(p.build_prompt(&[], "x"), "");
    }

    #[test]
    fn test_full_detail_prompter_builds() {
        let p = FullDetailPrompter;
        let skills: Vec<Arc<dyn Skill>> = vec![skill("deploy")];
        let out = p.build_prompt(&skills, "x");
        assert!(out.contains("### deploy"));
        assert!(out.contains("deploy brief"));
        assert!(out.contains("deploy detailed"));
    }
}
