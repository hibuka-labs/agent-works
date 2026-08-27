//! Composable prompt fragments for building system prompts.
//!
//! Each [`PromptFragment`] owns one concern (role definition, tool descriptions,
//! environment info, etc.) and renders itself independently. Fragments are
//! sorted by [`priority`](PromptFragment::priority) and concatenated by
//! [`compose_fragments`] to produce the final system prompt.
//!
//! # Architecture
//!
//! - **agent-works** (this module): trait definition + generic fragments
//!   (`EnvironmentFragment`, `DynamicToolsFragment`).
//! - **phi-agent**: application-specific fragments (`CoreInstructionsFragment`,
//!   `FocusFragment`, etc.) and the `build_system_prompt()` assembly entry point.
//! - **Consumers**: inject custom fragments via `build_system_prompt_with_fragments()`.

mod fragment;

pub use fragment::{FragmentContext, PromptFragment, compose_fragments};

// ── Generic fragments provided by agent-works ──────────────────────────────

/// Injects runtime environment information: OS, working directory, git branch.
///
/// Priority: 50 (middle — after core instructions, before tool descriptions).
#[derive(Clone)]
pub struct EnvironmentFragment {
    pub os: String,
    pub cwd: String,
    pub git_branch: Option<String>,
}

impl EnvironmentFragment {
    /// Auto-detect environment from the current process.
    pub fn detect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            cwd: std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            git_branch: detect_git_branch(),
        }
    }
}

impl PromptFragment for EnvironmentFragment {
    fn name(&self) -> &str {
        "environment"
    }

    fn priority(&self) -> i32 {
        50
    }

    fn render(&self, _ctx: &FragmentContext) -> Option<String> {
        let mut lines = vec![
            format!("[Environment]\n- OS: {}", self.os),
            format!("- Working directory: {}", self.cwd),
        ];
        if let Some(ref branch) = self.git_branch {
            lines.push(format!("- Git branch: {}", branch));
        }
        Some(lines.join("\n"))
    }
}

/// Dynamically generate tool descriptions from registered tools.
///
/// Iterates `ctx.tool_definitions` and produces a `[Tools]` section listing
/// each tool's name and description. Useful for consumers that want tool
/// documentation in the system prompt without hardcoding it.
///
/// Priority: 70 (after workflow/safety, before memory).
#[derive(Clone)]
pub struct DynamicToolsFragment;

impl PromptFragment for DynamicToolsFragment {
    fn name(&self) -> &str {
        "dynamic_tools"
    }

    fn priority(&self) -> i32 {
        70
    }

    fn render(&self, ctx: &FragmentContext) -> Option<String> {
        if ctx.tool_definitions.is_empty() {
            return None;
        }
        let mut lines = vec!["[Available Tools]".to_string()];
        for def in ctx.tool_definitions {
            let name = def
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("unknown");
            let desc = def
                .get("function")
                .and_then(|f| f.get("description"))
                .and_then(|d| d.as_str())
                .unwrap_or("");
            if desc.is_empty() {
                lines.push(format!("- `{}`", name));
            } else {
                lines.push(format!("- `{}` — {}", name, desc));
            }
        }
        Some(lines.join("\n"))
    }
}

/// Helper: detect current git branch via `git rev-parse`.
fn detect_git_branch() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_fragment_render() {
        let frag = EnvironmentFragment {
            os: "macos".into(),
            cwd: "/Users/test".into(),
            git_branch: Some("main".into()),
        };
        let ctx = FragmentContext {
            tool_definitions: &[],
            session_id: "s1",
        };
        let output = frag.render(&ctx).unwrap();
        assert!(output.contains("OS: macos"));
        assert!(output.contains("/Users/test"));
        assert!(output.contains("Git branch: main"));
    }

    #[test]
    fn test_environment_fragment_no_git() {
        let frag = EnvironmentFragment {
            os: "linux".into(),
            cwd: "/tmp".into(),
            git_branch: None,
        };
        let ctx = FragmentContext {
            tool_definitions: &[],
            session_id: "s1",
        };
        let output = frag.render(&ctx).unwrap();
        assert!(output.contains("OS: linux"));
        assert!(!output.contains("Git branch"));
    }

    #[test]
    fn test_dynamic_tools_fragment_with_tools() {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {}
            }
        });
        let frag = DynamicToolsFragment;
        let ctx = FragmentContext {
            tool_definitions: &[tool_def],
            session_id: "test",
        };
        let output = frag.render(&ctx).unwrap();
        assert!(output.contains("read_file"));
        assert!(output.contains("Read a file from disk"));
    }

    #[test]
    fn test_dynamic_tools_fragment_empty() {
        let frag = DynamicToolsFragment;
        let ctx = FragmentContext {
            tool_definitions: &[],
            session_id: "test",
        };
        assert!(frag.render(&ctx).is_none());
    }

    #[test]
    fn test_compose_orders_by_priority() {
        let env = EnvironmentFragment {
            os: "linux".into(),
            cwd: "/tmp".into(),
            git_branch: None,
        };
        let tool_def = serde_json::json!({
            "type": "function",
            "function": { "name": "test_tool", "description": "A test", "parameters": {} }
        });
        let fragments: Vec<Box<dyn PromptFragment>> = vec![
            Box::new(DynamicToolsFragment), // priority 70
            Box::new(env),                  // priority 50
        ];
        let ctx = FragmentContext {
            tool_definitions: &[tool_def],
            session_id: "test",
        };
        let result = compose_fragments(&fragments, &ctx);
        let env_pos = result.find("[Environment]").unwrap();
        let tools_pos = result.find("[Available Tools]").unwrap();
        assert!(
            env_pos < tools_pos,
            "Environment (50) should come before Tools (70)"
        );
    }
}
