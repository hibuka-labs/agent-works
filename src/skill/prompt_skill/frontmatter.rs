//! PromptSkill: implements `Skill` from a Markdown file with YAML frontmatter.
//!
//! Prompt Skills are lightweight instruction files that get injected into the
//! system prompt to guide AI behavior. Compatible with Trae/Cursor/Claude rules format.
//!
//! Supports the [Agent Skills](https://agentskills.io) open standard
//! (Anthropic 2025.12).

use serde::Deserialize;

// ── Frontmatter schema ──

/// Skill argument/parameter placeholder definition.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillArg {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub required: bool,
}

fn default_true() -> bool {
    true
}

/// Lifetime scope of a skill body once triggered (skill-lifetime spec D1-D3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillScope {
    /// Default: the host injects the body into its system prompt for the rest
    /// of the session (matches the Claude Code ecosystem contract — loaded
    /// skill content stays in context).
    Session,
    /// One-shot: the host injects the body as an ephemeral turn input and
    /// strips it at turn end (the ephemeral-injection path).
    Turn,
}

impl SkillScope {
    /// Parse the raw frontmatter value. Missing/empty/unknown values fall back
    /// to [`SkillScope::Session`] — the ecosystem default; unknown values warn.
    pub fn parse(raw: Option<&str>) -> Self {
        match raw.map(str::trim).filter(|s| !s.is_empty()) {
            None => Self::Session,
            Some("session") => Self::Session,
            Some("turn") => Self::Turn,
            Some(other) => {
                tracing::warn!(
                    value = other,
                    "unknown skill scope in SKILL.md frontmatter, falling back to 'session'"
                );
                Self::Session
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SkillFrontmatter {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub author: Option<String>,

    // ── Agent Skills open-standard fields (agentskills.io) ──
    /// Tools allowed when this skill is active (comma-separated or YAML list).
    /// Empty = all tools allowed.
    #[serde(default, alias = "allowed-tools")]
    pub allowed_tools: Vec<String>,

    /// Tools disallowed when this skill is active.
    #[serde(default, alias = "disallowed-tools")]
    pub disallowed_tools: Vec<String>,

    /// Override model selection. `None` / `"inherit"` = use parent model.
    #[serde(default)]
    pub model: Option<String>,

    /// Whether the user can manually invoke this skill via `/skill-name`.
    #[serde(default = "default_true", alias = "user-invocable")]
    pub user_invocable: bool,

    /// If true, the LLM cannot auto-trigger this skill (for side-effectful skills
    /// like deploy/commit).
    #[serde(default, alias = "disable-model-invocation")]
    pub disable_model_invocation: bool,

    /// Lifetime of the body once triggered. `session` (default): the host
    /// injects it into its system prompt until session end. `turn`: one-shot —
    /// the host injects it as an ephemeral turn input and strips it at turn
    /// end. Raw string; parsed by [`SkillScope::parse`] with warn-on-unknown so
    /// a typo can never fail the whole SKILL.md load.
    #[serde(default)]
    pub scope: Option<String>,

    /// Parameter placeholders (e.g. `[{name: branch, description: "Target branch"}]`).
    /// Body uses `$branch` to reference these.
    #[serde(default)]
    pub arguments: Vec<SkillArg>,

    /// If `"fork"`, the skill runs in an isolated sub-agent context.
    #[serde(default)]
    pub context: Option<String>,

    /// Gitignore-style glob patterns. The skill is only activated when changed
    /// files match these patterns.
    #[serde(default)]
    pub paths: Vec<String>,
}

/// Split a Markdown file into (frontmatter, body) at the `---` delimiters.
pub(super) fn split_frontmatter(content: &str) -> Result<(&str, &str), String> {
    let trimmed = content.trim_start();

    if !trimmed.starts_with("---") {
        return Err("Missing YAML frontmatter (file must start with ---)".to_string());
    }

    // Find the closing ---
    let after_first = &trimmed[3..];
    let closing = after_first
        .find("\n---")
        .or_else(|| after_first.find("\r\n---"))
        .ok_or("Missing closing --- for frontmatter")?;

    let frontmatter_str = &after_first[..closing];
    let body_start = closing + 4; // skip past "\n---"
    let body = after_first[body_start..]
        .trim_start_matches('\n')
        .trim_start_matches('\r');

    Ok((frontmatter_str, body))
}
