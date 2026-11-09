//! PromptSkill: implements `Skill` from a Markdown file with YAML frontmatter.
//!
//! Prompt Skills are lightweight instruction files that get injected into the
//! system prompt to guide AI behavior. Compatible with Trae/Cursor/Claude rules format.
//!
//! Supports the [Agent Skills](https://agentskills.io) open standard
//! (Anthropic 2025.12).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::skill::{Skill, SkillParam};

mod frontmatter;
use frontmatter::split_frontmatter;
pub use frontmatter::{SkillArg, SkillFrontmatter, SkillScope};

// ── PromptSkill ──

/// A skill loaded from a Markdown file with YAML frontmatter.
///
/// Implements `Skill` — can be registered into `SkillRegistry`
/// and used for system prompt injection.
///
/// Supports two loading modes:
/// - `from_markdown(content)` — parse a single SKILL.md string (no directory context)
/// - `from_dir(path)` — load from a standard skill directory with SKILL.md,
///   optional scripts/, references/, and templates/ subdirectories
///
/// ## Memory note
///
/// `PromptSkill` uses `Box::leak` to convert `String` fields into `&'static str`
/// references required by the `Skill` trait. This is acceptable for one-time
/// startup loading (a few KB of "leaked" memory), but is **not** compatible with
/// frequent hot-reload cycles — each reload adds new strings to the leaked pool.
/// If you use `hot-reload`, prefer event-triggered rather than timer-based reloading.
#[derive(Debug)]
pub struct PromptSkill {
    frontmatter: SkillFrontmatter,
    /// The full Markdown content (including frontmatter)
    content: String,
    /// The Markdown body (excluding frontmatter)
    body: String,
    /// Directory on disk (set via `from_dir()`)
    skill_dir: Option<PathBuf>,
    /// Path to SKILL.md file (set via `from_dir()`)
    source_path: Option<PathBuf>,
    /// Leaked name for &'static str return
    static_name: &'static str,
    /// Leaked version for &'static str return
    static_version: &'static str,
    /// Leaked category for &'static str return
    static_category: &'static str,
    /// Leaked author for &'static str return
    static_author: &'static str,
    /// Leaked tags for &'static [&'static str] return
    static_tags: Vec<&'static str>,
    /// Leaked allowed_tools
    static_allowed_tools: Vec<String>,
    /// Leaked disallowed_tools
    static_disallowed_tools: Vec<String>,
    /// Leaked paths
    static_paths: Vec<String>,
}

impl PromptSkill {
    /// Parse a Markdown string with YAML frontmatter into a `PromptSkill`.
    ///
    /// Expected format:
    /// ```markdown
    /// ---
    /// name: my-skill
    /// description: A short description
    /// category: 运维
    /// tags: [tag1, tag2]
    /// version: "1.0"
    /// author: ops-agent
    /// ---
    ///
    /// # Skill content here
    /// ```
    pub fn from_markdown(content: &str) -> Result<Self, String> {
        let (frontmatter_str, body) = split_frontmatter(content)?;

        let frontmatter: SkillFrontmatter = serde_yaml::from_str(frontmatter_str)
            .map_err(|e| format!("Frontmatter parse error: {e}"))?;

        if frontmatter.name.is_empty() {
            return Err("Skill name is empty".to_string());
        }

        if frontmatter.description.is_empty() {
            return Err("Skill description is empty".to_string());
        }

        // Validate name format (alphanumeric + hyphens)
        if !frontmatter
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(format!(
                "Invalid skill name '{}': only alphanumeric, hyphens, and underscores allowed",
                frontmatter.name
            ));
        }

        let static_name: &'static str = Box::leak(frontmatter.name.clone().into_boxed_str());
        let static_version: &'static str = Box::leak(
            frontmatter
                .version
                .clone()
                .unwrap_or_else(|| "1.0".to_string())
                .into_boxed_str(),
        );
        let static_category: &'static str =
            Box::leak(frontmatter.category.clone().into_boxed_str());
        let static_author: &'static str = Box::leak(
            frontmatter
                .author
                .clone()
                .unwrap_or_default()
                .into_boxed_str(),
        );
        let static_tags: Vec<&'static str> = frontmatter
            .tags
            .iter()
            .map(|t| Box::leak(t.clone().into_boxed_str()) as &'static str)
            .collect();
        let static_allowed_tools: Vec<String> = frontmatter.allowed_tools.clone();
        let static_disallowed_tools: Vec<String> = frontmatter.disallowed_tools.clone();
        let static_paths: Vec<String> = frontmatter.paths.clone();

        Ok(Self {
            frontmatter,
            content: content.to_string(),
            body: body.to_string(),
            skill_dir: None,
            source_path: None,
            static_name,
            static_version,
            static_category,
            static_author,
            static_tags,
            static_allowed_tools,
            static_disallowed_tools,
            static_paths,
        })
    }

    /// The raw frontmatter (for introspection).
    pub fn frontmatter(&self) -> &SkillFrontmatter {
        &self.frontmatter
    }

    /// Lifetime scope once triggered (frontmatter `scope:`; missing/unknown
    /// values fall back to `session`).
    pub fn scope(&self) -> SkillScope {
        SkillScope::parse(self.frontmatter.scope.as_deref())
    }

    /// The Markdown body (without frontmatter).
    pub fn body(&self) -> &str {
        &self.body
    }

    /// The skill directory on disk, if loaded via `from_dir()`.
    pub fn skill_dir_path(&self) -> Option<&Path> {
        self.skill_dir.as_deref()
    }

    /// Path to the `scripts/` subdirectory, if it exists.
    pub fn scripts_dir(&self) -> Option<PathBuf> {
        self.skill_dir
            .as_ref()
            .map(|d| d.join("scripts"))
            .filter(|p| p.is_dir())
    }

    /// Path to the `references/` subdirectory, if it exists.
    pub fn references_dir(&self) -> Option<PathBuf> {
        self.skill_dir
            .as_ref()
            .map(|d| d.join("references"))
            .filter(|p| p.is_dir())
    }

    /// Path to the `templates/` subdirectory, if it exists.
    pub fn templates_dir(&self) -> Option<PathBuf> {
        self.skill_dir
            .as_ref()
            .map(|d| d.join("templates"))
            .filter(|p| p.is_dir())
    }

    /// Load a skill from a standard directory structure.
    ///
    /// Expected layout:
    /// ```text
    /// skill-name/
    ///   SKILL.md          (required — YAML frontmatter + Markdown body)
    ///   scripts/          (optional)
    ///   references/       (optional)
    ///   templates/        (optional)
    /// ```
    ///
    /// The directory name must match the `name` field in SKILL.md frontmatter.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let skill_md = path.join("SKILL.md");

        if !skill_md.exists() {
            return Err(format!("SKILL.md not found in {}", path.display()));
        }

        let content = std::fs::read_to_string(&skill_md)
            .map_err(|e| format!("Failed to read {}: {}", skill_md.display(), e))?;

        let mut skill = Self::from_markdown(&content)?;

        // Validate directory name matches skill name
        if let Some(dir_name) = path.file_name().and_then(|n| n.to_str())
            && dir_name != skill.frontmatter.name
        {
            return Err(format!(
                "Directory name '{}' does not match skill name '{}'",
                dir_name, skill.frontmatter.name
            ));
        }

        skill.skill_dir = Some(path.to_path_buf());
        skill.source_path = Some(path.join("SKILL.md"));
        Ok(skill)
    }

    /// Scan a directory for skill subdirectories.
    ///
    /// Each subdirectory that contains a `SKILL.md` file is loaded as a
    /// [`PromptSkill`]. Subdirectories without `SKILL.md` are silently skipped.
    /// Subdirectories whose name starts with `_` are treated as disabled by
    /// user convention and silently skipped, even if they contain a SKILL.md.
    ///
    /// Returns an empty `Vec` if the directory does not exist.
    pub fn scan_dir(path: impl AsRef<Path>) -> Result<Vec<Self>, String> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Ok(Vec::new());
        }

        let mut skills = Vec::new();
        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory {}: {}", path.display(), e))?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let entry_path = entry.path();
            // `_`-prefixed directories are a user convention for disabled/hidden
            // skills (users share `~/.claude/skills` with other agents and park
            // skills there by renaming) — skip silently, no warning.
            if entry_path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('_'))
            {
                continue;
            }
            if entry_path.is_dir() && entry_path.join("SKILL.md").exists() {
                match Self::from_dir(&entry_path) {
                    Ok(skill) => skills.push(skill),
                    Err(e) => {
                        tracing::warn!(
                            dir = %entry_path.display(),
                            error = %e,
                            "skipping invalid skill directory"
                        );
                    }
                }
            }
        }

        // Sort by name for deterministic ordering
        skills.sort_by(|a, b| a.frontmatter.name.cmp(&b.frontmatter.name));
        Ok(skills)
    }

    /// Resolve variables in the skill body.
    ///
    /// Supported variables:
    /// - `$ARGUMENTS` — raw text after `/skill-name`
    /// - `$name` — named parameter value (from frontmatter `arguments` list)
    /// - `$PHI_SKILL_DIR` — path to the skill directory (for referencing scripts/)
    pub fn resolve_body(&self, params: &HashMap<String, String>, raw_arguments: &str) -> String {
        let mut result = self.body.clone();

        // $ARGUMENTS
        result = result.replace("$ARGUMENTS", raw_arguments);

        // $PHI_SKILL_DIR
        if let Some(dir) = &self.skill_dir {
            result = result.replace("$PHI_SKILL_DIR", &dir.to_string_lossy());
        }

        // Named parameters: $param_name
        // Sort by key length descending to prevent prefix clobbering.
        // Example: with keys ["host", "hostname"], $hostname must be substituted
        // before $host, otherwise $hostname becomes "Xname" instead of "Y".
        let mut sorted_keys: Vec<&String> = params.keys().collect();
        sorted_keys.sort_by_key(|b| std::cmp::Reverse(b.len()));
        for key in sorted_keys {
            let value = &params[key];
            let placeholder = format!("${}", key);
            result = result.replace(&placeholder, value);
        }

        result
    }
}

impl Skill for PromptSkill {
    fn name(&self) -> &'static str {
        self.static_name
    }

    fn brief_description(&self) -> String {
        self.frontmatter.description.clone()
    }

    fn detailed_description(&self) -> String {
        self.content.clone()
    }

    fn tools(&self) -> Vec<Arc<dyn agent_base::Tool>> {
        vec![]
    }

    fn parameters(&self) -> &[SkillParam] {
        &[]
    }

    fn version(&self) -> &'static str {
        self.static_version
    }

    fn tags(&self) -> &[&'static str] {
        &self.static_tags
    }

    fn author(&self) -> &'static str {
        self.static_author
    }

    fn category(&self) -> &'static str {
        self.static_category
    }

    fn allowed_tools(&self) -> &[String] {
        &self.static_allowed_tools
    }

    fn disallowed_tools(&self) -> &[String] {
        &self.static_disallowed_tools
    }

    fn model_override(&self) -> Option<&str> {
        self.frontmatter
            .model
            .as_deref()
            .filter(|m| *m != "inherit")
    }

    fn is_user_invocable(&self) -> bool {
        self.frontmatter.user_invocable
    }

    fn disable_model_invocation(&self) -> bool {
        self.frontmatter.disable_model_invocation
    }

    fn context_mode(&self) -> Option<&str> {
        self.frontmatter.context.as_deref()
    }

    fn path_patterns(&self) -> &[String] {
        &self.static_paths
    }

    fn skill_dir(&self) -> Option<&Path> {
        self.skill_dir.as_deref()
    }

    fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    fn read_reference(&self, relative_path: &str) -> Result<String, String> {
        let ref_dir = self.references_dir().ok_or_else(|| {
            format!(
                "Skill '{}' has no references/ directory",
                self.frontmatter.name
            )
        })?;

        // Prevent path traversal: canonicalize both paths for reliable comparison
        // (symlinks, case-insensitive FS, etc.)
        let ref_dir_canon = ref_dir
            .canonicalize()
            .map_err(|e| format!("Failed to resolve references/ directory: {}", e))?;

        let resolved = ref_dir.join(relative_path);
        let canonical = resolved.canonicalize().map_err(|e| {
            format!(
                "Failed to resolve reference path '{}': {}",
                relative_path, e
            )
        })?;

        if !canonical.starts_with(&ref_dir_canon) {
            return Err(format!(
                "Path traversal denied: '{}' is outside references/",
                relative_path
            ));
        }

        std::fs::read_to_string(&canonical)
            .map_err(|e| format!("Failed to read reference '{}': {}", relative_path, e))
    }
}

// ── Fuzz exports ──
#[cfg(feature = "fuzzing")]
pub mod fuzz_exports {
    pub fn split_frontmatter(content: &str) -> Result<(&str, &str), String> {
        super::split_frontmatter(content)
    }
}

#[cfg(test)]
mod tests;
