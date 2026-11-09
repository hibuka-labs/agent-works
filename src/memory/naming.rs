use std::path::Path;

use agent_base::{AgentError, AgentResult};

use super::MEMORY_TYPES;

// ─────────────────────────────────────────────────────────────────────────────
// Project slug
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a workspace path into a Claude Code style project slug.
///
/// `/Users/xxx/my.project` -> `-Users-xxx-my-project`: trailing `/` stripped,
/// every non-alphanumeric character becomes `-` (matching Claude Code).
/// Slugs longer than 200 chars are truncated. The slug embeds the user name and the absolute path, so
/// memories are naturally isolated per machine and per workspace. Bound to the
/// startup cwd (not the git root), matching Claude Code's behavior.
pub fn project_slug(cwd: &Path) -> String {
    let s = cwd.to_string_lossy();
    let trimmed = s.trim_end_matches('/');
    // Match Claude Code: every non-alphanumeric char becomes `-`
    let slug: String = trimmed
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    // Claude Code truncates at 200 chars
    if slug.len() > 200 {
        slug[..200].to_string()
    } else {
        slug
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Name validation (path-escape guard)
// ─────────────────────────────────────────────────────────────────────────────

/// Check a memory name against `^[a-z0-9][a-z0-9-]*$`.
///
/// The name is joined into a file path (`<name>.md`), so anything outside this
/// charset (slashes, dots, underscores, uppercase) is rejected *before* any
/// path is built — `../../etc/passwd` cannot escape the memory directory.
pub fn is_valid_memory_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Validate a memory name, returning a descriptive error for the LLM.
pub fn validate_memory_name(name: &str) -> AgentResult<()> {
    if is_valid_memory_name(name) {
        Ok(())
    } else {
        Err(AgentError::internal(format!(
            "invalid memory name `{name}`: must match ^[a-z0-9][a-z0-9-]*$ (kebab-case slug, no slashes/dots/underscores)"
        )))
    }
}

/// Validate a memory type against the four allowed values.
pub fn validate_memory_type(memory_type: &str) -> AgentResult<()> {
    if MEMORY_TYPES.contains(&memory_type) {
        Ok(())
    } else {
        Err(AgentError::internal(format!(
            "invalid memory type `{memory_type}`: must be one of {}",
            MEMORY_TYPES.join(" | ")
        )))
    }
}
