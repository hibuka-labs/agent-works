use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use agent_base::{AgentError, AgentResult};

use super::frontmatter::{parse_memory_file, render_memory_file_merged};
use super::{
    DEFAULT_INDEX_FILENAME, atomic_write, index_add, index_remove, now_iso8601, project_slug,
    validate_memory_name, validate_memory_type,
};

// ─────────────────────────────────────────────────────────────────────────────
// MemoryConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Memory configuration (consumer-injected policy, Phase 9b).
///
/// The framework owns format + tools + index sync; the consumer decides where
/// memories live and how they are pitched in the system prompt.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Memory root directory, e.g. `~/.claude/projects/<slug>/memory/`.
    pub memory_root: PathBuf,
    /// Index file name (default `"MEMORY.md"`).
    pub index_filename: String,
    /// System prompt template. Placeholders substituted at build time:
    /// `{memory_root}`, `{index_content}`, `{tools_description}`.
    pub prompt_template: String,
}

/// Default prompt template for Claude Code compatible deployments.
pub const CLAUDE_COMPATIBLE_TEMPLATE: &str = r#"## Memory

You have a persistent memory directory at `{memory_root}`. Memories survive across sessions and are shared with Claude Code (same directory, same format — what you write here, Claude Code reads, and vice versa).

### Current memory index

The snapshot below was taken at startup and may be stale — call `memory_list` to refresh whenever unsure.

{index_content}

### Tools

{tools_description}

### How to remember

- `memory_write` stores one memory as `<name>.md` and syncs the index automatically. `name` must match `^[a-z0-9][a-z0-9-]*$`; writing to an existing `name` overwrites it in place (update instead of duplicating; prefer overwriting rolling `project` memories like status notes over creating new ones).
- The `description` is the recall key — one line (no newlines) that lets you tell at a glance whether the memory matters for the current task.
- Types: `user` (who the user is: role, preferences, working style), `feedback` (rules and corrections the user gave you), `project` (ongoing work, decisions, status), `reference` (pointers to external resources).
- Link related memories with `[[memory-name]]` in the body.

### When to use memory

- The user explicitly asks to remember something ("remember this", "save that").
- You learn a durable preference, correction, or project fact worth carrying into future sessions.

### When NOT to use memory

- Transient details that only matter within this session.
- Facts already recorded in the codebase (code structure, git history, config files).

### How to recall

Scan the index above by `description` and call `memory_read(name)` only for the relevant entries. Recall is selective by design — do not read everything."#;

impl MemoryConfig {
    /// Claude Code compatible configuration for `workspace`.
    ///
    /// Storage: `~/.claude/projects/<slug>/memory/` with the slug computed
    /// from the workspace path (same algorithm as Claude Code), so both tools
    /// share one memory directory.
    pub fn claude_compatible(workspace: &Path) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self::custom(
            home.join(".claude")
                .join("projects")
                .join(project_slug(workspace))
                .join("memory"),
            CLAUDE_COMPATIBLE_TEMPLATE.to_string(),
        )
    }

    /// Custom configuration: the consumer computes `memory_root` itself
    /// (different product, different interop target) and supplies its own
    /// prompt template.
    pub fn custom(memory_root: PathBuf, prompt_template: String) -> Self {
        Self {
            memory_root,
            index_filename: DEFAULT_INDEX_FILENAME.to_string(),
            prompt_template,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryStore
// ─────────────────────────────────────────────────────────────────────────────

/// One memory directory + its in-process index lock.
///
/// All four memory tools share one `Arc<MemoryStore>`. File IO is synchronous
/// and bounded by the critical section (a read, one write, one rename), so a
/// `std::sync::Mutex` is the right primitive — never hold it across an `await`
/// (the tools don't).
pub struct MemoryStore {
    memory_root: PathBuf,
    index_filename: String,
    /// Serializes index read-modify-write within this process, so concurrent
    /// turns appending rows cannot lose updates.
    index_lock: Mutex<()>,
}

/// A row in the memory listing.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryListEntry {
    pub name: String,
    pub description: String,
    pub memory_type: Option<String>,
    pub modified: Option<String>,
}

impl MemoryStore {
    pub fn new(memory_root: PathBuf, index_filename: impl Into<String>) -> Self {
        Self {
            memory_root,
            index_filename: index_filename.into(),
            index_lock: Mutex::new(()),
        }
    }

    pub fn memory_root(&self) -> &Path {
        &self.memory_root
    }

    pub fn index_filename(&self) -> &str {
        &self.index_filename
    }

    pub fn index_path(&self) -> PathBuf {
        self.memory_root.join(&self.index_filename)
    }

    fn memory_path(&self, name: &str) -> PathBuf {
        // Caller must have run validate_memory_name first.
        self.memory_root.join(format!("{name}.md"))
    }

    /// Read the index file, or the empty string when it doesn't exist.
    pub fn read_index(&self) -> String {
        fs::read_to_string(self.index_path()).unwrap_or_default()
    }

    /// Ensure the memory directory exists (first-use auto-init).
    pub fn ensure_dir(&self) -> AgentResult<()> {
        fs::create_dir_all(&self.memory_root).map_err(|e| {
            AgentError::internal(format!(
                "failed to create memory dir {}: {e}",
                self.memory_root.display()
            ))
        })
    }

    /// Create or overwrite the memory `name` and sync the index row.
    ///
    /// Order: file first, index second — a crash between the two leaves an
    /// unindexed file (recoverable via [`rebuild_index`]), never a dangling
    /// index row.
    pub fn write_memory(
        &self,
        name: &str,
        description: &str,
        memory_type: &str,
        body: &str,
        origin_session_id: Option<&str>,
    ) -> AgentResult<()> {
        validate_memory_name(name)?;
        validate_memory_type(memory_type)?;
        // Single-line descriptions only — and not merely `\n`: Unicode line/
        // paragraph separators (U+2028/U+2029/U+0085) read as line breaks in
        // a prompt, so they could forge extra index rows.
        if description
            .chars()
            .any(|c| matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}' | '\u{85}'))
        {
            return Err(AgentError::internal(
                "invalid memory description: must be a single line (no newlines)",
            ));
        }
        self.ensure_dir()?;
        // Merge into the file's existing frontmatter so keys owned by the
        // other writer (Claude Code) survive the update.
        let file_path = self.memory_path(name);
        let original = fs::read_to_string(&file_path).ok();
        let file = render_memory_file_merged(
            original.as_deref(),
            name,
            description,
            memory_type,
            origin_session_id,
            Some(&now_iso8601()),
            body,
        )?;
        atomic_write(&file_path, &file)?;
        let _guard = self.index_lock.lock().unwrap();
        index_add(&self.memory_root, &self.index_filename, name, description)
    }

    /// Read one memory's full raw file content (frontmatter + body).
    pub fn read_memory(&self, name: &str) -> AgentResult<String> {
        validate_memory_name(name)?;
        let path = self.memory_path(name);
        fs::read_to_string(&path).map_err(|_| {
            AgentError::internal(format!(
                "memory `{name}` not found (looked at {})",
                path.display()
            ))
        })
    }

    /// Delete the memory `name`: remove its index row first, then the file.
    /// Order matches write's crash invariant: after a crash mid-delete, the
    /// worst case is an orphaned file (no dangling index row).
    pub fn delete_memory(&self, name: &str) -> AgentResult<()> {
        validate_memory_name(name)?;
        let path = self.memory_path(name);
        if !path.exists() {
            return Err(AgentError::internal(format!(
                "memory `{name}` not found (looked at {})",
                path.display()
            )));
        }
        // P2-3 fix: index first, file second — crash-safe (orphan file, no dangling row).
        {
            let _guard = self.index_lock.lock().unwrap();
            index_remove(&self.memory_root, &self.index_filename, name)?;
        }
        fs::remove_file(&path)
            .map_err(|e| AgentError::internal(format!("failed to delete memory `{name}`: {e}")))
    }

    /// List all parseable memories, sorted by name. Corrupt files are skipped
    /// with a warning (fault tolerance) and never break the listing.
    pub fn list_memories(&self) -> Vec<MemoryListEntry> {
        let mut entries = Vec::new();
        let Ok(dir) = fs::read_dir(&self.memory_root) else {
            return entries; // directory missing → no memories yet
        };
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension() != Some("md".as_ref())
                || path.file_name() == Some(self.index_filename.as_ref())
            {
                continue;
            }
            if let Ok(doc) = parse_memory_file(&path) {
                entries.push(MemoryListEntry {
                    name: doc.name,
                    description: doc.description,
                    memory_type: doc.metadata.as_ref().map(|m| m.memory_type.clone()),
                    modified: doc.metadata.as_ref().and_then(|m| m.modified.clone()),
                });
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }
}
