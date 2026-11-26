use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use agent_base::{AgentError, AgentResult};

use super::frontmatter::{EntryFields, parse_memory_file, render_memory_file_merged};
use super::{
    DEFAULT_INDEX_FILENAME, atomic_write, index_remove, now_iso8601, project_slug,
    render_index_from_disk, validate_memory_name, validate_memory_type,
};

// ─────────────────────────────────────────────────────────────────────────────
// MemoryLimits (design §5) — the write-path gates
// ─────────────────────────────────────────────────────────────────────────────

/// Write-path limits. Counts are mechanical (chars / bytes), never tokens.
///
/// Defaults are the design §5 table. A product may override any field via
/// [`MemoryConfig`]; `enabled: false` short-circuits every gate (the rollback
/// seam). The per-entry character gates (`hook`, `body`) and the aggregate
/// capacity gates (`index`, `store`) are enforced hard on write; the `*_warn`
/// / `*_soft` fields are nudge thresholds only (P1) and are carried here so a
/// product override is complete but not yet enforced.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryLimits {
    /// Master switch: `false` disables every hard gate (rollback seam).
    pub enabled: bool,
    /// Max chars in an index-row hook (= the `description`). Hard reject.
    pub hook_max_chars: usize,
    /// Max chars in a memory body. Hard reject.
    pub body_max_chars: usize,
    /// Max bytes of the whole index (hot-zone capacity). Hard reject.
    pub index_max_bytes: usize,
    /// Index size at which to nudge (P1). Not enforced in P0.
    pub index_warn_bytes: usize,
    /// Max total on-disk bytes of the store. Hard reject.
    pub store_max_bytes: usize,
    /// Entry count at which to nudge (P1). Not enforced in P0.
    pub store_soft_entries: usize,
    /// Max entries in the store (hot + cold). Hard reject.
    pub store_hard_entries: usize,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            enabled: true,
            hook_max_chars: 80,
            body_max_chars: 3000,
            index_max_bytes: 10_240,
            index_warn_bytes: 8192,
            store_max_bytes: 2_097_152,
            store_soft_entries: 300,
            store_hard_entries: 500,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LimitRejection — the five-field failure contract (design §6)
// ─────────────────────────────────────────────────────────────────────────────

/// A write-path limit rejection, rendered to the LLM as the wire shape
/// `{"ok": false, "rule", "limit", "actual", "name", "fix", "candidate"}`
/// (design §6). The rejection names which rule broke, the actual value against
/// the limit, and a `candidate` the model can retry with in one shot — for the
/// per-entry length rules the candidate is a mechanical trim that passes; for
/// the aggregate capacity rules there is no one-shot value and `candidate` is
/// empty while `fix` points at the cleanup (`memory_tidy`).
#[derive(Debug, Clone, PartialEq)]
pub struct LimitRejection {
    pub rule: String,
    pub limit: usize,
    pub actual: usize,
    pub name: String,
    pub fix: String,
    pub candidate: String,
}

impl LimitRejection {
    /// Render the wire shape (the `ok: false` JSON the tool returns verbatim).
    pub fn to_wire(&self) -> String {
        serde_json::json!({
            "ok": false,
            "rule": self.rule,
            "limit": self.limit,
            "actual": self.actual,
            "name": self.name,
            "fix": self.fix,
            "candidate": self.candidate,
        })
        .to_string()
    }
}

/// Outcome of a write: committed, or rejected by a limit with the failure
/// contract. Hard errors (bad name/type, IO failure) are `Err`, not `Rejected`
/// — a limit rejection is an expected, self-explanatory outcome the model can
/// act on, distinct from a fault.
#[derive(Debug, Clone, PartialEq)]
pub enum WriteOutcome {
    /// The memory landed and the index was synced.
    Written { created: bool },
    /// A write-path limit refused the write; nothing was persisted.
    Rejected(LimitRejection),
}

/// Truncate `s` to at most `max` chars on a char boundary (the mechanical
/// `candidate` trim that passes the length gate on retry).
fn trim_to_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

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
    /// Write-path limits (design §5). Defaults are the framework values; a
    /// product overrides any field (e.g. phimint's `[memory]` config section).
    pub limits: MemoryLimits,
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

Scan the index above by `description` and open only the relevant ones — the index row names the file, and it lives beside `MEMORY.md` in the memory directory. Recall is selective by design — do not read everything."#;

impl MemoryConfig {
    /// Claude Code compatible configuration for `workspace`.
    ///
    /// Storage: `~/.claude/projects/<slug>/memory/` with the slug computed
    /// from the workspace path (same algorithm as Claude Code), so both tools
    /// share one memory directory.
    pub fn claude_compatible(workspace: &Path) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| {
            tracing::warn!("home directory not found; memory root falls back to temp dir");
            std::env::temp_dir()
        });
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
            limits: MemoryLimits::default(),
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
    limits: MemoryLimits,
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
    /// A store with the default [`MemoryLimits`].
    pub fn new(memory_root: PathBuf, index_filename: impl Into<String>) -> Self {
        Self::with_limits(memory_root, index_filename, MemoryLimits::default())
    }

    /// A store with explicit write-path limits (product override entry).
    pub fn with_limits(
        memory_root: PathBuf,
        index_filename: impl Into<String>,
        limits: MemoryLimits,
    ) -> Self {
        Self {
            memory_root,
            index_filename: index_filename.into(),
            limits,
            index_lock: Mutex::new(()),
        }
    }

    pub fn memory_root(&self) -> &Path {
        &self.memory_root
    }

    pub fn index_filename(&self) -> &str {
        &self.index_filename
    }

    pub fn limits(&self) -> &MemoryLimits {
        &self.limits
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

    /// Rebuild the index listing from disk — the source of truth (design §4).
    ///
    /// This is what `memory_list` and prompt injection read: a fresh truth
    /// projection (see [`render_index_from_disk`]), never the possibly-stale
    /// `MEMORY.md` snapshot. Use [`read_index`] only to inspect the snapshot
    /// Claude Code sees.
    pub fn render_index(&self) -> AgentResult<String> {
        render_index_from_disk(&self.memory_root, &self.index_filename)
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

    /// Create or overwrite a memory with no lifecycle fields. See
    /// [`write_memory_entry`] for the full contract.
    pub fn write_memory(
        &self,
        name: &str,
        description: &str,
        memory_type: &str,
        body: &str,
        origin_session_id: Option<&str>,
    ) -> AgentResult<WriteOutcome> {
        self.write_memory_entry(
            name,
            description,
            memory_type,
            body,
            origin_session_id,
            &EntryFields::default(),
        )
    }

    /// Create or overwrite the memory `name`, then regenerate the index.
    ///
    /// Write order (design §6): body `temp+rename` first, index **last**. The
    /// index is regenerated from disk (the source of truth) and re-checked
    /// against `index_max_bytes`; if it would overflow, the body is rolled back
    /// to its pre-write state and the write is rejected — never a half-written
    /// account. If the final index write fails, the body is likewise restored
    /// (compensation), so the directory and the index never disagree.
    ///
    /// Returns [`WriteOutcome::Written`] on commit, or
    /// [`WriteOutcome::Rejected`] carrying the five-field failure contract when
    /// a hard limit refuses the write. Hard errors (bad name/type, IO failure)
    /// remain `Err` — a limit rejection is an expected, self-explanatory
    /// outcome, distinct from a fault.
    pub fn write_memory_entry(
        &self,
        name: &str,
        description: &str,
        memory_type: &str,
        body: &str,
        origin_session_id: Option<&str>,
        entry: &EntryFields,
    ) -> AgentResult<WriteOutcome> {
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

        // Per-entry gates (hook / body) need no disk state: reject before
        // touching disk. These carry a one-shot `candidate` trim.
        if let Some(rej) = self.check_entry_limits(name, description, body) {
            return Ok(WriteOutcome::Rejected(rej));
        }

        // Hold the index lock across the whole body→index commit so a rollback
        // is atomic with respect to concurrent regenerators (a body that is
        // later rolled back must never appear in another writer's index).
        let _guard = self.index_lock.lock().unwrap();

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
            entry,
            body,
        )?;

        // Aggregate store gates (entry count / total bytes) against the
        // projected post-write state.
        if let Some(rej) = self.check_store_limits(name, original.as_deref(), &file) {
            return Ok(WriteOutcome::Rejected(rej));
        }

        let created = original.is_none();
        // Body first (temp+rename) …
        atomic_write(&file_path, &file)?;

        // … then regenerate the index from disk and enforce the index gate.
        let index_content = render_index_from_disk(&self.memory_root, &self.index_filename)?;
        if self.limits.enabled && index_content.len() > self.limits.index_max_bytes {
            // Roll the body back to its pre-write state; the on-disk index is
            // still at its pre-write state (we never wrote it), so the account
            // and the matter agree.
            self.restore_pre_write(&file_path, original.as_deref());
            return Ok(WriteOutcome::Rejected(LimitRejection {
                rule: "index_max_bytes".to_string(),
                limit: self.limits.index_max_bytes,
                actual: index_content.len(),
                name: name.to_string(),
                fix: format!(
                    "the index is {}/{} bytes — run `memory_tidy` or archive \
                     entries to shrink it, then retry",
                    index_content.len(),
                    self.limits.index_max_bytes
                ),
                candidate: String::new(),
            }));
        }
        // Index last. On failure, compensate by restoring the body so we never
        // leave "row in index, file missing" — the forbidden half-account. The
        // restore is best-effort; if it too fails we leave an orphan file
        // (invisible, non-blocking) and warn.
        if let Err(e) = atomic_write(&self.index_path(), &index_content) {
            self.restore_pre_write(&file_path, original.as_deref());
            tracing::warn!(
                error = %e,
                name,
                "index write failed after body write; body restored (best-effort)"
            );
            return Err(e);
        }
        Ok(WriteOutcome::Written { created })
    }

    /// Per-entry gates: hook (`description`) and body length. Returns the
    /// failure contract for the first broken gate, with a one-shot `candidate`
    /// trim that passes on retry.
    fn check_entry_limits(
        &self,
        name: &str,
        description: &str,
        body: &str,
    ) -> Option<LimitRejection> {
        if !self.limits.enabled {
            return None;
        }
        let hook_chars = description.chars().count();
        if hook_chars > self.limits.hook_max_chars {
            return Some(LimitRejection {
                rule: "hook_max_chars".to_string(),
                limit: self.limits.hook_max_chars,
                actual: hook_chars,
                name: name.to_string(),
                fix: format!(
                    "compress the hook to <={} chars; move the detail into the body",
                    self.limits.hook_max_chars
                ),
                candidate: trim_to_chars(description, self.limits.hook_max_chars),
            });
        }
        let body_chars = body.chars().count();
        if body_chars > self.limits.body_max_chars {
            return Some(LimitRejection {
                rule: "body_max_chars".to_string(),
                limit: self.limits.body_max_chars,
                actual: body_chars,
                name: name.to_string(),
                fix: format!(
                    "the body is {body_chars} chars — one fact per memory; trim \
                     it to <={} chars or split the memory",
                    self.limits.body_max_chars
                ),
                candidate: trim_to_chars(body, self.limits.body_max_chars),
            });
        }
        None
    }

    /// Aggregate store gates: entry count and total on-disk bytes, checked
    /// against the projected post-write state. `original` is the file's prior
    /// content (absent on create).
    fn check_store_limits(
        &self,
        name: &str,
        original: Option<&str>,
        new_file: &str,
    ) -> Option<LimitRejection> {
        if !self.limits.enabled {
            return None;
        }
        let (count, total_bytes) = self.store_stats();
        // Entry-count gate: only a create adds an entry.
        if original.is_none() && count + 1 > self.limits.store_hard_entries {
            return Some(LimitRejection {
                rule: "store_hard_entries".to_string(),
                limit: self.limits.store_hard_entries,
                actual: count + 1,
                name: name.to_string(),
                fix: format!(
                    "the store already holds {count} entries (limit {}) — \
                     archive or delete some (`memory_tidy` / `memory_delete`), then retry",
                    self.limits.store_hard_entries
                ),
                candidate: String::new(),
            });
        }
        // Byte gate: projected total after this write (replace old bytes with new).
        let old_bytes = original.map(str::len).unwrap_or(0);
        let projected = total_bytes - old_bytes + new_file.len();
        if projected > self.limits.store_max_bytes {
            return Some(LimitRejection {
                rule: "store_max_bytes".to_string(),
                limit: self.limits.store_max_bytes,
                actual: projected,
                name: name.to_string(),
                fix: format!(
                    "the store would reach {projected} bytes (limit {}) — \
                     archive or delete entries (`memory_tidy`), then retry",
                    self.limits.store_max_bytes
                ),
                candidate: String::new(),
            });
        }
        None
    }

    /// `(entry_count, total_bytes)` of the memory files on disk (the index
    /// file itself is excluded).
    fn store_stats(&self) -> (usize, usize) {
        let mut count = 0usize;
        let mut total = 0usize;
        let Ok(dir) = fs::read_dir(&self.memory_root) else {
            return (0, 0);
        };
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension() != Some("md".as_ref())
                || path.file_name() == Some(self.index_filename.as_ref())
            {
                continue;
            }
            count += 1;
            total += entry.metadata().map(|m| m.len() as usize).unwrap_or(0);
        }
        (count, total)
    }

    /// Restore `path` to its pre-write state: rewrite `original` when the file
    /// existed before, else remove it. Best-effort — a failure here leaves an
    /// orphan file (invisible, non-blocking) and is logged, never raised.
    fn restore_pre_write(&self, path: &Path, original: Option<&str>) {
        let result = match original {
            Some(content) => fs::write(path, content),
            None => fs::remove_file(path),
        };
        if let Err(e) = result {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "write rollback failed; leaving an orphan file"
            );
        }
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
