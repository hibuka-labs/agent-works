//! Persistent auto-memory (Phase 9b, L2) — Claude Code compatible.
//!
//! Memories live as Markdown files with YAML frontmatter under a
//! per-project directory (default: `~/.claude/projects/<slug>/memory/`),
//! indexed by `MEMORY.md`. The format, slug algorithm, and index layout are
//! byte-compatible with Claude Code's auto-memory, so both tools read and
//! write the same directory with zero migration.
//!
//! Layering (docs/phase9-plan.md "Framework Layering Design"):
//! - This module owns the *mechanism*: frontmatter (de)serialization, atomic
//!   file writes, row-level index sync, and a `std::sync::Mutex` serializing
//!   index read-modify-write within the process (file IO inside the critical
//!   section is synchronous and millisecond-scale, so no async lock needed).
//! - The consumer owns the *policy* via [`MemoryConfig`]: root path, index
//!   filename, and the prompt template injected into the system prompt.
//!
//! Concurrency model: in-process writes are serialized per [`MemoryStore`];
//! cross-process concurrency (phimint + Claude Code) relies on atomic
//! rename-based writes (last-write-wins, no torn files) plus [`rebuild_index`]
//! as the repair path.
//!
//! Shared-writer contract (the directory has a second writer — Claude Code —
//! and neither side can stop the other from writing):
//! - **Tolerant read**: any frontmatter this module doesn't know — missing
//!   stamps (`node_type`, `originSessionId`, …), unknown keys — still parses.
//!   Claude Code's documented format is a supported fixture, not an error.
//! - **Faithful write**: updates merge into the existing frontmatter instead of
//!   re-rendering it, so keys owned by the other writer survive (see
//!   [`MemoryStore::write_memory`]).
//! - **Conservative repair**: a row is identified structurally — by its own
//!   link target `- [title](name.md)`, never by a substring of another line.
//!   An update refreshes only the row's `— description` tail; a rebuild
//!   re-emits existing rows byte-identical and in their curated order
//!   ([`index_add`], [`rebuild_index`]) — human titles live only in the index.
//!   A rebuild never erases a row it cannot re-derive.

/// Default index file name (Claude Code compatible).
pub const DEFAULT_INDEX_FILENAME: &str = "MEMORY.md";

/// Fixed `metadata.node_type` value for every memory file.
pub const MEMORY_NODE_TYPE: &str = "memory";

/// Allowed `metadata.type` values.
pub const MEMORY_TYPES: [&str; 4] = ["user", "feedback", "project", "reference"];

mod frontmatter;
mod index;
mod io;
mod naming;
mod store;

pub use frontmatter::{
    MemoryDoc, MemoryFrontmatter, MemoryFrontmatterMetadata, parse_memory_file, parse_memory_str,
    render_memory_file,
};
pub use index::rebuild_index;
pub(crate) use index::{index_add, index_remove};
pub(crate) use io::atomic_write;
pub use io::now_iso8601;
pub use naming::{is_valid_memory_name, project_slug, validate_memory_name, validate_memory_type};
pub use store::{CLAUDE_COMPATIBLE_TEMPLATE, MemoryConfig, MemoryListEntry, MemoryStore};

// ─────────────────────────────────────────────────────────────────────────────
// Tests (modules 1 & 2)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
