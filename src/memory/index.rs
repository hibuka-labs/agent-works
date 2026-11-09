use std::fs;
use std::path::Path;

use agent_base::{AgentError, AgentResult};

use super::atomic_write;
use super::frontmatter::parse_memory_file;

// ─────────────────────────────────────────────────────────────────────────────
// Index sync (row-level, never a full rewrite)
// ─────────────────────────────────────────────────────────────────────────────
//
// Index lines look like `- [Human Title](name.md) — short description`. The
// link text is a human-readable title (possibly Chinese) that only exists in
// the index — NOT in any frontmatter. Therefore add/remove must locate rows by
// the link target `(name.md)` and touch only those rows, leaving every other
// row (and its title) byte-identical. A refresh of a row replaces only its
// `— description` tail — the human title survives updates too (shared-index
// contract with Claude Code). New rows use the memory name as link text, the
// same shape [`rebuild_index`] generates.

/// Sanitize a description for single-line index rows.
///
/// The write path already rejects line breaks in descriptions; this scrub is
/// defense-in-depth for descriptions that reach a row from any other path
/// (e.g. regenerated rows for files a foreign writer created).
fn index_line_desc(desc: &str) -> String {
    desc.replace(['\n', '\r', '\u{2028}', '\u{2029}', '\u{85}'], " ")
        .trim()
        .to_string()
}

/// The link target of an index row line, if the line is a row at all:
/// `- [title](name.md)` → `name`.
///
/// Row identity is structural — the link target must be the row's first
/// `](…)` — so a description embedding another memory's `](x.md)` link can
/// never hijack that memory's row (descriptions are free text written by LLMs
/// and foreign tools).
fn row_link_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("- [")?;
    let close = rest.find("](")?;
    let target = rest[close + 2..].split(')').next()?;
    target.strip_suffix(".md")
}

/// `Some(head)` — everything up to and including `](name.md)` — when `line` is
/// `name`'s row. The head is preserved byte-identical on refresh so human
/// titles in the link text survive.
fn row_head<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    if row_link_name(line) != Some(name) {
        return None;
    }
    let close = line.find("](")?;
    line.get(..close + 2 + name.len() + ".md)".len())
}

/// Render a new index row: link text is the memory name — the shape
/// [`rebuild_index`] generates. Human titles exist only in rows the other
/// writer created and are never re-rendered here.
fn render_index_row(name: &str, desc: &str) -> String {
    format!("- [{name}]({name}.md) — {}", index_line_desc(desc))
}

/// Add (or update) the row for `name` in the index.
///
/// Existing rows for other names — including their human titles — are
/// preserved verbatim.
pub(crate) fn index_add(
    memory_root: &Path,
    index_filename: &str,
    name: &str,
    desc: &str,
) -> AgentResult<()> {
    // Self-healing: the very first write may race ahead of ensure_dir.
    fs::create_dir_all(memory_root).map_err(|e| {
        AgentError::internal(format!(
            "failed to create memory dir {}: {e}",
            memory_root.display()
        ))
    })?;
    let index_path = memory_root.join(index_filename);
    // P2-2 fix: distinguish NotFound (empty index) from real I/O errors.
    let content = match fs::read_to_string(&index_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            return Err(AgentError::internal(format!(
                "failed to read index {}: {e}",
                index_path.display()
            )));
        }
    };
    // Row identity is structural (the row's own link target), so a description
    // that embeds another `](name.md)` cannot hijack the refresh.
    let desc_line = index_line_desc(desc);

    let updated = if content.lines().any(|line| row_head(line, name).is_some()) {
        // Row exists: refresh only its `— description` tail. Everything up to
        // and including `](name.md)` — the human title in the link text — is
        // preserved byte-identical (Claude Code writes those titles).
        let mut out = content
            .lines()
            .map(|line| match row_head(line, name) {
                Some(head) => format!("{head} — {desc_line}"),
                None => line.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        if content.ends_with('\n') {
            out.push('\n');
        }
        out
    } else {
        // New row: append after the existing rows.
        let trimmed = content.trim_end();
        let mut out = String::new();
        if !trimmed.is_empty() {
            out.push_str(trimmed);
            out.push('\n');
        }
        out.push_str(&render_index_row(name, desc));
        out.push('\n');
        out
    };
    atomic_write(&index_path, &updated)
}

/// Remove the row for `name` from the index. Other rows are untouched.
pub(crate) fn index_remove(
    memory_root: &Path,
    index_filename: &str,
    name: &str,
) -> AgentResult<()> {
    fs::create_dir_all(memory_root).map_err(|e| {
        AgentError::internal(format!(
            "failed to create memory dir {}: {e}",
            memory_root.display()
        ))
    })?;
    let index_path = memory_root.join(index_filename);
    // P2-2 fix: distinguish NotFound (empty index) from real I/O errors.
    let content = match fs::read_to_string(&index_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(AgentError::internal(format!(
                "failed to read index {}: {e}",
                index_path.display()
            )));
        }
    };
    // Row identity is structural (the row's own link target), never a bare
    // substring of some other row's description.
    let mut updated = content
        .lines()
        .filter(|line| row_link_name(line) != Some(name))
        .collect::<Vec<_>>()
        .join("\n");
    if !updated.is_empty() && content.ends_with('\n') {
        updated.push('\n');
    }
    atomic_write(&index_path, &updated)
}

/// Find the existing index row for `name`, if any (same structural row
/// identity as [`index_add`] / [`index_remove`]). Returns the line
/// byte-identical — the human title in the link text must survive rebuilds.
fn find_row(index: &str, name: &str) -> Option<String> {
    index
        .lines()
        .find(|line| row_head(line, name).is_some())
        .map(str::to_string)
}

/// Rebuild the index from the memory directory (fault-tolerance path only).
///
/// Triggered when the index is lost or unreadable. Conservative repair
/// (shared-index contract with Claude Code): a file's existing row is re-emitted
/// byte-identical whenever one exists — human titles live only in the index and
/// cannot be recovered from the file. Rows are regenerated from frontmatter
/// only for files that have no row yet. Files that cannot be indexed at all
/// (corrupt frontmatter, name ≠ filename) keep their existing row rather than
/// being erased; only rows whose file is gone are dropped (dangling links).
pub fn rebuild_index(memory_root: &Path, index_filename: &str) -> AgentResult<()> {
    fs::create_dir_all(memory_root).map_err(|e| {
        AgentError::internal(format!(
            "failed to create memory dir {}: {e}",
            memory_root.display()
        ))
    })?;
    // Only "no index yet" is empty. Any other read error must abort: treating
    // it as empty would regenerate every human title away — exactly the loss
    // conservative repair exists to prevent.
    let old_index = match fs::read_to_string(memory_root.join(index_filename)) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            return Err(AgentError::internal(format!(
                "failed to read index {}: {e}",
                memory_root.join(index_filename).display()
            )));
        }
    };
    // Encounter order of the old index: kept rows survive in their curated
    // order (the prompt tells the LLM to scan top-down); rows generated for
    // files that have no row yet are appended after them, sorted by name.
    let old_pos: Vec<&str> = old_index.lines().filter_map(row_link_name).collect();
    let mut entries: Vec<(usize, String, String)> = Vec::new(); // (order, name, row)
    for entry in fs::read_dir(memory_root)? {
        let path = entry?.path();
        if path.extension() != Some("md".as_ref())
            || path.file_name() == Some(index_filename.as_ref())
        {
            continue;
        }
        let Some(name) = path.file_stem().map(|s| s.to_string_lossy().to_string()) else {
            continue;
        };
        let regenerated = parse_memory_file(&path)
            .ok()
            // Only index files whose own name matches their filename, so the
            // rebuilt index links resolve.
            .filter(|doc| doc.name == name)
            .map(|doc| render_index_row(&doc.name, &doc.description));
        let existing = find_row(&old_index, &name).map(|row| {
            let order = old_pos
                .iter()
                .position(|n| *n == name)
                .unwrap_or(usize::MAX);
            (order, row)
        });
        match existing.or_else(|| regenerated.map(|row| (usize::MAX, row))) {
            Some((order, row)) => entries.push((order, name, row)),
            None => tracing::warn!(
                file = %path.display(),
                "skipping unindexable memory file with no existing index row"
            ),
        }
    }
    entries.sort();
    let content = if entries.is_empty() {
        String::new()
    } else {
        entries
            .into_iter()
            .map(|(_, _, row)| row)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    atomic_write(&memory_root.join(index_filename), &content)
}
