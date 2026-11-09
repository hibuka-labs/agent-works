use std::fs;
use std::path::Path;

use agent_base::{AgentError, AgentResult};
use serde::{Deserialize, Serialize};

use super::MEMORY_NODE_TYPE;

// ─────────────────────────────────────────────────────────────────────────────
// Frontmatter
// ─────────────────────────────────────────────────────────────────────────────

/// YAML frontmatter `metadata:` block (Claude Code compatible field names).
///
/// Every field is optional on read: Claude Code's documented format carries
/// only `type`, and files written before a field existed must keep parsing
/// (tolerant read — the shared directory has a second writer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryFrontmatterMetadata {
    #[serde(default)]
    pub node_type: String,
    #[serde(rename = "type", default)]
    pub memory_type: String,
    #[serde(
        rename = "originSessionId",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub origin_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified: Option<String>,
}

/// YAML frontmatter of a memory file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryFrontmatter {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub metadata: Option<MemoryFrontmatterMetadata>,
}

/// A parsed memory file: frontmatter fields plus the free-form Markdown body.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDoc {
    pub name: String,
    pub description: String,
    pub metadata: Option<MemoryFrontmatterMetadata>,
    pub body: String,
}

/// Split `---\n<yaml>\n---\n<body>` into `(yaml, body)`.
///
/// Returns `None` when the file has no (complete) frontmatter block. The body
/// keeps everything after the closing `---` line with leading blank lines
/// trimmed (the conventional single blank separator is not part of the body).
fn split_frontmatter(raw: &str) -> Option<(String, String)> {
    // Both line-ending conventions: a foreign writer on Windows must not be
    // treated as "no frontmatter" — the merge would then drop every key.
    let rest = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))?;
    let mut cursor = 0usize;
    let mut close = None;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches(['\n', '\r']) == "---" {
            close = Some(cursor);
            break;
        }
        cursor += line.len();
    }
    let close = close?;
    let yaml = rest[..close].to_string();
    let after = &rest[close + 3..]; // skip the closing "---"
    let after = after
        .strip_prefix("\r\n")
        .or_else(|| after.strip_prefix('\n'))
        .unwrap_or(after);
    Some((yaml, after.trim_start_matches('\n').to_string()))
}

/// Parse a memory file into a [`MemoryDoc`].
///
/// Used by listing/index-rebuild paths; callers treat `Err` as "skip this
/// file" (fault tolerance for files Claude Code may have written badly).
pub fn parse_memory_file(path: &Path) -> AgentResult<MemoryDoc> {
    let raw = fs::read_to_string(path)?;
    parse_memory_str(&raw).ok_or_else(|| {
        AgentError::internal(format!(
            "{}: missing or malformed frontmatter",
            path.display()
        ))
    })
}

/// Parse memory file content (frontmatter + body).
pub fn parse_memory_str(raw: &str) -> Option<MemoryDoc> {
    let (yaml, body) = split_frontmatter(raw)?;
    let fm: MemoryFrontmatter = match serde_yaml::from_str(&yaml) {
        Ok(fm) => fm,
        Err(e) => {
            tracing::warn!(error = %e, "skipping memory file with invalid frontmatter");
            return None;
        }
    };
    if fm.name.is_empty() || fm.description.is_empty() {
        tracing::warn!("skipping memory file with empty name or description");
        return None;
    }
    Some(MemoryDoc {
        name: fm.name,
        description: fm.description,
        metadata: fm.metadata,
        body,
    })
}

/// Render a full memory file (frontmatter + body) for writing.
///
/// `origin_session_id` and `modified` are auto-filled by `memory_write` and
/// omitted when `None` (matching Claude Code's optional fields).
pub fn render_memory_file(
    name: &str,
    description: &str,
    memory_type: &str,
    origin_session_id: Option<&str>,
    modified: Option<&str>,
    body: &str,
) -> AgentResult<String> {
    render_memory_file_merged(
        None,
        name,
        description,
        memory_type,
        origin_session_id,
        modified,
        body,
    )
}

/// Render a memory file, merging into `original_raw` when updating.
///
/// Write-fidelity contract (shared directory, second writer = Claude Code):
/// frontmatter keys this module doesn't own — top-level or inside `metadata` —
/// are preserved across an update. Known keys are upserted; the session stamps
/// are set-or-keep (`None` never clears a stamp the other writer set). Two
/// hard limits, both lossy by necessity: YAML comments are not preserved, and
/// a `metadata` value that is not a mapping is left untouched (never destroy
/// what we cannot merge into). Best-effort merge: if the original frontmatter
/// won't parse as YAML the file is re-rendered from the known fields alone —
/// warned, because that drops every foreign key.
pub(super) fn render_memory_file_merged(
    original_raw: Option<&str>,
    name: &str,
    description: &str,
    memory_type: &str,
    origin_session_id: Option<&str>,
    modified: Option<&str>,
    body: &str,
) -> AgentResult<String> {
    use serde_yaml::{Mapping, Value};

    fn upsert(map: &mut Mapping, key: &str, value: &str) {
        map.insert(
            Value::String(key.to_string()),
            Value::String(value.to_string()),
        );
    }

    let mut map: Mapping = match original_raw.and_then(split_frontmatter) {
        Some((yaml, _)) => serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
            tracing::warn!(
                error = %e,
                "memory frontmatter not parseable; re-rendering from known fields (foreign keys dropped)"
            );
            Mapping::new()
        }),
        None => Mapping::new(),
    };
    upsert(&mut map, "name", name);
    upsert(&mut map, "description", description);
    // Merge into the original metadata mapping so unknown keys survive. A
    // non-mapping `metadata` is left untouched — faithful write: we never
    // destroy a value we can't merge into.
    let meta = match map.get(Value::String("metadata".to_string())) {
        Some(Value::Mapping(m)) => Some(m.clone()),
        None => Some(Mapping::new()),
        Some(other) => {
            tracing::warn!(
                value = ?other,
                "metadata is not a mapping; leaving it untouched (memory stamps not applied)"
            );
            None
        }
    };
    if let Some(mut meta) = meta {
        upsert(&mut meta, "node_type", MEMORY_NODE_TYPE);
        upsert(&mut meta, "type", memory_type);
        // Set-or-keep: `None` never clears a stamp — it may record the other
        // writer's provenance.
        if let Some(v) = origin_session_id {
            upsert(&mut meta, "originSessionId", v);
        }
        if let Some(v) = modified {
            upsert(&mut meta, "modified", v);
        }
        map.insert(Value::String("metadata".to_string()), Value::Mapping(meta));
    }

    // serde_yaml (never string concatenation) so descriptions with special
    // characters are quoted/escaped correctly.
    let yaml = serde_yaml::to_string(&map).map_err(|e| {
        AgentError::internal(format!("failed to serialize memory frontmatter: {e}"))
    })?;
    // `to_string` output always ends with '\n'.
    Ok(format!(
        "---\n{yaml}---\n\n{}",
        body.trim_start_matches('\n')
    ))
}
