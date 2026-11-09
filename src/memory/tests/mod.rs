use super::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::io::civil_from_days;

fn tmp_root() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("memory");
    (dir, root)
}

// ── Claude Code compatibility fixtures ──
//
// The memory directory is shared with Claude Code (Phase 9c contract).
// These fixtures pin the cross-writer format: if a fixture stops parsing,
// phimint silently loses memories Claude Code wrote.

/// Claude Code's documented format: `metadata` carries only `type` —
/// no `node_type`, no origin/modified stamps.
const CC_FORMAT_FILE: &str = "---\n\
name: cc-note\ndescription: written by claude code\n\
metadata:\n  type: feedback\n\
---\n\n来自 Claude Code 的事实\n";

/// A file carrying frontmatter keys neither writer owns (today): a
/// top-level `tags` and a `metadata.created`. Updates must not drop them.
const EXTRA_FIELDS_FILE: &str = "---\n\
name: extra\ndescription: has unknown fields\ntags: [a, b]\n\
metadata:\n  node_type: memory\n  type: project\n  created: 2026-01-01\n\
---\n\nbody\n";

// ── project_slug ──

#[test]
fn slug_encodes_absolute_path() {
    assert_eq!(
        project_slug(Path::new("/Users/xxx/project")),
        "-Users-xxx-project"
    );
}

#[test]
fn slug_matches_claude_code_example() {
    assert_eq!(
        project_slug(Path::new(
            "/Users/kangzengchen/source/buka/buka-works/phimint"
        )),
        "-Users-kangzengchen-source-buka-buka-works-phimint"
    );
}

#[test]
fn slug_strips_trailing_slash() {
    assert_eq!(project_slug(Path::new("/a/b/")), "-a-b");
}

#[test]
fn slug_is_empty_for_root() {
    assert_eq!(project_slug(Path::new("/")), "");
}

#[test]
fn slug_replaces_all_non_alphanumeric() {
    // Dots, underscores, spaces all become `-`
    assert_eq!(
        project_slug(Path::new("/Users/alice/my.project_v2")),
        "-Users-alice-my-project-v2"
    );
}

// ── name validation ──

#[test]
fn name_validation_accepts_kebab_case() {
    // `^[a-z0-9][a-z0-9-]*$` permits a trailing dash — harmless, and the
    // plan pins the contract to this exact pattern.
    for ok in [
        "a",
        "9",
        "valid-name",
        "a1-b2-c3",
        "9lives",
        "mem0ry",
        "trailing-",
    ] {
        assert!(is_valid_memory_name(ok), "`{ok}` should be valid");
    }
}

#[test]
fn name_validation_rejects_path_escape_and_bad_shapes() {
    for bad in [
        "../../etc/passwd",
        "..",
        ".",
        "-leading-dash",
        "Upper",
        "has space",
        "has_underscore",
        "has.dot",
        "",
        "a/b",
        "中文",
        "a\nb",
    ] {
        assert!(!is_valid_memory_name(bad), "`{bad}` should be invalid");
    }
}

#[test]
fn validate_memory_name_error_mentions_rule() {
    let err = validate_memory_name("../escape").unwrap_err().to_string();
    assert!(err.contains("^[a-z0-9][a-z0-9-]*$"), "{err}");
}

#[test]
fn memory_type_validation() {
    for t in MEMORY_TYPES {
        validate_memory_type(t).unwrap();
    }
    assert!(validate_memory_type("secret").is_err());
    assert!(validate_memory_type("").is_err());
}

// ── frontmatter round trip ──

#[test]
fn frontmatter_round_trip_preserves_fields() {
    let file = render_memory_file(
        "cargo-commit-discipline",
        "绝不主动 commit；commit 永不带 Cargo 文件",
        "feedback",
        Some("8ee16870-4dc8-4dd1-9d13-4fc7a9980f79"),
        Some("2026-09-04T11:33:28.939Z"),
        "**规则：**\n\n0. 绝不主动 commit\n",
    )
    .unwrap();

    let doc = parse_memory_str(&file).unwrap();
    assert_eq!(doc.name, "cargo-commit-discipline");
    assert_eq!(doc.description, "绝不主动 commit；commit 永不带 Cargo 文件");
    let md = doc.metadata.unwrap();
    assert_eq!(md.node_type, "memory");
    assert_eq!(md.memory_type, "feedback");
    assert_eq!(
        md.origin_session_id.as_deref(),
        Some("8ee16870-4dc8-4dd1-9d13-4fc7a9980f79")
    );
    assert_eq!(md.modified.as_deref(), Some("2026-09-04T11:33:28.939Z"));
    assert_eq!(doc.body, "**规则：**\n\n0. 绝不主动 commit\n");
}

#[test]
fn frontmatter_omits_optional_fields_when_none() {
    let file = render_memory_file("t", "d", "project", None, None, "b").unwrap();
    assert!(!file.contains("originSessionId"));
    assert!(!file.contains("modified"));
    // Required fields still present, in Claude Code layout.
    assert!(file.contains("node_type: memory"));
    assert!(file.contains("type: project"));
}

#[test]
fn frontmatter_escapes_special_yaml_characters() {
    let tricky = "has: colon, \"quotes\", #hash and 中文";
    let file = render_memory_file("t", tricky, "user", None, None, "b").unwrap();
    let doc = parse_memory_str(&file).unwrap();
    assert_eq!(doc.description, tricky);
}

#[test]
fn parse_rejects_missing_frontmatter() {
    assert!(parse_memory_str("just some text\n").is_none());
    assert!(parse_memory_str("---\nname: x\n").is_none()); // no closing ---
}

#[test]
fn parse_rejects_empty_name_or_description() {
    let bad = "---\nname: \"\"\ndescription: d\n---\n\nbody\n";
    assert!(parse_memory_str(bad).is_none());
    let bad = "---\nname: x\ndescription: \"\"\n---\n\nbody\n";
    assert!(parse_memory_str(bad).is_none());
}

#[test]
fn parse_tolerates_missing_metadata_block() {
    let raw = "---\nname: legacy\ndescription: no metadata\n---\n\nbody\n";
    let doc = parse_memory_str(raw).unwrap();
    assert_eq!(doc.name, "legacy");
    assert!(doc.metadata.is_none());
}

#[test]
fn parse_tolerates_claude_code_metadata_without_node_type() {
    let doc = parse_memory_str(CC_FORMAT_FILE).expect("Claude Code format must parse");
    assert_eq!(doc.name, "cc-note");
    let meta = doc.metadata.expect("metadata present");
    assert_eq!(meta.memory_type, "feedback");
}

#[test]
fn list_memories_includes_claude_code_format_files() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("cc-note.md"), CC_FORMAT_FILE).unwrap();
    std::fs::write(
        root.join("ours.md"),
        "---\nname: ours\ndescription: ours\nmetadata:\n  node_type: memory\n  type: project\n---\n\nb\n",
    )
    .unwrap();
    let names: Vec<_> = store.list_memories().into_iter().map(|e| e.name).collect();
    assert_eq!(names, vec!["cc-note".to_string(), "ours".to_string()]);
    drop(dir);
}

// ── timestamps ──

#[test]
fn civil_from_days_known_values() {
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    assert_eq!(civil_from_days(11_017), (2000, 3, 1));
    assert_eq!(civil_from_days(19_723), (2024, 1, 1));
}

#[test]
fn now_iso8601_has_canonical_shape() {
    let ts = now_iso8601();
    // 2026-09-13T08:15:30.123Z
    assert_eq!(ts.len(), 24, "{ts}");
    assert!(ts.ends_with('Z'));
    let bytes = ts.as_bytes();
    for (i, expected) in b"dddd-dd-ddTdd:dd:dd.dddZ".iter().enumerate() {
        let actual = bytes[i];
        match expected {
            b'd' => assert!(actual.is_ascii_digit(), "{ts}"),
            b'-' | b'T' | b':' | b'.' | b'Z' => assert_eq!(actual, *expected, "{ts}"),
            _ => unreachable!(),
        }
    }
    // Year is plausible (this code was written in 2026).
    let year: i32 = ts[..4].parse().unwrap();
    assert!((2024..=2100).contains(&year), "{ts}");
}

// ── atomic write ──

#[test]
fn atomic_write_writes_content_and_leaves_no_tmp() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    let target = root.join("file.md");
    atomic_write(&target, "hello").unwrap();
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "hello");
    let leftovers: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert_eq!(leftovers, vec!["file.md".to_string()]);
    drop(dir);
}

#[test]
fn atomic_write_overwrites_previous_content() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    let target = root.join("file.md");
    atomic_write(&target, "old").unwrap();
    atomic_write(&target, "new").unwrap();
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
    drop(dir);
}

// ── index_add ──

#[test]
fn index_add_creates_index_on_first_write() {
    let (dir, root) = tmp_root();
    index_add(&root, "MEMORY.md", "alpha", "first memory").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(index, "- [alpha](alpha.md) — first memory\n");
    drop(dir);
}

#[test]
fn index_add_appends_without_touching_other_rows() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    // Claude Code style rows with human titles (Chinese link text).
    std::fs::write(
        root.join("MEMORY.md"),
        "- [提交纪律](cargo-commit-discipline.md) — 绝不主动 commit\n- [工作原则](working-principles.md) — 高内聚低耦合\n",
    )
    .unwrap();
    index_add(&root, "MEMORY.md", "new-entry", "brand new").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index,
        "- [提交纪律](cargo-commit-discipline.md) — 绝不主动 commit\n\
         - [工作原则](working-principles.md) — 高内聚低耦合\n\
         - [new-entry](new-entry.md) — brand new\n"
    );
    drop(dir);
}

#[test]
fn index_add_update_preserves_human_title() {
    // Shared-index contract: a refresh touches only the `— desc` tail.
    // The link text (often a human title written by Claude Code) survives.
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("MEMORY.md"),
        "- [旧标题](target.md) — 旧描述\n- [工作原则](working-principles.md) — 高内聚低耦合\n",
    )
    .unwrap();
    index_add(&root, "MEMORY.md", "target", "new description").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index,
        "- [旧标题](target.md) — new description\n\
         - [工作原则](working-principles.md) — 高内聚低耦合\n"
    );
    drop(dir);
}

#[test]
fn index_add_is_name_scoped_not_prefix_scoped() {
    // `test.md` must not match the row for `test-long.md`.
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("MEMORY.md"),
        "- [Long](test-long.md) — long desc\n",
    )
    .unwrap();
    index_add(&root, "MEMORY.md", "test", "short desc").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index,
        "- [Long](test-long.md) — long desc\n- [test](test.md) — short desc\n"
    );
    drop(dir);
}

#[test]
fn index_add_sanitizes_newlines_in_description() {
    let (dir, root) = tmp_root();
    index_add(&root, "MEMORY.md", "multi", "line one\nline two").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(index.lines().count(), 1, "{index}");
    assert!(index.contains("line one line two"), "{index}");
    drop(dir);
}

// ── index_remove ──

#[test]
fn index_remove_deletes_only_target_row() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("MEMORY.md"),
        "- [A](a.md) — a\n- [B](b.md) — b\n- [C](c.md) — c\n",
    )
    .unwrap();
    index_remove(&root, "MEMORY.md", "b").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(index, "- [A](a.md) — a\n- [C](c.md) — c\n");
    drop(dir);
}

#[test]
fn index_remove_last_row_leaves_empty_index() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("MEMORY.md"), "- [A](a.md) — a\n").unwrap();
    index_remove(&root, "MEMORY.md", "a").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(index, "");
    drop(dir);
}

#[test]
fn index_remove_missing_name_is_noop() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("MEMORY.md"), "- [A](a.md) — a\n").unwrap();
    index_remove(&root, "MEMORY.md", "ghost").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(index, "- [A](a.md) — a\n");
    drop(dir);
}

#[test]
fn index_ops_tolerate_missing_index_file() {
    let (dir, root) = tmp_root();
    index_add(&root, "MEMORY.md", "a", "a desc").unwrap();
    index_remove(&root, "MEMORY.md", "a").unwrap();
    assert_eq!(std::fs::read_to_string(root.join("MEMORY.md")).unwrap(), "");
    drop(dir);
}

// ── rebuild_index ──

#[test]
fn rebuild_index_recovers_from_memory_files() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("beta.md"),
        "---\nname: beta\ndescription: second\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n",
    )
    .unwrap();
    std::fs::write(
        root.join("alpha.md"),
        "---\nname: alpha\ndescription: first\nmetadata:\n  node_type: memory\n  type: user\n---\n\nbody\n",
    )
    .unwrap();
    // Corrupt file must be skipped, not fatal.
    std::fs::write(root.join("broken.md"), "not frontmatter at all\n").unwrap();
    // Index file itself must be excluded.
    std::fs::write(root.join("MEMORY.md"), "garbage").unwrap();

    rebuild_index(&root, "MEMORY.md").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index,
        "- [alpha](alpha.md) — first\n- [beta](beta.md) — second\n"
    );
    drop(dir);
}

#[test]
fn rebuild_index_skips_filename_name_mismatch() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("mismatch.md"),
        "---\nname: other-name\ndescription: d\n---\n\nbody\n",
    )
    .unwrap();
    rebuild_index(&root, "MEMORY.md").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index, "",
        "mismatched file must not produce a dangling link: {index}"
    );
    drop(dir);
}

#[test]
fn rebuild_index_preserves_human_title_rows() {
    // Conservative repair: a row we can regenerate is still re-emitted
    // byte-identical when one exists — the human title lives only in the
    // index and cannot be recovered from the file.
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("beta.md"),
        "---\nname: beta\ndescription: second\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n",
    )
    .unwrap();
    std::fs::write(root.join("MEMORY.md"), "- [人类标题](beta.md) — second\n").unwrap();
    rebuild_index(&root, "MEMORY.md").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(index, "- [人类标题](beta.md) — second\n");
    drop(dir);
}

#[test]
fn rebuild_index_keeps_rows_for_files_it_cannot_index() {
    // Files that fail our parse (corrupt, foreign format, name mismatch)
    // keep their existing rows — a rebuild never erases what it doesn't
    // understand. Rows whose file is gone are still dropped (dangling).
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("broken.md"), "no frontmatter here\n").unwrap();
    std::fs::write(
        root.join("mismatch.md"),
        "---\nname: other-name\ndescription: d\n---\n\nbody\n",
    )
    .unwrap();
    std::fs::write(
        root.join("MEMORY.md"),
        "- [破](broken.md) — 尽量保留\n\
         - [改名](mismatch.md) — 也要保留\n\
         - [stale](gone.md) — 文件已删\n",
    )
    .unwrap();
    rebuild_index(&root, "MEMORY.md").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index, "- [破](broken.md) — 尽量保留\n- [改名](mismatch.md) — 也要保留\n",
        "unparseable rows survive; stale rows go: {index}"
    );
    drop(dir);
}

mod merge;
mod store;
