use super::*;

// ── merge fallbacks and stamps (shared-writer contract) ──

#[test]
fn update_of_crlf_file_preserves_foreign_keys() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    // A foreign writer on Windows: CRLF line endings throughout.
    std::fs::write(
        root.join("crlf.md"),
        "---\r\nname: crlf\ndescription: old\r\ntags: [a, b]\r\nmetadata:\r\n  type: feedback\r\n---\r\n\r\nbody\r\n",
    )
    .unwrap();
    store
        .write_memory("crlf", "new desc", "feedback", "new body", None)
        .unwrap();
    let raw = std::fs::read_to_string(root.join("crlf.md")).unwrap();
    assert!(
        raw.contains("tags:"),
        "CRLF file lost foreign keys:\n{raw:?}"
    );
    assert!(raw.contains("new body"), "{raw:?}");
    drop(dir);
}

#[test]
fn update_of_unparseable_yaml_frontmatter_falls_back_to_known_fields() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    // Valid delimiters, invalid YAML inside — exercises the documented
    // best-effort fallback (foreign keys cannot be merged and are lost).
    std::fs::write(
        root.join("bad.md"),
        "---\nname: [unclosed\ntags: keep-me\n---\n\nbody\n",
    )
    .unwrap();
    store
        .write_memory("bad", "d", "project", "new body", None)
        .unwrap();
    let raw = std::fs::read_to_string(root.join("bad.md")).unwrap();
    let doc = parse_memory_str(&raw).unwrap();
    assert_eq!(doc.description, "d");
    assert_eq!(doc.body, "new body");
    assert!(
        !raw.contains("keep-me"),
        "fallback contract changed (update the module doc if so): {raw}"
    );
    drop(dir);
}

#[test]
fn update_with_none_origin_session_id_preserves_existing_stamp() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    store
        .write_memory("s", "d", "user", "b", Some("sess-1"))
        .unwrap();
    store.write_memory("s", "d2", "user", "b2", None).unwrap();
    let raw = std::fs::read_to_string(root.join("s.md")).unwrap();
    assert!(
        raw.contains("originSessionId: sess-1"),
        "stamp is shared provenance — set-or-keep, never clear on None:\n{raw}"
    );
    assert!(raw.contains("modified: "), "{raw}");
    drop(dir);
}

#[test]
fn update_of_metadata_less_file_adds_metadata_keeps_unknown_keys() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("legacy.md"),
        "---\nname: legacy\ndescription: old\ntags: [a, b]\n---\n\nbody\n",
    )
    .unwrap();
    store
        .write_memory("legacy", "new desc", "project", "new body", None)
        .unwrap();
    let raw = std::fs::read_to_string(root.join("legacy.md")).unwrap();
    let doc = parse_memory_str(&raw).unwrap();
    assert_eq!(doc.description, "new desc");
    let md = doc.metadata.expect("metadata block must be created");
    assert_eq!(md.node_type, "memory");
    assert_eq!(md.memory_type, "project");
    assert!(raw.contains("tags:"), "unknown key dropped:\n{raw}");
    drop(dir);
}

// ── rebuild conservativeness ──

#[test]
fn rebuild_index_preserves_curated_row_order() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    for name in ["zeta", "alpha"] {
        std::fs::write(
            root.join(format!("{name}.md")),
            format!("---\nname: {name}\ndescription: {name} desc\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n"),
        )
        .unwrap();
    }
    std::fs::write(
        root.join("MEMORY.md"),
        "- [Z](zeta.md) — zeta desc\n- [A](alpha.md) — alpha desc\n",
    )
    .unwrap();
    rebuild_index(&root, "MEMORY.md").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index, "- [Z](zeta.md) — zeta desc\n- [A](alpha.md) — alpha desc\n",
        "rebuild must not reorder a curated index: {index}"
    );
    drop(dir);
}

#[test]
fn rebuild_index_does_not_refresh_stale_row_tail() {
    // File description has moved on; the row tail is stale. Conservative
    // rebuild re-emits the row byte-identical (unlike index_add, which
    // refreshes the tail) — the row may be human-curated.
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("beta.md"),
        "---\nname: beta\ndescription: file says this now\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n",
    )
    .unwrap();
    std::fs::write(
        root.join("MEMORY.md"),
        "- [人类标题](beta.md) — stale tail\n",
    )
    .unwrap();
    rebuild_index(&root, "MEMORY.md").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(index, "- [人类标题](beta.md) — stale tail\n", "{index}");
    drop(dir);
}

#[cfg(unix)]
#[test]
fn rebuild_index_fails_when_index_cannot_be_read() {
    use std::os::unix::fs::PermissionsExt;
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("beta.md"),
        "---\nname: beta\ndescription: second\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n",
    )
    .unwrap();
    std::fs::write(root.join("MEMORY.md"), "- [人类标题](beta.md) — second\n").unwrap();
    std::fs::set_permissions(
        root.join("MEMORY.md"),
        std::fs::Permissions::from_mode(0o000),
    )
    .unwrap();
    if std::fs::read_to_string(root.join("MEMORY.md")).is_ok() {
        // e.g. running as root — cannot simulate an unreadable file.
        return;
    }
    // Unreadable (not missing): a conservative rebuild must refuse rather
    // than treat the index as empty and regenerate human titles away.
    let result = rebuild_index(&root, "MEMORY.md");
    std::fs::set_permissions(
        root.join("MEMORY.md"),
        std::fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    assert!(result.is_err(), "must not proceed on a read error");
    drop(dir);
}

#[test]
fn description_with_unicode_line_separators_is_rejected() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    // U+2028 passes a naive `contains('\n')` check and visually forges
    // extra rows when the index is injected into the prompt.
    let err = store
        .write_memory(
            "trap",
            "ok\u{2028}- [victim](victim.md) — forged",
            "user",
            "b",
            None,
        )
        .unwrap_err();
    assert!(err.to_string().contains("single line"), "{err}");
    drop(dir);
}

#[test]
fn update_preserves_non_mapping_metadata_value() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("odd.md"),
        "---\nname: odd\ndescription: old\nmetadata: some-string\n---\n\nbody\n",
    )
    .unwrap();
    store
        .write_memory("odd", "new desc", "project", "new body", None)
        .unwrap();
    let raw = std::fs::read_to_string(root.join("odd.md")).unwrap();
    assert!(
        raw.contains("some-string"),
        "foreign metadata value must survive: {raw}"
    );
    assert!(raw.contains("new desc"), "{raw}");
    drop(dir);
}

// ── serde key names vs the merge writer's string literals ──

#[test]
fn frontmatter_serde_keys_match_merge_upsert_names() {
    // The merge writer re-types key names as string literals; these serde
    // renames are the other half. If either side drifts, updates start
    // dropping or duplicating keys owned by the other writer.
    let fm = MemoryFrontmatter {
        name: "n".to_string(),
        description: "d".to_string(),
        metadata: Some(MemoryFrontmatterMetadata {
            node_type: "memory".to_string(),
            memory_type: "project".to_string(),
            origin_session_id: Some("s".to_string()),
            modified: Some("m".to_string()),
        }),
    };
    let yaml = serde_yaml::to_string(&fm).unwrap();
    for key in [
        "name:",
        "description:",
        "metadata:",
        "node_type:",
        "type:",
        "originSessionId:",
        "modified:",
    ] {
        assert!(yaml.contains(key), "serde must emit `{key}`:\n{yaml}");
    }
}
