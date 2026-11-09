use super::*;

// ── MemoryStore: write / update / read / delete / list ──

#[test]
fn store_first_write_creates_dir_file_and_index() {
    let (dir, root) = tmp_root();
    assert!(!root.exists());
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    store
        .write_memory("my-note", "a note", "project", "the body", Some("sess-1"))
        .unwrap();
    assert!(root.is_dir(), "memory dir must be auto-created");
    let raw = std::fs::read_to_string(root.join("my-note.md")).unwrap();
    assert!(raw.contains("name: my-note"));
    assert!(raw.contains("originSessionId: sess-1"));
    assert!(raw.contains("modified: "));
    assert!(raw.ends_with("the body"));
    let index = store.read_index();
    assert!(
        index.contains("- [my-note](my-note.md) — a note"),
        "{index}"
    );
    drop(dir);
}

#[test]
fn store_same_name_update_writes_single_file_with_second_content() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    store
        .write_memory("dup", "first desc", "project", "first body", None)
        .unwrap();
    store
        .write_memory("dup", "second desc", "feedback", "second body", None)
        .unwrap();

    let files: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".md") && n != "MEMORY.md")
        .collect();
    assert_eq!(files, vec!["dup.md".to_string()], "no duplicate files");

    let raw = std::fs::read_to_string(root.join("dup.md")).unwrap();
    assert!(raw.contains("second body"));
    assert!(!raw.contains("first body"));

    let index = store.read_index();
    assert_eq!(index.matches("(dup.md)").count(), 1, "{index}");
    assert!(index.contains("second desc"), "{index}");
    drop(dir);
}

#[test]
fn store_update_preserves_unknown_frontmatter_fields() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("extra.md"), EXTRA_FIELDS_FILE).unwrap();

    store
        .write_memory("extra", "updated desc", "project", "new body", None)
        .unwrap();

    let raw = std::fs::read_to_string(root.join("extra.md")).unwrap();
    let doc = parse_memory_str(&raw).unwrap();
    assert_eq!(doc.description, "updated desc");
    assert_eq!(doc.body, "new body");
    // Keys neither writer owns must survive the update (round-trip).
    assert!(raw.contains("tags:"), "top-level `tags` dropped:\n{raw}");
    assert!(
        raw.contains("[a, b]") || raw.contains("- a"),
        "`tags` values dropped:\n{raw}"
    );
    assert!(
        raw.contains("created: 2026-01-01"),
        "metadata.created dropped:\n{raw}"
    );
    drop(dir);
}

#[test]
fn store_read_returns_raw_file_content() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    store
        .write_memory("note", "d", "user", "remember this", None)
        .unwrap();
    let raw = store.read_memory("note").unwrap();
    assert!(raw.starts_with("---\n"));
    assert!(raw.contains("remember this"));
    drop(dir);
}

#[test]
fn store_read_missing_is_error() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root, "MEMORY.md");
    let err = store.read_memory("ghost").unwrap_err().to_string();
    assert!(err.contains("not found"), "{err}");
    drop(dir);
}

#[test]
fn store_delete_removes_file_and_index_row() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    store
        .write_memory("gone", "d1", "user", "b1", None)
        .unwrap();
    store
        .write_memory("kept", "d2", "user", "b2", None)
        .unwrap();

    store.delete_memory("gone").unwrap();
    assert!(!root.join("gone.md").exists());
    assert!(root.join("kept.md").exists());
    let index = store.read_index();
    assert!(!index.contains("(gone.md)"), "{index}");
    assert!(index.contains("(kept.md)"), "{index}");
    drop(dir);
}

#[test]
fn store_delete_missing_is_error() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root, "MEMORY.md");
    let err = store.delete_memory("ghost").unwrap_err().to_string();
    assert!(err.contains("not found"), "{err}");
    drop(dir);
}

#[test]
fn store_write_rejects_invalid_name_and_type() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    assert!(store.write_memory("../x", "d", "user", "b", None).is_err());
    assert!(
        store
            .write_memory("ok-name", "d", "wrong-type", "b", None)
            .is_err()
    );
    assert!(
        store
            .write_memory("ok-name", "multi\nline", "user", "b", None)
            .is_err()
    );
    assert!(
        !root.join("ok-name.md").exists(),
        "nothing may be written on rejection"
    );
    drop(dir);
}

#[test]
fn store_list_skips_corrupt_and_sorts() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    store
        .write_memory("zeta", "last", "project", "b", None)
        .unwrap();
    store
        .write_memory("alpha", "first", "user", "b", None)
        .unwrap();
    std::fs::write(root.join("corrupt.md"), "garbage without frontmatter").unwrap();

    let entries = store.list_memories();
    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["alpha", "zeta"]);
    assert_eq!(entries[0].memory_type.as_deref(), Some("user"));
    drop(dir);
}

#[test]
fn store_list_missing_dir_is_empty() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root, "MEMORY.md");
    assert!(store.list_memories().is_empty());
    assert_eq!(store.read_index(), "");
    drop(dir);
}

// ── concurrency (in-process) ──

#[test]
fn concurrent_writes_keep_index_complete() {
    let (dir, root) = tmp_root();
    let store = Arc::new(MemoryStore::new(root.clone(), "MEMORY.md"));
    const N: usize = 32;

    let mut handles = Vec::new();
    for i in 0..N {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            let name = format!("mem-{i:03}");
            store
                .write_memory(
                    &name,
                    &format!("desc {i}"),
                    "project",
                    &format!("body {i}"),
                    None,
                )
                .unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let index = store.read_index();
    for i in 0..N {
        let name = format!("mem-{i:03}");
        assert!(
            index.contains(&format!("({name}.md)")),
            "index is missing `{name}`:\n{index}"
        );
    }
    assert_eq!(index.lines().count(), N, "no rows may be lost:\n{index}");

    let files = store.list_memories();
    assert_eq!(files.len(), N);
    drop(dir);
}

#[test]
fn concurrent_same_name_writes_leave_consistent_state() {
    let (dir, root) = tmp_root();
    let store = Arc::new(MemoryStore::new(root.clone(), "MEMORY.md"));

    let mut handles = Vec::new();
    for i in 0..16 {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            store
                .write_memory(
                    "shared",
                    &format!("desc {i}"),
                    "user",
                    &format!("body {i}"),
                    None,
                )
                .unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // One file, one row, and the file parses back to whatever row exists.
    let files: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".md"))
        .collect();
    assert_eq!(files.len(), 2, "index + exactly one memory file: {files:?}");
    let index = store.read_index();
    assert_eq!(index.matches("(shared.md)").count(), 1, "{index}");
    drop(dir);
}

// ── shared-index anchor safety ──
//
// Row identity is the row's own link target (`- [title](name.md)`), never a
// substring of some other row's description — descriptions are free text
// and may embed links to other memories.

#[test]
fn index_add_anchor_inside_other_row_description_does_not_hijack() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("MEMORY.md"),
        "- [beta](beta.md) — see [x](alpha.md)\n",
    )
    .unwrap();
    index_add(&root, "MEMORY.md", "alpha", "new").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert!(
        index.contains("- [alpha](alpha.md) — new"),
        "alpha row must be created despite the anchor in beta's description: {index}"
    );
    assert!(
        index.contains("- [beta](beta.md) — see [x](alpha.md)"),
        "beta's row must be untouched: {index}"
    );
    drop(dir);
}

#[test]
fn index_remove_ignores_anchor_inside_other_rows_description() {
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("MEMORY.md"),
        "- [alpha](alpha.md) — mine\n- [beta](beta.md) — see [x](alpha.md)\n",
    )
    .unwrap();
    index_remove(&root, "MEMORY.md", "alpha").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index, "- [beta](beta.md) — see [x](alpha.md)\n",
        "only alpha's own row may go: {index}"
    );
    drop(dir);
}

#[test]
fn rebuild_binds_rows_by_row_link_not_description_substring() {
    // beta's description mentions alpha's link and sits ABOVE alpha's
    // human-titled row. Each file must bind to its own row.
    let (dir, root) = tmp_root();
    std::fs::create_dir_all(&root).unwrap();
    for name in ["alpha", "beta"] {
        std::fs::write(
            root.join(format!("{name}.md")),
            format!("---\nname: {name}\ndescription: d\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n"),
        )
        .unwrap();
    }
    std::fs::write(
        root.join("MEMORY.md"),
        "- [beta](beta.md) — see [x](alpha.md)\n- [人类](alpha.md) — 旧\n",
    )
    .unwrap();
    rebuild_index(&root, "MEMORY.md").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(index.lines().count(), 2, "no duplication, no loss: {index}");
    assert!(index.contains("- [人类](alpha.md) — 旧"), "{index}");
    assert!(
        index.contains("- [beta](beta.md) — see [x](alpha.md)"),
        "{index}"
    );
    drop(dir);
}
