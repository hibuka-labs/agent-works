use super::*;

// ── grandfathering (design §7): legacy entries load untouched; touching one
//    brings it under the new rules ──

/// Build a 153-entry fixture straight on disk (as if written before the limits
/// existed): 150 with short hooks and 3 with over-long (legacy) hooks. Bypasses
/// the store so the over-long hooks are not rejected at creation — they are
/// grandfathered state.
fn write_legacy_fixture(root: &std::path::Path) {
    std::fs::create_dir_all(root).unwrap();
    for i in 0..153 {
        let name = format!("legacy-{i:03}");
        // The last three carry an over-long hook (150 chars) — grandfathered.
        let desc = if i >= 150 {
            "L".repeat(150)
        } else {
            format!("legacy note {i:03}")
        };
        let file = format!(
            "---\nname: {name}\ndescription: {desc}\nmetadata:\n  node_type: memory\n  type: project\n---\n\nbody\n"
        );
        std::fs::write(root.join(format!("{name}.md")), file).unwrap();
    }
}

/// Value: protects=grandfathering — a 153-entry legacy store (including
/// over-long hooks) loads without error on the read path, and touching one
/// entry re-validates it under the new write rules (a legacy over-long hook is
/// refused on write even though it loads); fails_when=the read path retro-
/// enforces the limits and refuses to list legacy entries, or the write path
/// lets an over-long hook through because the entry already existed.
/// no_claim=the real 153-entry store's contents — this is a synthetic fixture
/// of the same shape.
#[test]
fn a_153_entry_legacy_fixture_loads_untouched_and_touching_applies_new_rules() {
    let (dir, root) = tmp_root();
    write_legacy_fixture(&root);
    let store = MemoryStore::new(root.clone(), "MEMORY.md");

    // 1) Load is lenient: all 153 come through, over-long hooks included.
    let entries = store.list_memories();
    assert_eq!(entries.len(), 153, "every legacy entry must load");
    let listing = store.render_index().unwrap();
    assert!(
        listing.contains("legacy-150"),
        "over-long entry must list: {listing}"
    );

    // 2) Touch is strict: rewriting a grandfathered over-long hook is refused
    //    under the new rules (the hook gate fires before any disk write).
    match store
        .write_memory("legacy-150", &"H".repeat(141), "project", "new body", None)
        .unwrap()
    {
        WriteOutcome::Rejected(rej) => assert_eq!(rej.rule, "hook_max_chars"),
        other => panic!("touching with an over-long hook must be refused: {other:?}"),
    }

    // 3) Touching with a hook that fits the new rules succeeds.
    let outcome = store
        .write_memory(
            "legacy-150",
            "now fits the hook budget",
            "project",
            "new body",
            None,
        )
        .unwrap();
    assert!(
        matches!(outcome, WriteOutcome::Written { created: false }),
        "a compliant touch must land: {outcome:?}"
    );
    drop(dir);
}
