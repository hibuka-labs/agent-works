use super::*;

// ── truth projection: the listing is rebuilt from disk, never the stale
//    MEMORY.md snapshot (design §4) ──

/// Value: protects=the source-of-truth rebuild — a memory file deleted by the
/// other writer (Claude Code) drops out of the listing at once, so a dangling
/// row never survives; fails_when=`memory_list` reads the raw MEMORY.md and
/// keeps showing a name whose file is gone.
/// no_claim=that Claude Code deletes often — this is the worst-case contract.
#[test]
fn a_file_deleted_by_the_other_writer_drops_out_of_the_listing() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    store
        .write_memory("kept", "k", "project", "b", None)
        .unwrap();
    store
        .write_memory("gone", "g", "project", "b", None)
        .unwrap();
    // The stale snapshot still has both rows.
    assert!(store.read_index().contains("(gone.md)"));
    // The other writer deletes `gone.md` behind the store's back.
    std::fs::remove_file(root.join("gone.md")).unwrap();

    let listing = store.render_index().unwrap();
    assert!(
        !listing.contains("(gone.md)"),
        "dangling row survived: {listing}"
    );
    assert!(
        listing.contains("(kept.md)"),
        "kept row must remain: {listing}"
    );
    drop(dir);
}

/// Value: protects=the disk-wins rule — the row's description comes from the
/// file's frontmatter, so a hand-edited MEMORY.md tail is corrected, not
/// trusted; fails_when=the listing is read from MEMORY.md and a hand edit
/// rewrites what the memory says. no_claim=the reverse (frontmatter hand edits
/// winning over a curated index tail is the intent).
#[test]
fn the_row_description_comes_from_the_file_not_a_hand_edited_index() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    store
        .write_memory("fact", "frontmatter desc", "project", "body", None)
        .unwrap();
    // Someone hand-edits the index tail to say something else.
    std::fs::write(
        root.join("MEMORY.md"),
        "- [fact](fact.md) — HAND-EDITED DESC\n",
    )
    .unwrap();

    let listing = store.render_index().unwrap();
    assert!(
        listing.contains("frontmatter desc"),
        "disk truth must win: {listing}"
    );
    assert!(
        !listing.contains("HAND-EDITED DESC"),
        "the hand edit must not leak into the listing: {listing}"
    );
    drop(dir);
}

/// Value: protects=the curated human title — the `[title]` link text lives only
/// in the index and is carried across a rebuild (a title cannot be recovered
/// from the file), while the description is re-derived from disk;
/// fails_when=a rebuild drops the human title and renames every row to its slug.
/// no_claim=the order beyond "prior rows first, new files appended sorted".
#[test]
fn a_rebuild_keeps_the_curated_human_title_but_refreshes_the_description() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    store
        .write_memory("beta", "old desc", "project", "body", None)
        .unwrap();
    // The other writer curates a human title in the index.
    std::fs::write(root.join("MEMORY.md"), "- [人类标题](beta.md) — old desc\n").unwrap();

    // A store write regenerates the whole index from disk.
    store
        .write_memory("alpha", "a", "project", "b", None)
        .unwrap();
    let index = store.read_index();
    assert!(
        index.contains("- [人类标题](beta.md)"),
        "the human title must survive the rebuild: {index}"
    );
    // beta's description is re-derived from its frontmatter ("old desc").
    assert!(index.contains("old desc"), "{index}");
    assert!(index.contains("(alpha.md)"), "new file appended: {index}");
    drop(dir);
}

/// Value: protects=the other writer's additions — a memory file dropped in
/// directly (Claude Code writes the file) shows up in the listing even though
/// it never went through `memory_write`; fails_when=the listing only reflects
/// files the store wrote, hiding what the other writer added.
/// no_claim=that its frontmatter is valid — a corrupt file is skipped.
#[test]
fn a_file_added_by_the_other_writer_appears_in_the_listing() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    std::fs::create_dir_all(&root).unwrap();
    // Claude Code writes a file straight to disk (no store, no MEMORY.md row).
    std::fs::write(
        root.join("foreign.md"),
        "---\nname: foreign\ndescription: added by cc\nmetadata:\n  type: project\n---\n\nbody\n",
    )
    .unwrap();

    let listing = store.render_index().unwrap();
    assert!(
        listing.contains("(foreign.md)") && listing.contains("added by cc"),
        "the other writer's file must appear: {listing}"
    );
    drop(dir);
}
