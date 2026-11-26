use super::*;

// ── write-path limit gates (design §5) + the five-field failure contract (§6) ──

/// Value: protects=the hook gate rejects an over-long description with the full
/// five-field failure contract and a `candidate` that passes on retry;
/// fails_when=the hook gate is dropped, or the contract loses a field, or the
/// candidate is not actually ≤ the limit (so the one-shot retry bounces).
/// no_claim=the candidate reads well — it is a mechanical char-trim; semantic
/// compression is the model's job (`fix` says so).
#[test]
fn hook_over_limit_rejects_with_five_field_contract_and_a_retryable_candidate() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    let long_hook = "h".repeat(141);
    match store
        .write_memory("big-hook", &long_hook, "project", "body", None)
        .unwrap()
    {
        WriteOutcome::Rejected(rej) => {
            assert_eq!(rej.rule, "hook_max_chars");
            assert_eq!(rej.limit, 80);
            assert_eq!(rej.actual, 141);
            assert_eq!(rej.name, "big-hook");
            assert!(!rej.fix.is_empty(), "must say how to fix");
            // The candidate is a one-shot pass: ≤ the limit and non-empty.
            assert!(!rej.candidate.is_empty(), "must hand back a candidate");
            assert!(
                rej.candidate.chars().count() <= 80,
                "candidate must fit the hook limit: {}",
                rej.candidate.chars().count()
            );
            // The wire shape carries the same five fields.
            let wire = serde_json::from_str::<serde_json::Value>(&rej.to_wire()).unwrap();
            for key in ["ok", "rule", "limit", "actual", "fix", "candidate"] {
                assert!(wire.get(key).is_some(), "wire missing `{key}`: {wire}");
            }
            assert_eq!(wire["ok"], false);
            // Nothing was persisted on rejection.
            assert!(!root.join("big-hook.md").exists(), "no file on rejection");
        }
        other => panic!("expected a hook rejection, got {other:?}"),
    }

    // Retry with the candidate → accepted in one shot.
    let candidate = {
        let rej = match store
            .write_memory("big-hook", &long_hook, "project", "body", None)
            .unwrap()
        {
            WriteOutcome::Rejected(r) => r,
            _ => unreachable!(),
        };
        rej.candidate
    };
    let outcome = store
        .write_memory("big-hook", &candidate, "project", "body", None)
        .unwrap();
    assert!(
        matches!(outcome, WriteOutcome::Written { created: true }),
        "retry with the candidate must succeed: {outcome:?}"
    );
    drop(dir);
}

/// Value: protects=the body gate rejects over the char cap and accepts at it;
/// fails_when=the body gate is dropped or off-by-one (3000 rejected / 3001
/// accepted). no_claim=that 3000 chars is a good budget — it is the configured
/// default, and the product may override it.
#[test]
fn body_over_limit_rejects_and_at_limit_succeeds() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md");
    // 3001 → reject.
    match store
        .write_memory("big-body", "d", "project", &"b".repeat(3001), None)
        .unwrap()
    {
        WriteOutcome::Rejected(rej) => {
            assert_eq!(rej.rule, "body_max_chars");
            assert_eq!(rej.limit, 3000);
            assert_eq!(rej.actual, 3001);
            assert!(!rej.fix.is_empty());
        }
        other => panic!("expected a body rejection, got {other:?}"),
    }
    // 3000 → accept.
    let outcome = store
        .write_memory("big-body", "d", "project", &"b".repeat(3000), None)
        .unwrap();
    assert!(
        matches!(outcome, WriteOutcome::Written { .. }),
        "3000 chars must be accepted: {outcome:?}"
    );
    drop(dir);
}

/// Value: protects=the index gate lands exactly on the cap (accepted) and
/// rejects one byte over with a full rollback — index and body both back to the
/// pre-write state (account-matter reconciliation); fails_when=the boundary is
/// off-by-one (10240 rejected or 10241 accepted) or a rejected write leaves the
/// body rolled back but the index moved (or vice versa).
/// no_claim=the exact per-row byte arithmetic — the test measures it at runtime.
#[test]
fn index_lands_on_10240_and_10241_is_rejected_with_full_rollback() {
    let (dir, root) = tmp_root();
    let store = MemoryStore::new(root.clone(), "MEMORY.md"); // default index_max_bytes = 10240
    let cap = store.limits().index_max_bytes;
    assert_eq!(cap, 10_240, "the default cap must be 10,240 bytes");

    // Learn the fixed byte cost of a uniform 5-char-name row (em-dash and all).
    store
        .write_memory("e0000", "x", "project", "b", None)
        .unwrap();
    let one_row = store.render_index().unwrap().len(); // row(desc=1) + "\n"
    let fixed = one_row - 2; // row content = fixed + desc; trailing newline = 1

    // Fill uniform rows (desc 50) leaving room to tune a final row onto the cap.
    let d = 50usize;
    let row_size = fixed + d + 1; // bytes a filler row adds
    let n = (cap - (fixed + 2)) / row_size;
    assert!(n >= 2, "need a few filler rows: n={n}");
    for k in 0..n {
        let name = format!("e{k:04}");
        store
            .write_memory(&name, &"p".repeat(d), "project", "b", None)
            .unwrap();
    }

    // Tune a final row (name sorts last → appended) to land EXACTLY on the cap.
    let size = store.render_index().unwrap().len();
    let desc_len = cap - size - fixed - 1;
    assert!(desc_len >= 1, "no room for the tuned row: size={size}");
    store
        .write_memory("zzzzz", &"q".repeat(desc_len), "project", "b", None)
        .unwrap();
    assert_eq!(
        store.render_index().unwrap().len(),
        cap,
        "must land exactly on 10,240"
    );

    // One byte over → reject + roll back. Update `zzzzz` to a longer description.
    let before_body = std::fs::read_to_string(root.join("zzzzz.md")).unwrap();
    let before_index = store.read_index();
    match store
        .write_memory("zzzzz", &"q".repeat(desc_len + 1), "project", "b", None)
        .unwrap()
    {
        WriteOutcome::Rejected(rej) => {
            assert_eq!(rej.rule, "index_max_bytes");
            assert_eq!(rej.limit, cap);
            assert_eq!(rej.actual, cap + 1, "actual must report the over-size");
        }
        other => panic!("expected an index rejection, got {other:?}"),
    }
    assert_eq!(
        store.read_index(),
        before_index,
        "index must be unchanged after a rejected write (rolled back)"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("zzzzz.md")).unwrap(),
        before_body,
        "body must be restored to its pre-write state after a rejected write"
    );
    drop(dir);
}

/// Value: protects=the hot-zone invariant — 100 writes/updates never push the
/// index past the cap (each accepted write re-checks the regenerated index);
/// fails_when=the gate is checked against a stale index so an over-cap write
/// slips through and the final index exceeds `index_max_bytes`.
/// no_claim=that the cap is the right size; only that it is never exceeded.
#[test]
fn a_hundred_writes_keep_the_index_within_the_cap() {
    let (dir, _root) = tmp_root();
    let store = MemoryStore::new(_root.clone(), "MEMORY.md");
    let cap = store.limits().index_max_bytes;
    let mut max_seen = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for i in 0..100 {
        // Each row carries a hook at the max width (80) so 100 rows overflow the
        // index cap — the gate must bite partway through and hold the invariant.
        let name = format!("w{i:03}");
        let outcome = store
            .write_memory(&name, &"d".repeat(80), "project", "body", None)
            .unwrap();
        match outcome {
            WriteOutcome::Written { .. } => accepted += 1,
            WriteOutcome::Rejected(_) => rejected += 1,
        }
        let size = store.render_index().unwrap().len();
        max_seen = max_seen.max(size);
    }
    assert!(accepted > 0, "some writes must land");
    assert_eq!(
        accepted + rejected,
        100,
        "every write is accepted or rejected"
    );
    // The invariant: the index never crossed the cap.
    assert!(
        max_seen <= cap,
        "index reached {max_seen}, over the {cap} cap"
    );
    assert!(
        rejected > 0,
        "the cap must bite under 100 fills: {accepted}"
    );
    drop(dir);
}

/// Value: protects=the rollback seam — `limits.enabled: false` short-circuits
/// the hard gates so an over-long write still lands (the one-switch undo for a
/// bad limits rollout); fails_when=the enabled flag is ignored and the gates
/// stay armed when a product turns them off.
/// no_claim=that disabling limits is a good idea — it is the emergency seam.
#[test]
fn disabling_limits_short_circuits_the_hard_gates() {
    let (dir, root) = tmp_root();
    let limits = MemoryLimits {
        enabled: false,
        ..Default::default()
    };
    let store = MemoryStore::with_limits(root.clone(), "MEMORY.md", limits);
    // Over hook AND over body: both would reject with limits on.
    let outcome = store
        .write_memory(
            "no-gate",
            &"h".repeat(200),
            "project",
            &"b".repeat(5000),
            None,
        )
        .unwrap();
    assert!(
        matches!(outcome, WriteOutcome::Written { .. }),
        "limits disabled must accept: {outcome:?}"
    );
    assert!(root.join("no-gate.md").exists());
    drop(dir);
}

/// Value: protects=the product override entry — a caller-supplied
/// `MemoryLimits` is what the store enforces (phimint's `[memory]` section
/// overrides the framework defaults); fails_when=the store ignores `with_limits`
/// and always enforces the defaults, so a product cannot tighten or loosen a
/// gate. no_claim=the default values themselves.
#[test]
fn the_store_enforces_the_limits_it_was_given_not_the_defaults() {
    let (dir, root) = tmp_root();
    let limits = MemoryLimits {
        hook_max_chars: 5, // tighten the hook gate
        ..Default::default()
    };
    let store = MemoryStore::with_limits(root.clone(), "MEMORY.md", limits);
    match store
        .write_memory("tiny", "123456", "project", "b", None)
        .unwrap()
    {
        WriteOutcome::Rejected(rej) => {
            assert_eq!(rej.rule, "hook_max_chars");
            assert_eq!(rej.limit, 5, "must enforce the override, not 80");
            assert_eq!(rej.actual, 6);
        }
        other => panic!("the 5-char override must reject a 6-char hook: {other:?}"),
    }
    // A 5-char hook passes under the tightened gate.
    assert!(matches!(
        store
            .write_memory("tiny", "12345", "project", "b", None)
            .unwrap(),
        WriteOutcome::Written { .. }
    ));
    drop(dir);
}
