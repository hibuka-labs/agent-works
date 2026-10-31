//! P2 并存对账: the same task on two paths — phi-tools' 22 tool round-trips
//! (Path A) vs. the `b.*` code face (Path B) — compared on what the model
//! actually pays: prompt-face tokens and step count.
//!
//! Numbers are measured here with one tokenizer on both sides; the step
//! account is spelled out and pinned to the real tool names so it cannot
//! quietly drift. A live-model run of both paths is the follow-up if the
//! structural account ever stops matching practice.

use phi_ns::NamespaceProvider;

/// The task both paths perform: open the page, find the link whose text is
/// "next", click it, screenshot where that lands.
const TASK_PATH_A: &[&str] = &[
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_screenshot",
];

/// Path B writes all of that in one cell — one round-trip.
const TASK_PATH_B_CELLS: usize = 1;

fn tokens(text: &str) -> (usize, usize) {
    (
        tiktoken_rs::cl100k_base_singleton()
            .encode_ordinary(text)
            .len(),
        tiktoken_rs::o200k_base_singleton()
            .encode_ordinary(text)
            .len(),
    )
}

#[test]
fn the_same_task_costs_less_on_the_b_face() {
    // ── Path A: the 22-tool face as it rides in an LLM request ──
    let toolset = phi_tools::BrowserToolset::lazy(phi_tools::LaunchOptions::default());
    let tools = phi_tools::browser_tools(&toolset);
    assert_eq!(tools.len(), 22, "the suite is the 22-tool registry");
    let face_a = serde_json::to_string(
        &tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.schema(),
                })
            })
            .collect::<Vec<_>>(),
    )
    .expect("tool face serializes");

    // ── Path B: the seed list as it rides in the prompt ──
    let face_b = phi_ns_browser::PhiNsBrowser::new().api_doc();

    let (a_cl, a_o2) = tokens(&face_a);
    let (b_cl, b_o2) = tokens(&face_b);
    println!(
        "[reconcile] task: open page → find link 'next' → click → screenshot\n\
         [reconcile] Path A: 22 tool schemas, {} chars, cl100k={} tok, o200k={} tok, {} steps\n\
         [reconcile] Path B: b.* seed list, {} chars, cl100k={} tok, o200k={} tok, {} step",
        face_a.len(),
        a_cl,
        a_o2,
        TASK_PATH_A.len(),
        face_b.len(),
        b_cl,
        b_o2,
        TASK_PATH_B_CELLS,
    );

    // The claim under test: fewer prompt tokens and fewer round-trips.
    assert!(
        b_cl < a_cl,
        "the seed list ({b_cl}) must cost less than the 22-tool face ({a_cl})"
    );
    assert!(
        b_o2 < a_o2,
        "the seed list ({b_o2}) must cost less than the 22-tool face ({a_o2})"
    );
    assert!(
        TASK_PATH_B_CELLS < TASK_PATH_A.len(),
        "one cell must beat four tool round-trips"
    );

    // The step account cites only tools that still exist.
    for name in TASK_PATH_A {
        assert!(
            tools.iter().any(|t| t.name() == *name),
            "the account cites {name}, which the suite does not ship"
        );
    }
}

#[test]
fn both_faces_cover_the_task_end_to_end() {
    // A comparison is only fair if both paths can actually do the task:
    // Path A needs navigate + a way to find the link + click + screenshot,
    // Path B needs the same verbs behind one namespace.
    let toolset = phi_tools::BrowserToolset::lazy(phi_tools::LaunchOptions::default());
    let names: Vec<&str> = phi_tools::browser_tools(&toolset)
        .iter()
        .map(|t| t.name())
        .collect();
    for needed in [
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_screenshot",
    ] {
        assert!(names.contains(&needed), "Path A lost {needed}");
    }

    let methods: Vec<&str> = phi_ns_browser::PhiNsBrowser::new()
        .methods()
        .into_iter()
        .map(|m| m.name)
        .collect();
    for needed in ["b.goto", "b.query", "b.click", "b.screenshot"] {
        assert!(methods.contains(&needed), "Path B lost {needed}");
    }
}
