//! P2 acceptance (sdk-assembly §10) — the assembled stack: `codexec` with the
//! real `phi-ns-browser` pack, lined up in the挂接层 where assembly lives
//! (§9.3: neither the engine nor the pack may know the other).
//!
//! Three claims are judged here:
//!
//! 1. the codexec §10 touchstone two cells run with `b.*` — state survives;
//! 2. the picture really reaches the model — `ImageHandle` for the code,
//!    `CellEvent::Image` for the engine, and a multimodal block on top;
//! 3. `b.click` is held while `b.screenshot` flows — the policy split.

use codexec::{
    Action, CellEvent, CodeCellExecutor, ExecutorBuilder, FileJournal, JournalHandle, Limits,
    MethodPolicy, Observation, PolicyDecision, PythonWorker,
};
use phi_ns::ImageSource;
use phi_ns_browser::PhiNsBrowser;
use std::path::Path;
use std::sync::Arc;

const FIXTURE: &str = r##"<!doctype html>
<html>
<head><title>P2 Fixture</title></head>
<body>
  <div class="toolbar"><button id="export-btn">导出</button></div>
  <p id="out">ready</p>
  <script>
    document.getElementById('export-btn').addEventListener('click', function () {
      document.getElementById('out').textContent = 'exported';
    });
  </script>
</body>
</html>
"##;

struct Stage {
    _dir: tempfile::TempDir,
    page_url: String,
    shots: std::path::PathBuf,
    journal: std::path::PathBuf,
    executor: CodeCellExecutor<PythonWorker>,
}

async fn stage(policy: std::sync::Arc<dyn codexec::Policy>) -> Stage {
    let dir = tempfile::tempdir().expect("tempdir");
    let page = dir.path().join("fixture.html");
    std::fs::write(&page, FIXTURE).expect("fixture writes");
    let page_url = format!("file://{}", page.display());
    let shots = dir.path().join("shots");
    let journal = dir.path().join("journal.jsonl");

    let pack = PhiNsBrowser::new().with_shot_dir(&shots);
    let limits = Limits::default();
    let file = FileJournal::open(&journal).expect("journal opens");
    let executor = ExecutorBuilder::new(JournalHandle::new(Box::new(file)))
        .with_policy(policy)
        .with_limits(limits)
        .with_ns(pack)
        .build(move |registry| PythonWorker::spawn(registry, limits))
        .await
        .expect("engine assembles");

    Stage {
        _dir: dir,
        page_url,
        shots,
        journal,
        executor,
    }
}

fn cells(codes: &[&str]) -> Vec<Action> {
    codes
        .iter()
        .map(|code| Action::CodeCell {
            intent: String::new(),
            code: (*code).to_string(),
        })
        .collect()
}

fn journal_rows(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

/// What the host layer sends upstream from one observation. The engine stays
/// format-neutral ([`Observation`] + [`CellEvent`]); building provider blocks
/// is the application's job, spelled out here so acceptance can point at it.
fn model_blocks(obs: &Observation) -> Vec<serde_json::Value> {
    let mut content = vec![serde_json::json!({"type": "text", "text": obs.stdout})];
    for event in &obs.events {
        if let CellEvent::Image(image) = event {
            content.push(serde_json::json!({
                "type": "image",
                // One codec, one home: phi-ns owns the wire shape.
                "source": image.to_wire(),
            }));
        }
    }
    content
}

/// The §10 touchstone with the real pack: cell 1 leaves `els` and `shot`
/// behind, cell 2 uses them — across *steps*, which is the resident
/// interpreter claim, not just one multi-block step.
#[tokio::test]
async fn the_touchstone_two_cells_run_with_the_browser_pack() {
    let mut s = stage(Arc::new(codexec::AllowAll) as Arc<dyn codexec::Policy>).await;

    let open = s
        .executor
        .execute(cells(&[&format!(
            "b.goto({url})\nels = b.query('button')\nshot = b.screenshot('the export button row')",
            url = serde_json::to_string(&s.page_url).unwrap()
        )]))
        .await;
    assert!(open.error.is_none(), "{open:?}");
    assert_eq!(open.events.len(), 1, "{open:?}");

    // Next step, same worker: the previous cell's names are still alive.
    let use_them = s
        .executor
        .execute(cells(&["print(els[0].text, '|', shot.alt)"]))
        .await;
    assert!(use_them.error.is_none(), "{use_them:?}");
    assert!(
        use_them.stdout.contains("导出 | the export button row"),
        "cell 2 must see cell 1's state:\n{}",
        use_them.stdout
    );

    // And the multi-block one-step form (D4): three blocks, one round-trip.
    let one_step = s
        .executor
        .execute(cells(&[
            "names = [el.text for el in els]",
            "shot2 = b.screenshot('second look')",
            "print(len(names), names[0], shot2.alt)",
        ]))
        .await;
    assert!(one_step.error.is_none(), "{one_step:?}");
    assert!(
        one_step.stdout.contains("1 导出 second look"),
        "multi-block step:\n{}",
        one_step.stdout
    );
}

/// "图片真的到模型": the code gets an `ImageHandle`, the engine gets a
/// `CellEvent::Image`, and both point at the same bytes — then the block a
/// host would send carries the picture plus its `alt` sentence.
#[tokio::test]
async fn the_screenshot_reaches_the_model_on_both_channels() {
    let mut s = stage(Arc::new(codexec::AllowAll) as Arc<dyn codexec::Policy>).await;

    let obs = s
        .executor
        .execute(cells(&[
            &format!(
                "b.goto({url})\nshot = b.screenshot('the export button row')\nprint(shot.alt)\nprint(shot.path)",
                url = serde_json::to_string(&s.page_url).unwrap()
            ),
        ]))
        .await;
    assert!(obs.error.is_none(), "{obs:?}");

    // Channel one: the worker-side handle the code can pass around.
    assert!(
        obs.stdout.contains("the export button row"),
        "ImageHandle.alt reached the text stream:\n{}",
        obs.stdout
    );

    // Channel two: the engine-side event.
    assert_eq!(obs.events.len(), 1, "{obs:?}");
    let CellEvent::Image(image) = &obs.events[0] else {
        panic!("expected an image event, got {:?}", obs.events[0]);
    };
    assert_eq!(image.mime, "image/png");
    assert_eq!(image.alt, "the export button row");
    let ImageSource::Path(path) = &image.source else {
        panic!("default channel is on-disk, got {:?}", image.source);
    };
    assert!(
        path.starts_with(&s.shots),
        "shot lands in the store: {path:?}"
    );
    let bytes = std::fs::read(path).expect("shot bytes");
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "a real PNG");
    // One source, two channels: the handle the code printed is this path.
    assert!(
        obs.stdout.contains(&path.display().to_string()),
        "ImageHandle.path and ImageRef.source disagree:\n{}",
        obs.stdout
    );

    // Rule 3's floor line, rule 1's silence on bytes.
    assert!(
        obs.stdout.contains("image: the export button row"),
        "the alt floor line is what survives a trimmed event:\n{}",
        obs.stdout
    );
    assert!(!obs.stdout.contains("inline_base64"), "{obs:?}");
    assert!(!obs.stdout.contains("iVBOR"), "{obs:?}");

    // The multimodal block a host builds from this: text + picture.
    let blocks = model_blocks(&obs);
    assert_eq!(blocks.len(), 2, "{blocks:?}");
    assert_eq!(blocks[0]["type"], "text");
    assert!(
        blocks[0]["text"]
            .as_str()
            .expect("text")
            .contains("image: the export button row")
    );
    assert_eq!(blocks[1]["type"], "image");
    assert_eq!(blocks[1]["source"]["mime"], "image/png");
    assert_eq!(blocks[1]["source"]["alt"], "the export button row");
    assert_eq!(
        blocks[1]["source"]["path"],
        path.display().to_string(),
        "the block points at the same bytes"
    );
}

/// The policy split the names exist for: `b.click` (input) is held while
/// `b.screenshot` (capture) flows — both `Device` under the hood, so only a
/// method-keyed table can say "input asks, capture flows". And every decision
/// lands in the journal.
#[tokio::test]
async fn click_is_held_while_screenshot_flows() {
    let policy = MethodPolicy::new(PolicyDecision::Allow).with("b.click", PolicyDecision::Ask);
    let mut s = stage(Arc::new(policy) as Arc<dyn codexec::Policy>).await;

    let open = s
        .executor
        .execute(cells(&[&format!(
            "b.goto({url})",
            url = serde_json::to_string(&s.page_url).unwrap()
        )]))
        .await;
    assert!(open.error.is_none(), "{open:?}");

    // 能拦: the click is held, the effect never ran.
    let held = s.executor.execute(cells(&["b.click('#export-btn')"])).await;
    assert!(held.is_policy_hold(), "{held:?}");
    assert!(held.events.is_empty(), "{held:?}");
    let page = s
        .executor
        .execute(cells(&[
            "print(b.eval(\"document.getElementById('out').textContent\"))",
        ]))
        .await;
    assert!(
        page.stdout.contains("ready"),
        "the held click left no side effect:\n{}",
        page.stdout
    );

    // The capture face flows under the same policy.
    let shot = s
        .executor
        .execute(cells(&["b.screenshot('after the hold')"]))
        .await;
    assert!(shot.error.is_none(), "{shot:?}");
    assert_eq!(shot.events.len(), 1, "{shot:?}");

    // 能记: one row per call, decisions and outcomes spelled out.
    let rows = journal_rows(&s.journal);
    let held_row = rows
        .iter()
        .find(|r| r.contains("\"fn_name\":\"b.click\""))
        .unwrap_or_else(|| panic!("no journal row for the held click: {rows:?}"));
    assert!(held_row.contains("\"decision\":\"ask\""), "{held_row}");
    assert!(held_row.contains("\"result\":\"blocked\""), "{held_row}");
    let shot_row = rows
        .iter()
        .find(|r| r.contains("\"fn_name\":\"b.screenshot\""))
        .unwrap_or_else(|| panic!("no journal row for the screenshot: {rows:?}"));
    assert!(shot_row.contains("\"decision\":\"allow\""), "{shot_row}");
    assert!(shot_row.contains("\"result\":\"dispatched\""), "{shot_row}");
}
