//! P3 device acceptance (sdk-assembly §10) — the assembled stack: `codexec`
//! with the real `desk.*` + `m.*` packs, lined up in the挂接层 where assembly
//! lives (§9.3: neither the engine nor the pack may know the other).
//!
//! CI runs against fake wires — the wire protocol and the host-side method
//! semantics are what these claims judge; the real stubs are exercised by the
//! real-device runs. Claims:
//!
//! 1. the initial `MethodPolicy` table (inventory §5) holds device input on
//!    both faces while captures flow — and every decision lands in the journal;
//! 2. a phone screenshot reaches the model on both channels (reuse of P2's
//!    multimodal path: `ImageHandle` for the code, `CellEvent::Image` for the
//!    engine);
//! 3. a forged call cannot dodge the chokepoint — undeclared names fail
//!    closed, declared names still journal;
//! 4. one capture id authorizes exactly one action attempt (inventory §3.2) —
//!    复用就拒, a held call never spends it, and faces keep their own authority.

use codexec::{
    Action, CellEvent, ExecutorBuilder, FileJournal, JournalHandle, Limits, MethodPolicy,
    Observation, PolicyDecision, PythonWorker,
};
use phi_ns::NamespaceProvider;
use phi_ns::NsError;
use phi_ns::remote::Transport;
use phi_ns_desktop::PhiNsDesktop;
use phi_ns_mobile::PhiNsMobile;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Inventory §5's Ask column — the device verbs that must hold a human.
/// Flow/Allow columns fall through to `Allow`; the fall-through for anything
/// not declared at all is `Deny` (fail-closed).
const DEVICE_ASK: &[&str] = &[
    // desk.* Ask(11)
    "desk.click",
    "desk.drag",
    "desk.move",
    "desk.scroll",
    "desk.key",
    "desk.hotkey",
    "desk.type",
    "desk.menu",
    "desk.clip_write",
    "desk.win_frame",
    "desk.escalate",
    // m.* Ask(14)
    "m.click",
    "m.scroll",
    "m.swipe",
    "m.long_press",
    "m.drag",
    "m.set_text",
    "m.select_text",
    "m.perform",
    "m.text",
    "m.press_key",
    "m.system",
    "m.launch",
    "m.file_write",
    "m.shell",
];

/// The MethodPolicy initial table: one row per declared device method,
/// generated from the packs' own declaration source — names come from
/// `methods()`, decisions from the inventory's columns. Unknown names deny.
fn device_policy() -> MethodPolicy {
    let methods: Vec<phi_ns::MethodSpec> = PhiNsDesktop::unconnected()
        .methods()
        .into_iter()
        .chain(PhiNsMobile::unconnected().methods())
        .collect();
    let mut policy = MethodPolicy::new(PolicyDecision::Deny);
    for m in &methods {
        let decision = if DEVICE_ASK.contains(&m.name) {
            PolicyDecision::Ask
        } else {
            PolicyDecision::Allow
        };
        policy = policy.with(m.name, decision);
    }
    policy
}

/// The one tiny screenshot every fake wire serves: PNG magic plus filler —
/// enough for the "a real PNG" byte check without an image encoder.
const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";
const PNG_B64: &str = "iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";

fn png_bytes() -> Vec<u8> {
    let mut bytes = PNG_MAGIC.to_vec();
    bytes.extend_from_slice(&[0u8; 32]);
    bytes
}

// ---- fake wires: the stubs CI substitutes for the real devices ----

#[derive(Default)]
struct FakePhone {
    seen: Mutex<Vec<(String, Value)>>,
}

impl FakePhone {
    fn last_op(&self) -> String {
        self.seen
            .lock()
            .unwrap()
            .last()
            .map(|(op, _)| op.clone())
            .expect("the phone wire was never called")
    }

    fn op_count(&self, op: &str) -> usize {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .filter(|(o, _)| o == op)
            .count()
    }
}

#[async_trait::async_trait]
impl Transport for FakePhone {
    async fn request(&self, method: &str, args: Value) -> Result<Value, NsError> {
        self.seen.lock().unwrap().push((method.to_string(), args));
        Ok(match method {
            "observe" => json!({
                "nodes": [{"id": "n1", "role": "button", "text": "Settings", "target_token": "t1"}],
                "revision": "7", "width": 1080, "height": 2400, "png": PNG_B64,
            }),
            _ => json!({"ok": true}),
        })
    }
}

/// The cua stub's answers. `desk.shot` writes the capture file the way the
/// real stub does and returns its image wire plus the geometry the look bound
/// (the desktop capture descriptor, inventory §3.2).
struct FakeCua {
    shots: std::path::PathBuf,
    seen: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl Transport for FakeCua {
    async fn request(&self, method: &str, _args: Value) -> Result<Value, NsError> {
        self.seen.lock().unwrap().push(method.to_string());
        Ok(match method {
            "desk.shot" => {
                let path = self.shots.join("desk-shot.png");
                std::fs::write(&path, png_bytes()).expect("capture writes");
                json!({
                    "path": path.display().to_string(),
                    "mime": "image/png",
                    "alt": "the whole desktop",
                    "width": 1440, "height": 900, "rotation": 0,
                })
            }
            "desk.win_state" => json!({
                "window": "w1", "revision": "3",
                "els": [{"token": "e1", "role": "button", "text": "Save"}],
            }),
            _ => json!({"ok": true}),
        })
    }
}

// ---- the assembled stack ----

struct Stage {
    _dir: tempfile::TempDir,
    shots: std::path::PathBuf,
    journal: std::path::PathBuf,
    executor: codexec::CodeCellExecutor<PythonWorker>,
    phone: std::sync::Arc<FakePhone>,
    cua: std::sync::Arc<FakeCua>,
}

async fn stage(policy: std::sync::Arc<dyn codexec::Policy>) -> Stage {
    let dir = tempfile::tempdir().expect("tempdir");
    let shots = dir.path().join("shots");
    std::fs::create_dir_all(&shots).expect("shot store");
    let journal = dir.path().join("journal.jsonl");

    let phone = std::sync::Arc::new(FakePhone::default());
    let cua = std::sync::Arc::new(FakeCua {
        shots: shots.clone(),
        seen: Mutex::new(Vec::new()),
    });
    let limits = Limits::default();
    let file = FileJournal::open(&journal).expect("journal opens");
    let executor = ExecutorBuilder::new(JournalHandle::new(Box::new(file)))
        .with_policy(policy)
        .with_limits(limits)
        .with_ns(PhiNsMobile::new(phone.clone()).with_shot_dir(&shots))
        .with_ns(PhiNsDesktop::new(cua.clone()))
        .build(move |registry| PythonWorker::spawn(registry, limits))
        .await
        .expect("engine assembles");

    Stage {
        _dir: dir,
        shots,
        journal,
        executor,
        phone,
        cua,
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

/// What the host layer sends upstream from one observation (P2's shape, kept
/// identical so the two faces are compared in one language).
fn model_blocks(obs: &Observation) -> Vec<serde_json::Value> {
    let mut content = vec![serde_json::json!({"type": "text", "text": obs.stdout})];
    for event in &obs.events {
        if let CellEvent::Image(image) = event {
            content.push(serde_json::json!({
                "type": "image",
                "source": image.to_wire(),
            }));
        }
    }
    content
}

/// The initial table covers exactly the declared face — a name in the Ask
/// column that no pack declares is a typo, and a declared name missing from
/// the table would silently deny.
#[test]
fn the_initial_table_covers_every_declared_device_method() {
    let declared: Vec<&str> = PhiNsDesktop::unconnected()
        .methods()
        .into_iter()
        .chain(PhiNsMobile::unconnected().methods())
        .map(|m| m.name)
        .collect();
    assert_eq!(declared.len(), 41, "desk.* 21 + m.* 20");
    for name in DEVICE_ASK {
        assert!(
            declared.contains(name),
            "{name} is in the Ask column but undeclared"
        );
    }
    assert_eq!(DEVICE_ASK.len(), 25, "desk Ask(11) + m Ask(14)");
}

/// 能拦 + 能记: device input is held on both faces — the effect never reaches
/// the wire — while captures flow under the same table. Every decision is one
/// journal row.
#[tokio::test]
async fn device_input_is_held_while_captures_flow() {
    let mut s = stage(Arc::new(device_policy()) as Arc<dyn codexec::Policy>).await;

    // A capture on each face flows.
    let look = s
        .executor
        .execute(cells(&[
            "obs = m.observe('ax')",
            "print(obs.nodes[0].text)",
        ]))
        .await;
    assert!(look.error.is_none(), "{look:?}");
    assert!(look.stdout.contains("Settings"), "{}", look.stdout);
    assert_eq!(s.phone.last_op(), "observe");

    let shot = s.executor.execute(cells(&["desk.shot()"])).await;
    assert!(shot.error.is_none(), "{shot:?}");

    // Input on each face is held; nothing reaches either wire.
    let tap = s.executor.execute(cells(&["m.click([10, 20])"])).await;
    assert!(tap.is_policy_hold(), "{tap:?}");
    assert!(tap.events.is_empty(), "{tap:?}");
    assert_eq!(
        s.phone.last_op(),
        "observe",
        "the held tap never reached the phone"
    );

    let click = s.executor.execute(cells(&["desk.click([5, 5])"])).await;
    assert!(click.is_policy_hold(), "{click:?}");
    let cua_ops = s.cua.seen.lock().unwrap().clone();
    assert_eq!(
        cua_ops,
        ["desk.shot"],
        "the held click never reached the desktop"
    );

    // 能记: one row per call, decisions and outcomes spelled out.
    let rows = journal_rows(&s.journal);
    for (name, decision, result) in [
        ("m.observe", "allow", "dispatched"),
        ("desk.shot", "allow", "dispatched"),
        ("m.click", "ask", "blocked"),
        ("desk.click", "ask", "blocked"),
    ] {
        let row = rows
            .iter()
            .find(|r| r.contains(&format!("\"fn_name\":\"{name}\"")))
            .unwrap_or_else(|| panic!("no journal row for {name}: {rows:?}"));
        assert!(
            row.contains(&format!("\"decision\":\"{decision}\"")),
            "{row}"
        );
        assert!(row.contains(&format!("\"result\":\"{result}\"")), "{row}");
    }
}

/// 多模态 (P2's channel, reused): the phone screenshot lands as an
/// `ImageHandle` for the code and a `CellEvent::Image` for the engine — one
/// source, two channels — and the host block carries the picture.
#[tokio::test]
async fn the_phone_screenshot_reaches_the_model_on_both_channels() {
    let mut s = stage(Arc::new(codexec::AllowAll) as Arc<dyn codexec::Policy>).await;

    let obs = s
        .executor
        .execute(cells(&[
            "shot = m.observe('vision').image",
            "print(shot.alt)",
            "print(shot.path)",
        ]))
        .await;
    assert!(obs.error.is_none(), "{obs:?}");

    // Channel one: the worker-side handle.
    assert!(
        obs.stdout.contains("phone screen 1080x2400"),
        "ImageHandle.alt reached the text stream:\n{}",
        obs.stdout
    );

    // Channel two: the engine-side event.
    assert_eq!(obs.events.len(), 1, "{obs:?}");
    let CellEvent::Image(image) = &obs.events[0] else {
        panic!("expected an image event, got {:?}", obs.events[0]);
    };
    assert_eq!(image.mime, "image/png");
    assert_eq!(image.alt, "phone screen 1080x2400");
    let phi_ns::ImageSource::Path(path) = &image.source else {
        panic!("default channel is on-disk, got {:?}", image.source);
    };
    assert!(
        path.starts_with(&s.shots),
        "shot lands in the store: {path:?}"
    );
    let bytes = std::fs::read(path).expect("shot bytes");
    assert_eq!(&bytes[..8], PNG_MAGIC, "a real PNG");
    assert!(
        obs.stdout.contains(&path.display().to_string()),
        "ImageHandle.path and ImageRef.source disagree:\n{}",
        obs.stdout
    );

    // The multimodal block a host builds from this.
    let blocks = model_blocks(&obs);
    assert_eq!(blocks.len(), 2, "{blocks:?}");
    assert_eq!(blocks[1]["type"], "image");
    assert_eq!(blocks[1]["source"]["mime"], "image/png");
    assert_eq!(blocks[1]["source"]["alt"], "phone screen 1080x2400");
    assert!(
        !obs.stdout.contains("iVBOR"),
        "no image bytes in the text stream:\n{}",
        obs.stdout
    );
}

/// 审计绕不过: a forged `_phi_call` line hits the same chokepoint as a real
/// one. Undeclared names fail closed (nothing to journal); declared names
/// journal before any effect — the model cannot write a line that acts
/// without a row.
#[tokio::test]
async fn a_forged_call_cannot_dodge_the_chokepoint() {
    let mut s = stage(Arc::new(codexec::AllowAll) as Arc<dyn codexec::Policy>).await;

    // A name no pack declared: fail closed.
    let forged = s
        .executor
        .execute(cells(&["_phi_call('m.steal', [])"]))
        .await;
    assert!(
        forged.error.is_some(),
        "undeclared names must fail: {forged:?}"
    );
    let rows = journal_rows(&s.journal);
    assert!(
        rows.iter().all(|r| !r.contains("m.steal")),
        "a name that never existed journals nothing: {rows:?}"
    );

    // A declared name with garbage arguments: the row lands (write-ahead)
    // even though the call then fails its own contract. Observe first so the
    // capture gate is not what refuses — the contract failure is the point.
    let sloppy = s
        .executor
        .execute(cells(&[
            "m.observe('ax')",
            "_phi_call('m.click', ['not a target'])",
        ]))
        .await;
    assert!(
        sloppy.error.is_some(),
        "garbage targets must fail: {sloppy:?}"
    );
    let err = sloppy.error.clone().expect("just asserted").to_string();
    assert!(
        err.contains("target must be a node handle"),
        "the call failed its own contract, not the gate: {err}"
    );
    let rows = journal_rows(&s.journal);
    let row = rows
        .iter()
        .find(|r| r.contains("\"fn_name\":\"m.click\""))
        .unwrap_or_else(|| panic!("the forged-but-declared line still journals: {rows:?}"));
    // One attempt per look: the capture is spent at admission and never
    // refunded, even when the attempt then fails its own contract.
    assert!(row.contains("\"capture_id\":\"capture_m_0001\""), "{row}");
}

/// 一次性台账 (inventory §3.2): one look mints one capture id, one act spends
/// it — a second act off the same look is refused at the chokepoint (复用就拒),
/// the refusal journals, and nothing reaches the wire. A fresh look restores
/// the authority.
#[tokio::test]
async fn one_capture_authorizes_one_action_attempt() {
    let mut s = stage(Arc::new(codexec::AllowAll) as Arc<dyn codexec::Policy>).await;

    let look = s
        .executor
        .execute(cells(&[
            "obs = m.observe('ax')",
            "print('id', obs.capture_id)",
            "print('bind', obs.capture['revision'], obs.capture['width'], obs.capture['height'])",
            "m.click([10, 20])",
            "print('acted')",
        ]))
        .await;
    assert!(look.error.is_none(), "{look:?}");
    // The chokepoint stamps the one-use id on the look's return.
    assert!(look.stdout.contains("id capture_m_0001"), "{}", look.stdout);
    // The binding names the view the id authorizes — tree revision + geometry.
    assert!(look.stdout.contains("bind 7 1080 2400"), "{}", look.stdout);
    assert!(look.stdout.contains("acted"), "{}", look.stdout);

    // 能记: the spent capture lands in the journal row at the chokepoint.
    let rows = journal_rows(&s.journal);
    let spent = rows
        .iter()
        .find(|r| r.contains("\"fn_name\":\"m.click\""))
        .unwrap_or_else(|| panic!("no journal row for the click: {rows:?}"));
    assert!(
        spent.contains("\"capture_id\":\"capture_m_0001\""),
        "{spent}"
    );
    assert!(spent.contains("\"result\":\"dispatched\""), "{spent}");

    // 复用就拒: the second act off the same look errors before any effect.
    let reuse = s.executor.execute(cells(&["m.click([30, 40])"])).await;
    let err = reuse
        .error
        .clone()
        .unwrap_or_else(|| panic!("reuse must be refused: {reuse:?}"))
        .to_string();
    assert!(err.contains("capture gate refused m.click"), "{err}");
    assert!(err.contains("already consumed"), "{err}");
    assert!(err.contains("look again"), "{err}");

    // The refusal journals blocked, naming the capture it sought.
    let rows = journal_rows(&s.journal);
    let clicks: Vec<&String> = rows
        .iter()
        .filter(|r| r.contains("\"fn_name\":\"m.click\""))
        .collect();
    assert_eq!(clicks.len(), 2, "dispatched then blocked: {rows:?}");
    assert!(
        clicks[0].contains("\"result\":\"dispatched\""),
        "{}",
        clicks[0]
    );
    assert!(
        clicks[1].contains("\"result\":\"blocked\""),
        "{}",
        clicks[1]
    );
    assert!(
        clicks[1].contains("\"capture_id\":\"capture_m_0001\""),
        "{}",
        clicks[1]
    );

    // The refused act never reached the wire: one action attempt total.
    assert_eq!(
        s.phone.op_count("action"),
        1,
        "the refused act must not touch the phone"
    );

    // A fresh look restores the authority.
    let again = s
        .executor
        .execute(cells(&[
            "obs = m.observe('ax')",
            "print('id2', obs.capture_id)",
            "m.click([40, 50])",
        ]))
        .await;
    assert!(again.error.is_none(), "{again:?}");
    assert!(
        again.stdout.contains("id2 capture_m_0002"),
        "{}",
        again.stdout
    );
    assert_eq!(s.phone.op_count("action"), 2);
}

/// A held call never spends the capture (inventory §3.2): nothing was
/// authorized to act, so the look's authority survives the hold — the next
/// allowed act still spends it exactly once.
#[tokio::test]
async fn a_held_call_never_spends_the_capture() {
    // m.click is held; every other verb flows, so the difference between
    // "held" and "spent" is observable in one table.
    let policy = MethodPolicy::new(PolicyDecision::Allow).with("m.click", PolicyDecision::Ask);
    let mut s = stage(Arc::new(policy) as Arc<dyn codexec::Policy>).await;

    let look = s
        .executor
        .execute(cells(&["obs = m.observe('ax')", "print(obs.capture_id)"]))
        .await;
    assert!(look.error.is_none(), "{look:?}");
    assert!(look.stdout.contains("capture_m_0001"), "{}", look.stdout);

    // The held tap: the step halts, the wire is untouched, the capture lives.
    let held = s.executor.execute(cells(&["m.click([10, 20])"])).await;
    assert!(held.is_policy_hold(), "{held:?}");
    assert_eq!(s.phone.op_count("action"), 0, "the held tap never landed");

    // The same look still authorizes the next allowed act.
    let act = s.executor.execute(cells(&["m.text('hello')"])).await;
    assert!(act.error.is_none(), "{act:?}");
    assert_eq!(s.phone.op_count("action"), 1);
    let rows = journal_rows(&s.journal);
    let row = rows
        .iter()
        .find(|r| r.contains("\"fn_name\":\"m.text\""))
        .unwrap_or_else(|| panic!("no journal row for m.text: {rows:?}"));
    assert!(row.contains("\"capture_id\":\"capture_m_0001\""), "{row}");

    // And exactly once: that act spent it.
    let twice = s.executor.execute(cells(&["m.text('again')"])).await;
    let err = twice
        .error
        .clone()
        .unwrap_or_else(|| panic!("the second act must be refused: {twice:?}"))
        .to_string();
    assert!(err.contains("already consumed"), "{err}");
    assert_eq!(
        s.phone.op_count("action"),
        1,
        "the refused act never landed"
    );
}

/// Face isolation (inventory §3.2): looks authorize acts on their own face
/// only — a phone capture never authorizes a desktop click, and the other way
/// round. Each face mints its own id.
#[tokio::test]
async fn each_face_owns_its_capture_authority() {
    let mut s = stage(Arc::new(codexec::AllowAll) as Arc<dyn codexec::Policy>).await;

    // A phone look is not a desktop authority: desk never looked, so the act
    // is refused no matter how fresh the phone view is.
    let cross = s
        .executor
        .execute(cells(&[
            "obs = m.observe('ax')",
            "print('m', obs.capture_id)",
            "desk.click([7, 7])",
        ]))
        .await;
    let err = cross
        .error
        .clone()
        .unwrap_or_else(|| panic!("cross-face must be refused: {cross:?}"))
        .to_string();
    assert!(err.contains("capture gate refused desk.click"), "{err}");
    assert!(err.contains("no live capture on face `desk`"), "{err}");
    assert!(
        cross.stdout.contains("m capture_m_0001"),
        "{}",
        cross.stdout
    );

    // The phone face's own look authorizes the phone face's own act.
    let own = s.executor.execute(cells(&["m.click([10, 20])"])).await;
    assert!(own.error.is_none(), "{own:?}");

    // And the other way: a desktop look is not a phone authority — m's one
    // attempt is spent, and the fresh desktop capture cannot vouch for another.
    let cross = s
        .executor
        .execute(cells(&[
            "shot = desk.shot()",
            "print('d', shot.capture_id)",
            "m.click([1, 2])",
        ]))
        .await;
    let err = cross
        .error
        .clone()
        .unwrap_or_else(|| panic!("cross-face must be refused: {cross:?}"))
        .to_string();
    assert!(err.contains("capture gate refused m.click"), "{err}");
    assert!(err.contains("already consumed"), "{err}");
    assert!(
        cross.stdout.contains("d capture_desk_0001"),
        "{}",
        cross.stdout
    );

    // The desktop face's own look authorizes the desktop face's own act.
    let own = s.executor.execute(cells(&["desk.click([5, 5])"])).await;
    assert!(own.error.is_none(), "{own:?}");

    // Wire: exactly the two authorized acts landed; the refused ones never
    // reached either device.
    assert_eq!(
        s.cua.seen.lock().unwrap().clone(),
        ["desk.shot", "desk.click"]
    );
    assert_eq!(s.phone.op_count("action"), 1);
}
