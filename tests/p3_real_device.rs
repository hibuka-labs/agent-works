//! P3 real-device acceptance (D9, the living half): the assembled stack over
//! the real wires — the cua stub sidecar driving this Mac's desktop through
//! `cua-driver-local`, the mu broker driving one Android phone. The fake-wire
//! suite (`p3_device_acceptance.rs`) pins the host-side semantics; this one
//! answers only what a living device can:
//!
//! 1. the lookers really mint captures over the cua wire — a real PNG and
//!    cua's own capture id come back, so the id threading is device truth,
//!    not stub-level invention;
//! 2. the focused-window heuristic picks the window a human would mean, and
//!    element tokens click into it;
//! 3. the sidecar-spawned `cua-driver-local mcp` attaches with usable TCC
//!    attribution (screen recording + accessibility);
//! 4. the phone face refuses what the device refuses — a stale node is a
//!    rejection, AdbIME off is `UNSUPPORTED_CAPABILITY`, never fake success.
//!
//! CI never runs these: every test is `#[ignore]`d and gated on an env flag.
//!
//! ```text
//! # desktop (this Mac; CuaDriverLocal.app needs Accessibility + Screen Recording;
//! #           the display must be awake and unlocked — a locked session makes
//! #           captures black and AX trees empty)
//! cargo build --manifest-path ../phi-ns-desktop/stub/Cargo.toml
//! PHI_NS_REAL_DESKTOP=1 cargo test --test p3_real_device desktop_ -- --ignored --nocapture
//!
//! # phone (one adb device; mu-broker takes SERIAL AGENT_JAR)
//! cargo build --manifest-path ../../demo/OpenGCUA/packages/sdk/Cargo.toml -p mu-adapter-android --bin mu-broker
//! PHI_NS_MU_JAR=/path/to/agent.jar PHI_NS_ANDROID_SERIAL=<serial> PHI_NS_REAL_MOBILE=1 \
//!     cargo test --test p3_real_device mobile_ -- --ignored --nocapture
//! ```
//!
//! Side effects are deliberate and contained: the desktop tests open one
//! untitled TextEdit document and type into it — they never quit TextEdit
//! (the human at the keyboard may have unsaved docs) — and the phone tests
//! bounce to Home and open Settings once. Every act below is either a
//! wallpaper/document tap or a pointer move; nothing destructive is clicked.

use codexec::{
    Action, AllowAll, CodeCellExecutor, ExecutorBuilder, FileJournal, JournalHandle, Limits,
    MethodPolicy, PolicyDecision, PythonWorker,
};
use phi_ns::remote::ProcessTransport;
use phi_ns_desktop::PhiNsDesktop;
use phi_ns_mobile::PhiNsMobile;
use serial_test::serial;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";

// ---- where the real wires live ----

/// The cua stub sidecar binary (phi-wire-protocol in, `cua-driver-local mcp` out).
fn stub_bin() -> PathBuf {
    if let Ok(p) = std::env::var("PHI_NS_CUA_STUB_BIN") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../phi-ns-desktop/stub/target/debug/phi-ns-cua-stub")
}

/// The cua driver the stub wraps. The local installer names it
/// `cua-driver-local` (installed app: CuaDriverLocal.app).
fn driver_bin() -> String {
    if let Ok(d) = std::env::var("CUA_DRIVER_BIN") {
        return d;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let local = Path::new(&home).join(".local/bin/cua-driver-local");
    if local.exists() {
        return local.display().to_string();
    }
    "cua-driver-local".into()
}

/// The mu Android broker (`mu-broker SERIAL AGENT_JAR`).
fn mu_broker_bin() -> PathBuf {
    if let Ok(p) = std::env::var("MU_BROKER_BIN") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../demo/OpenGCUA/packages/sdk/target/debug/mu-broker")
}

fn mu_jar() -> PathBuf {
    PathBuf::from(std::env::var("PHI_NS_MU_JAR").expect(
        "PHI_NS_MU_JAR=/path/to/agent.jar — build it with `python3 packages/sdk/scripts/build_android.py`",
    ))
}

fn android_serial() -> String {
    std::env::var("PHI_NS_ANDROID_SERIAL").expect("PHI_NS_ANDROID_SERIAL=<adb device serial>")
}

/// These tests act on a living machine — they run only when asked, twice:
/// `#[ignore]` keeps them out of `cargo test`, the flag keeps a stray
/// `-- --ignored` from typing into whatever is frontmost.
fn require(flag: &str, face: &str) {
    assert_eq!(
        std::env::var(flag).as_deref(),
        Ok("1"),
        "{face} acceptance runs on demand: set {flag}=1 and pass -- --ignored"
    );
}

// ---- the assembled stack over real transports ----

struct Stage {
    _dir: tempfile::TempDir,
    journal: PathBuf,
    executor: CodeCellExecutor<PythonWorker>,
    wire: Arc<ProcessTransport>,
}

impl Stage {
    /// One execution against this stage's engine.
    async fn run(&mut self, codes: &[&str]) -> codexec::Observation {
        let actions = codes
            .iter()
            .map(|code| Action::CodeCell {
                intent: String::new(),
                code: (*code).to_string(),
            })
            .collect();
        self.executor.execute(actions).await
    }

    fn rows(&self) -> Vec<String> {
        std::fs::read_to_string(&self.journal)
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.to_string())
            .collect()
    }

    /// The one journal row a call is allowed to own.
    fn row_for(&self, method: &str) -> String {
        let rows = self.rows();
        rows.iter()
            .find(|r| r.contains(&format!("\"fn_name\":\"{method}\"")))
            .unwrap_or_else(|| panic!("no journal row for {method}: {rows:?}"))
            .clone()
    }

    async fn shutdown(&self) {
        let _ = self.wire.shutdown().await;
    }
}

/// Desktop acceptance needs somebody's screen to be on: an asleep or locked
/// session returns frozen black captures and empty AX trees, which read like
/// driver bugs instead of "nobody is at the machine". Fail fast on that.
fn require_awake_desktop() {
    // CoreGraphics says both flags; python3 is already a hard dependency here
    // (every cell runs in one) and ctypes needs no extra packages.
    const PROBE: &str = r#"import ctypes, ctypes.util as u
cg = ctypes.CDLL(u.find_library('CoreGraphics'))
cf = ctypes.CDLL(u.find_library('CoreFoundation'))
cg.CGMainDisplayID.restype = ctypes.c_uint32
cg.CGDisplayIsAsleep.argtypes = [ctypes.c_uint32]
cg.CGDisplayIsAsleep.restype = ctypes.c_bool
cg.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p
cf.CFDictionaryGetValue.restype = ctypes.c_void_p
cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cf.CFStringCreateWithCString.restype = ctypes.c_void_p
cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
cf.CFBooleanGetValue.restype = ctypes.c_bool
cf.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
asleep = cg.CGDisplayIsAsleep(cg.CGMainDisplayID())
d = cg.CGSessionCopyCurrentDictionary()
locked = False
if d:
    k = cf.CFStringCreateWithCString(None, b'CGSSessionScreenIsLocked', 0x08000100)
    v = cf.CFDictionaryGetValue(d, k)
    locked = bool(v) and cf.CFBooleanGetValue(v)
print('asleep' if asleep else 'locked' if locked else 'awake')
"#;
    let out = std::process::Command::new("python3")
        .args(["-c", PROBE])
        .output()
        .expect("python3 runs (codexec cells need it anyway)");
    let state = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(
        state, "awake",
        "this Mac is not hosting a live desktop ({state}) — unlock it and re-run"
    );
}

/// The desktop face: the stub sidecar around `cua-driver-local`.
async fn desktop_stage(policy: Arc<dyn codexec::Policy>) -> Stage {
    require_awake_desktop();
    let stub = stub_bin();
    assert!(
        stub.exists(),
        "build the sidecar first: cargo build --manifest-path ../phi-ns-desktop/stub/Cargo.toml (missing {})",
        stub.display()
    );
    let args = vec![
        "--driver".to_string(),
        driver_bin(),
        "--shots".to_string(),
        // Captures land in the stage's temp store via the pack's shot dir below.
        std::env::temp_dir().display().to_string(),
    ];
    let dir = tempfile::tempdir().expect("tempdir");
    let shots = dir.path().join("shots");
    std::fs::create_dir_all(&shots).expect("shot store");
    let journal = dir.path().join("journal.jsonl");
    let wire = ProcessTransport::spawn_with_timeout(
        &stub.display().to_string(),
        &args,
        &BTreeMap::new(),
        30_000,
    )
    .unwrap_or_else(|e| panic!("the cua stub sidecar must spawn: {e}"));
    let limits = Limits::default();
    let file = FileJournal::open(&journal).expect("journal opens");
    let executor = ExecutorBuilder::new(JournalHandle::new(Box::new(file)))
        .with_policy(policy)
        .with_limits(limits)
        .with_ns(PhiNsDesktop::new(wire.clone()))
        .build(move |registry| PythonWorker::spawn(registry, limits))
        .await
        .expect("engine assembles");
    Stage {
        _dir: dir,
        journal,
        executor,
        wire,
    }
}

/// The phone face: the mu Android broker over one leased adb device.
async fn mobile_stage(policy: Arc<dyn codexec::Policy>) -> Stage {
    let broker = mu_broker_bin();
    assert!(
        broker.exists(),
        "build the broker first: cargo build --manifest-path ../../demo/OpenGCUA/packages/sdk/Cargo.toml -p mu-adapter-android --bin mu-broker (missing {})",
        broker.display()
    );
    let jar = mu_jar();
    assert!(
        jar.exists(),
        "the Android agent jar is missing: {}",
        jar.display()
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let shots = dir.path().join("shots");
    std::fs::create_dir_all(&shots).expect("shot store");
    let journal = dir.path().join("journal.jsonl");
    let args = vec![android_serial(), jar.display().to_string()];
    let wire = ProcessTransport::spawn_with_timeout(
        &broker.display().to_string(),
        &args,
        &BTreeMap::new(),
        30_000,
    )
    .unwrap_or_else(|e| panic!("the mu broker must spawn: {e}"));
    let limits = Limits::default();
    let file = FileJournal::open(&journal).expect("journal opens");
    let executor = ExecutorBuilder::new(JournalHandle::new(Box::new(file)))
        .with_policy(policy)
        .with_limits(limits)
        .with_ns(PhiNsMobile::new(wire.clone()).with_shot_dir(&shots))
        .build(move |registry| PythonWorker::spawn(registry, limits))
        .await
        .expect("engine assembles");
    Stage {
        _dir: dir,
        journal,
        executor,
        wire,
    }
}

// ---- small readers over the cell stdout ----

/// Value printed by `print('<key>', <value>)`.
fn stdout_field(stdout: &str, key: &str) -> String {
    let prefix = format!("{key} ");
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{key} …` line in stdout:\n{stdout}"))
        .to_string()
}

fn png_dims(path: &str) -> (u32, u32) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("capture file {path}: {e}"));
    assert_eq!(
        &bytes[..8],
        PNG_MAGIC,
        "a real PNG, got {} bytes",
        bytes.len()
    );
    let dims = |from: usize| {
        u32::from_be_bytes([
            bytes[from],
            bytes[from + 1],
            bytes[from + 2],
            bytes[from + 3],
        ])
    };
    // PNG layout: signature, IHDR length+tag, then width and height.
    (dims(16), dims(20))
}

/// Put one fresh, focused TextEdit document on screen — the desktop acts need
/// somewhere harmless to land. Never quits TextEdit afterwards.
fn front_textedit() {
    // Activate first: `make new document` alone opens the window without
    // fronting the app, and the focused-window heuristic then sees the old
    // frontmost app instead of the document.
    let made = std::process::Command::new("osascript")
        .args([
            "-e",
            "tell application \"TextEdit\" to activate",
            "-e",
            "tell application \"TextEdit\" to make new document",
        ])
        .status()
        .expect("osascript runs");
    assert!(made.success(), "TextEdit must open for the desktop acts");
    // The document window appears and takes focus within a beat or two.
    std::thread::sleep(std::time::Duration::from_secs(2));
}

/// Assert the focused-window pick landed on TextEdit's own process. Identity
/// by pid survives the localized UI (this Mac says 文本编辑/未命名), unlike
/// matching app or window names in the listing.
fn assert_landed_on_textedit(stdout: &str) {
    let out = std::process::Command::new("pgrep")
        .args(["-x", "TextEdit"])
        .output()
        .expect("pgrep runs");
    let pid = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    assert!(!pid.is_empty(), "TextEdit is running");
    // Python dict repr of the resolved window: {'pid': N, 'window_id': M}.
    // Trailing comma keeps pid 123 from matching inside pid 1234.
    assert!(
        stdout.contains(&format!("'pid': {pid},")),
        "the focused window is TextEdit's (pid {pid}):\n{stdout}"
    );
}

// ---- desktop: the cua wire, this Mac ----

#[tokio::test]
#[serial]
#[ignore = "real desktop: PHI_NS_REAL_DESKTOP=1 — see the module docs"]
async fn desktop_a_real_shot_mints_one_capture_and_writes_a_real_png() {
    require("PHI_NS_REAL_DESKTOP", "desktop");
    let mut s = desktop_stage(Arc::new(AllowAll)).await;

    let look = s
        .run(&[
            "shot = desk.shot()",
            "print('id', shot.capture_id)",
            "print('path', shot.path)",
            "print('screen', desk.screen())",
        ])
        .await;
    assert!(look.error.is_none(), "{look:?}");
    // The chokepoint minted exactly one id for this look.
    assert!(
        look.stdout.contains("id capture_desk_0001"),
        "{}",
        look.stdout
    );
    // TCC question: screen recording really flows — a full PNG on disk.
    let path = stdout_field(&look.stdout, "path");
    let (w, h) = png_dims(&path);
    assert!(w > 0 && h > 0, "capture dimensions come back: {w}x{h}");
    let bytes = std::fs::read(&path).expect("capture bytes");
    assert!(
        bytes.len() > 10_000,
        "a real screen capture, not a stub: {} bytes",
        bytes.len()
    );

    let row = s.row_for("desk.shot");
    assert!(row.contains("\"decision\":\"allow\""), "{row}");
    assert!(row.contains("\"result\":\"dispatched\""), "{row}");
    s.shutdown().await;
}

#[tokio::test]
#[serial]
#[ignore = "real desktop: PHI_NS_REAL_DESKTOP=1 — see the module docs"]
async fn desktop_win_state_finds_the_editor_and_a_token_click_lands() {
    require("PHI_NS_REAL_DESKTOP", "desktop");
    front_textedit();
    let mut s = desktop_stage(Arc::new(AllowAll)).await;

    // The focused-window heuristic (no window argument) must pick the
    // document the human just made — that is what "frontmost" means here.
    let look = s
        .run(&[
            "print('wins', desk.windows())",
            "state = desk.win_state()",
            "print('id', state.capture_id)",
            "print('window', state.window)",
            "target = None",
            // One cell compiles as one Python block — the loop body lives in
            // the same string as its `for`, not as follow-up cells.
            "for el in state.els:\n    print('el', el.role, repr(el.text))\n    role = el.role.lower().replace(' ', '')\n    if target is None and 'textarea' in role:\n        target = el",
            "if target is None:\n    raise RuntimeError('no text-area element among %d els — see `el` lines' % len(state.els))",
            // Live fumble guard: every run must land in the document IT opened.
            // Content there means the focused pick grabbed a leftover document
            // — `el.text` is the AX name (an empty view is named "First Text
            // View"), so the document content is the raw row's `value`.
            "raw_ta = next((e for e in (state.raw or []) if e.get('element_token') == target.token), {})\nif (raw_ta.get('value') or '').strip():\n    raise RuntimeError('the focused pick is not the fresh empty document: %r' % raw_ta.get('value'))",
            "print('clicking', target.role, repr(target.text))",
            "target.click()",
            "print('clicked')",
        ])
        .await;
    assert!(look.error.is_none(), "{look:?}");
    assert!(
        look.stdout.contains("id capture_desk_0001"),
        "{}",
        look.stdout
    );
    assert_landed_on_textedit(&look.stdout);
    assert!(look.stdout.contains("clicked"), "{}", look.stdout);

    // 能记: the token click spent the look's one capture.
    let row = s.row_for("desk.click");
    assert!(row.contains("\"result\":\"dispatched\""), "{row}");
    assert!(
        row.contains("\"capture_id\":\"capture_desk_0001\""),
        "{row}"
    );
    s.shutdown().await;
}

#[tokio::test]
#[serial]
#[ignore = "real desktop: PHI_NS_REAL_DESKTOP=1 — see the module docs"]
async fn desktop_one_capture_authorizes_one_coordinate_click() {
    require("PHI_NS_REAL_DESKTOP", "desktop");
    front_textedit();
    let mut s = desktop_stage(Arc::new(AllowAll)).await;

    // Center of the fresh document window: a caret placement, nothing more.
    // Coordinates ride the look's native capture id — the device-side
    // binding is the whole point of threading it through.
    let look = s
        .run(&[
            "import struct",
            "shot = desk.shot()",
            "print('id', shot.capture_id)",
            "f = open(shot.path, 'rb'); f.read(16); w, h = struct.unpack('>II', f.read(8)); f.close()",
            "print('png', w, h)",
            "desk.click([w // 2, h // 2])",
            "print('clicked')",
        ])
        .await;
    assert!(look.error.is_none(), "{look:?}");
    assert!(
        look.stdout.contains("id capture_desk_0001"),
        "{}",
        look.stdout
    );
    assert!(look.stdout.contains("clicked"), "{}", look.stdout);

    // 复用就拒: the second act off the same look is refused at the chokepoint
    // before it can ever reach the screen.
    let reuse = s.run(&["desk.click([1, 1])"]).await;
    let err = reuse
        .error
        .clone()
        .unwrap_or_else(|| panic!("reuse must be refused: {reuse:?}"))
        .to_string();
    assert!(err.contains("already consumed"), "{err}");
    assert!(err.contains("look again"), "{err}");

    let rows = s.rows();
    let clicks: Vec<&String> = rows
        .iter()
        .filter(|r| r.contains("\"fn_name\":\"desk.click\""))
        .collect();
    assert_eq!(clicks.len(), 2, "dispatched then blocked: {rows:?}");
    assert!(
        clicks[0].contains("\"result\":\"dispatched\""),
        "{}",
        clicks[0]
    );
    assert!(
        clicks[0].contains("\"capture_id\":\"capture_desk_0001\""),
        "{}",
        clicks[0]
    );
    assert!(
        clicks[1].contains("\"result\":\"blocked\""),
        "{}",
        clicks[1]
    );
    s.shutdown().await;
}

#[tokio::test]
#[serial]
#[ignore = "real desktop: PHI_NS_REAL_DESKTOP=1 — see the module docs"]
async fn desktop_no_look_is_refused_and_a_hold_never_spends() {
    require("PHI_NS_REAL_DESKTOP", "desktop");

    // No look, no authority: refused at the gate, no wire call, no click.
    let mut s = desktop_stage(Arc::new(AllowAll)).await;
    let lookless = s.run(&["desk.click([1, 1])"]).await;
    let err = lookless
        .error
        .clone()
        .unwrap_or_else(|| panic!("a lookless act must be refused: {lookless:?}"))
        .to_string();
    assert!(err.contains("no live capture"), "{err}");
    assert!(err.contains("capture gate refused desk.click"), "{err}");
    s.shutdown().await;

    // A held call never spends the capture: desk.click asks a human and
    // stops; the look's authority survives the hold and still expires after
    // exactly one real act (a pointer move is the harmless consumer).
    let policy = MethodPolicy::new(PolicyDecision::Allow).with("desk.click", PolicyDecision::Ask);
    let mut s = desktop_stage(Arc::new(policy)).await;
    let look = s
        .run(&["shot = desk.shot()", "print('id', shot.capture_id)"])
        .await;
    assert!(look.error.is_none(), "{look:?}");
    assert!(
        look.stdout.contains("id capture_desk_0001"),
        "{}",
        look.stdout
    );

    let held = s.run(&["desk.click([2, 2])"]).await;
    assert!(held.is_policy_hold(), "{held:?}");

    let act = s.run(&["desk.move([10, 10])", "print('moved')"]).await;
    assert!(act.error.is_none(), "{act:?}");
    assert!(act.stdout.contains("moved"), "{}", act.stdout);
    let row = s.row_for("desk.move");
    assert!(
        row.contains("\"capture_id\":\"capture_desk_0001\""),
        "{row}"
    );

    let twice = s.run(&["desk.move([11, 11])"]).await;
    let err = twice
        .error
        .clone()
        .unwrap_or_else(|| panic!("the second act must be refused: {twice:?}"))
        .to_string();
    assert!(err.contains("already consumed"), "{err}");
    s.shutdown().await;
}

#[tokio::test]
#[serial]
#[ignore = "real desktop: PHI_NS_REAL_DESKTOP=1 — see the module docs"]
async fn desktop_blind_type_needs_no_capture_and_the_text_lands() {
    require("PHI_NS_REAL_DESKTOP", "desktop");
    front_textedit();
    let mut s = desktop_stage(Arc::new(AllowAll)).await;

    // Focus the document through an element token (this spends one capture),
    // then type blind: keyboard input draws no capture authority at all.
    let marked = "phi-ns real-device";
    let typed = s
        .run(&[
            "state = desk.win_state()",
            "print('id', state.capture_id)",
            "print('window', state.window)",
            "target = None",
            // One cell compiles as one Python block — the loop body lives in
            // the same string as its `for`, not as follow-up cells.
            "for el in state.els:\n    print('el', el.role, repr(el.text))\n    role = el.role.lower().replace(' ', '')\n    if target is None and 'textarea' in role:\n        target = el",
            "if target is None:\n    raise RuntimeError('no text-area element among %d els — see `el` lines' % len(state.els))",
            // Live fumble guard: typing into a leftover document would append
            // and still "land" — the fresh document must carry no content.
            "raw_ta = next((e for e in (state.raw or []) if e.get('element_token') == target.token), {})\nif (raw_ta.get('value') or '').strip():\n    raise RuntimeError('the focused pick is not the fresh empty document: %r' % raw_ta.get('value'))",
            "target.click()",
            &format!("desk.type({marked:?})"),
            "print('typed')",
            "print('clip-out', desk.clip_write('phi-ns-clip-mark'))",
            "print('clip-in', desk.clip_read())",
        ])
        .await;
    assert!(typed.error.is_none(), "{typed:?}");
    assert_landed_on_textedit(&typed.stdout);
    assert!(typed.stdout.contains("typed"), "{}", typed.stdout);
    // A closed loop that has nothing to do with the GUI: the clipboard pair.
    assert!(
        typed.stdout.contains("phi-ns-clip-mark"),
        "clip_write then clip_read round-trips:\n{}",
        typed.stdout
    );

    // On-screen evidence: the accessibility tree sees what was typed.
    let seen = s
        .run(&[
            "state = desk.win_state()",
            "print('raw', state.raw)",
            "for el in state.els: print('el', el.role, repr(el.text))",
        ])
        .await;
    assert!(seen.error.is_none(), "{seen:?}");
    // TextEdit applies macOS sentence capitalization to a document's first
    // letter (live: "phi-ns real-device" landed as "Phi-ns real-device"). The
    // engine typed the marker exactly; the editor's substitution is not ours.
    // Accept the exact marker or that one substitution — an engine-side upcase
    // of the whole string would match neither.
    let mut marker_chars = marked.chars();
    let sentence_capped = match marker_chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + marker_chars.as_str(),
        None => String::new(),
    };
    assert!(
        seen.stdout.contains(marked) || seen.stdout.contains(&sentence_capped),
        "the typed text is visible to a fresh win_state (as {marked:?} or {sentence_capped:?}):\n{}",
        seen.stdout
    );
    s.shutdown().await;
}

// ---- phone: the mu wire over one Android device ----

#[tokio::test]
#[serial]
#[ignore = "real phone: PHI_NS_REAL_MOBILE=1 — see the module docs"]
async fn mobile_describe_and_observe_tell_the_truth() {
    require("PHI_NS_REAL_MOBILE", "mobile");
    let mut s = mobile_stage(Arc::new(AllowAll)).await;

    let look = s
        .run(&[
            "print('describe', m.describe())",
            "obs = m.observe('ax')",
            "print('id', obs.capture_id)",
            "print('nodes', len(obs.nodes))",
            "print('img', obs.image.path)",
        ])
        .await;
    assert!(look.error.is_none(), "{look:?}");
    assert!(look.stdout.contains("id capture_m_0001"), "{}", look.stdout);
    let nodes: usize = stdout_field(&look.stdout, "nodes")
        .parse()
        .expect("node count");
    assert!(
        nodes >= 1,
        "a real UI tree, not an empty one:\n{}",
        look.stdout
    );
    let (w, h) = png_dims(&stdout_field(&look.stdout, "img"));
    assert!(w > 0 && h > 0, "the phone shot carries dimensions: {w}x{h}");
    s.shutdown().await;
}

#[tokio::test]
#[serial]
#[ignore = "real phone: PHI_NS_REAL_MOBILE=1 — see the module docs"]
async fn mobile_one_look_authorizes_one_center_tap() {
    require("PHI_NS_REAL_MOBILE", "mobile");
    let mut s = mobile_stage(Arc::new(AllowAll)).await;

    // Home first; the center of the launcher is wallpaper — the device's own
    // no-op tap, the safest real click there is.
    let look = s
        .run(&[
            "import struct, time",
            "m.system('home')",
            "time.sleep(0.5)",
            "obs = m.observe('ax')",
            "print('id', obs.capture_id)",
            "f = open(obs.image.path, 'rb'); f.read(16); w, h = struct.unpack('>II', f.read(8)); f.close()",
            "print('png', w, h)",
            "m.click([w // 2, h // 2])",
            "print('tapped')",
        ])
        .await;
    assert!(look.error.is_none(), "{look:?}");
    assert!(look.stdout.contains("id capture_m_0001"), "{}", look.stdout);
    assert!(look.stdout.contains("tapped"), "{}", look.stdout);

    // 复用就拒: one look, one action attempt — the device never sees the second.
    let reuse = s.run(&["m.click([1, 1])"]).await;
    let err = reuse
        .error
        .clone()
        .unwrap_or_else(|| panic!("reuse must be refused: {reuse:?}"))
        .to_string();
    assert!(err.contains("already consumed"), "{err}");

    let rows = s.rows();
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
        clicks[0].contains("\"capture_id\":\"capture_m_0001\""),
        "{}",
        clicks[0]
    );
    assert!(
        clicks[1].contains("\"result\":\"blocked\""),
        "{}",
        clicks[1]
    );
    s.shutdown().await;
}

#[tokio::test]
#[serial]
#[ignore = "real phone: PHI_NS_REAL_MOBILE=1 — see the module docs"]
async fn mobile_text_reports_adbime_instead_of_faking() {
    require("PHI_NS_REAL_MOBILE", "mobile");
    let mut s = mobile_stage(Arc::new(AllowAll)).await;

    // m.text types through AdbIME (inventory §3.5): with the IME active it
    // types; without it the device refuses `UNSUPPORTED_CAPABILITY`. Both are
    // truth. What must never happen is a silent "ok" that typed nothing.
    let probe = s
        .run(&[
            "print('describe', m.describe())",
            "obs = m.observe('ax')",
            "print('id', obs.capture_id)",
            "print(m.text('phi-ns ime probe'))",
            "print('typed')",
        ])
        .await;
    if let Some(err) = probe.error.clone() {
        let err = err.to_string();
        assert!(
            err.contains("UNSUPPORTED_CAPABILITY"),
            "a missing AdbIME must refuse by name, got: {err}"
        );
    } else {
        assert!(probe.stdout.contains("typed"), "{}", probe.stdout);
    }
    assert!(
        probe.stdout.contains("id capture_m_0001"),
        "{}",
        probe.stdout
    );
    s.shutdown().await;
}

#[tokio::test]
#[serial]
#[ignore = "real phone: PHI_NS_REAL_MOBILE=1 — see the module docs"]
async fn mobile_a_stale_node_is_a_rejection_not_a_maybe() {
    require("PHI_NS_REAL_MOBILE", "mobile");
    let mut s = mobile_stage(Arc::new(AllowAll)).await;

    // Observe Home, then walk away to Settings: the node handle from the old
    // tree is stale, and the device must say so (§3.2) — never act on it.
    let stale = s
        .run(&[
            "import time",
            "m.system('home')",
            "time.sleep(0.5)",
            "obs = m.observe('ax')",
            "print('id', obs.capture_id)",
            "print('node', obs.nodes[0])",
            "m.launch('com.android.settings')",
            "time.sleep(1.5)",
            "m.click(obs.nodes[0])",
            "print('clicked stale')",
        ])
        .await;
    let err = stale
        .error
        .clone()
        .unwrap_or_else(|| panic!("a stale node must be refused: {stale:?}"))
        .to_string();
    assert!(
        ["STALE", "NOT_FOUND", "INVALID", "REJECTED"]
            .iter()
            .any(|code| err.contains(code)),
        "the refusal names a rejection, not a maybe: {err}"
    );
    assert!(
        stale.stdout.contains("id capture_m_0001"),
        "{}",
        stale.stdout
    );
    // 能记: the refused attempt still owns (and spent) its capture — 一看一试.
    let row = s.row_for("m.click");
    assert!(row.contains("\"capture_id\":\"capture_m_0001\""), "{row}");
    assert!(row.contains("\"result\":\"dispatched\""), "{row}");

    // Put the human's phone back where it was.
    let _ = s.run(&["m.system('home')"]).await;
    s.shutdown().await;
}
