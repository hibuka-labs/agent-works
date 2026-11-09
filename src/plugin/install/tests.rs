use std::path::{Path, PathBuf};

use super::super::manifest::ResolvedCapability;
use super::super::registry::{PackageStatus, PluginRegistry, ToolPin, hex_sha256};
use super::build::pin_server_argv;
use super::source::{extract_archive, normalize_relative};
use super::*;
use crate::script_tool::ScriptLawError;

/// Platform-independent absolute path (test criterion under the same law
/// as `Path::is_absolute`).
fn abs(rel: &str) -> PathBuf {
    std::env::current_dir().unwrap().join(rel)
}

/// Write a script and set the executable bit (unix) — the install law
/// requires directly-executed `run[0]` to be executable.
fn write_script(path: &Path, contents: &str) {
    std::fs::write(path, contents).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = std::fs::metadata(path).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(path, perm).unwrap();
    }
}

fn write_basic_plugin(dir: &Path) {
    std::fs::create_dir_all(dir.join("scripts")).unwrap();
    std::fs::create_dir_all(dir.join("wasm")).unwrap();
    std::fs::create_dir_all(dir.join("server")).unwrap();
    std::fs::write(
        dir.join("plugin.json"),
        serde_json::json!({
            "schemaVersion": 1,
            "name": "my-tools",
            "version": "0.1.0",
            "tools": [
                {
                    "name": "deploy", "form": "script",
                    "description": "deploy it",
                    "run": ["scripts/deploy.sh", "--env", "${env}"],
                    "parameters": {"env": {"type": "string"}},
                    "timeout_secs": 120
                },
                {
                    "name": "wiki", "form": "mcp",
                    "server": {
                        "command": ["python", "server/main.py"],
                        "env": {"WIKI_TOKEN": "${ENV:WIKI_TOKEN}"}
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    write_script(&dir.join("scripts/deploy.sh"), "#!/bin/sh\necho deployed\n");
    std::fs::write(dir.join("server/main.py"), "print('wiki')\n").unwrap();
}

#[tokio::test]
async fn plan_then_commit_installs_and_pins() {
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    write_basic_plugin(&source_dir);

    let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .expect("plan");

    // Review digest: full argv + absolute paths (mechanism side of UX
    // criterion 3).
    let deploy = plan
        .digest
        .tools
        .iter()
        .find(|t| t.identity == "plugin.my-tools.deploy")
        .unwrap();
    let run_abs = plugins_root
        .join("my-tools")
        .join("scripts/deploy.sh")
        .to_string_lossy()
        .into_owned();
    assert_eq!(deploy.summary, format!("{run_abs} --env ${{env}}"));
    assert_eq!(deploy.timeout_secs, Some(120));

    // Before human confirmation: neither the final directory nor the
    // registry exists.
    assert!(!plugins_root.join("my-tools").exists());
    assert!(!PluginRegistry::path_in(&plugins_root).exists());

    // Standard install trio: env only flows as ${ENV:VAR} references
    // (never resolved).
    let mut registry = PluginRegistry::new();
    let installed = commit_install(&plan, &mut registry).expect("commit");
    assert!(plugins_root.join("my-tools/plugin.json").is_file());
    assert!(registry.get("my-tools").is_some());

    let pins = &registry.get("my-tools").unwrap().tools;
    let script_pin = pins
        .iter()
        .find_map(|t| match t {
            ToolPin::Script { identity, run } if identity == "plugin.my-tools.deploy" => {
                Some(run.clone())
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(script_pin[0], run_abs, "run[0] pinned to an absolute path");
    assert!(matches!(
        &pins[1],
        ToolPin::Mcp { env, command: Some(cmd), .. }
            if env.get("WIKI_TOKEN").map(String::as_str) == Some("${ENV:WIKI_TOKEN}")
                && cmd[0] == "python"
                && cmd[1].ends_with("server/main.py")
                && Path::new(&cmd[1]).is_absolute()
    ));
    assert_eq!(
        installed.identity_keys,
        vec!["plugin.my-tools.deploy", "plugin.my-tools.wiki"]
    );

    // The manifest hash in the registry matches the one on disk (load-time
    // re-verification passes).
    let report = registry.verify(&plugins_root.join("my-tools"));
    assert_eq!(report.package, PackageStatus::Trusted);
}

#[tokio::test]
async fn discard_leaves_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    write_basic_plugin(&source_dir);

    let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .expect("plan");
    plan.discard();
    assert!(!plugins_root.join("my-tools").exists());
    let leftovers = std::fs::read_dir(&plugins_root)
        .map(|rd| rd.count())
        .unwrap_or(0);
    assert_eq!(leftovers, 0, "staging must be cleaned up too");
}

#[tokio::test]
async fn replace_reinstall_swaps_pin() {
    // "Re-run plugin add to re-review": installing the same name again =
    // replace + re-pin.
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    write_basic_plugin(&source_dir);

    let mut registry = PluginRegistry::new();
    let plan = plan_install(&PluginSource::Path(source_dir.clone()), &plugins_root)
        .await
        .unwrap();
    commit_install(&plan, &mut registry).unwrap();

    // The author changed the script content and installed again.
    write_script(
        &source_dir.join("scripts/deploy.sh"),
        "#!/bin/sh\necho v2\n",
    );
    let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .unwrap();
    commit_install(&plan, &mut registry).unwrap();

    let report = registry.verify(&plugins_root.join("my-tools"));
    assert_eq!(report.package, PackageStatus::Trusted);
}

#[tokio::test]
async fn remove_clears_dir_registry_and_ledger() {
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    write_basic_plugin(&source_dir);

    let mut registry = PluginRegistry::new();
    let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .unwrap();
    let installed = commit_install(&plan, &mut registry).unwrap();

    // Record a ledger tier first (simulate prior use).
    let mut ledger = crate::approval_ledger::ApprovalLedger::new();
    ledger.set_tier(
        "plugin.my-tools.deploy",
        crate::approval_ledger::ApprovalTier::AlwaysAllow,
    );
    ledger.set_tier(
        "plugin.other.search",
        crate::approval_ledger::ApprovalTier::AlwaysAllow,
    );

    let report = remove_plugin(&plugins_root, &mut registry, "my-tools").expect("remove");
    assert!(report.dir_removed);
    assert!(report.registry_removed);
    assert!(registry.get("my-tools").is_none());
    assert!(!installed.plugin_dir.exists());

    // One of the three places: plugin.my-tools.* wiped, others untouched.
    let swept = report.sweep_ledger(&mut ledger);
    assert_eq!(swept, 1);
    assert_eq!(
        ledger.tier_of("plugin.my-tools.deploy"),
        crate::approval_ledger::ApprovalTier::Ask
    );
    assert_eq!(
        ledger.tier_of("plugin.other.search"),
        crate::approval_ledger::ApprovalTier::AlwaysAllow
    );
    assert_eq!(
        report.identity_keys,
        vec!["plugin.my-tools.deploy", "plugin.my-tools.wiki"]
    );
}

#[tokio::test]
async fn remove_unknown_plugin_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let mut registry = PluginRegistry::new();
    let err = remove_plugin(&plugins_root, &mut registry, "ghost").unwrap_err();
    assert!(matches!(err, InstallError::NotInstalled { name } if name == "ghost"));
}

// ── zip-slip / archive channel ────────────────────────────────────

#[test]
fn normalize_relative_rejects_traversal() {
    assert!(normalize_relative(Path::new("ok/inner.txt")).is_some());
    assert!(normalize_relative(Path::new("./ok")).is_some());
    assert!(normalize_relative(Path::new("../escape")).is_none());
    assert!(normalize_relative(Path::new("a/../../b")).is_none());
    assert!(normalize_relative(Path::new("/abs")).is_none());
    assert!(normalize_relative(Path::new("")).is_none());
}

#[test]
fn tar_gz_zip_slip_is_rejected() {
    // Build a tar.gz with a `../evil` entry (bypassing tar::Builder's path
    // law by writing the GNU header directly — exactly how real malicious
    // packages are made).
    let mut raw = Vec::new();
    {
        let encoder = flate2::write::GzEncoder::new(&mut raw, flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(5);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        header.as_gnu_mut().unwrap().name[..7].copy_from_slice(b"../evil");
        header.set_cksum();
        builder.append(&header, &b"pwned"[..]).expect("append");
        builder.finish().expect("finish");
    }

    let tmp = tempfile::tempdir().unwrap();
    let err = extract_archive(&raw, ArchiveFormat::TarGz, tmp.path()).unwrap_err();
    assert!(
        matches!(err, InstallError::ZipSlip { ref entry } if entry.contains("evil")),
        "{err:?}"
    );
    assert!(!tmp.path().join("../evil").exists());
}

#[test]
fn tar_gz_extracts_clean_archive() {
    let mut raw = Vec::new();
    {
        let encoder = flate2::write::GzEncoder::new(&mut raw, flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(5);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "plugin.json", &b"hello"[..])
            .unwrap();
        builder.finish().unwrap();
    }
    let tmp = tempfile::tempdir().unwrap();
    extract_archive(&raw, ArchiveFormat::TarGz, tmp.path()).unwrap();
    assert_eq!(
        std::fs::read_to_string(tmp.path().join("plugin.json")).unwrap(),
        "hello"
    );
}

// ── source parsing ────────────────────────────────────────────────────

#[test]
fn source_parse_channels() {
    assert_eq!(
        PluginSource::parse("https://x.example/my-tools.git").unwrap(),
        PluginSource::Git {
            url: "https://x.example/my-tools.git".into(),
            rev: None
        }
    );
    assert_eq!(
        PluginSource::parse("git@github.com:me/my-tools.git").unwrap(),
        PluginSource::Git {
            url: "git@github.com:me/my-tools.git".into(),
            rev: None
        }
    );
    assert_eq!(
        PluginSource::parse("https://x.example/my-tools.tar.gz").unwrap(),
        PluginSource::ArchiveUrl {
            url: "https://x.example/my-tools.tar.gz".into(),
            format: ArchiveFormat::TarGz
        }
    );
    assert!(matches!(
        PluginSource::parse("https://x.example/page.html"),
        Err(InstallError::BadRequest { .. })
    ));
    assert!(matches!(
        PluginSource::parse("/definitely/not/here"),
        Err(InstallError::BadRequest { .. })
    ));
    assert_eq!(
        PluginSource::parse("packs/my-tools.zip").unwrap(),
        PluginSource::ArchivePath {
            path: PathBuf::from("packs/my-tools.zip"),
            format: ArchiveFormat::Zip
        }
    );
}

#[test]
fn pin_server_argv_pins_paths_keeps_interpreters() {
    let root = abs("plugin-root-test");
    let pinned = pin_server_argv(
        &[
            "python".into(),
            "server/main.py".into(),
            abs("/usr/bin/python3").to_string_lossy().into_owned(),
        ],
        &root,
    )
    .unwrap();
    assert_eq!(pinned[0], "python", "bare-name interpreter kept as-is");
    assert_eq!(
        pinned[1],
        root.join("server/main.py").to_string_lossy().into_owned(),
        "in-package relative path pinned absolute"
    );
    assert!(
        Path::new(&pinned[2]).is_absolute(),
        "explicit absolute interpreter kept as-is"
    );

    // Path-shaped arguments that escape the plugin root are rejected.
    let err = pin_server_argv(&["python".into(), "../evil.py".into()], &root).unwrap_err();
    assert!(
        matches!(err, ScriptLawError::EscapedPluginRoot { .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn wasm_missing_at_install_is_reported() {
    // plugin.json lists a wasm tool but the .wasm is missing → cannot
    // install (fail-closed); a half package is never pinned into the
    // registry.
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    std::fs::create_dir_all(&source_dir).unwrap();
    std::fs::write(
            source_dir.join("plugin.json"),
            serde_json::json!({
                "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
                "tools": [{"name": "search", "form": "wasm", "module": "wasm/s.wasm", "description": "d"}]
            })
            .to_string(),
        )
        .unwrap();

    let err = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .unwrap_err();
    assert!(matches!(err, InstallError::Io(_)), "{err:?}");
}

#[tokio::test]
async fn script_missing_at_install_is_reported() {
    // Same law as missing wasm files: the manifest lists a script tool but
    // the file is absent (acceptance fixture: a missing script file must
    // block a half-package install) → cannot install, and the error says
    // which file is missing.
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    std::fs::create_dir_all(source_dir.join("scripts")).unwrap();
    std::fs::write(
        source_dir.join("plugin.json"),
        serde_json::json!({
            "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
            "tools": [{
                "name": "deploy", "form": "script",
                "description": "deploy it",
                "run": ["scripts/deploy.sh", "--env", "${env}"],
                "parameters": {"env": {"type": "string"}}
            }]
        })
        .to_string(),
    )
    .unwrap();

    let err = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("scripts/deploy.sh") && err.to_string().contains("does not exist"),
        "missing files must be named: {err}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn script_not_executable_at_install_is_reported() {
    // `run[0]` is the script body itself, executed directly — no
    // executable bit = it would not run even if installed, so reject at
    // install time (chmod +x and come back).
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    std::fs::create_dir_all(source_dir.join("scripts")).unwrap();
    std::fs::write(
        source_dir.join("plugin.json"),
        serde_json::json!({
            "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
            "tools": [{
                "name": "deploy", "form": "script",
                "description": "deploy it",
                "run": ["scripts/deploy.sh"]
            }]
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(source_dir.join("scripts/deploy.sh"), "#!/bin/sh\necho hi\n").unwrap();
    // 0644 — no executable bit.

    let err = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("not executable"),
        "a missing executable bit must be stated clearly: {err}"
    );
}

#[tokio::test]
async fn script_interpreter_form_checks_the_script_arg_not_the_interpreter() {
    // Interpreter form: the executable bit is checked on the interpreter
    // (PATH resolution, not the installer's business), but the script
    // argument must be in the package.
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    std::fs::create_dir_all(source_dir.join("scripts")).unwrap();
    std::fs::write(
        source_dir.join("plugin.json"),
        serde_json::json!({
            "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
            "tools": [{
                "name": "deploy", "form": "script",
                "description": "deploy it",
                "run": ["cmd", "/c", "scripts/deploy.bat"]
            }]
        })
        .to_string(),
    )
    .unwrap();

    let err = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("deploy.bat") && err.to_string().contains("does not exist"),
        "the interpreter form checks the script argument: {err}"
    );
}

#[tokio::test]
async fn capability_pinned_with_resolved_values() {
    let tmp = tempfile::tempdir().unwrap();
    let plugins_root = tmp.path().join("plugins");
    let source_dir = tmp.path().join("src-plugin");
    std::fs::create_dir_all(source_dir.join("wasm")).unwrap();
    std::fs::write(source_dir.join("wasm/s.wasm"), b"wasm-bytes").unwrap();
    std::fs::write(
        source_dir.join("plugin.json"),
        serde_json::json!({
            "schemaVersion": 1, "name": "my-tools", "version": "0.1.0",
            "tools": [{
                "name": "search", "form": "wasm", "module": "wasm/s.wasm", "description": "d",
                "capabilities": [
                    {"type": "fs:read", "paths": ["./data"]},
                    {"type": "net:http", "hosts": ["API.Example.com"]}
                ]
            }]
        })
        .to_string(),
    )
    .unwrap();

    let plan = plan_install(&PluginSource::Path(source_dir), &plugins_root)
        .await
        .expect("plan");
    match &plan.pin.tools[0] {
        ToolPin::Wasm {
            capabilities,
            wasm_sha256,
            module,
            ..
        } => {
            let data_abs = plugins_root
                .join("my-tools")
                .join("data")
                .to_string_lossy()
                .into_owned();
            assert_eq!(
                capabilities,
                &vec![
                    ResolvedCapability::FsRead {
                        paths: vec![data_abs]
                    },
                    ResolvedCapability::NetHttp {
                        hosts: vec!["api.example.com".into()]
                    }
                ]
            );
            assert_eq!(*wasm_sha256, hex_sha256(b"wasm-bytes"));
            assert!(Path::new(module).is_absolute());
        }
        other => panic!("expected wasm pin, got {other:?}"),
    }
}
