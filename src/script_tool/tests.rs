use std::path::{Path, PathBuf};
use std::time::Duration;

use agent_base::tool::Tool;
use agent_base::{AgentError, ToolContext};

use crate::tool_provider::{ToolIdentity, ToolProvider};

use super::*;
use serde_json::json;

fn s(v: &str) -> String {
    v.to_string()
}

/// Platform absolute path (the test criterion follows the same law as `Path::is_absolute`, runs
/// the same on unix/windows).
fn abs(rel: &str) -> String {
    std::env::current_dir()
        .unwrap()
        .join(rel)
        .to_string_lossy()
        .into_owned()
}

fn entry(run: &[&str]) -> ScriptToolEntry {
    ScriptToolEntry::new(
        "deploy",
        "deploy the project",
        run.iter().map(|s| s.to_string()).collect(),
    )
    .expect("entry must pass the path law")
}

// ── Path law (anti-shadowing) ────────────────────────────────────────────

#[test]
fn run_law_accepts_absolute_program() {
    let program = abs("deploy.sh");
    assert!(check_run_law(&[s(&program), s("--env"), s("prod")]).is_ok());
}

#[test]
fn run_law_rejects_relative_program_global_anti_shadowing() {
    // Global tools.json + relative path = another project shadowing a same-named script → reject.
    for program in ["./deploy.sh", "deploy.sh", "../deploy.sh", "~/deploy.sh"] {
        let err = check_run_law(&[s(program)]).unwrap_err();
        assert!(
            matches!(err, ScriptLawError::RelativeProgram { .. }),
            "{program}: {err}"
        );
    }
}

#[test]
fn run_law_interpreter_whitelist_requires_absolute_script_arg() {
    // Whitelisted interpreter: the script file argument must be an absolute path.
    for (interpreter, flag, script_rel) in [
        ("powershell", "-File", "deploy.ps1"),
        ("pwsh", "-f", "deploy.ps1"),
        ("powershell.exe", "-File", "deploy.ps1"),
        ("cmd", "/c", "deploy.cmd"),
        ("cmd.exe", "/C", "deploy.cmd"),
    ] {
        let script = abs(script_rel);
        let ok = vec![s(interpreter), s(flag), s(&script), s("prod")];
        assert!(check_run_law(&ok).is_ok(), "{interpreter} absolute script");

        let bad = vec![s(interpreter), s(flag), s("./deploy.ps1")];
        let err = check_run_law(&bad).unwrap_err();
        assert!(
            matches!(err, ScriptLawError::RelativeScriptArg { .. }),
            "{interpreter} relative script: {err}"
        );

        // No script argument found → cannot pin → reject.
        let missing = vec![s(interpreter), s("-Command"), s("echo hi")];
        let err = check_run_law(&missing).unwrap_err();
        assert!(
            matches!(err, ScriptLawError::InterpreterMissingScriptArg { .. }),
            "{interpreter} missing script arg: {err}"
        );
    }
}

#[test]
fn run_law_empty_run_rejected() {
    assert_eq!(check_run_law(&[]).unwrap_err(), ScriptLawError::EmptyRun);
}

#[test]
fn non_interpreter_bare_name_is_not_exempt() {
    // `npx` is not on the whitelist → run[0] must still be an absolute path.
    let err = check_run_law(&[s("npx"), s("-y"), s("pkg")]).unwrap_err();
    assert!(matches!(err, ScriptLawError::RelativeProgram { .. }));
}

// ── In-plugin path pinning (the same law, plugins are not exempt) ─────────────────────

#[test]
fn pin_from_plugin_root_pins_relative_script_absolute() {
    let root = PathBuf::from(abs("plugins/my-tools"));
    let pinned =
        pin_from_plugin_root(&[s("scripts/deploy.sh"), s("--env"), s("${env}")], &root).unwrap();
    assert_eq!(
        Path::new(&pinned[0]),
        root.join("scripts/deploy.sh").as_path(),
        "package-relative path pinned to an absolute path under the plugin root"
    );
    assert!(Path::new(&pinned[0]).is_absolute());
    // After pinning it must pass the same path law
    assert!(check_run_law(&pinned).is_ok());
}

#[test]
fn pin_from_plugin_root_rejects_escape_relative_and_absolute() {
    let root = PathBuf::from(abs("plugins/my-tools"));
    let outside = abs("plugins/other/x.sh");
    for bad in [
        vec![s("../outside.sh")],
        vec![s("scripts/../../outside.sh")],
        vec![s(&outside)],
    ] {
        let err = pin_from_plugin_root(&bad, &root).unwrap_err();
        assert!(
            matches!(err, ScriptLawError::EscapedPluginRoot { .. }),
            "{bad:?}: {err}"
        );
    }
}

#[test]
fn pin_from_plugin_root_pins_interpreter_script_arg() {
    let root = PathBuf::from(abs("plugins/my-tools"));
    let pinned = pin_from_plugin_root(
        &[
            s("powershell"),
            s("-File"),
            s("scripts/deploy.ps1"),
            s("-env"),
            s("prod"),
        ],
        &root,
    )
    .unwrap();
    assert_eq!(
        pinned[0], "powershell",
        "the interpreter itself is not pinned (whitelist)"
    );
    assert_eq!(
        Path::new(&pinned[2]),
        root.join("scripts/deploy.ps1").as_path(),
        "the interpreter's script argument is pinned"
    );
    // An escaping interpreter script argument is rejected just the same
    let err =
        pin_from_plugin_root(&[s("powershell"), s("-File"), s("../x.ps1")], &root).unwrap_err();
    assert!(matches!(err, ScriptLawError::EscapedPluginRoot { .. }));
}

// ── ${param} substitution (argv, fail-closed, no re-expansion) ──────────────────────

#[test]
fn substitute_replaces_whole_and_inline_placeholders() {
    let run = [
        s("/abs/deploy.sh"),
        s("--env"),
        s("${env}"),
        s("--tag=${tag}"),
    ];
    let args = json!({"env": "prod", "tag": "v1"});
    let got = substitute_params("script.deploy", &run, &args).unwrap();
    assert_eq!(got, vec!["/abs/deploy.sh", "--env", "prod", "--tag=v1"]);
}

#[test]
fn substitute_missing_param_is_tool_error() {
    let run = [s("/abs/deploy.sh"), s("${env}")];
    let err = substitute_params("script.deploy", &run, &json!({})).unwrap_err();
    assert!(matches!(err, AgentError::ToolArgsInvalid { name, .. } if name == "script.deploy"));
}

#[test]
fn substitute_non_scalar_param_is_tool_error() {
    let run = [s("/abs/deploy.sh"), s("${env}")];
    for bad in [
        json!({"env": null}),
        json!({"env": ["a"]}),
        json!({"env": {"a": 1}}),
    ] {
        let err = substitute_params("script.deploy", &run, &bad).unwrap_err();
        assert!(matches!(err, AgentError::ToolArgsInvalid { .. }), "{bad}");
    }
}

#[test]
fn substitute_scalar_params_stringify() {
    let run = [s("${n}"), s("${b}"), s("${s}")];
    let args = json!({"n": 42, "b": true, "s": "txt"});
    let got = substitute_params("script.deploy", &run, &args).unwrap();
    assert_eq!(got, vec!["42", "true", "txt"]);
}

#[test]
fn substitute_does_not_reexpand_values() {
    // ${x} inside a parameter value is not re-expanded; bare $FOO / non-${ forms are untouched.
    let run = [s("${a}"), s("$FOO"), s("plain")];
    let args = json!({"a": "${b}", "b": "NOPE"});
    let got = substitute_params("script.deploy", &run, &args).unwrap();
    assert_eq!(got, vec!["${b}", "$FOO", "plain"]);
}

#[test]
fn substitute_unclosed_placeholder_fails_closed() {
    let err = substitute_params(
        "script.deploy",
        &[s("/abs/x.sh ${env")],
        &json!({"env": "p"}),
    )
    .unwrap_err();
    assert!(matches!(err, AgentError::ToolArgsInvalid { .. }));
}

#[test]
fn substitute_non_object_args_fails_closed() {
    let err = substitute_params("script.deploy", &[s("x")], &json!(["a"])).unwrap_err();
    assert!(matches!(err, AgentError::ToolArgsInvalid { .. }));
}

// ── serde (tools.json entry form, parsing enforces the laws) ───────────────────────────

#[test]
fn entry_serde_round_trips_design_shape() {
    let script = abs("proj/deploy.sh");
    let raw = json!({
        "name": "deploy",
        "description": "部署当前项目到 staging",
        "parameters": { "env": { "type": "string", "description": "目标环境" } },
        "run": [script, "--env", "${env}"],
        "timeout_secs": 120
    });
    let entry: ScriptToolEntry = serde_json::from_value(raw).unwrap();
    assert_eq!(entry.name, "deploy");
    assert_eq!(entry.timeout_ms(), 120_000);
    assert_eq!(entry.parameters["env"].ty, ParamType::String);
    let back = serde_json::to_value(&entry).unwrap();
    assert_eq!(back["timeout_secs"], 120);
    assert_eq!(back["run"][0], script);
}

#[test]
fn entry_serde_rejects_shadowable_run() {
    let raw = json!({
        "name": "deploy",
        "description": "d",
        "run": ["deploy.sh"]
    });
    let err = serde_json::from_value::<ScriptToolEntry>(raw).unwrap_err();
    assert!(err.to_string().contains("deploy.sh"), "{err}");
}

#[test]
fn entry_serde_rejects_bad_name() {
    let raw = json!({
        "name": "Deploy",
        "description": "d",
        "run": [abs("deploy.sh")]
    });
    assert!(serde_json::from_value::<ScriptToolEntry>(raw).is_err());
}

#[test]
fn parameters_envelope_requires_all_scalars() {
    let script = abs("deploy.sh");
    let mut entry = entry(&[script.as_str()]);
    entry.parameters.insert(
        "env".into(),
        ParamSpec {
            ty: ParamType::String,
            description: Some("目标环境".into()),
        },
    );
    let envelope = entry.parameters_envelope();
    assert_eq!(envelope["type"], "object");
    assert_eq!(envelope["properties"]["env"]["type"], "string");
    assert_eq!(envelope["required"], json!(["env"]));
    assert_eq!(envelope["additionalProperties"], false);
}

// ── Execution semantics (actually run on unix; Windows semantics see the §win tests + docs) ────

#[cfg(unix)]
mod unix_exec {
    use super::*;

    #[tokio::test]
    async fn runs_argv_without_shell_and_substitutes_params() {
        let entry = entry(&["/bin/echo", "${msg}"]);
        let ctx = ToolContext::for_test();
        let out = execute_entry("script.echo", &entry, &json!({"msg": "hello"}), &ctx)
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(agent_base::tool::content_text(&out), "hello");
    }

    #[tokio::test]
    async fn env_inherits_and_overrides() {
        // argv goes through no shell; env override verification relies on /bin/sh -c (an absolute run[0] is legal).
        // Since edition 2024 env::set_var is unsafe (races with multithreaded env access).
        unsafe { std::env::set_var("SCRIPT_TOOL_TEST_INHERITED", "inh") };
        let mut entry = entry(&[
            "/bin/sh",
            "-c",
            r#"echo "$SCRIPT_TOOL_TEST_INHERITED/$SCRIPT_TOOL_TEST_SET""#,
        ]);
        entry
            .env
            .insert("SCRIPT_TOOL_TEST_SET".into(), "set".into());
        let ctx = ToolContext::for_test();
        let out = execute_entry("script.env", &entry, &json!({}), &ctx)
            .await
            .unwrap();
        assert_eq!(agent_base::tool::content_text(&out), "inh/set");
    }

    #[tokio::test]
    async fn non_zero_exit_reports_stderr_summary() {
        let entry = entry(&["/bin/sh", "-c", "echo boom >&2; exit 3"]);
        let ctx = ToolContext::for_test();
        let err = execute_entry("script.fail", &entry, &json!({}), &ctx)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("boom"), "{msg}");
        assert!(msg.contains("3"), "{msg}");
    }

    #[tokio::test]
    async fn timeout_kills_whole_process_tree() {
        // Grandchild process sleeps 30 and records its pid; after the timeout the whole tree must die.
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("grandchild.pid");
        let script = format!("sleep 30 & echo $! > {}; wait", pidfile.display());
        let mut entry = entry(&["/bin/sh", "-c", &script]);
        entry.timeout_ms = Some(1_000);
        let ctx = ToolContext::for_test();
        let err = execute_entry("script.tree", &entry, &json!({}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::ToolTimeout), "{err}");

        let pid_raw = std::fs::read_to_string(&pidfile).expect("grandchild pidfile");
        let pid: i32 = pid_raw.trim().parse().unwrap();
        for _ in 0..50 {
            let alive = unsafe { libc::kill(pid, 0) == 0 };
            if !alive {
                return; // the grandchild died with the tree
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("grandchild {pid} survived the tree kill");
    }

    #[tokio::test]
    async fn cancel_token_stops_script_and_reports_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("grandchild.pid");
        let script = format!("sleep 30 & echo $! > {}; wait", pidfile.display());
        let entry = entry(&["/bin/sh", "-c", &script]);
        let ctx = ToolContext::for_test();
        let token = ctx.cancel_token.clone();
        let run =
            tokio::spawn(
                async move { execute_entry("script.cancel", &entry, &json!({}), &ctx).await },
            );
        // Wait for the pidfile to appear (the tree has grown) before cancelling
        for _ in 0..100 {
            if pidfile.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        token.cancel();
        let err = run.await.unwrap().unwrap_err();
        assert!(matches!(err, AgentError::Cancelled), "{err}");
    }

    #[tokio::test]
    async fn output_truncation_follows_kernel_convention() {
        let entry = entry(&["/bin/sh", "-c", "seq 1 1000"]);
        let mut ctx = ToolContext::for_test();
        ctx.max_output_chars = Some(60);
        let out = execute_entry("script.big", &entry, &json!({}), &ctx)
            .await
            .unwrap();
        let text = agent_base::tool::content_text(&out);
        assert!(text.contains(TRUNCATION_MARKER), "{text}");
        assert!(
            text.chars().count() < 200,
            "still huge: {}",
            text.chars().count()
        );
        // Both head and tail are present (seq runs from 1 to 1000)
        assert!(text.starts_with('1'), "{text}");
        assert!(text.contains("1000"), "{text}");
    }

    #[test]
    fn truncation_never_exceeds_the_budget() {
        // Regression: a budget below the marker floor used to return the
        // marker itself — still over budget — and the engine then rejected
        // the whole call (`Tool '…' output exceeds the …-char limit`),
        // losing the result entirely.
        for max in 0..40usize {
            let out = truncate_output(&"x".repeat(500), Some(max));
            assert!(
                out.chars().count() <= max,
                "max={max} produced {} chars",
                out.chars().count()
            );
        }
    }

    #[tokio::test]
    async fn missing_script_file_reports_script_is_gone() {
        let entry = entry(&["/definitely/not/here/deploy.sh"]);
        let ctx = ToolContext::for_test();
        let err = execute_entry("script.gone", &entry, &json!({}), &ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("script is gone"), "{err}");
    }
}

// ── Provider / Tool wiring ───────────────────────────────────────────────

#[tokio::test]
async fn provider_lists_scripts_and_rejects_foreign_keys() {
    #[cfg(unix)]
    let entry = entry(&["/bin/echo", "hi"]);
    #[cfg(windows)]
    let entry = entry(&[r"C:\Windows\System32\cmd.exe", "/c", r"C:\proj\deploy.cmd"]);

    let provider = ScriptProvider::new([entry.clone()]);
    let listed = provider.list().await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].identity.to_key(), "script.deploy");
    assert_eq!(listed[0].timeout_ms, Some(DEFAULT_TIMEOUT_MS));

    let ctx = ToolContext::for_test();
    for foreign in [
        ToolIdentity::builtin("read_file").unwrap(),
        ToolIdentity::mcp("figma", "get_file").unwrap(),
        ToolIdentity::plugin("my-tools", "deploy").unwrap(),
        ToolIdentity::script("other").unwrap(),
    ] {
        let err = provider.call(&foreign, &json!({}), &ctx).await.unwrap_err();
        assert!(
            matches!(err, AgentError::ToolNotFound { .. }),
            "{foreign:?}"
        );
    }

    let own = entry.identity();
    provider.call(&own, &json!({}), &ctx).await.unwrap();
}

#[tokio::test]
async fn provider_health_detects_gone_script() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("deploy.sh");
    std::fs::write(&script, "#!/bin/sh\necho ok\n").unwrap();
    let entry = entry(&[script.to_str().unwrap()]);
    let provider = ScriptProvider::new([entry.clone()]);
    provider.health().await.unwrap();

    std::fs::remove_file(&script).unwrap();
    let err = provider.health().await.unwrap_err();
    assert!(err.to_string().contains("is gone"), "{err}");
}

#[tokio::test]
async fn script_tool_plugs_into_tool_table() {
    let entry = entry(&["/bin/echo", "hi"]);
    let tool = ScriptTool::new(entry.clone());
    assert_eq!(tool.name(), "script.deploy");
    assert_eq!(tool.timeout_ms(), Some(DEFAULT_TIMEOUT_MS));
    assert_eq!(tool.metadata().origin, "script");
    assert_eq!(tool.schema()["type"], "object");

    let plugin_tool = ScriptTool::with_key("plugin.my-tools.deploy", entry);
    assert_eq!(plugin_tool.name(), "plugin.my-tools.deploy");
}
