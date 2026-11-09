//! Plugin install lifecycle: `plugin add` / `plugin remove` (human review before
//! install → write to disk → pin → uninstall).
//!
//! The channels are capped at **path + git + archive**; this does not grow into a
//! package manager (marketplace / central catalog deferred).
//!
//! - **Human review before install**: [`plan_install`] only stages + produces a
//!   [`ReviewDigest`] (full manifest + capability breakdown); only **after human
//!   confirmation** does [`commit_install`] write to disk (product-level gate;
//!   plugins requested by an agent are never allowed through by default);
//! - **Pinned values**: manifest hash, each .wasm hash, `run`/module/server paths
//!   absolutized, resolved values for capabilities' paths/hosts — all pinned into
//!   [`PluginRegistry`] (the authority **outside** the plugin directory);
//! - **zip-slip**: reject if an archive member path escapes the extraction
//!   directory;
//! - **git dependency made explicit** (Windows): the git URL channel depends on a
//!   `git` executable on `PATH` (may embed gix later);
//! - **Uninstall cleans all three places**: tool table (identity_keys returned to
//!   the caller for deregistration) / ledger ([`RemovalReport::sweep_ledger`]) /
//!   registry entries.
//!
//! secret rule: env/headers only flow as `${ENV:VAR}` references; this module
//! never resolves them and never writes literals.

use std::path::{Path, PathBuf};

use super::super::manifest::{ManifestError, PluginManifest, PluginToolSpec, resolve_capabilities};
use super::super::registry::{PluginPin, ToolPin, hex_sha256};
use crate::script_tool::{
    ScriptLawError, check_script_file_on_disk, pin_from_plugin_root, resolve_under_root,
    script_file_index, script_file_is_program,
};

use super::types::*;

/// Staging → plan: parse the manifest, compute pinned values against the
/// **final directory**, produce the review digest.
pub(super) fn build_plan(
    staging_dir: PathBuf,
    plugins_root: &Path,
) -> Result<InstallPlan, InstallError> {
    let manifest_path = staging_dir.join("plugin.json");
    let manifest_raw = std::fs::read(&manifest_path).map_err(InstallError::Io)?;
    let manifest_text = String::from_utf8_lossy(&manifest_raw).into_owned();
    let manifest = PluginManifest::from_json(&manifest_text).map_err(InstallError::Manifest)?;

    let plugin_root = plugins_root.join(&manifest.name);
    let digest = manifest
        .review(&plugin_root)
        .map_err(InstallError::Manifest)?;

    let pin = build_pin(&manifest, &manifest_raw, &plugin_root, &staging_dir)?;

    Ok(InstallPlan {
        digest,
        pin,
        plugin_root,
        staging: StagingLayout { dir: staging_dir },
        plugins_root: plugins_root.to_path_buf(),
    })
}

pub(super) fn build_pin(
    manifest: &PluginManifest,
    manifest_raw: &[u8],
    plugin_root: &Path,
    staging_dir: &Path,
) -> Result<PluginPin, InstallError> {
    let mut tools = Vec::with_capacity(manifest.tools.len());
    for tool in &manifest.tools {
        let identity = tool.identity(&manifest.name).map_err(|source| {
            InstallError::Manifest(ManifestError::BadName {
                name: tool.name().to_string(),
                source,
            })
        })?;
        match tool {
            PluginToolSpec::Wasm(t) => {
                let module = resolve_under_root(plugin_root, &t.module)
                    .map_err(|source| InstallError::PathLaw { source })?;
                // Hashes are computed over the staged content (bytes unchanged
                // after the rename).
                let staged_module = staging_dir.join(&t.module);
                let bytes = std::fs::read(&staged_module).map_err(InstallError::Io)?;
                let capabilities = resolve_capabilities(&t.capabilities, plugin_root)
                    .map_err(|source| InstallError::PathLaw { source })?;
                tools.push(ToolPin::Wasm {
                    identity,
                    module,
                    wasm_sha256: hex_sha256(&bytes),
                    capabilities,
                });
            }
            PluginToolSpec::Script(t) => {
                let run = pin_from_plugin_root(&t.run, plugin_root)
                    .map_err(|source| InstallError::PathLaw { source })?;
                // At install time the script files must be present in the
                // package — the same fail-closed law as missing wasm files
                // (acceptance fixture: a missing script file must block a
                // half-package install). We check the staged content (bytes
                // unchanged after the rename); if something is missing, report
                // which file is missing.
                let index =
                    script_file_index(&t.run).map_err(|source| InstallError::PathLaw { source })?;
                let require_executable = script_file_is_program(&t.run)
                    .map_err(|source| InstallError::PathLaw { source })?;
                check_script_file_on_disk(&staging_dir.join(&t.run[index]), require_executable)
                    .map_err(|source| InstallError::PathLaw { source })?;
                tools.push(ToolPin::Script { identity, run });
            }
            PluginToolSpec::Mcp(t) => {
                let command = match &t.server.command {
                    Some(cmd) => Some(
                        pin_server_argv(cmd, plugin_root)
                            .map_err(|source| InstallError::PathLaw { source })?,
                    ),
                    None => None,
                };
                tools.push(ToolPin::Mcp {
                    label: t.name.clone(),
                    command,
                    env: t.server.env.clone(),
                    url: t.server.url.clone(),
                    headers: t.server.headers.clone(),
                });
            }
        }
    }

    Ok(PluginPin {
        name: manifest.name.clone(),
        version: manifest.version.clone(),
        source: "installed".into(),
        manifest_sha256: hex_sha256(manifest_raw),
        tools,
    })
}

/// Argv pinning law for an in-package MCP server (same anti-substitution intent
/// as `run[0]`, trimmed to the server context):
///
/// - **Path-shaped element** (contains a separator / starts with `.`) =
///   in-package relative path → resolve to an absolute path under the plugin
///   root and pin it (lexical normalization + reject escape from the plugin
///   root);
/// - **Bare name** (`python` / `npx` …) = interpreter/launcher resolved via
///   PATH lookup, kept as-is — the server's author experience comes first
///   (design example `["python", "server/main.py"]`);
/// - **Absolute path** (`/usr/bin/python3`) = explicit interpreter, kept as-is.
pub(super) fn pin_server_argv(
    argv: &[String],
    plugin_root: &Path,
) -> Result<Vec<String>, ScriptLawError> {
    argv.iter()
        .map(|arg| {
            let pathish = arg.contains('/') || arg.contains('\\') || arg.starts_with('.');
            if !pathish || Path::new(arg).is_absolute() {
                return Ok(arg.clone());
            }
            resolve_under_root(plugin_root, arg)
        })
        .collect()
}
