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

use std::path::Path;

use super::super::registry::{PluginRegistry, ToolPin};

use super::build::*;
use super::source::*;
use super::types::*;

/// `plugin add` step one: materialize the source into staging, validate the
/// manifest, compute pinned values, produce the review digest.
///
/// **Writes nothing to disk** (nothing into `<plugins_root>/<name>/`, nothing to
/// the registry) — only after a human has reviewed [`InstallPlan::digest`] and
/// confirmed does [`commit_install`] run.
pub async fn plan_install(
    source: &PluginSource,
    plugins_root: &Path,
) -> Result<InstallPlan, InstallError> {
    let staging_dir = staging_dir_for(plugins_root);
    std::fs::create_dir_all(&staging_dir).map_err(InstallError::Io)?;

    match materialize(source, &staging_dir).await {
        Ok(()) => {}
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging_dir);
            return Err(e);
        }
    }
    match build_plan(staging_dir, plugins_root) {
        Ok(plan) => Ok(plan),
        Err(e) => Err(e),
    }
}

/// `plugin add` step two (**after human confirmation**): staging → final
/// directory, pinned into the registry.
pub fn commit_install(
    plan: &InstallPlan,
    registry: &mut PluginRegistry,
) -> Result<InstalledPlugin, InstallError> {
    // Replacement semantics for re-adding: delete the old directory first
    // (same-name plugin slot).
    if plan.plugin_root.exists() {
        std::fs::remove_dir_all(&plan.plugin_root).map_err(InstallError::Io)?;
    }
    if let Some(parent) = plan.plugin_root.parent() {
        std::fs::create_dir_all(parent).map_err(InstallError::Io)?;
    }
    std::fs::rename(&plan.staging.dir, &plan.plugin_root).map_err(InstallError::Io)?;

    registry.insert(plan.pin.clone());
    let registry_path = PluginRegistry::path_in(&plan.plugins_root);
    registry
        .save(&registry_path)
        .map_err(InstallError::Registry)?;

    let identity_keys = plan
        .pin
        .tools
        .iter()
        .flat_map(|t| match t {
            // The real tool keys of the mcp form expand at load time by server
            // tool name; here we give label keys, and the loader fills them in.
            ToolPin::Wasm { identity, .. } | ToolPin::Script { identity, .. } => {
                vec![identity.clone()]
            }
            ToolPin::Mcp { label, .. } => vec![format!("plugin.{}.{label}", plan.pin.name)],
        })
        .collect();

    Ok(InstalledPlugin {
        pin: plan.pin.clone(),
        plugin_dir: plan.plugin_root.clone(),
        identity_keys,
    })
}

/// `plugin remove`: clean all three places (tool-table keys returned, ledger
/// prefix sweepable, registry pins pulled), delete the install directory.
pub fn remove_plugin(
    plugins_root: &Path,
    registry: &mut PluginRegistry,
    name: &str,
) -> Result<RemovalReport, InstallError> {
    let pin = registry
        .remove(name)
        .ok_or_else(|| InstallError::NotInstalled {
            name: name.to_string(),
        })?;

    let registry_path = PluginRegistry::path_in(plugins_root);
    registry
        .save(&registry_path)
        .map_err(InstallError::Registry)?;

    let plugin_dir = plugins_root.join(name);
    let dir_removed = if plugin_dir.exists() {
        std::fs::remove_dir_all(&plugin_dir).map_err(InstallError::Io)?;
        true
    } else {
        false
    };

    let mut identity_keys = Vec::new();
    for tool in &pin.tools {
        match tool {
            ToolPin::Wasm { identity, .. } | ToolPin::Script { identity, .. } => {
                identity_keys.push(identity.clone());
            }
            ToolPin::Mcp { label, .. } => {
                identity_keys.push(format!("plugin.{name}.{label}"));
            }
        }
    }

    Ok(RemovalReport {
        name: name.to_string(),
        identity_keys,
        ledger_prefix: format!("plugin.{name}."),
        dir_removed,
        registry_removed: true,
    })
}
