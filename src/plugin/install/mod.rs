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

// Doc-link targets for the module docs above; nothing in this facade uses them.
#[allow(unused_imports)]
use super::manifest::ReviewDigest;
#[allow(unused_imports)]
use super::registry::PluginRegistry;

mod build;
mod ops;
mod source;
mod types;

pub use ops::*;
pub use types::*;

#[cfg(test)]
mod tests;
