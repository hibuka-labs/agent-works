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

use std::path::PathBuf;

use super::super::manifest::{ManifestError, ReviewDigest};
use super::super::registry::{PluginPin, RegistryError};
use crate::script_tool::ScriptLawError;

/// Archive format (archive channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    /// `.zip`。
    Zip,
    /// `.tar.gz` / `.tgz`。
    TarGz,
}

/// Install source (channels capped at path + git + archive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginSource {
    /// Local directory (copy install).
    Path(PathBuf),
    /// git URL (**depends on a git executable**; on Windows it must be on
    /// PATH — made explicit).
    Git {
        /// Repository URL.
        url: String,
        /// Optional rev/tag/branch.
        rev: Option<String>,
    },
    /// Local archive.
    ArchivePath {
        /// Archive path.
        path: PathBuf,
        /// Format.
        format: ArchiveFormat,
    },
    /// Remote archive URL.
    ArchiveUrl {
        /// Download URL.
        url: String,
        /// Format.
        format: ArchiveFormat,
    },
}

impl PluginSource {
    /// Parse the source from `plugin add` arguments (used by the product CLI).
    ///
    /// `*.git` / `git@` → git; `http(s)://` + `.zip|.tar.gz|.tgz` → archive URL;
    /// anything else is treated as a local path (directory or archive file).
    pub fn parse(spec: &str) -> Result<Self, InstallError> {
        let lower = spec.to_ascii_lowercase();
        if lower.ends_with(".git") || spec.starts_with("git@") {
            return Ok(Self::Git {
                url: spec.to_string(),
                rev: None,
            });
        }
        if spec.starts_with("https://") || spec.starts_with("http://") {
            if let Some(format) = archive_format_of(&lower) {
                return Ok(Self::ArchiveUrl {
                    url: spec.to_string(),
                    format,
                });
            }
            return Err(InstallError::BadRequest {
                reason: format!("unsupported URL `{spec}` (git URL or .zip/.tar.gz/.tgz archive)"),
            });
        }
        let path = PathBuf::from(spec);
        if let Some(format) = archive_format_of(&lower) {
            return Ok(Self::ArchivePath { path, format });
        }
        if path.is_dir() {
            return Ok(Self::Path(path));
        }
        Err(InstallError::BadRequest {
            reason: format!("`{spec}` is not a plugin directory or archive"),
        })
    }
}

fn archive_format_of(lower_spec: &str) -> Option<ArchiveFormat> {
    if lower_spec.ends_with(".zip") {
        Some(ArchiveFormat::Zip)
    } else if lower_spec.ends_with(".tar.gz") || lower_spec.ends_with(".tgz") {
        Some(ArchiveFormat::TarGz)
    } else {
        None
    }
}

/// Install errors. fail-closed: nothing is written to disk if any step is invalid.
#[derive(Debug)]
pub enum InstallError {
    /// IO failure.
    Io(std::io::Error),
    /// Manifest parse/validation failure.
    Manifest(ManifestError),
    /// Path-law violation (escape from the plugin root, etc.).
    PathLaw { source: ScriptLawError },
    /// git channel failure (including **git not on PATH** — the real-world
    /// friction made explicit for Windows).
    Git(String),
    /// Download failure.
    Download(String),
    /// Archive extraction failure.
    Archive(String),
    /// **zip-slip**: member path escapes the extraction directory.
    ZipSlip { entry: String },
    /// Registry write failure.
    Registry(RegistryError),
    /// Invalid arguments.
    BadRequest { reason: String },
    /// Plugin not installed (on remove).
    NotInstalled { name: String },
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "install io: {e}"),
            Self::Manifest(e) => write!(f, "install manifest: {e}"),
            Self::PathLaw { source } => write!(f, "install path law: {source}"),
            Self::Git(e) => write!(
                f,
                "git channel failed: {e} (git URLs need the git executable on PATH; on Windows install Git for Windows)"
            ),
            Self::Download(e) => write!(f, "download failed: {e}"),
            Self::Archive(e) => write!(f, "archive extract failed: {e}"),
            Self::ZipSlip { entry } => {
                write!(
                    f,
                    "archive entry `{entry}` escapes the extract root (zip-slip)"
                )
            }
            Self::Registry(e) => write!(f, "registry: {e}"),
            Self::BadRequest { reason } => write!(f, "{reason}"),
            Self::NotInstalled { name } => write!(f, "plugin `{name}` is not installed"),
        }
    }
}

impl std::error::Error for InstallError {}

/// Staging layout (the output of `plan_install`, the input of `commit_install`).
#[derive(Debug)]
pub struct StagingLayout {
    /// Staging directory (contents already in the plugin's full layout).
    pub dir: PathBuf,
}

/// An install plan awaiting human review: staged content + review digest + pinned values.
///
/// **[`commit_install`] only after human confirmation** (product-level gate);
/// [`InstallPlan::discard`] clears the staging area if not installing.
#[derive(Debug)]
pub struct InstallPlan {
    /// Review digest (full manifest + capability breakdown + absolute paths).
    pub digest: ReviewDigest,
    /// Pinned values (paths already absolutized against the **final install dir**).
    pub pin: PluginPin,
    /// Final install directory (`<plugins_root>/<name>`).
    pub plugin_root: PathBuf,
    /// Staging layout.
    pub staging: StagingLayout,
    pub(super) plugins_root: PathBuf,
}

impl InstallPlan {
    /// Abandon the install: clear staging, write nothing to disk.
    pub fn discard(self) {
        let _ = std::fs::remove_dir_all(&self.staging.dir);
    }
}

/// Install result.
#[derive(Debug, Clone)]
pub struct InstalledPlugin {
    /// The pins (already written into the registry).
    pub pin: PluginPin,
    /// Install directory.
    pub plugin_dir: PathBuf,
    /// Tool identity keys (for tool-table registration / ledger cleanup).
    pub identity_keys: Vec<String>,
}

/// Uninstall report: receipts for the three-place cleanup (tool table / ledger /
/// registry).
#[derive(Debug, Clone)]
pub struct RemovalReport {
    /// Plugin name.
    pub name: String,
    /// Tool identity keys (the caller deregisters these from the tool table).
    pub identity_keys: Vec<String>,
    /// Ledger prefix (`plugin.{name}.`).
    pub ledger_prefix: String,
    /// Whether the install directory has been deleted.
    pub dir_removed: bool,
    /// Whether the registry entry has been removed.
    pub registry_removed: bool,
}

impl RemovalReport {
    /// Sweep the ledger: wipe all tiers of `plugin.{name}.*` (no orphaned memory).
    pub fn sweep_ledger(&self, ledger: &mut crate::approval_ledger::ApprovalLedger) -> usize {
        ledger.remove_prefix(&self.ledger_prefix)
    }
}
