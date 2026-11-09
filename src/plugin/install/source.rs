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

use super::types::*;

// ── internal: materialize / plan ──────────────────────────────────────────

pub(super) fn staging_dir_for(plugins_root: &Path) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    plugins_root.join(format!(".staging-{stamp}"))
}

pub(super) async fn materialize(source: &PluginSource, staging: &Path) -> Result<(), InstallError> {
    match source {
        PluginSource::Path(path) => {
            copy_dir(path, staging)?;
            Ok(())
        }
        PluginSource::Git { url, rev } => git_clone(url, rev.as_deref(), staging).await,
        PluginSource::ArchivePath { path, format } => {
            let raw = std::fs::read(path).map_err(InstallError::Io)?;
            extract_archive(&raw, *format, staging)
        }
        PluginSource::ArchiveUrl { url, format } => {
            let raw = download(url).await?;
            extract_archive(&raw, *format, staging)
        }
    }
}

pub(super) async fn download(url: &str) -> Result<Vec<u8>, InstallError> {
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| InstallError::Download(format!("{url}: {e}")))?;
    if !response.status().is_success() {
        return Err(InstallError::Download(format!(
            "{url}: HTTP {}",
            response.status()
        )));
    }
    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| InstallError::Download(format!("{url}: {e}")))
}

/// git channel: `git clone` (+ optional rev). Depends on a git executable
/// (explicit for Windows).
pub(super) async fn git_clone(
    url: &str,
    rev: Option<&str>,
    staging: &Path,
) -> Result<(), InstallError> {
    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("clone")
        .arg("--depth")
        .arg("1")
        .arg(url)
        .arg(staging)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    if let Some(rev) = rev {
        cmd.arg("--branch").arg(rev);
    }
    let output = cmd
        .output()
        .await
        .map_err(|e| InstallError::Git(format!("spawn git: {e}")))?;
    if !output.status.success() {
        return Err(InstallError::Git(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(())
}

pub(super) fn copy_dir(from: &Path, to: &Path) -> Result<(), InstallError> {
    std::fs::create_dir_all(to).map_err(InstallError::Io)?;
    for entry in std::fs::read_dir(from).map_err(InstallError::Io)? {
        let entry = entry.map_err(InstallError::Io)?;
        let target = to.join(entry.file_name());
        let kind = entry.file_type().map_err(InstallError::Io)?;
        if kind.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target).map_err(InstallError::Io)?;
        }
    }
    Ok(())
}

/// Extract into staging (zip-slip check: reject if a member path escapes the
/// extraction directory).
pub(super) fn extract_archive(
    raw: &[u8],
    format: ArchiveFormat,
    staging: &Path,
) -> Result<(), InstallError> {
    match format {
        ArchiveFormat::Zip => extract_zip(raw, staging),
        ArchiveFormat::TarGz => extract_tar_gz(raw, staging),
    }
}

pub(super) fn extract_zip(raw: &[u8], staging: &Path) -> Result<(), InstallError> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(raw))
        .map_err(|e| InstallError::Archive(e.to_string()))?;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| InstallError::Archive(e.to_string()))?;
        let Some(rel) = file.enclosed_name() else {
            return Err(InstallError::ZipSlip {
                entry: file.name().to_string(),
            });
        };
        let target = staging.join(&rel);
        if file.is_dir() {
            std::fs::create_dir_all(&target).map_err(InstallError::Io)?;
        } else {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(InstallError::Io)?;
            }
            let mut out = std::fs::File::create(&target).map_err(InstallError::Io)?;
            std::io::copy(&mut file, &mut out).map_err(InstallError::Io)?;
        }
    }
    Ok(())
}

pub(super) fn extract_tar_gz(raw: &[u8], staging: &Path) -> Result<(), InstallError> {
    let decoder = flate2::read::GzDecoder::new(raw);
    let mut archive = tar::Archive::new(decoder);
    for entry in archive
        .entries()
        .map_err(|e| InstallError::Archive(e.to_string()))?
    {
        let mut entry = entry.map_err(|e| InstallError::Archive(e.to_string()))?;
        let path = entry
            .path()
            .map_err(|e| InstallError::Archive(e.to_string()))?;
        // zip-slip: after lexical normalization it must still be under the
        // extraction directory.
        let Some(rel) = normalize_relative(&path) else {
            return Err(InstallError::ZipSlip {
                entry: path.display().to_string(),
            });
        };
        let target = staging.join(&rel);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&target).map_err(InstallError::Io)?;
        } else {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(InstallError::Io)?;
            }
            let mut out = std::fs::File::create(&target).map_err(InstallError::Io)?;
            std::io::copy(&mut entry, &mut out).map_err(InstallError::Io)?;
        }
    }
    Ok(())
}

/// Lexical normalization of an archive member path: absolute path / `..`
/// escape → `None` (zip-slip).
pub(super) fn normalize_relative(path: &Path) -> Option<PathBuf> {
    use std::path::Component;

    if path.is_absolute() {
        return None;
    }
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => return None, // reject ParentDir / RootDir / Prefix outright
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}
