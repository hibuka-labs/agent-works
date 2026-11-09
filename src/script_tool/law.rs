use std::path::{Component, Path, PathBuf};

use super::INTERPRETER_WHITELIST;

/// Path-law violation at definition/install time. Fail-closed: reject what is illegal, no compromises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptLawError {
    /// Tool name is invalid (character set / reserved word / too long).
    BadName { name: String },
    /// `run` is empty.
    EmptyRun,
    /// `run[0]` is not an absolute path and not in the interpreter whitelist — rejected to prevent shadowing.
    RelativeProgram { program: String },
    /// The interpreter lacks a script file argument (`-File`/`-f`/`/c` not found), so the script path cannot be pinned.
    InterpreterMissingScriptArg { interpreter: String },
    /// The interpreter's script file argument is not an absolute path.
    RelativeScriptArg { interpreter: String, arg: String },
    /// A path inside a plugin escapes the plugin root after resolution (`../` escape / absolute path outside).
    EscapedPluginRoot { path: String },
    /// The script file does not exist at install/mount time (or is not a regular file).
    ScriptFileMissing { path: String },
    /// The script body (the non-interpreter form's `run[0]`) has no executable bit.
    ScriptFileNotExecutable { path: String },
}

impl std::fmt::Display for ScriptLawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadName { name } => write!(f, "script tool name `{name}` is invalid"),
            Self::EmptyRun => write!(f, "`run` must not be empty"),
            Self::RelativeProgram { program } => write!(
                f,
                "`run[0]` must be an absolute path (got `{program}`); \
                 the only exception is the interpreter whitelist {INTERPRETER_WHITELIST:?} \
                 with an absolute script file argument"
            ),
            Self::InterpreterMissingScriptArg { interpreter } => write!(
                f,
                "interpreter `{interpreter}` needs a script file argument \
                 (`-File`/`-f` for powershell/pwsh, `/c` for cmd) so its absolute path can be pinned"
            ),
            Self::RelativeScriptArg { interpreter, arg } => write!(
                f,
                "script file argument `{arg}` of interpreter `{interpreter}` must be an absolute path"
            ),
            Self::EscapedPluginRoot { path } => {
                write!(f, "script path `{path}` escapes the plugin root")
            }
            Self::ScriptFileMissing { path } => write!(
                f,
                "script file `{path}` does not exist (must be a regular file)"
            ),
            Self::ScriptFileNotExecutable { path } => write!(
                f,
                "script file `{path}` is not executable (chmod +x) — it runs directly as `run[0]`"
            ),
        }
    }
}

impl std::error::Error for ScriptLawError {}

/// Interpreter whitelist check: takes the file stem of `run[0]` (lowercased).
fn interpreter_of(program: &str) -> Option<&'static str> {
    let stem = Path::new(program)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    INTERPRETER_WHITELIST
        .iter()
        .find(|name| **name == stem)
        .copied()
}

/// Index of the interpreter's "script file argument": powershell/pwsh → after `-File`/`-f`;
/// cmd → after `/c`. Not found (or no argument after the flag) = cannot pin → error.
/// Not an interpreter → `Ok(None)` (the script is `run[0]` itself).
fn interpreter_script_arg_index(run: &[String]) -> Result<Option<usize>, ScriptLawError> {
    let Some((program, args)) = run.split_first() else {
        return Err(ScriptLawError::EmptyRun);
    };
    let Some(interpreter) = interpreter_of(program) else {
        return Ok(None);
    };
    let flags: &[&str] = if interpreter == "cmd" {
        &["/c"]
    } else {
        &["-file", "-f"]
    };
    for (offset, arg) in args.iter().enumerate() {
        if flags.iter().any(|f| arg.eq_ignore_ascii_case(f)) {
            // absolute index of the flag itself within run
            let flag_index = offset + 1;
            return run.get(flag_index + 1).map(|_| Some(flag_index + 1)).ok_or(
                ScriptLawError::InterpreterMissingScriptArg {
                    interpreter: interpreter.to_string(),
                },
            );
        }
    }
    Err(ScriptLawError::InterpreterMissingScriptArg {
        interpreter: interpreter.to_string(),
    })
}

/// Index of the "script file" element: interpreter → its script argument position; otherwise 0.
pub fn script_file_index(run: &[String]) -> Result<usize, ScriptLawError> {
    if run.is_empty() {
        return Err(ScriptLawError::EmptyRun);
    }
    Ok(interpreter_script_arg_index(run)?.unwrap_or(0))
}

/// Path law (anti-shadowing): `run[0]` is an absolute path, or the interpreter whitelist + an absolute script file argument.
pub fn check_run_law(run: &[String]) -> Result<(), ScriptLawError> {
    let Some((program, _)) = run.split_first() else {
        return Err(ScriptLawError::EmptyRun);
    };
    match interpreter_script_arg_index(run)? {
        Some(index) => {
            let arg = &run[index];
            if Path::new(arg).is_absolute() {
                Ok(())
            } else {
                Err(ScriptLawError::RelativeScriptArg {
                    interpreter: program.clone(),
                    arg: arg.clone(),
                })
            }
        }
        None => {
            if Path::new(program).is_absolute() {
                Ok(())
            } else {
                Err(ScriptLawError::RelativeProgram {
                    program: program.clone(),
                })
            }
        }
    }
}

/// Whether the script body is `run[0]` itself (non-interpreter form). Interpreter form → `false`
/// (the script is read in, no executable bit needed). Invalid/empty errors out on the same criteria as [`check_run_law`].
pub fn script_file_is_program(run: &[String]) -> Result<bool, ScriptLawError> {
    Ok(interpreter_script_arg_index(run)?.is_none())
}

/// Script file check at install/mount time (fail-closed, the same law as a missing wasm file):
/// it must be a regular file; when the script body is executed directly it must also have the
/// executable bit on unix.
///
/// `path` is supplied by the caller: plugin install checks the **staging area** contents (bytes
/// unchanged after the rename), global mount checks the final path. Not checked at load time —
/// archived entries are left to self-check at execution time (see [`ScriptToolEntry::check_laws`]);
/// once the file comes back the tool is alive again.
pub fn check_script_file_on_disk(
    path: &Path,
    require_executable: bool,
) -> Result<(), ScriptLawError> {
    let missing = ScriptLawError::ScriptFileMissing {
        path: path.display().to_string(),
    };
    let meta = std::fs::metadata(path).map_err(|_| missing.clone())?;
    if !meta.is_file() {
        return Err(missing);
    }
    #[cfg(unix)]
    if require_executable {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o111 == 0 {
            return Err(ScriptLawError::ScriptFileNotExecutable {
                path: path.display().to_string(),
            });
        }
    }
    #[cfg(not(unix))]
    let _ = require_executable;
    Ok(())
}

/// Lexical normalization (does not touch the file system): consumes `.`, `..` steps back;
/// stepping back past the start = escape → `None`.
fn normalize_lexical(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    Some(out)
}

/// Resolve an in-plugin path (relative to the plugin root) to an absolute path and pin it.
///
/// - Relative path → resolved under `plugin_root`; absolute path → verified to be inside the plugin root;
/// - After lexical normalization it must still be under the plugin root (`../` escape → [`ScriptLawError::EscapedPluginRoot`]).
pub fn resolve_under_root(plugin_root: &Path, path: &str) -> Result<String, ScriptLawError> {
    let escape = |p: &str| ScriptLawError::EscapedPluginRoot {
        path: p.to_string(),
    };
    let root_norm = normalize_lexical(plugin_root).ok_or_else(|| escape(path))?;
    let joined = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        root_norm.join(path)
    };
    let normalized = normalize_lexical(&joined).ok_or_else(|| escape(path))?;
    if normalized.starts_with(&root_norm) {
        Ok(normalized.to_string_lossy().into_owned())
    } else {
        Err(escape(path))
    }
}

/// Plugin script pinning at install time: resolve the script file element to an absolute path
/// under the plugin root, then enforce the path law.
///
/// The same law as global scripts' "`run[0]` anti-shadowing" rule; plugins are not exempt.
pub fn pin_from_plugin_root(
    run: &[String],
    plugin_root: &Path,
) -> Result<Vec<String>, ScriptLawError> {
    let index = script_file_index(run)?;
    let mut pinned = run.to_vec();
    pinned[index] = resolve_under_root(plugin_root, &run[index])?;
    check_run_law(&pinned)?;
    Ok(pinned)
}
