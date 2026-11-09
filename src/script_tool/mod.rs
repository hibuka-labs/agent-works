//! Script-tier adapter (P1) — the runtime law for global script tools
//! (`script.{name}`) and the plugin `script` form.
//!
//! Spec text: `phimint/notes/tool-socket-design.md`, sections
//! "global script tool entry / script runtime semantics":
//!
//! - **Absolute path pinning (anti-shadowing)**: `run[0]` must be an absolute
//!   path. The only exception is the interpreter whitelist
//!   (`powershell` / `pwsh` / `cmd`), in which case the **script file argument**
//!   must be an absolute path (powershell: after `-File`/`-f`; cmd: after `/c`).
//!   A relative path = running that other project's same-named script in that
//!   project; rejected at definition/install time.
//! - **Scripts inside plugins obey the same law**: package-relative paths are
//!   resolved to absolute paths at install time by [`pin_from_plugin_root`]
//!   and pinned (lexical normalization + escape from the plugin root → reject).
//! - **argv spawn, no shell**: `run` substitutes `${param}` per element; a
//!   missing or non-scalar parameter → tool error for that call (fail-closed);
//!   parameter values are not re-expanded.
//! - cwd = current project root (pinned at product-layer registration in
//!   [`ScriptToolEntry::cwd`]; the framework does not sniff paths);
//!   env = inherited + entry overrides.
//! - timeout defaults to 60s; on timeout the **whole process tree** is killed
//!   (POSIX process group + `killpg`; Windows `taskkill /T /F` +
//!   `CREATE_NO_WINDOW`).
//! - Output exceeding `max_output_chars` reuses the kernel-tools truncation
//!   convention (first 1/3 + last 2/3 + `...[output truncated]`; the convention
//!   is copied verbatim, no cross-crate code import).
//! - exit ≠ 0 → tool error (with a stderr summary). If the script file is gone
//!   at execution time → a "script is gone" error, **no silent file swap**.

/// Default timeout (milliseconds).
pub const DEFAULT_TIMEOUT_MS: u64 = 60_000;

/// Interpreter whitelist: the **only** exception where `run[0]` need not be an absolute path.
pub const INTERPRETER_WHITELIST: [&str; 3] = ["powershell", "pwsh", "cmd"];

/// Output truncation marker (same convention as phi-kernel-tools `local_shell::truncate_output`).
/// The elision marker of the output truncation convention (one vocabulary
/// across the family: kernel tools, script tools, code cells).
pub(crate) const TRUNCATION_MARKER: &str = "...[output truncated]";

mod exec;
mod law;
mod provider;
mod subst;
mod types;

pub use exec::execute_entry;
pub use law::{
    ScriptLawError, check_run_law, check_script_file_on_disk, pin_from_plugin_root,
    resolve_under_root, script_file_index, script_file_is_program,
};
pub use provider::{ScriptProvider, ScriptTool};
pub use subst::substitute_params;
pub use types::{ExposureSpec, ParamSpec, ParamType, ScriptToolEntry, parameters_envelope};

// The `code-exec` feature form consumes this re-export (code_act /
// code_cell); the default-form lint cannot see those consumers.
#[allow(unused_imports)]
pub(crate) use exec::truncate_output;

#[cfg(test)]
mod tests;
