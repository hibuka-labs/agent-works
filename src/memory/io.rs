use std::fs;
use std::path::Path;

use agent_base::AgentResult;

// ─────────────────────────────────────────────────────────────────────────────
// Timestamps (ISO 8601 without a chrono dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Current UTC time as ISO 8601 with milliseconds, e.g. `2026-09-13T08:15:30.123Z`.
pub fn now_iso8601() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs() as i64;
    let millis = d.subsec_millis();
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (y, m, dd) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{dd:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60
    )
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 → (y, m, d).
pub(super) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic write
// ─────────────────────────────────────────────────────────────────────────────

/// Atomic file write: write to a unique sibling `.tmp` file, then `fs::rename`
/// over the target. A crash mid-write leaves the tmp remnant behind (dot-
/// prefixed, never scanned as a memory) but the target file always either has
/// its old content or its full new content — never a torn mix.
///
/// The tmp name is unique per call (monotonic counter): two threads writing
/// the same target concurrently must not share a tmp path, or one thread's
/// rename would steal the other's tmp file mid-flight.
pub(crate) fn atomic_write(path: &Path, content: &str) -> AgentResult<()> {
    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "memory-index".to_string());
    let tmp = path.with_file_name(format!(".{file_name}.{unique}.tmp"));
    fs::write(&tmp, content)?;
    fs::rename(&tmp, path)?;
    Ok(())
}
