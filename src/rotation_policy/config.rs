//! Configuration and message builders for the token-budget window strategy.
//!
//! Pure data + pure functions: no I/O, no runtime. The window lifecycle
//! state machine lives in [`super::core`].

use agent_base::engine::estimate_messages_tokens;
use agent_base::types::ChatMessage;

/// Configuration for the token-budget window strategy.
///
/// **Work-room semantics**: `work_budget` is the conversation space ABOVE
/// the window's fixed base (system prompt + per-window boilerplate), not the
/// total window size. The reset point is `base + work_budget + buffer`, so
/// any positive budget produces a window that fits — the "budget below
/// system-prompt overhead" failure class is structurally impossible. Use
/// [`TokenBudgetConfig::with_work_budget`] to keep the reminder/buffer
/// thresholds proportional when overriding the default.
#[derive(Clone, Debug)]
pub struct TokenBudgetConfig {
    /// Work-room budget: tokens of conversation space above the fixed
    /// window base. Corresponds to "how much room the model has to work
    /// with" — the base is added by the framework.
    pub work_budget: usize,
    /// Reminder threshold: inject the reminder when remaining work room
    /// ≤ this value.
    pub reminder_threshold: usize,
    /// Reminder message template. `{n_remaining}` is replaced with estimated
    /// remaining work-room tokens.
    pub reminder_template: String,
    /// Fallback prompt injected when the work budget is exhausted (window
    /// about to reset).
    pub fallback_prompt: String,
    /// Extra work-room overshoot tolerated past `work_budget` before the
    /// reset fires. Allows the model one more turn to save state after
    /// the fallback.
    pub fallback_buffer: usize,
    /// Guidance message injected once per window start (explains the
    /// private context-management tools).
    pub guidance_message: String,
    /// User seed message injected as the LAST message of every new window.
    /// Satisfies the engine's compaction-output contract (at least one
    /// non-System message — System maps to the top-level `system` parameter,
    /// and an all-System window is an HTTP 400) and gives the model a turn
    /// to reconstruct state.
    pub seed_message: String,
    /// Minimum viable work budget as a multiple of the base overhead.
    /// Below this no single turn of work fits in the room (one tool result
    /// overflows it) and the session thrashes. Construction clamps the
    /// work budget up to `max(min_work_multiple × base,
    /// min_absolute_work_room)` and scales the reminder/buffer bands to
    /// match.
    pub min_work_multiple: f64,
    /// Absolute floor for the work room, independent of base size. Tool
    /// results don't shrink with the system prompt — an app with a tiny
    /// prompt still needs room for real tool output, so the viability floor
    /// is whichever is larger: the proportional bound or this constant.
    pub min_absolute_work_room: usize,
    /// Token cap for the mechanical user-message trail carried into a fresh
    /// window. User messages are the task trail — facts, not model output —
    /// so they are re-injected verbatim (newest-first under this cap, oldest
    /// truncated) instead of asking the model to recover them from history
    /// (cf. codex `build_compacted_history`).
    pub user_trail_max_tokens: usize,
}

impl Default for TokenBudgetConfig {
    fn default() -> Self {
        Self {
            // Target: ~256K context window × 90% ≈ 230K total.
            // work_budget = 230K − base_overhead − fallback_buffer.
            // With base_overhead ≈ 15-20K and buffer = 10%, work_budget ≈ 210K.
            work_budget: 210_000,
            reminder_threshold: 42_000,
            reminder_template: "You have approximately {n_remaining} tokens left in this context window. \
                If you have important state, decisions, or progress to record, save them to your notes now \
                using the notes tools. This window may close soon."
                .to_string(),
            fallback_prompt: "This context window is closing. Before it resets, write a handoff \
                summary to your notes — a note file named `thread_hint.md` is injected \
                automatically into the next window, so put the current task, key decisions, \
                and progress there. Your previous conversation history also remains \
                available via the history tools."
                .to_string(),
            fallback_buffer: 21_000,
            guidance_message: "You have access to two private tools for context management:\n\
                - `history`: read-only access to previous context windows (list, search, read)\n\
                - `notes`: read-write persistent scratchpad that survives window transitions\n\
                Use these silently to maintain continuity. Never disclose them to the user."
                .to_string(),
            seed_message: DEFAULT_SEED_MESSAGE.to_string(),
            min_work_multiple: 2.0,
            min_absolute_work_room: 4_096,
            user_trail_max_tokens: 2_000,
        }
    }
}

/// Default user seed message for new windows (see
/// [`TokenBudgetConfig::seed_message`]).
pub const DEFAULT_SEED_MESSAGE: &str = "Your previous context window was archived. The user \
    messages above are your task trail — the last one is your current task. \
    Check notes/thread_hint.md for any handoff from the previous window; \
    if missing, write one after you understand the task. \
    Previous conversation history is available via the history tools — consult them \
    only if you need context the trail doesn't provide. \
    Do NOT re-read files you already covered. Continue the task directly.";

impl TokenBudgetConfig {
    /// Config with proportional reminder/buffer thresholds for a custom
    /// work budget (reminder = 20%, buffer = 10% of `work_budget`).
    pub fn with_work_budget(work_budget: usize) -> Self {
        Self {
            work_budget,
            reminder_threshold: work_budget / 5,
            fallback_buffer: work_budget / 10,
            ..Default::default()
        }
    }

    /// Hard reset threshold in absolute estimated tokens:
    /// `base_overhead + work_budget + fallback_buffer`.
    pub fn hard_limit(&self, base_overhead: usize) -> usize {
        base_overhead + self.work_budget + self.fallback_buffer
    }
}

/// The window-info system message injected at the start of each window.
pub fn build_context_window_info(
    previous_window_id: usize,
    current_window_id: usize,
) -> ChatMessage {
    ChatMessage::system(format!(
        "<context_window>\n\
         Previous context window id: {:03}\n\
         Current context window id: {:03}\n\
         </context_window>",
        previous_window_id, current_window_id
    ))
}

/// Build the reminder developer message for a given remaining work-room
/// token count.
///
/// Uses an ephemeral system message so it's auto-cleaned after the turn and
/// does not pollute persisted conversation history.
pub fn build_reminder_message(config: &TokenBudgetConfig, remaining: usize) -> ChatMessage {
    let content = config
        .reminder_template
        .replace("{n_remaining}", &remaining.to_string());
    ChatMessage::system_ephemeral(content)
}

/// Build the fallback developer message (work budget exhausted, window
/// about to reset).
///
/// This is NOT ephemeral — it persists so the model can see it in the new
/// window's initial context if needed.
pub fn build_fallback_message(config: &TokenBudgetConfig) -> ChatMessage {
    ChatMessage::system(config.fallback_prompt.clone())
}

/// Token estimate of the fixed content every window starts with:
/// system prompt + window info + guidance + seed (no thread hint — the
/// best case; app-provided hints add on top).
///
/// Call ONCE at build time — the system prompt is process-constant, so the
/// result never changes for the lifetime of the agent. The estimate is a
/// char count (~4 chars/token Latin, ~1.5 CJK), not a tokenizer call.
pub fn token_budget_base_overhead(
    config: &TokenBudgetConfig,
    system_prompt: Option<&str>,
) -> usize {
    let mut base = Vec::new();
    if let Some(sp) = system_prompt {
        base.push(ChatMessage::system(sp.to_string()));
    }
    base.push(build_context_window_info(1, 2));
    base.push(ChatMessage::system(config.guidance_message.clone()));
    base.push(ChatMessage::user(config.seed_message.clone()));
    estimate_messages_tokens(&base)
}

/// State tracker for the token budget across compaction calls within a
/// single window.
///
/// Uses atomics for interior mutability (the
/// [`ContextCompaction::compact`](agent_base::ContextCompaction::compact)
/// method takes `&self`, not `&mut self`).
pub struct TokenBudgetState {
    /// Current window ID (1-indexed, incremented after each reset).
    pub window_id: std::sync::atomic::AtomicUsize,
    /// Whether a reminder has been sent in the current window.
    pub reminder_sent: std::sync::atomic::AtomicBool,
    /// Consecutive resets of windows that never had a quiet turn — feeds the
    /// futility brake (see `TokenBudgetCore::evaluate`).
    breathless_resets: std::sync::atomic::AtomicUsize,
    /// Whether the current window ever evaluated to `None` (i.e. completed
    /// real work instead of dying immediately).
    had_breath: std::sync::atomic::AtomicBool,
    /// Futility brake engaged: window rotation permanently paused (the room
    /// provably cannot hold one turn of work).
    braked: std::sync::atomic::AtomicBool,
}

impl TokenBudgetState {
    pub fn new() -> Self {
        Self {
            window_id: std::sync::atomic::AtomicUsize::new(1),
            reminder_sent: std::sync::atomic::AtomicBool::new(false),
            breathless_resets: std::sync::atomic::AtomicUsize::new(0),
            had_breath: std::sync::atomic::AtomicBool::new(false),
            braked: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Advance to the next window and reset per-window state. Returns the
    /// new window ID.
    pub fn advance_window(&self) -> usize {
        self.reminder_sent
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.had_breath
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.window_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    pub fn current_window_id(&self) -> usize {
        self.window_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn has_sent_reminder(&self) -> bool {
        self.reminder_sent
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn mark_reminder_sent(&self) {
        self.reminder_sent
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record that the current window had a quiet evaluation — real work
    /// fits in the room.
    pub fn mark_breath(&self) {
        self.had_breath
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn had_breath(&self) -> bool {
        self.had_breath.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Count a reset of a window that never had a quiet turn. Returns the
    /// consecutive breathless-reset count (1-indexed).
    pub fn record_breathless_reset(&self) -> usize {
        self.breathless_resets
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    /// A window that did real work was archived — the room is viable; clear
    /// the consecutive-breathless counter.
    pub fn clear_breathless(&self) {
        self.breathless_resets
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn engage_brake(&self) {
        self.braked
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn braked(&self) -> bool {
        self.braked.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for TokenBudgetState {
    fn default() -> Self {
        Self::new()
    }
}
