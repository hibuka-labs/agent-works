//! The token-budget decision state machine (pure, no I/O).
//!
//! [`TokenBudgetCore`] owns the phase logic (reminder → fallback → reset),
//! the fixed base overhead (computed once at construction), and the
//! per-window [`TokenBudgetState`]. The application's `ContextCompaction`
//! shell performs I/O at the [`TokenBudgetAction::Reset`] point: archive the
//! old window, pull any app-specific seed context (e.g. notes), call
//! [`TokenBudgetCore::build_reset_messages`], then
//! [`TokenBudgetCore::commit_reset`].

use super::config::{
    TokenBudgetConfig, TokenBudgetState, build_context_window_info, build_fallback_message,
    build_reminder_message, token_budget_base_overhead,
};
use agent_base::engine::{estimate_messages_tokens, first_system_prompt};
use agent_base::types::ChatMessage;

/// Consecutive windows that die before their first quiet evaluation before
/// the futility brake pauses rotation (see [`TokenBudgetCore::evaluate`]).
const FUTILITY_RESET_LIMIT: usize = 3;

/// What the shell should do for this compaction check.
#[derive(Debug)]
pub enum TokenBudgetAction {
    /// Below the reminder band — nothing to do.
    None,
    /// Inject this ephemeral reminder (model should save state to notes).
    Reminder(ChatMessage),
    /// Inject this fallback prompt (last turn before the reset).
    Fallback(ChatMessage),
    /// Work budget + buffer exhausted — archive the current window, then
    /// install a fresh one.
    Reset {
        /// Window being archived.
        previous_window: usize,
        /// Window that becomes current after
        /// [`TokenBudgetCore::commit_reset`].
        new_window: usize,
    },
}

/// Pure decision core for the token-budget window strategy.
pub struct TokenBudgetCore {
    config: TokenBudgetConfig,
    /// Fixed per-window overhead in estimated tokens. Computed ONCE here —
    /// never per turn (the system prompt is process-constant).
    base_overhead: usize,
    state: TokenBudgetState,
}

impl TokenBudgetCore {
    /// Build the core. `system_prompt` is the composed prompt (constant for
    /// the process lifetime); the base overhead is estimated from it once,
    /// here.
    ///
    /// A work budget below the viability floor
    /// (`max(min_work_multiple × base, min_absolute_work_room)`) cannot
    /// hold even one turn of work — a single tool result overflows the
    /// room, every window dies to its first reconstruction step, and the
    /// session thrashes (reset → seed → reconstruct → reset). The budget is
    /// clamped up to the floor (with the reminder/buffer bands rescaled)
    /// rather than proceeding into a known-broken configuration. The
    /// absolute term protects small-prompt apps: tool output doesn't
    /// shrink with the system prompt.
    pub fn new(config: TokenBudgetConfig, system_prompt: Option<&str>) -> Self {
        let base_overhead = token_budget_base_overhead(&config, system_prompt);
        let proportional = (config.min_work_multiple * base_overhead as f64).ceil() as usize;
        let floor = proportional.max(config.min_absolute_work_room);
        let effective = config.work_budget.max(floor);
        let clamped = effective != config.work_budget;
        if clamped {
            tracing::warn!(
                requested = config.work_budget,
                effective = effective,
                base_overhead,
                "work budget below the viability floor \
                 (max(min_work_multiple × base, min_absolute_work_room)); \
                 clamping to the floor"
            );
        }
        Self {
            config: TokenBudgetConfig {
                work_budget: effective.max(1),
                // Scale the bands with the (possibly clamped) room; clamp to
                // ≥1 so the reset branch can never form a fixed point
                // (work + buffer == 0 would make every fresh window
                // immediately exhausted again — the reset loop).
                fallback_buffer: if clamped {
                    effective / 10
                } else {
                    config.fallback_buffer
                }
                .max(1),
                reminder_threshold: if clamped {
                    effective / 5
                } else {
                    config.reminder_threshold
                }
                .max(1),
                ..config
            },
            base_overhead,
            state: TokenBudgetState::new(),
        }
    }

    /// Fixed per-window overhead (estimated tokens).
    pub fn base_overhead(&self) -> usize {
        self.base_overhead
    }

    /// Absolute reset threshold (base + work + buffer).
    pub fn hard_limit(&self) -> usize {
        self.config.hard_limit(self.base_overhead)
    }

    /// Per-tool-result soft cap: a single tool result may consume at most
    /// a third of the work room, so a fresh window always holds room for
    /// 2-3 results plus the model's own turn (session 20260909_e7053736:
    /// untruncated 3-4k reads at a 6.5k room left space for exactly one).
    /// Floored at 512 so tiny rooms still get usable output; the engine's
    /// `max_message_tokens` valve stays the hard upper bound on top of
    /// this.
    pub fn max_result_tokens(&self) -> usize {
        (self.config.work_budget / 3).max(512)
    }

    /// Current window ID.
    pub fn window_id(&self) -> usize {
        self.state.current_window_id()
    }

    /// Whether the futility brake has paused window rotation.
    pub fn braked(&self) -> bool {
        self.state.braked()
    }

    pub fn config(&self) -> &TokenBudgetConfig {
        &self.config
    }

    pub fn state(&self) -> &TokenBudgetState {
        &self.state
    }

    /// Work-room tokens consumed so far (`total − base`).
    pub fn work_used(&self, total_tokens: usize) -> usize {
        total_tokens.saturating_sub(self.base_overhead)
    }

    /// Work-room tokens remaining.
    pub fn work_remaining(&self, total_tokens: usize) -> usize {
        self.config
            .work_budget
            .saturating_sub(self.work_used(total_tokens))
    }

    /// Run one phase check. Marks the reminder flag when Reminder/Fallback
    /// fires (once per window); the shell injects the returned message at
    /// the END of the list — right after the latest tool result, where the
    /// model's attention actually is (a front-positioned nudge is lost in
    /// the middle and ignored).
    ///
    /// One-turn reset hold: a window that crosses every band in a single
    /// jump still gets exactly one Fallback turn ("save state now") before
    /// the Reset — the handoff note is what makes the next window cheap.
    ///
    /// Futility brake: when consecutive windows die before their first
    /// quiet evaluation (`None`), the room evidently cannot hold even one
    /// turn of work — the tool results alone overflow it. Resetting again
    /// would just burn API calls and shred history (session
    /// 20260908_8b5fcb45: 41 resets in 3 minutes), so after
    /// [`FUTILITY_RESET_LIMIT`] breathless resets the rotation pauses
    /// permanently and evaluate returns [`TokenBudgetAction::None`].
    pub fn evaluate(&self, total_tokens: usize) -> TokenBudgetAction {
        if self.state.braked() {
            return TokenBudgetAction::None;
        }

        let used = self.work_used(total_tokens);

        // Phase 3: work budget + buffer exhausted → full window reset
        if used >= self.config.work_budget + self.config.fallback_buffer {
            // Guarantee one save-state turn: a window that crossed every
            // band in a single jump (one big tool result / long answer)
            // would otherwise rotate with no nudge at all — the model never
            // saves notes, every new window re-reconstructs from scratch
            // (session 20260908_33f6a029). Hold the reset for one turn and
            // ask for the handoff first.
            if !self.state.has_sent_reminder() {
                self.state.mark_reminder_sent();
                return TokenBudgetAction::Fallback(build_fallback_message(&self.config));
            }
            if self.state.had_breath() {
                // The room held real work before filling — it's viable.
                self.state.clear_breathless();
            } else if self.state.record_breathless_reset() >= FUTILITY_RESET_LIMIT {
                self.state.engage_brake();
                tracing::error!(
                    breathless_resets = FUTILITY_RESET_LIMIT,
                    work_budget = self.config.work_budget,
                    base_overhead = self.base_overhead,
                    "futility brake engaged: consecutive windows died before \
                     completing a turn — the work budget cannot hold one turn \
                     of work; window rotation is PAUSED. Raise the budget."
                );
                return TokenBudgetAction::None;
            }
            let previous_window = self.state.current_window_id();
            return TokenBudgetAction::Reset {
                previous_window,
                new_window: previous_window + 1,
            };
        }

        if self.state.has_sent_reminder() {
            self.state.mark_breath();
            return TokenBudgetAction::None;
        }

        // Phase 2: budget exhausted but within the buffer → fallback. One
        // last turn for the model to save state before the reset.
        if used >= self.config.work_budget {
            self.state.mark_reminder_sent();
            return TokenBudgetAction::Fallback(build_fallback_message(&self.config));
        }

        // Phase 1: approaching the budget → reminder (once per window).
        if self.work_remaining(total_tokens) <= self.config.reminder_threshold {
            self.state.mark_reminder_sent();
            let remaining = self.work_remaining(total_tokens);
            return TokenBudgetAction::Reminder(build_reminder_message(
                &self.config,
                remaining,
            ));
        }

        self.state.mark_breath();
        TokenBudgetAction::None
    }

    /// Assemble the fresh window's message list: preserved system prompt +
    /// window info + optional app-provided thread hint + guidance + seed.
    ///
    /// A thread hint large enough to push the fresh window past the hard
    /// limit is dropped (with a warning) — the base must always fit.
    /// Does NOT advance window state; call [`TokenBudgetCore::commit_reset`]
    /// after the shell has archived and installed the list.
    pub fn build_reset_messages(
        &self,
        old_messages: &[ChatMessage],
        thread_hint: Option<String>,
        previous_window: usize,
    ) -> Vec<ChatMessage> {
        let has_hint = thread_hint.is_some();
        let mut new_messages = self.assemble_window(old_messages, thread_hint, previous_window);

        // Defense in depth: the fresh window must fit under the hard limit.
        // With work-room semantics the base always does — unless an
        // app-provided hint blew past it. Drop the hint rather than reset
        // into another overflow.
        let total = estimate_messages_tokens(&new_messages);
        if total >= self.hard_limit() && has_hint {
            tracing::warn!(
                total,
                hard_limit = self.hard_limit(),
                "thread hint pushes the fresh window past the hard limit; \
                 dropping the hint"
            );
            new_messages = self.assemble_window(old_messages, None, previous_window);
        }

        new_messages
    }

    /// Advance the window state after a successful reset. Returns the new
    /// window ID.
    pub fn commit_reset(&self) -> usize {
        self.state.advance_window()
    }

    /// The fixed window skeleton: system prompt + window info + hint +
    /// user trail + guidance + seed.
    fn assemble_window(
        &self,
        old_messages: &[ChatMessage],
        thread_hint: Option<String>,
        previous_window: usize,
    ) -> Vec<ChatMessage> {
        let current = self.state.current_window_id();
        let mut messages = Vec::new();
        if let Some(system_prompt) = first_system_prompt(old_messages) {
            messages.push(system_prompt);
        }
        messages.push(build_context_window_info(previous_window, current));
        if let Some(hint) = thread_hint {
            messages.push(ChatMessage::system(hint));
        }
        messages.extend(self.collect_user_trail(old_messages));
        messages.push(ChatMessage::system(self.config.guidance_message.clone()));
        messages.push(ChatMessage::user(self.config.seed_message.clone()));
        messages
    }

    /// Mechanically carry the old window's LAST real user message into the
    /// fresh window (codex `build_compacted_history` style). Only the last
    /// one — older user messages are past tasks already answered; carrying
    /// all of them wastes the model's limited work room and leads to
    /// redundant file reads (session 20260908_cb460b2c: the model re-read
    /// the entire UI module every window because all three old user messages
    /// were present, with "看下 ui 模块" always last).
    fn collect_user_trail(&self, old_messages: &[ChatMessage]) -> Vec<ChatMessage> {
        for msg in old_messages.iter().rev() {
            if let ChatMessage::User { content, .. } = msg {
                if content == &self.config.seed_message {
                    continue;
                }
                let tokens = estimate_messages_tokens(std::slice::from_ref(msg));
                let carried = if tokens <= self.config.user_trail_max_tokens {
                    content.clone()
                } else {
                    let ratio = self.config.user_trail_max_tokens as f64 / tokens as f64;
                    let max_chars = ((content.chars().count() as f64) * ratio) as usize;
                    format!("{}…", content.chars().take(max_chars).collect::<String>())
                };
                return vec![ChatMessage::user(carried)];
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> TokenBudgetConfig {
        // Tiny work budget with minimal boilerplate so the base (~25 tokens
        // with the short guidance/seed) stays well under the reset point.
        // Band-machine tests need the exact budget to survive construction,
        // so both viability floors are opted out here.
        TokenBudgetConfig {
            work_budget: 100,
            reminder_threshold: 20,
            fallback_buffer: 10,
            guidance_message: "Use notes/history.".to_string(),
            seed_message: "Reconstruct state.".to_string(),
            min_work_multiple: 0.0,
            min_absolute_work_room: 0,
            ..Default::default()
        }
    }

    fn core_with_sys_prompt() -> TokenBudgetCore {
        TokenBudgetCore::new(small_config(), Some("You are a helpful assistant."))
    }

    #[test]
    fn base_overhead_computed_once() {
        let core = core_with_sys_prompt();
        // system prompt (~7) + window info (~20) + guidance (~5) + seed (~4)
        assert!(core.base_overhead() > 0);
        assert!(core.base_overhead() < 100);
        // Stable across instances — it's stored, not recomputed.
        assert_eq!(
            core.base_overhead(),
            TokenBudgetCore::new(small_config(), Some("You are a helpful assistant."))
                .base_overhead()
        );
    }

    #[test]
    fn hard_limit_is_base_plus_work_plus_buffer() {
        let core = core_with_sys_prompt();
        assert_eq!(core.hard_limit(), core.base_overhead() + 100 + 10);
    }

    #[test]
    fn below_threshold_no_action() {
        let core = core_with_sys_prompt();
        let total = core.base_overhead() + 50; // work used 50 < 80 (100-20)
        assert!(matches!(core.evaluate(total), TokenBudgetAction::None));
    }

    #[test]
    fn at_reminder_band_injects_reminder() {
        let core = core_with_sys_prompt();
        let total = core.base_overhead() + 85; // remaining 15 ≤ 20
        match core.evaluate(total) {
            TokenBudgetAction::Reminder(msg) => match msg {
                ChatMessage::System { content, ephemeral } => {
                    assert!(ephemeral);
                    assert!(content.contains("15"));
                }
                _ => panic!("expected System reminder"),
            },
            _ => panic!("expected Reminder"),
        }
    }

    #[test]
    fn reminder_only_once_per_window() {
        let core = core_with_sys_prompt();
        let total = core.base_overhead() + 85;
        assert!(matches!(core.evaluate(total), TokenBudgetAction::Reminder(_)));
        assert!(matches!(core.evaluate(total), TokenBudgetAction::None));
    }

    #[test]
    fn work_exhausted_injects_fallback() {
        let core = core_with_sys_prompt();
        let total = core.base_overhead() + 100; // used == work budget
        match core.evaluate(total) {
            TokenBudgetAction::Fallback(msg) => match msg {
                ChatMessage::System { content, ephemeral } => {
                    assert!(!ephemeral);
                    assert!(content.contains("closing"));
                    // The handoff loop: the fallback must name the file that
                    // the shell auto-injects into the next window.
                    assert!(content.contains("thread_hint.md"));
                }
                _ => panic!("expected System fallback"),
            },
            _ => panic!("expected Fallback"),
        }
    }

    #[test]
    fn buffer_exhausted_gets_fallback_then_reset() {
        let core = core_with_sys_prompt();
        let total = core.base_overhead() + 111; // used 111 ≥ 100 + 10
        // One-turn hold: the first crossing asks for the handoff...
        assert!(matches!(core.evaluate(total), TokenBudgetAction::Fallback(_)));
        // ...the next check rotates.
        match core.evaluate(total) {
            TokenBudgetAction::Reset {
                previous_window,
                new_window,
            } => {
                assert_eq!(previous_window, 1);
                assert_eq!(new_window, 2);
            }
            _ => panic!("expected Reset"),
        }
    }

    #[test]
    fn reminder_already_sent_resets_immediately() {
        let core = core_with_sys_prompt();
        // A window that already had its nudge (reminder band) skips the
        // hold: reminder → reset without a second Fallback.
        let band = core.base_overhead() + 85; // remaining 15 ≤ 20
        assert!(matches!(core.evaluate(band), TokenBudgetAction::Reminder(_)));
        let total = core.base_overhead() + 111;
        assert!(matches!(core.evaluate(total), TokenBudgetAction::Reset { .. }));
    }

    #[test]
    fn zero_budget_clamped_never_loops() {
        // work=0 buffer=0 would make every fresh window immediately
        // exhausted — the clamp must keep the reset branch honest.
        let config = TokenBudgetConfig {
            work_budget: 0,
            fallback_buffer: 0,
            ..small_config()
        };
        let core = TokenBudgetCore::new(config, Some("sys"));
        // Fresh window: used 0 vs clamped work(1)+buffer(1) → no reset.
        assert!(!matches!(
            core.evaluate(core.base_overhead()),
            TokenBudgetAction::Reset { .. }
        ));
    }

    #[test]
    fn reset_messages_shape_and_seed() {
        let core = core_with_sys_prompt();
        let old = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        let msgs = core.build_reset_messages(&old, None, 1);
        // sys + window info + user trail [user("hi")] + guidance + seed
        assert_eq!(msgs.len(), 5);
        assert!(matches!(&msgs[0], ChatMessage::System { content, ephemeral: false }
            if content.contains("helpful assistant")));
        assert!(matches!(&msgs[1], ChatMessage::System { content, .. }
            if content.contains("context_window")));
        // User trail preserves the real user message.
        assert!(matches!(&msgs[2], ChatMessage::User { content, .. }
            if content == "hi"));
        assert!(matches!(&msgs.last().unwrap(), ChatMessage::User { .. }));
    }

    #[test]
    fn reset_messages_include_thread_hint() {
        let core = core_with_sys_prompt();
        let old = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
        let msgs = core.build_reset_messages(&old, Some("<thread_hint>note</thread_hint>".into()), 1);
        // sys + window info + thread_hint + user trail [user("hi")] + guidance + seed
        assert_eq!(msgs.len(), 6);
        assert!(matches!(&msgs[2], ChatMessage::System { content, .. }
            if content.contains("thread_hint")));
    }

    #[test]
    fn oversized_thread_hint_dropped() {
        // work budget 100 → hard limit ≈ base + 110. A hint of ~1000 tokens
        // pushes the fresh window past it and must be dropped.
        let core = core_with_sys_prompt();
        let old = vec![ChatMessage::system("sys"), ChatMessage::user("hi")];
        let hint = format!("<thread_hint>{}</thread_hint>", "x".repeat(4000));
        let msgs = core.build_reset_messages(&old, Some(hint), 1);
        assert_eq!(msgs.len(), 5); // hint dropped, trail kept
        assert!(estimate_messages_tokens(&msgs) < core.hard_limit());
    }

    #[test]
    fn user_trail_excludes_seed_messages() {
        let core = core_with_sys_prompt();
        let old = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("Reconstruct state."), // this IS the seed
            ChatMessage::user("real task"),
        ];
        let trail = core.collect_user_trail(&old);
        assert_eq!(trail.len(), 1);
        assert!(matches!(&trail[0], ChatMessage::User { content, .. }
            if content == "real task"));
    }

    #[test]
    fn user_trail_truncates_long_message() {
        let core = core_with_sys_prompt();
        let long_msg = "a".repeat(16_000); // ~4000 tokens, exceeds cap
        let old = vec![
            ChatMessage::system("sys"),
            ChatMessage::user(&long_msg),
        ];
        let trail = core.collect_user_trail(&old);
        assert_eq!(trail.len(), 1);
        match &trail[0] {
            ChatMessage::User { content, .. } => {
                assert!(content.len() < 16_000, "truncated: {}", content.len());
                assert!(content.ends_with("…"));
            }
            _ => panic!("expected truncated user message"),
        }
    }

    #[test]
    fn user_trail_carries_only_last_message() {
        let core = core_with_sys_prompt();
        let old = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("first task"),
            ChatMessage::assistant("done"),
            ChatMessage::user("current task"),
        ];
        let trail = core.collect_user_trail(&old);
        assert_eq!(trail.len(), 1);
        assert!(matches!(&trail[0], ChatMessage::User { content, .. }
            if content == "current task"));
    }

    #[test]
    fn commit_reset_advances_window_and_resets_reminder() {
        let core = core_with_sys_prompt();
        let total = core.base_overhead() + 85;
        assert!(matches!(core.evaluate(total), TokenBudgetAction::Reminder(_)));
        assert!(core.state().has_sent_reminder());

        assert_eq!(core.commit_reset(), 2);
        assert_eq!(core.window_id(), 2);
        assert!(!core.state().has_sent_reminder());
    }

    #[test]
    fn with_work_budget_scales_thresholds() {
        let c = TokenBudgetConfig::with_work_budget(10_000);
        assert_eq!(c.work_budget, 10_000);
        assert_eq!(c.reminder_threshold, 2_000);
        assert_eq!(c.fallback_buffer, 1_000);
    }

    #[test]
    fn default_config_values() {
        let c = TokenBudgetConfig::default();
        assert_eq!(c.work_budget, 96_000);
        assert_eq!(c.reminder_threshold, 19_200);
        assert_eq!(c.fallback_buffer, 9_600);
        assert_eq!(c.min_work_multiple, 2.0);
        assert!(!c.seed_message.is_empty());
    }

    #[test]
    fn work_budget_below_floor_is_clamped() {
        // A room below min_work_multiple × base cannot hold one turn of
        // work — construction clamps it up (scaling the bands) instead of
        // just warning.
        let config = TokenBudgetConfig::with_work_budget(1);
        let core =
            TokenBudgetCore::new(config, Some("a reasonably long system prompt for the floor"));
        let floor = (core.base_overhead() as f64 * 2.0).ceil() as usize;
        assert!(core.config().work_budget >= floor);
        assert_eq!(
            core.config().reminder_threshold,
            core.config().work_budget / 5
        );
        assert_eq!(
            core.config().fallback_buffer,
            core.config().work_budget / 10
        );
    }

    #[test]
    fn absolute_floor_protects_small_prompt_apps() {
        // With a tiny system prompt the proportional bound (2×base) is
        // smaller than the absolute work-room floor — tool output doesn't
        // shrink with the prompt, so the constant term must win.
        let config = TokenBudgetConfig::with_work_budget(1);
        let core = TokenBudgetCore::new(config, Some("hi"));
        assert!(core.base_overhead() * 2 < 4_096, "premise: proportional floor below the absolute one");
        assert_eq!(core.config().work_budget, 4_096);
        assert_eq!(core.max_result_tokens(), 4_096 / 3);
    }

    #[test]
    fn proportional_floor_wins_for_big_prompts() {
        // phimint's regime: base ~3264 → 2×base (6528) > 4096, so the
        // multiple stays the binding floor (matches session
        // 20260909_e7053736 numbers exactly).
        let prompt = "x".repeat(13_000);
        let config = TokenBudgetConfig::with_work_budget(1);
        let core = TokenBudgetCore::new(config, Some(&prompt));
        assert_eq!(core.config().work_budget, core.base_overhead() * 2);
        assert!(core.config().work_budget > 4_096);
    }

    #[test]
    fn max_result_tokens_scales_and_floors() {
        let core = TokenBudgetCore::new(TokenBudgetConfig::with_work_budget(96_000), None);
        assert_eq!(core.max_result_tokens(), 32_000);
        // Tiny (pre-clamp) rooms still get a usable cap, not a 0-token one.
        let mut tiny = TokenBudgetConfig::with_work_budget(600);
        tiny.min_absolute_work_room = 0;
        tiny.min_work_multiple = 0.0;
        let core = TokenBudgetCore::new(tiny, Some("hi"));
        assert_eq!(core.max_result_tokens(), 512);
    }

    /// Total that blows straight past work + buffer with no quiet turn.
    fn over_budget(core: &TokenBudgetCore) -> usize {
        core.base_overhead() + core.config().work_budget + core.config().fallback_buffer + 1
    }

    #[test]
    fn futility_brake_pauses_rotation_after_breathless_resets() {
        let core = core_with_sys_prompt();
        let total = over_budget(&core);

        // The first two breathless resets fire (the shell archives + commits).
        // Each window gets its one-turn Fallback hold first.
        for window in 1..=2 {
            assert!(
                matches!(core.evaluate(total), TokenBudgetAction::Fallback(_)),
                "window {window} hold should fire"
            );
            assert!(
                matches!(core.evaluate(total), TokenBudgetAction::Reset { .. }),
                "breathless reset {window} should fire"
            );
            core.commit_reset();
        }
        // The third breathless window engages the brake instead of resetting.
        assert!(matches!(core.evaluate(total), TokenBudgetAction::Fallback(_)));
        assert!(matches!(core.evaluate(total), TokenBudgetAction::None));
        assert!(core.braked());
        // From now on nothing resets, however large the conversation grows.
        assert!(matches!(
            core.evaluate(total + 10_000),
            TokenBudgetAction::None
        ));
        assert_eq!(core.window_id(), 3);
    }

    #[test]
    fn window_with_breath_clears_brake_counter() {
        let core = core_with_sys_prompt();
        let over = over_budget(&core);

        // One breathless reset (with its Fallback hold)...
        assert!(matches!(core.evaluate(over), TokenBudgetAction::Fallback(_)));
        assert!(matches!(core.evaluate(over), TokenBudgetAction::Reset { .. }));
        core.commit_reset();
        // ...then a window with a quiet turn (real work fits)...
        assert!(matches!(
            core.evaluate(core.base_overhead() + 1),
            TokenBudgetAction::None
        ));
        // ...then overflowing again must still reset — the counter was cleared.
        assert!(matches!(core.evaluate(over), TokenBudgetAction::Fallback(_)));
        assert!(matches!(core.evaluate(over), TokenBudgetAction::Reset { .. }));
        assert!(!core.braked());
    }
}
