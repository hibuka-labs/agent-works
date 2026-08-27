/// Default guard configuration
pub struct DefaultGuardConfig {
    /// Maximum retries for reasoning-only responses
    pub reasoning_only_max_strikes: usize,
    /// Maximum retries for empty responses
    pub empty_response_max_strikes: usize,
    /// Nudge message for reasoning-only responses
    pub reasoning_only_nudge: String,
    /// Nudge message for empty responses
    pub empty_response_nudge: String,
    /// Whether to use LLM judge for text-only after tools
    pub use_llm_judge: bool,
    /// Timeout in seconds for the LLM judge call
    pub judge_timeout_secs: u64,
    /// Skip judge if response is longer than this (likely complete)
    pub judge_skip_threshold: usize,
    /// Whether to trust LLM when judge fails or times out.
    /// - true: fail-open (trust the model, end loop)
    /// - false: fail-closed (don't trust, force continue)
    pub judge_fail_open: bool,
    /// Enable short-response detection (merged from CompletionGateMiddleware).
    /// When the user input is long but the model response is very short,
    /// treat it as potentially incomplete and nudge/judge accordingly.
    pub detect_short_response: bool,
    /// Minimum user input character count to trigger short-response detection.
    pub short_response_min_input: usize,
    /// Maximum LLM output character count to be considered a short response.
    pub short_response_max_output: usize,
    /// Nudge message for short responses
    pub short_response_nudge: String,
    /// Number of recent user messages to include in the judge prompt.
    /// Helps the judge understand context like "继续" after a multi-turn discussion.
    pub recent_user_count: usize,
}

impl Default for DefaultGuardConfig {
    fn default() -> Self {
        Self {
            reasoning_only_max_strikes: 3,
            empty_response_max_strikes: 3,
            reasoning_only_nudge: "You produced internal reasoning but no tool call \
                and no final answer. Make a decision now: call a tool to make progress, \
                or write your final answer as plain text."
                .to_string(),
            empty_response_nudge: "Your response was empty. Please provide a response \
                with either a tool call or your final answer."
                .to_string(),
            use_llm_judge: true,
            judge_timeout_secs: 10,
            judge_skip_threshold: 256,
            judge_fail_open: false, // Default: don't trust LLM on judge failure
            detect_short_response: true,
            short_response_min_input: 128,
            short_response_max_output: 64,
            short_response_nudge: "Your response may be incomplete — \
                you may need to continue."
                .to_string(),
            recent_user_count: 5,
        }
    }
}
