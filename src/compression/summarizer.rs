//! LLM summarisation for context compression.
//!
//! Provides the [`summarize`] function that calls an LLM to produce a compact
//! handoff summary of an older conversation block.  The summary is designed for
//! *another* LLM to pick up seamlessly — it preserves the original goal, key
//! decisions, tool findings, and clear next steps.

use agent_base::{AgentResult, ChatMessage, StreamChunk, StreamClient};
use futures_util::stream::StreamExt;

/// Handoff-style summarisation prompt template.
///
/// Framed as a "context checkpoint compaction" — tells the LLM this is a
/// handoff to another model, not a self-summary.  Placeholders `{goal}`,
/// `{lang}`, `{max_chars}`, and `{transcript}` are filled by [`build_prompt`].
///
/// User messages are preserved verbatim (handled by the compactor), so the
/// summarizer only receives assistant and tool responses.  The prompt
/// explicitly instructs the LLM to describe *what happened* without
/// reproducing assistant text verbatim (poems, code, articles, etc.).
const SUMMARIZATION_PROMPT: &str = "\
You are performing a CONTEXT CHECKPOINT COMPACTION. \
Create a handoff summary for another LLM that will resume the task.

The original goal of this session was: {goal}

User messages have been preserved separately. \
Summarize ONLY the assistant responses and tool results below.

Include:
- What the assistant did (tools called, actions taken, results found)
- Key decisions made and important constraints discovered
- What remains to be done (clear next steps)
{lang}
Do NOT reproduce assistant text replies verbatim (poems, articles, code examples, etc.). \
Only describe what was done, not the content itself.

Be concise, structured, and focused on helping the next LLM seamlessly continue the work. \
Do not repeat work that has already been done. \
Output ONLY the summary text, no preamble, about {max_chars} characters max.

=== ASSISTANT AND TOOL RESPONSES ===
{transcript}";

/// Language instruction injected when CJK content is detected.
const LANG_INSTRUCTION_CJK: &str =
    "Respond in the same language as the conversation (CJK detected).";

/// No extra language instruction for predominantly Latin text — English is the
/// default and the prompt itself is English.
const LANG_INSTRUCTION_DEFAULT: &str = "";

// ── Public API ───────────────────────────────────────────────────────────────

/// Summarise a transcript block via the LLM.
///
/// * `client` — the LLM client to call.
/// * `transcript` — serialised older conversation (output of `serialize_block`).
/// * `original_goal` — first user message, truncated to avoid blowing the prompt.
/// * `max_chars` — target max length for the summary.
/// * `on_progress` — optional callback invoked with cumulative character count
///   as the LLM streams in. Useful for showing "generating summary... X chars".
///
/// Returns the summary text, truncated to `max_chars` if the LLM over-shoots.
pub async fn summarize(
    client: &dyn StreamClient,
    transcript: &str,
    original_goal: &str,
    max_chars: usize,
    on_progress: Option<&(dyn Fn(usize) + Sync)>,
) -> AgentResult<String> {
    if max_chars == 0 {
        return Ok(String::new());
    }

    let lang = language_instruction(&format!("{original_goal}\n{transcript}"));
    let prompt = build_prompt(original_goal, lang, max_chars, transcript);

    let system = ChatMessage::system(
        "You are a conversation summarizer for an AI agent that can call tools \
         (browser, shell, search, etc.).",
    );
    let user = ChatMessage::user(prompt);

    let mut stream = client.stream(&[system, user], &[], None, None).await?;
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk? {
            StreamChunk::Text(t) => {
                text.push_str(&t);
                if let Some(cb) = on_progress {
                    cb(text.len());
                }
            }
            StreamChunk::Stop { .. } => break,
            _ => {}
        }
    }

    Ok(truncate_summary_output(&text, max_chars))
}

// ── Prompt construction ──────────────────────────────────────────────────────

/// Build the summarisation prompt with all placeholders filled.
///
/// Uses single-pass character replacement to prevent placeholder pollution:
/// if `original_goal` contains `{lang}` etc. as literal text, it is preserved
/// verbatim (unlike a `.replace()` chain which would corrupt it).
fn build_prompt(goal: &str, lang: &str, max_chars: usize, transcript: &str) -> String {
    // Single-pass replacement: walk the template, emit literal chars or fill
    // placeholders.  Content of goal/transcript is never re-scanned.
    let mut out = String::with_capacity(SUMMARIZATION_PROMPT.len() + goal.len() + transcript.len());
    let mut chars = SUMMARIZATION_PROMPT.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            // Try to match a placeholder.
            let rest: String = chars.clone().take_while(|ch| *ch != '}').collect();
            match rest.as_str() {
                "goal" => {
                    out.push_str(goal);
                    // Skip past the closing '}'.
                    for _ in 0..=rest.len() {
                        chars.next();
                    }
                }
                "lang" => {
                    out.push_str(lang);
                    for _ in 0..=rest.len() {
                        chars.next();
                    }
                }
                "max_chars" => {
                    out.push_str(&max_chars.to_string());
                    for _ in 0..=rest.len() {
                        chars.next();
                    }
                }
                "transcript" => {
                    out.push_str(transcript);
                    for _ in 0..=rest.len() {
                        chars.next();
                    }
                }
                _ => out.push(c),
            }
        } else {
            out.push(c);
        }
    }
    out
}

// ── Language detection ───────────────────────────────────────────────────────

/// Detect the dominant script of a text and return the appropriate language
/// instruction for the summarisation prompt.
///
/// Returns `LANG_INSTRUCTION_CJK` when ≥ 20 % of non-whitespace characters
/// are CJK, `LANG_INSTRUCTION_DEFAULT` otherwise.
pub fn language_instruction(text: &str) -> &'static str {
    let meaningful: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if meaningful.is_empty() {
        return LANG_INSTRUCTION_DEFAULT;
    }
    let cjk_count = meaningful.iter().filter(|c| is_cjk(**c)).count();
    if cjk_count * 5 >= meaningful.len() {
        LANG_INSTRUCTION_CJK
    } else {
        LANG_INSTRUCTION_DEFAULT
    }
}

/// Returns `true` if `c` is a CJK ideograph, kana, hangul, or punctuation.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Unified Ideographs Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{3000}'..='\u{303F}' // CJK Symbols and Punctuation
        | '\u{FF00}'..='\u{FFEF}' // Fullwidth Forms
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
    )
}

// ── Output truncation ────────────────────────────────────────────────────────

/// Truncate a summary to `max_chars` (front 80 % + rear 20 %).
///
/// Preserves the beginning (which typically contains the most important
/// context) and the end (recent conclusions / next steps), dropping the
/// middle when the LLM over-shoots.
pub fn truncate_summary_output(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }
    // Reserve 1 char for the '…' separator.
    let budget = max_chars.saturating_sub(1);
    let front = (budget as f64 * 0.8) as usize;
    let rear = budget.saturating_sub(front);
    let front_s: String = text.chars().take(front).collect();
    let rear_s: String = text
        .chars()
        .rev()
        .take(rear)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{front_s}…{rear_s}")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::ResponseFormat;

    // ── Test helpers ──────────────────────────────────────────────────────

    /// Mock that captures the prompt sent to the LLM.
    struct PromptCapture {
        captured: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        response: String,
    }

    #[async_trait::async_trait]
    impl StreamClient for PromptCapture {
        async fn stream(
            &self,
            messages: &[ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<
            std::pin::Pin<
                Box<dyn futures_core::Stream<Item = AgentResult<agent_base::StreamChunk>> + Send>,
            >,
        > {
            // Capture the user message content (same as chat()).
            for msg in messages {
                if let ChatMessage::User { content, .. } = msg {
                    self.captured.lock().unwrap().push(content.clone());
                }
            }
            let response = self.response.clone();
            Ok(Box::pin(futures_util::stream::once(async move {
                Ok(agent_base::StreamChunk::Text(response))
            })))
        }

        async fn chat(
            &self,
            messages: &[ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&ResponseFormat>,
        ) -> AgentResult<String> {
            for msg in messages {
                if let ChatMessage::User { content, .. } = msg {
                    self.captured.lock().unwrap().push(content.clone());
                }
            }
            Ok(self.response.clone())
        }

        fn capabilities(&self) -> agent_base::LlmCapabilities {
            agent_base::LlmCapabilities::default()
        }
    }

    // ── language_instruction ──────────────────────────────────────────────

    #[test]
    fn test_language_instruction_cjk() {
        assert_eq!(
            language_instruction("用户问了关于日志分析的问题，发现了5次操作"),
            LANG_INSTRUCTION_CJK
        );
    }

    #[test]
    fn test_language_instruction_english() {
        assert_eq!(
            language_instruction("The user asked about log analysis, found 5 operations"),
            LANG_INSTRUCTION_DEFAULT
        );
    }

    #[test]
    fn test_language_instruction_mostly_latin_with_some_cjk() {
        assert_eq!(
            language_instruction("The user asked about 日志 analysis of the system"),
            LANG_INSTRUCTION_DEFAULT
        );
    }

    #[test]
    fn test_language_instruction_mixed_heavy_cjk() {
        assert_eq!(
            language_instruction("分析日志时发现 operations 有5次 user asked 分析"),
            LANG_INSTRUCTION_CJK
        );
    }

    #[test]
    fn test_language_instruction_empty() {
        assert_eq!(language_instruction(""), LANG_INSTRUCTION_DEFAULT);
    }

    #[test]
    fn test_language_instruction_whitespace_only() {
        assert_eq!(language_instruction("   \n\t  "), LANG_INSTRUCTION_DEFAULT);
    }

    #[test]
    fn test_language_instruction_hangul() {
        assert_eq!(
            language_instruction("사용자가 로그 분석에 대해 물었습니다"),
            LANG_INSTRUCTION_CJK
        );
    }

    // ── truncate_summary_output ───────────────────────────────────────────

    #[test]
    fn test_truncate_short_text() {
        assert_eq!(truncate_summary_output("short", 100), "short");
    }

    #[test]
    fn test_truncate_long_text_preserves_ends() {
        let text = "a".repeat(500) + "TAIL";
        let result = truncate_summary_output(&text, 100);
        assert!(result.chars().count() <= 100);
        assert!(result.starts_with('a'));
        assert!(result.contains("TAIL"));
        assert!(result.contains('…'));
    }

    #[test]
    fn test_truncate_exact_boundary() {
        let text = "x".repeat(100);
        assert_eq!(truncate_summary_output(&text, 100), text);
    }

    #[test]
    fn test_truncate_zero() {
        assert_eq!(truncate_summary_output("anything", 0), "");
    }

    // ── build_prompt ──────────────────────────────────────────────────────

    #[test]
    fn test_build_prompt_all_placeholders_filled() {
        let prompt = build_prompt("fix the bug", LANG_INSTRUCTION_CJK, 5000, "user: hello");
        assert!(prompt.contains("fix the bug"));
        assert!(prompt.contains("CJK detected"));
        assert!(prompt.contains("5000"));
        assert!(prompt.contains("user: hello"));
        // No unfilled placeholders.
        assert!(!prompt.contains("{goal}"));
        assert!(!prompt.contains("{lang}"));
        assert!(!prompt.contains("{transcript}"));
        assert!(!prompt.contains("{max_chars}"));
    }

    #[test]
    fn test_build_prompt_goal_with_placeholder_literals_not_polluted() {
        // goal contains literal {lang} and {max_chars} — must NOT be replaced.
        let goal = "按 {lang} 字段分组，max={max_chars}";
        let prompt = build_prompt(goal, LANG_INSTRUCTION_CJK, 5000, "data");
        assert!(
            prompt.contains("按 {lang} 字段分组，max={max_chars}"),
            "literal placeholders in goal must survive: {prompt}"
        );
        // The actual {lang} and {max_chars} placeholders should still be filled.
        assert!(prompt.contains("CJK detected"));
        assert!(prompt.contains("5000"));
    }

    #[test]
    fn test_build_prompt_transcript_with_goal_placeholder_not_polluted() {
        let transcript = "user: use {goal} as the key";
        let prompt = build_prompt("real goal", LANG_INSTRUCTION_DEFAULT, 1000, transcript);
        assert!(
            prompt.contains("use {goal} as the key"),
            "literal {{goal}} in transcript must survive: {prompt}"
        );
        assert!(prompt.contains("real goal"));
    }

    // ── summarize (mock-client integration) ───────────────────────────────

    #[tokio::test]
    async fn test_summarize_prompt_contains_goal_and_lang() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let client = std::sync::Arc::new(PromptCapture {
            captured: captured.clone(),
            response: "a summary".into(),
        });

        // CJK goal → should inject CJK instruction.
        let _ = summarize(
            client.as_ref(),
            "tool output here",
            "分析服务器日志中的延迟问题",
            5000,
            None,
        )
        .await
        .unwrap();

        let prompts = captured.lock().unwrap();
        assert_eq!(prompts.len(), 1);
        let prompt = &prompts[0];
        assert!(
            prompt.contains("分析服务器日志中的延迟问题"),
            "goal missing"
        );
        assert!(prompt.contains("CJK detected"), "lang instruction missing");
        assert!(prompt.contains("5000"), "max_chars missing");
        assert!(prompt.contains("tool output here"), "transcript missing");
    }

    #[tokio::test]
    async fn test_summarize_output_truncated() {
        let long_response = "x".repeat(2000);
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let client = std::sync::Arc::new(PromptCapture {
            captured: captured.clone(),
            response: long_response,
        });

        let result = summarize(client.as_ref(), "t", "g", 100, None)
            .await
            .unwrap();
        assert!(result.chars().count() <= 100);
    }

    #[tokio::test]
    async fn test_summarize_max_chars_zero() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let client = std::sync::Arc::new(PromptCapture {
            captured: captured.clone(),
            response: "ignored".into(),
        });

        let result = summarize(client.as_ref(), "t", "g", 0, None).await.unwrap();
        assert!(result.is_empty());
        // Should not even call the LLM.
        assert!(captured.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_summarize_returns_response_content() {
        // Test that summarize() returns the LLM's response content.
        let expected_summary =
            "User said hello and asked for a poem. Assistant provided a classical Chinese poem.";
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let client = std::sync::Arc::new(PromptCapture {
            captured: captured.clone(),
            response: expected_summary.into(),
        });

        let transcript = "[user] 你好\n[assistant] 你好！有什么我可以帮你的吗？\n[user] 来一首古诗";
        let result = summarize(client.as_ref(), transcript, "你好", 5000, None)
            .await
            .unwrap();

        // The result should be the mock response.
        assert_eq!(
            result, expected_summary,
            "summarize should return the LLM response"
        );

        // Verify the prompt was sent correctly.
        let prompts = captured.lock().unwrap();
        assert_eq!(prompts.len(), 1, "should have sent exactly one prompt");
        let prompt = &prompts[0];
        assert!(prompt.contains("你好"), "prompt should contain the goal");
        assert!(
            prompt.contains("来一首古诗"),
            "prompt should contain the transcript"
        );
        assert!(
            prompt.contains("CONTEXT CHECKPOINT COMPACTION"),
            "prompt should contain the compaction instruction"
        );
    }

    #[tokio::test]
    #[ignore] // Requires real API key: DEEPSEEK_API_KEY
    async fn test_summarize_with_real_deepseek_api() {
        // This test calls the real DeepSeek API to verify the summarization works.
        // It's skipped by default because it requires an API key.
        // Run with: cargo test test_summarize_with_real_deepseek_api -- --nocapture

        let api_key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            eprintln!("Skipping test: DEEPSEEK_API_KEY not set");
            return;
        }

        let base_url = std::env::var("DEEPSEEK_BASE_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com".to_string());

        // Create a real OpenAI client pointing to DeepSeek.
        let client =
            agent_base::OpenAiClient::new(api_key, "deepseek-chat".to_string(), Some(base_url));

        let transcript = "[user] 你好\n[assistant] 你好！有什么我可以帮你的吗？\n[user] 来一首古诗";
        let result = summarize(&client, transcript, "你好", 5000, None)
            .await
            .unwrap();

        eprintln!("=== DeepSeek API Response ===");
        eprintln!("{}", result);
        eprintln!("=============================");

        // The result should NOT start with "An alternate model reviewed".
        assert!(
            !result.starts_with("An alternate model reviewed"),
            "DeepSeek should follow the prompt, but got: {}",
            &result[..std::cmp::min(200, result.len())]
        );

        // The result should contain some summary content.
        assert!(
            result.len() > 10,
            "Summary should have meaningful content, got: {} chars",
            result.len()
        );
    }
}
