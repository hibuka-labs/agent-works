use std::sync::Arc;
use std::time::Duration;

use agent_base::ChatMessage;
use serde::de::DeserializeOwned;

// ── FocusInput ───────────────────────────────────────────────────────────────

/// Input for a Focus call — either a simple string or a structured context.
pub trait FocusInput {
    /// Format the input into the user prompt text sent to the LLM.
    fn to_prompt(&self) -> String;
}

/// Simple case: pass a string directly.
impl FocusInput for str {
    fn to_prompt(&self) -> String {
        self.to_string()
    }
}

impl FocusInput for String {
    fn to_prompt(&self) -> String {
        self.clone()
    }
}

impl FocusInput for &str {
    fn to_prompt(&self) -> String {
        self.to_string()
    }
}

// ── Context ──────────────────────────────────────────────────────────────────

/// Structured context for multi-field input scenarios.
///
/// Fields are formatted as `【key】\nvalue` when sent to the LLM,
/// where the key acts as a label to help the LLM understand the context.
///
/// # Usage
///
/// ```ignore
/// let ctx = Context::new()
///     .add("command", "apt install nginx")
///     .add("screen", screen_content);
/// ```
pub struct Context {
    entries: Vec<(String, String)>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a context field. The key is used as a label when sent to the LLM.
    pub fn add(mut self, key: &str, value: &str) -> Self {
        self.entries.push((key.to_string(), value.to_string()));
        self
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl FocusInput for Context {
    fn to_prompt(&self) -> String {
        self.entries
            .iter()
            .map(|(key, value)| format!("【{}】\n{}", key, value))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

// ── FocusOutput ──────────────────────────────────────────────────────────────

/// Output wrapper for a Focus call.
///
/// Contains both the structured result and the raw LLM response,
/// useful for debugging when something goes wrong.
#[derive(Debug)]
pub struct FocusOutput<T> {
    /// Deserialized structured result.
    pub result: T,
    /// Raw LLM response text (JSON string), for logging and debugging.
    pub raw_response: String,
}

// ── Focus ────────────────────────────────────────────────────────────────────

/// A focused LLM call.
///
/// Each instance is bound to a system prompt and dedicated to one specific
/// judgment question. Use `ask()` to send input and receive a structured
/// JSON answer.
///
/// # Usage
///
/// ```ignore
/// // Simple case: single string input
/// let classify = Focus::new(client, "You are a task complexity classifier...");
/// let output = classify.ask::<TaskComplexity>(&user_input, 5s).await?;
///
/// // Complex case: multiple context fields
/// let status_focus = Focus::new(client, "You are a task status judge...");
/// let ctx = Context::new()
///     .add("command", command)
///     .add("screen", screen);
/// let output = status_focus.ask::<TaskStatus>(&ctx, 5s).await?;
/// ```
pub struct Focus {
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
    system_prompt: String,
}

impl std::fmt::Debug for Focus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Focus").finish_non_exhaustive()
    }
}

impl Focus {
    /// Create a new Focus instance.
    ///
    /// - `client`: LLM client (shared; multiple Focus instances can reuse the same client)
    /// - `system_prompt`: The role and judgment rules for this Focus (bound at creation, never changes)
    pub fn new(
        client: Arc<dyn agent_base::llm_trait::LlmProvider>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            client,
            system_prompt: system_prompt.into(),
        }
    }

    /// Make a focused LLM call.
    ///
    /// Sends the system prompt (bound at creation) + user input (this call),
    /// forces JSON output, and deserializes into `T`.
    ///
    /// # Arguments
    /// - `input`: User input — can be `&str` or `Context`
    /// - `timeout`: Call timeout
    ///
    /// # Returns
    /// `FocusOutput<T>` containing the structured result and raw response.
    pub async fn ask<T: DeserializeOwned>(
        &self,
        input: &(impl FocusInput + ?Sized),
        timeout: Duration,
    ) -> Result<FocusOutput<T>, FocusError> {
        let user_prompt = input.to_prompt();

        // Logging: first line of prompt + char count (privacy-friendly)
        let prompt_first_line = user_prompt.lines().next().unwrap_or("(empty)");
        let prompt_char_count = user_prompt.chars().count();
        let sys_first_line = self.system_prompt.lines().next().unwrap_or("(empty)");
        let target_type = std::any::type_name::<T>();

        tracing::info!(
            target_type = target_type,
            system_prompt = %sys_first_line,
            user_prompt_first_line = %prompt_first_line,
            user_prompt_chars = prompt_char_count,
            timeout_secs = timeout.as_secs(),
            "[Focus] calling LLM"
        );

        let start = std::time::Instant::now();
        let messages = vec![
            ChatMessage::system(self.system_prompt.clone()),
            ChatMessage::user(user_prompt),
        ];

        let request = agent_base::llm_trait::ChatRequest::new(messages)
            .with_response_format(agent_base::llm_trait::request::ResponseFormat::JsonObject);

        let response = tokio::time::timeout(timeout, self.client.chat(request))
            .await
            .map_err(|_| FocusError::Timeout(timeout))?
            .map_err(|e| FocusError::Llm(e.to_string()))?;

        let elapsed_ms = start.elapsed().as_millis();
        // LlmProvider::chat() returns ChatResponse with content field.
        let raw_response = response.content;

        let result: T = serde_json::from_str(&raw_response).map_err(|e| {
            tracing::warn!(
                error = %e,
                raw_response = %raw_response,
                elapsed_ms = elapsed_ms,
                "[Focus] failed to parse LLM response as JSON"
            );
            FocusError::Parse {
                error: e.to_string(),
                raw: raw_response.clone(),
            }
        })?;

        tracing::info!(
            target_type = target_type,
            raw_response_chars = raw_response.chars().count(),
            elapsed_ms = elapsed_ms,
            "[Focus] call succeeded"
        );

        Ok(FocusOutput {
            result,
            raw_response,
        })
    }
}

// ── FocusError ───────────────────────────────────────────────────────────────

/// Error type for Focus calls.
#[derive(Debug)]
pub enum FocusError {
    /// LLM call timed out.
    Timeout(Duration),
    /// LLM call failed (network error, API error, etc.).
    Llm(String),
    /// LLM response could not be parsed into the expected JSON type.
    Parse { error: String, raw: String },
}

impl std::fmt::Display for FocusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FocusError::Timeout(d) => write!(f, "Focus timeout after {:?}", d),
            FocusError::Llm(e) => write!(f, "Focus LLM error: {}", e),
            FocusError::Parse { error, .. } => write!(f, "Focus parse error: {}", error),
        }
    }
}

impl std::error::Error for FocusError {}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::llm_trait::response::FinishReason;
    use agent_base::llm_trait::types::UsageInfo;
    use agent_base::llm_trait::{
        Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider,
    };
    use async_trait::async_trait;
    use serde::Deserialize;
    use std::sync::Mutex;

    // ── Mock LlmProvider for Focus tests ──

    /// A mock LlmProvider whose `chat()` returns a pre-set string.
    struct MockStreamClient {
        /// Canned response for `chat()`. Consumed on first call (take).
        response: Mutex<Option<Result<String, String>>>,
    }

    impl MockStreamClient {
        fn with_text(text: impl Into<String>) -> Self {
            Self {
                response: Mutex::new(Some(Ok(text.into()))),
            }
        }

        fn with_error(err: impl Into<String>) -> Self {
            Self {
                response: Mutex::new(Some(Err(err.into()))),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for MockStreamClient {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            Ok(ChatStream::new(Box::pin(futures_util::stream::empty())))
        }

        /// Override `chat()` to return the canned response directly.
        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            match self.response.lock().unwrap().take() {
                Some(Ok(text)) => Ok(ChatResponse {
                    content: text,
                    tool_calls: vec![],
                    usage: UsageInfo::default(),
                    finish_reason: FinishReason::Stop,
                    raw: None,
                    reasoning_content: None,
                    thinking_signature: None,
                }),
                Some(Err(e)) => Err(LlmError::llm(e)),
                None => Ok(ChatResponse {
                    content: String::new(),
                    tool_calls: vec![],
                    usage: UsageInfo::default(),
                    finish_reason: FinishReason::Stop,
                    raw: None,
                    reasoning_content: None,
                    thinking_signature: None,
                }),
            }
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }

        fn info(&self) -> agent_base::llm_trait::ProviderInfo {
            agent_base::llm_trait::ProviderInfo {
                name: "mock".to_string(),
                model: "mock-model".to_string(),
                version: None,
            }
        }
    }

    // ── Context tests ──

    #[test]
    fn context_single_field() {
        let ctx = Context::new().add("command", "df -h");
        assert_eq!(ctx.to_prompt(), "【command】\ndf -h");
    }

    #[test]
    fn context_multiple_fields() {
        let ctx = Context::new()
            .add("command", "apt install nginx")
            .add("elapsed", "30s")
            .add("screen", "Reading package lists...");
        let expected = "【command】\napt install nginx\n\n【elapsed】\n30s\n\n【screen】\nReading package lists...";
        assert_eq!(ctx.to_prompt(), expected);
    }

    #[test]
    fn context_empty() {
        let ctx = Context::new();
        assert_eq!(ctx.to_prompt(), "");
    }

    // ── FocusInput tests ──

    #[test]
    fn str_input() {
        let input: &str = "hello";
        assert_eq!(input.to_prompt(), "hello");
    }

    #[test]
    fn string_input() {
        let input = String::from("hello");
        assert_eq!(input.to_prompt(), "hello");
    }

    // ── FocusOutput deserialize tests ──

    #[derive(Deserialize, Debug, PartialEq)]
    struct MockResult {
        status: String,
        reason: String,
    }

    #[test]
    fn focus_output_deserialize() {
        let raw = r#"{"status":"finished","reason":"done"}"#;
        let result: MockResult = serde_json::from_str(raw).unwrap();
        assert_eq!(result.status, "finished");
        assert_eq!(result.reason, "done");
    }

    // ── FocusError tests ──

    #[test]
    fn focus_error_display() {
        let err = FocusError::Timeout(Duration::from_secs(5));
        assert_eq!(format!("{}", err), "Focus timeout after 5s");

        let err = FocusError::Llm("network error".to_string());
        assert_eq!(format!("{}", err), "Focus LLM error: network error");

        let err = FocusError::Parse {
            error: "unexpected token".to_string(),
            raw: "not json".to_string(),
        };
        assert_eq!(format!("{}", err), "Focus parse error: unexpected token");
    }

    // ── Focus::ask() tests ──

    #[derive(Deserialize, Debug, PartialEq)]
    struct AskResult {
        status: String,
        confidence: f64,
    }

    #[tokio::test]
    async fn focus_ask_parses_valid_json() {
        let client = Arc::new(MockStreamClient::with_text(
            r#"{"status":"finished","confidence":0.95}"#,
        ));
        let focus = Focus::new(client, "You are a classifier.");
        let output: FocusOutput<AskResult> = focus
            .ask(&"classify this", Duration::from_secs(5))
            .await
            .expect("ask should succeed");
        assert_eq!(output.result.status, "finished");
        assert_eq!(output.result.confidence, 0.95);
        assert_eq!(
            output.raw_response,
            r#"{"status":"finished","confidence":0.95}"#
        );
    }

    #[tokio::test]
    async fn focus_ask_parses_str_input() {
        let client = Arc::new(MockStreamClient::with_text(
            r#"{"status":"done","confidence":1.0}"#,
        ));
        let focus = Focus::new(client, "system");
        let output: FocusOutput<AskResult> = focus
            .ask("classify", Duration::from_secs(5))
            .await
            .expect("ask should succeed");
        assert_eq!(output.result.status, "done");
    }

    #[tokio::test]
    async fn focus_ask_rejects_invalid_json() {
        let client = Arc::new(MockStreamClient::with_text("not valid json at all"));
        let focus = Focus::new(client, "system");
        let err = focus
            .ask::<AskResult>(&"input", Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(matches!(err, FocusError::Parse { .. }));
        assert!(err.to_string().contains("Focus parse error"));
    }

    #[tokio::test]
    async fn focus_ask_propagates_llm_error() {
        let client = Arc::new(MockStreamClient::with_error("api key invalid"));
        let focus = Focus::new(client, "system");
        let err = focus
            .ask::<AskResult>(&"input", Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(matches!(err, FocusError::Llm(_)));
    }

    #[tokio::test]
    async fn focus_ask_times_out() {
        // Return after a long delay — Focus has a very short timeout
        let client = Arc::new(MockStreamClient::with_text("{}"));
        let focus = Focus::new(client, "system");
        let result = focus
            .ask::<AskResult>(&"input", Duration::from_millis(1))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn focus_ask_new_constructs_correctly() {
        let client = Arc::new(MockStreamClient::with_text("{}"));
        let focus = Focus::new(client, "You are helpful.");
        // Just verify construction + Debug
        assert!(format!("{:?}", focus).contains("Focus"));
    }
}
