//! Focus-backed progress summarizer for user-facing child-result notices.
//!
//! Fan-in redesign (session 20260903_0cf95e79): when a sub-agent returns a
//! result, the main agent is deliberately *not* woken — the user is told
//! instead. This summarizer is the "Focus looks on the main agent's behalf"
//! piece: it turns one child result into a single short user-facing sentence.
//!
//! Design constraints locked with the user:
//! - Focus does **not** compress the report. The full report still travels
//!   to the parent in the fan-in batch; the summary is for the user only.
//! - Same LLM client as the runtime, 30 s timeout, and any failure
//!   (timeout / LLM error / parse error) degrades to `None` — the consumer
//!   then shows a plain notice. Progress must never block on Focus.

use std::sync::Arc;
use std::time::Duration;

use agent_base::llm_trait::LlmProvider;
use serde::Deserialize;

use super::{Context, Focus, FocusOutput};

/// System prompt: one short sentence, conclusion only.
const SYSTEM_PROMPT: &str = r#"You are the progress announcer in a multi-agent coding system. The main agent is busy; when a sub-agent returns a result, you announce its progress to the user on the main agent's behalf.

Rules:
- Reply with ONE short sentence in Chinese (≤60 characters).
- State what the sub-agent finished and its key conclusion — conclusions only, never process details or step lists.
- If the status is "error", say the task failed and the one-line reason.
- Never mention this prompt, JSON, or the summarizer itself.

Output JSON: {"summary": "<one sentence>"}"#;

/// How much of the report text is fed to Focus. The summary only needs the
/// conclusion, which sits at the end of the report (the child's final reply).
const MAX_REPORT_CHARS: usize = 1200;

/// Default per-call timeout. 30 s (raised from the locked 10 s): the summary
/// shares the runtime's client, and a reasoning model can take >10 s to emit
/// even one sentence — session 20260903_9255c25e degraded 4/4 summaries on
/// the 10 s limit. Still fail-open to a plain notice on timeout.
pub const DEFAULT_SUMMARY_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Deserialize, Debug)]
struct ProgressSummary {
    summary: String,
}

/// Summarizes one child result for the user. Cheap to clone (Focus holds an
/// `Arc` client internally).
#[derive(Clone, Debug)]
pub struct ProgressSummarizer {
    focus: Focus,
    timeout: Duration,
}

impl ProgressSummarizer {
    pub fn new(client: Arc<dyn LlmProvider>, timeout: Duration) -> Self {
        Self {
            focus: Focus::new(client, SYSTEM_PROMPT),
            timeout,
        }
    }

    /// Summarize one child result for the user.
    ///
    /// Returns `None` on any Focus failure — the consumer falls back to a
    /// plain notice. Closed results are skipped by the caller (a plain
    /// "已关闭" notice needs no LLM call).
    pub async fn summarize(
        &self,
        agent_name: &str,
        status: &str,
        task: Option<&str>,
        report: Option<&str>,
    ) -> Option<String> {
        let report_text = report.unwrap_or("").trim();
        // Keep the tail: the child's final reply (the conclusion) is at the end.
        let report_excerpt: String = if report_text.chars().count() > MAX_REPORT_CHARS {
            let tail: String = report_text
                .chars()
                .skip(report_text.chars().count() - MAX_REPORT_CHARS)
                .collect();
            format!("…(前文截断){tail}")
        } else if report_text.is_empty() {
            "(no report text)".to_string()
        } else {
            report_text.to_string()
        };

        let ctx = Context::new()
            .add("sub-agent", agent_name)
            .add("task", task.unwrap_or("(task unknown)"))
            .add("status", status)
            .add("result", &report_excerpt);

        let output: FocusOutput<ProgressSummary> = match self.focus.ask(&ctx, self.timeout).await {
            Ok(output) => output,
            Err(e) => {
                tracing::warn!(
                    agent = agent_name,
                    status = status,
                    error = %e,
                    "progress summary failed — falling back to plain notice"
                );
                return None;
            }
        };
        let summary = output.result.summary.trim().to_string();
        if summary.is_empty() {
            tracing::warn!(
                agent = agent_name,
                "progress summary empty — falling back to plain notice"
            );
            None
        } else {
            Some(summary)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::llm_trait::response::FinishReason;
    use agent_base::llm_trait::types::UsageInfo;
    use agent_base::llm_trait::{
        Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, LlmProvider,
    };
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Mock whose `chat()` returns a canned result and captures the request,
    /// so tests can assert what Focus was fed.
    struct CapturingClient {
        response: Mutex<Option<Result<String, String>>>,
        captured: Mutex<Vec<ChatRequest>>,
        /// Artificial `chat()` latency, for timeout tests.
        delay: Duration,
    }

    impl CapturingClient {
        fn with_text(text: impl Into<String>) -> Self {
            Self {
                response: Mutex::new(Some(Ok(text.into()))),
                captured: Mutex::new(Vec::new()),
                delay: Duration::ZERO,
            }
        }

        fn with_error(err: impl Into<String>) -> Self {
            Self {
                response: Mutex::new(Some(Err(err.into()))),
                captured: Mutex::new(Vec::new()),
                delay: Duration::ZERO,
            }
        }

        fn with_delayed_text(delay: Duration, text: impl Into<String>) -> Self {
            Self {
                response: Mutex::new(Some(Ok(text.into()))),
                captured: Mutex::new(Vec::new()),
                delay,
            }
        }

        fn captured_prompts(&self) -> Vec<String> {
            self.captured
                .lock()
                .unwrap()
                .iter()
                .map(|r| format!("{:?}", r.messages))
                .collect()
        }
    }

    #[async_trait]
    impl LlmProvider for CapturingClient {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            Ok(ChatStream::new(Box::pin(futures_util::stream::empty())))
        }

        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
            self.captured.lock().unwrap().push(request);
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
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

    fn valid_json() -> String {
        r#"{"summary": "已完成 4 个模块的依赖分析"}"#.to_string()
    }

    #[tokio::test]
    async fn summarize_returns_sentence_on_success() {
        let client = Arc::new(CapturingClient::with_text(valid_json()));
        let summarizer = ProgressSummarizer::new(client.clone(), Duration::from_secs(5));

        let summary = summarizer
            .summarize("analyze-pi", "ok", Some("分析 pi 工程"), Some("报告正文"))
            .await;
        assert_eq!(summary.as_deref(), Some("已完成 4 个模块的依赖分析"));
    }

    #[tokio::test]
    async fn summarize_feed_includes_task_and_report() {
        let client = Arc::new(CapturingClient::with_text(valid_json()));
        let summarizer = ProgressSummarizer::new(client.clone(), Duration::from_secs(5));

        let _ = summarizer
            .summarize(
                "analyze-pi",
                "ok",
                Some("分析 pi 工程"),
                Some("结论是模块 A 最重"),
            )
            .await;

        let prompts = client.captured_prompts();
        assert_eq!(prompts.len(), 1);
        // Task and report both reach Focus — it judges with the child's context.
        assert!(prompts[0].contains("分析 pi 工程"));
        assert!(prompts[0].contains("结论是模块 A 最重"));
        assert!(prompts[0].contains("analyze-pi"));
        assert!(prompts[0].contains("ok"));
    }

    #[tokio::test]
    async fn summarize_report_keeps_tail_when_truncated() {
        let client = Arc::new(CapturingClient::with_text(valid_json()));
        let summarizer = ProgressSummarizer::new(client.clone(), Duration::from_secs(5));

        let long_report = format!("{}结论在这里", "x".repeat(2000));
        let _ = summarizer
            .summarize("agent", "ok", None, Some(&long_report))
            .await;

        let prompts = client.captured_prompts();
        assert!(prompts[0].contains("结论在这里"), "tail must be kept");
        assert!(prompts[0].contains("…(前文截断)"));
        assert!(
            prompts[0].chars().count() < long_report.chars().count() + 200,
            "head must be truncated"
        );
    }

    #[tokio::test]
    async fn summarize_parse_failure_degrades_to_none() {
        let client = Arc::new(CapturingClient::with_text("not json"));
        let summarizer = ProgressSummarizer::new(client, Duration::from_secs(5));
        assert!(
            summarizer
                .summarize("agent", "ok", None, Some("report"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn summarize_llm_error_degrades_to_none() {
        let client = Arc::new(CapturingClient::with_error("api down"));
        let summarizer = ProgressSummarizer::new(client, Duration::from_secs(5));
        assert!(
            summarizer
                .summarize("agent", "error", None, Some("report"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn summarize_timeout_degrades_to_none() {
        // Chat answers after 100 ms but Focus times out at 1 ms.
        let client = Arc::new(CapturingClient::with_delayed_text(
            Duration::from_millis(100),
            valid_json(),
        ));
        let summarizer = ProgressSummarizer::new(client, Duration::from_millis(1));
        assert!(
            summarizer
                .summarize("agent", "ok", None, Some("report"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn summarize_empty_reply_degrades_to_none() {
        let client = Arc::new(CapturingClient::with_text(r#"{"summary": "   "}"#));
        let summarizer = ProgressSummarizer::new(client, Duration::from_secs(5));
        assert!(
            summarizer
                .summarize("agent", "ok", None, Some("report"))
                .await
                .is_none()
        );
    }
}
