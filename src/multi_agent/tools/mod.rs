//! Multi-agent LLM tools.
//!
//! Six tools that the LLM uses to manage sub-agents:
//! spawn, send_message, followup_task, wait, list, close.
//!
//! Each tool holds an `Arc<MultiAgentRuntime>` and delegates to its methods.

use std::sync::Arc;

use agent_base::{AgentResult, Tool, ToolContext, ToolControlFlow, TypedTool};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::runtime::MultiAgentRuntime;

// ---------------------------------------------------------------------------
// Helper: create tool instances
// ---------------------------------------------------------------------------

/// Create all 6 multi-agent tools, sharing the same runtime.
pub fn create_all_tools(runtime: Arc<MultiAgentRuntime>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(SpawnAgentTool::new(runtime.clone())),
        Arc::new(SendMessageTool::new(runtime.clone())),
        Arc::new(FollowupTaskTool::new(runtime.clone())),
        Arc::new(WaitAgentTool::new(runtime.clone())),
        Arc::new(ListAgentsTool::new(runtime.clone())),
        Arc::new(CloseAgentTool::new(runtime)),
    ]
}

// ---------------------------------------------------------------------------
// 1. spawn_agent
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SpawnAgentArgs {
    task_name: String,
    message: String,
    #[serde(default)]
    agent_type: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    fork_history: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpawnAgentOutput {
    agent_path: String,
    message: String,
}

pub struct SpawnAgentTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl SpawnAgentTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for SpawnAgentTool {
    type Args = SpawnAgentArgs;
    type Output = SpawnAgentOutput;

    fn name(&self) -> &'static str {
        "spawn_agent"
    }

    fn description(&self) -> &'static str {
        "Spawn a new sub-agent to execute a task independently.\n\
         The sub-agent runs concurrently and reports results via its mailbox.\n\
         Use agent_type for preset roles, or system_prompt for custom instructions."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task_name": {
                    "type": "string",
                    "description": "Unique name for this sub-agent (used in agent path)"
                },
                "message": {
                    "type": "string",
                    "description": "Initial task description for the sub-agent"
                },
                "agent_type": {
                    "type": "string",
                    "description": "Optional role type that maps to a preset configuration"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional custom system prompt (overrides agent_type)"
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override"
                },
                "reasoning_effort": {
                    "type": "string",
                    "description": "Optional reasoning effort (low/medium/high)"
                },
                "fork_history": {
                    "type": "string",
                    "description": "Optional history: 'none' (default), 'all', or a number N"
                }
            },
            "required": ["task_name", "message"]
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
    }

    async fn call_typed(
        &self,
        args: Self::Args,
        _ctx: &ToolContext,
    ) -> AgentResult<Self::Output> {
        let system_prompt = args
            .system_prompt
            .or(args
                .agent_type
                .map(|t| format!("You are a {} specialist.", t)))
            .unwrap_or_else(|| args.message.clone());

        let depth = 1;
        let tool_count = 0;

        match self
            .runtime
            .spawn_child(&args.task_name, system_prompt, depth, tool_count)
            .await
        {
            Ok(agent_path) => {
                let _ = self
                    .runtime
                    .send_task(&agent_path, args.message.clone(), true);
                Ok(SpawnAgentOutput {
                    agent_path,
                    message: "Agent spawned successfully".to_string(),
                })
            }
            Err(e) => Ok(SpawnAgentOutput {
                agent_path: String::new(),
                message: format!("Failed to spawn agent: {}", e),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// 2. send_message
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SendMessageArgs {
    agent_path: String,
    message: String,
}

#[derive(Debug, Serialize)]
pub struct SendMessageOutput {
    delivered: bool,
}

pub struct SendMessageTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl SendMessageTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for SendMessageTool {
    type Args = SendMessageArgs;
    type Output = SendMessageOutput;

    fn name(&self) -> &'static str {
        "send_message"
    }

    fn description(&self) -> &'static str {
        "Send a message to a sub-agent without triggering execution.\n\
         The message is queued and delivered with the next followup_task."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_path": {
                    "type": "string",
                    "description": "Target agent path (e.g., 'root/searcher')"
                },
                "message": {
                    "type": "string",
                    "description": "Message content to deliver"
                }
            },
            "required": ["agent_path", "message"]
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
    }

    async fn call_typed(
        &self,
        args: Self::Args,
        _ctx: &ToolContext,
    ) -> AgentResult<Self::Output> {
        let delivered = self
            .runtime
            .send_message(&args.agent_path, args.message)
            .unwrap_or(false);
        Ok(SendMessageOutput { delivered })
    }
}

// ---------------------------------------------------------------------------
// 3. followup_task
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct FollowupTaskArgs {
    agent_path: String,
    task: String,
    #[serde(default = "default_interrupt")]
    interrupt: bool,
}

fn default_interrupt() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct FollowupTaskOutput {
    accepted: bool,
    agent_path: String,
}

pub struct FollowupTaskTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl FollowupTaskTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for FollowupTaskTool {
    type Args = FollowupTaskArgs;
    type Output = FollowupTaskOutput;

    fn name(&self) -> &'static str {
        "followup_task"
    }

    fn description(&self) -> &'static str {
        "Send a task to a sub-agent and trigger execution.\n\
         Returns immediately. Use wait_agent to collect results.\n\
         Set interrupt=false to queue after current task completes."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_path": {
                    "type": "string",
                    "description": "Target agent path (e.g., 'root/searcher')"
                },
                "task": {
                    "type": "string",
                    "description": "Task description for the sub-agent"
                },
                "interrupt": {
                    "type": "boolean",
                    "description": "Whether to interrupt current task (default: true)"
                }
            },
            "required": ["agent_path", "task"]
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
    }

    async fn call_typed(
        &self,
        args: Self::Args,
        _ctx: &ToolContext,
    ) -> AgentResult<Self::Output> {
        match self
            .runtime
            .send_task(&args.agent_path, args.task, args.interrupt)
        {
            Ok(accepted) => Ok(FollowupTaskOutput {
                accepted,
                agent_path: args.agent_path,
            }),
            Err(e) => Ok(FollowupTaskOutput {
                accepted: false,
                agent_path: format!("error: {}", e),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// 4. wait_agent
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct WaitAgentArgs {
    #[serde(default)]
    agent_path: Option<String>,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    120_000
}

#[derive(Debug, Serialize)]
pub struct WaitAgentOutput {
    status: String,
    result: Option<String>,
    agent_path: Option<String>,
    has_more: bool,
}

pub struct WaitAgentTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl WaitAgentTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for WaitAgentTool {
    type Args = WaitAgentArgs;
    type Output = WaitAgentOutput;

    fn name(&self) -> &'static str {
        "wait_agent"
    }

    fn description(&self) -> &'static str {
        "Wait for a sub-agent to complete and return its result.\n\
         If agent_path is omitted, waits for ANY sub-agent.\n\
         Returns timeout if no agent completes within the timeout.\n\
         Check has_more for additional pending results."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_path": {
                    "type": "string",
                    "description": "Optional: specific agent to wait for. Omit for any."
                },
                "timeout_ms": {
                    "type": "number",
                    "description": "Max wait time in ms (default: 120000 = 2 min)"
                }
            },
            "required": []
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
    }

    async fn call_typed(
        &self,
        args: Self::Args,
        _ctx: &ToolContext,
    ) -> AgentResult<Self::Output> {
        let result = self
            .runtime
            .wait_for_result(args.agent_path.as_deref(), args.timeout_ms)
            .await;

        Ok(WaitAgentOutput {
            status: result.status,
            result: result.result,
            agent_path: result.agent_path,
            has_more: result.has_more,
        })
    }
}

// ---------------------------------------------------------------------------
// 5. list_agents
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ListAgentsArgs {}

#[derive(Debug, Serialize)]
pub struct ListAgentItem {
    agent_path: String,
    status: String,
    tool_count: usize,
}

pub struct ListAgentsTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl ListAgentsTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for ListAgentsTool {
    type Args = ListAgentsArgs;
    type Output = Vec<ListAgentItem>;

    fn name(&self) -> &'static str {
        "list_agents"
    }

    fn description(&self) -> &'static str {
        "List all active sub-agents and their status.\n\
         Status: idle (ready), running (executing), done (completed, awaiting close or new task)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
    }

    async fn call_typed(
        &self,
        _args: Self::Args,
        _ctx: &ToolContext,
    ) -> AgentResult<Self::Output> {
        let agents = self.runtime.list_agents();
        Ok(agents
            .into_iter()
            .map(|a| ListAgentItem {
                agent_path: a.agent_path,
                status: a.status,
                tool_count: a.tool_count,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// 6. close_agent
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CloseAgentArgs {
    agent_path: String,
}

#[derive(Debug, Serialize)]
pub struct CloseAgentOutput {
    closed: bool,
    previous_status: String,
    message: String,
}

pub struct CloseAgentTool {
    runtime: Arc<MultiAgentRuntime>,
}

impl CloseAgentTool {
    pub fn new(runtime: Arc<MultiAgentRuntime>) -> Self {
        Self { runtime }
    }
}

#[async_trait::async_trait]
impl TypedTool for CloseAgentTool {
    type Args = CloseAgentArgs;
    type Output = CloseAgentOutput;

    fn name(&self) -> &'static str {
        "close_agent"
    }

    fn description(&self) -> &'static str {
        "Close a sub-agent and release its resources.\n\
         Immediately stops the agent (aborts current task) and removes it.\n\
         Pending wait_agent calls for this agent return status='closed'."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "agent_path": {
                    "type": "string",
                    "description": "Agent path to close (e.g., 'root/searcher')"
                }
            },
            "required": ["agent_path"]
        })
    }

    fn control_flow() -> ToolControlFlow {
        ToolControlFlow::Continue
    }

    fn format_output(&self, output: Self::Output) -> String {
        serde_json::to_string(&output).unwrap_or_default()
    }

    async fn call_typed(
        &self,
        args: Self::Args,
        _ctx: &ToolContext,
    ) -> AgentResult<Self::Output> {
        match self.runtime.close_agent(&args.agent_path) {
            Ok(result) => Ok(CloseAgentOutput {
                closed: result.closed,
                previous_status: result.previous_status,
                message: result.message,
            }),
            Err(e) => Ok(CloseAgentOutput {
                closed: false,
                previous_status: "unknown".to_string(),
                message: e,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::{Language, LlmClient, ToolControlFlow};
    use std::pin::Pin;
    use tokio_util::sync::CancellationToken;

    use crate::multi_agent::config::MultiAgentConfig;
    use crate::multi_agent::runtime::MultiAgentRuntime;

    // ── Minimal mock LLM client ──

    struct StubClient;

    #[async_trait::async_trait]
    impl LlmClient for StubClient {
        async fn chat(
            &self,
            _messages: &[agent_base::ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&agent_base::ResponseFormat>,
        ) -> AgentResult<serde_json::Value> {
            Ok(serde_json::json!({"choices": [{"message": {"content": "ok"}}]}))
        }

        async fn chat_stream(
            &self,
            _messages: &[agent_base::ChatMessage],
            _tools: &[serde_json::Value],
            _reasoning: Option<&agent_base::ReasoningConfig>,
            _response_format: Option<&agent_base::ResponseFormat>,
        ) -> AgentResult<Pin<Box<dyn futures_core::Stream<Item = AgentResult<agent_base::StreamChunk>> + Send>>> {
            let chunks: Vec<AgentResult<agent_base::StreamChunk>> = vec![
                Ok(agent_base::StreamChunk::Text("ok".to_string())),
                Ok(agent_base::StreamChunk::Stop),
            ];
            Ok(Box::pin(futures_util::stream::iter(chunks)))
        }

        fn capabilities(&self) -> agent_base::LlmCapabilities {
            agent_base::LlmCapabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_vision: false,
                supports_thinking: false,
                max_context_tokens: None,
                max_output_tokens: None,
            }
        }
    }

    fn make_runtime() -> Arc<MultiAgentRuntime> {
        let client = Arc::new(StubClient);
        let cancel = CancellationToken::new();
        Arc::new(MultiAgentRuntime::new(
            MultiAgentConfig::enabled(),
            client,
            vec![],
            cancel,
            None,
            Language::En,
        ))
    }

    fn make_tool_ctx() -> ToolContext {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<agent_base::UserEvent>();
        ToolContext {
            session_id: agent_base::SessionId::new(1),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: Language::En,
            cancel_token: CancellationToken::new(),
        }
    }

    // ── name / description / schema / control_flow ──

    #[test]
    fn test_spawn_agent_tool_metadata() {
        let t = SpawnAgentTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "spawn_agent");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["required"].as_array().unwrap().contains(&"task_name".into()));
        assert!(schema["required"].as_array().unwrap().contains(&"message".into()));
        assert!(matches!(SpawnAgentTool::control_flow(), ToolControlFlow::Continue));
    }

    #[test]
    fn test_send_message_tool_metadata() {
        let t = SendMessageTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "send_message");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["required"].as_array().unwrap().contains(&"agent_path".into()));
        assert!(matches!(SendMessageTool::control_flow(), ToolControlFlow::Continue));
    }

    #[test]
    fn test_followup_task_tool_metadata() {
        let t = FollowupTaskTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "followup_task");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.parameters_schema();
        assert!(schema["required"].as_array().unwrap().contains(&"agent_path".into()));
        assert!(matches!(FollowupTaskTool::control_flow(), ToolControlFlow::Continue));
    }

    #[test]
    fn test_wait_agent_tool_metadata() {
        let t = WaitAgentTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "wait_agent");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["required"].as_array().unwrap().is_empty());
        assert!(matches!(WaitAgentTool::control_flow(), ToolControlFlow::Continue));
    }

    #[test]
    fn test_list_agents_tool_metadata() {
        let t = ListAgentsTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "list_agents");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(matches!(ListAgentsTool::control_flow(), ToolControlFlow::Continue));
    }

    #[test]
    fn test_close_agent_tool_metadata() {
        let t = CloseAgentTool::new(make_runtime());
        assert_eq!(agent_base::TypedTool::name(&t), "close_agent");
        assert!(!agent_base::TypedTool::description(&t).is_empty());
        let schema = t.parameters_schema();
        assert!(schema["required"].as_array().unwrap().contains(&"agent_path".into()));
        assert!(matches!(CloseAgentTool::control_flow(), ToolControlFlow::Continue));
    }

    // ── format_output ──

    #[test]
    fn test_spawn_agent_format_output() {
        let t = SpawnAgentTool::new(make_runtime());
        let out = t.format_output(SpawnAgentOutput {
            agent_path: "root/w1".into(),
            message: "ok".into(),
        });
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["agent_path"], "root/w1");
        assert_eq!(v["message"], "ok");
    }

    #[test]
    fn test_send_message_format_output() {
        let t = SendMessageTool::new(make_runtime());
        let out = t.format_output(SendMessageOutput { delivered: true });
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["delivered"], true);
    }

    #[test]
    fn test_followup_task_format_output() {
        let t = FollowupTaskTool::new(make_runtime());
        let out = t.format_output(FollowupTaskOutput { accepted: true, agent_path: "root/w1".into() });
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["accepted"], true);
    }

    #[test]
    fn test_wait_agent_format_output() {
        let t = WaitAgentTool::new(make_runtime());
        let out = t.format_output(WaitAgentOutput {
            status: "timeout".into(),
            result: None,
            agent_path: None,
            has_more: false,
        });
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["status"], "timeout");
        assert_eq!(v["has_more"], false);
    }

    #[test]
    fn test_close_agent_format_output() {
        let t = CloseAgentTool::new(make_runtime());
        let out = t.format_output(CloseAgentOutput {
            closed: true,
            previous_status: "idle".into(),
            message: "done".into(),
        });
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["closed"], true);
    }

    // ── call_typed for tools that don't need spawn ──

    #[tokio::test]
    async fn test_list_agents_call_empty() {
        let rt = make_runtime();
        let t = ListAgentsTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t.call_typed(ListAgentsArgs {}, &ctx).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_close_agent_nonexistent() {
        let rt = make_runtime();
        let t = CloseAgentTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(CloseAgentArgs { agent_path: "root/ghost".into() }, &ctx)
            .await
            .unwrap();
        assert!(!result.closed);
        assert_eq!(result.previous_status, "unknown");
    }

    #[tokio::test]
    async fn test_send_message_nonexistent() {
        let rt = make_runtime();
        let t = SendMessageTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(SendMessageArgs { agent_path: "root/ghost".into(), message: "hi".into() }, &ctx)
            .await
            .unwrap();
        assert!(!result.delivered);
    }

    #[tokio::test]
    async fn test_followup_task_nonexistent() {
        let rt = make_runtime();
        let t = FollowupTaskTool::new(rt);
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(
                FollowupTaskArgs { agent_path: "root/ghost".into(), task: "do".into(), interrupt: true },
                &ctx,
            )
            .await
            .unwrap();
        assert!(!result.accepted);
    }

    // ── send_message + followup_task + wait result round-trip ──

    #[tokio::test]
    async fn test_send_message_and_followup_task_roundtrip() {
        let rt = make_runtime();

        // Spawn a child first
        let path = rt
            .spawn_child("worker", "you are a worker".into(), 1, 0)
            .await
            .unwrap();

        // Send a message (no execution trigger)
        let t = SendMessageTool::new(rt.clone());
        let ctx = make_tool_ctx();
        let result = t
            .call_typed(SendMessageArgs { agent_path: path.clone(), message: "context info".into() }, &ctx)
            .await
            .unwrap();
        assert!(result.delivered);

        // Send a task (triggers execution, drains pending messages)
        let t2 = FollowupTaskTool::new(rt.clone());
        let result2 = t2
            .call_typed(
                FollowupTaskArgs { agent_path: path.clone(), task: "do work".into(), interrupt: true },
                &ctx,
            )
            .await
            .unwrap();
        assert!(result2.accepted);

        // Wait for result
        let t3 = WaitAgentTool::new(rt.clone());
        let result3 = t3
            .call_typed(
                WaitAgentArgs { agent_path: Some(path.clone()), timeout_ms: 5000 },
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(result3.status, "ok");
        assert!(result3.result.is_some());

        // Close
        let t4 = CloseAgentTool::new(rt.clone());
        let result4 = t4
            .call_typed(CloseAgentArgs { agent_path: path.clone() }, &ctx)
            .await
            .unwrap();
        assert!(result4.closed);
    }

    // ── spawn_agent call_typed ──

    #[tokio::test]
    async fn test_spawn_agent_call() {
        let rt = make_runtime();
        let t = SpawnAgentTool::new(rt.clone());
        let ctx = make_tool_ctx();

        let result = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "helper".into(),
                    message: "do something".into(),
                    agent_type: None,
                    system_prompt: Some("you are a helper".into()),
                    model: None,
                    reasoning_effort: None,
                    fork_history: None,
                },
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(result.agent_path, "root/helper");
        assert!(result.message.contains("spawned"));

        // Verify the agent shows up in list
        let t2 = ListAgentsTool::new(rt);
        let list = t2.call_typed(ListAgentsArgs {}, &ctx).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].agent_path, "root/helper");
    }

    // ── spawn with auto-generated system prompt from agent_type ──

    #[tokio::test]
    async fn test_spawn_agent_with_agent_type() {
        let rt = make_runtime();
        let t = SpawnAgentTool::new(rt);
        let ctx = make_tool_ctx();

        let result = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "searcher".into(),
                    message: "search for info".into(),
                    agent_type: Some("researcher".into()),
                    system_prompt: None,
                    model: None,
                    reasoning_effort: None,
                    fork_history: None,
                },
                &ctx,
            )
            .await
            .unwrap();

        assert_eq!(result.agent_path, "root/searcher");
    }

    // ── spawn limit exceeded ──

    #[tokio::test]
    async fn test_spawn_agent_limit_exceeded() {
        let client = Arc::new(StubClient);
        let cancel = CancellationToken::new();
        let config = MultiAgentConfig { enabled: true, max_sub_agents: 1, max_agent_depth: 1 };
        let rt = Arc::new(MultiAgentRuntime::new(config, client, vec![], cancel, None, Language::En));

        let t = SpawnAgentTool::new(rt.clone());
        let ctx = make_tool_ctx();

        // First spawn succeeds
        let r1 = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "first".into(),
                    message: "task".into(),
                    agent_type: None,
                    system_prompt: None,
                    model: None,
                    reasoning_effort: None,
                    fork_history: None,
                },
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(r1.agent_path, "root/first");

        // Second spawn fails
        let r2 = t
            .call_typed(
                SpawnAgentArgs {
                    task_name: "second".into(),
                    message: "task".into(),
                    agent_type: None,
                    system_prompt: None,
                    model: None,
                    reasoning_effort: None,
                    fork_history: None,
                },
                &ctx,
            )
            .await
            .unwrap();

        assert!(r2.agent_path.is_empty());
        assert!(r2.message.contains("Failed"));
    }

    // ── create_all_tools ──

    #[test]
    fn test_create_all_tools_returns_six() {
        let rt = make_runtime();
        let tools = create_all_tools(rt);
        assert_eq!(tools.len(), 6);

        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"spawn_agent"));
        assert!(names.contains(&"send_message"));
        assert!(names.contains(&"followup_task"));
        assert!(names.contains(&"wait_agent"));
        assert!(names.contains(&"list_agents"));
        assert!(names.contains(&"close_agent"));
    }
}
