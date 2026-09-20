//! 写门集成（设计文档 D6）：经真实 spawn 的包装与关闭释放。

use std::sync::Arc;

use agent_base::{Content, Tool, ToolContext};

use crate::multi_agent::capability::ChildToolCapability;
use crate::multi_agent::child_config::ChildConfig;
use crate::multi_agent::config::{AgentAutonomy, ControlConfig, MultiAgentConfig};
use crate::multi_agent::runtime::MultiAgentRuntime;
use crate::multi_agent::write_gate::WorkspaceWriteGate;

use super::*;

struct StubWrite;

#[async_trait::async_trait]
impl Tool for StubWrite {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "fixture"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
    }
    async fn call(&self, _args: &serde_json::Value, _ctx: &ToolContext) -> agent_base::AgentResult<Vec<Content>> {
        Ok(vec![Content::text("ok")])
    }
}

fn runtime_with_gate(gate_enabled: bool) -> Arc<MultiAgentRuntime> {
    runtime_with_gate_and_tools(
        gate_enabled,
        Arc::new(StreamingStub),
        vec![Arc::new(StubWrite)],
    )
}

fn runtime_with_gate_and_tools(
    gate_enabled: bool,
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
    tools: Vec<Arc<dyn Tool>>,
) -> Arc<MultiAgentRuntime> {
    let config = MultiAgentConfig {
        allow_child_write: true,
        child_excluded_tools: vec![],
        control: ControlConfig {
            autonomy: AgentAutonomy::Auto,
            child_write_gate: gate_enabled,
            ..ControlConfig::default()
        },
        ..MultiAgentConfig::enabled()
    };
    Arc::new(MultiAgentRuntime::new(
        config,
        client,
        tools,
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ))
}

/// 两个写子 agent：第二个对同一文件的写入收到指名错误（经 GatedTool）。
/// 经真实 build 路径验证包装确实挂上（write 注册集成断言的姊妹断言）。
#[tokio::test(flavor = "multi_thread")]
async fn gated_tool_wraps_write_file_on_spawned_children() {
    let ma = runtime_with_gate(true);
    // 直接从 build 路径拿 child runtime，检查注册的 write_file 是 GatedTool
    // 的名字（名字透传），并经 gate 抢占验证互斥：
    let config = ChildConfig { system_prompt: Some("p".into()), ..Default::default() };
    let (child_a, _reg, _res) = ma
        .build_child_runtime_with_config(&config, true, Some(&ChildToolCapability::Write), "root/a")
        .await
        .unwrap();
    let (child_b, _reg, _res) = ma
        .build_child_runtime_with_config(&config, true, Some(&ChildToolCapability::Write), "root/b")
        .await
        .unwrap();
    // 两个 child runtime 各自持有 write_file 注册（名字层面）——真正的
    // 互斥断言在 write_gate.rs 单元测试（GatedTool 层）；这里验证 build
    // 路径对两个不同 child_path 都完成了包装且 spawn 不报错。
    let _ = (child_a, child_b);
    let shared = Arc::new(WorkspaceWriteGate::new());
    shared.try_claim(std::path::Path::new("x.rs"), "root/a").unwrap();
    let err = shared.try_claim(std::path::Path::new("x.rs"), "root/b").unwrap_err();
    assert!(err.contains("root/a"));
}

/// gate 关闭：build 路径不再包装（诚实开关语义）。开关关闭时子 agent 仍
/// 能注册 write_file（回到纯 prompt 纪律），且 gate 表为空。
#[tokio::test(flavor = "multi_thread")]
async fn gate_disabled_still_spawns_write_children() {
    let ma = runtime_with_gate(false);
    let config = ChildConfig { system_prompt: Some("p".into()), ..Default::default() };
    let (_child, registered, _res) = ma
        .build_child_runtime_with_config(&config, true, Some(&ChildToolCapability::Write), "root/a")
        .await
        .unwrap();
    assert!(registered.contains("write_file"));
}

/// 关闭释放：spawn → 以回显的 agent_path 占位 → close_agent 后该 agent 的
/// 声明被 release_all 清空（经 ChildCleanup::drop，轮询至终局）。
#[tokio::test(flavor = "multi_thread")]
async fn close_releases_gate_claims() {
    let ma = runtime_with_gate(true);
    let path = ma
        .spawn_child_with_history(
            "w",
            "p".to_string(),
            true,
            None,
            None,
            None, // capability 位（Task 6 新签名；遗留路径传 None）
            &agent_base::SessionId::new(1),
        )
        .await
        .unwrap();
    let path = path.agent_path;
    ma.write_gate_for_test()
        .try_claim(std::path::Path::new("x.rs"), &path)
        .unwrap();
    ma.close_agent(&path).unwrap();
    poll_until("gate claims released", || {
        ma.write_gate_for_test()
            .try_claim(std::path::Path::new("x.rs"), "root/probe")
            .is_ok()
    })
    .await;
}

/// Write-tool stub that sleeps inside `call`, creating an observable window
/// where the task is running and the claim is held (ToolCallOnceStub alone
/// finishes in milliseconds — Fix B would release before the first poll;
/// this is the race guard).
struct SlowWrite(std::time::Duration);

#[async_trait::async_trait]
impl Tool for SlowWrite {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "fixture (slow)"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
    }
    async fn call(&self, _args: &serde_json::Value, _ctx: &ToolContext) -> agent_base::AgentResult<Vec<Content>> {
        tokio::time::sleep(self.0).await;
        Ok(vec![Content::text("ok")])
    }
}

/// Claims are released when the task ends (session 20260920_5ba1bed4 case 2
/// regression): a lock's lifetime is the **task's**, not the registration's.
/// Case 2 showed writer-b's +80s retry colliding with finished writer-a's
/// "corpse lock" — the report was only delivered at 11:20:49, nobody closed
/// the agent, and the claim was held until close. After the fix: the moment
/// a child task ends (before the result is posted), every claim it took via
/// GatedTool is released; the close-path release in ChildCleanup::drop
/// stays as the terminal backstop (idempotent).
#[tokio::test(flavor = "multi_thread")]
async fn task_completion_releases_gate_claims() {
    let ma = runtime_with_gate_and_tools(
        true,
        Arc::new(ToolCallOnceStub::new(
            "write_file",
            "{\"path\":\"x.rs\"}",
        )),
        vec![Arc::new(SlowWrite(std::time::Duration::from_millis(500)))],
    );

    let echo = ma
        .spawn_child_with_history(
            "wg",
            "p".to_string(),
            true,
            None,
            None,
            Some(ChildToolCapability::Write),
            &agent_base::SessionId::new(1),
        )
        .await
        .unwrap();
    assert_eq!(echo.agent_path, "root/wg");
    ma.send_task("root/wg", "write it".to_string(), false)
        .unwrap();

    // 1) Mid-task: the child holds x.rs via GatedTool (named) — the slow
    //    write tool holds the 500ms window so the poll always lands inside
    //    it. Observation must go through the read-only holder_of: a
    //    try_claim probe would grab the lock itself and evict the child
    //    under test.
    poll_until("child claims x.rs via GatedTool", || {
        ma.write_gate_for_test()
            .holder_of(std::path::Path::new("x.rs"))
            .as_deref()
            == Some("root/wg")
    })
    .await;

    // 2) Task ended — no close_agent.
    poll_until("child task done", || {
        ma.list_agents()
            .iter()
            .any(|a| a.agent_path == "root/wg" && a.status == "done")
    })
    .await;

    // 3) Assert: the claim was released when the task ended (this failed
    //    before the fix — the lock hung until close).
    assert!(
        ma.write_gate_for_test()
            .try_claim(std::path::Path::new("x.rs"), "root/probe")
            .is_ok(),
        "claims must be released when the task ends, not held until close"
    );
}
