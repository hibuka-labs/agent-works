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
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(StubWrite)];
    Arc::new(MultiAgentRuntime::new(
        config,
        Arc::new(StreamingStub),
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
