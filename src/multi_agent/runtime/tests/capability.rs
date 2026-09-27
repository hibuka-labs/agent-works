//! 能力解析器的接线测试（设计文档 v3 外部声音发现 1 的回归锁）：
//! "write 子 agent 能注册 write_file" 是**集成断言**——验证 build 路径
//! 逐字消费解析器输出的排除集，而不是单元断言。

use std::sync::Arc;

use agent_base::{Content, Tool, ToolContext};

use crate::multi_agent::capability::{ChildToolCapability, resolve_capability};
use crate::multi_agent::child_config::ChildConfig;
use crate::multi_agent::config::{AgentAutonomy, ControlConfig, MultiAgentConfig};
use crate::multi_agent::runtime::MultiAgentRuntime;

use super::*;

struct StubWriteFile;

#[async_trait::async_trait]
impl Tool for StubWriteFile {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn description(&self) -> &'static str {
        "Write a file"
    }
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
    }
    async fn call(
        &self,
        _args: &serde_json::Value,
        _ctx: &ToolContext,
    ) -> agent_base::AgentResult<Vec<Content>> {
        Ok(vec![Content::text("ok")])
    }
}

fn write_enabled_config(autonomy: AgentAutonomy, allow: bool) -> MultiAgentConfig {
    MultiAgentConfig {
        child_permission_mode: crate::multi_agent::config::ChildPermissionMode::Full,
        allow_child_write: allow,
        child_excluded_tools: vec!["write_file".to_string()],
        control: ControlConfig {
            autonomy,
            ..ControlConfig::default()
        },
        ..MultiAgentConfig::enabled()
    }
}

fn runtime_with_config(
    config: MultiAgentConfig,
    client: Arc<dyn agent_base::llm_trait::LlmProvider>,
) -> Arc<MultiAgentRuntime> {
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(NoopReadFileTool) as Arc<dyn Tool>,
        Arc::new(StubWriteFile),
    ];
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

/// THE integration assertion: write 子 agent 的注册集确实含 write_file
/// （Auto + allow，部署排除表里虽有 write_file，但被写豁免救回）。
#[tokio::test(flavor = "multi_thread")]
async fn write_capability_child_registers_write_file() {
    let ma = runtime_with_config(
        write_enabled_config(AgentAutonomy::Auto, true),
        Arc::new(StreamingStub),
    );
    let config = ChildConfig {
        system_prompt: Some("prompt".into()),
        ..Default::default()
    };
    let (_child, registered, _res) = ma
        .build_child_runtime_with_config(
            &config,
            true,
            Some(&ChildToolCapability::Write),
            "root/test",
        )
        .await
        .expect("spawn builds");
    assert!(
        registered.contains("write_file"),
        "write child must hold write_file, got {registered:?}"
    );
    assert!(registered.contains("read_file"));
}

/// CRITICAL 回归镜像（框架侧）：默认 spawn（Some(ReadOnly) —— LLM 面缺省）
/// 注册集不含任何写工具。
#[tokio::test(flavor = "multi_thread")]
async fn default_spawn_child_never_registers_write_tools() {
    let ma = runtime_with_config(
        write_enabled_config(AgentAutonomy::Auto, true),
        Arc::new(StreamingStub),
    );
    let config = ChildConfig {
        system_prompt: Some("prompt".into()),
        ..Default::default()
    };
    let (_child, registered, _res) = ma
        .build_child_runtime_with_config(
            &config,
            true,
            Some(&ChildToolCapability::ReadOnly),
            "root/test",
        )
        .await
        .expect("spawn builds");
    assert!(
        !registered.contains("write_file"),
        "read-only child must not hold write_file"
    );
    assert!(registered.contains("read_file"));
}

/// 遗留路径（capability=None）字节不变式：write_file 被部署排除表拦下。
#[tokio::test(flavor = "multi_thread")]
async fn legacy_path_still_excludes_by_table() {
    let ma = runtime_with_config(
        write_enabled_config(AgentAutonomy::Auto, true),
        Arc::new(StreamingStub),
    );
    let config = ChildConfig {
        system_prompt: Some("prompt".into()),
        ..Default::default()
    };
    let (_child, registered, _res) = ma
        .build_child_runtime_with_config(&config, true, None, "root/test")
        .await
        .expect("spawn builds");
    assert!(
        !registered.contains("write_file"),
        "legacy path keeps the deployment table"
    );
}

/// 解析器输出的排除集与 build 消费的一致性（框架层 CRITICAL 断言）：
/// phimint 形状的部署表下，ReadOnly 解析后的排除集覆盖全部写工具。
#[test]
fn read_only_resolution_covers_all_write_tools() {
    let table: Vec<String> = ["write_file", "task_output"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let res = resolve_capability(
        Some(&ChildToolCapability::ReadOnly),
        AgentAutonomy::Auto,
        true,
        &table,
        &[
            "write_file".to_string(),
            "edit_file".to_string(),
            "execute_command".to_string(),
        ],
    );
    for t in ["write_file", "edit_file", "execute_command"] {
        assert!(res.excluded_tools.contains(t));
    }
    assert!(res.excluded_tools.contains("task_output"));
}

/// spawn 级回显（外部声音发现 9）：write 请求在 allow=false 部署下携带
/// 降级原因；allowed 时 registered_tools 含 write_file。
#[tokio::test(flavor = "multi_thread")]
async fn spawn_echo_carries_registered_and_degradation() {
    // allowed 部署：echo.registered_tools 含 write_file，无降级。
    let ma = runtime_with_config(
        write_enabled_config(AgentAutonomy::Auto, true),
        Arc::new(StreamingStub),
    );
    let echo = ma
        .spawn_child_with_history(
            "echo-w",
            "prompt".to_string(),
            true,
            None,
            None,
            Some(ChildToolCapability::Write),
            &agent_base::SessionId::new(1),
        )
        .await
        .unwrap();
    assert!(echo.registered_tools.contains("write_file"));
    assert!(echo.degraded_reason.is_none());
    ma.close_agent(&echo.agent_path).unwrap();

    // 关闭部署：降级原因回显。
    let ma2 = runtime_with_config(
        write_enabled_config(AgentAutonomy::Auto, false),
        Arc::new(StreamingStub),
    );
    let echo2 = ma2
        .spawn_child_with_history(
            "echo-r",
            "prompt".to_string(),
            true,
            None,
            None,
            Some(ChildToolCapability::Write),
            &agent_base::SessionId::new(1),
        )
        .await
        .unwrap();
    let why = echo2.degraded_reason.expect("degraded echo must explain");
    assert!(why.contains("allow_child_write"));
    ma2.close_agent(&echo2.agent_path).unwrap();
}

/// T8 接线断言：list_agents 回显每个子 agent 实际注册的工具集；close 后
/// 条目随 ChildCleanup::drop 消失（与写门同一终局清理点）。
#[tokio::test(flavor = "multi_thread")]
async fn list_agents_echoes_spawned_tools_and_close_clears_them() {
    let ma = runtime_with_config(
        write_enabled_config(AgentAutonomy::Auto, true),
        Arc::new(StreamingStub),
    );
    let echo = ma
        .spawn_child_with_history(
            "echo-l",
            "prompt".to_string(),
            true,
            None,
            None,
            Some(ChildToolCapability::Write),
            &agent_base::SessionId::new(1),
        )
        .await
        .unwrap();
    let row = ma
        .list_agents()
        .into_iter()
        .find(|a| a.agent_path == echo.agent_path)
        .expect("spawned agent listed");
    assert!(
        row.spawned_tools.contains(&"write_file".to_string()),
        "list_agents must echo the registered set, got {:?}",
        row.spawned_tools
    );
    ma.close_agent(&echo.agent_path).unwrap();
    poll_until("spawned_tools echo cleared", || {
        !ma.list_agents()
            .iter()
            .any(|a| a.agent_path == echo.agent_path)
    })
    .await;
}

/// Session 20260919_f65b754c: read-only children echoed their full 14-tool
/// read/search baseline on every `list_agents` poll — accurate but token
/// burn, and it contradicted the list_agents contract ("empty spawned_tools
/// ⇒ read-only"). The echo is capability news only: read-only spawns leave
/// no entry, so the omission IS the fact. The one-time spawn message still
/// carries the complete registered set.
#[tokio::test]
async fn read_only_spawn_leaves_no_spawned_tools_echo() {
    let ma = runtime_with_config(
        write_enabled_config(AgentAutonomy::Auto, true),
        Arc::new(StreamingStub),
    );
    let echo = ma
        .spawn_child_with_history(
            "ro-l",
            "prompt".to_string(),
            false,
            None,
            None,
            Some(ChildToolCapability::ReadOnly),
            &agent_base::SessionId::new(1),
        )
        .await
        .unwrap();
    let row = ma
        .list_agents()
        .into_iter()
        .find(|a| a.agent_path == echo.agent_path)
        .expect("spawned agent listed");
    assert!(
        row.spawned_tools.is_empty(),
        "read-only child must not echo its baseline tool set, got {:?}",
        row.spawned_tools
    );
}
