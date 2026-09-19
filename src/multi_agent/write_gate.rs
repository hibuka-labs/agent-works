//! 子 agent 间文件级写互斥（设计 2026-09-19 D6）。
//!
//! 进程内 Mutex 表：CanonicalPath → 持有者 agent_path。**任务期持有**——
//! 声明在子 agent 生命周期内有效（不是单次调用期），第二个子 agent 一写
//! 就得到指名错误，向父报告，父重新分界。`try_claim` 永不阻塞：无等待即
//! 无死锁面。释放钩子挂在 `ChildCleanup::drop`（正常 close / panic /
//! abort 全走这一处）；超时 reaper 不释放（agent 存活，声明继续有效）。
//!
//! 诚实边界：`execute_command` 的 shell 重定向在门外——gate 只包
//! `write_file`/`edit_file` 主通道（框架已知的注册名），shell 通道靠任务
//! 分界纪律 + 父 prompt 指引。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_base::{Content, Tool, ToolContext};

/// gate 拦截的工具（框架已知的注册名，设计 D6）。
pub const WRITE_GATE_TOOLS: &[&str] = &["write_file", "edit_file"];

/// 进程内写门。一个
/// [`MultiAgentRuntime`](crate::multi_agent::runtime::MultiAgentRuntime)
/// 一个实例（跨子 agent 共享才能互斥）；父 agent 豁免（gate 只包子
/// agent 的工具实例）。
#[derive(Default)]
pub struct WorkspaceWriteGate {
    holders: Mutex<HashMap<PathBuf, String>>,
}

impl WorkspaceWriteGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// 空闲 → 占用并 Ok；已被自己持有 → 幂等 Ok；他人持有 → 立即
    /// Err（指名，不不等待）。
    pub fn try_claim(&self, path: &Path, agent: &str) -> Result<(), String> {
        // canonicalize 让同一文件的不同相对/绝对拼写指向同一条目；
        // 失败（尚不存在的文件）退回原路径——同拼写仍互斥，诚实降级。
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let mut holders = self.holders.lock().unwrap();
        match holders.get(&canonical) {
            Some(owner) if owner == agent => Ok(()),
            Some(owner) => Err(format!("file locked by {owner}")),
            None => {
                holders.insert(canonical, agent.to_string());
                Ok(())
            }
        }
    }

    /// agent 关闭路径调用：释放它持有的全部声明。
    pub fn release_all(&self, agent: &str) {
        self.holders
            .lock()
            .unwrap()
            .retain(|_, owner| owner != agent);
    }
}

/// 包装写工具：调用前 `try_claim(args.path, agent_path)`，失败即指名错误
/// （子 agent 向父报告冲突，父重新分界——响亮失败，非阻塞）。
pub struct GatedTool {
    inner: Arc<dyn Tool>,
    gate: Arc<WorkspaceWriteGate>,
    agent_path: String,
}

impl GatedTool {
    pub fn new(inner: Arc<dyn Tool>, gate: Arc<WorkspaceWriteGate>, agent_path: String) -> Self {
        Self {
            inner,
            gate,
            agent_path,
        }
    }
}

#[async_trait::async_trait]
impl Tool for GatedTool {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn description(&self) -> &'static str {
        self.inner.description()
    }
    fn schema(&self) -> serde_json::Value {
        self.inner.schema()
    }
    async fn call(
        &self,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> agent_base::AgentResult<Vec<Content>> {
        let path = args
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if path.is_empty() {
            return Err(agent_base::AgentError::internal(format!(
                "{}: missing `path` argument; write gate cannot claim",
                self.inner.name()
            )));
        }
        match self.gate.try_claim(Path::new(path), &self.agent_path) {
            Ok(()) => self.inner.call(args, ctx).await,
            Err(why) => Err(agent_base::AgentError::internal(format!(
                "{}: {why}",
                self.inner.name()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingTool {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &'static str {
            "write_file"
        }
        fn description(&self) -> &'static str {
            "fixture"
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}})
        }
        async fn call(
            &self,
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> agent_base::AgentResult<Vec<Content>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Content::text("written")])
        }
    }

    fn gated() -> (Arc<WorkspaceWriteGate>, Arc<CountingTool>, GatedTool) {
        let gate = Arc::new(WorkspaceWriteGate::new());
        let tool = Arc::new(CountingTool {
            calls: AtomicUsize::new(0),
        });
        let wrapped = GatedTool::new(tool.clone(), gate.clone(), "root/a".to_string());
        (gate, tool, wrapped)
    }

    fn args(path: &str) -> serde_json::Value {
        serde_json::json!({ "path": path })
    }

    #[tokio::test]
    async fn claim_then_delegate() {
        let (_g, tool, wrapped) = gated();
        let ctx = ToolContext::for_test();
        let out = wrapped.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        assert!(
            matches!(&out[0], Content::Text { text } if text == "written"),
            "gated tool must delegate to the inner tool"
        );
        assert_eq!(tool.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn second_agent_gets_named_error() {
        let (gate, _t, wrapped_a) = gated();
        let ctx = ToolContext::for_test();
        wrapped_a.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        let wrapped_b = GatedTool::new(
            Arc::new(CountingTool {
                calls: AtomicUsize::new(0),
            }),
            gate,
            "root/b".to_string(),
        );
        let err = wrapped_b
            .call(&args("tmp/x.txt"), &ctx)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("file locked by root/a"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn self_reclaim_is_idempotent() {
        let (_g, tool, wrapped) = gated();
        let ctx = ToolContext::for_test();
        wrapped.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        wrapped.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        assert_eq!(
            tool.calls.load(Ordering::SeqCst),
            2,
            "自己重入幂等放行"
        );
    }

    #[tokio::test]
    async fn release_all_frees_the_file() {
        let (gate, _t, wrapped_a) = gated();
        let ctx = ToolContext::for_test();
        wrapped_a.call(&args("tmp/x.txt"), &ctx).await.unwrap();
        gate.release_all("root/a");
        let wrapped_b = GatedTool::new(
            Arc::new(CountingTool {
                calls: AtomicUsize::new(0),
            }),
            gate,
            "root/b".to_string(),
        );
        wrapped_b
            .call(&args("tmp/x.txt"), &ctx)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn missing_path_argument_fails_loud() {
        let (_g, tool, wrapped) = gated();
        let ctx = ToolContext::for_test();
        let err = wrapped
            .call(&serde_json::json!({ "other": 1 }), &ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("missing `path`"), "{err}");
        assert_eq!(tool.calls.load(Ordering::SeqCst), 0, "no claim, no call");
    }

    #[test]
    fn distinct_spellings_of_same_file_collide() {
        let gate = WorkspaceWriteGate::new();
        let dir = std::env::temp_dir().join("phimint_gate_test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("f.txt");
        std::fs::write(&file, "x").unwrap();
        gate.try_claim(&file, "root/a").unwrap();
        let err = gate
            .try_claim(&dir.join("./f.txt"), "root/b")
            .unwrap_err();
        assert!(err.contains("root/a"));
        gate.release_all("root/a");
        gate.try_claim(&file, "root/b").unwrap();
    }
}
