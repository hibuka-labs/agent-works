//! 子 agent 工具能力解析（设计文档 2026-09-19 D2/D3/D5）。
//!
//! LLM 只选择工具面（能调什么），永远不选择审批语义（D1 正交）。
//! 本模块是排除规则的**单一编码点**：原 `build_child_runtime_with_config`
//! 的排除集构造（含 Manual 合并）迁入 [`resolve_capability`]，输出的
//! `excluded_tools` 由注册循环逐字消费——循环零改动，改的是门内的名单。
//!
//! 规则（与载体无关）：写能力 = `write_tools` 成员豁免自己在排除表中的
//! 条目；`read_only` = `write_tools` 强制并入排除集（有牙的真只读）。

use std::collections::BTreeSet;

use super::config::AgentAutonomy;
use super::preset::ChildPreset;

/// spawn 请求的语义能力（schema 层的 `tools` 参数映射到这里）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildToolCapability {
    /// 强制真只读：write_tools 并入排除集。
    ReadOnly,
    /// 请求写能力：write_tools 成员从排除集中豁免。
    Write,
    /// 命名 preset：展开其白名单 + 角色 prompt（未知名由工具层先校验，
    /// 此处兜底为只读，无 panic 路径）。
    Preset(&'static str),
}

impl ChildToolCapability {
    /// 回显标签（spawn 输出 message 用）。
    pub fn label(&self) -> String {
        match self {
            Self::ReadOnly => "read_only".to_string(),
            Self::Write => "write".to_string(),
            Self::Preset(name) => format!("preset:{name}"),
        }
    }
}

/// 能力解析结果。降级不走 Result 错误通道——降级是成功解析附带说明。
#[derive(Debug, Clone, Default)]
pub struct CapabilityResolution {
    /// per-spawn 调整后的排除集（注册循环原样消费）。
    pub excluded_tools: BTreeSet<String>,
    /// preset 白名单（None = 不设白名单，沿用 ChildConfig.tool_names）。
    pub tool_names: Option<BTreeSet<String>>,
    /// preset 角色 prompt（拼在脚手架之前）。
    pub role_prompt: Option<String>,
    /// preset 的 max_turns。
    pub max_turns: Option<u32>,
    /// 降级原因（Some = 本次写请求被降级为只读）。
    pub degraded_reason: Option<String>,
}

/// 能力解析（纯函数，单元可测）。
///
/// `requested = None` 是遗留程序化路径（ChildBuilder / 旧 spawn_*）：
/// 完全跳过能力层，排除集与升级前逐字相同。LLM 面的 spawn 传
/// `Some`（schema 缺省 read_only）。
pub fn resolve_capability(
    requested: Option<&ChildToolCapability>,
    autonomy: AgentAutonomy,
    allow_child_write: bool,
    child_excluded_tools: &[String],
    write_tools: &[String],
) -> CapabilityResolution {
    // 第一道闸：Manual 合并（与升级前一致，能力层无法越过它）。
    let base: BTreeSet<String> = if autonomy == AgentAutonomy::Manual {
        child_excluded_tools
            .iter()
            .chain(write_tools)
            .cloned()
            .collect()
    } else {
        child_excluded_tools.iter().cloned().collect()
    };

    let Some(cap) = requested else {
        // 遗留程序化路径：无能力层，base 原样（Auto 不并 write_tools）。
        return CapabilityResolution {
            excluded_tools: base,
            ..Default::default()
        };
    };

    // preset 展开：白名单 + 角色 prompt + max_turns。
    let (tool_names, role_prompt, max_turns, preset_wants_write) = match cap {
        ChildToolCapability::Preset(name) => match ChildPreset::by_name(name) {
            Some(preset) => {
                let wants = preset
                    .config
                    .tool_names
                    .as_ref()
                    .map(|set| set.iter().any(|t| write_tools.contains(t)))
                    .unwrap_or(false);
                (
                    preset.config.tool_names.clone(),
                    preset.config.system_prompt.clone(),
                    preset.config.max_turns,
                    wants,
                )
            }
            // 工具层已校验名字；这里兜底为只读（无 panic 路径）。
            None => (None, None, None, false),
        },
        _ => (None, None, None, false),
    };

    let wants_write = match cap {
        ChildToolCapability::Write => true,
        ChildToolCapability::Preset(_) => preset_wants_write,
        ChildToolCapability::ReadOnly => false,
    };
    // Manual 永不批准；Auto 下看部署开关。
    let permitted = wants_write && autonomy == AgentAutonomy::Auto && allow_child_write;

    let excluded_tools = if permitted {
        // 写豁免：write_tools 成员从排除集中移除。
        base.into_iter()
            .filter(|t| !write_tools.contains(t))
            .collect()
    } else {
        // 只读强制：write_tools 并入排除集。
        let mut set = base;
        for t in write_tools {
            set.insert(t.clone());
        }
        set
    };

    let degraded_reason = if !wants_write {
        None
    } else if autonomy == AgentAutonomy::Manual {
        Some("deployment autonomy is Manual; children are hard read-only".to_string())
    } else if !allow_child_write {
        Some("allow_child_write=false; request degraded to read-only".to_string())
    } else {
        None // wants_write 且 Auto 且 allow ⇒ permitted，不可达
    };

    CapabilityResolution {
        excluded_tools,
        tool_names,
        role_prompt,
        max_turns,
        degraded_reason,
    }
}

/// read-only nudge 三条件（设计文档 D3.1，per-child 纯函数）：
/// `autonomy == Manual`（强制，压倒一切）
/// || `child_read_only`（强制全员 nudge 的偏执开关，默认开——框架默认部署
///    与今日行为逐字相同）
/// || 解析后排除集仍含全部 write_tools（子 agent 实际无写工具）。
pub fn read_only_nudge(
    autonomy: AgentAutonomy,
    child_read_only: bool,
    excluded_tools: &BTreeSet<String>,
    write_tools: &[String],
) -> bool {
    autonomy == AgentAutonomy::Manual
        || child_read_only
        || write_tools.iter().all(|t| excluded_tools.contains(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn excluded(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    const WRITE_TOOLS: &[&str] = &["write_file", "edit_file", "execute_command"];

    fn write_tools() -> Vec<String> {
        excluded(WRITE_TOOLS)
    }

    /// None（遗留路径）：Auto 不并 write_tools —— 与升级前逐字相同
    /// （preset fixture 回归哨兵的前提）。
    #[test]
    fn none_capability_keeps_legacy_base_set() {
        let res = resolve_capability(None, AgentAutonomy::Auto, false, &[], &write_tools());
        assert!(res.excluded_tools.is_empty());
        assert!(res.tool_names.is_none());
        assert!(res.degraded_reason.is_none());
    }

    #[test]
    fn none_capability_manual_merges_write_tools() {
        let res = resolve_capability(None, AgentAutonomy::Manual, false, &[], &write_tools());
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
    }

    /// read_only 强制化（v3）：Auto 下也并入 write_tools。
    #[test]
    fn read_only_forces_write_tools_into_exclusion() {
        let res = resolve_capability(
            Some(&ChildToolCapability::ReadOnly),
            AgentAutonomy::Auto,
            true,
            &[],
            &write_tools(),
        );
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t), "{t} must be excluded");
        }
    }

    /// 写能力 + 允许：write_tools 成员从排除集豁免（含部署表里误入的）。
    #[test]
    fn write_permitted_exempt_write_tools() {
        let table = excluded(&["write_file", "task_output"]);
        let res = resolve_capability(
            Some(&ChildToolCapability::Write),
            AgentAutonomy::Auto,
            true,
            &table,
            &write_tools(),
        );
        assert!(res.excluded_tools.contains("task_output"));
        assert!(!res.excluded_tools.contains("write_file"));
        assert!(!res.excluded_tools.contains("edit_file"));
        assert!(!res.excluded_tools.contains("execute_command"));
        assert!(res.degraded_reason.is_none());
        assert!(res.tool_names.is_none(), "write 不产生白名单");
    }

    /// 写能力 + 部署关闭：降级为只读 + 原因。
    #[test]
    fn write_denied_by_flag_degrades_with_reason() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Write),
            AgentAutonomy::Auto,
            false,
            &[],
            &write_tools(),
        );
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
        let why = res.degraded_reason.expect("must carry degraded reason");
        assert!(
            why.contains("allow_child_write"),
            "reason should name the flag: {why}"
        );
    }

    /// Manual 优先级最高：写请求被压回只读 + 原因。
    #[test]
    fn write_denied_by_manual_autonomy() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Write),
            AgentAutonomy::Manual,
            true,
            &[],
            &write_tools(),
        );
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
        let why = res.degraded_reason.expect("Manual denial must explain");
        assert!(why.contains("Manual"), "{why}");
    }

    /// preset（含写工具）+ 允许：白名单展开，排除集豁免写成员。
    #[test]
    fn preset_with_write_tools_permitted_expands() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Preset("coder")),
            AgentAutonomy::Auto,
            true,
            &[],
            &write_tools(),
        );
        let whitelist = res.tool_names.expect("coder preset carries whitelist");
        assert!(whitelist.contains("write_file"));
        assert!(whitelist.contains("read_file"));
        assert!(!res.excluded_tools.contains("write_file"));
        assert!(res.role_prompt.is_some());
        assert_eq!(res.max_turns, Some(64));
        assert!(res.degraded_reason.is_none());
    }

    /// preset（含写工具）+ 关闭：白名单保留角色（其中的写条目由 §5.4
    /// warn+drop 拦下）。
    #[test]
    fn preset_with_write_tools_denied_degrades() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Preset("tester")),
            AgentAutonomy::Auto,
            false,
            &[],
            &write_tools(),
        );
        assert!(res.tool_names.is_some(), "preset 角色保留");
        assert!(res.degraded_reason.is_some());
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
    }

    /// 只读 preset（researcher）：等价 read_only 的排除集语义。
    #[test]
    fn read_only_preset_forces_read_only_exclusion() {
        let res = resolve_capability(
            Some(&ChildToolCapability::Preset("researcher")),
            AgentAutonomy::Auto,
            true,
            &[],
            &write_tools(),
        );
        for t in WRITE_TOOLS {
            assert!(res.excluded_tools.contains(*t));
        }
        assert!(res.degraded_reason.is_none(), "只读请求无降级");
        assert_eq!(res.max_turns, Some(32));
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(ChildToolCapability::ReadOnly.label(), "read_only");
        assert_eq!(ChildToolCapability::Write.label(), "write");
        assert_eq!(ChildToolCapability::Preset("coder").label(), "preset:coder");
    }

    /// D3.1：nudge 三条件（per-child，依据解析器输出的排除集）。
    #[test]
    fn nudge_three_conditions() {
        let read_only_excluded: BTreeSet<String> =
            WRITE_TOOLS.iter().map(|s| s.to_string()).collect();
        let write_exempted: BTreeSet<String> = ["task_output".to_string()].into_iter().collect();

        // 条件一：Manual 强制。
        assert!(super::read_only_nudge(
            AgentAutonomy::Manual,
            false,
            &write_exempted,
            &write_tools()
        ));
        // 条件二：child_read_only 偏执开关（默认部署路径）。
        assert!(super::read_only_nudge(
            AgentAutonomy::Auto,
            true,
            &write_exempted,
            &write_tools()
        ));
        // 条件三：解析后排除集仍含全部 write_tools（子 agent 实际无写工具）。
        assert!(super::read_only_nudge(
            AgentAutonomy::Auto,
            false,
            &read_only_excluded,
            &write_tools()
        ));
        // 写子 agent（豁免后）：无 nudge。
        assert!(!super::read_only_nudge(
            AgentAutonomy::Auto,
            false,
            &write_exempted,
            &write_tools()
        ));
    }
}
