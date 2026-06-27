use std::collections::HashMap;
use std::sync::Arc;

use agent_base::{
    AgentError, AgentResult, PlanStore, Tool, ToolContext, ToolControlFlow, ToolOutput,
};
use async_trait::async_trait;
use serde_json::{json, Value};

use super::registry::SkillRegistry;

/// Tool that lets the LLM invoke a template skill, generating an ExecutionPlan.
///
/// The LLM calls `apply_skill(skill_name, params)` and receives a plan_id
/// it can then execute via the existing `execute_plan` tool.
pub struct ApplySkillTool {
    registry: Arc<SkillRegistry>,
    plan_store: Arc<dyn PlanStore>,
}

impl ApplySkillTool {
    pub fn new(registry: Arc<SkillRegistry>, plan_store: Arc<dyn PlanStore>) -> Self {
        Self {
            registry,
            plan_store,
        }
    }
}

#[async_trait]
impl Tool for ApplySkillTool {
    fn name(&self) -> &'static str {
        "apply_skill"
    }

    fn definition(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "apply_skill",
                "description": "应用一个预定义的运维技能模板，生成执行计划。使用前建议先调用 get_skill_detail 查看详细说明。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "技能名称"
                        },
                        "params": {
                            "type": "object",
                            "description": "技能参数，键值对形式，如 {\"target_host\": \"prod-1\", \"service_name\": \"nginx\"}"
                        }
                    },
                    "required": ["skill_name"]
                }
            }
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<ToolOutput> {
        let skill_name = args
            .get("skill_name")
            .and_then(Value::as_str)
            .unwrap_or("");

        if skill_name.is_empty() {
            return Ok(ToolOutput {
                summary: "请提供 skill_name 参数".to_string(),
                raw: None,
                control_flow: ToolControlFlow::Break,
                truncation: None,
            });
        }

        // Parse params from the args — convert all JSON values to strings
        let params: HashMap<String, String> = args
            .get("params")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| {
                        let s = match v {
                            Value::String(s) => s.clone(),
                            Value::Null => String::new(),
                            _ => v.to_string(), // numbers, bools, arrays, objects → JSON text
                        };
                        (k.clone(), s)
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Apply the skill — generates ExecutionPlan
        let plan = match self.registry.apply(skill_name, &params).await {
            Ok(Some(plan)) => plan,
            Ok(None) => {
                // List available skills for a helpful error
                let available: Vec<String> = self
                    .registry
                    .list()
                    .await
                    .into_iter()
                    .filter(|s| s.has_plan)
                    .map(|s| s.name)
                    .collect();

                return Ok(ToolOutput {
                    summary: format!(
                        "技能 '{}' 不存在或不是模板型技能。可用的模板型技能: {}",
                        skill_name,
                        if available.is_empty() {
                            "(无)".to_string()
                        } else {
                            available.join(", ")
                        }
                    ),
                    raw: None,
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                });
            }
            Err(e) => {
                return Ok(ToolOutput {
                    summary: format!("应用技能 '{}' 失败: {}", skill_name, e),
                    raw: None,
                    control_flow: ToolControlFlow::Break,
                    truncation: None,
                });
            }
        };

        let plan_id = plan.id.clone();
        let steps_total = plan.total_steps();
        let steps_summary: Vec<String> = plan
            .all_steps()
            .map(|s| format!("- {}", s.description))
            .collect();

        // Save to PlanStore so execute_plan can pick it up
        self.plan_store
            .save_plan(&plan, json!({"source": "skill", "skill_name": skill_name}))
            .await
            .map_err(|e| {
                AgentError::internal(format!("Failed to store plan from skill '{}': {}", skill_name, e))
            })?;

        let summary = format!(
            "已从技能 '{}' 生成执行计划 (plan_id: {}, {} 个步骤):\n{}",
            skill_name,
            plan_id,
            steps_total,
            steps_summary.join("\n")
        );

        tracing::info!(
            skill_name = skill_name,
            plan_id = plan_id,
            steps = steps_total,
            "apply_skill: plan generated"
        );

        Ok(ToolOutput {
            summary,
            raw: Some(json!({
                "plan_id": plan_id,
                "skill_name": skill_name,
                "steps": steps_total,
            })),
            control_flow: ToolControlFlow::Break,
            truncation: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::{InMemoryPlanStore, PlanStep};
    use crate::skill::{Skill, SkillParam, SkillParamType};
    use std::sync::Arc;

    struct TemplateSkill {
        params: Vec<SkillParam>,
    }

    impl Skill for TemplateSkill {
        fn name(&self) -> &'static str {
            "test-template"
        }
        fn brief_description(&self) -> String {
            "Test template skill".to_string()
        }
        fn detailed_description(&self) -> String {
            "# Test\nA test template skill.".to_string()
        }
        fn tools(&self) -> Vec<Arc<dyn agent_base::Tool>> {
            vec![]
        }
        fn plan_steps(
            &self,
            params: &HashMap<String, String>,
        ) -> Option<Vec<PlanStep>> {
            let host = params.get("target_host").cloned().unwrap_or_default();
            Some(vec![PlanStep::tool_call(
                "step-1",
                "Check disk",
                "execute_command",
                json!({"command": "df -h", "target_host": host}),
            )])
        }
        fn parameters(&self) -> &[SkillParam] {
            &self.params
        }
    }

    #[tokio::test]
    async fn test_apply_skill_tool_success() {
        let registry = Arc::new(SkillRegistry::new());
        let skill = Arc::new(TemplateSkill {
            params: vec![SkillParam {
                name: "target_host".to_string(),
                description: "Target host".to_string(),
                param_type: SkillParamType::HostRef,
                required: true,
                default: None,
            }],
        });
        registry.register(skill).await;

        let plan_store = Arc::new(InMemoryPlanStore::new());
        let tool = ApplySkillTool::new(registry, plan_store.clone());

        let args = json!({
            "skill_name": "test-template",
            "params": {"target_host": "prod-1"}
        });

        let result = tool.call(&args, &dummy_ctx()).await.unwrap();
        assert!(result.summary.contains("plan_id"));
        assert!(result.summary.contains("Check disk"));

        // Verify plan was saved
        let raw = result.raw.unwrap();
        let plan_id = raw["plan_id"].as_str().unwrap();
        let saved = plan_store.load_plan(plan_id).await.unwrap();
        assert!(saved.is_some());
        assert_eq!(saved.unwrap().plan.total_steps(), 1);
    }

    #[tokio::test]
    async fn test_apply_skill_tool_missing_skill() {
        let registry = Arc::new(SkillRegistry::new());
        let plan_store = Arc::new(InMemoryPlanStore::new());
        let tool = ApplySkillTool::new(registry, plan_store);

        let args = json!({"skill_name": "nonexistent"});
        let result = tool.call(&args, &dummy_ctx()).await.unwrap();
        assert!(result.summary.contains("不存在"));
    }

    fn dummy_ctx() -> ToolContext {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ToolContext {
            session_id: agent_base::SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: agent_base::Language::Zh,
            cancel_token: tokio_util::sync::CancellationToken::new(),
        }
    }
}
