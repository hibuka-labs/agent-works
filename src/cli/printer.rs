use std::io::{self, Write};

use agent_base::{AgentResult, RuntimeEvent};

#[allow(clippy::type_complexity)]
pub struct CliEventPrinter<W: Write = io::Stdout> {
    pub assistant_prefix_printed: bool,
    pub custom_handlers: Vec<Box<dyn Fn(&RuntimeEvent) -> Option<String> + Send>>,
    pub write: W,
}

impl Default for CliEventPrinter {
    fn default() -> Self {
        Self {
            assistant_prefix_printed: false,
            custom_handlers: Vec::new(),
            write: io::stdout(),
        }
    }
}
impl CliEventPrinter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<W: Write> CliEventPrinter<W> {
    pub fn with_writer(write: W) -> Self {
        Self {
            assistant_prefix_printed: false,
            custom_handlers: Vec::new(),
            write,
        }
    }

    pub fn handle(&mut self, event: RuntimeEvent) -> AgentResult<()> {
        for handler in &self.custom_handlers {
            if let Some(output) = handler(&event) {
                self.finish();
                write!(self.write, "{output}")
                    .map_err(|e| agent_base::AgentError::internal(format!("write failed: {e}")))?;
                self.write.flush().map_err(|e| {
                    agent_base::AgentError::internal(format!("flush stdout failed: {e}"))
                })?;
                return Ok(());
            }
        }

        match event {
            RuntimeEvent::TextDelta { text, .. } => {
                if !self.assistant_prefix_printed {
                    write!(self.write, "Assistant > ").map_err(|e| {
                        agent_base::AgentError::internal(format!("write failed: {e}"))
                    })?;
                    self.assistant_prefix_printed = true;
                }
                write!(self.write, "{text}")
                    .map_err(|e| agent_base::AgentError::internal(format!("write failed: {e}")))?;
                self.write.flush().map_err(|e| {
                    agent_base::AgentError::internal(format!("flush stdout failed: {e}"))
                })?;
            }
            RuntimeEvent::ThoughtDelta { text, .. } => {
                write!(self.write, "\x1b[90m{text}\x1b[0m")
                    .map_err(|e| agent_base::AgentError::internal(format!("write failed: {e}")))?;
                self.write.flush().map_err(|e| {
                    agent_base::AgentError::internal(format!("flush stdout failed: {e}"))
                })?;
            }
            RuntimeEvent::ToolCallStarted {
                tool_name,
                args_json,
                ..
            } => {
                self.finish();
                writeln!(self.write, "[Tool Start] {tool_name} {args_json}")
                    .map_err(|e| agent_base::AgentError::internal(format!("write failed: {e}")))?;
            }
            RuntimeEvent::ToolCallFinished {
                tool_name, summary, ..
            } => {
                self.finish();
                writeln!(self.write, "[Tool Done] {tool_name}")
                    .map_err(|e| agent_base::AgentError::internal(format!("write failed: {e}")))?;
                for line in summary.lines() {
                    writeln!(self.write, "  {line}").map_err(|e| {
                        agent_base::AgentError::internal(format!("write failed: {e}"))
                    })?;
                }
            }
            RuntimeEvent::AwaitingApproval { .. } => {
                self.finish();
            }
            RuntimeEvent::RunFinished { .. } => {
                self.finish();
            }
            RuntimeEvent::PlanUpdated {
                objective, plan, ..
            } => {
                self.finish();
                writeln!(self.write, "[Plan] {objective}")
                    .map_err(|e| agent_base::AgentError::internal(format!("write failed: {e}")))?;
                for (i, item) in plan.iter().enumerate() {
                    let status_icon = match item.status {
                        agent_base::PlanStepStatus::Completed => "✅",
                        agent_base::PlanStepStatus::InProgress => "🔄",
                        agent_base::PlanStepStatus::Pending => "⏳",
                    };
                    writeln!(self.write, "  {status_icon} {}. {}", i + 1, item.step).map_err(
                        |e| agent_base::AgentError::internal(format!("write failed: {e}")),
                    )?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn finish(&mut self) {
        if self.assistant_prefix_printed {
            let _ = writeln!(self.write);
            self.assistant_prefix_printed = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::{PlanItem, SessionId};

    #[test]
    fn test_text_delta() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        let event1 = RuntimeEvent::TextDelta {
            session_id: SessionId::new(0),
            text: "hello".to_string(),
            agent_id: None,
            trace_id: None,
        };

        assert!(cep.handle(event1).is_ok());
        let event2 = RuntimeEvent::TextDelta {
            session_id: SessionId::new(0),
            text: " world".to_string(),
            agent_id: None,
            trace_id: None,
        };
        assert!(cep.handle(event2).is_ok());

        let output = String::from_utf8(cep.write).unwrap();
        assert_eq!(output, "Assistant > hello world");
    }
    #[test]
    fn test_thought_delta() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        let event = RuntimeEvent::ThoughtDelta {
            session_id: SessionId::new(0),
            text: "thought delta".to_string(),
            agent_id: None,
            trace_id: None,
        };
        let agent_result = cep.handle(event);
        assert!(agent_result.is_ok());
        let output = String::from_utf8(cep.write).unwrap();
        assert_eq!(output, "\x1b[90mthought delta\x1b[0m");
    }
    #[test]
    fn test_tool_call_started() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        let event = RuntimeEvent::ToolCallStarted {
            session_id: SessionId::new(0),
            tool_name: "tool_started".to_string(),
            args_json: "tool_args".to_string(),
            agent_id: None,
            trace_id: None,
        };

        let agent_result = cep.handle(event);
        assert!(agent_result.is_ok());

        let output = String::from_utf8(cep.write).unwrap();
        assert_eq!(output, "[Tool Start] tool_started tool_args\n");
    }
    #[test]
    fn test_tool_call_finished() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        let event = RuntimeEvent::ToolCallFinished {
            session_id: SessionId::new(0),
            tool_name: "tool_name".to_string(),
            summary: "tool call finished\nthis is a test\n".to_string(),
            agent_id: None,
            trace_id: None,
            denied: false,
                            details: None,
        };
        let agent_result = cep.handle(event);
        assert!(agent_result.is_ok());
        let output = String::from_utf8(cep.write).unwrap();
        assert!(
            output.starts_with("[Tool Done] tool_name")
                && output.lines().any(|line| line == "  tool call finished")
                && output.lines().any(|line| line == "  this is a test")
        );
    }
    #[test]
    fn test_plan_updated() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        let event = RuntimeEvent::PlanUpdated {
            session_id: SessionId::new(0),
            objective: "object".to_string(),
            explanation: None,
            plan: vec![
                PlanItem {
                    step: "step_a".to_string(),
                    status: agent_base::PlanStepStatus::Completed,
                },
                PlanItem {
                    step: "step_b".to_string(),
                    status: agent_base::PlanStepStatus::InProgress,
                },
                PlanItem {
                    step: "step_c".to_string(),
                    status: agent_base::PlanStepStatus::Pending,
                },
            ],
            agent_id: None,
            trace_id: None,
        };
        let agent_result = cep.handle(event);
        assert!(agent_result.is_ok());
        let output = String::from_utf8(cep.write).unwrap();
        assert!(
            output.starts_with("[Plan] object")
                && output.lines().any(|line| line == "  ✅ 1. step_a")
                && output.lines().any(|line| line == "  🔄 2. step_b")
                && output.lines().any(|line| line == "  ⏳ 3. step_c")
        );
    }

    #[test]
    fn test_finish() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        cep.assistant_prefix_printed = true;
        cep.finish();
        assert!(!cep.assistant_prefix_printed);
        assert_eq!(String::from_utf8(cep.write).unwrap(), "\n");
    }

    #[test]
    fn test_custom_handlers() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        cep.custom_handlers = vec![Box::new(|event| match event {
            RuntimeEvent::TextDelta { text, .. } => Some(format!("Custom handler {}", text)),
            _ => None,
        })];
        let event = RuntimeEvent::TextDelta {
            session_id: SessionId::new(0),
            text: "mock string".to_string(),
            agent_id: None,
            trace_id: None,
        };
        let agent_result = cep.handle(event);
        assert!(agent_result.is_ok());
        let output = String::from_utf8(cep.write).unwrap();
        assert_eq!(output, "Custom handler mock string");
    }

    #[test]
    fn test_new_and_default() {
        let default = CliEventPrinter::default();
        assert!(!default.assistant_prefix_printed);
        assert!(default.custom_handlers.is_empty());

        let new = CliEventPrinter::new();
        assert!(!new.assistant_prefix_printed);
    }

    #[test]
    fn test_custom_handler_none_falls_through() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        cep.custom_handlers = vec![Box::new(|_event| None)];
        let event = RuntimeEvent::TextDelta {
            session_id: SessionId::new(0),
            text: "falls through".to_string(),
            agent_id: None,
            trace_id: None,
        };
        assert!(cep.handle(event).is_ok());
        assert_eq!(
            String::from_utf8(cep.write).unwrap(),
            "Assistant > falls through"
        );
    }

    #[test]
    fn test_awaiting_approval_and_run_finished() {
        let mut cep = CliEventPrinter::with_writer(Vec::new());
        cep.assistant_prefix_printed = true;

        let awaiting = RuntimeEvent::AwaitingApproval {
            session_id: SessionId::new(0),
            request: agent_base::ApprovalRequest {
                title: "t".to_string(),
                message: "m".to_string(),
                action_key: None,
                risk_level: agent_base::RiskLevel::Safe,
                raw: None,
            },
            agent_id: None,
            trace_id: None,
        };
        assert!(cep.handle(awaiting).is_ok());
        assert!(!cep.assistant_prefix_printed);

        // RunFinished with no prefix printed is a no-op.
        let finished = RuntimeEvent::RunFinished {
            session_id: SessionId::new(0),
            agent_id: None,
            trace_id: None,
        };
        assert!(cep.handle(finished).is_ok());
    }
}
