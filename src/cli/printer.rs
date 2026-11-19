use std::io::{self, Write};

use agent_base::{AgentResult, RuntimeEvent};

/// Prints [`RuntimeEvent`]s to an arbitrary [`Write`] sink (stdout by default).
///
/// The sink is boxed rather than generic so tests can plug in an in-memory
/// buffer without monomorphizing every call site.
#[allow(clippy::type_complexity)]
pub struct CliEventPrinter {
    pub assistant_prefix_printed: bool,
    pub custom_handlers: Vec<Box<dyn Fn(&RuntimeEvent) -> Option<String> + Send>>,
    sink: Box<dyn Write + Send>,
}

impl Default for CliEventPrinter {
    fn default() -> Self {
        Self::with_sink(io::stdout())
    }
}

impl CliEventPrinter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Route all output through `sink` instead of stdout.
    pub fn with_sink(sink: impl Write + Send + 'static) -> Self {
        Self {
            assistant_prefix_printed: false,
            custom_handlers: Vec::new(),
            sink: Box::new(sink),
        }
    }

    pub fn handle(&mut self, event: RuntimeEvent) -> AgentResult<()> {
        for handler in &self.custom_handlers {
            if let Some(output) = handler(&event) {
                self.finish();
                self.emit(&output)?;
                return Ok(());
            }
        }

        match event {
            RuntimeEvent::TextDelta { text, .. } => {
                if !self.assistant_prefix_printed {
                    self.emit("Assistant > ")?;
                    self.assistant_prefix_printed = true;
                }
                self.emit(&text)?;
                self.flush_sink()?;
            }
            RuntimeEvent::ThoughtDelta { text, .. } => {
                self.emit(&format!("\x1b[90m{text}\x1b[0m"))?;
                self.flush_sink()?;
            }
            RuntimeEvent::ActionStarted {
                tool_name,
                args_json,
                ..
            } => {
                self.finish();
                self.emit_line(&format!("[Tool Start] {tool_name} {args_json}"))?;
            }
            RuntimeEvent::ActionFinished {
                tool_name, summary, ..
            } => {
                self.finish();
                self.emit_line(&format!("[Tool Done] {tool_name}"))?;
                for line in summary.lines() {
                    self.emit_line(&format!("  {line}"))?;
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
                self.emit_line(&format!("[Plan] {objective}"))?;
                for (i, item) in plan.iter().enumerate() {
                    let status_icon = match item.status {
                        agent_base::PlanStepStatus::Completed => "✅",
                        agent_base::PlanStepStatus::InProgress => "🔄",
                        agent_base::PlanStepStatus::Pending => "⏳",
                    };
                    self.emit_line(&format!("  {status_icon} {}. {}", i + 1, item.step))?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn finish(&mut self) {
        if self.assistant_prefix_printed {
            let _ = writeln!(self.sink);
            self.assistant_prefix_printed = false;
        }
    }

    fn emit(&mut self, fragment: &str) -> AgentResult<()> {
        self.sink
            .write_all(fragment.as_bytes())
            .map_err(|e| agent_base::AgentError::internal(format!("write failed: {e}")))
    }

    fn emit_line(&mut self, line: &str) -> AgentResult<()> {
        self.emit(line)?;
        self.emit("\n")
    }

    fn flush_sink(&mut self) -> AgentResult<()> {
        self.sink
            .flush()
            .map_err(|e| agent_base::AgentError::internal(format!("flush failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_base::{PlanItem, SessionId};

    /// A `Write` that records everything, so assertions can read it back.
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct SharedTape(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedTape {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn shared_printer() -> (CliEventPrinter, SharedTape) {
        let tape = SharedTape(Arc::new(Mutex::new(Vec::new())));
        (CliEventPrinter::with_sink(tape.clone()), tape)
    }

    fn read_back(tape: &SharedTape) -> String {
        String::from_utf8(tape.0.lock().unwrap().clone()).unwrap()
    }

    fn text_delta(text: &str) -> RuntimeEvent {
        RuntimeEvent::TextDelta {
            session_id: SessionId::new(0),
            text: text.to_string(),
            agent_id: None,
            trace_id: None,
        }
    }

    #[test]
    fn consecutive_text_deltas_share_one_assistant_prefix() {
        let (mut p, tape) = shared_printer();
        p.handle(text_delta("hello")).unwrap();
        p.handle(text_delta(" world")).unwrap();
        assert_eq!(read_back(&tape), "Assistant > hello world");
    }

    #[test]
    fn thought_delta_is_wrapped_in_dim_ansi() {
        let (mut p, tape) = shared_printer();
        p.handle(RuntimeEvent::ThoughtDelta {
            session_id: SessionId::new(0),
            text: "thought delta".to_string(),
            agent_id: None,
            trace_id: None,
        })
        .unwrap();
        assert_eq!(read_back(&tape), "\x1b[90mthought delta\x1b[0m");
    }

    #[test]
    fn action_started_prints_name_and_args_on_one_line() {
        let (mut p, tape) = shared_printer();
        p.handle(RuntimeEvent::ActionStarted {
            session_id: SessionId::new(0),
            tool_name: "tool_started".to_string(),
            args_json: "tool_args".to_string(),
            agent_id: None,
            trace_id: None,
        })
        .unwrap();
        assert_eq!(read_back(&tape), "[Tool Start] tool_started tool_args\n");
    }

    #[test]
    fn action_finished_indents_every_summary_line() {
        let (mut p, tape) = shared_printer();
        p.handle(RuntimeEvent::ActionFinished {
            session_id: SessionId::new(0),
            tool_name: "tool_name".to_string(),
            summary: "tool call finished\nthis is a test\n".to_string(),
            agent_id: None,
            trace_id: None,
            denied: false,
            details: None,
        })
        .unwrap();
        let out = read_back(&tape);
        assert!(out.starts_with("[Tool Done] tool_name"));
        assert!(out.lines().any(|l| l == "  tool call finished"));
        assert!(out.lines().any(|l| l == "  this is a test"));
    }

    #[test]
    fn plan_updated_renders_status_icons_in_order() {
        let (mut p, tape) = shared_printer();
        p.handle(RuntimeEvent::PlanUpdated {
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
        })
        .unwrap();
        let out = read_back(&tape);
        assert!(out.starts_with("[Plan] object"));
        assert!(out.lines().any(|l| l == "  ✅ 1. step_a"));
        assert!(out.lines().any(|l| l == "  🔄 2. step_b"));
        assert!(out.lines().any(|l| l == "  ⏳ 3. step_c"));
    }

    #[test]
    fn finish_emits_one_newline_and_clears_the_prefix_flag() {
        let (mut p, tape) = shared_printer();
        p.assistant_prefix_printed = true;
        p.finish();
        assert!(!p.assistant_prefix_printed);
        assert_eq!(read_back(&tape), "\n");
    }

    #[test]
    fn a_matching_custom_handler_short_circuits_the_default_path() {
        let (mut p, tape) = shared_printer();
        p.custom_handlers = vec![Box::new(|event| match event {
            RuntimeEvent::TextDelta { text, .. } => Some(format!("Custom handler {}", text)),
            _ => None,
        })];
        p.handle(text_delta("mock string")).unwrap();
        assert_eq!(read_back(&tape), "Custom handler mock string");
    }

    #[test]
    fn new_and_default_start_with_a_clean_slate() {
        let default = CliEventPrinter::default();
        assert!(!default.assistant_prefix_printed);
        assert!(default.custom_handlers.is_empty());

        let new = CliEventPrinter::new();
        assert!(!new.assistant_prefix_printed);
    }

    #[test]
    fn a_handler_returning_none_falls_through_to_normal_rendering() {
        let (mut p, tape) = shared_printer();
        p.custom_handlers = vec![Box::new(|_event| None)];
        p.handle(text_delta("falls through")).unwrap();
        assert_eq!(read_back(&tape), "Assistant > falls through");
    }

    #[test]
    fn approval_and_run_finished_close_the_line_without_extra_output() {
        let (mut p, tape) = shared_printer();
        p.assistant_prefix_printed = true;

        p.handle(RuntimeEvent::AwaitingApproval {
            session_id: SessionId::new(0),
            request: agent_base::ApprovalRequest {
                title: "t".to_string(),
                message: "m".to_string(),
                action_key: None,
                risk_level: agent_base::RiskLevel::Safe,
                raw: None,
                source: None,
            },
            agent_id: None,
            trace_id: None,
        })
        .unwrap();
        assert!(!p.assistant_prefix_printed);

        // RunFinished with no prefix printed is a no-op.
        p.handle(RuntimeEvent::RunFinished {
            session_id: SessionId::new(0),
            agent_id: None,
            trace_id: None,
        })
        .unwrap();
        assert_eq!(read_back(&tape), "\n");
    }
}
