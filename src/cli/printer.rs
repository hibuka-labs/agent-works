use std::io::{self, Write};

use agent_base::{RuntimeEvent, AgentResult};

pub struct CliEventPrinter {
    pub assistant_prefix_printed: bool,
    pub custom_handlers: Vec<Box<dyn Fn(&RuntimeEvent) -> Option<String> + Send>>,
}

impl Default for CliEventPrinter {
    fn default() -> Self {
        Self {
            assistant_prefix_printed: false,
            custom_handlers: Vec::new(),
        }
    }
}

impl CliEventPrinter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn handle(&mut self, event: RuntimeEvent) -> AgentResult<()> {
        for handler in &self.custom_handlers {
            if let Some(output) = handler(&event) {
                self.finish();
                print!("{output}");
                io::stdout()
                    .flush()
                    .map_err(|e| agent_base::AgentError::internal(format!("flush stdout failed: {e}")))?;
                return Ok(());
            }
        }

        match event {
            RuntimeEvent::TextDelta { text, .. } => {
                if !self.assistant_prefix_printed {
                    print!("Assistant > ");
                    self.assistant_prefix_printed = true;
                }
                print!("{text}");
                io::stdout()
                    .flush()
                    .map_err(|e| agent_base::AgentError::internal(format!("flush stdout failed: {e}")))?;
            }
            RuntimeEvent::ThoughtDelta { text, .. } => {
                print!("\x1b[90m{text}\x1b[0m");
                io::stdout()
                    .flush()
                    .map_err(|e| agent_base::AgentError::internal(format!("flush stdout failed: {e}")))?;
            }
            RuntimeEvent::ToolCallStarted {
                tool_name,
                args_json,
                ..
            } => {
                self.finish();
                println!("[Tool Start] {tool_name} {args_json}");
            }
            RuntimeEvent::ToolCallFinished {
                tool_name, summary, ..
            } => {
                self.finish();
                println!("[Tool Done] {tool_name}");
                for line in summary.lines() {
                    println!("  {line}");
                }
            }
            RuntimeEvent::AwaitingApproval { .. } => {
                self.finish();
            }
            RuntimeEvent::RunFinished { .. } => {
                self.finish();
            }
            RuntimeEvent::PlanGenerated { plan, .. } => {
                self.finish();
                println!("[Plan Generated] id={}", plan.id);
                for phase in &plan.phases {
                    println!("  Phase: {}", phase.title);
                    for (i, step) in phase.steps.iter().enumerate() {
                        println!("    {}. {}", i + 1, step.description);
                    }
                }
            }
            RuntimeEvent::PlanStepStarted {
                step_id,
                step_description,
                ..
            } => {
                self.finish();
                println!("[Plan Step Start] {step_id} - {step_description}");
            }
            RuntimeEvent::PlanStepCompleted {
                step_id,
                success,
                result,
                ..
            } => {
                self.finish();
                println!(
                    "[Plan Step Done] {step_id} ({})",
                    if success { "success" } else { "failed" }
                );
                if let Some(result) = result {
                    for line in result.lines() {
                        println!("  {line}");
                    }
                }
            }
            RuntimeEvent::PlanCompleted {
                plan_id, success, ..
            } => {
                self.finish();
                println!(
                    "[Plan Done] {plan_id} ({})",
                    if success { "success" } else { "failed" }
                );
            }
            RuntimeEvent::PlanFailed {
                plan_id, error, ..
            } => {
                self.finish();
                println!("[Plan Failed] {plan_id} - {error}");
            }
            _ => {}
        }
        Ok(())
    }

    pub fn finish(&mut self) {
        if self.assistant_prefix_printed {
            println!();
            self.assistant_prefix_printed = false;
        }
    }
}
