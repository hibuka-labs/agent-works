use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use agent_base::{AgentError, AgentResult, AgentRuntime, SessionId};

#[allow(clippy::type_complexity)]
pub struct CliRepl {
    runtime: AgentRuntime,
    session_id: Option<SessionId>,
    shell_commands: HashMap<String, Box<dyn Fn(&str) -> bool + Send + Sync>>,
}

impl CliRepl {
    pub fn new(runtime: AgentRuntime) -> Self {
        Self {
            runtime,
            session_id: None,
            shell_commands: HashMap::new(),
        }
    }

    pub fn register_shell_command(
        &mut self,
        prefix: &str,
        handler: Box<dyn Fn(&str) -> bool + Send + Sync>,
    ) {
        self.shell_commands.insert(prefix.to_string(), handler);
    }

    pub async fn run(&mut self) -> AgentResult<()> {
        if self.session_id.is_none() {
            self.session_id = Some(self.runtime.create_session().await);
        }

        println!(">>> Agent REPL started");
        println!(">>> Commands: .help, .reset, .exit");

        loop {
            print!("\nUser > ");
            io::stdout()
                .flush()
                .map_err(|e| AgentError::internal(format!("flush: {e}")))?;

            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .map_err(|e| AgentError::internal(format!("read stdin: {e}")))?;
            let input = input.trim();

            if input.is_empty() {
                continue;
            }

            if let Some(cmd) = input.strip_prefix('.') {
                match cmd {
                    "exit" | "quit" => break,
                    "reset" => {
                        self.session_id = Some(self.runtime.create_session().await);
                        println!(">>> Session reset");
                        continue;
                    }
                    "help" => {
                        println!(">>> .exit - quit");
                        println!(">>> .reset - new session");
                        println!(">>> .help - this message");
                        continue;
                    }
                    _ => {
                        let (prefix, _) = cmd.split_once(' ').unwrap_or((cmd, ""));
                        if let Some(handler) = self.shell_commands.get(prefix)
                            && handler(cmd)
                        {
                            continue;
                        }
                        println!(">>> Unknown command: .{cmd}");
                        continue;
                    }
                }
            }

            let sid = self.session_id.clone().expect("session not initialized");
            let printer = Arc::new(Mutex::new(super::printer::CliEventPrinter::new()));
            let printer_clone = printer.clone();
            match self
                .runtime
                .run_turn(sid, input, move |event| {
                    printer_clone.lock().unwrap().handle(event)
                })
                .await
            {
                Ok(_) => printer.lock().unwrap().finish(),
                Err(err) => {
                    printer.lock().unwrap().finish();
                    eprintln!("[Error] {err}");
                }
            }
        }

        Ok(())
    }
}
