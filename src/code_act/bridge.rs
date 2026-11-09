use std::sync::{Arc, Mutex};

use agent_base::{ApprovalDecision, ApprovalRequest, AskAnswer, AskOption, ExecCtx, RiskLevel};
use async_trait::async_trait;
use codexec::{ConsentRequest, InteractionBridge, PolicyDecision};
use phi_cap::Capability;
use serde_json::Value;

/// The cell-interaction bridge (T10b, OV-6): codexec's [`InteractionBridge`]
/// over the run's [`ExecCtx`]. The bridge outlives any one step (the engine
/// holds it from assembly) while the context arrives with `execute` — so it
/// keeps an owned handle for the step and rebuilds the context per call, which
/// keeps the suspension discipline (answer / cancel / watchdog) in exactly one
/// place: `ExecCtx`'s own methods.
#[derive(Default)]
pub struct ExecCtxBridge {
    armed: Mutex<Option<ExecCtx>>,
}

impl ExecCtxBridge {
    /// An un-armed bridge: nothing suspends until [`Self::arm`] runs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Arm with the step's run context (a handle clone — the step owns the
    /// original).
    pub fn arm(&self, ctx: &ExecCtx) {
        *self.armed.lock().unwrap() = Some(clone_ctx(ctx));
    }

    /// Disarm at the end of the step.
    pub fn disarm(&self) {
        *self.armed.lock().unwrap() = None;
    }

    /// The armed handle — or why the suspension cannot happen (which fails
    /// closed on the engine side).
    fn current(&self) -> Result<ExecCtx, String> {
        self.armed
            .lock()
            .unwrap()
            .as_ref()
            .map(clone_ctx)
            .ok_or_else(|| "no run context armed (interaction outside a step)".to_string())
    }
}

/// `ExecCtx` is a handle (token + channel + deadline), so cloning the handle
/// is as good as the original. (Struct literal: those fields are the type.)
fn clone_ctx(ctx: &ExecCtx) -> ExecCtx {
    ExecCtx {
        cancel: ctx.cancel.clone(),
        interact: Arc::clone(&ctx.interact),
        interact_timeout: ctx.interact_timeout,
    }
}

#[async_trait]
impl InteractionBridge for ExecCtxBridge {
    async fn consent(&self, req: &ConsentRequest<'_>) -> Result<PolicyDecision, String> {
        let ctx = self.current()?;
        let request = ApprovalRequest {
            title: req.fn_name.to_string(),
            message: consent_message(req),
            action_key: Some(req.fn_name.to_string()),
            risk_level: risk_of(req.cap),
            raw: req.args.cloned(),
            source: None,
        };
        match ctx.request_consent(request).await {
            Ok(ApprovalDecision::AllowOnce) | Ok(ApprovalDecision::AllowAlways) => {
                Ok(PolicyDecision::Allow)
            }
            Ok(ApprovalDecision::Deny) => Ok(PolicyDecision::Deny),
            // Cancel / watchdog / a dead channel: fail closed, and the reason
            // ("interaction cancelled") rides the hold the model sees.
            Err(e) => Err(e.to_string()),
        }
    }

    async fn ask(&self, question: &str, options: &[Value]) -> Result<Value, String> {
        let ctx = self.current()?;
        let options = options
            .iter()
            .enumerate()
            .map(|(i, o)| ask_option(o, i))
            .collect::<Result<Vec<_>, _>>()?;
        match ctx.ask_user(question, &options).await {
            Ok(AskAnswer::Picked { id, label }) => {
                Ok(serde_json::json!({"picked": {"id": id, "label": label}}))
            }
            Ok(AskAnswer::Skipped) => Ok(serde_json::json!({"skipped": true})),
            // The "Chat about this" pseudo-row: the user's question comes back
            // verbatim under `clarify` (wire shape — the cell reads it as the
            // ask.user return value).
            Ok(AskAnswer::Clarify { text }) => Ok(serde_json::json!({"clarify": text})),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// What the human sees on the consent dialog: the call and its payload (the
/// digest only when the chokepoint holds nothing else — the escape hatch).
fn consent_message(req: &ConsentRequest<'_>) -> String {
    match req.args {
        Some(wire) => format!("{} {wire}", req.fn_name),
        None => format!("{} (args digest {})", req.fn_name, req.args_digest),
    }
}

/// Risk for the approval dialog: reads are safe, anything that writes or
/// leaves the box is sensitive — a cell is code, so sensitive and never
/// destructive (the product's approval face owns the final wording).
fn risk_of(cap: Capability) -> RiskLevel {
    match cap {
        Capability::Pure | Capability::ReadFs => RiskLevel::Safe,
        _ => RiskLevel::Sensitive,
    }
}

/// Wire `{"id", "label", "note"?}` → the ask vocabulary. A bad shape fails
/// the interaction instead of guessing.
fn ask_option(wire: &Value, i: usize) -> Result<AskOption, String> {
    let obj = wire
        .as_object()
        .ok_or_else(|| format!("ask.user option {i} must be an object"))?;
    let field = |k: &str| {
        obj.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("ask.user option {i} needs a string `{k}`"))
    };
    Ok(AskOption {
        id: field("id")?,
        label: field("label")?,
        note: obj.get("note").and_then(Value::as_str).map(str::to_string),
    })
}
