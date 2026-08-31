//! Child runtime construction (B split from `runtime.rs`).
//!
//! Assembly of a child `AgentRuntime` from a [`ChildConfig`]:
//! the read-only nudge, the permission triple (prompt note / tool policy /
//! approval handler), the "先排除,再白名单" tool pipeline with post-exclusion
//! validation (§5.4), and the usage hook (§7.2). The permission-resolution
//! helpers (`spawn_permission` et al., §7.5) live here because they are the
//! build path's input; the spawn chain calls them via
//! [`build_child_runtime_with_config`](MultiAgentRuntime::build_child_runtime_with_config).

use super::*;

impl MultiAgentRuntime {
    /// Legacy entry point used by the positional spawn path: registers every
    /// non-excluded business tool, no whitelist, no turn/context overrides.
    /// Retained for the existing unit tests' 2-arg call sites; the live
    /// spawn path uses [`build_child_runtime_with_config`](Self::build_child_runtime_with_config)
    /// directly (behavior byte-identical for `tool_names = None`).
    #[cfg(test)]
    pub(super) async fn build_child_runtime(
        &self,
        system_prompt: String,
        full_permission: bool,
    ) -> AgentResult<AgentRuntime> {
        let config = ChildConfig {
            system_prompt: Some(system_prompt),
            full_permission: Some(full_permission),
            ..Default::default()
        };
        let (runtime, _spawned) = self
            .build_child_runtime_with_config(&config, full_permission)
            .await?;
        Ok(runtime)
    }

    /// Build a child runtime from a [`ChildConfig`] (design §5.4).
    ///
    /// Returns the runtime plus the **actually registered** tool names (the
    /// "echo" the spawn output carries so the parent can see what the child
    /// really got — §5.4 review M-3). `full_permission` is already resolved
    /// by the caller via [`effective_permission`](Self::effective_permission).
    pub(super) async fn build_child_runtime_with_config(
        &self,
        config: &ChildConfig,
        full_permission: bool,
    ) -> AgentResult<(AgentRuntime, BTreeSet<String>)> {
        let system_prompt = config.system_prompt.clone().unwrap_or_default();

        // Read-only nudge (framework layer). The framework cannot classify which
        // business tools mutate state, so this is a *suggestion* only — it does
        // not gate any tool. The business layer hard-gates mutating tools via
        // `child_excluded_tools` when it needs a guarantee. This is orthogonal to
        // `full_permission` (approval vs. deny): even a full-permission child is
        // nudged not to write when `child_read_only` is set. Under `Manual`
        // (§7.5) the nudge is forced on regardless — belt and braces over the
        // hard layers (exclusion + approval floor), never a substitute.
        let system_prompt = if self.effective_read_only_nudge() {
            format!(
                "{}\n\nYou are a read-only sub-agent: investigate, analyze, and report your findings in your final answer. Do not modify the workspace, mutate state, or run side-effecting commands — the parent agent owns all changes and will apply them based on your report.",
                system_prompt
            )
        } else {
            system_prompt
        };

        let (prompt, policy, approval): (
            String,
            Option<Arc<dyn ToolPolicy>>,
            Arc<dyn ApprovalHandler>,
        ) = if full_permission {
            // Full: no tool policy → every tool auto-approves (= current behaviour).
            (system_prompt, None, Arc::new(AllowAllApprovalHandler))
        } else {
            // None: policy = parent's (if any) or DenyAllToolPolicy fallback.
            let note = "If a tool call is rejected for lack of permission, explain in your final answer that you lacked permission for that action.";
            let policy: Arc<dyn ToolPolicy> = match &self.tool_policy {
                Some(p) => p.clone(),
                None => Arc::new(DenyAllToolPolicy),
            };
            // Codex-style delegation: rather than hard-denying the child's own
            // approval requests, route the decision up to the parent's approval
            // handler (human-in-the-loop / auto). Falls back to DenyAll only when
            // the parent itself carries no handler — which keeps the "no policy,
            // no handler → read-only" invariant intact.
            let approval: Arc<dyn ApprovalHandler> = match &self.approval_handler {
                Some(h) => h.clone(),
                None => Arc::new(DenyAllApprovalHandler),
            };
            (
                format!("{}\n\n{}", system_prompt, note),
                Some(policy),
                approval,
            )
        };

        let mut builder = AgentBuilder::new(self.client.clone())
            .system_prompt(prompt)
            .approval_handler(approval)
            .language(self.language.clone());

        if let Some(p) = policy {
            builder = builder.tool_policy(p);
        }

        // Turn / context-window overrides (design §5.1 mapping table; wired
        // straight to agent-base's builder). `None` = inherit framework
        // default — i.e. the pre-config behaviour.
        if let Some(max_turns) = config.max_turns {
            builder = builder.execution_max_turns(max_turns);
        }
        if let Some(context_window) = config.context_window {
            builder = builder.context_window(context_window);
        }

        // Register business tools (NOT multi-agent tools): global exclusion
        // first, then the per-child whitelist (§5.4 "先排除,再白名单").
        // `registered` records the names actually handed to the child — the
        // post-exclusion, post-whitelist set that validation runs against.
        //
        // §7.5: under `Manual` the deployment-declared write set joins the
        // exclusion *before* the whitelist runs, so a child asking for
        // `write_file` gets the reduced set (warn, not error) — the hard
        // floor of "children are read-only" is this merge, not the nudge.
        let excluded: BTreeSet<&str> = if self.autonomy == AgentAutonomy::Manual {
            self.child_excluded_tools
                .iter()
                .chain(&self.write_tools)
                .map(|s| s.as_str())
                .collect()
        } else {
            self.child_excluded_tools
                .iter()
                .map(|s| s.as_str())
                .collect()
        };
        let mut registered: BTreeSet<String> = BTreeSet::new();
        for tool in &self.business_tools {
            let name = tool.name();
            if excluded.contains(name) {
                tracing::debug!(
                    tool = name,
                    "skipping excluded business tool for child runtime"
                );
                continue;
            }
            if let Some(allow) = &config.tool_names
                && !allow.contains(name)
            {
                continue;
            }
            builder = builder.register_tool_arc(tool.clone());
            registered.insert(name.to_string());
        }

        // Post-exclusion whitelist validation (§5.4, review M-3). The check
        // runs against `registered` — NOT the pre-exclusion business-tool
        // list — so a preset that asks for a globally-excluded tool cannot
        // "validate" while the child silently ends up read-only (the fake
        // completion trap named in §6.1).
        //
        // Three outcomes for each requested name:
        //   1. registered            → satisfied;
        //   2. excluded by deployment→ warn + drop it (intentional tightening;
        //      §5.4:298 — don't error, don't silently fake the permission);
        //   3. resolves nowhere      → ToolNotFound (unknown/typo/hallucinated
        //      name — fail loud per §5.4:297).
        //
        // NOTE (doc tension for stage-1 report): §6.1's prose says case 2
        // should be a hard `ToolNotFound`, while §5.4:298, the M-3 change
        // note (line 11), and the stage-1 acceptance list (§12:997) all say
        // warn. This follows the acceptance criterion (warn); see report.
        if let Some(allow) = &config.tool_names {
            for wanted in allow {
                if registered.contains(wanted) {
                    continue;
                }
                if excluded.contains(wanted.as_str()) {
                    tracing::warn!(
                        tool = %wanted,
                        "requested tool is globally excluded from children; \
                         spawning with the reduced set (deployment-side tightening)"
                    );
                    continue;
                }
                return Err(AgentError::ToolNotFound {
                    name: wanted.clone(),
                });
            }
        }

        if let Some(ref recovery) = self.error_recovery {
            builder = builder.error_recovery(recovery.clone());
        }

        let child = builder.build()?;

        // Hook A (§5.4 / review M-4): child usage feeds the rollout budget.
        // `run_turn`'s return value carries no usage, so the child runtime's
        // turn-end callback is the only metering path — registered on the
        // *child*, hence the parent's own spend is never metered (§7.2).
        // Always on: with `child_max_tokens = None` the budget is unlimited,
        // but `ControlStatus::used_tokens` stays observable.
        let budget = Arc::clone(self.control.budget());
        child.on_turn_end(move |ctx| {
            if let Some(usage) = &ctx.usage {
                budget.record_usage(usage_total(usage));
            }
        });

        if let Some(effort) = self.child_reasoning_effort.clone() {
            child.set_reasoning_effort(effort).await;
        }
        Ok((child, registered))
    }

    /// Resolve the effective full-permission flag for a spawn, applying the
    /// configured [`ChildPermissionMode`]. `Full`/`None` override the LLM-supplied
    /// flag; `PerSpawn` lets the LLM decide.
    pub(super) fn effective_permission(&self, full_permission: bool) -> bool {
        match self.child_permission_mode {
            ChildPermissionMode::Full => true,
            ChildPermissionMode::None => false,
            ChildPermissionMode::PerSpawn => full_permission,
        }
    }

    /// The permission a spawn **actually** gets, after the deployment autonomy
    /// mode (§7.5 layer ②, the approval floor).
    ///
    /// `Manual` forces `None`-mode semantics regardless of
    /// [`ChildPermissionMode`]: policy with the parent (or `DenyAll` when the
    /// parent carries none), approvals delegated to the parent's handler —
    /// a child can never raise itself to `Full` in a manual deployment, and
    /// the LLM-supplied `full_permission` flag is ignored (the same
    /// attack-surface logic that keeps the bit out of `ChildConfig`).
    /// `Auto` is exactly [`effective_permission`](Self::effective_permission).
    pub(super) fn spawn_permission(&self, requested: Option<bool>) -> bool {
        match self.autonomy {
            AgentAutonomy::Manual => false,
            AgentAutonomy::Auto => self.effective_permission(requested.unwrap_or(false)),
        }
    }

    /// Whether the read-only prompt nudge applies (§7.5 layer ③).
    ///
    /// `child_read_only` as configured, forced on by `Manual`. Pure function
    /// of the two deployment bits — unit-testable without a runtime.
    pub(super) fn effective_read_only_nudge(&self) -> bool {
        self.child_read_only || self.autonomy == AgentAutonomy::Manual
    }
}
