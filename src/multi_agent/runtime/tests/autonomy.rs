use super::*;

/// Runtime with explicit (mode, read-only nudge, autonomy) bits for the
/// pure-helper checks — no spawn needed.
fn make_runtime_bits(
    mode: ChildPermissionMode,
    read_only: bool,
    autonomy: AgentAutonomy,
) -> Arc<MultiAgentRuntime> {
    let config = MultiAgentConfig {
        child_permission_mode: mode,
        child_read_only: read_only,
        control: ControlConfig {
            autonomy,
            ..Default::default()
        },
        ..MultiAgentConfig::enabled()
    };
    Arc::new(MultiAgentRuntime::new(
        config,
        Arc::new(StreamingStub),
        vec![],
        tokio_util::sync::CancellationToken::new(),
        None,
        agent_base::Language::En,
        None,
        None,
    ))
}

#[test]
fn autonomy_floors_permission_and_forces_nudge() {
    use ChildPermissionMode::{Full, None as NoPerm, PerSpawn};

    // Auto ≡ effective_permission, unchanged across all modes.
    let auto_full = make_runtime_bits(Full, true, AgentAutonomy::Auto);
    assert!(auto_full.spawn_permission(Some(false)));
    assert!(auto_full.spawn_permission(None));
    let auto_no = make_runtime_bits(NoPerm, true, AgentAutonomy::Auto);
    assert!(!auto_no.spawn_permission(Some(true)));
    let auto_per = make_runtime_bits(PerSpawn, true, AgentAutonomy::Auto);
    assert!(auto_per.spawn_permission(Some(true)));
    assert!(!auto_per.spawn_permission(None));

    // Manual: the approval floor holds regardless of mode or request —
    // the LLM cannot self-escalate (§7.5, same logic as B4).
    let manual_full = make_runtime_bits(Full, false, AgentAutonomy::Manual);
    assert!(!manual_full.spawn_permission(Some(true)));

    // Nudge (D3.1): the per-child three-condition rule lives in
    // `capability::read_only_nudge` — Manual forces it on, the
    // `child_read_only` switch forces it on, and a child whose exclusion
    // set no longer covers the write set (i.e. a write-capable child)
    // gets no nudge.
    // (Deployment-bits derivation was removed with the per-child rule;
    // the full matrix is covered by `capability.rs::nudge_three_conditions`.)
    let write_tools: Vec<String> = vec!["write_file".to_string(), "edit_file".to_string()];
    let no_exclusions = std::collections::BTreeSet::new();
    assert!(crate::multi_agent::capability::read_only_nudge(
        AgentAutonomy::Manual, false, &no_exclusions, &write_tools
    ));
    assert!(!crate::multi_agent::capability::read_only_nudge(
        AgentAutonomy::Auto, false, &no_exclusions, &write_tools
    ));
    assert!(crate::multi_agent::capability::read_only_nudge(
        AgentAutonomy::Auto, true, &no_exclusions, &write_tools
    ));
}
