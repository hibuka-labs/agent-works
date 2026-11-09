use std::collections::BTreeMap;
use std::path::Path;

use agent_base::tool::Tool;
use agent_base::{AgentError, AgentResult, Content, ToolContext, ToolExposure};
use async_trait::async_trait;
use serde_json::Value;

use crate::tool_provider::{ToolDescriptor, ToolIdentity, ToolProvider};

use super::exec::execute_entry;
use super::law::script_file_index;
use super::types::*;

/// Global script tool supply: namespace `script.{name}`, a name → entry table.
///
/// Contract invariant: `call` rejects every foreign key (builtin / `mcp.` / `plugin.` all get
/// [`AgentError::ToolNotFound`]).
#[derive(Debug, Default)]
pub struct ScriptProvider {
    entries: BTreeMap<String, ScriptToolEntry>,
}

impl ScriptProvider {
    /// Build the table in bulk; **duplicate names keep the first one** (design: skip the later one
    /// at load time and warn — the warning lives at the product layer).
    pub fn new(entries: impl IntoIterator<Item = ScriptToolEntry>) -> Self {
        let mut provider = Self::default();
        for entry in entries {
            provider.entries.entry(entry.name.clone()).or_insert(entry);
        }
        provider
    }

    /// Add/replace one entry.
    pub fn insert(&mut self, entry: ScriptToolEntry) -> Option<ScriptToolEntry> {
        self.entries.insert(entry.name.clone(), entry)
    }

    /// Remove one entry (used by `/tools remove`; ledger cleanup is the product layer's job, see approval_ledger).
    pub fn remove(&mut self, name: &str) -> Option<ScriptToolEntry> {
        self.entries.remove(name)
    }

    pub fn get(&self, name: &str) -> Option<&ScriptToolEntry> {
        self.entries.get(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[async_trait]
impl ToolProvider for ScriptProvider {
    async fn list(&self) -> AgentResult<Vec<ToolDescriptor>> {
        Ok(self
            .entries
            .values()
            .map(|entry| ToolDescriptor {
                identity: entry.identity(),
                description: entry.description.clone(),
                parameters: entry.parameters_envelope(),
                timeout_ms: Some(entry.timeout_ms()),
                exposure: entry.exposure.into(),
            })
            .collect())
    }

    async fn call(
        &self,
        identity: &ToolIdentity,
        args: &Value,
        ctx: &ToolContext,
    ) -> AgentResult<Vec<Content>> {
        let ToolIdentity::Script { name } = identity else {
            return Err(AgentError::ToolNotFound {
                name: identity.to_key(),
            });
        };
        let Some(entry) = self.entries.get(name) else {
            return Err(AgentError::ToolNotFound {
                name: identity.to_key(),
            });
        };
        execute_entry(&identity.to_key(), entry, args, ctx).await
    }

    async fn health(&self) -> AgentResult<()> {
        // Cheap probe: is the script file still there.
        for entry in self.entries.values() {
            let index = script_file_index(&entry.run)
                .map_err(|e| AgentError::Internal(format!("script.{}: {e}", entry.name)))?;
            let path = Path::new(&entry.run[index]);
            if !path.is_file() {
                return Err(AgentError::Internal(format!(
                    "script.{} is gone: {}",
                    entry.name,
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

/// A single script tool (plugs into the [`Tool`] table; `Box::leak` naming follows the existing
/// `McpToolAdapter` precedent).
///
/// Global scripts use [`ScriptTool::new`] (key = `script.{name}`); the plugin `script` form uses
/// [`ScriptTool::with_key`] in P2 to mount `plugin.{plugin}.{tool}`.
pub struct ScriptTool {
    name: &'static str,
    description: &'static str,
    schema: Value,
    entry: ScriptToolEntry,
}

impl ScriptTool {
    /// Global script tool: key = `script.{name}`.
    pub fn new(entry: ScriptToolEntry) -> Self {
        let key = format!("script.{}", entry.name);
        Self::with_key(&key, entry)
    }

    /// Explicit identity key (plugin `script` form: `plugin.{plugin}.{tool}`).
    pub fn with_key(identity_key: &str, entry: ScriptToolEntry) -> Self {
        let name: &'static str = Box::leak(identity_key.to_string().into_boxed_str());
        let description: &'static str = Box::leak(entry.description.clone().into_boxed_str());
        let schema = entry.parameters_envelope();
        Self {
            name,
            description,
            schema,
            entry,
        }
    }

    pub fn entry(&self) -> &ScriptToolEntry {
        &self.entry
    }
}

#[async_trait]
impl Tool for ScriptTool {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        execute_entry(self.name, &self.entry, args, ctx).await
    }

    fn timeout_ms(&self) -> Option<u64> {
        Some(self.entry.timeout_ms())
    }

    fn metadata(&self) -> agent_base::ToolMetadata {
        agent_base::ToolMetadata {
            name: self.name.to_string(),
            description: self.description.to_string(),
            origin: "script".to_string(),
            version: "unknown".to_string(),
            requirements: vec![],
        }
    }

    fn exposure(&self) -> ToolExposure {
        self.entry.exposure.into()
    }
}
