//! Memory tools (Phase 9b): `memory_write` / `memory_read` / `memory_list` /
//! `memory_delete`.
//!
//! Dedicated tools instead of generic `read_file`/`write_file` so the LLM
//! cannot write broken frontmatter, the index is synced automatically, and
//! writes are serialized through the store's mutex. All tools share one
//! `Arc<MemoryStore>` bound to a single memory directory.

use std::sync::Arc;

use agent_base::{Tool, ToolContext};

use crate::memory::MemoryStore;

mod memory_delete;
mod memory_list;
mod memory_read;
mod memory_write;

pub use memory_delete::MemoryDeleteTool;
pub use memory_list::MemoryListTool;
pub use memory_read::MemoryReadTool;
pub use memory_write::MemoryWriteTool;

/// Fixed, LLM-facing description of the four memory tools. Substituted into
/// the `{tools_description}` placeholder of the memory prompt template.
pub const MEMORY_TOOLS_DESCRIPTION: &str = "\
- `memory_write(name, description, type, body)` — create or update one memory and sync the index.
- `memory_read(name)` — read one memory's full content.
- `memory_list()` — return the current index (refreshes the stale startup snapshot).
- `memory_delete(name)` — delete a memory and remove its index line.";

/// Create the four memory tools bound to `store`.
pub fn create_memory_tools(store: Arc<MemoryStore>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(MemoryWriteTool::new(Arc::clone(&store))),
        Arc::new(MemoryReadTool::new(Arc::clone(&store))),
        Arc::new(MemoryListTool::new(Arc::clone(&store))),
        Arc::new(MemoryDeleteTool::new(store)),
    ]
}

/// Best-effort session reference for `metadata.originSessionId`: the session's
/// external id when the consumer provides one (e.g. a UUID), else the numeric id.
fn session_ref(ctx: &ToolContext) -> String {
    ctx.session_id
        .external_id
        .clone()
        .unwrap_or_else(|| ctx.session_id.id.to_string())
}

/// Shared test helpers for the per-tool test modules.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use agent_base::Content;

    /// Extract the text of the first `Content` in a tool result (the memory
    /// tools always return exactly one text content).
    pub(crate) fn first_text(out: &[Content]) -> String {
        match out.first() {
            Some(Content::Text { text }) => text.clone(),
            _ => String::new(),
        }
    }

    /// A `ToolContext` whose session carries an external id (UUID-like).
    pub(crate) fn ctx_with_external_id(id: &str) -> ToolContext {
        let mut ctx = ToolContext::for_test();
        ctx.session_id.external_id = Some(id.to_string());
        ctx
    }
}
