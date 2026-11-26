//! `memory_write`: create or update one memory file and sync the index.
//!
//! Validates `name` against `^[a-z0-9][a-z0-9-]*$` *before* any path is built
//! (path-escape guard), requires a single-line `description`, one of the four
//! `type` values, and auto-fills `metadata.node_type` / `originSessionId` /
//! `modified`. The MEMORY.md row is added/updated by the store under the
//! index lock — the LLM never touches the index itself.

use std::sync::Arc;

use agent_base::{AgentError, AgentResult, Content, Tool, ToolContext};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::memory::{MemoryStore, WriteOutcome};
use crate::tools::session_ref;

#[derive(Debug, Clone, Deserialize)]
struct WriteArgs {
    name: String,
    description: String,
    #[serde(rename = "type")]
    memory_type: String,
    body: String,
}

/// Create or update a memory. See module docs for the validation contract.
pub struct MemoryWriteTool {
    store: Arc<MemoryStore>,
}

impl MemoryWriteTool {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for MemoryWriteTool {
    fn name(&self) -> &'static str {
        "memory_write"
    }

    fn description(&self) -> &'static str {
        "Create or update one persistent memory — a markdown file with YAML frontmatter — and \
         sync its MEMORY.md index row. The `description` is the one-line recall key shown in \
         the index: specific enough to judge relevance at a glance, or nobody will open it.\n\n\
         Use this for a fact that should outlive this session: who the user is, a rule they gave \
         you, where a project stands, a pointer to something external. Recall is selective by \
         design — one fact per memory, named by what it is.\n\n\
         Do NOT use it for this session's working state: that is the notes scratchpad, and a \
         plan or a hand-off in the memory store is noise the next session reads past. Do NOT \
         record what the codebase already says — code structure, git history and config files \
         are their own record, and a memory that duplicates them goes stale the moment they \
         change.\n\n\
         Writing an existing `name` overwrites it in place — update, never duplicate. Read what \
         is there first if you have not: an overwrite takes the previous body with it, and no \
         tool below this one will say so.\n\n\
         The write also owns the index row, so the index cannot drift from the files. Deleting \
         is `memory_delete` for the same reason: a file removed without its row leaves a name \
         pointing at nothing."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "pattern": "^[a-z0-9][a-z0-9-]*$",
                    "description": "kebab-case slug, stored as <name>.md (e.g. `cargo-commit-discipline`)"
                },
                "description": {
                    "type": "string",
                    "description": "one-line summary used for recall in the index; must not contain newlines"
                },
                "type": {
                    "type": "string",
                    "enum": ["user", "feedback", "project", "reference"],
                    "description": "user = who the user is; feedback = rules the user gave; project = ongoing work/status; reference = external pointers"
                },
                "body": {
                    "type": "string",
                    "description": "free-form markdown body of the memory (use Why: / How to apply: sections and [[other-name]] links where helpful)"
                }
            },
            "required": ["name", "description", "type", "body"],
            "additionalProperties": false
        })
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let parsed: WriteArgs =
            serde_json::from_value(args.clone()).map_err(|e| AgentError::ToolArgsInvalid {
                name: self.name().to_string(),
                raw: format!("{args}: {e}"),
            })?;

        match self.store.write_memory(
            &parsed.name,
            &parsed.description,
            &parsed.memory_type,
            &parsed.body,
            Some(&session_ref(ctx)),
        )? {
            // A limit refused the write: surface the five-field failure contract
            // as the tool result (design §6 wire shape), so the model reads the
            // rule, the actual value, and a `candidate` it can retry with in one
            // shot. This is an expected, self-explanatory rejection — not a fault.
            WriteOutcome::Rejected(rej) => Ok(vec![Content::text(rej.to_wire())]),
            WriteOutcome::Written { created } => {
                let created_or_updated = if created { "created" } else { "updated" };
                let path = self.store.memory_root().join(format!("{}.md", parsed.name));
                Ok(vec![Content::text(format!(
                    "Memory `{}` {created_or_updated} at {} and the index was synced.",
                    parsed.name,
                    path.display()
                ))])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_store() -> (tempfile::TempDir, Arc<MemoryStore>) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(MemoryStore::new(dir.path().join("memory"), "MEMORY.md"));
        (dir, store)
    }

    fn tool(store: &Arc<MemoryStore>) -> MemoryWriteTool {
        MemoryWriteTool::new(Arc::clone(store))
    }

    #[tokio::test]
    async fn writes_file_and_syncs_index() {
        let (dir, store) = temp_store();
        let out = tool(&store)
            .call(
                &json!({
                    "name": "cargo-commit-discipline",
                    "description": "绝不主动 commit",
                    "type": "feedback",
                    "body": "0. 绝不主动 commit\n1. commit 永不带 Cargo 文件"
                }),
                &ToolContext::for_test(),
            )
            .await
            .unwrap();

        let text = crate::tools::test_support::first_text(&out);
        assert!(text.contains("cargo-commit-discipline"), "{text}");
        assert!(text.contains("created"), "{text}");

        let root = dir.path().join("memory");
        let raw = std::fs::read_to_string(root.join("cargo-commit-discipline.md")).unwrap();
        assert!(raw.contains("name: cargo-commit-discipline"));
        assert!(raw.contains("type: feedback"));
        assert!(raw.contains("node_type: memory"));
        assert!(raw.contains("modified: "));
        // Verify via parse-back (serde_yaml may quote scalar values).
        let parsed = crate::memory::parse_memory_str(&raw).unwrap();
        let md = parsed.metadata.unwrap();
        assert_eq!(md.origin_session_id.as_deref(), Some("0"), "{raw}");
        assert!(md.modified.is_some(), "{raw}");
        assert!(raw.contains("commit 永不带 Cargo 文件"));

        let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
        assert!(
            index.contains(
                "- [cargo-commit-discipline](cargo-commit-discipline.md) — 绝不主动 commit"
            ),
            "{index}"
        );
    }

    #[tokio::test]
    async fn same_name_write_reports_updated_and_keeps_one_row() {
        let (_dir, store) = temp_store();
        let t = tool(&store);
        let ctx = ToolContext::for_test();
        t.call(
            &json!({"name": "dup", "description": "v1", "type": "project", "body": "one"}),
            &ctx,
        )
        .await
        .unwrap();
        let out = t
            .call(
                &json!({"name": "dup", "description": "v2", "type": "project", "body": "two"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(crate::tools::test_support::first_text(&out).contains("updated"));

        let index = store.read_index();
        assert_eq!(index.matches("(dup.md)").count(), 1, "{index}");
        assert!(index.contains("v2"), "{index}");
        let raw = store.read_memory("dup").unwrap();
        assert!(raw.contains("two"));
    }

    #[tokio::test]
    async fn rejects_path_escape_name() {
        let (_dir, store) = temp_store();
        let err = tool(&store)
            .call(
                &json!({"name": "../../etc/passwd", "description": "d", "type": "user", "body": "b"}),
                &ToolContext::for_test(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("^[a-z0-9][a-z0-9-]*$"), "{err}");
    }

    #[tokio::test]
    async fn rejects_invalid_type() {
        let (_dir, store) = temp_store();
        let err = tool(&store)
            .call(
                &json!({"name": "valid", "description": "d", "type": "secret", "body": "b"}),
                &ToolContext::for_test(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must be one of"), "{err}");
    }

    #[tokio::test]
    async fn rejects_multiline_description() {
        let (_dir, store) = temp_store();
        let err = tool(&store)
            .call(
                &json!({"name": "valid", "description": "line1\nline2", "type": "user", "body": "b"}),
                &ToolContext::for_test(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("single line"), "{err}");
    }

    #[tokio::test]
    async fn missing_args_is_tool_args_invalid() {
        let (_dir, store) = temp_store();
        let err = tool(&store)
            .call(&json!({"name": "only-name"}), &ToolContext::for_test())
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::ToolArgsInvalid { .. }), "{err}");
    }

    #[tokio::test]
    async fn uses_external_session_id_when_present() {
        let (_dir, store) = temp_store();
        let ctx = crate::tools::test_support::ctx_with_external_id(
            "8ee16870-4dc8-4dd1-9d13-4fc7a9980f79",
        );
        tool(&store)
            .call(
                &json!({"name": "sess", "description": "d", "type": "reference", "body": "b"}),
                &ctx,
            )
            .await
            .unwrap();
        let raw = store.read_memory("sess").unwrap();
        assert!(
            raw.contains("originSessionId: 8ee16870-4dc8-4dd1-9d13-4fc7a9980f79"),
            "{raw}"
        );
    }

    #[test]
    fn schema_requires_all_fields_and_enums_type() {
        let schema = tool(&temp_store().1).schema();
        let required = schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 4);
        assert_eq!(
            schema["properties"]["name"]["pattern"],
            "^[a-z0-9][a-z0-9-]*$"
        );
        let variants: Vec<_> = schema["properties"]["type"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(variants, vec!["user", "feedback", "project", "reference"]);
    }

    /// Value: protects=the wire failure contract the model reads — a limit
    /// rejection comes back as the `{"ok": false, rule, limit, actual, fix,
    /// candidate}` tool result (not a bare fault), so the model sees all five
    /// fields and can retry with `candidate` in one shot; fails_when=the
    /// rejection is flattened into an opaque error string, or the candidate is
    /// missing/unusable so the retry bounces.
    /// no_claim=that the model will use the candidate — only that it is offered
    /// and that it passes when used.
    #[tokio::test]
    async fn a_limit_rejection_returns_the_five_field_contract_and_the_candidate_retries_ok() {
        let (_dir, store) = temp_store();
        let t = tool(&store);
        let ctx = ToolContext::for_test();
        let long_hook = "h".repeat(141);

        // Reject: the tool result is the `ok: false` contract.
        let out = t
            .call(
                &json!({"name": "big", "description": long_hook, "type": "project", "body": "b"}),
                &ctx,
            )
            .await
            .unwrap();
        let text = crate::tools::test_support::first_text(&out);
        let wire: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
            panic!("rejection must be the JSON wire shape, got {text}: {e}");
        });
        assert_eq!(wire["ok"], false, "{text}");
        for key in ["rule", "limit", "actual", "fix", "candidate"] {
            assert!(wire.get(key).is_some(), "contract missing `{key}`: {text}");
        }
        assert_eq!(wire["rule"], "hook_max_chars", "{text}");
        assert_eq!(wire["limit"], 80, "{text}");
        assert_eq!(wire["actual"], 141, "{text}");
        let candidate = wire["candidate"].as_str().unwrap().to_string();
        assert!(!candidate.is_empty(), "{text}");

        // Retry with the candidate → created.
        let out2 = t
            .call(
                &json!({"name": "big", "description": candidate, "type": "project", "body": "b"}),
                &ctx,
            )
            .await
            .unwrap();
        let text2 = crate::tools::test_support::first_text(&out2);
        assert!(
            text2.contains("created"),
            "candidate retry must pass: {text2}"
        );
    }
}
