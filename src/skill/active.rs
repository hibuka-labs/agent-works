//! Session-active skill names (single-injection guard, survey P4).
//!
//! A skill "goes active" when its body is baked into the session's system
//! prompt (host-side activation — e.g. a user slash command). This set is the
//! shared fact behind two guards: [`render_catalog`](super::render_catalog)
//! marks active entries, and the model-facing [`SkillTool`](super::SkillTool)
//! refuses to re-inject a body that is already in the prompt. Before these,
//! the bake path and the tool path were mutually unaware and one session
//! paid for the same skill twice (~52K tokens, 20260914_50cf809d).

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

/// Shared, mutable set of active skill names.
///
/// Cheap to clone (Arc-backed): the host holds one handle to mutate at
/// activation/resume time, while [`SkillResolver`](super::resolver::SkillResolver)
/// carries a clone for the tool and the catalog to consult. Empty (the
/// default) means nothing is baked and every lookup behaves as before.
#[derive(Clone, Debug, Default)]
pub struct ActiveSkillSet {
    names: Arc<RwLock<HashSet<String>>>,
}

impl ActiveSkillSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a skill name active (idempotent).
    pub fn mark_active(&self, name: impl Into<String>) {
        self.names.write().unwrap().insert(name.into());
    }

    /// Is this skill's body already baked into the system prompt?
    pub fn is_active(&self, name: &str) -> bool {
        self.names.read().unwrap().contains(name)
    }

    /// Replace the whole set — host sync after (re)loading its own list
    /// (session switch must drop the previous session's names).
    pub fn replace(&self, names: impl IntoIterator<Item = String>) {
        let mut guard = self.names.write().unwrap();
        guard.clear();
        guard.extend(names);
    }

    /// Snapshot of the active names, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.names.read().unwrap().iter().cloned().collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_query_replace_are_shared_across_clones() {
        let set = ActiveSkillSet::default();
        let handle = set.clone();
        assert!(!set.is_active("review"));

        handle.mark_active("review");
        assert!(set.is_active("review"), "clone writes are visible to all");
        assert_eq!(set.names(), vec!["review".to_string()]);

        set.replace(["ship".to_string()]);
        assert!(!set.is_active("review"), "replace drops stale names");
        assert!(handle.is_active("ship"));
    }
}
