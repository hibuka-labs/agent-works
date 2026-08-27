//! Core trait and composition logic for prompt fragments.

/// Context passed to each fragment during rendering.
pub struct FragmentContext<'a> {
    /// Tool definitions as JSON (OpenAI function-calling format).
    pub tool_definitions: &'a [serde_json::Value],
    /// Current session ID.
    pub session_id: &'a str,
}

/// Composable prompt fragment — each fragment owns one concern.
///
/// # Priority convention
///
/// | Range  | Typical use                     |
/// |--------|----------------------------------|
/// | 10–19  | Core role/personality            |
/// | 20–29  | Thinking methodology             |
/// | 50–59  | Safety rules, environment info   |
/// | 60–69  | Workflow instructions            |
/// | 70–79  | Tool descriptions                |
/// | 80–89  | Feature-specific (multi-agent)   |
/// | 90–99  | Network environment, memory      |
///
/// Lower priority renders first (closer to the top of the system prompt).
pub trait PromptFragment: Send + Sync + dyn_clone::DynClone {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Lower = rendered first. Default: 100.
    fn priority(&self) -> i32 {
        100
    }

    /// Render this fragment's content. Return `None` to skip.
    fn render(&self, ctx: &FragmentContext) -> Option<String>;
}

dyn_clone::clone_trait_object!(PromptFragment);

/// Sort fragments by priority (ascending) and concatenate with double newlines.
///
/// Fragments that return `None` from `render()` are skipped.
/// Empty result returns an empty string.
pub fn compose_fragments(fragments: &[Box<dyn PromptFragment>], ctx: &FragmentContext) -> String {
    let mut sorted: Vec<&Box<dyn PromptFragment>> = fragments.iter().collect();
    sorted.sort_by_key(|f| f.priority());
    sorted
        .iter()
        .filter_map(|f| f.render(ctx))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct AlphaFragment;
    impl PromptFragment for AlphaFragment {
        fn name(&self) -> &str {
            "alpha"
        }
        fn priority(&self) -> i32 {
            20
        }
        fn render(&self, _ctx: &FragmentContext) -> Option<String> {
            Some("ALPHA".to_string())
        }
    }

    #[derive(Clone)]
    struct BetaFragment;
    impl PromptFragment for BetaFragment {
        fn name(&self) -> &str {
            "beta"
        }
        fn priority(&self) -> i32 {
            10
        }
        fn render(&self, _ctx: &FragmentContext) -> Option<String> {
            Some("BETA".to_string())
        }
    }

    #[derive(Clone)]
    struct SkipFragment;
    impl PromptFragment for SkipFragment {
        fn name(&self) -> &str {
            "skip"
        }
        fn render(&self, _ctx: &FragmentContext) -> Option<String> {
            None
        }
    }

    #[test]
    fn compose_sorts_by_priority() {
        let fragments: Vec<Box<dyn PromptFragment>> = vec![
            Box::new(AlphaFragment), // priority 20
            Box::new(BetaFragment),  // priority 10
        ];
        let ctx = FragmentContext {
            tool_definitions: &[],
            session_id: "test",
        };
        let result = compose_fragments(&fragments, &ctx);
        // Beta (10) should come before Alpha (20)
        assert_eq!(result, "BETA\n\nALPHA");
    }

    #[test]
    fn compose_skips_none() {
        let fragments: Vec<Box<dyn PromptFragment>> = vec![
            Box::new(AlphaFragment),
            Box::new(SkipFragment),
            Box::new(BetaFragment),
        ];
        let ctx = FragmentContext {
            tool_definitions: &[],
            session_id: "test",
        };
        let result = compose_fragments(&fragments, &ctx);
        assert_eq!(result, "BETA\n\nALPHA");
    }

    #[test]
    fn compose_empty_fragments() {
        let fragments: Vec<Box<dyn PromptFragment>> = vec![];
        let ctx = FragmentContext {
            tool_definitions: &[],
            session_id: "test",
        };
        let result = compose_fragments(&fragments, &ctx);
        assert!(result.is_empty());
    }

    #[test]
    fn compose_all_none_returns_empty() {
        let fragments: Vec<Box<dyn PromptFragment>> =
            vec![Box::new(SkipFragment), Box::new(SkipFragment)];
        let ctx = FragmentContext {
            tool_definitions: &[],
            session_id: "test",
        };
        let result = compose_fragments(&fragments, &ctx);
        assert!(result.is_empty());
    }

    #[test]
    fn default_priority_is_100() {
        #[derive(Clone)]
        struct DefaultPriorityFragment;
        impl PromptFragment for DefaultPriorityFragment {
            fn name(&self) -> &str {
                "default"
            }
            fn render(&self, _ctx: &FragmentContext) -> Option<String> {
                Some("default".to_string())
            }
        }
        assert_eq!(DefaultPriorityFragment.priority(), 100);
    }
}
