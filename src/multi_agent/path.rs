//! Tree-style agent path identifier.
//!
//! `AgentPath` represents a position in the agent tree, e.g. `root/searcher/worker-1`.
//! It is used for routing messages between agents and for event attribution.
//!
//! With `max_agent_depth=1` (the default), paths are always `root` or `root/<child-name>`.

use std::fmt;

/// A tree-style path identifying an agent in the agent hierarchy.
///
/// # Format
///
/// Path segments are separated by `/`. The root agent is always `root`.
/// Child agents have paths like `root/searcher` or `root/analyzer`.
///
/// # Examples
///
/// ```rust
/// use agent_works::multi_agent::AgentPath;
///
/// let root = AgentPath::root();
/// assert!(root.is_root());
/// assert_eq!(root.depth(), 0);
/// assert_eq!(root.to_string(), "root");
///
/// let child = root.join("searcher");
/// assert!(!child.is_root());
/// assert_eq!(child.depth(), 1);
/// assert_eq!(child.to_string(), "root/searcher");
///
/// let parsed = "root/searcher".parse::<AgentPath>().unwrap();
/// assert_eq!(parsed, child);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AgentPath {
    /// Path segments from root to this agent.
    /// The first segment is always `"root"`.
    segments: Vec<String>,
}

impl AgentPath {
    /// Create the root agent path.
    pub fn root() -> Self {
        Self {
            segments: vec!["root".to_string()],
        }
    }

    /// Create a child path by appending a segment.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use agent_works::multi_agent::AgentPath;
    ///
    /// let root = AgentPath::root();
    /// let searcher = root.join("searcher");
    /// assert_eq!(searcher.to_string(), "root/searcher");
    /// ```
    pub fn join(&self, name: &str) -> Self {
        let mut segments = self.segments.clone();
        segments.push(name.to_string());
        Self { segments }
    }

    /// Parse from a string like `"root/searcher/worker-1"`.
    ///
    /// Returns `None` if the string is empty, contains an empty segment, or does not
    /// start with `"root"`.
    pub fn parse(path: &str) -> Option<Self> {
        if path.is_empty() {
            return None;
        }
        let segments: Vec<String> = path.split('/').map(|s| s.to_string()).collect();
        // Validate: no empty segments
        if segments.iter().any(|s| s.is_empty()) {
            return None;
        }
        // First segment must be "root"
        if segments.first().map(|s| s.as_str()) != Some("root") {
            return None;
        }
        Some(Self { segments })
    }

    /// Returns `true` if this is the root agent.
    pub fn is_root(&self) -> bool {
        self.segments.len() == 1
    }

    /// Returns the depth from root.
    ///
    /// Root has depth 0, `root/searcher` has depth 1, `root/searcher/worker-1` has depth 2.
    pub fn depth(&self) -> usize {
        self.segments.len() - 1
    }

    /// Returns the name of this agent (the last segment).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use agent_works::multi_agent::AgentPath;
    ///
    /// let path = AgentPath::root().join("searcher");
    /// assert_eq!(path.name(), "searcher");
    /// assert_eq!(AgentPath::root().name(), "root");
    /// ```
    pub fn name(&self) -> &str {
        self.segments.last().map(|s| s.as_str()).unwrap_or("root")
    }

    /// Returns the parent path, or `None` if this is the root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use agent_works::multi_agent::AgentPath;
    ///
    /// let child = AgentPath::root().join("searcher");
    /// assert_eq!(child.parent(), Some(AgentPath::root()));
    /// assert_eq!(AgentPath::root().parent(), None);
    /// ```
    pub fn parent(&self) -> Option<Self> {
        if self.is_root() {
            return None;
        }
        let mut segments = self.segments.clone();
        segments.pop();
        Some(Self { segments })
    }

    /// Returns the segments of this path.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }
}

impl fmt::Display for AgentPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.segments.join("/"))
    }
}

impl std::str::FromStr for AgentPath {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        AgentPath::parse(s).ok_or_else(|| format!("invalid agent path: '{}'", s))
    }
}

impl serde::Serialize for AgentPath {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for AgentPath {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        AgentPath::parse(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid agent path: '{}'", s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_path() {
        let root = AgentPath::root();
        assert!(root.is_root());
        assert_eq!(root.depth(), 0);
        assert_eq!(root.name(), "root");
        assert_eq!(root.to_string(), "root");
        assert_eq!(root.parent(), None);
    }

    #[test]
    fn join_child() {
        let root = AgentPath::root();
        let child = root.join("searcher");
        assert!(!child.is_root());
        assert_eq!(child.depth(), 1);
        assert_eq!(child.name(), "searcher");
        assert_eq!(child.to_string(), "root/searcher");
        assert_eq!(child.parent(), Some(root));
    }

    #[test]
    fn join_nested() {
        let root = AgentPath::root();
        let grandchild = root.join("searcher").join("worker-1");
        assert!(!grandchild.is_root());
        assert_eq!(grandchild.depth(), 2);
        assert_eq!(grandchild.name(), "worker-1");
        assert_eq!(grandchild.to_string(), "root/searcher/worker-1");
        assert_eq!(grandchild.parent(), Some(root.join("searcher")));
    }

    #[test]
    fn parse_valid() {
        assert_eq!("root".parse::<AgentPath>().unwrap(), AgentPath::root());
        assert_eq!(
            "root/searcher".parse::<AgentPath>().unwrap(),
            AgentPath::root().join("searcher")
        );
        assert_eq!(
            "root/a/b".parse::<AgentPath>().unwrap(),
            AgentPath::root().join("a").join("b")
        );
    }

    #[test]
    fn parse_invalid() {
        assert!("".parse::<AgentPath>().is_err());
        assert!("/root".parse::<AgentPath>().is_err()); // empty first segment
        assert!("not-root/a".parse::<AgentPath>().is_err()); // doesn't start with root
        assert!("root/".parse::<AgentPath>().is_err()); // trailing empty
        assert!("root//a".parse::<AgentPath>().is_err()); // empty middle segment
    }

    #[test]
    fn display_roundtrip() {
        let paths = vec!["root", "root/searcher", "root/a/b"];
        for s in paths {
            let parsed: AgentPath = s.parse().unwrap();
            assert_eq!(parsed.to_string(), s);
        }
    }

    #[test]
    fn serde_roundtrip() {
        let path = AgentPath::root().join("searcher");
        let json = serde_json::to_string(&path).unwrap();
        assert_eq!(json, "\"root/searcher\"");
        let deserialized: AgentPath = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, path);
    }

    #[test]
    fn serde_invalid() {
        assert!(serde_json::from_str::<AgentPath>("\"\"").is_err());
        assert!(serde_json::from_str::<AgentPath>("\"not-root\"").is_err());
    }

    #[test]
    fn segments_accessor() {
        let path = AgentPath::root().join("a").join("b");
        assert_eq!(path.segments(), &["root", "a", "b"]);
    }

    #[test]
    fn ordering() {
        let a = AgentPath::root().join("a");
        let b = AgentPath::root().join("b");
        let a1 = AgentPath::root().join("a").join("1");
        assert!(a < b);
        assert!(a < a1);
    }

    // ── proptest: AgentPath ──

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a valid AgentPath by building from root + random segments.
        fn arb_agent_path() -> impl Strategy<Value = AgentPath> {
            prop::collection::vec(r"[a-zA-Z0-9_-]{1,20}", 0..5).prop_map(|segments| {
                let mut path = AgentPath::root();
                for seg in segments {
                    path = path.join(&seg);
                }
                path
            })
        }

        proptest! {
            #[test]
            fn parse_never_panics(s in ".*") {
                let _ = AgentPath::parse(&s);
            }

            #[test]
            fn roundtrip_to_string_parse(path in arb_agent_path()) {
                let s = path.to_string();
                let reparsed = AgentPath::parse(&s).unwrap();
                assert_eq!(path, reparsed, "roundtrip failed for {:?}", s);
            }

            #[test]
            fn parsed_path_starts_with_root(s in "[a-zA-Z0-9/_-]{1,50}") {
                if let Some(path) = AgentPath::parse(&s) {
                    assert_eq!(path.segments()[0], "root",
                        "parsed path must start with 'root', got {:?}", path.segments());
                }
            }

            #[test]
            fn depth_equals_segments_minus_1(path in arb_agent_path()) {
                assert_eq!(path.depth(), path.segments().len() - 1);
            }

            #[test]
            fn parent_of_non_root_is_some(path in arb_agent_path()) {
                if !path.is_root() {
                    let parent = path.parent().unwrap();
                    assert_eq!(parent.depth(), path.depth() - 1);
                    assert_eq!(parent.to_string().len() + 1 + path.name().len(), path.to_string().len());
                }
            }
        }
    }
}
