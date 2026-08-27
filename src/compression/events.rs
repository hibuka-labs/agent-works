//! Typed compression lifecycle events.
//!
//! [`CompressionEvent`] replaces inline `serde_json::json!()` construction in
//! the middleware with a compile-time-safe enum.  Events are transported via
//! [`agent_base::UserEvent::Structured`] so no changes to `agent-base` are
//! required.
//!
//! Lifecycle: `Preparing → Started → Progress (0..N) → Completed | Failed`

use serde::{Deserialize, Serialize};

/// Compression trigger type.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum CompressionTrigger {
    /// Automatic compression when token threshold is reached.
    Auto,
    /// Manual compression via /compact command.
    Manual,
    /// Inline compaction within the react loop (after tool execution).
    InlineCompaction,
}

/// Context compression lifecycle events.
///
/// Emitted as `UserEvent::Structured { event_type: "compression", data: ... }`
/// so consumers can match on the typed enum for compile-time safety.
///
/// ## Lifecycle
///
/// ```text
/// Preparing → Started → Progress (0..N) → Completed | Failed
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum CompressionEvent {
    /// Pre-processing — saving snapshot, about to start.
    Preparing {
        /// Session that owns this compression.
        session_id: u64,
        /// Estimated token count before compression.
        tokens_before: usize,
        /// Number of messages in the session.
        msg_count: usize,
        /// What triggered this compression.
        trigger: CompressionTrigger,
    },
    /// Compression started — LLM call about to begin.
    Started {
        /// Session that owns this compression.
        session_id: u64,
        /// Estimated token count before compression.
        tokens_before: usize,
        /// Number of messages in the session.
        msg_count: usize,
        /// What triggered this compression.
        trigger: CompressionTrigger,
    },
    /// Streaming progress — cumulative character count from LLM.
    Progress {
        /// Session that owns this compression.
        session_id: u64,
        /// Cumulative characters received from the summarisation LLM so far.
        chars: usize,
    },
    /// Compression completed — session messages replaced.
    Completed {
        /// Session that owns this compression.
        session_id: u64,
        /// Estimated token count before compression.
        tokens_before: usize,
        /// Estimated token count after compression.
        tokens_after: usize,
        /// Reduction percentage (positive = saved tokens).
        reduction_pct: i32,
        /// Message count before compression.
        msg_count_before: usize,
        /// Message count after compression.
        msg_count_after: usize,
        /// What triggered this compression.
        trigger: CompressionTrigger,
    },
    /// Compression failed — session restored from snapshot.
    Failed {
        /// Session that owns this compression.
        session_id: u64,
        /// Human-readable error description.
        error: String,
        /// What triggered this compression.
        trigger: CompressionTrigger,
    },
}

impl CompressionEvent {
    /// Convert into an [`agent_base::UserEvent::Structured`] for transport.
    pub fn into_user_event(self) -> agent_base::UserEvent {
        agent_base::UserEvent::Structured {
            event_type: "compression".to_string(),
            data: serde_json::to_value(self).expect("CompressionEvent is always serializable"),
        }
    }

    /// Try to extract a [`CompressionEvent`] from a [`agent_base::UserEvent`].
    ///
    /// Returns `None` if the event is not a compression event or if
    /// deserialization fails.
    pub fn from_user_event(event: &agent_base::UserEvent) -> Option<Self> {
        match event {
            agent_base::UserEvent::Structured { event_type, data }
                if event_type == "compression" =>
            {
                serde_json::from_value(data.clone()).ok()
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Serialization round-trip ─────────────────────────────────────────

    #[test]
    fn test_preparing_roundtrip() {
        let event = CompressionEvent::Preparing {
            session_id: 42,
            tokens_before: 4200,
            msg_count: 8,
            trigger: CompressionTrigger::Auto,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: CompressionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn test_started_roundtrip() {
        let event = CompressionEvent::Started {
            session_id: 42,
            tokens_before: 4200,
            msg_count: 8,
            trigger: CompressionTrigger::Manual,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: CompressionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn test_progress_roundtrip() {
        let event = CompressionEvent::Progress {
            session_id: 42,
            chars: 755,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: CompressionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn test_completed_roundtrip() {
        let event = CompressionEvent::Completed {
            session_id: 42,
            tokens_before: 4200,
            tokens_after: 3800,
            reduction_pct: 10,
            msg_count_before: 20,
            msg_count_after: 8,
            trigger: CompressionTrigger::Auto,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: CompressionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn test_failed_roundtrip() {
        let event = CompressionEvent::Failed {
            session_id: 42,
            error: "LLM timeout".to_string(),
            trigger: CompressionTrigger::Manual,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: CompressionEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    // ── JSON shape (snake_case wire format) ──────────────────────────────

    #[test]
    fn test_json_uses_snake_case() {
        let event = CompressionEvent::Completed {
            session_id: 1,
            tokens_before: 100,
            tokens_after: 50,
            reduction_pct: 50,
            msg_count_before: 10,
            msg_count_after: 5,
            trigger: CompressionTrigger::Auto,
        };
        let json = serde_json::to_value(&event).unwrap();

        // phase tag
        assert_eq!(json["phase"], "completed");
        // snake_case keys
        assert!(json.get("tokens_before").is_some(), "expected snake_case");
        assert!(
            json.get("tokensAfter").is_none(),
            "camelCase must not appear"
        );
        assert!(json.get("msg_count_before").is_some());
        assert!(json.get("msg_count_after").is_some());
        assert!(json.get("reduction_pct").is_some());
        assert!(json.get("session_id").is_some());
    }

    #[test]
    fn test_trigger_json_values() {
        let auto = serde_json::to_value(&CompressionTrigger::Auto).unwrap();
        assert_eq!(auto, "auto");

        let manual = serde_json::to_value(&CompressionTrigger::Manual).unwrap();
        assert_eq!(manual, "manual");
    }

    // ── into_user_event / from_user_event ────────────────────────────────

    #[test]
    fn test_into_user_event_and_back() {
        let event = CompressionEvent::Preparing {
            session_id: 42,
            tokens_before: 4200,
            msg_count: 8,
            trigger: CompressionTrigger::Auto,
        };
        let user_event = event.clone().into_user_event();

        // Verify the wrapper.
        match &user_event {
            agent_base::UserEvent::Structured { event_type, .. } => {
                assert_eq!(event_type, "compression");
            }
            other => panic!("expected Structured, got {:?}", other),
        }

        // Round-trip back.
        let recovered = CompressionEvent::from_user_event(&user_event).unwrap();
        assert_eq!(event, recovered);
    }

    #[test]
    fn test_from_user_event_ignores_non_compression() {
        let other = agent_base::UserEvent::Structured {
            event_type: "other".to_string(),
            data: serde_json::json!({}),
        };
        assert!(CompressionEvent::from_user_event(&other).is_none());
    }

    #[test]
    fn test_from_user_event_returns_none_on_malformed_data() {
        let bad = agent_base::UserEvent::Structured {
            event_type: "compression".to_string(),
            data: serde_json::json!({ "phase": "unknown_phase" }),
        };
        assert!(CompressionEvent::from_user_event(&bad).is_none());
    }

    // ── Deserialization from legacy inline JSON (compatibility) ──────────

    #[test]
    fn test_deserialize_progress_from_json() {
        // Matches the existing inline JSON shape emitted by the middleware.
        let json = serde_json::json!({
            "phase": "progress",
            "session_id": 1,
            "chars": 42,
        });
        let event: CompressionEvent = serde_json::from_value(json).unwrap();
        assert_eq!(
            event,
            CompressionEvent::Progress {
                session_id: 1,
                chars: 42
            }
        );
    }

    #[test]
    fn test_deserialize_start_from_json() {
        let json = serde_json::json!({
            "phase": "start",
            "tokens_before": 4200,
            "msg_count": 8,
        });
        // Note: current middleware uses "start", but we use "started" in the enum.
        // This test verifies what happens with the old "start" value.
        let result: Result<CompressionEvent, _> = serde_json::from_value(json);
        // "start" != "started", so this should fail — documenting the breaking change.
        assert!(result.is_err(), "\"start\" should not match \"started\"");
    }

    // ── proptest: CompressionEvent serde roundtrip ──

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a random CompressionTrigger.
        fn arb_trigger() -> impl Strategy<Value = CompressionTrigger> {
            prop_oneof![
                Just(CompressionTrigger::Auto),
                Just(CompressionTrigger::Manual),
                Just(CompressionTrigger::InlineCompaction),
            ]
        }

        /// Generate a random CompressionEvent.
        fn arb_event() -> impl Strategy<Value = CompressionEvent> {
            prop_oneof![
                (0u64..10000, 0usize..100000, 0usize..1000, arb_trigger()).prop_map(
                    |(sid, tokens, msgs, trigger)| CompressionEvent::Preparing {
                        session_id: sid,
                        tokens_before: tokens,
                        msg_count: msgs,
                        trigger,
                    }
                ),
                (0u64..10000, 0usize..100000, 0usize..1000, arb_trigger()).prop_map(
                    |(sid, tokens, msgs, trigger)| CompressionEvent::Started {
                        session_id: sid,
                        tokens_before: tokens,
                        msg_count: msgs,
                        trigger,
                    }
                ),
                (0u64..10000, 0usize..100000).prop_map(|(sid, chars)| CompressionEvent::Progress {
                    session_id: sid,
                    chars,
                }),
                (
                    0u64..10000,
                    0usize..100000,
                    0usize..100000,
                    -100i32..100,
                    0usize..1000,
                    0usize..1000,
                    arb_trigger()
                )
                    .prop_map(|(sid, tb, ta, pct, mb, ma, trigger)| {
                        CompressionEvent::Completed {
                            session_id: sid,
                            tokens_before: tb,
                            tokens_after: ta,
                            reduction_pct: pct,
                            msg_count_before: mb,
                            msg_count_after: ma,
                            trigger,
                        }
                    }),
                (0u64..10000, "[a-z ]{0,50}", arb_trigger()).prop_map(|(sid, err, trigger)| {
                    CompressionEvent::Failed {
                        session_id: sid,
                        error: err,
                        trigger,
                    }
                }),
            ]
        }

        proptest! {
            #[test]
            fn serde_roundtrip(event in arb_event()) {
                let json = serde_json::to_value(&event).unwrap();
                let reparsed: CompressionEvent = serde_json::from_value(json).unwrap();
                assert_eq!(event, reparsed, "serde roundtrip failed");
            }

            #[test]
            fn json_keys_are_snake_case(event in arb_event()) {
                let json = serde_json::to_value(&event).unwrap();
                if let Some(obj) = json.as_object() {
                    for key in obj.keys() {
                        // No uppercase letters in keys (camelCase would have them)
                        assert!(!key.chars().any(|c| c.is_ascii_uppercase()),
                            "key {:?} is not snake_case in {:?}", key, json);
                    }
                }
            }

            #[test]
            fn phase_tag_always_present(event in arb_event()) {
                let json = serde_json::to_value(&event).unwrap();
                assert!(json.get("phase").is_some(),
                    "missing 'phase' tag in {:?}", json);
                assert!(json["phase"].is_string(),
                    "'phase' should be a string in {:?}", json);
            }
        }
    }
}
