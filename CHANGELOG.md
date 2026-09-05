# Changelog

All notable changes to `agent-works` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-09-06

### Added
- Full multi-agent orchestration layer (spawn / track / fan-in) per design doc
  stages 1-4, built on the push-based delivery model.
- `spawn_agent` gains a `task` field with Focus-based prompt expansion.
- Push-based child results with fan-in batch injection: child results are
  handed to the parent LLM as a single intact batch instead of trickling in.
- Fact-derived child status machine; the delivery gap is surfaced as facts
  (`results_handed_over` / `pending_results`).
- The `multi_agent` feature now implies `focus` (the fan-in coordinator uses
  Focus for user-facing progress summaries).
- `loom-check` feature to model-check the multi_agent atomic gates with loom
  (off by default; run `cargo test --features multi_agent,loom-check --lib loom`).
- `multi_agent_control` benchmark and `multi_agent` example.

### Fixed
- Child agent now returns only the last assistant message.
- Watcher emits the plain progress notice first and the Focus summary as a
  follow-up.
- Guard never judges completion after rejected tool calls.
