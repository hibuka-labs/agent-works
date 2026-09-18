# Changelog

All notable changes to `agent-works` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-09-18

### Added
- **Auto-memory system** (behind `memory` feature): persistent project memory
  compatible with Claude Code storage (`~/.claude/projects/<slug>/memory/`).
  Includes `MemoryStore`, `MemoryConfig`, memory tools (write/read/list/delete),
  frontmatter serde, atomic writes, line-level index management, and builder
  integration across all 4 build paths. 66 unit + 6 integration tests.
- **Session-scope slash commands**: slash-activated skills get `Session` scope
  (stays in effect for the whole session instead of single-turn). Catalog
  trigger copy distinguishes session-scope from single-turn skills.
- **Active Skills bake sanitization** (`demote_h2_headings`): prevents skill
  body `##` headings from colliding with prompt-surgery anchors.
- **`closing` terminal fact** on `AgentEntry`: force-killed children count as
  settled for fan-in quiescence immediately; `note_closing()` method +
  `status()` derives to `Closed` on close.
- **Held-batch reaper** (90s liveness backstop): hands over stranded reports
  when quiescence is unachievable rather than deadlocking.
- **Shutdown flush**: watcher force-hands-over held reports on cancel/hub-drop.

### Changed
- Default work budget increased from 96K → 210K (aligned with ~256K context
  window × 90% minus base overhead).

## [0.7.0] - 2026-09-11

### Added
- **`rotation_policy` module** (renamed from `token_budget`): `TokenBudgetConfig`,
  `TokenBudgetCore`, `TokenBudgetState`, `TokenBudgetAction`,
  `DEFAULT_SEED_MESSAGE`, `build_context_window_info`, `token_budget_base_overhead`.
- **Skill catalog discovery** (behind `prompt_skill`): filesystem-based skill
  resolver, catalog refresh middleware, telemetry, and `SkillTool` — enables
  runtime skill injection via prompt without dedicated tools.
- **`agent_instructions_paths`** on `AgentBuilder`: read instruction files at
  build time (e.g. CLAUDE.md, INSTRUCTIONS.md) and append to the system prompt;
  supports multiple paths with priority ordering.
- **Child `model` field now functional**: `ChildConfig.model` is applied via
  `AgentRuntime::set_model_override()` so sub-agents can use a different model
  tier (e.g. "lite").

### Changed
- Error results now emit a single synchronous Progress carrying the error's
  first line (ANSI/control sanitized, truncated at 160 chars) instead of a terse
  notice plus a detached Focus paraphrase — the user sees the real reason in the
  TUI the moment the child fails.

### Removed
- `token_budget` module (renamed to `rotation_policy`).

### Changed
- Error results now emit a single synchronous Progress carrying the error's
  first line (ANSI/control sanitized, truncated at 160 chars) instead of a
  terse notice plus a detached Focus paraphrase — the user sees the real
  reason in the TUI the moment the child fails (session 20260906_0f6d4341).
  Ok results are unchanged (plain notice, then Focus summary).

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
