# CLAUDE.md — agent-works

Batteries-included Agent toolbox (MCP, Skills, built-in tools, Focus).

## Rules

### Dependencies
- `Cargo.toml` uses **pure version deps** (no `path`). The committed state is clean.
- To debug against a local dependency: temporarily add `path`, **DO NOT commit** it.

### Publishing
After making changes to this crate:

1. Bump version in `Cargo.toml`
2. Commit and push to GitHub
3. `cargo publish --registry crates-io`

### Downstream crates
When publishing a new version, update the dep in:
- [ ] phi-agent
- [ ] phi-bard
- [ ] ops/*

### Pre-push
```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings
```
