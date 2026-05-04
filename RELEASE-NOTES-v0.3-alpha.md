# shutup-mcp v0.3.0-alpha Release Notes

## Summary

This release fixes the alpha implementation so the documented intent-filtering workflow is usable.

## Added

- `--intent` CLI argument.
- `--serve` mode for MCP stdio proxy usage.
- Explicit `shutup__set_intent` runtime control tool.
- Intent-aware `tools/list` filtering.
- `fake` embedder for deterministic tests and CI.
- Lightweight CI workflow.
- Honest alpha README positioning and security notes.

## Changed

- Removed unverified performance claims.
- Reframed dynamic intent detection as explicit intent control.
- Tests no longer download sentence-transformers models.
- Dependencies split optional embedding backends.

## Known limitations

- The proxy focuses on tools/list and tools/call.
- Upstream sessions are opened per call in this alpha.
- Resources and prompts are not fully proxied.
- This is not an authorization or DLP layer.
