# Implementation hardening — September 2026

Serialize real MCP SDK results at the JSON-RPC boundary.

## Changes

Upstream CallToolResult models are converted with model_dump(mode="json", by_alias=True, exclude_none=True). Regression coverage uses the actual MCP type and JSON serialization.

## Validation

```sh
python -m pytest -q
```

## Compatibility and remaining limits

Dictionary responses for the control tool remain unchanged. No upstream network service is needed by the regression test.
