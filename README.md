# shutup-mcp

> Experimental MCP tool-list filtering proxy for large tool catalogs.

`shutup-mcp` sits between an MCP client and one or more MCP servers. It aggregates tool definitions and exposes only the top-k tools that match the current intent.

This repository is an **alpha implementation seed**. It is useful for experimenting with tool-list compression, but it is not yet a full production MCP gateway.

---

## What It Does

- Reads a Claude Desktop-style MCP config.
- Discovers configured MCP servers.
- Fetches and prefixes upstream tools.
- Builds a hybrid retrieval index over tool names and descriptions.
- Filters `tools/list` results using:
  - explicit CLI intent;
  - a runtime `shutup__set_intent` tool;
  - fallback behavior when no intent is known.
- Routes `tools/call` to the correct upstream server.
- Supports local embedding backends:
  - `sentence-transformers`;
  - `ollama`.

---

## What It Does Not Yet Do

This alpha does not yet provide:

- full MCP request proxying for every capability;
- persistent upstream sessions for every server;
- guaranteed client-side dynamic intent detection across all MCP clients;
- production authentication or sandboxing;
- benchmark-backed token or latency claims.

Earlier README versions included strong reduction metrics. Those are removed until reproducible benchmarks are added.

---

## Install

```bash
pip install shutup-mcp
```

For local development:

```bash
pip install -e ".[dev]"
```

---

## CLI Usage

### One-shot tool filtering

```bash
shutup \
  --config ~/Library/Application\ Support/Claude/claude_desktop_config.json \
  --intent "read and write local files" \
  --top-k 5
```

This prints a JSON array of filtered tool definitions.

### Run as MCP proxy

```bash
shutup \
  --config ~/Library/Application\ Support/Claude/claude_desktop_config.json \
  --intent "work with GitHub issues" \
  --serve \
  --top-k 5
```

If `--intent` is supplied, `tools/list` returns only top-k tools matching that intent.

If no intent is supplied, the proxy exposes a small control tool:

```text
shutup__set_intent
```

Calling this tool updates the current intent, after which `tools/list` can be filtered.

---

## Claude Desktop Configuration

Example:

```json
{
  "mcpServers": {
    "shutup": {
      "command": "shutup",
      "args": [
        "--config",
        "/absolute/path/to/claude_desktop_config.json",
        "--serve",
        "--intent",
        "work with GitHub issues",
        "--top-k",
        "5"
      ]
    }
  }
}
```

---

## Example MCP Server Config

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    }
  }
}
```

---

## Embedder Options

| Backend | Description | Privacy | Setup |
|---|---|---|---|
| `sentence-transformers` | Local model such as `all-MiniLM-L6-v2` | Local after download | Downloads model on first use |
| `ollama` | Ollama embedding model | Local | Requires Ollama running |
| `fake` | Deterministic lightweight test embedder | Local | For tests and CI only |

---

## Command Options

```text
shutup --config PATH [--intent TEXT] [--top-k K] [--embedder BACKEND] [--serve]
```

| Option | Description | Default |
|---|---|---|
| `--config` | Path to Claude Desktop MCP config | required |
| `--intent` | Current user task intent | none |
| `--top-k` | Number of tools to expose | 5 |
| `--embedder` | `sentence-transformers`, `ollama`, or `fake` | `sentence-transformers` |
| `--serve` | Run as an MCP stdio proxy | false |

---

## Runtime Intent Tool

The proxy exposes a control tool:

```text
shutup__set_intent
```

Input:

```json
{
  "intent": "create and triage GitHub issues"
}
```

Output:

```json
{
  "ok": true,
  "intent": "create and triage GitHub issues"
}
```

This provides an explicit client-controlled intent update path.

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Tests use the lightweight `fake` embedder and do not download embedding models.

---

## Security Notes

`shutup-mcp` reads and launches MCP servers from a config file. Treat that config file as executable configuration.

Do not use untrusted server configs.

This project filters tool visibility; it does not enforce policy, authorization, sandboxing, or data-loss prevention.

---

## License

MIT
