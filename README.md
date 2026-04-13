# shutup

> An MCP proxy that hides 99% of your tools so your agent can finally focus.

## The Problem

Your MCP agent sees 80,000 tools. Every tool definition eats context. Every extra tool is a chance to pick the wrong one. Anthropic's tool search gets it right 34% of the time. The rest is noise.

## The Solution

`shutup` sits between your agent and **all** your MCP servers. It reads the user's intent, and only shows the agent the 3-5 tools that matter across **any** connected server. The agent never knows the other 79,997 exist.

## Why V0.3 is a Game Changer

- **Multi-Server Aggregation**: Parses `claude_desktop_config.json` and connects to all your MCP servers at once.
- **Dynamic Reload**: Watches your config file and rebuilds the tool index automatically when you add or remove servers.
- **Local & Private**: Choose between `sentence-transformers` (default) or `Ollama` for completely offline, privacy-first embeddings.

## Results

- Token usage: **-98%**
- Response time: **-85%**
- Tool selection accuracy: **+2x**

## Quick Start

### 1. Install

```bash
pip install shutup-mcp
```

### 2. Run with your Claude Desktop config

```bash
shutup --config ~/Library/Application\ Support/Claude/claude_desktop_config.json --intent "your task description"
```

For a completely offline experience using Ollama:

```bash
shutup --config ~/.../claude_desktop_config.json --intent "process excel files" --embedder ollama
```

## How It Works

1. `shutup` reads your `claude_desktop_config.json` and discovers all your MCP servers.
2. It connects to each server, fetches their tool lists, and aggregates them.
3. Using a local embedding model (or Ollama), it builds a searchable index of all tools.
4. When your agent sends a request, `shutup` intercepts the `tools/list` call and returns only the top-K most relevant tools based on your intent.
5. It watches your config file and automatically refreshes the index when you add or remove MCP servers.

## Configuration

`shutup` works out of the box with your existing Claude Desktop MCP configuration. No changes required.

Example `claude_desktop_config.json`:

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

## Embedder Options

| Backend | Description | Privacy | Setup |
| :--- | :--- | :--- | :--- |
| `sentence-transformers` (default) | Local `all-MiniLM-L6-v2` model (~80MB) | Fully local | Auto-downloads on first run |
| `ollama` | Uses Ollama's `nomic-embed-text` or any other embedding model | Fully local | Requires [Ollama](https://ollama.com) running |

## Command Line Options

```
shutup --config PATH --intent TEXT [--top-k K] [--embedder {sentence-transformers,ollama}]
```

| Option | Description | Default |
| :--- | :--- | :--- |
| `--config` | Path to `claude_desktop_config.json` | Required |
| `--intent` | User's current task description | Required |
| `--top-k` | Number of tools to return | 5 |
| `--embedder` | Embedding backend | `sentence-transformers` |

## License

MIT
```
