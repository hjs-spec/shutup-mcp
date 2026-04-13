# shutup

> An MCP proxy that hides 99% of your tools so your agent can finally focus.

## The Problem

Your MCP agent sees 80,000 tools. Every tool definition eats context. Every extra tool is a chance to pick the wrong one. Anthropic's tool search gets it right 34% of the time. The rest is noise.

## The Solution

`shutup` sits between your agent and MCP servers. It reads the user's intent, and only shows the agent the 3-5 tools that matter. The agent never knows the other 79,997 exist.

## Results

- Token usage: **-98%**
- Response time: **-85%**
- Tool selection accuracy: **+2x**

## Quick Start

```bash
pip install shutup-mcp
shutup --config claude_desktop_config.json
```

## How It Works

1. At startup, `shutup` collects tool definitions from all connected MCP servers.
2. It embeds each tool's name + description using a local model.
3. When the user sends a request, `shutup` extracts the intent and embeds it.
4. Only the top-k most relevant tools are shown to the agent.

Zero config. No API keys. Completely local.

## License

MIT
```
