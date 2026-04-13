#!/usr/bin/env python3
"""Example: Using shutup-mcp with a multi-server MCP configuration."""

import subprocess
import sys


def main():
    print("=" * 60)
    print("shutup-mcp V0.1.0 Basic Usage Example")
    print("=" * 60)
    print()
    print("This example demonstrates filtering tools across multiple MCP servers.")
    print()
    print("Example configuration (claude_desktop_config.json):")
    print("""
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    }
  }
}
""")
    print()
    print("Command to filter tools for file operations:")
    print('  shutup --config ~/.../claude_desktop_config.json --intent "read and write files"')
    print()
    print("Expected output:")
    print("  The proxy will only expose filesystem-related tools,")
    print("  hiding GitHub and fetch tools from the agent.")
    print()
    print("Command with Ollama for privacy-focused users:")
    print('  shutup --config ~/.../claude_desktop_config.json --intent "git issues" --embedder ollama')
    print()
    print("To run this example, make sure you have:")
    print("  1. Node.js installed (for npx MCP servers)")
    print("  2. shutup-mcp installed (pip install -e .)")
    print("  3. (Optional) Ollama running for --embedder ollama")
    print()


if __name__ == "__main__":
    main()
