#!/usr/bin/env python3
"""Example: Using shutup-mcp with explicit intent filtering."""

import subprocess
import tempfile
import json
from pathlib import Path


def main():
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
            },
        }
    }

    print("Example config:")
    print(json.dumps(config, indent=2))
    print()
    print("Example command:")
    print('shutup --config claude_desktop_config.json --intent "read and write files" --top-k 5')
    print()
    print("For CI tests, use --embedder fake.")
