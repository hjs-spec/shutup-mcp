#!/usr/bin/env python3
"""Example: Using shutup-mcp with a local filesystem MCP server."""

import subprocess
import sys

def main():
    print("=" * 60)
    print("shutup-mcp Basic Usage Example")
    print("=" * 60)
    print()
    print("This example demonstrates filtering tools for a filesystem MCP server.")
    print()
    print("Command:")
    print("  shutup --intent 'read and write files' -- npx @modelcontextprotocol/server-filesystem /tmp")
    print()
    print("The upstream server provides these tools:")
    print("  - read_file")
    print("  - write_file")
    print("  - list_directory")
    print("  - delete_file")
    print("  - move_file")
    print("  - get_file_info")
    print()
    print("With intent 'read and write files', shutup will likely return:")
    print("  - read_file")
    print("  - write_file")
    print()
    print("To run this example, make sure you have:")
    print("  1. Node.js installed (for npx)")
    print("  2. shutup-mcp installed (pip install -e .)")
    print()

if __name__ == "__main__":
    main()
