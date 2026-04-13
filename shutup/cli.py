"""Command-line interface for shutup-mcp."""

import argparse
import asyncio
import os
import sys

from .proxy import ShutupProxy


def main():
    """Entry point for the shutup command."""
    parser = argparse.ArgumentParser(
        description="shutup - An MCP proxy that filters tools based on user intent.",
        usage="shutup --intent 'your task description' -- upstream-command [args...]"
    )
    parser.add_argument(
        "--intent", "-i",
        type=str,
        default=None,
        help="User's current task or intent description."
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of tools to return (default: 5)."
    )
    parser.add_argument(
        "upstream",
        nargs=argparse.REMAINDER,
        help="The upstream MCP server command and its arguments."
    )

    args = parser.parse_args()

    if not args.upstream:
        print("Error: No upstream MCP server command provided.", file=sys.stderr)
        print("Example: shutup --intent 'process excel' -- npx @modelcontextprotocol/server-filesystem /tmp", file=sys.stderr)
        sys.exit(1)

    # Pass intent and top_k via environment for proxy to pick up
    if args.intent:
        os.environ["SHUTUP_INTENT"] = args.intent
    os.environ["SHUTUP_TOP_K"] = str(args.top_k)

    # The proxy will read upstream command from sys.argv in its own logic
    # We need to modify sys.argv so that proxy._get_upstream_params() works
    sys.argv = ["shutup-mcp"] + args.upstream

    # Run the proxy
    intent = os.environ.get("SHUTUP_INTENT")
    top_k = int(os.environ.get("SHUTUP_TOP_K", "5"))
    proxy = ShutupProxy(intent=intent, top_k=top_k)

    try:
        asyncio.run(proxy.run())
    except KeyboardInterrupt:
        print("\n[shutup] Shutting down.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
