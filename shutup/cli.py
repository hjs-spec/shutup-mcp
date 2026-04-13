"""Command-line interface for shutup-mcp."""

import argparse
import asyncio
import sys
from pathlib import Path

from .proxy import ShutupProxy


def main():
    """Entry point for the shutup command."""
    parser = argparse.ArgumentParser(
        description="shutup - An MCP proxy that filters tools based on user intent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use with sentence-transformers (default)
  shutup --config ~/.../claude_desktop_config.json --intent "read and write files"

  # Use with Ollama for fully offline embeddings
  shutup --config ~/.../claude_desktop_config.json --intent "process excel" --embedder ollama

  # Return top 3 tools instead of default 5
  shutup --config ~/.../claude_desktop_config.json --intent "github issues" --top-k 3
""",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to MCP config file (e.g., claude_desktop_config.json)."
    )
    parser.add_argument(
        "--intent", "-i",
        type=str,
        required=True,
        help="User's current task or intent description."
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of tools to return (default: 5)."
    )
    parser.add_argument(
        "--embedder", "-e",
        type=str,
        choices=["sentence-transformers", "ollama"],
        default="sentence-transformers",
        help="Embedding backend to use (default: sentence-transformers)."
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"[shutup] Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[shutup] Starting with config: {config_path}", file=sys.stderr)
    print(f"[shutup] Intent: '{args.intent}'", file=sys.stderr)
    print(f"[shutup] Top-K: {args.top_k}", file=sys.stderr)
    print(f"[shutup] Embedder: {args.embedder}", file=sys.stderr)

    proxy = ShutupProxy(
        config_path=config_path,
        intent=args.intent,
        top_k=args.top_k,
        embedder_backend=args.embedder,
    )

    try:
        asyncio.run(proxy.run())
    except KeyboardInterrupt:
        print("\n[shutup] Shutting down.", file=sys.stderr)
        proxy.shutdown()
        sys.exit(0)
    except Exception as e:
        print(f"[shutup] Fatal error: {e}", file=sys.stderr)
        proxy.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
