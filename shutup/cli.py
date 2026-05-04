"""Command line interface for shutup-mcp."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .proxy import ShutupProxy
from .server_manager import ServerManager
from .retriever import HybridRetriever


async def run_filter(config: Path, intent: str, top_k: int, embedder: str) -> list[dict]:
    manager = ServerManager(config)
    manager.load_config()
    tools = await manager.fetch_all_tools()
    retriever = HybridRetriever(embedder_backend=embedder)
    retriever.build_index(tools)
    if intent:
        return retriever.retrieve(intent, top_k=top_k)
    return tools[:top_k]


async def run_proxy(config: Path, intent: str, top_k: int, embedder: str) -> None:
    proxy = ShutupProxy(config_path=config, intent=intent, top_k=top_k, embedder_backend=embedder)
    await proxy.serve_stdio()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter large MCP tool lists using intent-aware retrieval.")
    parser.add_argument("--config", required=True, help="Path to Claude Desktop MCP config JSON")
    parser.add_argument("--intent", default="", help="Current task intent used to filter tools")
    parser.add_argument("--top-k", type=int, default=5, help="Number of tools to expose")
    parser.add_argument(
        "--embedder",
        default="sentence-transformers",
        choices=["sentence-transformers", "ollama", "fake"],
        help="Embedding backend",
    )
    parser.add_argument("--serve", action="store_true", help="Run as MCP stdio proxy")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Path(args.config).expanduser()

    if args.serve:
        asyncio.run(run_proxy(config, args.intent, args.top_k, args.embedder))
        return

    results = asyncio.run(run_filter(config, args.intent, args.top_k, args.embedder))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
