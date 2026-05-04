"""MCP proxy implementation for shutup-mcp.

This module focuses on tools/list and tools/call. It exposes an explicit
`shutup__set_intent` control tool so clients can update retrieval intent before
asking for tools/list.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .retriever import HybridRetriever
from .server_manager import ServerManager


CONTROL_TOOL_NAME = "shutup__set_intent"


def control_tool_definition() -> dict:
    return {
        "name": CONTROL_TOOL_NAME,
        "description": "Set the current intent used by shutup-mcp to filter tools/list results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "The user's current task or tool-selection intent."
                }
            },
            "required": ["intent"]
        }
    }


class ShutupProxy:
    """A minimal MCP JSON-RPC stdio proxy for tool filtering."""

    def __init__(
        self,
        config_path: Path,
        intent: str = "",
        top_k: int = 5,
        embedder_backend: str = "sentence-transformers",
    ):
        self.config_path = Path(config_path)
        self.current_intent = intent or ""
        self.top_k = top_k
        self.server_manager = ServerManager(self.config_path)
        self.retriever = HybridRetriever(embedder_backend=embedder_backend)
        self._indexed = False

    async def initialize(self) -> None:
        self.server_manager.load_config()
        tools = await self.server_manager.fetch_all_tools()
        self.retriever.build_index(tools)
        self._indexed = True

    async def filter_tools(self, intent: Optional[str] = None, top_k: Optional[int] = None) -> List[dict]:
        if not self._indexed:
            await self.initialize()

        query = intent if intent is not None else self.current_intent
        k = top_k or self.top_k

        if query:
            return [control_tool_definition()] + self.retriever.retrieve(query, top_k=k)

        # No intent yet: expose only control tool plus a small safe preview.
        preview = self.retriever.tools[: min(k, len(self.retriever.tools))]
        return [control_tool_definition()] + preview

    async def handle_list_tools(self) -> dict:
        tools = await self.filter_tools()
        return {"tools": tools}

    async def handle_call_tool(self, name: str, arguments: Optional[dict]) -> dict:
        arguments = arguments or {}

        if name == CONTROL_TOOL_NAME:
            intent = str(arguments.get("intent", "")).strip()
            self.current_intent = intent
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"ok": True, "intent": self.current_intent}, ensure_ascii=False)
                    }
                ]
            }

        server = self.server_manager.get_server_for_tool(name)
        if server is None:
            raise ValueError(f"Unknown tool or missing server prefix: {name}")

        # Connect per call for alpha simplicity.
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client
        except Exception as exc:
            raise RuntimeError("mcp package is required to call upstream tools") from exc

        upstream_name = self.server_manager.upstream_tool_name(name)
        params = server.to_server_params()

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(upstream_name, arguments)

    async def handle_json_rpc(self, request: dict) -> Optional[dict]:
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params") or {}

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "shutup-mcp", "version": "0.3.0-alpha"}
                    }
                }

            if method == "tools/list":
                result = await self.handle_list_tools()
                return {"jsonrpc": "2.0", "id": req_id, "result": result}

            if method == "tools/call":
                result = await self.handle_call_tool(params.get("name"), params.get("arguments") or {})
                return {"jsonrpc": "2.0", "id": req_id, "result": result}

            # Notification: initialized
            if method == "notifications/initialized":
                return None

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not implemented by alpha proxy: {method}"}
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)}
            }

    async def serve_stdio(self) -> None:
        await self.initialize()

        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = await self.handle_json_rpc(request)
                if response is not None:
                    print(json.dumps(response, ensure_ascii=False), flush=True)
            except Exception as exc:
                error = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
                print(json.dumps(error), flush=True)
