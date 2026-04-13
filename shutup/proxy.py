"""MCP proxy that filters tools based on user intent."""

import asyncio
import json
import sys
from typing import Optional, List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from .embedder import ToolEmbedder


class ShutupProxy:
    """
    An MCP proxy that intercepts tools/list requests and returns only
    the most relevant tools for the current user intent.
    """

    def __init__(self, intent: Optional[str] = None, top_k: int = 5):
        """
        Initialize the proxy.

        Args:
            intent: User's current task description. If not provided,
                    the proxy will attempt to extract it from the first
                    user message (future enhancement).
            top_k: Number of tools to return to the agent.
        """
        self.intent = intent
        self.top_k = top_k
        self.embedder = ToolEmbedder()
        self.upstream_tools: List[Tool] = []
        self.server = Server("shutup-mcp")

        # Register handlers
        self.server.list_tools()(self.handle_list_tools)
        self.server.call_tool()(self.handle_call_tool)

    async def fetch_upstream_tools(self, server_params: StdioServerParameters) -> List[Tool]:
        """Connect to upstream MCP server and fetch its tool list."""
        print(f"[shutup] Connecting to upstream MCP server: {server_params.command} ...", file=sys.stderr)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(f"[shutup] Upstream server provides {len(tools)} tools.", file=sys.stderr)
                return tools

    async def handle_list_tools(self) -> List[Tool]:
        """Intercept tools/list and return filtered tools based on intent."""
        if not self.upstream_tools:
            # If no tools cached yet, return empty list
            return []

        if not self.intent:
            # No intent provided, return all tools (fallback)
            print("[shutup] Warning: No intent provided, returning all tools.", file=sys.stderr)
            return self.upstream_tools

        # Convert upstream tools to dict format for embedder
        tool_dicts = [
            {"name": t.name, "description": t.description or ""}
            for t in self.upstream_tools
        ]

        # Build index if not already built or if tools changed
        self.embedder.build_index(tool_dicts)

        # Search for relevant tools
        relevant = self.embedder.search(self.intent, self.top_k)

        # Map back to original Tool objects
        relevant_names = {t["name"] for t in relevant}
        filtered_tools = [t for t in self.upstream_tools if t.name in relevant_names]

        print(f"[shutup] Intent: '{self.intent}' -> returning {len(filtered_tools)}/{len(self.upstream_tools)} tools.", file=sys.stderr)

        return filtered_tools

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Forward tool calls to upstream server."""
        # This is a simplified proxy: we need to maintain upstream connection.
        # For MVP, we can re-establish connection per tool call (not efficient but works).
        # Future version: keep persistent session.

        # For now, just forward by reconnecting
        server_params = self._get_upstream_params()
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return result

    def _get_upstream_params(self) -> StdioServerParameters:
        """Parse upstream server configuration from environment."""
        # In MVP, we read UPSTREAM_COMMAND and UPSTREAM_ARGS from env
        command = sys.argv[1] if len(sys.argv) > 1 else None
        if not command:
            raise ValueError("Upstream MCP server command not provided as argument")

        # Simple parsing: first arg is command, rest are args
        args = sys.argv[2:] if len(sys.argv) > 2 else []

        return StdioServerParameters(command=command, args=args)

    async def run(self):
        """Start the proxy server."""
        # Fetch upstream tools first
        upstream_params = self._get_upstream_params()
        self.upstream_tools = await self.fetch_upstream_tools(upstream_params)

        # Start the MCP server (stdio)
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
