"""MCP proxy that filters tools based on user intent across multiple servers."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from .embedder import ToolEmbedder, create_embedder
from .server_manager import ServerManager


class ShutupProxy:
    """
    An MCP proxy that intercepts tools/list requests and returns only
    the most relevant tools for the current user intent across all
    configured MCP servers.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        intent: Optional[str] = None,
        top_k: int = 5,
        embedder_backend: str = "sentence-transformers",
    ):
        """
        Initialize the proxy.

        Args:
            config_path: Path to MCP config file (e.g., claude_desktop_config.json).
            intent: User's current task description.
            top_k: Number of tools to return to the agent.
            embedder_backend: "sentence-transformers" or "ollama".
        """
        self.config_path = config_path
        self.intent = intent
        self.top_k = top_k
        self.embedder_backend = embedder_backend

        self.server_manager = ServerManager(config_path)
        self.tool_embedder = ToolEmbedder(backend=embedder_backend)

        self.server = Server("shutup-mcp")

        # Register MCP handlers
        self.server.list_tools()(self.handle_list_tools)
        self.server.call_tool()(self.handle_call_tool)

        # Cache for upstream connections
        self._upstream_sessions: Dict[str, ClientSession] = {}

    async def initialize(self) -> None:
        """Fetch tools from all upstream servers and build the embedding index."""
        tools = await self.server_manager.fetch_all_tools()
        if tools:
            tool_dicts = [
                {"name": t.name, "description": t.description or ""}
                for t in tools
            ]
            self.tool_embedder.build_index(tool_dicts)
        else:
            print("[shutup] Warning: No tools found from upstream servers.", file=sys.stderr)

    async def handle_list_tools(self) -> List[Tool]:
        """Intercept tools/list and return filtered tools based on intent."""
        all_tools = self.server_manager.get_all_tools()

        if not all_tools:
            return []

        if not self.intent:
            print("[shutup] Warning: No intent provided, returning all tools.", file=sys.stderr)
            return all_tools

        # Search for relevant tools
        relevant_dicts = self.tool_embedder.search(self.intent, self.top_k)
        relevant_names = {t["name"] for t in relevant_dicts}

        filtered_tools = [t for t in all_tools if t.name in relevant_names]

        print(
            f"[shutup] Intent: '{self.intent}' -> returning {len(filtered_tools)}/{len(all_tools)} tools.",
            file=sys.stderr,
        )

        return filtered_tools

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Forward tool calls to the appropriate upstream server."""
        # Find which server this tool belongs to
        if "__" not in name:
            raise ValueError(f"Tool name '{name}' does not have server prefix.")

        server_name, original_tool_name = name.split("__", 1)
        server_config = self.server_manager.servers.get(server_name)

        if not server_config:
            raise ValueError(f"Unknown server '{server_name}' for tool '{name}'.")

        # Connect to the upstream server and call the tool
        params = server_config.to_server_params()
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(original_tool_name, arguments)
                    return result
        except Exception as e:
            print(f"[shutup] Error calling tool '{name}': {e}", file=sys.stderr)
            raise

    async def run(self) -> None:
        """Start the proxy server."""
        await self.initialize()

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )

    def shutdown(self) -> None:
        """Clean up resources."""
        self.server_manager.stop_observer()
