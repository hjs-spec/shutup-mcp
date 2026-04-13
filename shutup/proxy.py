"""MCP proxy that filters tools dynamically based on user intent."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, CallToolResult, TextContent

from .retriever import HybridRetriever
from .server_manager import ServerManager


class ShutupProxy:
    """
    An MCP proxy that intercepts requests and dynamically filters tools
    based on the user's current intent.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        top_k: int = 5,
        embedder_backend: str = "sentence-transformers",
    ):
        """
        Initialize the proxy.
        
        Args:
            config_path: Path to MCP config file (e.g., claude_desktop_config.json).
            top_k: Number of tools to return to the agent.
            embedder_backend: "sentence-transformers" or "ollama".
        """
        self.config_path = config_path
        self.top_k = top_k
        self.embedder_backend = embedder_backend

        self.server_manager = ServerManager(config_path)
        self.retriever = HybridRetriever(embedder_backend=embedder_backend)

        self.server = Server("shutup-mcp")

        # Register MCP handlers
        self.server.list_tools()(self.handle_list_tools)
        self.server.call_tool()(self.handle_call_tool)

        # Cache for upstream connections
        self._upstream_sessions: Dict[str, ClientSession] = {}
        
        # Track the current intent (extracted from user messages)
        self._current_intent: str = ""

    async def initialize(self) -> None:
        """Fetch tools from all upstream servers and build the hybrid index."""
        tools = await self.server_manager.fetch_all_tools()
        if tools:
            tool_dicts = [
                {"name": t.name, "description": t.description or ""}
                for t in tools
            ]
            self.retriever.build_index(tool_dicts)
        else:
            print("[shutup] Warning: No tools found from upstream servers.", file=sys.stderr)

    def _extract_intent_from_message(self, message: Dict[str, Any]) -> Optional[str]:
        """
        Extract user intent from an MCP message.
        
        In stdio MCP, user messages come through as prompts/get or tools/call
        with natural language content.
        """
        # Look for text content in various possible locations
        if "content" in message:
            if isinstance(message["content"], str):
                return message["content"]
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
        if "params" in message:
            params = message["params"]
            if isinstance(params, dict):
                if "arguments" in params and isinstance(params["arguments"], dict):
                    # Check for common query fields
                    for field in ["query", "message", "prompt", "text", "input"]:
                        if field in params["arguments"]:
                            return params["arguments"][field]
        return None

    async def handle_list_tools(self) -> List[Tool]:
        """Intercept tools/list and return dynamically filtered tools."""
        all_tools = self.server_manager.get_all_tools()

        if not all_tools:
            return []

        if not self._current_intent:
            print("[shutup] No intent detected yet, returning all tools.", file=sys.stderr)
            return all_tools

        # Retrieve relevant tools using hybrid search
        relevant_dicts = self.retriever.retrieve(self._current_intent, self.top_k)
        relevant_names = {t["name"] for t in relevant_dicts}

        filtered_tools = [t for t in all_tools if t.name in relevant_names]

        print(
            f"[shutup] Intent: '{self._current_intent[:50]}...' -> "
            f"returning {len(filtered_tools)}/{len(all_tools)} tools.",
            file=sys.stderr,
        )

        return filtered_tools

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Forward tool calls to the appropriate upstream server.
        Also extracts intent from the call for future filtering.
        """
        # Update intent from this call
        if "query" in arguments:
            self._current_intent = str(arguments["query"])
        elif "message" in arguments:
            self._current_intent = str(arguments["message"])
        elif "prompt" in arguments:
            self._current_intent = str(arguments["prompt"])

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
                    
                    # Convert result to proper MCP format
                    content = []
                    if hasattr(result, 'content'):
                        for item in result.content:
                            if hasattr(item, 'text'):
                                content.append(TextContent(type="text", text=item.text))
                            else:
                                content.append(item)
                    elif isinstance(result, dict):
                        content.append(TextContent(type="text", text=json.dumps(result)))
                    else:
                        content.append(TextContent(type="text", text=str(result)))
                    
                    return CallToolResult(content=content, isError=False)
        except Exception as e:
            print(f"[shutup] Error calling tool '{name}': {e}", file=sys.stderr)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {e}")],
                isError=True
            )

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
