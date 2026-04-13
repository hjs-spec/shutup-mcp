"""Manages multiple MCP server connections and tool aggregation."""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool


@dataclass
class ServerConfig:
    """Configuration for a single MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)

    def to_server_params(self) -> StdioServerParameters:
        return StdioServerParameters(command=self.command, args=self.args)


class ConfigFileHandler(FileSystemEventHandler):
    """Watchdog handler for config file changes."""

    def __init__(self, manager: "ServerManager"):
        self.manager = manager

    def on_modified(self, event):
        if event.src_path == str(self.manager.config_path):
            print("[shutup] Config file changed. Reloading...", file=sys.stderr)
            # Schedule reload in the running event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.reload())
            except RuntimeError:
                # No running loop, can't reload automatically
                print("[shutup] Warning: Cannot reload config automatically.", file=sys.stderr)


class ServerManager:
    """Manages multiple MCP server connections and tool aggregation."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the server manager.

        Args:
            config_path: Path to claude_desktop_config.json or similar MCP config.
        """
        self.config_path = config_path
        self.servers: Dict[str, ServerConfig] = {}
        self.all_tools: List[Tool] = []
        self._observer: Optional[Observer] = None
        self._setup_observer()

    def _setup_observer(self) -> None:
        """Set up file watcher for config changes."""
        if self.config_path and self.config_path.exists():
            try:
                from watchdog.observers import Observer
                handler = ConfigFileHandler(self)
                self._observer = Observer()
                self._observer.schedule(handler, str(self.config_path.parent), recursive=False)
                self._observer.start()
                print(f"[shutup] Watching config file for changes: {self.config_path}", file=sys.stderr)
            except ImportError:
                print("[shutup] Warning: watchdog not installed. Config auto-reload disabled.", file=sys.stderr)
            except Exception as e:
                print(f"[shutup] Warning: Could not start file watcher: {e}", file=sys.stderr)

    def stop_observer(self) -> None:
        """Stop the file watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=1)

    def load_config(self) -> List[ServerConfig]:
        """Load MCP server configurations from a JSON config file.

        Returns:
            List of ServerConfig objects.
        """
        if not self.config_path or not self.config_path.exists():
            print("[shutup] No config file found. Returning empty server list.", file=sys.stderr)
            return []

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"[shutup] Error reading config file: {e}", file=sys.stderr)
            return []

        servers = []
        mcp_servers = config.get("mcpServers", {})
        for name, server_conf in mcp_servers.items():
            if isinstance(server_conf, dict) and "command" in server_conf:
                cmd = server_conf["command"]
                args = server_conf.get("args", [])
            elif isinstance(server_conf, str):
                # Simple format: just a command string
                cmd_parts = server_conf.split()
                cmd = cmd_parts[0]
                args = cmd_parts[1:]
            else:
                print(f"[shutup] Warning: Skipping invalid server config for '{name}'", file=sys.stderr)
                continue

            servers.append(ServerConfig(name=name, command=cmd, args=args))
            self.servers[name] = servers[-1]

        print(f"[shutup] Loaded {len(servers)} MCP servers from config.", file=sys.stderr)
        return servers

    async def _fetch_tools_from_server(self, server: ServerConfig) -> List[Tool]:
        """Connect to a single MCP server and fetch its tool list."""
        try:
            params = server.to_server_params()
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    return list(tools_result.tools)
        except Exception as e:
            print(f"[shutup] Error fetching tools from '{server.name}': {e}", file=sys.stderr)
            return []

    async def fetch_all_tools(self) -> List[Tool]:
        """Connect to all configured servers and aggregate their tools."""
        servers = self.load_config()
        all_tools = []

        for server in servers:
            tools = await self._fetch_tools_from_server(server)
            # Prefix tool names with server name to avoid conflicts
            for tool in tools:
                # Create a new Tool object with prefixed name
                prefixed_tool = Tool(
                    name=f"{server.name}__{tool.name}",
                    description=tool.description,
                    inputSchema=tool.inputSchema,
                )
                all_tools.append(prefixed_tool)
            print(f"[shutup] Fetched {len(tools)} tools from '{server.name}'", file=sys.stderr)

        self.all_tools = all_tools
        print(f"[shutup] Total aggregated tools: {len(all_tools)}", file=sys.stderr)
        return all_tools

    async def reload(self) -> List[Tool]:
        """Reload configuration and refresh tool list."""
        print("[shutup] Reloading configuration...", file=sys.stderr)
        self.servers.clear()
        return await self.fetch_all_tools()

    def get_all_tools(self) -> List[Tool]:
        """Return the current aggregated tool list."""
        return self.all_tools

    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """Find a tool by its full prefixed name."""
        for tool in self.all_tools:
            if tool.name == name:
                return tool
        return None

    def get_server_for_tool(self, tool_name: str) -> Optional[ServerConfig]:
        """Extract server name from prefixed tool name and return config."""
        if "__" not in tool_name:
            return None
        server_name = tool_name.split("__", 1)[0]
        return self.servers.get(server_name)
