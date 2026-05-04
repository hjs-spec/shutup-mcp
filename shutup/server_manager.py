"""MCP server configuration management."""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception:  # pragma: no cover - allows unit tests without MCP runtime.
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


@dataclass
class ServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None

    def to_server_params(self):
        if StdioServerParameters is None:
            raise RuntimeError("mcp package is required to connect to MCP servers")
        return StdioServerParameters(command=self.command, args=self.args, env=self.env)


class ServerManager:
    """Load MCP server configs and fetch tool definitions."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.servers: List[ServerConfig] = []
        self.all_tools: List[dict] = []

    def load_config(self) -> List[ServerConfig]:
        if not self.config_path.exists():
            print(f"[shutup] Config not found: {self.config_path}", file=sys.stderr)
            self.servers = []
            return []

        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw_servers = data.get("mcpServers", {})
        servers: List[ServerConfig] = []

        for name, value in raw_servers.items():
            if isinstance(value, str):
                parts = shlex.split(value)
                if not parts:
                    continue
                servers.append(ServerConfig(name=name, command=parts[0], args=parts[1:]))
            elif isinstance(value, dict):
                command = value.get("command")
                if not command:
                    continue
                servers.append(
                    ServerConfig(
                        name=name,
                        command=command,
                        args=list(value.get("args", [])),
                        env=value.get("env"),
                    )
                )

        self.servers = servers
        return servers

    async def fetch_all_tools(self) -> List[dict]:
        if not self.servers:
            self.load_config()

        tools: List[dict] = []
        for server in self.servers:
            try:
                server_tools = await self.fetch_tools_for_server(server)
                for tool in server_tools:
                    tool_dict = self._tool_to_dict(tool)
                    original_name = tool_dict.get("name", "")
                    prefixed = dict(tool_dict)
                    prefixed["name"] = f"{server.name}__{original_name}"
                    prefixed["server"] = server.name
                    prefixed["upstream_name"] = original_name
                    tools.append(prefixed)
            except Exception as exc:
                print(f"[shutup] Failed to fetch tools from {server.name}: {exc}", file=sys.stderr)

        self.all_tools = tools
        return tools

    async def fetch_tools_for_server(self, server: ServerConfig):
        if ClientSession is None or stdio_client is None:
            raise RuntimeError("mcp package is required to fetch tools")
        params = server.to_server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools

    def get_server_for_tool(self, tool_name: str) -> Optional[ServerConfig]:
        if "__" not in tool_name:
            return None
        server_name, _ = tool_name.split("__", 1)
        for server in self.servers:
            if server.name == server_name:
                return server
        return None

    def upstream_tool_name(self, tool_name: str) -> str:
        if "__" not in tool_name:
            return tool_name
        return tool_name.split("__", 1)[1]

    def stop_observer(self) -> None:
        # Placeholder for future watchdog integration.
        return None

    def _tool_to_dict(self, tool) -> dict:
        if isinstance(tool, dict):
            return dict(tool)
        data = {}
        for key in ["name", "description", "inputSchema", "input_schema"]:
            if hasattr(tool, key):
                data[key] = getattr(tool, key)
        if "description" not in data:
            data["description"] = ""
        return data
