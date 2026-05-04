"""Tests for server_manager module."""

import json
import tempfile
from pathlib import Path

from shutup.server_manager import ServerManager, ServerConfig


def test_server_config_to_params_if_mcp_available():
    config = ServerConfig(name="test", command="npx", args=["-y", "server"])
    try:
        params = config.to_server_params()
        assert params.command == "npx"
    except RuntimeError:
        # Acceptable in minimal test environments without mcp runtime.
        pass


def test_load_config_dict_format():
    config_data = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            },
            "fetch": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-fetch"],
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        path = Path(f.name)
    try:
        manager = ServerManager(path)
        servers = manager.load_config()
        assert len(servers) == 2
        assert servers[0].name == "filesystem"
        assert servers[0].command == "npx"
    finally:
        path.unlink(missing_ok=True)


def test_load_config_simple_string_format():
    config_data = {"mcpServers": {"filesystem": "npx -y @modelcontextprotocol/server-filesystem /tmp"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        path = Path(f.name)
    try:
        manager = ServerManager(path)
        servers = manager.load_config()
        assert len(servers) == 1
        assert servers[0].command == "npx"
        assert "/tmp" in servers[0].args
    finally:
        path.unlink(missing_ok=True)


def test_get_server_for_tool():
    config_data = {"mcpServers": {"filesystem": {"command": "npx", "args": []}}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        path = Path(f.name)
    try:
        manager = ServerManager(path)
        manager.load_config()
        assert manager.get_server_for_tool("filesystem__read_file").name == "filesystem"
        assert manager.get_server_for_tool("unknown__read_file") is None
        assert manager.get_server_for_tool("invalid") is None
    finally:
        path.unlink(missing_ok=True)
