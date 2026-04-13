"""Tests for server_manager module."""

import json
import tempfile
from pathlib import Path
import pytest
from shutup.server_manager import ServerManager, ServerConfig


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_to_server_params(self):
        config = ServerConfig(
            name="test",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        params = config.to_server_params()
        assert params.command == "npx"
        assert params.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


class TestServerManager:
    """Tests for ServerManager."""

    @pytest.fixture
    def sample_config_file(self):
        """Create a temporary MCP config file."""
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        yield Path(temp_path)
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def simple_config_file(self):
        """Create a config with simple string format."""
        config_data = {
            "mcpServers": {
                "filesystem": "npx -y @modelcontextprotocol/server-filesystem /tmp"
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        yield Path(temp_path)
        Path(temp_path).unlink(missing_ok=True)

    def test_load_config_dict_format(self, sample_config_file):
        manager = ServerManager(sample_config_file)
        servers = manager.load_config()
        assert len(servers) == 2
        assert servers[0].name == "filesystem"
        assert servers[0].command == "npx"
        assert servers[1].name == "fetch"

    def test_load_config_simple_string_format(self, simple_config_file):
        manager = ServerManager(simple_config_file)
        servers = manager.load_config()
        assert len(servers) == 1
        assert servers[0].name == "filesystem"
        assert servers[0].command == "npx"
        assert "/tmp" in servers[0].args

    def test_load_config_no_file(self):
        manager = ServerManager(Path("/nonexistent/config.json"))
        servers = manager.load_config()
        assert servers == []

    def test_stop_observer(self, sample_config_file):
        manager = ServerManager(sample_config_file)
        manager.stop_observer()
        # No exception means success

    @pytest.mark.asyncio
    async def test_fetch_all_tools(self, sample_config_file):
        """This test requires actual MCP servers to be available via npx."""
        manager = ServerManager(sample_config_file)
        try:
            tools = await manager.fetch_all_tools()
            # Even if servers fail, we should get a list (possibly empty)
            assert isinstance(tools, list)
        except Exception as e:
            pytest.skip(f"MCP servers not available: {e}")

    def test_get_server_for_tool(self, sample_config_file):
        manager = ServerManager(sample_config_file)
        manager.load_config()
        # Add a mock tool with prefixed name
        manager.all_tools = []  # Not needed for this test
        config = manager.get_server_for_tool("filesystem__read_file")
        assert config is not None
        assert config.name == "filesystem"

        config = manager.get_server_for_tool("nonexistent__tool")
        assert config is None

        config = manager.get_server_for_tool("invalid_name")
        assert config is None
