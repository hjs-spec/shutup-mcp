"""Tests for proxy filtering behavior."""

import pytest
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from shutup.proxy import ShutupProxy, CONTROL_TOOL_NAME


@pytest.mark.asyncio
async def test_filter_tools_with_intent(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"mcpServers": {}}')

    proxy = ShutupProxy(config_path=config, intent="github issue", embedder_backend="fake", top_k=1)

    async def fake_init():
        proxy.retriever.build_index([
            {"name": "filesystem__read_file", "description": "Read a file"},
            {"name": "github__create_issue", "description": "Create a GitHub issue"},
        ])
        proxy._indexed = True

    monkeypatch.setattr(proxy, "initialize", fake_init)
    tools = await proxy.filter_tools()
    names = [t["name"] for t in tools]
    assert names[0] == CONTROL_TOOL_NAME
    assert "github__create_issue" in names


@pytest.mark.asyncio
async def test_set_intent_tool(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"mcpServers": {}}')
    proxy = ShutupProxy(config_path=config, embedder_backend="fake")
    result = await proxy.handle_call_tool(CONTROL_TOOL_NAME, {"intent": "read files"})
    assert proxy.current_intent == "read files"
    assert result["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_upstream_mcp_model_serializes_to_json_rpc(monkeypatch, tmp_path):
    import mcp
    import mcp.client.stdio
    from mcp.types import CallToolResult, TextContent

    upstream = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)

    class Session:
        def __init__(self, *args): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def initialize(self): pass
        async def call_tool(self, name, arguments): return upstream

    @asynccontextmanager
    async def transport(params):
        yield None, None

    monkeypatch.setattr(mcp, "ClientSession", Session)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", transport)
    proxy = ShutupProxy(config_path=tmp_path / "config.json", embedder_backend="fake")
    monkeypatch.setattr(proxy.server_manager, "get_server_for_tool", lambda name: SimpleNamespace(to_server_params=lambda: None))
    monkeypatch.setattr(proxy.server_manager, "upstream_tool_name", lambda name: "read")
    response = await proxy.handle_json_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "fs__read", "arguments": {}}})
    decoded = json.loads(json.dumps(response))
    assert decoded["result"]["content"][0]["text"] == "ok"
    assert decoded["result"]["isError"] is False
