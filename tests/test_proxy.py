"""Tests for proxy filtering behavior."""

import pytest

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
