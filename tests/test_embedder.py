"""Tests for lightweight embedder behavior."""

import numpy as np
from shutup.embedder import FakeEmbedder, ToolEmbedder, create_embedder


def test_fake_embedder_single_and_batch():
    embedder = FakeEmbedder(dimension=16)
    single = embedder.encode("read file")
    batch = embedder.encode(["read file", "github issue"])
    assert isinstance(single, np.ndarray)
    assert single.shape == (16,)
    assert batch.shape == (2, 16)


def test_create_fake_embedder():
    assert isinstance(create_embedder("fake"), FakeEmbedder)


def test_tool_embedder_search():
    tools = [
        {"name": "read_file", "description": "Read contents of a file"},
        {"name": "github_issue", "description": "Create GitHub issues"},
    ]
    index = ToolEmbedder(backend="fake")
    index.build_index(tools)
    results = index.search("github issue", top_k=1)
    assert results[0]["name"] == "github_issue"
