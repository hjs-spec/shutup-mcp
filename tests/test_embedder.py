"""Tests for embedder module."""

import pytest
import numpy as np
from shutup.embedder import (
    ToolEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)


class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder."""

    def test_init_default(self):
        embedder = SentenceTransformerEmbedder()
        assert embedder.get_dimension() > 0

    def test_encode_single_string(self):
        embedder = SentenceTransformerEmbedder()
        result = embedder.encode("test string")
        assert isinstance(result, np.ndarray)
        assert result.shape == (embedder.get_dimension(),)

    def test_encode_list_of_strings(self):
        embedder = SentenceTransformerEmbedder()
        texts = ["first string", "second string"]
        result = embedder.encode(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, embedder.get_dimension())


class TestToolEmbedder:
    """Tests for ToolEmbedder."""

    @pytest.fixture
    def sample_tools(self):
        return [
            {"name": "read_file", "description": "Read contents of a file"},
            {"name": "write_file", "description": "Write data to a file"},
            {"name": "list_directory", "description": "List files in a directory"},
            {"name": "http_request", "description": "Make HTTP requests to APIs"},
            {"name": "github_issue", "description": "Create or update GitHub issues"},
        ]

    def test_build_index(self, sample_tools):
        embedder = ToolEmbedder(backend="sentence-transformers")
        embedder.build_index(sample_tools)
        assert len(embedder.tools) == 5
        assert embedder.embeddings is not None
        assert embedder.embeddings.shape == (5, embedder.embedder.get_dimension())

    def test_search_returns_top_k(self, sample_tools):
        embedder = ToolEmbedder(backend="sentence-transformers")
        embedder.build_index(sample_tools)

        results = embedder.search("read a file", top_k=2)
        assert len(results) == 2
        assert results[0]["name"] == "read_file"

    def test_search_handles_empty_query(self, sample_tools):
        embedder = ToolEmbedder(backend="sentence-transformers")
        embedder.build_index(sample_tools)

        results = embedder.search("")
        assert len(results) == 5

    def test_search_with_no_index(self):
        embedder = ToolEmbedder(backend="sentence-transformers")
        results = embedder.search("anything")
        assert results == []

    def test_get_tool_by_name(self, sample_tools):
        embedder = ToolEmbedder(backend="sentence-transformers")
        embedder.build_index(sample_tools)

        tool = embedder.get_tool_by_name("read_file")
        assert tool is not None
        assert tool["description"] == "Read contents of a file"

        tool = embedder.get_tool_by_name("nonexistent")
        assert tool is None


class TestCreateEmbedder:
    """Tests for create_embedder factory function."""

    def test_create_default(self):
        embedder = create_embedder()
        assert isinstance(embedder, SentenceTransformerEmbedder)

    def test_create_sentence_transformers_explicit(self):
        embedder = create_embedder(backend="sentence-transformers")
        assert isinstance(embedder, SentenceTransformerEmbedder)

    def test_create_ollama_requires_ollama_running(self):
        """This test will fail if Ollama is not running locally."""
        try:
            embedder = create_embedder(backend="ollama")
            # If we get here without error, Ollama is running
            assert embedder.get_dimension() > 0
        except SystemExit:
            pytest.skip("Ollama is not running or not installed")
