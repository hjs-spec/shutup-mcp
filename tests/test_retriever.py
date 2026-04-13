"""Tests for HybridRetriever."""

import pytest
import numpy as np
from shutup.retriever import HybridRetriever, SearchResult
from shutup.embedder import SentenceTransformerEmbedder


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    @pytest.fixture
    def sample_tools(self):
        return [
            {"name": "read_file", "description": "Read contents of a file from filesystem"},
            {"name": "write_file", "description": "Write data to a file on disk"},
            {"name": "list_directory", "description": "List files and folders in a directory"},
            {"name": "http_request", "description": "Make HTTP requests to external APIs"},
            {"name": "github_create_issue", "description": "Create a new issue on GitHub"},
            {"name": "github_list_issues", "description": "List open issues in a GitHub repository"},
            {"name": "slack_send_message", "description": "Send a message to a Slack channel"},
        ]

    @pytest.fixture
    def retriever(self):
        return HybridRetriever(embedder_backend="sentence-transformers")

    def test_build_index(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        assert len(retriever.tools) == 7
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape == (7, retriever.embedder.get_dimension())
        assert retriever.bm25 is not None
        assert len(retriever.tokenized_corpus) == 7

    def test_retrieve_returns_top_k(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        results = retriever.retrieve("read a file", top_k=2)
        assert len(results) == 2
        # The most relevant should be file-related
        assert results[0]["name"] in ["read_file", "write_file", "list_directory"]

    def test_retrieve_handles_keyword_match(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        results = retriever.retrieve("github issue", top_k=3)
        names = [r["name"] for r in results]
        assert "github_create_issue" in names
        assert "github_list_issues" in names

    def test_retrieve_with_empty_index(self, retriever):
        results = retriever.retrieve("anything")
        assert results == []

    def test_vector_search(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        vector_results = retriever._vector_search("read file")
        assert len(vector_results) == 7
        # Results should be sorted by score
        assert vector_results[0].score >= vector_results[-1].score

    def test_bm25_search(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        bm25_results = retriever._bm25_search("create github issue")
        assert len(bm25_results) == 7
        # Highest score should go to github_create_issue
        best_idx = max(bm25_results, key=lambda x: x.score).tool_index
        assert retriever.tools[best_idx]["name"] == "github_create_issue"

    def test_rrf_fusion_merges_results(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        vector_results = retriever._vector_search("slack message")
        bm25_results = retriever._bm25_search("slack message")
        
        merged = retriever._rrf_fusion(vector_results, bm25_results, top_k=3)
        assert len(merged) == 3
        # Slack tool should appear in top results
        names = [t["name"] for t in merged]
        assert "slack_send_message" in names

    def test_debouncing_caches_results(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        
        # First query
        results1 = retriever.retrieve("read file", top_k=3)
        # Very similar query should hit cache
        results2 = retriever.retrieve("read a file", top_k=3)
        
        # Results should be identical (cached)
        assert [t["name"] for t in results1] == [t["name"] for t in results2]
        assert retriever._last_query == "read a file"

    def test_debouncing_refreshes_on_different_query(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        
        results1 = retriever.retrieve("read file", top_k=3)
        # Very different query should bypass cache
        results2 = retriever.retrieve("slack message", top_k=3)
        
        assert retriever._last_query == "slack message"
        # Results should differ
        assert [t["name"] for t in results1] != [t["name"] for t in results2]

    def test_jaccard_similarity(self, retriever):
        sim = retriever._jaccard_similarity("read file", "read a file")
        assert sim > 0.5
        sim2 = retriever._jaccard_similarity("read file", "send slack message")
        assert sim2 < 0.3

    def test_get_tool_by_name(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        tool = retriever.get_tool_by_name("read_file")
        assert tool is not None
        assert tool["description"] == "Read contents of a file from filesystem"
        
        tool = retriever.get_tool_by_name("nonexistent")
        assert tool is None

    def test_add_tool(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        new_tool = {"name": "delete_file", "description": "Delete a file from filesystem"}
        retriever.add_tool(new_tool)
        
        assert len(retriever.tools) == 8
        assert retriever.embeddings.shape[0] == 8
        assert len(retriever.tokenized_corpus) == 8
        
        # Verify new tool is retrievable
        results = retriever.retrieve("delete a file", top_k=2)
        assert results[0]["name"] == "delete_file"

    def test_remove_tool(self, retriever, sample_tools):
        retriever.build_index(sample_tools)
        assert retriever.remove_tool("read_file") is True
        assert len(retriever.tools) == 6
        assert retriever.embeddings.shape[0] == 6
        assert len(retriever.tokenized_corpus) == 6
        
        assert retriever.remove_tool("nonexistent") is False
