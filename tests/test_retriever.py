"""Tests for HybridRetriever."""

from shutup.retriever import HybridRetriever
from shutup.embedder import FakeEmbedder


def sample_tools():
    return [
        {"name": "read_file", "description": "Read contents of a file from filesystem"},
        {"name": "write_file", "description": "Write data to a file on disk"},
        {"name": "github_create_issue", "description": "Create a new issue on GitHub"},
        {"name": "github_list_issues", "description": "List open issues in a GitHub repository"},
        {"name": "slack_send_message", "description": "Send a message to a Slack channel"},
    ]


def test_retrieve_keyword_match():
    retriever = HybridRetriever(embedder=FakeEmbedder())
    retriever.build_index(sample_tools())
    results = retriever.retrieve("github issue", top_k=2)
    names = [r["name"] for r in results]
    assert "github_create_issue" in names


def test_empty_index():
    retriever = HybridRetriever(embedder=FakeEmbedder())
    assert retriever.retrieve("anything") == []


def test_add_and_remove_tool():
    retriever = HybridRetriever(embedder=FakeEmbedder())
    retriever.build_index(sample_tools())
    retriever.add_tool({"name": "delete_file", "description": "Delete a file"})
    assert retriever.get_tool_by_name("delete_file") is not None
    assert retriever.remove_tool("delete_file") is True
    assert retriever.get_tool_by_name("delete_file") is None


def test_jaccard_similarity():
    retriever = HybridRetriever(embedder=FakeEmbedder())
    assert retriever._jaccard_similarity("read file", "read a file") > 0.5
    assert retriever._jaccard_similarity("read file", "send slack message") < 0.5
