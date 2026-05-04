"""Hybrid semantic/keyword retrieval for MCP tool definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re
import numpy as np

from .embedder import BaseEmbedder, create_embedder


@dataclass
class SearchResult:
    tool_index: int
    score: float


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class HybridRetriever:
    """Hybrid vector + lexical retriever with simple RRF fusion."""

    def __init__(self, embedder: Optional[BaseEmbedder] = None, embedder_backend: str = "sentence-transformers"):
        self.embedder = embedder or create_embedder(embedder_backend)
        self.tools: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.tokenized_corpus: List[List[str]] = []
        self._last_query: Optional[str] = None
        self._last_results: Optional[List[dict]] = None

    def build_index(self, tools: List[dict]) -> None:
        self.tools = tools or []
        if not self.tools:
            self.embeddings = None
            self.tokenized_corpus = []
            return

        texts = [self._tool_text(t) for t in self.tools]
        self.embeddings = self.embedder.encode(texts)
        self.tokenized_corpus = [tokenize(t) for t in texts]
        self._last_query = None
        self._last_results = None

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.tools:
            return []

        query = query or ""
        if self._last_query is not None and self._last_results is not None:
            if self._jaccard_similarity(query, self._last_query) > 0.75:
                return self._last_results[:top_k]

        vector = self._vector_search(query)
        lexical = self._bm25_search(query)
        results = self._rrf_fusion(vector, lexical, top_k=top_k)
        self._last_query = query
        self._last_results = results
        return results

    def _tool_text(self, tool: dict) -> str:
        return f"{tool.get('name', '')} {tool.get('description', '')}"

    def _vector_search(self, query: str) -> List[SearchResult]:
        if self.embeddings is None:
            return []
        q = self.embedder.encode([query])[0]
        scores = np.dot(self.embeddings, q) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q) + 1e-8
        )
        order = np.argsort(scores)[::-1]
        return [SearchResult(int(i), float(scores[i])) for i in order]

    def _bm25_search(self, query: str) -> List[SearchResult]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return [SearchResult(i, 0.0) for i in range(len(self.tools))]

        scores = []
        for i, doc in enumerate(self.tokenized_corpus):
            score = 0.0
            doc_set = set(doc)
            for t in q_tokens:
                if t in doc_set:
                    score += 1.0 + doc.count(t) * 0.1
            scores.append(SearchResult(i, score))
        return sorted(scores, key=lambda r: r.score, reverse=True)

    def _rrf_fusion(self, a: List[SearchResult], b: List[SearchResult], top_k: int = 5, k: int = 60) -> List[dict]:
        scores = {}
        for rank, result in enumerate(a):
            scores[result.tool_index] = scores.get(result.tool_index, 0.0) + 1.0 / (k + rank + 1)
        for rank, result in enumerate(b):
            scores[result.tool_index] = scores.get(result.tool_index, 0.0) + 1.0 / (k + rank + 1)
        order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.tools[i] for i, _ in order[:top_k]]

    def _jaccard_similarity(self, a: str, b: str) -> float:
        sa, sb = set(tokenize(a)), set(tokenize(b))
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def add_tool(self, tool: dict) -> None:
        self.tools.append(tool)
        self.build_index(self.tools)

    def remove_tool(self, name: str) -> bool:
        old_len = len(self.tools)
        self.tools = [t for t in self.tools if t.get("name") != name]
        changed = len(self.tools) != old_len
        if changed:
            self.build_index(self.tools)
        return changed

    def get_tool_by_name(self, name: str) -> Optional[dict]:
        for tool in self.tools:
            if tool.get("name") == name:
                return tool
        return None
