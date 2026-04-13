"""
Hybrid retriever for MCP tools.
Combines semantic vector search with BM25 keyword search using RRF fusion.
"""

import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict

try:
    from rank_bm25 import BM25Okapi
    import nltk
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import word_tokenize
except ImportError:
    print("[shutup] Error: rank-bm25 or nltk not installed. Run: pip install rank-bm25 nltk", file=sys.stderr)
    sys.exit(1)

from .embedder import BaseEmbedder, create_embedder


@dataclass
class SearchResult:
    """Single search result from a retriever."""
    tool_index: int
    score: float


class HybridRetriever:
    """
    Hybrid tool retriever combining vector embeddings and BM25 keyword search.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results without score normalization.
    """
    
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        embedder_backend: str = "sentence-transformers",
        rrf_k: int = 60,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedder: Pre-configured embedder instance.
            embedder_backend: Backend to use if embedder not provided.
            rrf_k: RRF smoothing constant (default 60).
        """
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = create_embedder(embedder_backend)
        
        self.rrf_k = rrf_k
        self.tools: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        
        # Cache for previous retrieval results (debouncing)
        self._last_query: str = ""
        self._last_results: List[Dict[str, Any]] = []
        self._last_top_k: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        try:
            return word_tokenize(text.lower())
        except LookupError:
            nltk.download('punkt', quiet=True)
            return word_tokenize(text.lower())

    def build_index(self, tools: List[Dict[str, Any]]) -> None:
        """
        Build both vector and BM25 indices from tool definitions.
        
        Args:
            tools: List of tool dicts with 'name' and 'description' fields.
        """
        if not tools:
            print("[shutup] Warning: No tools provided to build index.", file=sys.stderr)
            self.tools = []
            self.embeddings = None
            self.bm25 = None
            return

        self.tools = tools
        
        # Build vector index
        texts = [f"{t['name']}: {t.get('description', '')}" for t in tools]
        print(f"[shutup] Building vector embeddings for {len(texts)} tools...", file=sys.stderr)
        self.embeddings = self.embedder.encode(texts)
        
        # Build BM25 index
        print(f"[shutup] Building BM25 index for {len(texts)} tools...", file=sys.stderr)
        self.tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print("[shutup] Hybrid index built.", file=sys.stderr)

    def _vector_search(self, query: str) -> List[SearchResult]:
        """Perform semantic vector search."""
        if self.embeddings is None or len(self.tools) == 0:
            return []
        
        query_embedding = self.embedder.encode([query])[0]
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        return [SearchResult(i, float(scores[i])) for i in range(len(scores))]

    def _bm25_search(self, query: str) -> List[SearchResult]:
        """Perform BM25 keyword search."""
        if self.bm25 is None or len(self.tools) == 0:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        return [SearchResult(i, float(scores[i])) for i in range(len(scores))]

    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge vector and BM25 results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) across all result lists.
        k=60 is the industry standard, balancing precision and consensus.
        """
        # Sort by score descending
        vector_results.sort(key=lambda x: x.score, reverse=True)
        bm25_results.sort(key=lambda x: x.score, reverse=True)
        
        # Compute RRF scores
        rrf_scores: Dict[int, float] = {}
        
        for rank, result in enumerate(vector_results, start=1):
            rrf_scores[result.tool_index] = rrf_scores.get(result.tool_index, 0) + (
                1.0 / (self.rrf_k + rank)
            )
        
        for rank, result in enumerate(bm25_results, start=1):
            rrf_scores[result.tool_index] = rrf_scores.get(result.tool_index, 0) + (
                1.0 / (self.rrf_k + rank)
            )
        
        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        
        return [self.tools[i] for i in sorted_indices[:top_k]]

    def retrieve(self, query: str, top_k: int = 5, debounce_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant tools using hybrid search.
        
        Args:
            query: User's intent or task description.
            top_k: Number of tools to return.
            debounce_threshold: Jaccard similarity threshold for cache reuse.
                               If new query is very similar to previous, return cached.
        
        Returns:
            List of tool dicts ordered by relevance.
        """
        if not self.tools:
            return []
        
        # Debouncing: avoid unnecessary recomputation for very similar queries
        if self._last_query and self._last_top_k == top_k:
            similarity = self._jaccard_similarity(query, self._last_query)
            if similarity > debounce_threshold:
                return self._last_results
        
        # Parallel retrieval (in production, these could be async)
        vector_results = self._vector_search(query)
        bm25_results = self._bm25_search(query)
        
        # RRF fusion
        results = self._rrf_fusion(vector_results, bm25_results, top_k)
        
        # Update cache
        self._last_query = query
        self._last_results = results
        self._last_top_k = top_k
        
        return results

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts for debouncing."""
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 and not words2:
            return 1.0
        return len(words1 & words2) / len(words1 | words2)

    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a tool by its name."""
        for tool in self.tools:
            if tool.get("name") == name:
                return tool
        return None

    def add_tool(self, tool: Dict[str, Any]) -> None:
        """Add a single tool and update indices."""
        self.tools.append(tool)
        text = f"{tool['name']}: {tool.get('description', '')}"
        
        # Update vector index
        new_embedding = self.embedder.encode([text])[0]
        if self.embeddings is None:
            self.embeddings = np.array([new_embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])
        
        # Update BM25 index
        self.tokenized_corpus.append(self._tokenize(text))
        if self.bm25 is None:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            # Rebuild BM25 (simplest approach; for frequent updates, consider incremental)
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name and rebuild indices."""
        for i, tool in enumerate(self.tools):
            if tool.get("name") == name:
                self.tools.pop(i)
                self.build_index(self.tools)
                return True
        return False
