"""Embedding backends for shutup-mcp."""

from __future__ import annotations

import hashlib
import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np


class BaseEmbedder(ABC):
    """Base class for embedding backends."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformers backend.

    The import is lazy so tests and lightweight installs do not download or
    import the model unless this backend is actually used.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers backend requires sentence-transformers. "
                "Install the package or use --embedder fake/ollama."
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if single:
            return vectors[0]
        return vectors

    def get_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


class OllamaEmbedder(BaseEmbedder):
    """Ollama embedding backend."""

    def __init__(self, model: str = "nomic-embed-text"):
        try:
            import ollama
        except Exception as exc:
            raise RuntimeError("ollama backend requires the ollama Python package.") from exc
        self.ollama = ollama
        self.model = model
        # nomic-embed-text dimension is typically 768.
        self.dimension = 768

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vectors = []
        for text in texts:
            response = self.ollama.embeddings(model=self.model, prompt=text)
            vectors.append(response["embedding"])
        arr = np.array(vectors, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        arr = arr / norms
        if single:
            return arr[0]
        return arr

    def get_dimension(self) -> int:
        return self.dimension


class FakeEmbedder(BaseEmbedder):
    """Deterministic lightweight embedder for tests and CI.

    It hashes tokens into a fixed-size vector. It is not semantically strong,
    but it is deterministic, fast, and has no network/model dependency.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=float)
            tokens = str(text).lower().replace("_", " ").replace("-", " ").split()
            if not tokens:
                tokens = [""]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec) + 1e-8
            vectors.append(vec / norm)
        arr = np.vstack(vectors)
        if single:
            return arr[0]
        return arr

    def get_dimension(self) -> int:
        return self.dimension


def create_embedder(backend: str = "sentence-transformers", **kwargs) -> BaseEmbedder:
    """Create an embedder instance."""

    if backend == "ollama":
        return OllamaEmbedder(**kwargs)
    if backend == "fake":
        return FakeEmbedder(**kwargs)
    return SentenceTransformerEmbedder(**kwargs)


class ToolEmbedder:
    """High-level vector tool index."""

    def __init__(self, embedder: Optional[BaseEmbedder] = None, backend: str = "sentence-transformers"):
        self.embedder = embedder or create_embedder(backend)
        self.tools: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, tools: List[dict]) -> None:
        if not tools:
            print("[shutup] Warning: No tools provided to build index.", file=sys.stderr)
            self.tools = []
            self.embeddings = None
            return

        self.tools = tools
        texts = [f"{t.get('name', '')}: {t.get('description', '')}" for t in tools]
        self.embeddings = self.embedder.encode(texts)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.tools or self.embeddings is None:
            return []

        query_embedding = self.embedder.encode([query])[0]
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )

        k = min(top_k, len(self.tools))
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.tools[i] for i in top_indices]

    def get_tool_by_name(self, name: str) -> Optional[dict]:
        for tool in self.tools:
            if tool.get("name") == name:
                return tool
        return None
