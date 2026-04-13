"""Embedding abstraction layer with multiple backend support."""

import sys
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for the given text(s).

        Args:
            texts: A single string or a list of strings to embed.

        Returns:
            A numpy array of shape (n_texts, embedding_dim).
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of the generated embeddings."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding provider using Sentence Transformers (local, no API key)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the SentenceTransformer embedder.

        Args:
            model_name: Name of the sentence-transformers model.
                        Default "all-MiniLM-L6-v2" is ~80MB and runs locally.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print(
                "[shutup] Error: sentence-transformers not installed. "
                "Run: pip install sentence-transformers",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[shutup] Loading embedding model: {model_name} ...", file=sys.stderr)
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        print(
            f"[shutup] Model loaded. Embedding dimension: {self._dimension}",
            file=sys.stderr,
        )

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        return self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

    def get_dimension(self) -> int:
        return self._dimension


class OllamaEmbedder(BaseEmbedder):
    """Embedding provider using Ollama (local server, privacy-focused)."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        """Initialize the Ollama embedder.

        Args:
            model_name: Name of the Ollama embedding model.
                        Default "nomic-embed-text" is recommended.
            host: Ollama server URL.
        """
        try:
            import ollama
        except ImportError:
            print(
                "[shutup] Error: ollama package not installed. "
                "Run: pip install ollama",
                file=sys.stderr,
            )
            sys.exit(1)

        self.client = ollama.Client(host=host)
        self.model_name = model_name
        self.host = host

        print(
            f"[shutup] Connecting to Ollama at {host} with model '{model_name}' ...",
            file=sys.stderr,
        )
        try:
            test_embedding = self.client.embed(model=self.model_name, input="test")
            self._dimension = len(test_embedding["embeddings"][0])
            print(
                f"[shutup] Ollama connected. Embedding dimension: {self._dimension}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"[shutup] Error: Could not connect to Ollama. "
                f"Is it running? ({e})",
                file=sys.stderr,
            )
            print(
                "[shutup] Install Ollama from https://ollama.com and run: "
                "ollama pull nomic-embed-text",
                file=sys.stderr,
            )
            sys.exit(1)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Ollama."""
        if isinstance(texts, str):
            texts = [texts]
        response = self.client.embed(model=self.model_name, input=texts)
        return np.array(response["embeddings"])

    def get_dimension(self) -> int:
        return self._dimension


def create_embedder(backend: str = "sentence-transformers", **kwargs) -> BaseEmbedder:
    """Factory function to create an embedder instance.

    Args:
        backend: "sentence-transformers" or "ollama".
        **kwargs: Additional arguments passed to the embedder constructor.

    Returns:
        A BaseEmbedder instance.
    """
    if backend == "ollama":
        return OllamaEmbedder(**kwargs)
    else:
        return SentenceTransformerEmbedder(**kwargs)


class ToolEmbedder:
    """High-level tool indexing and search using an embedder backend."""

    def __init__(self, embedder: Optional[BaseEmbedder] = None, backend: str = "sentence-transformers"):
        """Initialize the tool embedder.

        Args:
            embedder: Optional pre-configured embedder instance.
            backend: Embedder backend to use if embedder is not provided.
        """
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = create_embedder(backend)

        self.tools: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, tools: List[dict]) -> None:
        """Build the embedding index from a list of tool definitions.

        Each tool dict should have 'name' and 'description' fields.
        """
        if not tools:
            print("[shutup] Warning: No tools provided to build index.", file=sys.stderr)
            self.tools = []
            self.embeddings = None
            return

        self.tools = tools
        texts = [f"{t['name']}: {t.get('description', '')}" for t in tools]
        print(f"[shutup] Building embeddings for {len(texts)} tools...", file=sys.stderr)
        self.embeddings = self.embedder.encode(texts)
        print("[shutup] Embedding index built.", file=sys.stderr)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for the top_k most relevant tools given a user intent query.

        Args:
            query: The user's intent or task description.
            top_k: Number of tools to return.

        Returns:
            List of tool dicts ordered by relevance (most relevant first).
        """
        if not self.tools or self.embeddings is None:
            return []

        if len(self.tools) == 0:
            return []

        query_embedding = self.embedder.encode([query])[0]

        # Cosine similarity
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )

        # Get indices of top_k scores
        if top_k >= len(self.tools):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [self.tools[i] for i in top_indices]

    def get_tool_by_name(self, name: str) -> Optional[dict]:
        """Retrieve a tool by its name."""
        for tool in self.tools:
            if tool.get("name") == name:
                return tool
        return None
