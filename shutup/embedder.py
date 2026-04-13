"""Embedding and similarity search for MCP tools."""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any


class ToolEmbedder:
    """Generates embeddings for tool descriptions and finds most relevant tools."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence-transformers model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                        Default is "all-MiniLM-L6-v2" (runs locally, ~80MB).
        """
        print(f"[shutup] Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.tools: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.array([])

    def build_index(self, tools: List[Dict[str, Any]]) -> None:
        """
        Build the embedding index from a list of tool definitions.

        Each tool should have 'name' and 'description' fields.
        """
        if not tools:
            print("[shutup] Warning: No tools provided to build index.")
            self.tools = []
            self.embeddings = np.array([])
            return

        self.tools = tools
        texts = [f"{t['name']}: {t.get('description', '')}" for t in tools]
        print(f"[shutup] Building embeddings for {len(texts)} tools...")
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        print("[shutup] Embedding index built.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the top_k most relevant tools given a user intent query.

        Args:
            query: The user's intent or task description.
            top_k: Number of tools to return.

        Returns:
            List of tool definitions ordered by relevance (most relevant first).
        """
        if len(self.tools) == 0:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

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

    def update_tool(self, tool_name: str, new_description: str) -> None:
        """
        Update a single tool's description and re-compute its embedding.
        (Simplified: rebuild full index for now)
        """
        for i, tool in enumerate(self.tools):
            if tool.get("name") == tool_name:
                self.tools[i]["description"] = new_description
                break
        # For MVP, just rebuild the whole index
        if self.tools:
            self.build_index(self.tools)
