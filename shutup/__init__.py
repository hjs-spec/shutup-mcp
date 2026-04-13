"""shutup - An MCP proxy that shows agents only the tools they actually need."""

from .proxy import ShutupProxy
from .retriever import HybridRetriever
from .embedder import create_embedder, BaseEmbedder
from .server_manager import ServerManager

__version__ = "0.2.0"
__all__ = ["ShutupProxy", "HybridRetriever", "create_embedder", "BaseEmbedder", "ServerManager"]
