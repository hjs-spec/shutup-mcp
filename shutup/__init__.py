"""shutup - An MCP proxy that shows agents only the tools they actually need."""

from .proxy import ShutupProxy
from .embedder import ToolEmbedder

__version__ = "0.1.0"
__all__ = ["ShutupProxy", "ToolEmbedder"]
