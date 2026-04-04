"""brain_agent.core.llm — LLM provider layer for Brain Agent v2.

Public API
----------
LLMProvider
    Abstract base class all providers must implement.
OllamaProvider
    Local Ollama-backed provider (chat + JSON generation).
GeminiEmbeddingProvider
    Google Gemini ``text-embedding-004`` via the free-tier API.
LocalEmbeddingProvider
    Offline sentence-transformers fallback for embeddings.
ToolCallParser
    Parse XML-formatted ``<tool>`` calls from LLM responses.
ToolCall
    Dataclass representing a single parsed tool invocation.
"""

from .embeddings import GeminiEmbeddingProvider, LocalEmbeddingProvider
from .provider import LLMProvider, OllamaProvider
from .tool_parser import ToolCall, ToolCallParser

__all__ = [
    "GeminiEmbeddingProvider",
    "LLMProvider",
    "LocalEmbeddingProvider",
    "OllamaProvider",
    "ToolCall",
    "ToolCallParser",
]
