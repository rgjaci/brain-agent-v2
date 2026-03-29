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

from .provider import LLMProvider, OllamaProvider
from .embeddings import GeminiEmbeddingProvider, LocalEmbeddingProvider
from .tool_parser import ToolCallParser, ToolCall

__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "GeminiEmbeddingProvider",
    "LocalEmbeddingProvider",
    "ToolCallParser",
    "ToolCall",
]
