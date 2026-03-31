from __future__ import annotations

import hashlib
import logging
import math
import time
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Thread-safe in-memory LRU-style cache for embeddings.

    Items are keyed by the SHA-256 of the input text, so identical strings
    across sessions always produce cache hits without collision risk.

    Args:
        cache_dir:        Optional ``Path`` for future on-disk persistence
                          (directory is created if it does not exist).
        max_memory_items: Maximum number of embeddings held in RAM before the
                          oldest 20 % are evicted.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_items: int = 10_000,
    ) -> None:
        self._memory: dict[str, list[float]] = {}
        self._max_items = max_memory_items
        self._lock = threading.Lock()
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[list[float]]:
        """Return cached embedding for *text*, or ``None`` on a cache miss."""
        key = self._key(text)
        with self._lock:
            return self._memory.get(key)

    def set(self, text: str, embedding: list[float]) -> None:
        """Store *embedding* for *text*, evicting old entries when full."""
        key = self._key(text)
        with self._lock:
            if len(self._memory) >= self._max_items:
                # Evict oldest 20 % (dict preserves insertion order in Python 3.7+)
                cutoff = self._max_items // 5
                to_remove = list(self._memory.keys())[:cutoff]
                for k in to_remove:
                    del self._memory[k]
            self._memory[key] = embedding

    def __len__(self) -> int:
        with self._lock:
            return len(self._memory)

    def clear(self) -> None:
        """Flush all cached embeddings from memory."""
        with self._lock:
            self._memory.clear()


# ---------------------------------------------------------------------------
# Gemini embedding provider
# ---------------------------------------------------------------------------


class GeminiEmbeddingProvider:
    """Embed texts using Google's *text-embedding-004* model (free tier).

    Free-tier limits (as of 2025):
        - 1 500 requests / minute
        - Up to 100 texts per batch request

    The provider is cache-aware and rate-limit-aware out of the box.

    Args:
        api_key:   Gemini API key.  Obtain one at
                   https://aistudio.google.com/app/apikey
        cache_dir: Optional directory for the embedding cache.
        model:     Gemini embedding model name (default ``text-embedding-004``).
        dims:      Expected dimensionality of returned vectors (default 768).

    Raises:
        ValueError:  When *api_key* is empty.
        ImportError: When ``google-genai`` is not installed.
    """

    MODEL = "models/gemini-embedding-001"
    DIMS = 768
    RATE_LIMIT_RPM = 1_500  # Free tier
    BATCH_SIZE = 100         # Max texts per API request

    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[Path] = None,
        model: str = MODEL,
        dims: int = DIMS,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY required. "
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        self.api_key = api_key
        self.model = model
        self.dims = dims
        self._cache = EmbeddingCache(cache_dir)
        self._last_request: float = 0.0
        self._min_interval: float = 60.0 / self.RATE_LIMIT_RPM

        try:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "google-genai package required. "
                "Install with: pip install google-genai"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if necessary to stay within the API rate limit."""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Send a single batched embedding request to Gemini.

        Args:
            texts: List of strings (max ``BATCH_SIZE`` items).

        Returns:
            List of embedding vectors in the same order as *texts*.

        Raises:
            Exception: Re-raises any API-level errors after logging.
        """
        self._rate_limit()
        try:
            from google.genai import types as genai_types

            result = self._client.models.embed_content(
                model=self.model,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    outputDimensionality=self.dims,
                ),
            )
            return [e.values for e in result.embeddings]
        except Exception as exc:
            logger.error("Gemini embedding error: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings, using the cache and batching automatically.

        Cache hits are returned immediately; remaining texts are batched into
        groups of ``BATCH_SIZE`` and sent to the Gemini API.

        Args:
            texts: Strings to embed.  Order is preserved in the return value.

        Returns:
            List of embedding vectors aligned with *texts*.
        """
        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results  # type: ignore[return-value]

        # Batch-embed uncached texts
        for batch_start in range(0, len(uncached_texts), self.BATCH_SIZE):
            batch = uncached_texts[batch_start : batch_start + self.BATCH_SIZE]
            embeddings = self._embed_batch(batch)

            for j, embedding in enumerate(embeddings):
                idx = uncached_indices[batch_start + j]
                results[idx] = embedding
                self._cache.set(uncached_texts[batch_start + j], embedding)

        return results  # type: ignore[return-value]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Convenience wrapper around :meth:`embed`.
        """
        return self.embed([query])[0]

    # ------------------------------------------------------------------
    # Similarity utilities
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute the cosine similarity between two embedding vectors.

        Returns a value in ``[-1, 1]`` (typically ``[0, 1]`` for text).
        Returns ``0.0`` if either vector has zero magnitude.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Local / offline embedding provider
# ---------------------------------------------------------------------------


class LocalEmbeddingProvider:
    """Offline embedding provider backed by *sentence-transformers*.

    Useful as a fallback when the Gemini API is unavailable (e.g. no internet
    access, quota exhausted).

    Args:
        model_name: Any `sentence-transformers
                    <https://www.sbert.net/docs/pretrained_models.html>`_
                    model name.  Defaults to the lightweight
                    ``all-MiniLM-L6-v2`` (384 dims, ~80 MB).

    Raises:
        ImportError: When ``sentence-transformers`` is not installed.

    Example::

        provider = LocalEmbeddingProvider()
        vectors = provider.embed(["hello world", "foo bar"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            self.dims: int = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for offline mode. "
                "Install with: pip install sentence-transformers"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings using the local model.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors (Python lists of ``float``).
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([query])[0]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute the cosine similarity between two embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
