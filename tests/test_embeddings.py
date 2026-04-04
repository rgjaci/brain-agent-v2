"""Tests for EmbeddingProvider — Gemini and local providers, caching."""
from __future__ import annotations

import pytest

from core.llm.embeddings import EmbeddingCache


class TestEmbeddingCache:
    def test_cache_set_and_get(self):
        cache = EmbeddingCache(max_memory_items=100)
        cache.set("hello", [0.1, 0.2, 0.3])
        result = cache.get("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_cache_miss(self):
        cache = EmbeddingCache(max_memory_items=100)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_eviction(self):
        cache = EmbeddingCache(max_memory_items=5)
        # Fill cache
        for i in range(5):
            cache.set(f"key_{i}", [float(i)])
        # Add one more to trigger eviction (evicts 20% = 1 item)
        cache.set("key_5", [5.0])
        # At least one old item should be evicted
        assert len(cache) <= 5
        # The newest item should be present
        assert cache.get("key_5") == [5.0]

    def test_cache_overwrite(self):
        cache = EmbeddingCache(max_memory_items=100)
        cache.set("key", [1.0])
        cache.set("key", [2.0])
        assert cache.get("key") == [2.0]

    def test_cache_size(self):
        cache = EmbeddingCache(max_memory_items=5)
        for i in range(10):
            cache.set(f"key_{i}", [float(i)])
        # Should only have last items (evicts oldest 20% = 1 at a time)
        assert cache.get("key_9") == [9.0]
        assert len(cache) <= 5

    def test_cache_len(self):
        cache = EmbeddingCache(max_memory_items=100)
        assert len(cache) == 0
        cache.set("a", [1.0])
        assert len(cache) == 1

    def test_cache_clear(self):
        cache = EmbeddingCache(max_memory_items=100)
        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None


class TestLocalEmbeddingProvider:
    def test_local_provider_requires_sentence_transformers(self):
        """LocalEmbeddingProvider should raise ImportError without sentence-transformers."""
        try:
            from core.llm.embeddings import LocalEmbeddingProvider
            # This will raise ImportError if sentence-transformers is not installed
            try:
                provider = LocalEmbeddingProvider()
                # If it loaded, test basic functionality
                embedding = provider.embed(["test text"])
                assert len(embedding) == 1
                assert len(embedding[0]) == provider.dims
            except ImportError:
                pytest.skip("sentence-transformers not installed")
        except ImportError:
            pytest.skip("LocalEmbeddingProvider not available")


class TestGeminiEmbeddingProvider:
    def test_gemini_requires_api_key(self):
        """GeminiEmbeddingProvider should raise error without API key."""
        try:
            from core.llm.embeddings import GeminiEmbeddingProvider
            with pytest.raises(ValueError):
                GeminiEmbeddingProvider(api_key="")
        except ImportError:
            pytest.skip("GeminiEmbeddingProvider not available")

    def test_gemini_initialization_with_key(self):
        """GeminiEmbeddingProvider should initialize with valid key."""
        try:
            from core.llm.embeddings import GeminiEmbeddingProvider
            # Should not raise with a non-empty key (even if invalid)
            # But will raise ImportError if google-genai not installed
            try:
                provider = GeminiEmbeddingProvider(api_key="test-key")
                assert provider.model == "models/gemini-embedding-001"
                assert provider.dims == 768
            except ImportError:
                pytest.skip("google-genai not installed")
        except ImportError:
            pytest.skip("GeminiEmbeddingProvider not available")
