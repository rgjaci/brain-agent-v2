"""Shared pytest fixtures for brain_agent test suite."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── async support ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


# ── in-memory database ────────────────────────────────────────────────────────

@pytest.fixture
def db():
    from core.memory.database import MemoryDatabase
    return MemoryDatabase(":memory:")


# ── mock LLM ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="mocked response")
    llm.generate_json = MagicMock(return_value=[])
    llm.count_tokens = MagicMock(return_value=10)
    return llm


# ── mock embedder ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.embed_texts = AsyncMock(return_value=[[0.1] * 768])
    emb.embed_query = AsyncMock(return_value=[0.1] * 768)
    return emb


# ── knowledge graph ───────────────────────────────────────────────────────────

@pytest.fixture
def kg(db):
    from core.memory.kg import KnowledgeGraph
    return KnowledgeGraph(db)


# ── writer ────────────────────────────────────────────────────────────────────

@pytest.fixture
def writer(mock_llm, mock_embedder, db, kg):
    from core.memory.writer import MemoryWriter
    return MemoryWriter(mock_llm, mock_embedder, db, kg)


# ── reader ────────────────────────────────────────────────────────────────────

@pytest.fixture
def reader(db, mock_embedder, kg, mock_llm):
    from core.memory.reader import MemoryReader
    return MemoryReader(db, mock_embedder, kg, mock_llm)


# ── reranker ──────────────────────────────────────────────────────────────────

@pytest.fixture
def feedback(db):
    from core.memory.feedback import RetrievalFeedbackCollector
    return RetrievalFeedbackCollector(db)


@pytest.fixture
def reranker(feedback):
    from core.memory.reranker import Reranker
    return Reranker(feedback_collector=feedback)


# ── procedures ────────────────────────────────────────────────────────────────

@pytest.fixture
def procedures(db):
    from core.memory.procedures import ProcedureStore
    return ProcedureStore(db)
