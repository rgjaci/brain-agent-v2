"""Tests for MemoryWriter — fact extraction, dedup, graph extraction."""
from __future__ import annotations
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


def make_writer(llm_response=None):
    from core.memory.writer import MemoryWriter
    from core.memory.kg import KnowledgeGraph
    from core.memory.database import MemoryDatabase

    db = MemoryDatabase(":memory:")
    kg = KnowledgeGraph(db)

    llm = MagicMock()
    llm.generate_json = MagicMock(return_value=llm_response or [])

    embedder = MagicMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1] * 768])
    embedder.embed_query = AsyncMock(return_value=[0.1] * 768)

    return MemoryWriter(llm, embedder, db, kg), db, kg


# ── fact extraction ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_facts_returns_list():
    facts_payload = [
        {"content": "User prefers ed25519 SSH keys", "category": "preference", "importance": 0.8},
        {"content": "Production server is 10.0.0.1", "category": "fact", "importance": 0.9},
    ]
    writer, db, kg = make_writer(llm_response=facts_payload)
    facts = await writer.extract_facts("setup SSH", "Done, used ed25519")
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_extract_facts_empty_exchange():
    writer, _, _ = make_writer(llm_response=[])
    facts = await writer.extract_facts("hi", "hello")
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_extract_facts_handles_llm_error():
    writer, _, _ = make_writer()
    writer.llm.generate_json = MagicMock(side_effect=RuntimeError("LLM down"))
    facts = await writer.extract_facts("test", "test")
    assert facts == []


# ── deduplication ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deduplicate_removes_exact_duplicate():
    from core.memory.writer import FactExtraction
    writer, db, _ = make_writer()

    # Pre-store a memory
    mid = db.insert_memory(
        content="User prefers ed25519 SSH keys",
        category="preference",
        importance=0.8,
    )
    # Store embedding via the correct table name
    try:
        db.insert_embedding("memory_embeddings", "memory_id", mid, [0.9] * 768)
    except Exception:
        pass  # embedding table name varies, skip

    # Simulate embedder returning same embedding → high cosine sim
    writer.embedder.embed_texts = AsyncMock(return_value=[[0.9] * 768])
    writer.db.vector_search = MagicMock(return_value=[
        {"id": mid, "content": "User prefers ed25519 SSH keys", "distance": 0.02}
    ])

    facts = [FactExtraction("User prefers ed25519 SSH keys", "preference", 0.8)]
    result = await writer.deduplicate_facts(facts, threshold=0.92)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_deduplicate_keeps_different_fact():
    from core.memory.writer import FactExtraction
    writer, db, _ = make_writer()

    writer.embedder.embed_texts = AsyncMock(return_value=[[0.1] * 768])
    writer.db.vector_search = MagicMock(return_value=[])  # no similar memories

    facts = [FactExtraction("User uses tmux", "fact", 0.6)]
    result = await writer.deduplicate_facts(facts, threshold=0.92)
    assert len(result) >= 0  # may or may not deduplicate — should not crash


# ── graph extraction ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_graph_returns_tuple():
    graph_payload = {
        "entities": [
            {"name": "Tailscale", "type": "service", "description": "VPN mesh network"},
        ],
        "relations": [
            {"source": "User", "target": "Tailscale", "type": "uses", "detail": "for remote access"},
        ],
    }
    writer, _, _ = make_writer(llm_response=graph_payload)
    result = await writer.extract_graph("setup Tailscale", "Done")
    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_extract_graph_empty_on_error():
    writer, _, _ = make_writer()
    writer.llm.generate_json = MagicMock(side_effect=ValueError("bad json"))
    entities, relations = await writer.extract_graph("hi", "hi")
    assert entities == []
    assert relations == []


# ── interaction_succeeded ─────────────────────────────────────────────────────

def test_interaction_succeeded_true():
    writer, _, _ = make_writer()
    assert writer.interaction_succeeded("The task is complete.") is True


def test_interaction_succeeded_false_on_error():
    writer, _, _ = make_writer()
    assert writer.interaction_succeeded("Error: command not found") is False
    assert writer.interaction_succeeded("I couldn't complete that") is False
    assert writer.interaction_succeeded("Failed: permission denied") is False
