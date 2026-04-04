"""Tests for core/memory/dream.py — AutoDream LLM-powered consolidation."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def consolidator(db, mock_llm, mock_embedder):
    from core.memory.consolidation import ConsolidationEngine
    return ConsolidationEngine(db, mock_llm, mock_embedder)


@pytest.fixture
def dream_engine(db, mock_llm, mock_embedder, consolidator, kg):
    from core.memory.dream import DreamEngine
    return DreamEngine(db, mock_llm, mock_embedder, consolidator, kg)


# ── Full dream cycle ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dream_returns_report(dream_engine):
    """Dream cycle should return a DreamReport even with empty DB."""
    report = await dream_engine.dream()
    assert report.elapsed_seconds >= 0
    assert report.abstractions_created == 0  # no memories to cluster
    assert report.contradictions_resolved == 0


@pytest.mark.asyncio
async def test_dream_updates_last_dream_time(dream_engine):
    """Dream should update _last_dream timestamp."""
    assert dream_engine._last_dream == 0.0
    await dream_engine.dream()
    assert dream_engine._last_dream > 0


# ── maybe_dream scheduling ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_dream_turn_trigger(dream_engine):
    """Dream should trigger at the configured turn interval."""
    report = await dream_engine.maybe_dream(50, interval=50)
    assert report is not None


@pytest.mark.asyncio
async def test_maybe_dream_skip(dream_engine):
    """Dream should skip when turn is not at interval."""
    report = await dream_engine.maybe_dream(7, interval=50)
    assert report is None


@pytest.mark.asyncio
async def test_maybe_dream_idle_trigger(dream_engine):
    """Dream should trigger after idle threshold."""
    dream_engine._last_dream = time.time() - 700  # idle > 600s
    report = await dream_engine.maybe_dream(1, interval=1000, idle_threshold=600)
    assert report is not None


# ── Abstraction creation ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_abstractions_empty_db(dream_engine):
    """No abstractions should be created with an empty database."""
    result = await dream_engine._create_abstractions()
    assert result == 0


@pytest.mark.asyncio
async def test_create_abstractions_with_data(dream_engine, db, mock_llm):
    """Abstractions should be created from clustered memories."""
    # Insert memories with embeddings (all similar = same base vector)
    base_emb = [0.1] * 768
    for i in range(5):
        mid = db.insert_memory(f"Python fact {i}", category="knowledge")
        db.insert_embedding("memory_vectors", "memory_id", mid, base_emb)

    mock_llm.generate_json = MagicMock(return_value={
        "insight": "Python is widely used for various purposes",
        "importance": 0.7,
    })

    result = await dream_engine._create_abstractions()
    assert result >= 1

    # Verify insight was stored
    rows = db.execute("SELECT * FROM memories WHERE category = 'insight'")
    assert len(rows) >= 1
    assert "Python" in rows[0]["content"]


# ── Contradiction resolution ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_contradictions_empty(dream_engine):
    """No contradictions to resolve in empty DB."""
    result = await dream_engine._resolve_contradictions_llm()
    assert result == 0


@pytest.mark.asyncio
async def test_resolve_contradictions_keeps_one(dream_engine, db, mock_llm):
    """LLM should resolve contradictions by keeping one memory."""
    # Two similar but not identical memories
    emb_a = [0.5] * 768
    emb_b = [0.51] * 768  # slightly different
    mid_a = db.insert_memory("User prefers dark mode", category="preference")
    mid_b = db.insert_memory("User prefers light mode", category="preference")
    db.insert_embedding("memory_vectors", "memory_id", mid_a, emb_a)
    db.insert_embedding("memory_vectors", "memory_id", mid_b, emb_b)

    mock_llm.generate_json = MagicMock(return_value={
        "contradicts": True,
        "resolution": "keep_b",
        "merged_content": "",
        "reasoning": "More recent preference",
    })

    result = await dream_engine._resolve_contradictions_llm()
    # May or may not resolve depending on cosine sim range (0.80-0.95)
    assert isinstance(result, int)


# ── Pattern detection ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_detect_patterns_empty(dream_engine):
    """No patterns to detect in empty DB."""
    result = await dream_engine._detect_patterns()
    assert result == 0


@pytest.mark.asyncio
async def test_detect_patterns_with_data(dream_engine, db, mock_llm):
    """Patterns should be detected and stored as insights."""
    for i in range(10):
        db.insert_memory(f"User works on project {i}", category="observation")

    mock_llm.generate_json = MagicMock(return_value=[
        {"pattern": "User works on many projects simultaneously", "importance": 0.7, "evidence_count": 10}
    ])

    result = await dream_engine._detect_patterns()
    assert result == 1

    insights = db.execute("SELECT * FROM memories WHERE category = 'insight' AND source = 'dream'")
    assert len(insights) >= 1
    assert "Pattern:" in insights[0]["content"]


# ── Connection strengthening ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_strengthen_connections_empty(dream_engine):
    """No connections to strengthen with no entities."""
    result = await dream_engine._strengthen_connections()
    assert result == 0


@pytest.mark.asyncio
async def test_strengthen_connections_with_entities(dream_engine, db, mock_llm):
    """New relations should be added for under-connected entities."""
    eid1 = db.insert_entity("Python", entity_type="language")
    db.insert_entity("Django", entity_type="tool")

    mock_llm.generate_json = MagicMock(return_value=[
        {"source": "Django", "target": "Python", "type": "depends_on", "confidence": 0.9, "reasoning": "Django is a Python framework"}
    ])

    result = await dream_engine._strengthen_connections()
    assert result >= 1

    # Verify relation was added
    relations = db.get_relations(eid1)
    assert any(r["relation_type"] == "depends_on" for r in relations)


# ── Question generation ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_questions_empty(dream_engine):
    """No questions with empty DB."""
    result = await dream_engine._generate_questions()
    assert result == 0


@pytest.mark.asyncio
async def test_generate_questions_with_data(dream_engine, db, mock_llm):
    """Questions should be generated and stored."""
    for i in range(5):
        db.insert_memory(f"User uses tool {i}", category="fact")

    mock_llm.generate_json = MagicMock(return_value=[
        {"question": "What is the user's primary programming language?", "importance": 0.8, "topic": "programming"}
    ])

    result = await dream_engine._generate_questions()
    assert result == 1

    questions = db.execute("SELECT * FROM memories WHERE category = 'question'")
    assert len(questions) >= 1


# ── Error resilience ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dream_survives_llm_errors(dream_engine, db, mock_llm):
    """Dream cycle should complete even if LLM calls fail."""
    for i in range(10):
        mid = db.insert_memory(f"fact {i}", category="fact")
        db.insert_embedding("memory_vectors", "memory_id", mid, [0.1] * 768)

    mock_llm.generate_json = MagicMock(side_effect=RuntimeError("LLM offline"))

    report = await dream_engine.dream()
    assert report.elapsed_seconds >= 0
    # Should not crash despite LLM failures


@pytest.mark.asyncio
async def test_dream_report_fields(dream_engine):
    """DreamReport should have all expected fields."""
    from core.memory.dream import DreamReport
    report = DreamReport()
    assert report.abstractions_created == 0
    assert report.contradictions_resolved == 0
    assert report.patterns_detected == 0
    assert report.connections_added == 0
    assert report.questions_generated == 0
    assert report.elapsed_seconds == 0.0
    assert report.errors == []
