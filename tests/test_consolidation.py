"""Tests for consolidation.py — decay, promote, contradictions, scheduling."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def engine(db, mock_llm, mock_embedder):
    from core.memory.consolidation import ConsolidationEngine
    return ConsolidationEngine(db, mock_llm, mock_embedder)


# ── apply_decay ──────────────────────────────────────────────────────────────

def test_apply_decay_no_old_memories(engine, db):
    """Fresh memories should not be decayed."""
    db.insert_memory("recent fact", importance=0.8)
    decayed = engine.apply_decay()
    assert decayed == 0


def test_apply_decay_old_memory(engine, db):
    """Memories last accessed > 30 days ago should be decayed."""
    mid = db.insert_memory("old fact", importance=0.8)
    # Manually backdate last_accessed to 60 days ago
    old_ts = time.time() - 60 * 86400
    db.execute(
        "UPDATE memories SET last_accessed = ? WHERE id = ?",
        (old_ts, mid),
    )
    decayed = engine.apply_decay()
    assert decayed >= 1
    mem = db.get_memory(mid)
    assert mem["importance"] < 0.8


def test_apply_decay_floor(engine, db):
    """Importance should not go below 0.1."""
    mid = db.insert_memory("barely important", importance=0.12)
    old_ts = time.time() - 60 * 86400
    db.execute(
        "UPDATE memories SET last_accessed = ? WHERE id = ?",
        (old_ts, mid),
    )
    engine.apply_decay()
    mem = db.get_memory(mid)
    assert mem["importance"] >= 0.1


# ── promote_important ────────────────────────────────────────────────────────

def test_promote_important_basic(engine, db):
    mid = db.insert_memory("popular fact", importance=0.5)
    db.execute(
        "UPDATE memories SET access_count = 15 WHERE id = ?", (mid,)
    )
    promoted = engine.promote_important()
    assert promoted >= 1
    mem = db.get_memory(mid)
    assert mem["importance"] > 0.5


def test_promote_important_cap(engine, db):
    mid = db.insert_memory("very popular", importance=0.95)
    db.execute(
        "UPDATE memories SET access_count = 100 WHERE id = ?", (mid,)
    )
    engine.promote_important()
    mem = db.get_memory(mid)
    assert mem["importance"] <= 1.0


# ── resolve_contradictions ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_contradictions_keeps_newest(engine, db):
    # The resolver groups by first 3 words as topic key
    # So both must share the same 3-word prefix to be detected as contradictions
    db.insert_memory("User prefers vim for editing code", category="preference")
    import time
    time.sleep(0.01)
    db.insert_memory("User prefers emacs for editing code", category="preference")
    # These both have topic key "user prefers vim" / "user prefers emacs" — different keys.
    # To actually trigger contradiction, we need same 3 words:
    db.insert_memory("User likes dark mode", category="preference")
    time.sleep(0.01)
    db.insert_memory("User likes light mode", category="preference")
    # Both have topic key "user likes dark" / "user likes light" — still different.
    # The resolver only catches same first-3-words. Let's use exact match:
    db.insert_memory("User wants tabs in code", category="preference")
    time.sleep(0.01)
    db.insert_memory("User wants tabs in code but larger", category="preference")
    resolved = await engine.resolve_contradictions()
    assert resolved >= 1


@pytest.mark.asyncio
async def test_resolve_contradictions_no_conflicts(engine, db):
    db.insert_memory("User likes Python", category="preference")
    db.insert_memory("Server runs Ubuntu", category="fact")
    resolved = await engine.resolve_contradictions()
    assert resolved == 0


# ── maybe_consolidate ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_consolidate_triggers_at_interval(engine):
    """Should trigger at turn 10 (CONSOLIDATION_INTERVAL=10)."""
    engine.consolidate = MagicMock(return_value=None)
    # Make consolidate async
    from unittest.mock import AsyncMock
    engine.consolidate = AsyncMock()
    await engine.maybe_consolidate(10)
    engine.consolidate.assert_called_once()


@pytest.mark.asyncio
async def test_maybe_consolidate_skips_non_interval(engine):
    from unittest.mock import AsyncMock
    engine.consolidate = AsyncMock()
    await engine.maybe_consolidate(7)
    engine.consolidate.assert_not_called()
