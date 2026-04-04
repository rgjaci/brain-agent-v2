"""Tests for MemoryReader — RRF fusion, heuristic rerank, adaptive strategy."""
from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock


def make_reader():
    from core.memory.database import MemoryDatabase
    from core.memory.kg import KnowledgeGraph
    from core.memory.reader import MemoryReader

    db = MemoryDatabase(":memory:")
    embedder = MagicMock()
    embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
    kg = KnowledgeGraph(db)
    reader = MemoryReader(db, embedder, kg)
    return reader, db


def _insert_mem(db, content, category="fact", importance=0.5):
    """Insert a real memory row and return its id."""
    mid = db.insert_memory(content=content, category=category, importance=importance)
    return mid


# ── RRF fusion ────────────────────────────────────────────────────────────────

def test_rrf_fuse_merges_hits():
    reader, db = make_reader()
    # Insert real memories so get_memory() can fetch them
    id1 = _insert_mem(db, "fact a", importance=0.8)
    id2 = _insert_mem(db, "fact b", importance=0.5)
    id3 = _insert_mem(db, "fact c", importance=0.6)

    dense  = [{"id": id1}, {"id": id2}]
    sparse = [{"id": id2}, {"id": id3}]
    fused = reader.rrf_fuse(dense, sparse)
    ids = [m.id for m in fused]
    assert id1 in ids and id2 in ids and id3 in ids
    # id2 appeared in both lists — should rank higher than id1 or id3
    id2_score = next(m.rrf_score for m in fused if m.id == id2)
    id1_score = next(m.rrf_score for m in fused if m.id == id1)
    assert id2_score > id1_score


def test_rrf_fuse_empty_inputs():
    reader, _db = make_reader()
    assert reader.rrf_fuse([], []) == []


def test_rrf_fuse_single_list():
    reader, db = make_reader()
    mid = _insert_mem(db, "single fact")
    result = reader.rrf_fuse([{"id": mid}], [])
    assert len(result) == 1
    assert result[0].id == mid


def test_rrf_fuse_skips_missing_ids():
    reader, _db = make_reader()
    # ID 99999 doesn't exist in DB
    result = reader.rrf_fuse([{"id": 99999}], [])
    assert result == []


# ── heuristic rerank ──────────────────────────────────────────────────────────

def _make_retrieved_mem(db, content, category="fact", importance=0.5,
                        access_count=1, created_at=None):
    from core.memory.reader import RetrievedMemory
    if created_at is None:
        created_at = datetime.datetime.now().isoformat()
    mid = _insert_mem(db, content, category, importance)
    return RetrievedMemory(
        id=mid,
        content=content,
        category=category,
        importance=importance,
        access_count=access_count,
        rrf_score=1.0,
    )


def test_heuristic_rerank_boosts_recent():
    import time
    reader, db = make_reader()
    now_ts   = time.time()
    old_ts   = now_ts - 86400 * 60  # 60 days ago

    m1 = _make_retrieved_mem(db, "old fact",    importance=0.7)
    m2 = _make_retrieved_mem(db, "recent fact", importance=0.7)
    # created_at must be a float (unix timestamp) in metadata
    m1.metadata["created_at"] = old_ts
    m2.metadata["created_at"] = now_ts
    m1.rrf_score = 1.0
    m2.rrf_score = 1.0

    ranked = reader.heuristic_rerank([m1, m2], "some query")
    assert ranked[0].id == m2.id


def test_heuristic_rerank_boosts_procedure_for_how_to_query():
    import time
    reader, db = make_reader()
    now_ts = time.time()

    m1 = _make_retrieved_mem(db, "random fact",   category="fact",      importance=0.5)
    m2 = _make_retrieved_mem(db, "step by step",  category="procedure", importance=0.5)
    m1.metadata["created_at"] = now_ts
    m2.metadata["created_at"] = now_ts
    m1.rrf_score = 1.0
    m2.rrf_score = 1.0

    # "how to" query triggers 1.5x boost for procedure category
    ranked = reader.heuristic_rerank([m1, m2], "how to set up SSH")
    assert ranked[0].id == m2.id


def test_heuristic_rerank_returns_same_count():
    reader, db = make_reader()
    mems = [_make_retrieved_mem(db, f"fact {i}") for i in range(5)]
    for m in mems:
        m.rrf_score = 1.0
    ranked = reader.heuristic_rerank(mems, "test")
    assert len(ranked) == 5


# ── adaptive strategy ─────────────────────────────────────────────────────────

def test_adaptive_strategy_conservative_for_short():
    reader, _ = make_reader()
    assert reader.adaptive_strategy("hi", "") == "conservative"


def test_adaptive_strategy_aggressive_for_memory_query():
    reader, _ = make_reader()
    assert reader.adaptive_strategy("remember what I said about SSH?", "") == "aggressive"


def test_adaptive_strategy_normal_default():
    reader, _ = make_reader()
    strategy = reader.adaptive_strategy("How do I set up a cronjob for backups?", "")
    assert strategy in ("normal", "aggressive")


# ── entity extraction ─────────────────────────────────────────────────────────

def test_extract_query_entities_finds_capitalized():
    reader, _ = make_reader()
    entities = reader.extract_query_entities("Set up Tailscale on Ubuntu server")
    assert "Tailscale" in entities or "Ubuntu" in entities


def test_extract_query_entities_no_stopwords():
    reader, _ = make_reader()
    entities = reader.extract_query_entities("What is the best way to do this?")
    assert "What" not in entities
    assert "the" not in entities
