"""Integration-style tests for adaptive retrieval pipeline (MemoryReader end-to-end)."""
from __future__ import annotations
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


def make_reader_with_data():
    from core.memory.reader import MemoryReader
    from core.memory.database import MemoryDatabase
    from core.memory.kg import KnowledgeGraph

    db = MemoryDatabase(":memory:")
    embedder = MagicMock()
    embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
    kg = KnowledgeGraph(db)

    # Insert some test memories directly
    ids = []
    for content, category in [
        ("User prefers ed25519 SSH keys", "preference"),
        ("Production server IP is 10.0.0.1", "fact"),
        ("User corrected: use nginx not apache", "correction"),
        ("Python 3.12 is installed on the server", "fact"),
        ("remember: deploy with rsync, not scp", "preference"),
    ]:
        mid = db.insert_memory(content=content, category=category, importance=0.7)
        ids.append(mid)

    reader = MemoryReader(db, embedder, kg)
    return reader, db, kg, ids


# ── retrieve ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_returns_result():
    reader, db, kg, ids = make_reader_with_data()
    reader.db.vector_search = MagicMock(return_value=[])
    reader.db.fts_search = MagicMock(return_value=[])
    result = await reader.retrieve("SSH setup")
    assert hasattr(result, "memories") or isinstance(result, list)


@pytest.mark.asyncio
async def test_retrieve_conservative_strategy():
    reader, db, kg, ids = make_reader_with_data()
    reader.db.vector_search = MagicMock(return_value=[])
    reader.db.fts_search = MagicMock(return_value=[])
    # Very short query -> conservative
    result = await reader.retrieve("hi")
    assert result is not None


@pytest.mark.asyncio
async def test_retrieve_aggressive_strategy():
    reader, db, kg, ids = make_reader_with_data()
    reader.db.vector_search = MagicMock(return_value=[])
    reader.db.fts_search = MagicMock(return_value=[])
    result = await reader.retrieve("do you remember what I said about SSH keys?")
    assert result is not None


@pytest.mark.asyncio
async def test_retrieve_returns_memories_field():
    reader, db, kg, ids = make_reader_with_data()
    all_rows = db.execute(
        "SELECT id, content, category, importance, access_count, created_at FROM memories"
    )
    reader.db.vector_search = MagicMock(return_value=[{"id": r["id"]} for r in all_rows])
    reader.db.fts_search = MagicMock(return_value=[])
    result = await reader.retrieve("SSH")
    mems = result.memories if hasattr(result, "memories") else result
    assert isinstance(mems, list)


# ── strategy selection ────────────────────────────────────────────────────────

def test_strategy_short_query():
    reader, _, _, _ = make_reader_with_data()
    assert reader.adaptive_strategy("ok", "") == "conservative"


def test_strategy_memory_keywords():
    reader, _, _, _ = make_reader_with_data()
    strat = reader.adaptive_strategy("do you remember what I said?", "")
    assert strat == "aggressive"


def test_strategy_correction():
    reader, _, _, _ = make_reader_with_data()
    strat = reader.adaptive_strategy("Actually I prefer nginx", "")
    assert strat in ("aggressive", "normal")


def test_strategy_technical_normal():
    reader, _, _, _ = make_reader_with_data()
    strat = reader.adaptive_strategy(
        "How do I configure a systemd service to auto-restart?", ""
    )
    assert strat in ("normal", "aggressive")


# ── RRF pipeline ─────────────────────────────────────────────────────────────

def test_rrf_pipeline_deduplicates():
    reader, db, _, ids = make_reader_with_data()
    # Use real DB id
    shared_id = ids[0]
    dense  = [{"id": shared_id}, {"id": ids[1]}]
    sparse = [{"id": shared_id}, {"id": ids[2]}]
    fused = reader.rrf_fuse(dense, sparse)
    result_ids = [m.id for m in fused]
    assert result_ids.count(shared_id) == 1  # no duplicate


def test_rrf_fusion_score_ordering():
    reader, db, _, ids = make_reader_with_data()
    # id=ids[0] appears in both lists → higher RRF score
    dense  = [{"id": ids[0]}, {"id": ids[1]}]
    sparse = [{"id": ids[0]}, {"id": ids[2]}]
    fused = reader.rrf_fuse(dense, sparse)
    id_to_score = {m.id: m.rrf_score for m in fused}
    assert id_to_score[ids[0]] > id_to_score.get(ids[1], 0)
    assert id_to_score[ids[0]] > id_to_score.get(ids[2], 0)


def test_rrf_fuse_empty():
    reader, db, _, ids = make_reader_with_data()
    assert reader.rrf_fuse([], []) == []


def test_rrf_fuse_skips_missing_db_id():
    reader, db, _, ids = make_reader_with_data()
    result = reader.rrf_fuse([{"id": 99999}], [])
    assert result == []
