"""Tests for Reranker — heuristic scoring, best-at-edges."""
from __future__ import annotations

import datetime


def make_reranker():
    from core.memory.reranker import Reranker
    return Reranker(feedback_collector=None)


def _mem(id, content, importance=0.5, access_count=1, category="fact",
         created_at=None, rrf_score=1.0):
    """Return a dict matching what reranker.rerank() expects."""
    if created_at is None:
        created_at = datetime.datetime.now().isoformat()
    return {
        "id": id,
        "content": content,
        "importance": importance,
        "access_count": access_count,
        "category": category,
        "created_at": created_at,
        "rrf_score": rrf_score,
    }


# ── heuristic stage ───────────────────────────────────────────────────────────

def test_rerank_returns_same_count():
    r = make_reranker()
    mems = [_mem(i, f"fact {i}") for i in range(5)]
    result = r.rerank(mems, query="test query")
    assert len(result) == 5


def test_rerank_empty_list():
    r = make_reranker()
    assert r.rerank([], query="anything") == []


def test_rerank_boosts_high_importance():
    r = make_reranker()
    now = datetime.datetime.now().isoformat()
    mems = [
        _mem(1, "low importance",  importance=0.1, created_at=now),
        _mem(2, "high importance", importance=0.95, created_at=now),
    ]
    ranked = r.rerank(mems, query="test")
    assert ranked[0]["id"] == 2


def test_rerank_boosts_recent():
    r = make_reranker()
    recent = datetime.datetime.now().isoformat()
    old    = "2019-01-01T00:00:00"
    mems = [
        _mem(1, "old fact",    importance=0.5, created_at=old),
        _mem(2, "recent fact", importance=0.5, created_at=recent),
    ]
    ranked = r.rerank(mems, query="test")
    assert ranked[0]["id"] == 2


def test_rerank_boosts_frequent_access():
    r = make_reranker()
    now = datetime.datetime.now().isoformat()
    mems = [
        _mem(1, "rarely accessed", access_count=0,  importance=0.5, created_at=now),
        _mem(2, "often accessed",  access_count=50, importance=0.5, created_at=now),
    ]
    ranked = r.rerank(mems, query="test")
    assert ranked[0]["id"] == 2


def test_rerank_boosts_corrections():
    r = make_reranker()
    now = datetime.datetime.now().isoformat()
    mems = [
        _mem(1, "regular fact",    category="fact",       importance=0.5, created_at=now),
        _mem(2, "correction note", category="correction", importance=0.5, created_at=now),
    ]
    ranked = r.rerank(mems, query="test")
    assert ranked[0]["id"] == 2


# ── age computation ───────────────────────────────────────────────────────────

def test_compute_age_days_recent():
    r = make_reranker()
    now = datetime.datetime.now().isoformat()
    age = r.compute_age_days(now)
    assert 0 <= age < 1


def test_compute_age_days_old():
    r = make_reranker()
    age = r.compute_age_days("2020-01-01T00:00:00")
    assert age > 365 * 4


def test_compute_age_days_invalid_returns_fallback():
    r = make_reranker()
    age = r.compute_age_days("not-a-date")
    assert age >= 0


# ── best-at-edges ─────────────────────────────────────────────────────────────

def test_apply_best_at_edges_preserves_count():
    r = make_reranker()
    mems = [_mem(i, f"fact {i}", importance=i * 0.1) for i in range(8)]
    result = r.apply_best_at_edges(mems)
    assert len(result) == 8


def test_apply_best_at_edges_empty():
    r = make_reranker()
    assert r.apply_best_at_edges([]) == []


def test_apply_best_at_edges_single():
    r = make_reranker()
    mems = [_mem(1, "only")]
    assert len(r.apply_best_at_edges(mems)) == 1


def test_apply_best_at_edges_puts_best_at_edge():
    r = make_reranker()
    now = datetime.datetime.now().isoformat()
    # Create mems with varying importance; id=5 has highest importance
    mems = [_mem(i, f"fact {i}", importance=i * 0.1, created_at=now) for i in range(6)]
    ranked = r.rerank(mems, query="test")
    result = r.apply_best_at_edges(ranked)
    # Best item should be at position 0 or last
    best = max(ranked, key=lambda m: m.get("_rerank_score", m.get("importance", 0)))
    best_id = best["id"]
    ids = [m["id"] for m in result]
    assert ids[0] == best_id or ids[-1] == best_id
