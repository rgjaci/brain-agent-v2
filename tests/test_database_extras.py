"""Tests for database.py new methods and schema additions."""
from __future__ import annotations

# ── store_conversation / get_recent_messages ─────────────────────────────────

def test_store_conversation_returns_id(db):
    row_id = db.store_conversation("sess1", "user", "hello")
    assert isinstance(row_id, int) and row_id > 0


def test_get_recent_messages_order(db):
    db.store_conversation("s1", "user", "first")
    db.store_conversation("s1", "assistant", "second")
    db.store_conversation("s1", "user", "third")
    msgs = db.get_recent_messages("s1", limit=10)
    assert len(msgs) == 3
    assert msgs[0]["content"] == "first"
    assert msgs[-1]["content"] == "third"


def test_get_recent_messages_limit(db):
    for i in range(10):
        db.store_conversation("s2", "user", f"msg-{i}")
    msgs = db.get_recent_messages("s2", limit=3)
    assert len(msgs) == 3
    # Should be the 3 most recent, oldest-first
    assert msgs[0]["content"] == "msg-7"
    assert msgs[-1]["content"] == "msg-9"


def test_get_recent_messages_empty(db):
    msgs = db.get_recent_messages("nonexistent")
    assert msgs == []


# ── usefulness_score ─────────────────────────────────────────────────────────

def test_usefulness_score_default(db):
    mid = db.insert_memory("test fact", importance=0.5)
    mem = db.get_memory(mid)
    assert mem["usefulness_score"] == 0.5


def test_update_usefulness_positive(db):
    mid = db.insert_memory("useful fact")
    db.update_usefulness(mid, 0.3)
    mem = db.get_memory(mid)
    assert abs(mem["usefulness_score"] - 0.8) < 1e-9


def test_update_usefulness_clamp_high(db):
    mid = db.insert_memory("very useful")
    db.update_usefulness(mid, 0.9)  # 0.5 + 0.9 = 1.4 → clamped to 1.0
    mem = db.get_memory(mid)
    assert mem["usefulness_score"] == 1.0


def test_update_usefulness_clamp_low(db):
    mid = db.insert_memory("less useful")
    db.update_usefulness(mid, -0.8)  # 0.5 - 0.8 = -0.3 → clamped to 0.0
    mem = db.get_memory(mid)
    assert mem["usefulness_score"] == 0.0


# ── FTS5 search ──────────────────────────────────────────────────────────────

def test_fts5_search_basic(db):
    db.insert_memory("Python is a great programming language", category="fact")
    db.insert_memory("I like eating pizza on Fridays", category="preference")
    results = db.fts_search("Python programming", limit=5)
    assert len(results) >= 1
    assert results[0]["id"] is not None


# ── relations valid_from / valid_until ───────────────────────────────────────

def test_relation_valid_from_default(db):
    e1 = db.insert_entity("A", "concept")
    e2 = db.insert_entity("B", "concept")
    rid = db.insert_relation(e1, e2, "uses")
    rows = db.execute("SELECT valid_from, valid_until FROM relations WHERE id = ?", (rid,))
    assert len(rows) == 1
    assert rows[0]["valid_from"] is not None
    assert rows[0]["valid_until"] is None


# ── count_memories ───────────────────────────────────────────────────────────

def test_count_memories_empty(db):
    assert db.count_memories() == 0


def test_count_memories_after_inserts(db):
    db.insert_memory("fact one")
    db.insert_memory("fact two")
    db.insert_memory("fact three")
    assert db.count_memories() == 3
