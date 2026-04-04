"""Tests for memory source linking and memory ↔ entity linking."""
from __future__ import annotations

# ── Memory source linking ────────────────────────────────────────────────────

def test_link_memory_source(db):
    mid = db.insert_memory("Python is a programming language", category="knowledge")
    db.link_memory_source(mid, "conversation", "sess-123")
    sources = db.get_memory_sources(mid)
    assert len(sources) == 1
    assert sources[0]["source_type"] == "conversation"
    assert sources[0]["source_id"] == "sess-123"


def test_link_memory_multiple_sources(db):
    mid = db.insert_memory("Docker uses containers", category="knowledge")
    db.link_memory_source(mid, "conversation", "sess-1")
    db.link_memory_source(mid, "document", "doc-42")
    sources = db.get_memory_sources(mid)
    assert len(sources) == 2
    types = {s["source_type"] for s in sources}
    assert types == {"conversation", "document"}


def test_link_memory_source_idempotent(db):
    """Duplicate source links should be ignored (INSERT OR IGNORE)."""
    mid = db.insert_memory("fact", category="fact")
    db.link_memory_source(mid, "conversation", "sess-1")
    db.link_memory_source(mid, "conversation", "sess-1")  # duplicate
    sources = db.get_memory_sources(mid)
    assert len(sources) == 1


def test_get_memories_by_source(db):
    mid1 = db.insert_memory("fact A", category="fact")
    mid2 = db.insert_memory("fact B", category="fact")
    mid3 = db.insert_memory("unrelated fact", category="fact")
    db.link_memory_source(mid1, "conversation", "sess-1")
    db.link_memory_source(mid2, "conversation", "sess-1")
    db.link_memory_source(mid3, "conversation", "sess-2")

    results = db.get_memories_by_source("conversation", "sess-1")
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {mid1, mid2}


def test_get_memories_by_source_empty(db):
    results = db.get_memories_by_source("dream", "nonexistent")
    assert results == []


# ── Memory ↔ Entity linking ──────────────────────────────────────────────────

def test_link_memory_entity(db):
    mid = db.insert_memory("User uses Python for data science", category="fact")
    eid = db.insert_entity("Python", entity_type="language")
    db.link_memory_entity(mid, eid)

    entities = db.get_memory_entities(mid)
    assert len(entities) == 1
    assert entities[0]["name"] == "Python"


def test_link_memory_entity_idempotent(db):
    mid = db.insert_memory("fact", category="fact")
    eid = db.insert_entity("Docker", entity_type="tool")
    db.link_memory_entity(mid, eid)
    db.link_memory_entity(mid, eid)  # duplicate
    entities = db.get_memory_entities(mid)
    assert len(entities) == 1


def test_get_entity_memories(db):
    eid = db.insert_entity("Rust", entity_type="language")
    mid1 = db.insert_memory("User is learning Rust", category="fact")
    mid2 = db.insert_memory("Rust has a borrow checker", category="knowledge")
    db.insert_memory("Unrelated fact about Python", category="fact")
    db.link_memory_entity(mid1, eid)
    db.link_memory_entity(mid2, eid)

    memories = db.get_entity_memories(eid)
    assert len(memories) == 2
    ids = {m["id"] for m in memories}
    assert ids == {mid1, mid2}


def test_get_related_memories(db):
    """Memories sharing entities should be returned as related."""
    eid_py = db.insert_entity("Python", entity_type="language")
    eid_ml = db.insert_entity("ML", entity_type="concept")

    mid1 = db.insert_memory("User uses Python for ML", category="fact")
    mid2 = db.insert_memory("Python has great ML libraries", category="knowledge")
    mid3 = db.insert_memory("User prefers PyTorch", category="preference")

    # mid1 and mid2 share Python entity
    db.link_memory_entity(mid1, eid_py)
    db.link_memory_entity(mid2, eid_py)
    # mid1 and mid3 share ML entity
    db.link_memory_entity(mid1, eid_ml)
    db.link_memory_entity(mid3, eid_ml)

    related = db.get_related_memories(mid1)
    ids = {m["id"] for m in related}
    assert mid2 in ids  # shares Python
    assert mid3 in ids  # shares ML
    assert mid1 not in ids  # not self


def test_get_related_memories_empty(db):
    mid = db.insert_memory("isolated fact", category="fact")
    related = db.get_related_memories(mid)
    assert related == []


def test_memory_entity_link_multiple_entities(db):
    """A memory can link to multiple entities."""
    mid = db.insert_memory("User uses Docker and Kubernetes", category="fact")
    eid1 = db.insert_entity("Docker", entity_type="tool")
    eid2 = db.insert_entity("Kubernetes", entity_type="tool")
    db.link_memory_entity(mid, eid1)
    db.link_memory_entity(mid, eid2)

    entities = db.get_memory_entities(mid)
    assert len(entities) == 2
    names = {e["name"] for e in entities}
    assert names == {"Docker", "Kubernetes"}


# ── Table creation ───────────────────────────────────────────────────────────

def test_memory_sources_table_exists(db):
    """Verify memory_sources table was created during DB setup."""
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_sources'"
    )
    assert len(rows) == 1


def test_memory_entity_links_table_exists(db):
    """Verify memory_entity_links table was created during DB setup."""
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_entity_links'"
    )
    assert len(rows) == 1
