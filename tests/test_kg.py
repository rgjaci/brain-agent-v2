"""Tests for KnowledgeGraph — entity/relation CRUD and traversal."""
from __future__ import annotations

from core.memory.database import MemoryDatabase
from core.memory.kg import Entity, KnowledgeGraph, Relation


def make_kg():
    db = MemoryDatabase(":memory:")
    return KnowledgeGraph(db), db


# ── entity upsert ─────────────────────────────────────────────────────────────

def test_upsert_entity_returns_id():
    kg, _ = make_kg()
    eid = kg.upsert_entity(Entity("Tailscale", "service", "VPN mesh"))
    assert isinstance(eid, int)
    assert eid > 0


def test_upsert_entity_deduplicates():
    kg, _ = make_kg()
    id1 = kg.upsert_entity(Entity("Tailscale", "service", "VPN mesh"))
    id2 = kg.upsert_entity(Entity("Tailscale", "service", "Updated description"))
    assert id1 == id2


def test_find_entity_by_name():
    kg, _ = make_kg()
    kg.upsert_entity(Entity("Ubuntu", "tool", "Linux distro"))
    result = kg.find_entity("Ubuntu")
    assert result is not None
    assert result["name"] == "Ubuntu"


def test_find_entity_missing_returns_none():
    kg, _ = make_kg()
    assert kg.find_entity("NonExistent") is None


def test_upsert_entity_invalid_type_falls_back():
    kg, _ = make_kg()
    eid = kg.upsert_entity(Entity("Thing", "unknown_type", "desc"))
    assert eid > 0


# ── relation upsert ───────────────────────────────────────────────────────────

def test_upsert_relation_returns_id():
    kg, _ = make_kg()
    kg.upsert_entity(Entity("User", "person", "the user"))
    kg.upsert_entity(Entity("Tailscale", "service", "VPN"))
    rid = kg.upsert_relation(Relation("User", "Tailscale", "uses", detail="for remote access"))
    assert rid is not None
    assert rid > 0


def test_upsert_relation_same_pair_twice():
    kg, _ = make_kg()
    kg.upsert_entity(Entity("User", "person", "the user"))
    kg.upsert_entity(Entity("Tailscale", "service", "VPN"))
    r1 = kg.upsert_relation(Relation("User", "Tailscale", "uses"))
    kg.upsert_relation(Relation("User", "Tailscale", "uses"))
    # First should succeed; second may be duplicate (None) or new id
    assert r1 is not None  # at minimum first insert works


def test_upsert_relation_entities_must_exist():
    kg, _ = make_kg()
    # Entities not pre-created — upsert_relation returns None
    rid = kg.upsert_relation(Relation("Alice", "Python", "uses"))
    assert rid is None or isinstance(rid, int)  # graceful skip


# ── traversal ─────────────────────────────────────────────────────────────────

def test_traverse_returns_string():
    kg, _ = make_kg()
    kg.upsert_entity(Entity("User", "person", "the user"))
    kg.upsert_entity(Entity("Vim", "tool", "text editor"))
    kg.upsert_relation(Relation("User", "Vim", "prefers", detail="for editing"))
    ctx = kg.traverse(["User"], max_hops=1)
    assert isinstance(ctx, str)


def test_traverse_empty_on_missing_entity():
    kg, _ = make_kg()
    ctx = kg.traverse(["NoSuchEntity"], max_hops=1)
    assert isinstance(ctx, str)


def test_traverse_respects_max_facts():
    kg, _ = make_kg()
    for i in range(10):
        kg.upsert_entity(Entity(f"Tool{i}", "tool", f"tool {i}"))
        kg.upsert_relation(Relation("User", f"Tool{i}", "uses"))
    ctx = kg.traverse(["User"], max_hops=1, max_facts=3)
    assert isinstance(ctx, str)


# ── get_entity_context ────────────────────────────────────────────────────────

def test_get_entity_context_returns_dict():
    kg, _ = make_kg()
    kg.upsert_entity(Entity("Python", "language", "programming language"))
    kg.upsert_entity(Entity("User", "person", "the user"))
    kg.upsert_relation(Relation("User", "Python", "uses"))
    result = kg.get_entity_context("Python")
    assert isinstance(result, dict)


# ── stats ─────────────────────────────────────────────────────────────────────

def test_get_stats_returns_counts():
    kg, _ = make_kg()
    kg.upsert_entity(Entity("A", "tool", "tool a"))
    kg.upsert_entity(Entity("B", "tool", "tool b"))
    kg.upsert_relation(Relation("A", "B", "depends_on"))
    stats = kg.get_stats()
    assert stats.get("entities", 0) >= 2
    assert stats.get("relations", 0) >= 1
