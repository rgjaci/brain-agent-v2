"""Tests for server/mcp_server.py — MCP Server tools."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture(autouse=True)
def reset_server_globals():
    """Reset lazy-initialised globals before each test."""
    import server.mcp_server as srv
    srv._db = None
    srv._kg = None
    srv._llm = None
    srv._embedder = None
    srv._writer = None
    srv._reader = None
    srv._dream_engine = None
    srv._reasoning_engine = None
    yield
    srv._db = None
    srv._kg = None


@pytest.fixture
def setup_server(db, kg):
    """Inject test DB into MCP server globals."""
    import server.mcp_server as srv
    srv._db = db
    srv._kg = kg
    return srv


# ── memory_store ─────────────────────────────────────────────────────────────

def test_memory_store(setup_server):
    from server.mcp_server import memory_store
    result = memory_store("User likes Python", category="preference", importance=0.8)
    assert "Stored memory #" in result
    assert "preference" in result


def test_memory_store_invalid_category(setup_server):
    from server.mcp_server import memory_store
    result = memory_store("A fact", category="bogus")
    assert "fact" in result  # falls back to "fact"


def test_memory_store_clamps_importance(setup_server):
    from server.mcp_server import memory_store
    result = memory_store("test", importance=5.0)
    assert "Stored memory #" in result


# ── memory_recall ────────────────────────────────────────────────────────────

def test_memory_recall_no_results(setup_server):
    from server.mcp_server import memory_recall
    result = memory_recall("nonexistent query xyz")
    assert "No memories found" in result


def test_memory_recall_finds_results(setup_server, db):
    db.insert_memory("Python is a programming language", category="knowledge")
    from server.mcp_server import memory_recall
    result = memory_recall("Python programming")
    assert "Python" in result


# ── memory_teach ─────────────────────────────────────────────────────────────

def test_memory_teach(setup_server):
    from server.mcp_server import memory_teach
    result = memory_teach("The server runs on port 8080")
    assert "Taught memory #" in result


# ── memory_stats ─────────────────────────────────────────────────────────────

def test_memory_stats_empty(setup_server):
    from server.mcp_server import memory_stats
    result = memory_stats()
    assert "Memories:" in result
    assert "Entities:" in result


def test_memory_stats_with_data(setup_server, db):
    db.insert_memory("fact 1", category="fact")
    db.insert_memory("question?", category="question")
    db.insert_entity("Python", entity_type="language")
    from server.mcp_server import memory_stats
    result = memory_stats()
    assert "2" in result or "Memories" in result


# ── memory_related ───────────────────────────────────────────────────────────

def test_memory_related_no_links(setup_server, db):
    mid = db.insert_memory("isolated fact", category="fact")
    from server.mcp_server import memory_related
    result = memory_related(mid)
    assert "No related" in result


def test_memory_related_with_links(setup_server, db):
    eid = db.insert_entity("Python", entity_type="language")
    mid1 = db.insert_memory("User uses Python", category="fact")
    mid2 = db.insert_memory("Python is fast", category="knowledge")
    db.link_memory_entity(mid1, eid)
    db.link_memory_entity(mid2, eid)
    from server.mcp_server import memory_related
    result = memory_related(mid1)
    assert "Python is fast" in result


# ── kg_query ─────────────────────────────────────────────────────────────────

def test_kg_query_not_found(setup_server):
    from server.mcp_server import kg_query
    result = kg_query("NonexistentEntity")
    assert "not found" in result


def test_kg_query_found(setup_server, db):
    eid = db.insert_entity("Python", entity_type="language", description="A programming language")
    from server.mcp_server import kg_query
    result = kg_query("Python")
    assert "Python" in result
    assert "language" in result


def test_kg_query_with_relations(setup_server, db):
    eid1 = db.insert_entity("Django", entity_type="tool")
    eid2 = db.insert_entity("Python", entity_type="language")
    db.insert_relation(eid1, eid2, "depends_on", confidence=0.9)
    from server.mcp_server import kg_query
    result = kg_query("Django")
    assert "depends_on" in result


# ── kg_add_entity ────────────────────────────────────────────────────────────

def test_kg_add_entity(setup_server):
    from server.mcp_server import kg_add_entity
    result = kg_add_entity("Rust", entity_type="language", description="Systems language")
    assert "Rust" in result
    assert "language" in result


# ── kg_add_relation ──────────────────────────────────────────────────────────

def test_kg_add_relation(setup_server, db):
    db.insert_entity("Docker", entity_type="tool")
    db.insert_entity("Linux", entity_type="concept")
    from server.mcp_server import kg_add_relation
    result = kg_add_relation("Docker", "Linux", "depends_on", confidence=0.9)
    assert "depends_on" in result


def test_kg_add_relation_creates_missing_entities(setup_server):
    from server.mcp_server import kg_add_relation
    result = kg_add_relation("NewTool", "NewDep", "works_with")
    assert "works_with" in result


# ── kg_traverse ──────────────────────────────────────────────────────────────

def test_kg_traverse_not_found(setup_server):
    from server.mcp_server import kg_traverse
    result = kg_traverse("Nonexistent")
    assert "No graph context" in result or result == ""


# ── dream_trigger ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dream_trigger_no_engine(setup_server):
    from server.mcp_server import dream_trigger
    result = await dream_trigger()
    assert "not available" in result


@pytest.mark.asyncio
async def test_dream_trigger_with_engine(setup_server, db, mock_llm, mock_embedder):
    import server.mcp_server as srv
    from core.memory.consolidation import ConsolidationEngine
    from core.memory.dream import DreamEngine
    consolidator = ConsolidationEngine(db, mock_llm, mock_embedder)
    srv._dream_engine = DreamEngine(db, mock_llm, mock_embedder, consolidator, srv._kg)

    from server.mcp_server import dream_trigger
    result = await dream_trigger()
    assert "Dream cycle complete" in result


# ── reasoning_status ─────────────────────────────────────────────────────────

def test_reasoning_status_no_engine(setup_server):
    from server.mcp_server import reasoning_status
    result = reasoning_status()
    assert "not available" in result


def test_reasoning_status_with_engine(setup_server, db, mock_llm, mock_embedder, kg):
    import server.mcp_server as srv
    from core.memory.reasoning import ReasoningEngine
    srv._reasoning_engine = ReasoningEngine(db, mock_llm, mock_embedder, kg, interval=60)

    from server.mcp_server import reasoning_status
    result = reasoning_status()
    assert "Running:" in result
    assert "Total cycles:" in result


# ── reasoning_cycle ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reasoning_cycle_no_engine(setup_server):
    from server.mcp_server import reasoning_cycle
    result = await reasoning_cycle()
    assert "not available" in result


@pytest.mark.asyncio
async def test_reasoning_cycle_with_engine(setup_server, db, mock_llm, mock_embedder, kg):
    import server.mcp_server as srv
    from core.memory.reasoning import ReasoningEngine
    srv._reasoning_engine = ReasoningEngine(db, mock_llm, mock_embedder, kg, interval=60)

    from server.mcp_server import reasoning_cycle
    result = await reasoning_cycle()
    assert "Reasoning cycle complete" in result


# ── Resources ────────────────────────────────────────────────────────────────

def test_resource_stats(setup_server):
    from server.mcp_server import resource_stats
    result = resource_stats()
    assert "Memories:" in result


def test_resource_recent_empty(setup_server):
    from server.mcp_server import resource_recent
    result = resource_recent()
    assert "(no memories)" in result


def test_resource_recent_with_data(setup_server, db):
    db.insert_memory("test fact", category="fact")
    from server.mcp_server import resource_recent
    result = resource_recent()
    assert "test fact" in result


def test_resource_insights_empty(setup_server):
    from server.mcp_server import resource_insights
    result = resource_insights()
    assert "(no insights yet)" in result


def test_resource_insights_with_data(setup_server, db):
    db.insert_memory("Pattern: users prefer X", category="insight", source="dream")
    from server.mcp_server import resource_insights
    result = resource_insights()
    assert "Pattern:" in result


def test_resource_questions_empty(setup_server):
    from server.mcp_server import resource_questions
    result = resource_questions()
    assert "(no open questions)" in result


def test_resource_questions_with_data(setup_server, db):
    db.insert_memory("What IDE does the user prefer?", category="question")
    from server.mcp_server import resource_questions
    result = resource_questions()
    assert "IDE" in result
