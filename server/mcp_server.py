"""Brain Agent v2 — MCP Server.

Exposes the Brain Agent memory system via Model Context Protocol (MCP) for
integration with external agents like Claude Code, OpenClaw, etc.

Usage::

    # stdio transport (for Claude Code integration)
    python -m server.mcp_server

    # Or via main.py
    python main.py serve
    python main.py serve --transport sse --port 8765

Claude Code config (``~/.claude/claude_code_config.json``)::

    {
      "mcpServers": {
        "brain-agent": {
          "command": "python",
          "args": ["-m", "server.mcp_server"],
          "cwd": "/path/to/brain-agent-v2"
        }
      }
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── Create the MCP server ────────────────────────────────────────────────────

mcp = FastMCP(
    "brain-agent",
    instructions=(
        "Brain Agent memory system. Store, retrieve, and organise knowledge. "
        "Use memory_store to save facts, memory_recall to search, "
        "kg_traverse for entity relationships, and dream_trigger for "
        "LLM-powered memory consolidation."
    ),
)

# ── Lazy-initialised singletons ──────────────────────────────────────────────

_db = None
_kg = None
_llm = None
_embedder = None
_writer = None
_reader = None
_dream_engine = None
_reasoning_engine = None


def _init_components():
    """Lazily initialise database and components on first tool call."""
    global _db, _kg, _llm, _embedder, _writer, _reader
    global _dream_engine, _reasoning_engine

    if _db is not None:
        return

    from core.config import AgentConfig
    from core.memory.database import MemoryDatabase
    from core.memory.kg import KnowledgeGraph

    cfg = AgentConfig.load()
    db_path = cfg.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _db = MemoryDatabase(str(db_path))
    _kg = KnowledgeGraph(_db)

    # LLM — optional
    try:
        from core.llm.provider import OllamaProvider
        _llm = OllamaProvider.from_env()
    except Exception:
        logger.warning("LLM unavailable — some tools will be limited.")

    # Embedder — optional
    try:
        from core.llm.embeddings import GeminiEmbeddingProvider
        _embedder = GeminiEmbeddingProvider(api_key=cfg.gemini_api_key)
    except Exception:
        logger.warning("Embedder unavailable — semantic search disabled.")

    # Writer
    if _llm and _embedder:
        from core.memory.writer import MemoryWriter
        _writer = MemoryWriter(_llm, _embedder, _db, _kg)

    # Reader
    if _embedder:
        from core.memory.reader import MemoryReader
        _reader = MemoryReader(_db, _embedder, _kg, _llm)

    # Dream engine
    if _llm and _embedder and cfg.dream_enabled:
        from core.memory.consolidation import ConsolidationEngine
        from core.memory.dream import DreamEngine
        consolidator = ConsolidationEngine(_db, _llm, _embedder)
        _dream_engine = DreamEngine(_db, _llm, _embedder, consolidator, _kg)

    # Reasoning engine
    if _llm and _embedder and cfg.reasoning_enabled:
        from core.memory.reasoning import ReasoningEngine
        _reasoning_engine = ReasoningEngine(
            db=_db, llm=_llm, embedder=_embedder, kg=_kg,
            interval=cfg.reasoning_interval,
            max_cycles=cfg.reasoning_max_cycles_per_session,
        )

    logger.info("MCP server components initialised (db=%s).", db_path)


# ── Memory tools ─────────────────────────────────────────────────────────────

@mcp.tool()
def memory_store(
    content: str,
    category: str = "fact",
    importance: float = 0.7,
    source: str = "mcp",
) -> str:
    """Store a fact, observation, or piece of knowledge in the memory system.

    Args:
        content: The memory text to store (e.g. "User prefers dark mode").
        category: One of: fact, preference, observation, correction, knowledge,
                  rule, insight, question.
        importance: Score from 0.0 to 1.0 (higher = more important).
        source: Origin label (default: "mcp").

    Returns:
        Confirmation with the new memory ID.
    """
    _init_components()
    importance = max(0.1, min(1.0, importance))
    valid_categories = {
        "fact", "preference", "observation", "correction",
        "knowledge", "rule", "insight", "question",
    }
    if category not in valid_categories:
        category = "fact"
    mid = _db.insert_memory(
        content=content,
        category=category,
        source=source,
        importance=importance,
    )
    _db.link_memory_source(mid, "mcp", source)
    return f"Stored memory #{mid}: [{category}] {content}"


@mcp.tool()
def memory_recall(query: str, limit: int = 10) -> str:
    """Search memories by keyword or semantic similarity.

    Args:
        query: Search query (natural language or keywords).
        limit: Maximum number of results (default: 10).

    Returns:
        Formatted list of matching memories with categories and scores.
    """
    _init_components()
    safe = re.sub(r'[^\w\s]', ' ', query).strip()
    fts_query = " OR ".join(safe.split()) if safe else ""
    results = _db.fts_search(fts_query, limit=limit) if fts_query else []

    if not results:
        return f'No memories found for: "{query}"'

    lines = []
    for i, hit in enumerate(results, 1):
        mem = _db.get_memory(hit["id"])
        if mem:
            lines.append(
                f"{i}. [{mem.get('category', '?')}] "
                f"(importance={mem.get('importance', 0):.1f}) "
                f"{mem.get('content', '')}"
            )
    return "\n".join(lines)


@mcp.tool()
def memory_teach(fact: str, importance: float = 0.7) -> str:
    """Teach a specific fact directly to the memory system.

    Args:
        fact: The fact to store (e.g. "The server runs on port 8080").
        importance: Score from 0.0 to 1.0.

    Returns:
        Confirmation message.
    """
    _init_components()
    mid = _db.insert_memory(
        content=fact,
        category="fact",
        source="teach:mcp",
        importance=max(0.1, min(1.0, importance)),
    )
    return f"Taught memory #{mid}: {fact}"


@mcp.tool()
def memory_stats() -> str:
    """Get current memory system statistics.

    Returns:
        Counts of memories, entities, relations, procedures, and questions.
    """
    _init_components()

    def _n(q):
        rows = _db.execute(q)
        return int(rows[0]["n"]) if rows else 0

    mem_count = _n("SELECT COUNT(*) n FROM memories WHERE superseded_by IS NULL")
    ent_count = _n("SELECT COUNT(*) n FROM entities")
    rel_count = _n("SELECT COUNT(*) n FROM relations")
    proc_count = _n("SELECT COUNT(*) n FROM procedures")
    q_count = _n("SELECT COUNT(*) n FROM memories WHERE category = 'question' AND superseded_by IS NULL")
    insight_count = _n("SELECT COUNT(*) n FROM memories WHERE category = 'insight' AND superseded_by IS NULL")

    return (
        f"Memories:    {mem_count:,}\n"
        f"Entities:    {ent_count:,}\n"
        f"Relations:   {rel_count:,}\n"
        f"Procedures:  {proc_count:,}\n"
        f"Insights:    {insight_count:,}\n"
        f"Open questions: {q_count:,}"
    )


@mcp.tool()
def memory_related(memory_id: int) -> str:
    """Find memories related to a given memory (sharing entities).

    Args:
        memory_id: The memory ID to find relations for.

    Returns:
        List of related memories.
    """
    _init_components()
    related = _db.get_related_memories(memory_id, limit=10)
    if not related:
        return f"No related memories found for memory #{memory_id}."

    lines = [f"Memories related to #{memory_id}:"]
    for m in related:
        lines.append(
            f"  #{m['id']} [{m.get('category', '?')}] {m.get('content', '')}"
        )
    return "\n".join(lines)


# ── Knowledge graph tools ────────────────────────────────────────────────────

@mcp.tool()
def kg_query(entity_name: str) -> str:
    """Query the knowledge graph for an entity and its relations.

    Args:
        entity_name: Name of the entity to look up.

    Returns:
        Entity details and its relations.
    """
    _init_components()
    entity = _db.get_entity(name=entity_name)
    if not entity:
        return f'Entity "{entity_name}" not found.'

    lines = [
        f"Entity: {entity['name']} (type: {entity.get('entity_type', '?')})",
        f"Description: {entity.get('description', 'N/A')}",
        f"Importance: {entity.get('importance', 0):.2f}",
        "",
        "Relations:",
    ]

    relations = _db.get_relations(entity["id"])
    if not relations:
        lines.append("  (none)")
    else:
        for r in relations:
            lines.append(
                f"  {r.get('source_name', '?')} "
                f"→ {r.get('relation_type', '?')} → "
                f"{r.get('target_name', '?')} "
                f"(confidence={r.get('confidence', 0):.2f})"
            )

    # Also show linked memories
    entity_mems = _db.get_entity_memories(entity["id"])
    if entity_mems:
        lines.append("")
        lines.append("Linked memories:")
        for m in entity_mems[:5]:
            lines.append(f"  #{m['id']} {m.get('content', '')[:80]}")

    return "\n".join(lines)


@mcp.tool()
def kg_add_entity(
    name: str,
    entity_type: str = "concept",
    description: str = "",
) -> str:
    """Add or update a knowledge graph entity.

    Args:
        name: Entity name (e.g. "Python", "Docker").
        entity_type: One of: person, tool, concept, project, file, service,
                     language, config.
        description: Brief description.

    Returns:
        Confirmation with entity ID.
    """
    _init_components()
    eid = _db.upsert_entity(
        name=name,
        entity_type=entity_type,
        description=description,
        source="mcp",
    )
    return f"Entity #{eid}: {name} ({entity_type})"


@mcp.tool()
def kg_add_relation(
    source_entity: str,
    target_entity: str,
    relation_type: str = "works_with",
    confidence: float = 0.8,
) -> str:
    """Add a relation between two entities in the knowledge graph.

    Args:
        source_entity: Name of the source entity.
        target_entity: Name of the target entity.
        relation_type: One of: uses, prefers, part_of, depends_on, works_with,
                       manages, belongs_to, configured_with, causes, contradicts.
        confidence: Confidence score 0.0-1.0.

    Returns:
        Confirmation message.
    """
    _init_components()
    src = _db.get_entity(name=source_entity)
    tgt = _db.get_entity(name=target_entity)

    if not src:
        src_id = _db.upsert_entity(name=source_entity, source="mcp")
    else:
        src_id = src["id"]

    if not tgt:
        tgt_id = _db.upsert_entity(name=target_entity, source="mcp")
    else:
        tgt_id = tgt["id"]

    rid = _db.insert_relation(
        source_id=src_id,
        target_id=tgt_id,
        relation_type=relation_type,
        confidence=max(0.1, min(1.0, confidence)),
        source="mcp",
    )
    return f"Relation #{rid}: {source_entity} → {relation_type} → {target_entity}"


@mcp.tool()
def kg_traverse(entity_name: str, max_hops: int = 2) -> str:
    """Traverse the knowledge graph from an entity using BFS.

    Args:
        entity_name: Starting entity name.
        max_hops: Maximum traversal depth (default: 2).

    Returns:
        Formatted context string with discovered facts.
    """
    _init_components()
    context = _kg.traverse([entity_name], max_hops=min(max_hops, 3))
    return context if context else f'No graph context found for "{entity_name}".'


# ── Dream / Reasoning tools ─────────────────────────────────────────────────

@mcp.tool()
async def dream_trigger() -> str:
    """Trigger an AutoDream cycle (LLM-powered memory consolidation).

    This runs the full dream cycle: abstraction creation, contradiction
    resolution, pattern detection, connection strengthening, and question
    generation.

    Returns:
        Dream cycle report with counts.
    """
    _init_components()
    if not _dream_engine:
        return "AutoDream not available (requires LLM + embedder)."

    report = await _dream_engine.dream()
    return (
        f"Dream cycle complete ({report.elapsed_seconds:.1f}s):\n"
        f"  Abstractions created:    {report.abstractions_created}\n"
        f"  Contradictions resolved: {report.contradictions_resolved}\n"
        f"  Patterns detected:       {report.patterns_detected}\n"
        f"  Connections added:       {report.connections_added}\n"
        f"  Questions generated:     {report.questions_generated}"
    )


@mcp.tool()
def reasoning_status() -> str:
    """Check the status of the System 2 reasoning engine.

    Returns:
        Current status including cycle count and recent activity.
    """
    _init_components()
    if not _reasoning_engine:
        return "Reasoning engine not available."

    results = _reasoning_engine.recent_results
    lines = [
        f"Running: {_reasoning_engine.is_running}",
        f"Total cycles: {_reasoning_engine._cycle_count}",
        f"Interval: {_reasoning_engine.interval}s",
        "",
        "Recent results:",
    ]

    if not results:
        lines.append("  (none yet)")
    else:
        for r in results[-5:]:
            lines.append(
                f"  [{r.strategy}] focus={r.focus_topic} "
                f"insights={r.insights_generated} "
                f"questions={r.questions_generated} "
                f"rules={r.rules_derived} "
                f"({r.elapsed_seconds:.1f}s)"
            )

    return "\n".join(lines)


@mcp.tool()
async def reasoning_cycle() -> str:
    """Run a single System 2 reasoning cycle manually.

    Returns:
        Results of the reasoning cycle.
    """
    _init_components()
    if not _reasoning_engine:
        return "Reasoning engine not available."

    result = await _reasoning_engine.reasoning_cycle()
    return (
        f"Reasoning cycle complete ({result.elapsed_seconds:.1f}s):\n"
        f"  Strategy: {result.strategy}\n"
        f"  Focus: {result.focus_topic}\n"
        f"  Insights: {result.insights_generated}\n"
        f"  Questions: {result.questions_generated}\n"
        f"  Rules: {result.rules_derived}\n"
        f"  Connections: {result.connections_found}\n"
        f"  Answered: {result.questions_answered}"
    )


# ── MCP Resources ───────────────────────────────────────────────────────────

@mcp.resource("memory://stats")
def resource_stats() -> str:
    """Current memory system statistics."""
    return memory_stats()


@mcp.resource("memory://recent")
def resource_recent() -> str:
    """Most recent memories."""
    _init_components()
    rows = _db.execute(
        """
        SELECT id, content, category, importance, created_at
          FROM memories
         WHERE superseded_by IS NULL
         ORDER BY created_at DESC
         LIMIT 20
        """
    )
    lines = []
    for r in rows:
        lines.append(
            f"#{r['id']} [{r.get('category', '?')}] "
            f"(imp={r.get('importance', 0):.1f}) "
            f"{r.get('content', '')}"
        )
    return "\n".join(lines) if lines else "(no memories)"


@mcp.resource("memory://insights")
def resource_insights() -> str:
    """Recent insights from dreaming and reasoning."""
    _init_components()
    rows = _db.execute(
        """
        SELECT id, content, source, importance, created_at
          FROM memories
         WHERE category IN ('insight', 'rule')
           AND superseded_by IS NULL
         ORDER BY created_at DESC
         LIMIT 20
        """
    )
    lines = []
    for r in rows:
        lines.append(
            f"#{r['id']} [{r.get('source', '?')}] "
            f"(imp={r.get('importance', 0):.1f}) "
            f"{r.get('content', '')}"
        )
    return "\n".join(lines) if lines else "(no insights yet)"


@mcp.resource("memory://questions")
def resource_questions() -> str:
    """Open questions awaiting answers."""
    _init_components()
    rows = _db.execute(
        """
        SELECT id, content, importance, created_at
          FROM memories
         WHERE category = 'question'
           AND superseded_by IS NULL
         ORDER BY importance DESC, created_at DESC
         LIMIT 20
        """
    )
    lines = []
    for r in rows:
        lines.append(
            f"#{r['id']} (imp={r.get('importance', 0):.1f}) "
            f"{r.get('content', '')}"
        )
    return "\n".join(lines) if lines else "(no open questions)"


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
