"""Brain Agent v2 — Knowledge Graph module.

Provides :class:`KnowledgeGraph`, a thin layer over :class:`MemoryDatabase`
that adds entity/relation upsert helpers and a BFS graph traversal for
building LLM context strings.

Schema notes (from database.py):
  - ``relations`` table uses ``source_id`` / ``target_id`` columns.
  - There is NO ``valid_until`` column; expired-relation filtering is skipped.
  - ``upsert_entity`` and ``insert_relation`` accept a *dict* for ``properties``
    (the DB layer JSON-encodes it internally).
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Valid vocabularies
# ──────────────────────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = {
    "person",
    "tool",
    "concept",
    "project",
    "file",
    "service",
    "language",
    "config",
    "other",
}

VALID_RELATION_TYPES = {
    "uses",
    "prefers",
    "part_of",
    "causes",
    "depends_on",
    "contradicts",
    "instance_of",
    "located_in",
    "created_by",
    "configured_with",
    "works_with",
    "manages",
    "belongs_to",
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Entity:
    """Lightweight representation of a knowledge-graph entity."""

    name: str
    entity_type: str
    description: str = ""
    properties: dict = field(default_factory=dict)
    importance: float = 0.5
    source: str = ""
    id: int | None = None


@dataclass
class Relation:
    """Lightweight representation of a knowledge-graph relation."""

    source_name: str
    target_name: str
    relation_type: str
    properties: dict = field(default_factory=dict)
    confidence: float = 0.7
    source: str = ""
    detail: str = ""  # extra human-readable detail from extraction


# ──────────────────────────────────────────────────────────────────────────────
# KnowledgeGraph
# ──────────────────────────────────────────────────────────────────────────────


class KnowledgeGraph:
    """SQLite-backed knowledge graph for entity/relation storage and traversal.

    All persistence is delegated to a :class:`MemoryDatabase` instance whose
    schema is defined in ``database.py``.

    Args:
        db: An initialised :class:`MemoryDatabase` instance.
    """

    def __init__(self, db) -> None:
        self.db = db

    # ── Entity helpers ─────────────────────────────────────────────────────────

    def upsert_entity(self, entity: Entity) -> int:
        """Insert or update an entity and return its database ID.

        The ``entity_type`` is clamped to the allowed vocabulary; unknown types
        map to ``"other"``.  ``properties`` is passed as a dict — the DB layer
        handles JSON encoding.

        Args:
            entity: :class:`Entity` instance to persist.

        Returns:
            Primary key of the inserted / updated row.
        """
        safe_type = (
            entity.entity_type
            if entity.entity_type in VALID_ENTITY_TYPES
            else "other"
        )
        return self.db.upsert_entity(
            name=entity.name,
            entity_type=safe_type,
            description=entity.description,
            properties=entity.properties if entity.properties else None,
            importance=entity.importance,
            source=entity.source or "inferred",
        )

    def upsert_relation(self, relation: Relation) -> int | None:
        """Insert or update a directed relation between two named entities.

        Entity names are resolved to IDs via the database.  If either entity
        is missing the relation is skipped and ``None`` is returned.

        The ``relation_type`` falls back to ``"works_with"`` when unknown.
        Any ``detail`` text is merged into the ``properties`` dict.

        Args:
            relation: :class:`Relation` instance to persist.

        Returns:
            Primary key of the inserted row, or ``None`` if skipped.
        """
        source_entity = self.db.get_entity(name=relation.source_name)
        target_entity = self.db.get_entity(name=relation.target_name)

        if not source_entity or not target_entity:
            logger.debug(
                "Skipping relation — entity not found: %s → %s",
                relation.source_name,
                relation.target_name,
            )
            return None

        safe_type = (
            relation.relation_type
            if relation.relation_type in VALID_RELATION_TYPES
            else "works_with"
        )

        props = dict(relation.properties)
        if relation.detail:
            props["detail"] = relation.detail

        return self.db.insert_relation(
            source_id=source_entity["id"],
            target_id=target_entity["id"],
            relation_type=safe_type,
            properties=props if props else None,
            confidence=relation.confidence,
            source=relation.source or "inferred",
        )

    # ── Lookup helpers ─────────────────────────────────────────────────────────

    def find_entity(self, name: str) -> dict | None:
        """Find an entity by exact name (case-sensitive, delegated to DB).

        Args:
            name: Entity name to look up.

        Returns:
            Entity dict or ``None``.
        """
        return self.db.get_entity(name=name)

    # ── BFS traversal ─────────────────────────────────────────────────────────

    def traverse(
        self,
        entity_names: list[str],
        max_hops: int = 2,
        max_facts: int = 20,
    ) -> str:
        """BFS traversal from *entity_names* seeds.

        Walks the relation graph up to *max_hops* levels deep and assembles
        a human-readable context string suitable for injection into an LLM
        prompt.

        The actual SQL uses ``source_id`` / ``target_id`` — the column names
        present in the ``relations`` schema defined in ``database.py``.

        Args:
            entity_names: Seed entity names to start traversal from.
            max_hops:     Maximum BFS depth.
            max_facts:    Upper bound on returned facts.

        Returns:
            Newline-joined fact strings, or an empty string when nothing found.
        """
        all_facts: list[str] = []
        visited_ids: set[int] = set()

        for seed_name in entity_names:
            entity = self.find_entity(seed_name)

            if not entity:
                # Fall back to partial (LIKE) match
                rows = self.db.execute(
                    "SELECT * FROM entities WHERE name LIKE ? LIMIT 1",
                    (f"%{seed_name}%",),
                )
                entity = rows[0] if rows else None

            if not entity:
                logger.debug("Traversal: seed entity '%s' not found.", seed_name)
                continue

            # BFS queue: (entity_id, entity_name, current_depth)
            queue: list[tuple[int, str, int]] = [
                (entity["id"], entity["name"], 0)
            ]
            visited_ids.add(entity["id"])

            while queue and len(all_facts) < max_facts:
                current_id, current_name, depth = queue.pop(0)

                if depth >= max_hops:
                    continue

                # Fetch all relations where this entity is source OR target.
                # Note: column names are source_id / target_id (no valid_until).
                relations = self.db.execute(
                    """
                    SELECT
                        r.relation_type,
                        r.confidence,
                        r.properties,
                        CASE WHEN r.source_id = :eid
                             THEN e2.name  ELSE e1.name  END AS other_name,
                        CASE WHEN r.source_id = :eid
                             THEN e2.id    ELSE e1.id    END AS other_id,
                        CASE WHEN r.source_id = :eid
                             THEN e2.description
                             ELSE e1.description          END AS other_desc,
                        CASE WHEN r.source_id = :eid
                             THEN 'outgoing' ELSE 'incoming' END AS direction
                    FROM relations r
                    JOIN entities e1 ON r.source_id = e1.id
                    JOIN entities e2 ON r.target_id = e2.id
                    WHERE (r.source_id = :eid OR r.target_id = :eid)
                    ORDER BY r.confidence DESC
                    LIMIT 10
                    """,
                    {"eid": current_id},
                )

                for row in relations:
                    # Build a readable fact string
                    fact = f"{current_name} {row['relation_type']} {row['other_name']}"

                    other_desc = row.get("other_desc")
                    if other_desc:
                        fact += f" ({other_desc})"

                    props: dict = {}
                    raw_props = row.get("properties")
                    if raw_props:
                        if isinstance(raw_props, dict):
                            props = raw_props
                        else:
                            with contextlib.suppress(json.JSONDecodeError, TypeError):
                                props = json.loads(raw_props)
                    if props.get("detail"):
                        fact += f" — {props['detail']}"

                    all_facts.append(fact)

                    other_id = row.get("other_id")
                    if other_id and other_id not in visited_ids:
                        visited_ids.add(other_id)
                        queue.append((other_id, row["other_name"], depth + 1))

        return "\n".join(all_facts[:max_facts])

    # ── Reranking support ──────────────────────────────────────────────────────

    def has_connection(self, entity_name: str, memory_id: int) -> bool:
        """Check whether a named entity is mentioned in a specific memory.

        Used as a KG-bonus signal during retrieval reranking.  A simple
        substring match on the memory ``content`` is used (fast, no joins).

        Args:
            entity_name: Entity name to look for.
            memory_id:   Memory row to inspect.

        Returns:
            ``True`` when the entity name appears in the memory content.
        """
        mem = self.db.get_memory(memory_id)
        if not mem:
            return False

        entity = self.find_entity(entity_name)
        if not entity:
            return False

        content = mem.get("content", "").lower()
        return entity["name"].lower() in content

    # ── Full entity context ────────────────────────────────────────────────────

    def get_entity_context(self, entity_name: str) -> dict:
        """Return an entity together with all its relations.

        Args:
            entity_name: Name of the entity to fetch context for.

        Returns:
            Dict with keys ``"entity"`` and ``"relations"``, or ``{}`` when
            the entity is not found.
        """
        entity = self.find_entity(entity_name)
        if not entity:
            return {}

        relations = self.db.get_relations(entity["id"], max_results=30)
        return {
            "entity": dict(entity),
            "relations": [dict(r) for r in relations],
        }

    # ── Type inference heuristic ───────────────────────────────────────────────

    def infer_entity_type(self, name: str, description: str) -> str:
        """Heuristically guess an entity type from name and description tokens.

        Args:
            name:        Entity name string.
            description: Short description or empty string.

        Returns:
            One of the :data:`VALID_ENTITY_TYPES` strings.
        """
        name_lower = name.lower()
        desc_lower = description.lower()

        if any(
            w in name_lower
            for w in ("ssh", "nginx", "docker", "git", "npm", "pip", "make", "curl")
        ):
            return "tool"
        if any(
            w in name_lower
            for w in ("python", "javascript", "typescript", "rust", "go", "sql",
                      "bash", "ruby", "java", "c++", "kotlin")
        ):
            return "language"
        if any(w in desc_lower for w in ("service", "server", "api", "endpoint", "daemon")):
            return "service"
        if any(w in desc_lower for w in ("config", "setting", "preference", "option", "flag")):
            return "config"
        if any(w in desc_lower for w in ("project", "repo", "repository", "codebase")):
            return "project"
        if "/" in name or name.startswith("~") or name.endswith(
            (".py", ".js", ".ts", ".toml", ".yaml", ".yml", ".json", ".sh")
        ):
            return "file"

        return "concept"

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a summary of KG size.

        Returns:
            Dict with ``"entities"`` and ``"relations"`` counts.
        """
        entities_rows = self.db.execute("SELECT COUNT(*) AS n FROM entities")
        relations_rows = self.db.execute("SELECT COUNT(*) AS n FROM relations")
        return {
            "entities": entities_rows[0]["n"] if entities_rows else 0,
            "relations": relations_rows[0]["n"] if relations_rows else 0,
        }
