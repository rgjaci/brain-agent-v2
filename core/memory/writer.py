"""Brain Agent v2 — Memory Write Pipeline.

:class:`MemoryWriter` ingests a completed agent interaction (user message,
agent reply, tool calls) and persists three kinds of structured memory:

1. **Episodic facts** — atomic statements extracted by an LLM and stored as
   embedding-indexed memories in ``MemoryDatabase``.
2. **Knowledge-graph updates** — entities and relations extracted and upserted
   into :class:`KnowledgeGraph`.
3. **Procedures** — step-by-step workflows inferred from multi-tool
   interactions, stored in the ``procedures`` table.

All LLM calls are synchronous (OllamaProvider uses ``requests`` under the
hood) and are off-loaded to a thread pool via ``asyncio.to_thread`` so they
don't block the event loop.  Exceptions are always caught, logged, and
swallowed — the writer must never crash the agent.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

FACT_EXTRACTION_PROMPT = """Extract new facts from this exchange.
Only extract genuinely NEW information — things not obvious or common knowledge.
Focus on: user preferences, system configurations, project details, personal facts, corrections.

User: {user_msg}
Assistant: {agent_msg}

Return JSON array (or empty array [] if nothing new):
[{{"content": "...", "category": "fact|preference|observation|correction", "importance": 0.1-1.0}}]

Examples of good facts:
- "User's domain is nhproject.org"
- "User prefers ed25519 SSH keys"
- "Production server IP is 192.168.1.100"
- "User uses .venv/ for Python virtual environments"

Do NOT extract: greetings, generic statements, obvious truths."""

GRAPH_EXTRACTION_PROMPT = """Identify entities and relationships in this exchange.
Only extract SPECIFIC, named entities — not generic words.

User: {user_msg}
Assistant: {agent_msg}

Return JSON:
{{"entities": [
    {{"name": "...", "type": "person|tool|project|concept|service|config|language|file", "description": "one line description"}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "uses|prefers|part_of|depends_on|configured_with|works_with", "detail": "..."}}
  ]
}}

Examples:
- Entity: {{"name": "Tailscale", "type": "service", "description": "VPN mesh network service"}}
- Relation: {{"source": "User", "target": "ed25519", "type": "prefers", "detail": "for SSH authentication"}}"""

PROCEDURE_EXTRACTION_PROMPT = """A successful multi-step operation just completed. Extract a reusable procedure.

User request: {user_msg}
Agent response: {agent_msg}
Tools used: {tool_summary}

Return JSON describing the procedure:
{{
  "name": "short_snake_case_name",
  "description": "One sentence describing what this procedure accomplishes.",
  "trigger_pattern": "Natural language pattern that would trigger this procedure",
  "preconditions": ["condition1", "condition2"],
  "steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "warnings": ["Warning or gotcha to watch out for"],
  "context": "Any important contextual notes"
}}

Focus on the GENERAL pattern, not the specific values used this time.
If no clear reusable procedure exists, return {{}}."""


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FactExtraction:
    """A single atomic fact extracted from an interaction."""

    content: str
    category: str = "fact"
    importance: float = 0.5


@dataclass
class ProcedureExtraction:
    """A reusable step-by-step procedure inferred from a successful interaction."""

    name: str
    description: str
    trigger_pattern: str
    preconditions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    context: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# MemoryWriter
# ──────────────────────────────────────────────────────────────────────────────


class MemoryWriter:
    """Extracts and persists memories, KG updates, and procedures.

    Args:
        llm:     An :class:`OllamaProvider` instance for JSON extraction.
        embedder: A ``GeminiEmbeddingProvider`` (or compatible) instance whose
                  ``embed(text)`` method returns a ``list[float]``.
        db:      An initialised :class:`MemoryDatabase` instance.
        kg:      An initialised :class:`KnowledgeGraph` instance.
    """

    def __init__(self, llm, embedder, db, kg) -> None:
        self.llm = llm
        self.embedder = embedder
        self.db = db
        self.kg = kg

    # ── Public entry-point ─────────────────────────────────────────────────────

    async def process_interaction(
        self,
        user_msg: str,
        agent_msg: str,
        tool_calls: list,
        session_id: str,
    ) -> None:
        """Full write pipeline for one completed interaction.

        Steps:
          1. Extract facts (LLM).
          2. Extract graph entities & relations (LLM).
          3. Optionally extract a procedure when ≥2 tools were used and the
             interaction succeeded.
          4. Deduplicate facts against existing memories.
          5. Embed & store new facts.
          6. Upsert KG entities and relations.
          7. Persist procedure (if any).

        All sub-steps are wrapped in individual try/except blocks; a failure
        in any one step does not abort the others.

        Args:
            user_msg:   The user's message text.
            agent_msg:  The agent's reply text.
            tool_calls: List of tool call objects / dicts used in this turn.
            session_id: Current session identifier (stored as memory source).
        """
        # ── 1 & 2: Extraction (run concurrently) ───────────────────────────
        facts_task = asyncio.create_task(
            self.extract_facts(user_msg, agent_msg)
        )
        graph_task = asyncio.create_task(
            self.extract_graph(user_msg, agent_msg)
        )

        # ── 3: Procedure extraction (conditional) ──────────────────────────
        procedure: Optional[ProcedureExtraction] = None
        if len(tool_calls) >= 2 and self.interaction_succeeded(agent_msg):
            try:
                procedure = await self.extract_procedure(
                    user_msg, agent_msg, tool_calls
                )
            except Exception:
                logger.exception("Procedure extraction failed — skipping.")

        # Await concurrent tasks
        try:
            facts: list[FactExtraction] = await facts_task
        except Exception:
            logger.exception("Fact extraction failed — skipping.")
            facts = []

        try:
            entities_raw, relations_raw = await graph_task
        except Exception:
            logger.exception("Graph extraction failed — skipping.")
            entities_raw, relations_raw = [], []

        # ── 4: Deduplication ───────────────────────────────────────────────
        try:
            unique_facts = await self.deduplicate_facts(facts)
        except Exception:
            logger.exception("Deduplication failed — storing all facts.")
            unique_facts = facts

        # ── 5: Embed & store facts ─────────────────────────────────────────
        for fact in unique_facts:
            try:
                await self._store_fact(fact, session_id)
            except Exception:
                logger.exception("Failed to store fact: %s", fact.content[:80])

        # ── 6: Upsert KG ──────────────────────────────────────────────────
        for entity in entities_raw:
            try:
                self.kg.upsert_entity(entity)
            except Exception:
                logger.exception(
                    "Failed to upsert entity: %s", entity.name
                )

        for relation in relations_raw:
            try:
                self.kg.upsert_relation(relation)
            except Exception:
                logger.exception(
                    "Failed to upsert relation: %s → %s",
                    relation.source_name,
                    relation.target_name,
                )

        # ── 7: Store procedure ─────────────────────────────────────────────
        if procedure:
            try:
                await self._store_procedure(procedure)
            except Exception:
                logger.exception(
                    "Failed to store procedure: %s", procedure.name
                )

        logger.debug(
            "process_interaction done — %d facts, %d entities, %d relations, "
            "procedure=%s",
            len(unique_facts),
            len(entities_raw),
            len(relations_raw),
            procedure.name if procedure else None,
        )

    # ── Fact extraction ────────────────────────────────────────────────────────

    async def extract_facts(
        self, user_msg: str, agent_msg: str
    ) -> list[FactExtraction]:
        """Use the LLM to extract atomic facts from the exchange.

        Args:
            user_msg:  The user's message.
            agent_msg: The agent's reply.

        Returns:
            List of :class:`FactExtraction` instances (may be empty).
        """
        prompt = FACT_EXTRACTION_PROMPT.format(
            user_msg=user_msg, agent_msg=agent_msg
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await asyncio.to_thread(
                self.llm.generate_json, messages, None, 0.1
            )
        except Exception:
            logger.exception("LLM call failed in extract_facts.")
            return []

        if not isinstance(raw, list):
            logger.warning(
                "extract_facts: expected list, got %s — ignoring.", type(raw).__name__
            )
            return []

        facts: list[FactExtraction] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "").strip()
            if not content:
                continue
            category = item.get("category", "fact")
            importance = float(item.get("importance", 0.5))
            # Clamp to valid range
            importance = max(0.1, min(1.0, importance))
            facts.append(FactExtraction(
                content=content,
                category=category,
                importance=importance,
            ))

        logger.debug("extract_facts: %d fact(s) extracted.", len(facts))
        return facts

    # ── Graph extraction ───────────────────────────────────────────────────────

    async def extract_graph(
        self, user_msg: str, agent_msg: str
    ) -> tuple[list, list]:
        """Use the LLM to extract KG entities and relations from the exchange.

        Imports :class:`~brain_agent.core.memory.kg.Entity` and
        :class:`~brain_agent.core.memory.kg.Relation` locally to avoid
        circular imports when this module is used standalone.

        Args:
            user_msg:  The user's message.
            agent_msg: The agent's reply.

        Returns:
            Tuple of ``(list[Entity], list[Relation])``.
        """
        from core.memory.kg import Entity, Relation  # local import

        prompt = GRAPH_EXTRACTION_PROMPT.format(
            user_msg=user_msg, agent_msg=agent_msg
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await asyncio.to_thread(
                self.llm.generate_json, messages, None, 0.1
            )
        except Exception:
            logger.exception("LLM call failed in extract_graph.")
            return [], []

        if not isinstance(raw, dict):
            logger.warning(
                "extract_graph: expected dict, got %s — ignoring.", type(raw).__name__
            )
            return [], []

        entities: list[Entity] = []
        for item in raw.get("entities", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            if not name:
                continue
            etype = item.get("type", "concept")
            desc = item.get("description", "")
            entities.append(Entity(
                name=name,
                entity_type=etype,
                description=desc,
                importance=0.5,
                source="interaction",
            ))

        relations: list[Relation] = []
        for item in raw.get("relations", []):
            if not isinstance(item, dict):
                continue
            src = item.get("source", "").strip()
            tgt = item.get("target", "").strip()
            rtype = item.get("type", "works_with")
            detail = item.get("detail", "")
            if not src or not tgt:
                continue
            relations.append(Relation(
                source_name=src,
                target_name=tgt,
                relation_type=rtype,
                confidence=0.7,
                source="interaction",
                detail=detail,
            ))

        logger.debug(
            "extract_graph: %d entity/entities, %d relation(s).",
            len(entities),
            len(relations),
        )
        return entities, relations

    # ── Procedure extraction ───────────────────────────────────────────────────

    async def extract_procedure(
        self,
        user_msg: str,
        agent_msg: str,
        tool_calls: list,
    ) -> Optional[ProcedureExtraction]:
        """Use the LLM to infer a reusable procedure from a multi-tool interaction.

        Only called when ``len(tool_calls) >= 2`` and the interaction
        succeeded (see :meth:`interaction_succeeded`).

        Args:
            user_msg:   The user's message.
            agent_msg:  The agent's reply.
            tool_calls: List of tool call objects used this turn.

        Returns:
            :class:`ProcedureExtraction` or ``None`` when nothing useful found.
        """
        # Build a compact tool summary
        tool_names: list[str] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name") or tc.get("function", {}).get("name", "unknown")
            elif hasattr(tc, "name"):
                name = tc.name
            else:
                name = str(tc)
            tool_names.append(name)
        tool_summary = ", ".join(tool_names)

        prompt = PROCEDURE_EXTRACTION_PROMPT.format(
            user_msg=user_msg,
            agent_msg=agent_msg,
            tool_summary=tool_summary,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await asyncio.to_thread(
                self.llm.generate_json, messages, None, 0.1
            )
        except Exception:
            logger.exception("LLM call failed in extract_procedure.")
            return None

        if not isinstance(raw, dict) or not raw:
            return None

        name = raw.get("name", "").strip()
        description = raw.get("description", "").strip()
        if not name or not description:
            return None

        return ProcedureExtraction(
            name=name,
            description=description,
            trigger_pattern=raw.get("trigger_pattern", ""),
            preconditions=raw.get("preconditions", []),
            steps=raw.get("steps", []),
            warnings=raw.get("warnings", []),
            context=raw.get("context", ""),
        )

    # ── Success detection ──────────────────────────────────────────────────────

    def interaction_succeeded(self, agent_msg: str) -> bool:
        """Determine whether the agent interaction completed without errors.

        Checks for common failure phrases in the agent message (case-insensitive).

        Args:
            agent_msg: The agent's reply text.

        Returns:
            ``True`` when no error indicators are found.
        """
        failure_indicators = (
            "error:",
            "failed:",
            "i couldn't",
            "i'm unable",
            "i am unable",
            "unable to",
            "could not",
            "cannot complete",
            "i cannot",
            "i can't",
            "exception:",
            "traceback",
            "permission denied",
            "not found",
            "timed out",
        )
        lower = agent_msg.lower()
        return not any(indicator in lower for indicator in failure_indicators)

    # ── Deduplication ─────────────────────────────────────────────────────────

    async def deduplicate_facts(
        self,
        facts: list[FactExtraction],
        threshold: float = 0.92,
    ) -> list[FactExtraction]:
        """Remove facts that are too similar to existing memories.

        For each candidate fact, compute its embedding, search for near
        neighbours in the vector index, and discard the fact when cosine
        similarity exceeds *threshold*.

        Cosine similarity is computed properly by fetching neighbour
        embeddings from the database (L2 distance from sqlite-vec cannot
        be trivially converted to cosine similarity without unit-normalising).

        Args:
            facts:     Candidate facts from :meth:`extract_facts`.
            threshold: Cosine-similarity threshold above which a fact is
                       considered a duplicate.  Default is 0.92.

        Returns:
            Filtered list containing only novel facts.
        """
        import struct as _struct
        import math as _math

        def _cosine_sim(a: list[float], b: list[float]) -> float:
            if len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = _math.sqrt(sum(x * x for x in a))
            nb = _math.sqrt(sum(x * x for x in b))
            if na == 0.0 or nb == 0.0:
                return 0.0
            return dot / (na * nb)

        unique: list[FactExtraction] = []

        for fact in facts:
            try:
                embedding_result = await asyncio.to_thread(
                    self.embedder.embed, [fact.content]
                )
                # Handle both embed([text]) -> [[floats]] and embed(text) -> [floats]
                query_emb = embedding_result[0] if (
                    isinstance(embedding_result, list)
                    and embedding_result
                    and isinstance(embedding_result[0], (list, tuple))
                ) else embedding_result

                neighbours = self.db.vector_search(
                    embedding=query_emb,
                    table="memory_vectors",
                    id_col="memory_id",
                    limit=5,
                )
            except Exception:
                logger.exception(
                    "Deduplication check failed for: %s — keeping fact.",
                    fact.content[:60],
                )
                unique.append(fact)
                continue

            is_duplicate = False
            for neighbour in neighbours:
                nid = neighbour["id"]
                # Fetch neighbour embedding from DB for proper cosine similarity
                try:
                    rows = self.db.execute(
                        "SELECT embedding FROM memory_vectors WHERE memory_id = ?",
                        (nid,),
                    )
                    if not rows:
                        continue
                    blob = rows[0].get("embedding") if isinstance(rows[0], dict) else rows[0][0]
                    if blob is None:
                        continue
                    n_floats = len(blob) // 4
                    neighbor_emb = list(_struct.unpack(f"{n_floats}f", blob))
                    similarity = _cosine_sim(query_emb, neighbor_emb)
                except Exception:
                    continue

                if similarity >= threshold:
                    logger.debug(
                        "Duplicate fact skipped (sim=%.3f): %s",
                        similarity,
                        fact.content[:60],
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(fact)

        logger.debug(
            "deduplicate_facts: %d/%d facts are novel.",
            len(unique),
            len(facts),
        )
        return unique

    # ── Bootstrap / scan ingestion ─────────────────────────────────────────────

    async def extract_from_scan(self, scan_name: str, output: str) -> None:
        """Ingest environment scan output as episodic facts.

        Called by the bootstrap process to seed the memory store with system
        information gathered by shell commands (e.g. ``uname``, ``df``, ``env``).

        The output is passed through :meth:`extract_facts` with a synthetic
        agent message, then stored directly without deduplication (bootstrap
        runs once on a fresh DB).

        Args:
            scan_name: Human-readable label for the scan (e.g. ``"os_info"``).
            output:    Raw text output of the scan command.
        """
        user_msg = f"System scan: {scan_name}"
        agent_msg = output

        try:
            facts = await self.extract_facts(user_msg, agent_msg)
        except Exception:
            logger.exception("extract_from_scan: fact extraction failed for %s.", scan_name)
            facts = []

        for fact in facts:
            try:
                await self._store_fact(fact, source=f"scan:{scan_name}")
            except Exception:
                logger.exception(
                    "extract_from_scan: failed to store fact: %s", fact.content[:60]
                )

        # Also attempt graph extraction from the scan output
        try:
            entities, relations = await self.extract_graph(user_msg, agent_msg)
            for entity in entities:
                try:
                    entity.source = f"scan:{scan_name}"
                    self.kg.upsert_entity(entity)
                except Exception:
                    logger.exception("extract_from_scan: KG entity upsert failed.")
            for relation in relations:
                try:
                    relation.source = f"scan:{scan_name}"
                    self.kg.upsert_relation(relation)
                except Exception:
                    logger.exception("extract_from_scan: KG relation upsert failed.")
        except Exception:
            logger.exception(
                "extract_from_scan: graph extraction failed for %s.", scan_name
            )

        logger.info(
            "extract_from_scan: '%s' → %d fact(s) stored.", scan_name, len(facts)
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _store_fact(self, fact: FactExtraction, source: str = "agent") -> int:
        """Embed a fact and write it to the memories + vector tables.

        Args:
            fact:   The :class:`FactExtraction` to persist.
            source: Provenance label stored in the ``memories.source`` column.

        Returns:
            The new memory row ID.
        """
        # Insert memory row first (synchronous, fast)
        memory_id = self.db.insert_memory(
            content=fact.content,
            category=fact.category,
            source=source,
            importance=fact.importance,
            confidence=1.0,
        )

        # Embed and store vector (potentially slow — run in thread)
        try:
            embeddings = await asyncio.to_thread(
                self.embedder.embed, [fact.content]
            )
            self.db.insert_embedding(
                table="memory_vectors",
                id_col="memory_id",
                rowid=memory_id,
                embedding=embeddings[0],
            )
        except Exception:
            logger.exception(
                "Embedding failed for memory %d — stored without vector.", memory_id
            )

        return memory_id

    async def _store_procedure(self, proc: ProcedureExtraction) -> int:
        """Persist a :class:`ProcedureExtraction` to the ``procedures`` table.

        Also embeds the procedure description and stores it in
        ``procedure_vectors`` for similarity retrieval.

        Args:
            proc: The procedure to store.

        Returns:
            The new procedure row ID.
        """
        import json as _json

        cursor = self.db._conn.execute(
            """
            INSERT INTO procedures
                (name, description, trigger_pattern, preconditions,
                 steps, warnings, context, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                description     = excluded.description,
                trigger_pattern = excluded.trigger_pattern,
                preconditions   = excluded.preconditions,
                steps           = excluded.steps,
                warnings        = excluded.warnings,
                context         = excluded.context
            """,
            (
                proc.name,
                proc.description,
                proc.trigger_pattern,
                _json.dumps(proc.preconditions),
                _json.dumps(proc.steps),
                _json.dumps(proc.warnings),
                proc.context,
                "learned",
                __import__("time").time(),
            ),
        )
        proc_id: int = cursor.lastrowid  # type: ignore[assignment]

        # Embed description for similarity retrieval
        try:
            embed_text = f"{proc.name}: {proc.description}"
            embeddings = await asyncio.to_thread(
                self.embedder.embed, [embed_text]
            )
            self.db.insert_embedding(
                table="procedure_vectors",
                id_col="procedure_id",
                rowid=proc_id,
                embedding=embeddings[0],
            )
        except Exception:
            logger.exception(
                "Embedding failed for procedure %d ('%s').", proc_id, proc.name
            )

        logger.info("Stored procedure '%s' (id=%d).", proc.name, proc_id)
        return proc_id

    # ── Document chunk processing ─────────────────────────────────────────────

    async def process_document_chunk(
        self,
        chunk,
        session_id: str = "ingest",
    ) -> int:
        """Store a document chunk as a memory with category='document'.

        Also creates document and document_section rows to maintain the
        document hierarchy.

        Args:
            chunk:      A :class:`~.documents.DocumentChunk` instance.
            session_id: Session identifier for provenance.

        Returns:
            The new memory row ID.
        """
        import json as _json
        import time as _time

        source_path = chunk.source_path
        doc_id = chunk.metadata.get("doc_id")

        # Ensure a document row exists
        if doc_id is None:
            rows = self.db.execute(
                "SELECT id FROM documents WHERE source_path = ?",
                (source_path,),
            )
            if rows:
                doc_id = rows[0]["id"]
            else:
                cursor = self.db._conn.execute(
                    """INSERT INTO documents (title, source_path, doc_type, total_chunks, created_at)
                       VALUES (?, ?, 'text', ?, ?)""",
                    (chunk.metadata.get("file_name", source_path),
                     source_path, chunk.total_chunks, _time.time()),
                )
                doc_id = cursor.lastrowid

        # Create a document_section row
        self.db._conn.execute(
            """INSERT INTO document_sections (doc_id, title, level, position, created_at)
               VALUES (?, ?, 0, ?, ?)""",
            (doc_id, f"chunk-{chunk.chunk_index}", chunk.chunk_index, _time.time()),
        )

        # Store as memory
        meta = dict(chunk.metadata)
        meta["doc_id"] = doc_id
        meta["chunk_index"] = chunk.chunk_index
        meta["total_chunks"] = chunk.total_chunks

        fact = FactExtraction(
            content=chunk.content,
            category="document",
            importance=0.4,
        )
        memory_id = await self._store_fact(fact, source=f"doc:{source_path}")
        return memory_id

    # ── Contradiction detection ───────────────────────────────────────────────

    async def detect_contradictions(self, new_memory_id: int) -> list[int]:
        """Find existing memories that may contradict the newly stored one.

        If cosine similarity > 0.85 and content differs, marks old memories
        as superseded_by the new one.

        Args:
            new_memory_id: ID of the freshly stored memory.

        Returns:
            List of superseded memory IDs.
        """
        import struct as _struct
        import math as _math

        new_mem = self.db.get_memory(new_memory_id)
        if not new_mem:
            return []

        superseded: list[int] = []
        try:
            embedding_result = await asyncio.to_thread(
                self.embedder.embed, [new_mem["content"]]
            )
            query_emb = embedding_result[0] if (
                isinstance(embedding_result, list)
                and embedding_result
                and isinstance(embedding_result[0], (list, tuple))
            ) else embedding_result
            neighbours = self.db.vector_search(
                embedding=query_emb,
                table="memory_vectors",
                id_col="memory_id",
                limit=10,
            )
        except Exception:
            return []

        def _cosine_sim(a, b):
            if len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = _math.sqrt(sum(x * x for x in a))
            nb = _math.sqrt(sum(x * x for x in b))
            if na == 0.0 or nb == 0.0:
                return 0.0
            return dot / (na * nb)

        for neighbour in neighbours:
            nid = neighbour["id"]
            if nid == new_memory_id:
                continue
            try:
                rows = self.db.execute(
                    "SELECT embedding FROM memory_vectors WHERE memory_id = ?",
                    (nid,),
                )
                if not rows:
                    continue
                blob = rows[0].get("embedding") if isinstance(rows[0], dict) else rows[0][0]
                if blob is None:
                    continue
                n_floats = len(blob) // 4
                neighbor_emb = list(_struct.unpack(f"{n_floats}f", blob))
                similarity = _cosine_sim(query_emb, neighbor_emb)
            except Exception:
                continue

            if similarity > 0.85:
                old_mem = self.db.get_memory(nid)
                if old_mem and old_mem["content"] != new_mem["content"]:
                    self.db.mark_superseded(nid, new_memory_id)
                    superseded.append(nid)
                    logger.debug(
                        "Contradiction detected: memory %d superseded by %d (sim=%.3f)",
                        nid, new_memory_id, similarity,
                    )

        return superseded
