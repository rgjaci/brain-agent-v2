"""Brain Agent v2 — AutoDream Engine.

LLM-powered memory consolidation inspired by human REM sleep.  While the
existing :class:`ConsolidationEngine` uses heuristics (cosine thresholds,
decay curves), the DreamEngine leverages an LLM to:

* **Create abstractions** — cluster related memories and synthesise higher-
  level insights.
* **Resolve contradictions** — detect and reconcile semantically conflicting
  memories.
* **Detect patterns** — identify recurring themes and regularities across the
  knowledge base.
* **Strengthen connections** — infer missing knowledge-graph relations.
* **Generate questions** — surface knowledge gaps for future investigation.

Usage::

    engine = DreamEngine(db, llm, embedder, consolidator, kg)
    report = await engine.dream()          # full dream cycle
    await engine.maybe_dream(turn_count)   # auto-trigger on schedule
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.provider import OllamaProvider
    from .consolidation import ConsolidationEngine
    from .database import MemoryDatabase
    from .kg import KnowledgeGraph

logger = logging.getLogger(__name__)

# Default scheduling constants
DREAM_INTERVAL_TURNS: int = 50
DREAM_IDLE_THRESHOLD: int = 600  # seconds


# ── Prompt templates ─────────────────────────────────────────────────────────

ABSTRACTION_PROMPT = """You are a memory analyst. Given a cluster of related facts, create a single higher-level insight that summarises the key information.

Facts:
{facts}

Return a JSON object:
{{"insight": "A concise higher-level summary that captures the essence of these facts", "importance": 0.1-1.0}}

Rules:
- The insight should be MORE GENERAL than the individual facts
- It should capture the common theme or pattern
- Importance should reflect how useful this insight is"""

CONTRADICTION_PROMPT = """Analyse these two memories for contradictions:

Memory A: {memory_a}
Memory B: {memory_b}

Return a JSON object:
{{"contradicts": true/false, "resolution": "keep_a"|"keep_b"|"merge", "merged_content": "merged fact if resolution is merge, else empty string", "reasoning": "brief explanation"}}

Rules:
- Only mark as contradicting if they truly conflict (not just different aspects of the same topic)
- If one is more recent or specific, prefer it
- If they can be combined, merge them"""

PATTERN_PROMPT = """Analyse these recent memories and identify patterns, themes, or recurring topics:

{memories}

Return a JSON array of patterns found:
[{{"pattern": "description of the pattern", "importance": 0.1-1.0, "evidence_count": N}}]

Rules:
- Only report genuine patterns (appearing 2+ times)
- Rank by importance and frequency
- Be specific, not generic"""

CONNECTION_PROMPT = """Given this entity and its current relations, suggest missing connections:

Entity: {entity_name} (type: {entity_type})
Current relations: {relations}
Available entities: {other_entities}

Return a JSON array of suggested relations:
[{{"source": "entity_name", "target": "other_entity_name", "type": "uses|prefers|part_of|depends_on|works_with", "confidence": 0.1-1.0, "reasoning": "why this connection exists"}}]

Rules:
- Only suggest confident connections (>0.5)
- Base suggestions on the context, not guesses
- Limit to 3 most confident suggestions"""

QUESTION_PROMPT = """Based on these memories and knowledge gaps, generate questions that would help fill gaps:

{memories}

Known entities: {entities}

Return a JSON array of questions:
[{{"question": "What is...?", "importance": 0.1-1.0, "topic": "related topic"}}]

Rules:
- Focus on knowledge GAPS — things that are hinted at but not known
- Questions should be answerable from the user's future interactions
- Limit to 5 most important questions"""


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class DreamReport:
    """Summary of a dream cycle's results."""
    abstractions_created: int = 0
    contradictions_resolved: int = 0
    patterns_detected: int = 0
    connections_added: int = 0
    questions_generated: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ── DreamEngine ──────────────────────────────────────────────────────────────

class DreamEngine:
    """LLM-powered memory consolidation — the agent's REM sleep.

    Args:
        db:           Initialised :class:`MemoryDatabase`.
        llm:          LLM provider for reasoning.
        embedder:     Embedding provider for similarity computation.
        consolidator: Existing heuristic :class:`ConsolidationEngine`.
        kg:           :class:`KnowledgeGraph` for entity/relation access.
    """

    def __init__(
        self,
        db: MemoryDatabase,
        llm: OllamaProvider,
        embedder,
        consolidator: ConsolidationEngine,
        kg: KnowledgeGraph,
    ) -> None:
        self.db = db
        self.llm = llm
        self.embedder = embedder
        self.consolidator = consolidator
        self.kg = kg
        self._last_dream: float = 0.0

    # ── Scheduling ───────────────────────────────────────────────────────────

    async def maybe_dream(
        self,
        turn_count: int,
        interval: int = DREAM_INTERVAL_TURNS,
        idle_threshold: int = DREAM_IDLE_THRESHOLD,
    ) -> DreamReport | None:
        """Auto-trigger a dream cycle based on turn count or idle time.

        Args:
            turn_count:      Current 1-based turn index.
            interval:        Dream every N turns.
            idle_threshold:  Dream after this many seconds idle.

        Returns:
            :class:`DreamReport` if a dream ran, else ``None``.
        """
        now = time.time()
        idle_enough = (
            self._last_dream > 0
            and (now - self._last_dream) > idle_threshold
        )
        turn_trigger = turn_count > 0 and turn_count % interval == 0

        if turn_trigger or idle_enough:
            logger.info(
                "maybe_dream: triggering (turn=%d, idle=%.0fs).",
                turn_count, now - self._last_dream if self._last_dream else 0,
            )
            return await self.dream()
        return None

    # ── Full dream cycle ─────────────────────────────────────────────────────

    async def dream(self) -> DreamReport:
        """Run a full dream cycle: heuristic consolidation + LLM-powered steps.

        Returns:
            :class:`DreamReport` with counts and timing.
        """
        start = time.time()
        report = DreamReport()

        # Step 0: Run heuristic consolidation first
        try:
            await self.consolidator.consolidate()
        except Exception as exc:
            logger.warning("dream: heuristic consolidation failed — %s", exc)
            report.errors.append(f"consolidation: {exc}")

        # Step 1: Create abstractions from memory clusters
        try:
            report.abstractions_created = await self._create_abstractions()
        except Exception as exc:
            logger.warning("dream: abstraction creation failed — %s", exc)
            report.errors.append(f"abstractions: {exc}")

        # Step 2: LLM-powered contradiction resolution
        try:
            report.contradictions_resolved = await self._resolve_contradictions_llm()
        except Exception as exc:
            logger.warning("dream: contradiction resolution failed — %s", exc)
            report.errors.append(f"contradictions: {exc}")

        # Step 3: Pattern detection
        try:
            report.patterns_detected = await self._detect_patterns()
        except Exception as exc:
            logger.warning("dream: pattern detection failed — %s", exc)
            report.errors.append(f"patterns: {exc}")

        # Step 4: Connection strengthening
        try:
            report.connections_added = await self._strengthen_connections()
        except Exception as exc:
            logger.warning("dream: connection strengthening failed — %s", exc)
            report.errors.append(f"connections: {exc}")

        # Step 5: Question generation
        try:
            report.questions_generated = await self._generate_questions()
        except Exception as exc:
            logger.warning("dream: question generation failed — %s", exc)
            report.errors.append(f"questions: {exc}")

        report.elapsed_seconds = time.time() - start
        self._last_dream = time.time()

        logger.info(
            "dream: cycle complete in %.1fs — abstractions=%d, "
            "contradictions=%d, patterns=%d, connections=%d, questions=%d",
            report.elapsed_seconds,
            report.abstractions_created,
            report.contradictions_resolved,
            report.patterns_detected,
            report.connections_added,
            report.questions_generated,
        )

        return report

    # ── Step 1: Abstraction creation ─────────────────────────────────────────

    async def _create_abstractions(self, max_clusters: int = 5) -> int:
        """Cluster similar memories and create higher-level insights.

        Groups memories by embedding similarity, then asks the LLM to
        synthesise each cluster into a single insight.

        Returns:
            Number of abstractions created.
        """
        embeddings = self.db.get_all_embeddings(
            table="memory_vectors", id_col="memory_id", limit=500
        )
        if len(embeddings) < 3:
            return 0

        # Simple greedy clustering: pick a seed, gather neighbours
        used: set[int] = set()
        clusters: list[list[int]] = []

        for mem_id, emb in embeddings:
            if mem_id in used or len(clusters) >= max_clusters:
                break
            cluster = [mem_id]
            used.add(mem_id)

            for other_id, other_emb in embeddings:
                if other_id in used:
                    continue
                sim = _cosine_similarity(emb, other_emb)
                if sim > 0.75:
                    cluster.append(other_id)
                    used.add(other_id)
                    if len(cluster) >= 8:
                        break

            if len(cluster) >= 3:
                clusters.append(cluster)

        count = 0
        for cluster_ids in clusters:
            memories = []
            for mid in cluster_ids:
                mem = self.db.get_memory(mid)
                if mem:
                    memories.append(mem)

            if len(memories) < 3:
                continue

            facts_text = "\n".join(
                f"- {m['content']}" for m in memories
            )
            prompt = ABSTRACTION_PROMPT.format(facts=facts_text)

            try:
                result = await asyncio.to_thread(
                    self.llm.generate_json,
                    [{"role": "user", "content": prompt}],
                    None, 0.2,
                )
            except Exception:
                logger.debug("Abstraction LLM call failed for cluster.")
                continue

            if not isinstance(result, dict):
                continue

            insight = result.get("insight", "").strip()
            importance = float(result.get("importance", 0.6))
            if not insight:
                continue

            # Store as insight memory
            new_id = self.db.insert_memory(
                content=insight,
                category="insight",
                source="dream",
                importance=max(0.1, min(1.0, importance)),
            )
            # Link back to source memories
            for mid in cluster_ids:
                self.db.link_memory_source(new_id, "dream", f"cluster:{mid}")

            count += 1

        return count

    # ── Step 2: LLM contradiction resolution ─────────────────────────────────

    async def _resolve_contradictions_llm(self, max_pairs: int = 5) -> int:
        """Find and resolve contradictory memories using LLM judgment.

        Returns:
            Number of contradictions resolved.
        """
        embeddings = self.db.get_all_embeddings(
            table="memory_vectors", id_col="memory_id", limit=300
        )
        if len(embeddings) < 2:
            return 0

        # Find pairs with high similarity (potential contradictions)
        candidates: list[tuple[int, int, float]] = []
        for i, (id_a, emb_a) in enumerate(embeddings):
            for id_b, emb_b in embeddings[i + 1:]:
                sim = _cosine_similarity(emb_a, emb_b)
                if 0.80 < sim < 0.95:  # similar but not identical
                    candidates.append((id_a, id_b, sim))
                    if len(candidates) >= max_pairs * 2:
                        break
            if len(candidates) >= max_pairs * 2:
                break

        # Sort by similarity descending and take top pairs
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[:max_pairs]

        resolved = 0
        for id_a, id_b, _sim in candidates:
            mem_a = self.db.get_memory(id_a)
            mem_b = self.db.get_memory(id_b)
            if not mem_a or not mem_b:
                continue

            prompt = CONTRADICTION_PROMPT.format(
                memory_a=mem_a["content"],
                memory_b=mem_b["content"],
            )

            try:
                result = await asyncio.to_thread(
                    self.llm.generate_json,
                    [{"role": "user", "content": prompt}],
                    None, 0.1,
                )
            except Exception:
                continue

            if not isinstance(result, dict) or not result.get("contradicts"):
                continue

            resolution = result.get("resolution", "")
            if resolution == "keep_a":
                self.db.mark_superseded(id_b, id_a)
                resolved += 1
            elif resolution == "keep_b":
                self.db.mark_superseded(id_a, id_b)
                resolved += 1
            elif resolution == "merge":
                merged = result.get("merged_content", "").strip()
                if merged:
                    new_id = self.db.insert_memory(
                        content=merged,
                        category=mem_a.get("category", "fact"),
                        source="dream",
                        importance=max(
                            mem_a.get("importance", 0.5),
                            mem_b.get("importance", 0.5),
                        ),
                    )
                    self.db.mark_superseded(id_a, new_id)
                    self.db.mark_superseded(id_b, new_id)
                    self.db.link_memory_source(new_id, "dream", f"merge:{id_a}:{id_b}")
                    resolved += 1

        return resolved

    # ── Step 3: Pattern detection ────────────────────────────────────────────

    async def _detect_patterns(self, sample_size: int = 50) -> int:
        """Identify recurring patterns across recent memories.

        Returns:
            Number of patterns stored as insights.
        """
        rows = self.db.execute(
            """
            SELECT content, category FROM memories
             WHERE superseded_by IS NULL
             ORDER BY created_at DESC
             LIMIT ?
            """,
            (sample_size,),
        )
        if len(rows) < 5:
            return 0

        memories_text = "\n".join(
            f"- [{r['category']}] {r['content']}" for r in rows
        )
        prompt = PATTERN_PROMPT.format(memories=memories_text)

        try:
            result = await asyncio.to_thread(
                self.llm.generate_json,
                [{"role": "user", "content": prompt}],
                None, 0.2,
            )
        except Exception:
            return 0

        if not isinstance(result, list):
            return 0

        count = 0
        for item in result:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern", "").strip()
            importance = float(item.get("importance", 0.5))
            if not pattern:
                continue
            self.db.insert_memory(
                content=f"Pattern: {pattern}",
                category="insight",
                source="dream",
                importance=max(0.1, min(1.0, importance)),
            )
            count += 1

        return count

    # ── Step 4: Connection strengthening ─────────────────────────────────────

    async def _strengthen_connections(self, max_entities: int = 5) -> int:
        """Infer missing knowledge-graph relations for under-connected entities.

        Returns:
            Number of new relations added.
        """
        from .kg import VALID_RELATION_TYPES

        # Find entities with few relations
        rows = self.db.execute(
            """
            SELECT e.id, e.name, e.entity_type,
                   COUNT(r.id) AS rel_count
              FROM entities e
              LEFT JOIN relations r ON r.source_id = e.id OR r.target_id = e.id
             GROUP BY e.id
            HAVING rel_count < 3
             ORDER BY e.importance DESC
             LIMIT ?
            """,
            (max_entities,),
        )
        if not rows:
            return 0

        # Get all entity names for context
        all_entities = self.db.execute(
            "SELECT name, entity_type FROM entities LIMIT 50"
        )
        entity_list = ", ".join(
            f"{e['name']} ({e['entity_type']})" for e in all_entities
        )

        count = 0
        for row in rows:
            eid = row["id"]
            relations = self.db.get_relations(eid, max_results=10)
            rel_text = "\n".join(
                f"- {r.get('source_name', '?')} → {r.get('relation_type', '?')} → {r.get('target_name', '?')}"
                for r in relations
            ) or "(none)"

            prompt = CONNECTION_PROMPT.format(
                entity_name=row["name"],
                entity_type=row["entity_type"],
                relations=rel_text,
                other_entities=entity_list,
            )

            try:
                result = await asyncio.to_thread(
                    self.llm.generate_json,
                    [{"role": "user", "content": prompt}],
                    None, 0.2,
                )
            except Exception:
                continue

            if not isinstance(result, list):
                continue

            for suggestion in result:
                if not isinstance(suggestion, dict):
                    continue
                confidence = float(suggestion.get("confidence", 0))
                if confidence < 0.5:
                    continue
                rtype = suggestion.get("type", "works_with")
                if rtype not in VALID_RELATION_TYPES:
                    continue

                target_name = suggestion.get("target", "").strip()
                if not target_name:
                    continue

                target = self.db.get_entity(name=target_name)
                if not target:
                    continue

                self.db.insert_relation(
                    source_id=eid,
                    target_id=target["id"],
                    relation_type=rtype,
                    confidence=confidence,
                    source="dream",
                )
                count += 1

        return count

    # ── Step 5: Question generation ──────────────────────────────────────────

    async def _generate_questions(self, sample_size: int = 30) -> int:
        """Identify knowledge gaps and store questions for future investigation.

        Returns:
            Number of questions generated.
        """
        rows = self.db.execute(
            """
            SELECT content, category FROM memories
             WHERE superseded_by IS NULL
             ORDER BY created_at DESC
             LIMIT ?
            """,
            (sample_size,),
        )
        if len(rows) < 3:
            return 0

        entities = self.db.execute("SELECT name FROM entities LIMIT 30")
        entity_names = ", ".join(e["name"] for e in entities) or "(none)"

        memories_text = "\n".join(
            f"- [{r['category']}] {r['content']}" for r in rows
        )

        prompt = QUESTION_PROMPT.format(
            memories=memories_text,
            entities=entity_names,
        )

        try:
            result = await asyncio.to_thread(
                self.llm.generate_json,
                [{"role": "user", "content": prompt}],
                None, 0.3,
            )
        except Exception:
            return 0

        if not isinstance(result, list):
            return 0

        count = 0
        for item in result:
            if not isinstance(item, dict):
                continue
            question = item.get("question", "").strip()
            importance = float(item.get("importance", 0.4))
            if not question:
                continue
            new_id = self.db.insert_memory(
                content=question,
                category="question",
                source="dream",
                importance=max(0.1, min(1.0, importance)),
            )
            self.db.link_memory_source(new_id, "dream", "question_generation")
            count += 1

        return count
