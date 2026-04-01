"""Brain Agent v2 — System 2 Reasoning Engine.

A background LLM-powered deliberate thinking loop that continuously processes
memories to generate insights, formulate questions, discover connections, and
improve the knowledge base — analogous to human "System 2" (slow, deliberate)
thinking.

The engine cycles through several reasoning strategies:

* **Gap analysis** — identifies what is unknown about a topic.
* **Cross-domain connection** — finds links between disparate topics.
* **Rule inference** — derives general heuristics from observations.
* **Contradiction check** — detects inconsistencies across the knowledge base.
* **Procedure improvement** — suggests refinements to learned procedures.
* **Question answering** — attempts to answer previously generated questions.

Usage::

    engine = ReasoningEngine(db, llm, embedder, kg, config)
    await engine.start()       # run in background
    await engine.stop()        # graceful shutdown
    result = await engine.reasoning_cycle()  # single cycle (for testing)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .database import MemoryDatabase
    from .kg import KnowledgeGraph
    from ..config import AgentConfig

logger = logging.getLogger(__name__)

# ── Reasoning strategies ─────────────────────────────────────────────────────

STRATEGIES = [
    "gap_analysis",
    "cross_domain",
    "rule_inference",
    "contradiction_check",
    "procedure_improvement",
    "question_answering",
]

# ── Prompt templates ─────────────────────────────────────────────────────────

GAP_ANALYSIS_PROMPT = """You are analysing a knowledge base about a user. Given these memories about the topic "{topic}", identify knowledge gaps.

Memories:
{memories}

Return a JSON object:
{{"gaps": ["specific gap 1", "specific gap 2"], "questions": ["question to fill gap 1", "question to fill gap 2"], "importance": 0.1-1.0}}

Rules:
- Focus on PRACTICAL gaps (things that would help the agent serve the user better)
- Questions should be naturally answerable from future conversations
- Limit to 3 most important gaps"""

CROSS_DOMAIN_PROMPT = """Given these two sets of memories about different topics, identify connections:

Topic A ({topic_a}):
{memories_a}

Topic B ({topic_b}):
{memories_b}

Return a JSON object:
{{"connections": [{{"insight": "how A relates to B", "importance": 0.1-1.0}}], "new_relations": [{{"source": "entity_a", "target": "entity_b", "type": "uses|depends_on|works_with|part_of", "reasoning": "why"}}]}}

Rules:
- Only report genuine, meaningful connections
- Be specific, not generic"""

RULE_INFERENCE_PROMPT = """Analyse these observations and derive general rules or heuristics:

Observations:
{observations}

Return a JSON array of rules:
[{{"rule": "When X, then Y", "confidence": 0.1-1.0, "evidence_count": N, "category": "rule"}}]

Rules:
- Only derive rules with strong evidence (2+ supporting observations)
- Rules should be actionable and specific
- Limit to 3 most confident rules"""

CONTRADICTION_CHECK_PROMPT = """Check these memories for inconsistencies or contradictions:

{memories}

Return a JSON object:
{{"contradictions": [{{"memory_a": "content of first", "memory_b": "content of second", "explanation": "why they contradict", "resolution": "which is more likely correct and why"}}]}}

Rules:
- Only flag GENUINE contradictions, not complementary facts
- Provide clear reasoning for the resolution"""

PROCEDURE_IMPROVEMENT_PROMPT = """Analyse this procedure and its history. Suggest improvements:

Procedure: {name}
Description: {description}
Steps: {steps}
Success rate: {success_rate}
Last used: {last_used}

Related memories:
{related_memories}

Return a JSON object:
{{"improvements": ["improvement 1", "improvement 2"], "revised_steps": ["step 1", "step 2"], "confidence": 0.1-1.0}}

Rules:
- Only suggest improvements backed by evidence from related memories
- Keep the procedure's core purpose intact"""

QUESTION_ANSWERING_PROMPT = """Try to answer this question using ONLY the provided knowledge:

Question: {question}

Available knowledge:
{knowledge}

Return a JSON object:
{{"answerable": true/false, "answer": "the answer if answerable", "confidence": 0.1-1.0, "reasoning": "how you derived the answer"}}

Rules:
- Only answer if the knowledge ACTUALLY supports the answer
- Set confidence based on how well the knowledge covers the question
- If not answerable, explain what's missing"""


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ReasoningResult:
    """Result of one reasoning cycle."""
    strategy: str = ""
    focus_topic: str = ""
    insights_generated: int = 0
    questions_generated: int = 0
    rules_derived: int = 0
    connections_found: int = 0
    questions_answered: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class FocusTopic:
    """A topic selected for deliberate reasoning."""
    name: str
    source: str  # "entity", "category", "question"
    priority: float = 0.5
    last_reasoned: float = 0.0


# ── ReasoningEngine ──────────────────────────────────────────────────────────

class ReasoningEngine:
    """System 2 deliberate thinking — background LLM reasoning over memories.

    Args:
        db:       Initialised :class:`MemoryDatabase`.
        llm:      LLM provider for reasoning.
        embedder: Embedding provider.
        kg:       :class:`KnowledgeGraph` for entity/relation access.
        interval: Seconds between reasoning cycles.
        max_cycles: Maximum total cycles per session (safety limit).
        on_event: Optional callback for event notifications.
    """

    def __init__(
        self,
        db: MemoryDatabase,
        llm,
        embedder,
        kg: KnowledgeGraph,
        interval: int = 180,
        max_cycles: int = 100,
        on_event=None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.embedder = embedder
        self.kg = kg
        self.interval = interval
        self.max_cycles = max_cycles
        self.on_event = on_event

        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count: int = 0
        self._strategy_index: int = 0
        self._focus_history: list[FocusTopic] = []
        self._results_log: list[ReasoningResult] = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background reasoning loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("ReasoningEngine started (interval=%ds).", self.interval)

    async def stop(self) -> None:
        """Gracefully stop the reasoning loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("ReasoningEngine stopped after %d cycles.", self._cycle_count)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def recent_results(self) -> list[ReasoningResult]:
        """Return the last 10 reasoning results."""
        return self._results_log[-10:]

    async def _loop(self) -> None:
        """Main background loop — cycle through reasoning strategies."""
        while self._running and self._cycle_count < self.max_cycles:
            try:
                result = await self.reasoning_cycle()
                self._results_log.append(result)
            except Exception as exc:
                logger.warning("Reasoning cycle failed: %s", exc)
            await asyncio.sleep(self.interval)
        self._running = False

    # ── Single reasoning cycle ───────────────────────────────────────────────

    async def reasoning_cycle(self) -> ReasoningResult:
        """Execute one deliberate reasoning cycle.

        Selects a focus topic and applies the current strategy.

        Returns:
            :class:`ReasoningResult` with outcomes.
        """
        start = time.time()
        self._cycle_count += 1

        strategy = STRATEGIES[self._strategy_index % len(STRATEGIES)]
        self._strategy_index += 1

        focus = await self._select_focus(strategy)

        self._emit("reasoning_start", {
            "strategy": strategy,
            "focus": focus.name,
            "cycle": self._cycle_count,
        })

        result = ReasoningResult(strategy=strategy, focus_topic=focus.name)

        try:
            if strategy == "gap_analysis":
                result = await self._gap_analysis(focus)
            elif strategy == "cross_domain":
                result = await self._cross_domain(focus)
            elif strategy == "rule_inference":
                result = await self._rule_inference()
            elif strategy == "contradiction_check":
                result = await self._contradiction_check()
            elif strategy == "procedure_improvement":
                result = await self._procedure_improvement()
            elif strategy == "question_answering":
                result = await self._question_answering()
        except Exception as exc:
            logger.warning("Strategy %s failed: %s", strategy, exc)

        result.strategy = strategy
        result.focus_topic = focus.name
        result.elapsed_seconds = time.time() - start

        if result.insights_generated > 0 or result.questions_answered > 0:
            self._emit("reasoning_insight", {
                "strategy": strategy,
                "topic": focus.name,
                "insights": result.insights_generated,
                "questions_answered": result.questions_answered,
            })

        focus.last_reasoned = time.time()
        self._focus_history.append(focus)

        logger.debug(
            "reasoning_cycle: strategy=%s, focus=%s, insights=%d, "
            "questions=%d, rules=%d, elapsed=%.1fs",
            strategy, focus.name, result.insights_generated,
            result.questions_generated, result.rules_derived,
            result.elapsed_seconds,
        )

        return result

    # ── Focus selection ──────────────────────────────────────────────────────

    async def _select_focus(self, strategy: str) -> FocusTopic:
        """Select a topic to focus on for the current reasoning cycle.

        Priority factors:
        - Freshness: recently added memories rank higher
        - Open questions: topics with unanswered questions rank higher
        - Diversity: avoid fixating on one topic (round-robin penalty)
        """
        # For question_answering, pick a stored question
        if strategy == "question_answering":
            questions = self.db.execute(
                """
                SELECT id, content FROM memories
                 WHERE category = 'question' AND superseded_by IS NULL
                 ORDER BY created_at DESC LIMIT 5
                """
            )
            if questions:
                q = questions[0]
                return FocusTopic(
                    name=q["content"][:100],
                    source="question",
                    priority=0.8,
                )

        # For procedure_improvement, pick a procedure
        if strategy == "procedure_improvement":
            procs = self.db.execute(
                """
                SELECT name FROM procedures
                 ORDER BY last_used DESC NULLS LAST
                 LIMIT 5
                """
            )
            if procs:
                return FocusTopic(
                    name=procs[0]["name"],
                    source="procedure",
                    priority=0.7,
                )

        # For other strategies, pick an entity or category as focus
        entities = self.db.execute(
            """
            SELECT name, entity_type, importance FROM entities
             ORDER BY importance DESC, created_at DESC
             LIMIT 20
            """
        )

        # Apply round-robin penalty
        recent_topics = {f.name for f in self._focus_history[-5:]}
        best = None
        best_score = -1

        for entity in entities:
            score = entity.get("importance", 0.5)
            if entity["name"] in recent_topics:
                score *= 0.3  # penalise recently focused topics
            if score > best_score:
                best_score = score
                best = entity

        if best:
            return FocusTopic(
                name=best["name"],
                source="entity",
                priority=best_score,
            )

        # Fallback: use a category
        return FocusTopic(name="general", source="category", priority=0.3)

    # ── Strategy implementations ─────────────────────────────────────────────

    async def _gap_analysis(self, focus: FocusTopic) -> ReasoningResult:
        """Identify knowledge gaps about a topic."""
        result = ReasoningResult()

        memories = self.db.execute(
            """
            SELECT content FROM memories
             WHERE superseded_by IS NULL
               AND (content LIKE ? OR content LIKE ?)
             ORDER BY created_at DESC LIMIT 20
            """,
            (f"%{focus.name}%", f"%{focus.name.lower()}%"),
        )
        if not memories:
            return result

        mem_text = "\n".join(f"- {m['content']}" for m in memories)
        prompt = GAP_ANALYSIS_PROMPT.format(
            topic=focus.name, memories=mem_text,
        )

        raw = await asyncio.to_thread(
            self.llm.generate_json,
            [{"role": "user", "content": prompt}],
            None, 0.3,
        )

        if not isinstance(raw, dict):
            return result

        for question in raw.get("questions", []):
            if isinstance(question, str) and question.strip():
                self.db.insert_memory(
                    content=question.strip(),
                    category="question",
                    source="reasoning",
                    importance=float(raw.get("importance", 0.5)),
                )
                result.questions_generated += 1

        return result

    async def _cross_domain(self, focus: FocusTopic) -> ReasoningResult:
        """Find connections between the focus topic and another domain."""
        result = ReasoningResult()

        # Pick another entity as topic B
        entities = self.db.execute(
            "SELECT name FROM entities WHERE name != ? ORDER BY RANDOM() LIMIT 1",
            (focus.name,),
        )
        if not entities:
            return result
        topic_b = entities[0]["name"]

        memories_a = self.db.execute(
            "SELECT content FROM memories WHERE superseded_by IS NULL AND content LIKE ? LIMIT 10",
            (f"%{focus.name}%",),
        )
        memories_b = self.db.execute(
            "SELECT content FROM memories WHERE superseded_by IS NULL AND content LIKE ? LIMIT 10",
            (f"%{topic_b}%",),
        )

        if not memories_a or not memories_b:
            return result

        text_a = "\n".join(f"- {m['content']}" for m in memories_a)
        text_b = "\n".join(f"- {m['content']}" for m in memories_b)

        prompt = CROSS_DOMAIN_PROMPT.format(
            topic_a=focus.name, memories_a=text_a,
            topic_b=topic_b, memories_b=text_b,
        )

        raw = await asyncio.to_thread(
            self.llm.generate_json,
            [{"role": "user", "content": prompt}],
            None, 0.3,
        )

        if not isinstance(raw, dict):
            return result

        for conn in raw.get("connections", []):
            if isinstance(conn, dict) and conn.get("insight"):
                self.db.insert_memory(
                    content=f"Connection: {conn['insight']}",
                    category="insight",
                    source="reasoning",
                    importance=float(conn.get("importance", 0.5)),
                )
                result.insights_generated += 1
                result.connections_found += 1

        return result

    async def _rule_inference(self) -> ReasoningResult:
        """Derive general rules from observations."""
        result = ReasoningResult()

        observations = self.db.execute(
            """
            SELECT content FROM memories
             WHERE superseded_by IS NULL
               AND category IN ('observation', 'fact', 'knowledge')
             ORDER BY created_at DESC LIMIT 30
            """
        )
        if len(observations) < 5:
            return result

        obs_text = "\n".join(f"- {o['content']}" for o in observations)
        prompt = RULE_INFERENCE_PROMPT.format(observations=obs_text)

        raw = await asyncio.to_thread(
            self.llm.generate_json,
            [{"role": "user", "content": prompt}],
            None, 0.3,
        )

        if not isinstance(raw, list):
            return result

        for item in raw:
            if not isinstance(item, dict):
                continue
            rule = item.get("rule", "").strip()
            if not rule:
                continue
            confidence = float(item.get("confidence", 0.5))
            if confidence < 0.5:
                continue
            self.db.insert_memory(
                content=f"Rule: {rule}",
                category="rule",
                source="reasoning",
                importance=max(0.1, min(1.0, confidence)),
            )
            result.rules_derived += 1

        return result

    async def _contradiction_check(self) -> ReasoningResult:
        """Check for contradictions in the knowledge base."""
        result = ReasoningResult()

        memories = self.db.execute(
            """
            SELECT content, category FROM memories
             WHERE superseded_by IS NULL
             ORDER BY created_at DESC LIMIT 40
            """
        )
        if len(memories) < 5:
            return result

        mem_text = "\n".join(
            f"- [{m['category']}] {m['content']}" for m in memories
        )
        prompt = CONTRADICTION_CHECK_PROMPT.format(memories=mem_text)

        raw = await asyncio.to_thread(
            self.llm.generate_json,
            [{"role": "user", "content": prompt}],
            None, 0.2,
        )

        if not isinstance(raw, dict):
            return result

        for contra in raw.get("contradictions", []):
            if not isinstance(contra, dict):
                continue
            explanation = contra.get("explanation", "").strip()
            resolution = contra.get("resolution", "").strip()
            if explanation:
                self.db.insert_memory(
                    content=f"Contradiction detected: {explanation}. Resolution: {resolution}",
                    category="insight",
                    source="reasoning",
                    importance=0.7,
                )
                result.insights_generated += 1

        return result

    async def _procedure_improvement(self) -> ReasoningResult:
        """Suggest improvements to learned procedures."""
        result = ReasoningResult()

        procs = self.db.execute(
            "SELECT * FROM procedures ORDER BY last_used DESC NULLS LAST LIMIT 1"
        )
        if not procs:
            return result

        proc = procs[0]
        total = proc.get("success_count", 0) + proc.get("failure_count", 0)
        success_rate = (
            f"{proc.get('success_count', 0)}/{total}"
            if total > 0
            else "untested"
        )

        # Find related memories
        related = self.db.execute(
            "SELECT content FROM memories WHERE superseded_by IS NULL AND content LIKE ? LIMIT 10",
            (f"%{proc.get('name', '')}%",),
        )
        related_text = "\n".join(f"- {m['content']}" for m in related) or "(none)"

        prompt = PROCEDURE_IMPROVEMENT_PROMPT.format(
            name=proc.get("name", ""),
            description=proc.get("description", ""),
            steps=proc.get("steps", "[]"),
            success_rate=success_rate,
            last_used=proc.get("last_used", "never"),
            related_memories=related_text,
        )

        raw = await asyncio.to_thread(
            self.llm.generate_json,
            [{"role": "user", "content": prompt}],
            None, 0.2,
        )

        if not isinstance(raw, dict):
            return result

        improvements = raw.get("improvements", [])
        if improvements:
            self.db.insert_memory(
                content=f"Procedure improvement for '{proc.get('name', '')}': {'; '.join(str(i) for i in improvements)}",
                category="insight",
                source="reasoning",
                importance=0.6,
            )
            result.insights_generated += 1

        return result

    async def _question_answering(self) -> ReasoningResult:
        """Attempt to answer stored questions from existing knowledge."""
        result = ReasoningResult()

        questions = self.db.execute(
            """
            SELECT id, content FROM memories
             WHERE category = 'question' AND superseded_by IS NULL
             ORDER BY importance DESC, created_at DESC
             LIMIT 3
            """
        )
        if not questions:
            return result

        q = questions[0]

        # Gather all knowledge
        knowledge = self.db.execute(
            """
            SELECT content FROM memories
             WHERE superseded_by IS NULL AND category != 'question'
             ORDER BY importance DESC LIMIT 30
            """
        )
        if not knowledge:
            return result

        knowledge_text = "\n".join(f"- {k['content']}" for k in knowledge)
        prompt = QUESTION_ANSWERING_PROMPT.format(
            question=q["content"],
            knowledge=knowledge_text,
        )

        raw = await asyncio.to_thread(
            self.llm.generate_json,
            [{"role": "user", "content": prompt}],
            None, 0.3,
        )

        if not isinstance(raw, dict):
            return result

        if raw.get("answerable") and raw.get("answer"):
            confidence = float(raw.get("confidence", 0.5))
            if confidence >= 0.6:
                # Store the answer as knowledge
                self.db.insert_memory(
                    content=raw["answer"].strip(),
                    category="knowledge",
                    source="reasoning",
                    importance=confidence,
                )
                # Mark the question as answered (supersede it)
                answer_id = self.db.insert_memory(
                    content=f"Answered: {q['content']} → {raw['answer'].strip()}",
                    category="insight",
                    source="reasoning",
                    importance=confidence,
                )
                self.db.mark_superseded(q["id"], answer_id)
                result.questions_answered += 1
                result.insights_generated += 1

        return result

    # ── Utilities ────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, data: dict) -> None:
        """Fire event to listeners (e.g. TUI)."""
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass
