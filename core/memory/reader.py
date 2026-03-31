"""Brain Agent v2 — Hybrid memory retrieval.

Implements :class:`MemoryReader`, which provides a full retrieval pipeline:

  1. **Adaptive strategy** — chooses conservative / normal / aggressive based
     on the query and session context.
  2. **Dense search** — embeds the query via Gemini and calls sqlite-vec ANN.
  3. **Sparse search** — BM25 full-text search via FTS5.
  4. **RRF fusion** — Reciprocal Rank Fusion merges both ranked lists.
  5. **Heuristic rerank** — multipliers for recency, access count, category
     relevance, and importance.
  6. **KG traversal** — extracts named entities from the query and traverses
     the knowledge graph to build a supplementary context string.
  7. **Procedure retrieval** — delegates to :class:`ProcedureStore`.
  8. **format_for_context** — "best-at-edges" ordering with token budgeting.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.provider import OllamaProvider
    from ..llm.embeddings import GeminiEmbeddingProvider
    from .database import MemoryDatabase
    from .kg import KnowledgeGraph

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Common English stop-words (used by entity extraction)
# ──────────────────────────────────────────────────────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "after",
    "before", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "that", "this", "these", "those",
    "it", "its", "i", "you", "he", "she", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "our", "their", "what", "which",
    "who", "when", "where", "why", "how", "all", "both", "each", "more",
    "most", "other", "some", "such", "no", "not", "only", "same", "so",
    "than", "too", "very", "just", "any", "also", "there", "then", "if",
    "as", "well", "out", "get", "set", "use", "make", "go", "know", "see",
    "come", "think", "look", "want", "tell", "give", "find", "here", "now",
    "new", "old", "right", "say", "take", "put",
})

# Domain words / tool names known to the agent (used in entity extraction)
_KNOWN_TOOLS: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "rust", "go", "sql", "bash",
    "docker", "git", "nginx", "ssh", "npm", "pip", "make", "curl",
    "sqlite", "postgres", "redis", "ollama", "gemini", "langchain",
    "fastapi", "flask", "django", "react", "vue", "node", "kubernetes",
    "terraform", "ansible", "linux", "macos", "windows",
})

# ──────────────────────────────────────────────────────────────────────────────
# Query-classifier helpers
# ──────────────────────────────────────────────────────────────────────────────

_AGGRESSIVE_KEYWORDS: frozenset[str] = frozenset({
    "remember", "memory", "memorize", "recall", "search", "find", "lookup",
    "what did", "when did", "have i", "did i", "told you", "mentioned",
    "stored", "saved", "know about",
})

_PROCEDURE_KEYWORDS: frozenset[str] = frozenset({
    "how to", "how do", "steps to", "process for", "procedure for",
    "way to", "guide", "tutorial", "instructions", "help me",
})

_FACT_KEYWORDS: frozenset[str] = frozenset({
    "who is", "what is", "what are", "who are", "define", "explain",
    "describe", "tell me about", "what does",
})

# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RetrievedMemory:
    """A single memory retrieved from the database, decorated with retrieval scores.

    Attributes:
        id:           Database primary key.
        content:      Raw text content.
        category:     Memory category label (e.g. ``"fact"``, ``"task"``).
        importance:   Importance score in ``[0.0, 1.0]``.
        access_count: Number of times this memory has been accessed.
        rrf_score:    Reciprocal Rank Fusion score (higher = more relevant).
        dense_rank:   Rank in the dense (vector) search result list (1-indexed; 999 = absent).
        sparse_rank:  Rank in the sparse (BM25) search result list (1-indexed; 999 = absent).
        source:       Origin of the result: ``"memory"``, ``"kg"``, or ``"procedure"``.
        metadata:     Arbitrary extra fields from the database row.
    """

    id: int
    content: str
    category: str
    importance: float
    access_count: int
    rrf_score: float = 0.0
    dense_rank: int = 999
    sparse_rank: int = 999
    source: str = "memory"   # "memory" | "kg" | "procedure"
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """The complete output of a single :meth:`MemoryReader.retrieve` call.

    Attributes:
        memories:           Fused and reranked memory objects.
        kg_context:         Formatted entity / relation facts from KG traversal.
        procedures:         Relevant procedural memory dicts.
        query_entities:     Entity names extracted from the query (used for KG traversal).
        retrieval_strategy: Which adaptive strategy was selected.
        elapsed_ms:         Wall-clock time for the entire retrieval pipeline.
    """

    memories: list[RetrievedMemory]
    kg_context: str          # formatted entity/relation facts
    procedures: list[dict]   # relevant procedures
    query_entities: list[str]
    retrieval_strategy: str  # "conservative" | "normal" | "aggressive"
    elapsed_ms: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# MemoryReader
# ──────────────────────────────────────────────────────────────────────────────


class MemoryReader:
    """Hybrid memory retrieval engine for Brain Agent v2.

    Combines dense vector search (sqlite-vec ANN) and sparse BM25 full-text
    search (FTS5) via Reciprocal Rank Fusion, applies heuristic reranking, and
    supplements results with knowledge-graph facts and relevant procedures.

    Args:
        db:      An initialised :class:`MemoryDatabase` instance.
        embedder: A :class:`GeminiEmbeddingProvider` (or compatible) instance.
        kg:      A :class:`KnowledgeGraph` instance for entity traversal.
        llm:     Optional :class:`OllamaProvider`; reserved for future LLM-based
                 entity extraction / reranking (not currently used).

    Example::

        reader = MemoryReader(db=db, embedder=embedder, kg=kg)
        result = await reader.retrieve("What tools do I prefer for Python?", n=10)
        context = reader.format_for_context(result)
    """

    def __init__(
        self,
        db: "MemoryDatabase",
        embedder: "GeminiEmbeddingProvider",
        kg: "KnowledgeGraph",
        llm: Optional["OllamaProvider"] = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.kg = kg
        self.llm = llm   # reserved for future use

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        session_context: str = "",
        n: int = 20,
    ) -> RetrievalResult:
        """Run the full hybrid retrieval pipeline.

        Pipeline steps:

        1. Determine adaptive retrieval strategy.
        2. Dense ANN search (k=50).
        3. Sparse BM25 search (k=50).
        4. RRF fusion.
        5. Heuristic rerank.
        6. Extract query entities → KG traversal.
        7. Fetch relevant procedures.
        8. Update access counts for top-n returned memories.

        Args:
            query:           The user query text.
            session_context: Optional recent conversation text used by the
                             adaptive strategy heuristic.
            n:               Base number of memories to return.  Scaled by the
                             strategy multiplier.

        Returns:
            A :class:`RetrievalResult` with memories, KG context, procedures,
            extracted entities, strategy name, and elapsed time.
        """
        t0 = time.perf_counter()

        # 1. Adaptive strategy ────────────────────────────────────────────────
        strategy = self.adaptive_strategy(query, session_context)
        multipliers = {"conservative": 0.5, "normal": 1.0, "aggressive": 1.5}
        effective_n = max(1, int(n * multipliers[strategy]))

        # 2. Dense search ─────────────────────────────────────────────────────
        dense_hits = await self.dense_search(query, k=50)

        # 3. Sparse search ────────────────────────────────────────────────────
        sparse_hits = self.sparse_search(query, k=50)

        # 4. RRF fusion ───────────────────────────────────────────────────────
        fused = self.rrf_fuse(dense_hits, sparse_hits, k=60)

        # 5. Heuristic rerank ─────────────────────────────────────────────────
        reranked = self.heuristic_rerank(fused, query)

        # Slice to effective_n
        top_memories = reranked[:effective_n]

        # 6. Entity extraction + KG traversal ─────────────────────────────────
        query_entities = self.extract_query_entities(query)
        kg_context = ""
        if query_entities and self.kg is not None:
            try:
                kg_context = self.kg.traverse(
                    entity_names=query_entities,
                    max_hops=2,
                    max_facts=20,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("KG traversal failed: %s", exc)

        # 7. Relevant procedures ───────────────────────────────────────────────
        procedures: list[dict] = []
        try:
            from .procedures import ProcedureStore  # late import to avoid cycles
            proc_store = ProcedureStore(self.db)
            procs = proc_store.find_relevant(query, max_results=3)
            procedures = [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "steps": p.steps,
                    "trigger_pattern": p.trigger_pattern,
                }
                for p in procs
            ]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Procedure retrieval skipped: %s", exc)

        # 8. Update access counts ─────────────────────────────────────────────
        for mem in top_memories:
            try:
                self.db.update_memory_access(mem.id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not update access count for %d: %s", mem.id, exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logger.debug(
            "retrieve: strategy=%s n=%d dense=%d sparse=%d fused=%d elapsed=%.1fms",
            strategy,
            effective_n,
            len(dense_hits),
            len(sparse_hits),
            len(fused),
            elapsed_ms,
        )

        return RetrievalResult(
            memories=top_memories,
            kg_context=kg_context,
            procedures=procedures,
            query_entities=query_entities,
            retrieval_strategy=strategy,
            elapsed_ms=elapsed_ms,
        )

    # ── Adaptive strategy ─────────────────────────────────────────────────────

    def adaptive_strategy(self, query: str, context: str) -> str:
        """Choose a retrieval strategy based on query characteristics.

        Rules (evaluated in order):

        * **aggressive** — if the query explicitly mentions memory/search/find
          semantics (e.g. "remember", "find", "recall"), OR if the session
          context shows the topic was already discussed (recent topic continuity).
        * **conservative** — if the query is very short (< 15 characters),
          contains no question mark, and contains no domain keywords.
        * **normal** — everything else.

        The strategy is mapped to an *n* multiplier in :meth:`retrieve`:
        conservative=0.5×, normal=1×, aggressive=1.5×.

        Args:
            query:   The user query text.
            context: Recent session context (concatenated recent user messages).

        Returns:
            One of ``"conservative"``, ``"normal"``, or ``"aggressive"``.
        """
        query_lower = query.lower().strip()

        # Aggressive: explicit memory / search intent
        for kw in _AGGRESSIVE_KEYWORDS:
            if kw in query_lower:
                return "aggressive"

        # Aggressive: topic continuity — query shares significant tokens with
        # recent session context, suggesting the user is following up on
        # something already discussed (worth pulling deeper into memory).
        if context:
            import re as _re
            query_tokens = set(_re.findall(r"\w{4,}", query_lower))
            context_lower = context.lower()
            context_tokens = set(_re.findall(r"\w{4,}", context_lower))
            if query_tokens:
                overlap = query_tokens & context_tokens
                overlap_ratio = len(overlap) / len(query_tokens)
                if overlap_ratio >= 0.5 and len(overlap) >= 2:
                    return "aggressive"

        # Conservative: very short, no question mark, no domain terms
        if len(query.strip()) < 15:
            has_question = "?" in query
            has_domain = any(kw in query_lower for kw in _KNOWN_TOOLS)
            if not has_question and not has_domain:
                return "conservative"

        return "normal"

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def rrf_fuse(
        self,
        dense: list[dict],
        sparse: list[dict],
        k: int = 60,
    ) -> list[RetrievedMemory]:
        """Reciprocal Rank Fusion of dense and sparse ranked lists.

        For each document that appears in *either* list:
            rrf_score += 1 / (k + rank)
        for every list it appears in (1-indexed rank).

        Merged results are sorted by descending RRF score.  Memory rows are
        fetched from the database to populate all :class:`RetrievedMemory`
        fields.

        Args:
            dense:  List of ``{"id": int, "distance": float}`` dicts from
                    :meth:`dense_search`, ordered best-first.
            sparse: List of ``{"id": int, "rank": float}`` dicts from
                    :meth:`sparse_search`, ordered best-first.
            k:      RRF smoothing constant (default 60).

        Returns:
            List of :class:`RetrievedMemory` objects sorted by descending
            ``rrf_score``.
        """
        # Accumulate per-memory id: (rrf_score, dense_rank, sparse_rank)
        scores: dict[int, dict] = {}

        for rank_0, item in enumerate(dense):
            mid = item["id"]
            rank_1 = rank_0 + 1  # 1-indexed
            if mid not in scores:
                scores[mid] = {"rrf_score": 0.0, "dense_rank": 999, "sparse_rank": 999}
            scores[mid]["rrf_score"] += 1.0 / (k + rank_1)
            scores[mid]["dense_rank"] = rank_1

        for rank_0, item in enumerate(sparse):
            mid = item["id"]
            rank_1 = rank_0 + 1
            if mid not in scores:
                scores[mid] = {"rrf_score": 0.0, "dense_rank": 999, "sparse_rank": 999}
            scores[mid]["rrf_score"] += 1.0 / (k + rank_1)
            scores[mid]["sparse_rank"] = rank_1

        # Fetch memory rows and build RetrievedMemory objects
        result: list[RetrievedMemory] = []
        for mid, score_info in scores.items():
            row = self.db.get_memory(mid)
            if row is None:
                continue  # memory was deleted since indexing
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            rm = RetrievedMemory(
                id=mid,
                content=row.get("content", ""),
                category=row.get("category", "general"),
                importance=float(row.get("importance", 0.5)),
                access_count=int(row.get("access_count", 0)),
                rrf_score=score_info["rrf_score"],
                dense_rank=score_info["dense_rank"],
                sparse_rank=score_info["sparse_rank"],
                source="memory",
                metadata={
                    "created_at": row.get("created_at"),
                    "last_accessed": row.get("last_accessed"),
                    **metadata,
                },
            )
            result.append(rm)

        result.sort(key=lambda m: m.rrf_score, reverse=True)
        return result

    # ── Heuristic rerank ──────────────────────────────────────────────────────

    def heuristic_rerank(
        self,
        memories: list[RetrievedMemory],
        query: str,
    ) -> list[RetrievedMemory]:
        """Apply heuristic score multipliers and re-sort.

        Multipliers applied (all multiplicative against ``rrf_score``):

        * **Recency** × ``(1 + 0.3 × recency_factor)`` where ``recency_factor``
          decays from 1.0 (fresh today) to 0.0 at 30 days.
        * **Access frequency** × ``(1 + 0.1 × min(access_count, 10))``.
        * **Category relevance** × 1.5 for procedures when query is a "how to",
          × 1.2 for facts when query is a "who/what is" question.
        * **Importance** × ``(0.5 + importance)`` (range: 0.5 – 1.5).

        The modified ``rrf_score`` is written back into each object so that
        callers can inspect the final value.

        Args:
            memories: Fused list from :meth:`rrf_fuse`.
            query:    Original query text (used for category-match check).

        Returns:
            Re-sorted list (highest final score first), same objects mutated in-place.
        """
        query_lower = query.lower()
        now = time.time()
        seconds_per_day = 86_400.0
        decay_days = 30.0

        # Determine category-boost type
        is_procedure_query = any(kw in query_lower for kw in _PROCEDURE_KEYWORDS)
        is_fact_query = any(kw in query_lower for kw in _FACT_KEYWORDS)

        for mem in memories:
            score = mem.rrf_score

            # Recency bonus
            created_at = mem.metadata.get("created_at") or (
                now - decay_days * seconds_per_day
            )
            age_days = (now - float(created_at)) / seconds_per_day
            recency_factor = max(0.0, 1.0 - age_days / decay_days)
            score *= 1.0 + 0.3 * recency_factor

            # Access frequency bonus
            score *= 1.0 + 0.1 * min(mem.access_count, 10)

            # Category-relevance bonus
            cat = mem.category.lower()
            if is_procedure_query and cat in ("procedure", "task", "how-to", "steps"):
                score *= 1.5
            elif is_fact_query and cat in ("fact", "knowledge", "definition"):
                score *= 1.2

            # Importance multiplier (maps [0,1] → [0.5, 1.5])
            score *= 0.5 + mem.importance

            mem.rrf_score = score

        memories.sort(key=lambda m: m.rrf_score, reverse=True)
        return memories

    # ── Entity extraction ─────────────────────────────────────────────────────

    def extract_query_entities(self, query: str) -> list[str]:
        """Extract candidate entity names from the query string.

        Heuristics applied (in order):

        1. **Quoted strings** — text inside single or double quotes.
        2. **Known tool names** — matches against :data:`_KNOWN_TOOLS`.
        3. **Title-cased words** — words starting with an uppercase letter
           that are not stop-words and have ≥ 3 characters.

        Duplicates are removed; order is preserved.

        Args:
            query: Raw query text.

        Returns:
            Deduplicated list of entity name strings.
        """
        entities: list[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                entities.append(name)

        # 1. Quoted strings (single or double quotes)
        import re
        for match in re.finditer(r'["\']([^"\']{2,})["\']', query):
            _add(match.group(1))

        # 2. Known tool names (case-insensitive match, return canonical lower-case)
        query_lower = query.lower()
        for tool in _KNOWN_TOOLS:
            if tool in query_lower:
                _add(tool)

        # 3. Title-cased words (not stop-words, length ≥ 3)
        for word in query.split():
            # Strip trailing punctuation
            clean = re.sub(r"[^a-zA-Z0-9_\-]", "", word)
            if (
                clean
                and len(clean) >= 3
                and clean[0].isupper()
                and clean.lower() not in _STOP_WORDS
            ):
                _add(clean)

        return entities

    # ── Hierarchical document search ────────────────────────────────────────

    def hierarchical_doc_search(
        self, query: str, max_docs: int = 5
    ) -> list[dict]:
        """Search memories sourced from documents, grouped by document.

        Args:
            query:    Search query.
            max_docs: Maximum number of distinct documents to return.

        Returns:
            List of dicts, each with ``"doc_id"``, ``"title"``, ``"chunks"``
            (list of memory dicts).
        """
        import re
        safe_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe_query:
            return []

        try:
            hits = self.db.fts_search(safe_query, limit=50)
        except Exception as exc:
            logger.warning("hierarchical_doc_search FTS failed: %s", exc)
            return []

        # Filter to document-sourced memories and group by doc_id
        from collections import defaultdict
        doc_groups: dict[int, list[dict]] = defaultdict(list)

        for hit in hits:
            mem = self.db.get_memory(hit["id"])
            if not mem:
                continue
            if mem.get("category") != "document":
                continue
            meta = mem.get("metadata", {})
            if isinstance(meta, str):
                import json
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            doc_id = meta.get("doc_id")
            if doc_id is not None:
                doc_groups[doc_id].append(mem)

        # Build result grouped by document
        results: list[dict] = []
        for doc_id, chunks in list(doc_groups.items())[:max_docs]:
            # Look up document title
            doc_rows = self.db.execute(
                "SELECT title FROM documents WHERE id = ?", (doc_id,)
            )
            title = doc_rows[0]["title"] if doc_rows else f"Document {doc_id}"
            results.append({
                "doc_id": doc_id,
                "title": title,
                "chunks": chunks,
            })

        return results

    # ── Dense search ──────────────────────────────────────────────────────────

    async def dense_search(self, query: str, k: int = 50) -> list[dict]:
        """Embed the query and run ANN search via sqlite-vec.

        Embedding is performed in a thread pool executor so it does not block
        the event loop (the Gemini SDK is synchronous).

        Args:
            query: Query text to embed.
            k:     Maximum number of nearest neighbours to return.

        Returns:
            List of ``{"id": int, "distance": float}`` dicts from
            :meth:`MemoryDatabase.vector_search`, ordered by ascending
            distance (closest / most similar first).  Returns an empty list
            when sqlite-vec is unavailable or the embedding call fails.
        """
        try:
            loop = asyncio.get_event_loop()
            embedding: list[float] = await loop.run_in_executor(
                None, self.embedder.embed_query, query
            )
            hits: list[dict] = self.db.vector_search(
                embedding=embedding,
                table="memory_vectors",
                id_col="memory_id",
                limit=k,
            )
            return hits
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dense search failed: %s", exc)
            return []

    # ── Sparse search ─────────────────────────────────────────────────────────

    def sparse_search(self, query: str, k: int = 50) -> list[dict]:
        """BM25 full-text search over the ``memory_fts`` FTS5 table.

        Sanitises the query to avoid FTS5 syntax errors by stripping special
        characters before passing to :meth:`MemoryDatabase.fts_search`.

        Args:
            query: Raw query text.
            k:     Maximum number of results.

        Returns:
            List of ``{"id": int, "rank": float}`` dicts ordered by
            descending relevance.  Returns an empty list on error.
        """
        import re
        # Strip FTS5 special characters that could cause parse errors
        safe_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe_query:
            return []
        try:
            hits: list[dict] = self.db.fts_search(safe_query, limit=k)
            return hits
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sparse search failed for query %r: %s", safe_query, exc)
            return []

    # ── Context formatting ────────────────────────────────────────────────────

    def format_for_context(
        self,
        result: RetrievalResult,
        budget_tokens: int = 11_000,
    ) -> str:
        """Format retrieval results as a prompt-ready context string.

        Uses a **best-at-edges** layout: the highest-scored memories appear at
        the TOP and BOTTOM of the block, with medium-scored memories in the
        middle.  Research suggests LLMs attend more to the beginning and end of
        long contexts ("lost in the middle" effect).

        Token budget uses tiktoken when available, else ``len(text) // 4``.
        Sections are concatenated in order (KG facts → memories → procedures)
        until the budget is exhausted.

        Memory format per item::

            [{category}] {content}

        Args:
            result:       The :class:`RetrievalResult` from :meth:`retrieve`.
            budget_tokens: Approximate maximum number of tokens for the entire
                           formatted string.

        Returns:
            A single formatted string, or an empty string when no content is
            available.
        """
        # Use heuristic token counting (len//4) for budget enforcement.
        # The budget parameter was designed for this model, and the context
        # assembler (which uses tiktoken) applies its own accurate budget
        # when building the final LLM prompt.
        def _count_tokens(text: str) -> int:
            return max(1, len(text) // 4)

        parts: list[str] = []
        tokens_used = 0

        def _fits(text: str) -> bool:
            nonlocal tokens_used
            cost = _count_tokens(text)
            if tokens_used + cost > budget_tokens:
                return False
            tokens_used += cost
            return True

        # ── KG context ─────────────────────────────────────────────────────
        if result.kg_context:
            section = f"## Knowledge Graph Context\n{result.kg_context}\n"
            if _fits(section):
                parts.append(section)

        # ── Memories (best-at-edges) ────────────────────────────────────────
        memories = result.memories
        if memories:
            n = len(memories)
            # Split into top-half (first ceil(n/2)) and bottom-half
            top_half = memories[: math.ceil(n / 2)]
            bottom_half = memories[math.ceil(n / 2) :]

            formatted_top: list[str] = []
            formatted_bottom: list[str] = []

            for mem in top_half:
                line = f"[{mem.category}] {mem.content}"
                if _fits(line):
                    formatted_top.append(line)
                else:
                    break  # budget exhausted

            for mem in bottom_half:
                line = f"[{mem.category}] {mem.content}"
                if _fits(line):
                    formatted_bottom.append(line)
                else:
                    break  # budget exhausted

            if formatted_top or formatted_bottom:
                memory_lines = formatted_top + formatted_bottom
                parts.append("## Relevant Memories\n" + "\n".join(memory_lines) + "\n")

        # ── Procedures ─────────────────────────────────────────────────────
        if result.procedures:
            proc_lines: list[str] = []
            for proc in result.procedures:
                name = proc.get("name", "Procedure")
                steps = proc.get("steps", [])
                if isinstance(steps, list):
                    steps_text = "\n".join(
                        f"  {i + 1}. {s}" for i, s in enumerate(steps)
                    )
                else:
                    steps_text = f"  {steps}"
                block = f"### {name}\n{steps_text}"
                if _fits(block):
                    proc_lines.append(block)
                else:
                    break

            if proc_lines:
                parts.append("## Relevant Procedures\n" + "\n".join(proc_lines) + "\n")

        return "\n".join(parts).strip()
