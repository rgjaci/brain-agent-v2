"""Brain Agent v2 — Memory Consolidation Engine.

Runs periodically in the background to keep the memory store healthy:

* **Near-duplicate merging** — collapses semantically identical memories
  (cosine similarity > 0.95) into a single canonical record.
* **Contradiction resolution** — detects same-category memories that conflict
  with each other (e.g. two "User prefers X" facts for the same topic) and
  keeps only the most recent one.
* **Importance decay** — gradually reduces the importance score of memories
  that have not been accessed for 30+ days.
* **Importance promotion** — rewards frequently-accessed memories with a
  higher importance score.

Consolidation is triggered by :meth:`ConsolidationEngine.maybe_consolidate`
once every :data:`CONSOLIDATION_INTERVAL` turns.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

CONSOLIDATION_INTERVAL: int = 10   # turns between automatic consolidation runs


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two equal-length float vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in ``[-1, 1]``; returns 0.0 if either vector is
        zero-length or the two vectors have different lengths.
    """
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ──────────────────────────────────────────────────────────────────────────────
# ConsolidationEngine
# ──────────────────────────────────────────────────────────────────────────────

class ConsolidationEngine:
    """Periodic background job that keeps the memory store clean and compact.

    Args:
        db:      Initialised :class:`~.database.MemoryDatabase`.
        llm:     LLM provider used for semantic contradiction detection
                 (currently unused — contradiction resolution uses heuristics).
        embedder: Embedding model used to retrieve stored vectors for duplicate
                  detection.
    """

    def __init__(self, db, llm, embedder) -> None:
        self.db = db
        self.llm = llm
        self.embedder = embedder

    # ── Scheduling ────────────────────────────────────────────────────────────

    async def maybe_consolidate(self, turn_count: int) -> None:
        """Run consolidation if the current turn is a multiple of the interval.

        Designed to be awaited at the end of every agent turn:

        .. code-block:: python

            await engine.maybe_consolidate(turn_count)

        Args:
            turn_count: The 1-based index of the current conversation turn.
        """
        if turn_count > 0 and turn_count % CONSOLIDATION_INTERVAL == 0:
            logger.info(
                "maybe_consolidate: turn %d — triggering consolidation.", turn_count
            )
            await self.consolidate()
        else:
            logger.debug(
                "maybe_consolidate: turn %d — skipping (next at turn %d).",
                turn_count,
                CONSOLIDATION_INTERVAL * (turn_count // CONSOLIDATION_INTERVAL + 1),
            )

    # ── Full consolidation pass ───────────────────────────────────────────────

    async def consolidate(self) -> None:
        """Run all consolidation steps sequentially.

        Steps:

        1. :meth:`merge_near_duplicates` — vector-similarity deduplication
        2. :meth:`resolve_contradictions` — heuristic contradiction resolution
        3. :meth:`apply_decay` — importance decay for old memories
        4. :meth:`promote_important` — importance boost for popular memories
        """
        logger.info("consolidate: starting consolidation pass.")
        start = time.time()

        try:
            merges = await self.merge_near_duplicates()
            logger.info("consolidate: merged %d near-duplicate pairs.", merges)
        except Exception as exc:
            logger.warning("consolidate: merge_near_duplicates failed — %s", exc)
            merges = 0

        try:
            resolutions = await self.resolve_contradictions()
            logger.info(
                "consolidate: resolved %d contradictions.", resolutions
            )
        except Exception as exc:
            logger.warning("consolidate: resolve_contradictions failed — %s", exc)
            resolutions = 0

        try:
            decayed = self.apply_decay()
            logger.info("consolidate: decayed %d stale memories.", decayed)
        except Exception as exc:
            logger.warning("consolidate: apply_decay failed — %s", exc)
            decayed = 0

        try:
            promoted = self.promote_important()
            logger.info("consolidate: promoted %d frequently-accessed memories.", promoted)
        except Exception as exc:
            logger.warning("consolidate: promote_important failed — %s", exc)
            promoted = 0

        # 5. Conversation pruning
        try:
            pruned = self.db.prune_conversations(keep_last=100)
            logger.info("consolidate: pruned %d old conversation rows.", pruned)
        except Exception as exc:
            logger.warning("consolidate: prune_conversations failed — %s", exc)

        # 6. Conversation summarization (sessions older than 7 days)
        try:
            summaries = await self.summarize_old_sessions()
            logger.info("consolidate: summarized %d old sessions.", summaries)
        except Exception as exc:
            logger.warning("consolidate: summarize_old_sessions failed — %s", exc)

        elapsed = time.time() - start
        logger.info(
            "consolidate: pass complete in %.2fs — "
            "merges=%d, resolutions=%d, decayed=%d, promoted=%d, summaries=%d.",
            elapsed, merges, resolutions, decayed, promoted, summaries if 'summaries' in dir() else 0,
        )

    # ── Step 1 — Near-duplicate merging ──────────────────────────────────────

    async def merge_near_duplicates(self) -> int:
        """Detect and merge near-duplicate memories using vector cosine similarity.

        Algorithm:

        1. Fetch all stored embeddings from the ``memory_vectors`` virtual table.
        2. Compare every pair of embeddings (batched to avoid loading all at once).
        3. For pairs with cosine similarity > 0.95:

           * Keep the memory with the higher ``access_count`` (falling back to
             the higher ``id`` as a tie-breaker).
           * Mark the other memory as superseded via
             :meth:`~.database.MemoryDatabase.mark_superseded`.
           * Delete the superseded memory's embedding and memory row.

        Returns:
            Number of duplicate pairs merged.

        Note:
            Runs in a thread executor to avoid blocking the event loop during
            the O(N²) similarity computation.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._merge_near_duplicates_sync)

    def _merge_near_duplicates_sync(self) -> int:
        """Synchronous implementation of near-duplicate merging.

        Uses ANN vector search (sqlite-vec) per embedding instead of
        O(N²) pairwise comparison.  For each embedding, its 10 nearest
        neighbours are fetched; pairs with cosine similarity > 0.95 are
        merged.
        """
        import math as _math
        import struct as _struct

        def _cosine_sim(a: list[float], b: list[float]) -> float:
            if len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = _math.sqrt(sum(x * x for x in a))
            nb = _math.sqrt(sum(x * x for x in b))
            if na == 0.0 or nb == 0.0:
                return 0.0
            return dot / (na * nb)

        # Fetch all embeddings
        embeddings = self.db.get_all_embeddings(
            table="memory_vectors", id_col="memory_id", limit=5000
        )
        if len(embeddings) < 2:
            return 0

        merge_count = 0
        deleted_ids: set[int] = set()

        for mem_id, emb in embeddings:
            if mem_id in deleted_ids:
                continue

            # ANN search: find 10 nearest neighbours
            try:
                neighbours = self.db.vector_search(
                    embedding=emb,
                    table="memory_vectors",
                    id_col="memory_id",
                    limit=10,
                )
            except Exception:
                continue

            for neighbour in neighbours:
                nid = neighbour["id"]
                if nid == mem_id or nid in deleted_ids:
                    continue

                # Fetch neighbour embedding for proper cosine similarity
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
                    sim = _cosine_sim(emb, neighbor_emb)
                except Exception:
                    continue

                if sim > 0.95:
                    # Decide which memory to keep
                    mem_a = self.db.get_memory(mem_id)
                    mem_b = self.db.get_memory(nid)

                    if mem_a is None or mem_b is None:
                        continue

                    acc_a = int(mem_a.get("access_count", 0))
                    acc_b = int(mem_b.get("access_count", 0))

                    if acc_a >= acc_b:
                        keep_id, drop_id = mem_id, nid
                    else:
                        keep_id, drop_id = nid, mem_id

                    # Mark as superseded and delete the duplicate
                    try:
                        self.db.mark_superseded(drop_id, keep_id)
                        self.db.execute(
                            "DELETE FROM memories WHERE id = ?",
                            (drop_id,),
                        )
                        deleted_ids.add(drop_id)
                        merge_count += 1
                        logger.debug(
                            "merge_near_duplicates: merged %d → %d (sim=%.4f).",
                            drop_id, keep_id, sim,
                        )
                    except Exception as exc:
                        logger.warning(
                            "merge_near_duplicates: could not merge "
                            "%d → %d — %s.",
                            drop_id, keep_id, exc,
                        )

        return merge_count

    # ── Step 2 — Contradiction resolution ────────────────────────────────────

    async def resolve_contradictions(self) -> int:
        """Detect and resolve contradictory memories within the same category.

        Heuristic: two memories in the same category that both start with
        ``"User prefers"`` followed by the same key phrase are contradictory —
        the newer one is kept and the older is deleted.

        The method groups candidate memories by category and looks for pairs
        whose content has the same leading key token (the word immediately after
        ``"User prefers "``).

        Returns:
            Number of contradictions resolved.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._resolve_contradictions_sync)

    def _resolve_contradictions_sync(self) -> int:
        """Synchronous implementation of contradiction resolution."""
        # Fetch candidate memories (those starting with preference/correction phrases)
        rows = self.db.execute(
            """
            SELECT id, content, category, created_at
              FROM memories
             WHERE superseded_by IS NULL
               AND (
                     content LIKE 'User prefers%'
                  OR content LIKE 'User likes%'
                  OR content LIKE 'User dislikes%'
                  OR content LIKE 'User wants%'
               )
             ORDER BY category, created_at DESC
            """,
        )

        if not rows:
            return 0

        # Group by (category, topic_key)
        # topic_key = first two words of content after the leading phrase
        from collections import defaultdict
        groups: dict[tuple, list[dict]] = defaultdict(list)

        for row in rows:
            content = str(row.get("content", ""))
            category = str(row.get("category", ""))
            words = content.split()
            # Use first 3 words as the "topic key" (e.g. "User prefers dark")
            topic_key = " ".join(words[:3]).lower() if len(words) >= 3 else content.lower()
            groups[(category, topic_key)].append(row)

        resolution_count = 0

        for (_category, topic_key), group in groups.items():
            if len(group) <= 1:
                continue

            # Sort newest first (already sorted by created_at DESC above, but
            # re-sort here to be safe since defaultdict merges groups)
            group_sorted = sorted(
                group,
                key=lambda r: float(r.get("created_at", 0)),
                reverse=True,
            )

            # Keep the newest, delete the rest
            keep = group_sorted[0]
            for duplicate in group_sorted[1:]:
                try:
                    self.db.mark_superseded(
                        int(duplicate["id"]), int(keep["id"])
                    )
                    self.db.execute(
                        "DELETE FROM memories WHERE id = ?",
                        (int(duplicate["id"]),),
                    )
                    resolution_count += 1
                    logger.debug(
                        "resolve_contradictions: deleted contradicting memory "
                        "%d, kept %d (topic=%r).",
                        duplicate["id"], keep["id"], topic_key,
                    )
                except Exception as exc:
                    logger.warning(
                        "resolve_contradictions: could not delete memory %d — %s.",
                        duplicate["id"], exc,
                    )

        return resolution_count

    # ── Step 3 — Importance decay ─────────────────────────────────────────────

    def apply_decay(self) -> int:
        """Reduce importance of memories not accessed in the last 30 days.

        Executes the following SQL update:

        .. code-block:: sql

            UPDATE memories
               SET importance    = MAX(0.1, importance - 0.05),
                   updated_at    = CURRENT_TIMESTAMP
             WHERE last_accessed < datetime('now', '-30 days')
               AND importance    > 0.1

        Returns:
            Number of memory rows whose importance was reduced.
        """
        try:
            self.db.execute(
                """
                UPDATE memories
                   SET importance  = MAX(0.1, importance - 0.05)
                 WHERE last_accessed < (strftime('%s','now') - 2592000)
                   AND importance    > 0.1
                """,
            )
            # SQLite doesn't return rowcount through execute() easily;
            # query the change count via a separate approach.
            count_rows = self.db.execute("SELECT changes() AS n")
            count = int(count_rows[0]["n"]) if count_rows else 0
            return count
        except Exception as exc:
            logger.warning("apply_decay: SQL failed — %s", exc)
            return 0

    # ── Step 4 — Importance promotion ────────────────────────────────────────

    def promote_important(self) -> int:
        """Boost importance of memories that have been accessed 10+ times.

        Executes the following SQL update:

        .. code-block:: sql

            UPDATE memories
               SET importance = MIN(1.0, importance + 0.1)
             WHERE access_count >= 10
               AND importance   < 1.0

        Returns:
            Number of memory rows whose importance was increased.
        """
        try:
            self.db.execute(
                """
                UPDATE memories
                   SET importance = MIN(1.0, importance + 0.1)
                 WHERE access_count >= 10
                   AND importance   < 1.0
                """,
            )
            count_rows = self.db.execute("SELECT changes() AS n")
            count = int(count_rows[0]["n"]) if count_rows else 0
            return count
        except Exception as exc:
            logger.warning("promote_important: SQL failed — %s", exc)
            return 0

    # ── Step 6 — Conversation summarization ──────────────────────────────────

    async def summarize_old_sessions(self, max_age_days: int = 7) -> int:
        """Summarize conversation sessions older than *max_age_days*.

        For each old session:
        1. Fetch all messages in the session.
        2. Use the LLM to generate a concise summary.
        3. Store the summary as a ``conversation_summary`` memory.
        4. Delete the raw conversation rows to free space.

        Args:
            max_age_days: Minimum age of sessions to summarize.

        Returns:
            Number of sessions summarized.
        """
        if self.llm is None:
            logger.debug("summarize_old_sessions: no LLM available, skipping.")
            return 0

        cutoff = time.time() - (max_age_days * 86400)

        # Find old session IDs
        try:
            sessions = self.db.execute(
                """
                SELECT DISTINCT session_id, MIN(created_at) as first_msg
                  FROM conversations
                 WHERE created_at < ?
                 GROUP BY session_id
                 ORDER BY first_msg ASC
                 LIMIT 10
                """,
                (cutoff,),
            )
        except Exception as exc:
            logger.warning("summarize_old_sessions: query failed — %s", exc)
            return 0

        if not sessions:
            logger.debug("summarize_old_sessions: no old sessions to summarize.")
            return 0

        summary_count = 0
        for session_row in sessions:
            session_id = session_row["session_id"]
            try:
                # Fetch all messages for this session
                messages = self.db.execute(
                    """
                    SELECT role, content FROM conversations
                     WHERE session_id = ?
                     ORDER BY created_at ASC
                    """,
                    (session_id,),
                )

                if not messages:
                    continue

                # Build conversation text for summarization
                conv_text = "\n".join(
                    f"{m['role']}: {m['content'][:200]}" for m in messages[:50]
                )

                # Generate summary
                summary = await self._generate_session_summary(conv_text)
                if summary:
                    # Store summary as a memory
                    self.db.insert_memory(
                        content=f"[Session Summary] {summary}",
                        category="conversation_summary",
                        source=f"session:{session_id}",
                        importance=0.4,
                        confidence=0.8,
                    )
                    summary_count += 1
                    logger.debug(
                        "summarize_old_sessions: summarized session %s",
                        session_id,
                    )

                # Delete raw conversation rows
                self.db.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,),
                )

            except Exception as exc:
                logger.warning(
                    "summarize_old_sessions: failed for session %s — %s",
                    session_id, exc,
                )

        return summary_count

    async def _generate_session_summary(self, conversation: str) -> str:
        """Use the LLM to generate a concise session summary.

        Args:
            conversation: Full conversation text (truncated).

        Returns:
            Summary string, or empty string on failure.
        """
        prompt = (
            "Summarize this conversation in 2-3 sentences. "
            "Focus on: decisions made, facts learned, preferences expressed, "
            "and unresolved questions. Be concise.\n\n"
            f"Conversation:\n{conversation[:4000]}"
        )

        try:
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(  # type: ignore[union-attr]
                    [{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                ),
            )
            return summary.strip()
        except Exception as exc:
            logger.warning("_generate_session_summary failed: %s", exc)
            return ""
