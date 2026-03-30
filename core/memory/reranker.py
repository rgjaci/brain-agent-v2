"""Brain Agent v2 — Memory Reranker.

Three-stage reranker pipeline:

1. **Heuristic stage** (always active) — applies hand-crafted score multipliers
   based on recency, access frequency, importance, and category bonuses.
2. **Logistic-regression stage** (when weights are available) — blends the
   heuristic score with a trained LR probability to produce a final ranking.
3. **Cross-encoder stage** (reserved for future integration with a neural
   cross-encoder model).

After scoring, :meth:`apply_best_at_edges` reorders memories so that the
highest-scoring items appear at both the top *and* the bottom of the context
window — exploiting the empirical observation that LLMs attend more strongly to
the extremes of long prompts (the "lost in the middle" effect).
"""
from __future__ import annotations

import json
import logging
import math
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .reader import RetrievedMemory
    from .feedback import RetrievalFeedbackCollector

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────────────────────────────────────

class Reranker:
    """Score and reorder retrieved memories for optimal LLM context placement.

    Args:
        feedback_collector: Optional :class:`~.feedback.RetrievalFeedbackCollector`
                            used to score memories with learned LR weights.
                            When ``None`` only heuristic scoring is performed.
    """

    def __init__(self, feedback_collector: Optional[object] = None) -> None:
        self.feedback_collector = feedback_collector

    # ── Main reranking entry-point ────────────────────────────────────────────

    def rerank(
        self,
        memories: list,
        query: str,
        weights: Optional[dict] = None,
    ) -> list:
        """Score and sort memories using heuristic multipliers and optional LR.

        **Stage 1 — Heuristic scoring** (always applied):

        Starting from the memory's existing ``rrf_score`` (or 1.0 as a
        baseline), the following multipliers are applied in order:

        * **Recency** — ``× (1 + 0.3 × exp(−age_days / 30))``
          Decays smoothly from ×1.3 (brand-new) to ×1.0 (old).
        * **Access frequency** — ``× (1 + 0.1 × min(access_count, 10))``
          Up to an extra ×2.0 for heavily-accessed memories.
        * **Importance weight** — ``× (0.5 + importance)``
          Ranges from ×0.5 (importance=0) to ×1.5 (importance=1).
        * **Correction bonus** — ``× 1.5`` if ``category == "correction"``
        * **Preference bonus** — ``× 1.3`` if ``category == "preference"``

        **Stage 2 — LR blending** (only when *weights* are provided):

        ``final_score = 0.6 × heuristic_score + 0.4 × lr_score``

        where ``lr_score`` is the sigmoid output of the stored logistic model.

        Args:
            memories: List of memory dicts (each must have at least ``"id"``).
            query:    The query string (used for LR feature extraction).
            weights:  Optional weight dict from
                      :meth:`~.feedback.RetrievalFeedbackCollector.load_weights`.
                      When ``None``, only the heuristic stage runs.

        Returns:
            The same memory dicts, annotated with a ``"_rerank_score"`` key and
            sorted in descending order of that score.
        """
        if not memories:
            return memories

        scored: list[tuple[float, dict]] = []

        for rank, memory in enumerate(memories):
            heuristic_score = self._heuristic_score(memory, rank)

            if weights is not None and self.feedback_collector is not None:
                lr_score = self._lr_score(memory, query, rank, weights)
                final_score = 0.6 * heuristic_score + 0.4 * lr_score
            else:
                final_score = heuristic_score

            memory = dict(memory)   # shallow copy — don't mutate caller's list
            memory["_rerank_score"] = final_score
            scored.append((final_score, memory))

        scored.sort(key=lambda t: t[0], reverse=True)
        result = [m for _, m in scored]

        logger.debug(
            "rerank: %d memories scored (LR=%s).",
            len(result), weights is not None,
        )
        return result

    # ── Stage 1 — Heuristic scoring ───────────────────────────────────────────

    def _heuristic_score(self, memory: dict, rank: int) -> float:
        """Compute a heuristic score for a single memory.

        Args:
            memory: Memory dict.
            rank:   Zero-based position in the original retrieval ranking
                    (used only for potential future extensions).

        Returns:
            Non-negative float score.
        """
        base = float(memory.get("rrf_score", 1.0))
        if base <= 0.0:
            base = 1.0

        age_days = self.compute_age_days(memory.get("created_at"))
        access_count = int(memory.get("access_count", 0))
        importance = float(memory.get("importance", 0.5))
        category = str(memory.get("category", "")).lower()

        # Recency multiplier
        recency_mult = 1.0 + 0.3 * math.exp(-age_days / 30.0)

        # Access-frequency multiplier
        freq_mult = 1.0 + 0.1 * min(access_count, 10)

        # Importance multiplier
        importance_mult = 0.5 + importance

        # Category bonuses
        category_mult = 1.0
        if category == "correction":
            category_mult = 1.5
        elif category == "preference":
            category_mult = 1.3

        score = base * recency_mult * freq_mult * importance_mult * category_mult
        return score

    # ── Stage 2 — Logistic-regression scoring ─────────────────────────────────

    def _lr_score(
        self, memory: dict, query: str, rank: int, weights: dict
    ) -> float:
        """Score a memory with the stored logistic-regression model.

        Args:
            memory:  Memory dict.
            query:   Retrieval query string.
            rank:    Zero-based retrieval rank.
            weights: Dict with ``"weights"`` (feature→coef) and ``"bias"`` keys.

        Returns:
            Probability in ``[0, 1]``.
        """
        if self.feedback_collector is None:
            return 0.5

        try:
            features = self.feedback_collector.extract_features(memory, query, rank)
            w = weights.get("weights", {})
            b = float(weights.get("bias", 0.0))
            return self.feedback_collector.score_with_learned_weights(features, w, b)
        except Exception as exc:  # noqa: BLE001
            logger.warning("_lr_score failed: %s", exc)
            return 0.5

    # ── Best-at-edges reordering ──────────────────────────────────────────────

    def apply_best_at_edges(self, memories: list) -> list:
        """Reorder memories so the highest-scoring items sit at both edges.

        Addresses the "lost in the middle" problem where LLMs pay less
        attention to context items in the middle of a long list.

        Algorithm:

        1. Sort memories by ``"_rerank_score"`` descending (best → worst).
        2. Interleave them into the output list, alternating between the
           front and back:

           * rank-1 → position 0  (front)
           * rank-2 → position −1 (back)
           * rank-3 → position 1  (front+1)
           * rank-4 → position −2 (back−1)
           * … and so on

        Args:
            memories: List of memory dicts (with ``"_rerank_score"`` set).

        Returns:
            Reordered list with best memories at the edges.
        """
        if len(memories) <= 2:
            return memories

        sorted_mems = sorted(
            memories,
            key=lambda m: float(m.get("_rerank_score", 0.0)),
            reverse=True,
        )

        output: list = [None] * len(sorted_mems)
        front = 0
        back = len(output) - 1
        use_front = True

        for memory in sorted_mems:
            if use_front:
                output[front] = memory
                front += 1
            else:
                output[back] = memory
                back -= 1
            use_front = not use_front

        return output

    # ── Stage 3 — Cross-encoder reranking ────────────────────────────────────

    def cross_encoder_rerank(
        self, query: str, candidates: list, top_k: int = 10
    ) -> list:
        """Rerank candidates using a cross-encoder model.

        Tries to import ``sentence_transformers`` and use a cross-encoder.
        Falls back to returning ``candidates[:top_k]`` if not installed.

        Args:
            query:      The query string.
            candidates: List of memory dicts (must have ``"content"`` key).
            top_k:      Number of results to return.

        Returns:
            Reranked list of memory dicts, limited to *top_k*.
        """
        if not candidates:
            return []

        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [
                (query, str(c.get("content", "")))
                for c in candidates
            ]
            scores = model.predict(pairs)
            scored = list(zip(scores, candidates))
            scored.sort(key=lambda x: float(x[0]), reverse=True)
            return [c for _, c in scored[:top_k]]
        except ImportError:
            logger.debug(
                "cross_encoder_rerank: sentence-transformers not installed — "
                "returning top-k by existing order."
            )
            return candidates[:top_k]
        except Exception as exc:
            logger.warning("cross_encoder_rerank failed: %s", exc)
            return candidates[:top_k]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def compute_age_days(self, created_at) -> float:
        """Compute the age of a memory in days from its ``created_at`` value.

        Handles both Unix-timestamp floats/ints (as stored by
        :class:`~.database.MemoryDatabase`) and ISO-8601 date strings for
        forward-compatibility.

        Args:
            created_at: Unix timestamp (float or int), ISO-8601 string, or
                        ``None``.

        Returns:
            Age in fractional days. Returns 0.0 for unrecognised formats or
            future timestamps.
        """
        now = time.time()

        if created_at is None:
            return 0.0

        if isinstance(created_at, (int, float)):
            age_seconds = now - float(created_at)
            return max(0.0, age_seconds / 86400.0)

        if isinstance(created_at, str):
            # Try parsing common ISO-8601 variants
            import datetime

            formats = [
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    dt = datetime.datetime.strptime(created_at, fmt)
                    # Assume UTC if no timezone info
                    ts = dt.replace(
                        tzinfo=datetime.timezone.utc
                    ).timestamp()
                    return max(0.0, (now - ts) / 86400.0)
                except ValueError:
                    continue

            logger.debug(
                "compute_age_days: unrecognised created_at format %r — "
                "defaulting to 0 days.",
                created_at,
            )

        return 0.0
