"""Brain Agent v2 — Retrieval Feedback Collector.

Collects implicit feedback from subsequent conversation turns and trains a
pure-Python logistic-regression reranker that can be persisted to and loaded
from the MemoryDatabase config table.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalEvent:
    """Represents a single memory retrieval event within a session."""

    memory_id: int
    query: str
    rank: int
    rrf_score: float
    features: dict = field(default_factory=dict)
    was_referenced: bool = False   # filled in after the agent turn
    timestamp: float = field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────────────────────
# RetrievalFeedbackCollector
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalFeedbackCollector:
    """Collects implicit retrieval feedback and trains a logistic-regression reranker.

    Feedback is collected in two phases per agent turn:

    1. :meth:`record_retrieval` — called immediately after retrieval; stores
       feature vectors for every memory that was surfaced.
    2. :meth:`record_reference` — called after the LLM response is generated;
       marks which memory IDs were actually referenced in the answer.

    After enough data accumulates, :meth:`train_logistic_regression` fits a
    lightweight binary classifier (referenced = 1, ignored = 0) and
    :meth:`persist_weights` / :meth:`load_weights` round-trip the coefficients
    through the database ``config`` table.

    Args:
        db: An initialised :class:`~.database.MemoryDatabase` instance.
    """

    # Config-table key used to persist the trained weights.
    _WEIGHTS_KEY = "reranker_lr_weights"

    def __init__(self, db) -> None:
        self.db = db
        # Maps session_id → list[RetrievalEvent]
        self._buffer: dict[str, list[RetrievalEvent]] = {}

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_retrieval(
        self,
        query: str,
        retrieved_memories: list,
        session_id: str,
    ) -> None:
        """Store retrieval events for a query within a session.

        For every memory in *retrieved_memories* a :class:`RetrievalEvent` is
        created (including pre-computed features) and appended to the in-memory
        buffer keyed by *session_id*.

        Args:
            query:              The natural-language query that triggered retrieval.
            retrieved_memories: Ordered list of memory dicts as returned by the
                                reader (must include ``id``, ``rrf_score``, and
                                the fields used by :meth:`extract_features`).
            session_id:         Conversation session identifier.
        """
        if session_id not in self._buffer:
            self._buffer[session_id] = []

        for rank, memory in enumerate(retrieved_memories):
            memory_id = memory.get("id")
            if memory_id is None:
                continue

            rrf_score = float(memory.get("rrf_score", 0.0))
            features = self.extract_features(memory, query, rank)

            event = RetrievalEvent(
                memory_id=memory_id,
                query=query,
                rank=rank,
                rrf_score=rrf_score,
                features=features,
            )
            self._buffer[session_id].append(event)

        logger.debug(
            "record_retrieval: session=%s, query=%r, %d memories buffered.",
            session_id, query, len(retrieved_memories),
        )

    def record_reference(
        self,
        session_id: str,
        memory_ids_referenced: list[int],
    ) -> None:
        """Mark memories as referenced in the most recent agent response.

        Iterates over the event buffer for *session_id* and sets
        ``was_referenced = True`` for every event whose ``memory_id`` appears
        in *memory_ids_referenced*.

        Args:
            session_id:           Conversation session identifier.
            memory_ids_referenced: IDs of memories that were actually cited or
                                   used in the agent's answer.
        """
        referenced_set = set(memory_ids_referenced)
        events = self._buffer.get(session_id, [])
        count = 0
        for event in events:
            if event.memory_id in referenced_set:
                event.was_referenced = True
                count += 1

        logger.debug(
            "record_reference: session=%s, %d/%d events marked as referenced.",
            session_id, count, len(events),
        )

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_features(
        self, memory: dict, query: str, rank: int
    ) -> dict:
        """Build a feature vector for a single retrieval event.

        All values are normalised to roughly ``[0, 1]`` so that logistic
        regression converges reliably without a separate scaling step.

        Features:

        * ``rank_normalized``  — retrieval rank divided by 20 (capped at 1.0)
        * ``rrf_score``        — raw RRF score (already in ``[0, 1]``)
        * ``access_count_log`` — ``log1p(access_count)`` / 5  (soft-caps at ~148
          accesses mapping to 1.0)
        * ``importance``       — stored importance value in ``[0, 1]``
        * ``age_days``         — ``(now − created_at) / 86400``; negative ages
          are clamped to 0
        * ``category_match``   — 1.0 if the memory's category appears as a word
          in the query, else 0.0
        * ``query_len``        — number of query tokens divided by 50

        Args:
            memory: Memory dict (may include any subset of the fields above).
            query:  The retrieval query string.
            rank:   Zero-based retrieval rank.

        Returns:
            Dict mapping feature name → float value.
        """
        now = time.time()

        # rank_normalized
        rank_normalized = min(rank / 20.0, 1.0)

        # rrf_score — already in [0,1]
        rrf_score = float(memory.get("rrf_score", 0.0))

        # access_count_log
        access_count = int(memory.get("access_count", 0))
        access_count_log = math.log1p(access_count) / 5.0

        # importance
        importance = float(memory.get("importance", 0.5))

        # age_days
        created_at = memory.get("created_at", now)
        if isinstance(created_at, (int, float)):
            age_seconds = now - float(created_at)
        else:
            # Fallback: assume already 0 days old
            age_seconds = 0.0
        age_days = max(0.0, age_seconds / 86400.0)

        # category_match
        category = str(memory.get("category", "")).lower()
        query_words = set(query.lower().split())
        category_match = 1.0 if category and category in query_words else 0.0

        # query_len
        query_len = len(query.split()) / 50.0

        return {
            "rank_normalized": rank_normalized,
            "rrf_score": rrf_score,
            "access_count_log": access_count_log,
            "importance": importance,
            "age_days": age_days,
            "category_match": category_match,
            "query_len": query_len,
        }

    # ── Training data ─────────────────────────────────────────────────────────

    def get_training_data(self) -> tuple[list[list[float]], list[int]]:
        """Collect all buffered events as a (X, y) training dataset.

        Flushes the entire event buffer after collection so that subsequent
        training calls start from a clean state.

        Returns:
            A tuple ``(X, y)`` where ``X`` is a list of feature-value lists
            and ``y`` is the corresponding list of binary labels
            (``1`` if the memory was referenced, ``0`` otherwise).
        """
        X: list[list[float]] = []
        y: list[int] = []

        # Canonical feature order — must remain consistent across calls.
        feature_keys = [
            "rank_normalized",
            "rrf_score",
            "access_count_log",
            "importance",
            "age_days",
            "category_match",
            "query_len",
        ]

        for session_events in self._buffer.values():
            for event in session_events:
                feature_vec = [event.features.get(k, 0.0) for k in feature_keys]
                X.append(feature_vec)
                y.append(1 if event.was_referenced else 0)

        # Flush buffer
        self._buffer.clear()

        logger.debug(
            "get_training_data: %d samples collected (%d positive).",
            len(y), sum(y),
        )
        return X, y

    # ── Logistic regression ───────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable sigmoid function."""
        return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))

    def train_logistic_regression(self) -> Optional[dict]:
        """Fit a logistic-regression model on buffered feedback data.

        Uses full-batch gradient descent for up to 100 epochs with a fixed
        learning rate of 0.01.  Returns ``None`` if fewer than 10 samples are
        available (insufficient data for meaningful training).

        The returned dict contains:

        * ``"weights"`` — ``dict[str, float]`` mapping feature name to coefficient
        * ``"bias"``    — scalar bias term (float)
        * ``"loss"``    — final cross-entropy loss (float)
        * ``"n_samples"`` — number of training samples used

        Returns:
            Trained weight dict, or ``None`` if training was skipped.
        """
        X, y = self.get_training_data()

        if len(y) < 10:
            logger.info(
                "train_logistic_regression: only %d samples — need at least 10. "
                "Skipping training.",
                len(y),
            )
            return None

        feature_keys = [
            "rank_normalized",
            "rrf_score",
            "access_count_log",
            "importance",
            "age_days",
            "category_match",
            "query_len",
        ]

        n_features = len(feature_keys)
        n_samples = len(y)

        # Initialise weights and bias to zero
        weights = [0.0] * n_features
        bias = 0.0
        lr = 0.01
        max_epochs = 100

        final_loss = float("inf")

        for epoch in range(max_epochs):
            # Forward pass — compute predictions and loss
            predictions = []
            loss = 0.0
            for i in range(n_samples):
                z = bias + sum(weights[j] * X[i][j] for j in range(n_features))
                p = self._sigmoid(z)
                # Clip to avoid log(0)
                p_clipped = max(1e-15, min(1.0 - 1e-15, p))
                predictions.append(p)
                if y[i] == 1:
                    loss -= math.log(p_clipped)
                else:
                    loss -= math.log(1.0 - p_clipped)

            loss /= n_samples
            final_loss = loss

            # Backward pass — gradient descent
            grad_w = [0.0] * n_features
            grad_b = 0.0

            for i in range(n_samples):
                error = predictions[i] - y[i]
                grad_b += error
                for j in range(n_features):
                    grad_w[j] += error * X[i][j]

            grad_b /= n_samples
            for j in range(n_features):
                grad_w[j] /= n_samples
                weights[j] -= lr * grad_w[j]

            bias -= lr * grad_b

            if epoch % 20 == 0:
                logger.debug(
                    "LR training epoch %d/%d — loss=%.4f",
                    epoch, max_epochs, loss,
                )

        weight_dict = {feature_keys[j]: weights[j] for j in range(n_features)}

        result = {
            "weights": weight_dict,
            "bias": bias,
            "loss": final_loss,
            "n_samples": n_samples,
        }

        logger.info(
            "train_logistic_regression: trained on %d samples, final loss=%.4f.",
            n_samples, final_loss,
        )
        return result

    def score_with_learned_weights(
        self, features: dict, weights: dict, bias: float
    ) -> float:
        """Score a feature dict using pre-trained logistic-regression weights.

        Args:
            features: Feature dict as produced by :meth:`extract_features`.
            weights:  Feature-name → coefficient mapping from
                      :meth:`train_logistic_regression`.
            bias:     Scalar bias term.

        Returns:
            Probability in ``[0, 1]`` that the memory will be referenced.
        """
        z = bias + sum(weights.get(k, 0.0) * v for k, v in features.items())
        return self._sigmoid(z)

    # ── Persistence ───────────────────────────────────────────────────────────

    def persist_weights(self, weights: dict) -> None:
        """Save trained weights to the database ``config`` table.

        Creates the ``config`` table if it does not yet exist, then performs an
        ``INSERT OR REPLACE`` so that re-training always overwrites the previous
        checkpoint.

        Args:
            weights: Dict as returned by :meth:`train_logistic_regression`
                     (contains ``"weights"``, ``"bias"``, etc.).
        """
        # Ensure the config table exists
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
        )

        payload = json.dumps(weights)
        self.db.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (self._WEIGHTS_KEY, payload),
        )
        logger.info(
            "persist_weights: saved reranker weights to config table "
            "(n_features=%d).",
            len(weights.get("weights", {})),
        )

    def load_weights(self) -> Optional[dict]:
        """Load previously saved weights from the database ``config`` table.

        Returns:
            Weight dict (same structure as :meth:`train_logistic_regression`
            output) or ``None`` if no weights have been persisted yet.
        """
        try:
            rows = self.db.execute(
                "SELECT value FROM config WHERE key = ?",
                (self._WEIGHTS_KEY,),
            )
        except Exception:
            # Config table may not exist yet
            return None

        if not rows:
            return None

        try:
            return json.loads(rows[0]["value"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("load_weights: failed to deserialise weights — %s", exc)
            return None
