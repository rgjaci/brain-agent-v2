"""Brain Agent v2 — Procedural memory store.

Implements :class:`ProcedureStore`, which manages step-by-step procedures in
the ``procedures`` SQLite table.  Retrieval uses BM25 FTS (via a custom query
against the ``procedures`` table — *not* the ``memory_fts`` virtual table) and
ranks results with a **UCB1** (Upper Confidence Bound) score to balance
exploitation of high-success procedures with exploration of less-tried ones.

Table schema (from database.py)::

    CREATE TABLE IF NOT EXISTS procedures (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT    NOT NULL UNIQUE,
        description     TEXT    NOT NULL,
        trigger_pattern TEXT,
        preconditions   TEXT,          -- JSON list
        steps           TEXT    NOT NULL,  -- JSON list
        warnings        TEXT,          -- JSON list
        context         TEXT,
        source          TEXT    NOT NULL DEFAULT 'learned',
        success_count   INTEGER NOT NULL DEFAULT 0,
        failure_count   INTEGER NOT NULL DEFAULT 0,
        last_used       REAL,
        created_at      REAL    NOT NULL
    );

Note: The schema uses ``failure_count`` (not ``attempt_count``), so
``attempt_count`` is computed as ``success_count + failure_count + 1``.
The ``confidence`` column from the spec is also derived at runtime.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Valid procedure tiers (per spec Technique 3)
# ──────────────────────────────────────────────────────────────────────────────

VALID_PROCEDURE_TIERS = {
    1: "atomic",       # Low-level, specific actions
    2: "task",         # Mid-level, reusable task procedures
    3: "strategy",     # High-level, abstract strategy templates
}


@dataclass
class Procedure:
    """An in-memory representation of a stored procedure.

    Attributes:
        id:              Database primary key (``None`` before first persist).
        name:            Unique procedure name.
        description:     Human-readable summary.
        trigger_pattern: Regex or keyword pattern that suggests this procedure.
        preconditions:   List of prerequisite conditions (free-form strings).
        steps:           Ordered list of action steps.
        warnings:        List of caveats / warnings.
        context:         Free-form contextual notes.
        tier:            Procedure tier: 1=atomic, 2=task, 3=strategy.
        success_count:   Number of times this procedure was recorded as successful.
        attempt_count:   Total attempts (successes + failures + 1 to avoid log(0)).
        confidence:      Cached derived confidence in ``[0.0, 1.0]``.
    """

    id: int | None
    name: str
    description: str
    trigger_pattern: str
    preconditions: list[str]
    steps: list[str]
    warnings: list[str]
    context: str
    tier: int = 2  # Default to task-level procedures
    success_count: int = 0
    attempt_count: int = 1
    confidence: float = 0.5

    @property
    def tier_name(self) -> str:
        """Human-readable tier name."""
        return VALID_PROCEDURE_TIERS.get(self.tier, "unknown")


# ──────────────────────────────────────────────────────────────────────────────
# ProcedureStore
# ──────────────────────────────────────────────────────────────────────────────


class ProcedureStore:
    """CRUD layer and UCB-ranked retrieval for procedural memory.

    All persistence is delegated to a :class:`MemoryDatabase` instance via its
    generic :meth:`~MemoryDatabase.execute` method, because the ``procedures``
    table is a plain SQLite table (not FTS5) and the ``memory_fts`` virtual
    table covers only the ``memories`` table.

    Args:
        db: An initialised :class:`MemoryDatabase` instance.

    Example::

        store = ProcedureStore(db)
        procs = store.find_relevant("how to deploy a docker container")
        context_block = store.format_for_context(procs)
    """

    def __init__(self, db) -> None:
        self.db = db

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_procedure(row: dict) -> Procedure:
        """Convert a database row dict to a :class:`Procedure` dataclass.

        JSON-encoded columns (``steps``, ``preconditions``, ``warnings``) are
        decoded; any decode error falls back to wrapping the raw value in a
        list.
        """

        def _decode_json_list(value, fallback_key: str = "") -> list[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(v) for v in value]
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
                return [str(parsed)]
            except (json.JSONDecodeError, TypeError):
                return [str(value)] if value else []

        success_count = int(row.get("success_count") or 0)
        failure_count = int(row.get("failure_count") or 0)
        attempt_count = success_count + failure_count + 1  # +1 avoids log(0)
        confidence = success_count / attempt_count if attempt_count > 1 else 0.5
        tier = int(row.get("tier") or 2)  # Default to task-level

        return Procedure(
            id=row.get("id"),
            name=row.get("name", ""),
            description=row.get("description", ""),
            trigger_pattern=row.get("trigger_pattern") or "",
            preconditions=_decode_json_list(row.get("preconditions")),
            steps=_decode_json_list(row.get("steps")),
            warnings=_decode_json_list(row.get("warnings")),
            context=row.get("context") or "",
            tier=tier,
            success_count=success_count,
            attempt_count=attempt_count,
            confidence=confidence,
        )

    @staticmethod
    def _ucb_score(
        success_count: int,
        attempt_count: int,
        total_attempts: int,
    ) -> float:
        """Compute UCB1 score for a single procedure.

        Formula::

            ucb = success_rate + sqrt(2 * log(total_attempts + 1) / (attempt_count + 1))

        Where:
            * ``success_rate = success_count / attempt_count``
            * ``total_attempts`` is the sum of all procedure attempts (exploration
              term denominator).

        Args:
            success_count:  Number of successful executions.
            attempt_count:  Total executions (success + failure, min 1).
            total_attempts: Global total across all procedures.

        Returns:
            UCB1 score (higher = more worth exploring/exploiting).
        """
        success_rate = success_count / max(attempt_count, 1)
        exploration = math.sqrt(
            2.0 * math.log(total_attempts + 1) / (attempt_count + 1)
        )
        return success_rate + exploration

    @staticmethod
    def _text_relevance(query: str, procedure: Procedure) -> float:
        """Compute a simple keyword-overlap relevance score.

        Normalises both the query and the combined procedure text, then
        counts the fraction of unique query tokens that appear in the
        procedure text.

        Args:
            query:     User query string.
            procedure: Candidate :class:`Procedure`.

        Returns:
            Relevance score in ``[0.0, 1.0]``.
        """
        q_tokens = set(re.findall(r"\w+", query.lower()))
        combined = " ".join([
            procedure.name,
            procedure.description,
            procedure.trigger_pattern,
            " ".join(procedure.steps),
        ]).lower()
        p_tokens = set(re.findall(r"\w+", combined))

        if not q_tokens:
            return 0.0
        overlap = q_tokens & p_tokens
        return len(overlap) / len(q_tokens)

    # ── Public API ────────────────────────────────────────────────────────────

    def find_relevant(self, query: str, max_results: int = 3) -> list[Procedure]:
        """Find procedures relevant to *query* using FTS + UCB ranking.

        Steps:

        1. Sanitise the query for SQLite LIKE patterns.
        2. Search the ``procedures`` table for rows whose ``name``,
           ``description``, or ``trigger_pattern`` match any query token.
        3. Compute UCB1 score for each candidate.
        4. Combine UCB1 with text-relevance and sort descending.
        5. Return the top *max_results* procedures.

        Args:
            query:       User query text.
            max_results: Maximum number of :class:`Procedure` objects to return.

        Returns:
            List of :class:`Procedure` objects, best first.
        """
        if not query.strip():
            return []

        # Build a set of search tokens (≥ 3 chars, alphanumeric)
        tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3]
        if not tokens:
            return []

        # Build LIKE conditions for each token across the three text columns
        conditions: list[str] = []
        params: list[str] = []
        for token in tokens[:10]:  # guard against very long queries
            pattern = f"%{token}%"
            conditions.append(
                "(LOWER(name) LIKE ? OR LOWER(description) LIKE ? "
                "OR LOWER(trigger_pattern) LIKE ?)"
            )
            params.extend([pattern, pattern, pattern])

        where_clause = " OR ".join(conditions)
        sql = f"""
            SELECT id, name, description, trigger_pattern,
                   preconditions, steps, warnings, context, tier,
                   success_count, failure_count
              FROM procedures
             WHERE {where_clause}
             LIMIT 50
        """

        try:
            rows = self.db.execute(sql, params)
        except Exception as exc:
            logger.warning("ProcedureStore.find_relevant query failed: %s", exc)
            return []

        if not rows:
            return []

        # Compute total_attempts across all retrieved candidates for UCB
        total_attempts = sum(
            int(r.get("success_count", 0)) + int(r.get("failure_count", 0))
            for r in rows
        ) + 1  # +1 so log doesn't blow up on empty

        candidates: list[tuple[float, Procedure]] = []
        for row in rows:
            proc = self._row_to_procedure(row)
            ucb = self._ucb_score(
                success_count=proc.success_count,
                attempt_count=proc.attempt_count,
                total_attempts=total_attempts,
            )
            relevance = self._text_relevance(query, proc)
            combined_score = ucb * (relevance + 0.01)  # +0.01 avoids 0 * ucb = 0
            candidates.append((combined_score, proc))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [proc for _, proc in candidates[:max_results]]

    def find_by_tier(
        self, tier: int, max_results: int = 5
    ) -> list[Procedure]:
        """Find procedures by tier.

        Tiers (per spec Technique 3):
            1 = Atomic actions (low-level, specific)
            2 = Task procedures (mid-level, reusable)
            3 = Strategy templates (high-level, abstract)

        Args:
            tier:        Procedure tier (1, 2, or 3).
            max_results: Maximum number of procedures to return.

        Returns:
            List of :class:`Procedure` objects ordered by confidence.
        """
        if tier not in VALID_PROCEDURE_TIERS:
            logger.warning("Invalid procedure tier: %d", tier)
            return []

        sql = """
            SELECT id, name, description, trigger_pattern,
                   preconditions, steps, warnings, context, tier,
                   success_count, failure_count
              FROM procedures
             WHERE tier = ?
             ORDER BY success_count DESC
             LIMIT ?
        """
        try:
            rows = self.db.execute(sql, (tier, max_results))
        except Exception as exc:
            logger.warning("ProcedureStore.find_by_tier failed: %s", exc)
            return []

        return [self._row_to_procedure(row) for row in rows]

    def record_success(self, procedure_id: int) -> None:
        """Increment the success counter for a procedure.

        Also updates the ``last_used`` timestamp.

        Args:
            procedure_id: Primary key of the procedure row.
        """
        try:
            self.db.execute(
                """
                UPDATE procedures
                   SET success_count = success_count + 1,
                       last_used = ?
                 WHERE id = ?
                """,
                (time.time(), procedure_id),
            )
            logger.debug("Recorded success for procedure id=%d", procedure_id)
        except Exception as exc:
            logger.warning("record_success failed for id=%d: %s", procedure_id, exc)

    def record_failure(self, procedure_id: int) -> None:
        """Increment the failure counter for a procedure.

        Also updates the ``last_used`` timestamp.

        Args:
            procedure_id: Primary key of the procedure row.
        """
        try:
            self.db.execute(
                """
                UPDATE procedures
                   SET failure_count = failure_count + 1,
                       last_used = ?
                 WHERE id = ?
                """,
                (time.time(), procedure_id),
            )
            logger.debug("Recorded failure for procedure id=%d", procedure_id)
        except Exception as exc:
            logger.warning("record_failure failed for id=%d: %s", procedure_id, exc)

    def store(self, procedure: Procedure) -> int:
        """Persist a :class:`Procedure` to the database.

        Uses ``INSERT OR REPLACE`` so that updating an existing procedure by
        name is idempotent.  The returned ID is also written back into
        ``procedure.id``.

        Args:
            procedure: The :class:`Procedure` to store.

        Returns:
            The database primary key of the inserted / replaced row.

        Raises:
            RuntimeError: If the database write fails.
        """
        steps_json = json.dumps(procedure.steps)
        preconditions_json = json.dumps(procedure.preconditions)
        warnings_json = json.dumps(procedure.warnings)
        now = time.time()

        try:
            rows = self.db.execute(
                """
                INSERT INTO procedures
                    (name, description, trigger_pattern, preconditions,
                     steps, warnings, context, source,
                     success_count, failure_count, last_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description     = excluded.description,
                    trigger_pattern = excluded.trigger_pattern,
                    preconditions   = excluded.preconditions,
                    steps           = excluded.steps,
                    warnings        = excluded.warnings,
                    context         = excluded.context,
                    last_used       = excluded.last_used
                RETURNING id
                """,
                (
                    procedure.name,
                    procedure.description,
                    procedure.trigger_pattern,
                    preconditions_json,
                    steps_json,
                    warnings_json,
                    procedure.context,
                    "learned",
                    procedure.success_count,
                    max(0, procedure.attempt_count - procedure.success_count - 1),
                    now,
                    now,
                ),
            )
            # RETURNING gives us the id in the result set
            if rows and rows[0].get("id") is not None:
                new_id = int(rows[0]["id"])
            else:
                # Fallback: look it up by name
                lookup = self.db.execute(
                    "SELECT id FROM procedures WHERE name = ?", (procedure.name,)
                )
                if not lookup:
                    raise RuntimeError(
                        f"Could not determine id for procedure '{procedure.name}'"
                    )
                new_id = int(lookup[0]["id"])

            procedure.id = new_id
            logger.debug("Stored procedure '%s' with id=%d", procedure.name, new_id)
            return new_id

        except Exception as exc:
            logger.error("ProcedureStore.store failed: %s", exc)
            raise RuntimeError(f"Failed to store procedure '{procedure.name}': {exc}") from exc

    def get_all(self) -> list[Procedure]:
        """Return every procedure in the database.

        Results are ordered by descending success rate (most reliable first).

        Returns:
            List of :class:`Procedure` objects (may be empty).
        """
        try:
            rows = self.db.execute(
                """
                SELECT id, name, description, trigger_pattern,
                       preconditions, steps, warnings, context,
                       success_count, failure_count
                  FROM procedures
                 ORDER BY
                    CAST(success_count AS REAL) /
                    (success_count + failure_count + 1) DESC,
                    created_at DESC
                """
            )
            return [self._row_to_procedure(r) for r in rows]
        except Exception as exc:
            logger.warning("ProcedureStore.get_all failed: %s", exc)
            return []

    def format_for_context(
        self,
        procedures: list[Procedure],
        budget_tokens: int = 2_000,
    ) -> str:
        """Format a list of procedures as a prompt-ready context block.

        Each procedure is rendered as a numbered-step list::

            ### <Name>
            <description>

            Steps:
              1. <step 1>
              2. <step 2>
              ...

            Warnings:
              ⚠ <warning 1>

        Procedures are included in order until the token budget (estimated as
        ``len(text) // 4``) is exhausted.

        Args:
            procedures:    List of :class:`Procedure` objects to format.
            budget_tokens: Approximate maximum tokens for the output string.

        Returns:
            Formatted string, or an empty string when *procedures* is empty.
        """
        if not procedures:
            return ""

        blocks: list[str] = []
        tokens_used = 0

        for proc in procedures:
            lines: list[str] = [f"### {proc.name}"]

            if proc.description:
                lines.append(proc.description)

            if proc.preconditions:
                lines.append("\nPreconditions:")
                for cond in proc.preconditions:
                    lines.append(f"  • {cond}")

            if proc.steps:
                lines.append("\nSteps:")
                for i, step in enumerate(proc.steps, start=1):
                    lines.append(f"  {i}. {step}")

            if proc.warnings:
                lines.append("\nWarnings:")
                for warn in proc.warnings:
                    lines.append(f"  ⚠ {warn}")

            block = "\n".join(lines)
            cost = len(block) // 4
            if tokens_used + cost > budget_tokens:
                break
            blocks.append(block)
            tokens_used += cost

        return "\n\n".join(blocks)
