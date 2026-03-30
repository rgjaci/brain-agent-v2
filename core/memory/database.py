"""Brain Agent v2 — SQLite + sqlite-vec memory database.

Provides the :class:`MemoryDatabase` which manages all persistent storage for
the agent:  episodic memories, knowledge-graph entities & relations, procedural
memory, document hierarchies, conversation history, and retrieval-feedback
logging.  Vector indices use the ``sqlite-vec`` extension; full-text search
uses SQLite's built-in FTS5.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ──────────────────────────────────────────────────────────────────────────────

_DDL_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT    NOT NULL,
    category        TEXT    NOT NULL DEFAULT 'general',
    source          TEXT    NOT NULL DEFAULT 'user',
    importance      REAL    NOT NULL DEFAULT 0.5,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    created_at      REAL    NOT NULL,
    last_accessed   REAL,
    usefulness_score REAL   NOT NULL DEFAULT 0.5,
    superseded_by   INTEGER REFERENCES memories(id),
    metadata        TEXT,
    CONSTRAINT importance_range CHECK (importance BETWEEN 0.0 AND 1.0),
    CONSTRAINT confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
);
"""

_DDL_MEMORY_VECTORS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
    memory_id INTEGER PRIMARY KEY,
    embedding float[768]
);
"""

_DDL_MEMORY_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    category,
    source,
    content='memories',
    content_rowid='id'
);
"""

_DDL_MEMORY_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memory_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content, category, source)
    VALUES (new.id, new.content, new.category, new.source);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, category, source)
    VALUES ('delete', old.id, old.content, old.category, old.source);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_update AFTER UPDATE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, category, source)
    VALUES ('delete', old.id, old.content, old.category, old.source);
    INSERT INTO memory_fts(rowid, content, category, source)
    VALUES (new.id, new.content, new.category, new.source);
END;
"""

_DDL_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    entity_type     TEXT    NOT NULL DEFAULT 'concept',
    description     TEXT,
    properties      TEXT,
    importance      REAL    NOT NULL DEFAULT 0.5,
    source          TEXT    NOT NULL DEFAULT 'inferred',
    access_count    INTEGER NOT NULL DEFAULT 0,
    created_at      REAL    NOT NULL,
    last_accessed   REAL,
    CONSTRAINT uq_entity_name_type UNIQUE (name, entity_type)
);
"""

_DDL_ENTITY_VECTORS = """
CREATE VIRTUAL TABLE IF NOT EXISTS entity_vectors USING vec0(
    entity_id INTEGER PRIMARY KEY,
    embedding float[768]
);
"""

_DDL_RELATIONS = """
CREATE TABLE IF NOT EXISTS relations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id       INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type   TEXT    NOT NULL,
    properties      TEXT,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    source          TEXT    NOT NULL DEFAULT 'inferred',
    created_at      REAL    NOT NULL,
    valid_from      DATETIME DEFAULT CURRENT_TIMESTAMP,
    valid_until     DATETIME,
    CONSTRAINT confidence_range CHECK (confidence BETWEEN 0.0 AND 1.0)
);
"""

_DDL_PROCEDURES = """
CREATE TABLE IF NOT EXISTS procedures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    description     TEXT    NOT NULL,
    trigger_pattern TEXT,
    preconditions   TEXT,
    steps           TEXT    NOT NULL,
    warnings        TEXT,
    context         TEXT,
    source          TEXT    NOT NULL DEFAULT 'learned',
    success_count   INTEGER NOT NULL DEFAULT 0,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    last_used       REAL,
    created_at      REAL    NOT NULL
);
"""

_DDL_PROCEDURE_VECTORS = """
CREATE VIRTUAL TABLE IF NOT EXISTS procedure_vectors USING vec0(
    procedure_id INTEGER PRIMARY KEY,
    embedding float[768]
);
"""

_DDL_RETRIEVAL_LOG = """
CREATE TABLE IF NOT EXISTS retrieval_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT    NOT NULL,
    query_text          TEXT    NOT NULL,
    memory_id           INTEGER REFERENCES memories(id),
    retrieval_method    TEXT    NOT NULL,
    retrieval_rank      INTEGER NOT NULL DEFAULT 0,
    retrieval_score     REAL,
    was_in_context      INTEGER NOT NULL DEFAULT 0,
    was_useful          INTEGER,
    created_at          REAL    NOT NULL
);
"""

_DDL_RERANKER_TRAINING = """
CREATE TABLE IF NOT EXISTS reranker_training (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text      TEXT    NOT NULL,
    memory_id       INTEGER REFERENCES memories(id),
    features        TEXT    NOT NULL,
    label           INTEGER NOT NULL DEFAULT 0,
    created_at      REAL    NOT NULL
);
"""

_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    source_path     TEXT,
    doc_type        TEXT    NOT NULL DEFAULT 'text',
    summary         TEXT,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    created_at      REAL    NOT NULL,
    last_updated    REAL
);
"""

_DDL_DOCUMENT_SECTIONS = """
CREATE TABLE IF NOT EXISTS document_sections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id       INTEGER REFERENCES document_sections(id),
    title           TEXT,
    summary         TEXT,
    level           INTEGER NOT NULL DEFAULT 0,
    position        INTEGER NOT NULL DEFAULT 0,
    created_at      REAL    NOT NULL
);
"""

_DDL_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    metadata        TEXT,
    created_at      REAL    NOT NULL
);
"""

_DDL_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_memories_category    ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_source      ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_importance  ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_created_at  ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_superseded  ON memories(superseded_by);

CREATE INDEX IF NOT EXISTS idx_entities_name        ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type        ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_importance  ON entities(importance);

CREATE INDEX IF NOT EXISTS idx_relations_source     ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target     ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type       ON relations(relation_type);

CREATE INDEX IF NOT EXISTS idx_procedures_name      ON procedures(name);
CREATE INDEX IF NOT EXISTS idx_procedures_trigger   ON procedures(trigger_pattern);

CREATE INDEX IF NOT EXISTS idx_retrieval_session    ON retrieval_log(session_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_memory     ON retrieval_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_method     ON retrieval_log(retrieval_method);

CREATE INDEX IF NOT EXISTS idx_reranker_memory      ON reranker_training(memory_id);
CREATE INDEX IF NOT EXISTS idx_reranker_label       ON reranker_training(label);

CREATE INDEX IF NOT EXISTS idx_doc_sections_doc     ON document_sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_sections_parent  ON document_sections(parent_id);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_role    ON conversations(role);

CREATE INDEX IF NOT EXISTS idx_memories_usefulness ON memories(usefulness_score DESC);
CREATE INDEX IF NOT EXISTS idx_relations_valid     ON relations(valid_until);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _serialize_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into little-endian IEEE-754 bytes for sqlite-vec."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a :class:`sqlite3.Row` to a plain dict."""
    return dict(row)


def _now() -> float:
    """Current time as a Unix timestamp (seconds)."""
    return time.time()


# ──────────────────────────────────────────────────────────────────────────────
# MemoryDatabase
# ──────────────────────────────────────────────────────────────────────────────

class MemoryDatabase:
    """Persistent storage for Brain Agent v2.

    Wraps a SQLite database augmented with ``sqlite-vec`` for approximate
    nearest-neighbour vector search and FTS5 for BM25 full-text search.

    Args:
        db_path: Filesystem path to the SQLite database file.  The parent
                 directory is created automatically if it does not exist.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._vec_enabled: bool = False
        self._conn: sqlite3.Connection = self._connect()
        self._setup()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,   # autocommit; we manage transactions manually
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-32000;")   # ~32 MB page cache
        return conn

    # ── Extension loading ─────────────────────────────────────────────────────

    def _load_sqlite_vec(self) -> None:
        """Attempt to load the sqlite-vec extension.  Logs a warning on failure."""
        try:
            import sqlite_vec  # type: ignore[import]
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._vec_enabled = True
            logger.debug("sqlite-vec extension loaded successfully.")
        except ImportError:
            logger.warning(
                "sqlite-vec Python package not found.  "
                "Install it with: pip install sqlite-vec  "
                "Vector search will be disabled."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load sqlite-vec extension (%s).  "
                "Vector search will be disabled.",
                exc,
            )

    # ── Schema setup ──────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Create all tables, virtual tables, triggers, and indexes."""
        self._load_sqlite_vec()

        with self._conn:
            # Core tables
            self._conn.executescript(_DDL_MEMORIES)
            self._conn.executescript(_DDL_MEMORY_FTS)
            self._conn.executescript(_DDL_MEMORY_FTS_TRIGGERS)
            self._conn.executescript(_DDL_ENTITIES)
            self._conn.executescript(_DDL_RELATIONS)
            self._conn.executescript(_DDL_PROCEDURES)
            self._conn.executescript(_DDL_RETRIEVAL_LOG)
            self._conn.executescript(_DDL_RERANKER_TRAINING)
            self._conn.executescript(_DDL_DOCUMENTS)
            self._conn.executescript(_DDL_DOCUMENT_SECTIONS)
            self._conn.executescript(_DDL_CONVERSATIONS)
            self._conn.executescript(_DDL_INDEXES)

            # Vector tables — only when extension is available
            if self._vec_enabled:
                self._conn.executescript(_DDL_MEMORY_VECTORS)
                self._conn.executescript(_DDL_ENTITY_VECTORS)
                self._conn.executescript(_DDL_PROCEDURE_VECTORS)

        logger.debug(
            "MemoryDatabase schema initialised (vec_enabled=%s, path=%s).",
            self._vec_enabled,
            self.db_path,
        )

    # ── Generic query helper ──────────────────────────────────────────────────

    def execute(
        self, sql: str, params: tuple | list = ()
    ) -> list[dict[str, Any]]:
        """Execute an arbitrary SQL statement and return all rows as dicts.

        Useful for ad-hoc KG traversal queries from outside this class.

        Args:
            sql:    SQL statement to execute.
            params: Positional query parameters.

        Returns:
            List of row dictionaries (may be empty).
        """
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    # ── Memories ──────────────────────────────────────────────────────────────

    def insert_memory(
        self,
        content: str,
        category: str = "general",
        source: str = "user",
        importance: float = 0.5,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> int:
        """Insert a new episodic memory and return its row ID.

        Args:
            content:    The memory text.
            category:   High-level category label (e.g. ``"fact"``, ``"task"``).
            source:     Origin of the memory (e.g. ``"user"``, ``"agent"``).
            importance: Score in ``[0.0, 1.0]`` for retrieval prioritisation.
            confidence: Confidence in the accuracy of the memory.
            metadata:   Optional dict of extra attributes; stored as JSON.

        Returns:
            The ``id`` of the newly inserted row.
        """
        meta_json = json.dumps(metadata) if metadata else None
        now = _now()
        cursor = self._conn.execute(
            """
            INSERT INTO memories
                (content, category, source, importance, confidence,
                 created_at, last_accessed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (content, category, source, importance, confidence, now, now, meta_json),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_memory(self, memory_id: int) -> Optional[dict[str, Any]]:
        """Fetch a single memory by primary key.

        Args:
            memory_id: The ``id`` of the memory row.

        Returns:
            A dict representation of the row, or ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        result = _row_to_dict(row)
        if result.get("metadata"):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    def update_memory_access(self, memory_id: int) -> None:
        """Increment ``access_count`` and refresh ``last_accessed`` timestamp.

        Args:
            memory_id: Target memory row ID.
        """
        self._conn.execute(
            """
            UPDATE memories
               SET access_count = access_count + 1,
                   last_accessed = ?
             WHERE id = ?
            """,
            (_now(), memory_id),
        )

    def mark_superseded(self, old_memory_id: int, new_memory_id: int) -> None:
        """Link an old memory to its replacement.

        Args:
            old_memory_id: ID of the memory being superseded.
            new_memory_id: ID of the newer memory that replaces it.
        """
        self._conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE id = ?",
            (new_memory_id, old_memory_id),
        )

    # ── Embeddings ────────────────────────────────────────────────────────────

    def insert_embedding(
        self,
        table: str,
        id_col: str,
        rowid: int,
        embedding: list[float],
    ) -> None:
        """Insert or replace a vector embedding in a ``vec0`` virtual table.

        Args:
            table:     Name of the virtual table (e.g. ``"memory_vectors"``).
            id_col:    Name of the ID column in that table (e.g. ``"memory_id"``).
            rowid:     The integer key that links the vector to its parent row.
            embedding: Float list; must match the table's declared dimensions.

        Raises:
            RuntimeError: If the sqlite-vec extension is not available.
        """
        if not self._vec_enabled:
            logger.debug(
                "insert_embedding skipped — sqlite-vec not available "
                "(table=%s, rowid=%d).",
                table,
                rowid,
            )
            return
        blob = _serialize_embedding(embedding)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({id_col}, embedding) VALUES (?, ?)",
            (rowid, blob),
        )

    def get_all_embeddings(
        self, table: str, id_col: str, limit: int = 1000
    ) -> list[tuple[int, list[float]]]:
        """Retrieve all stored embeddings from a vec0 table.

        Useful for offline consolidation / re-clustering tasks.

        Args:
            table:  Name of the virtual vector table.
            id_col: Name of the integer ID column.
            limit:  Maximum number of rows to return.

        Returns:
            List of ``(id, embedding_floats)`` tuples.
        """
        if not self._vec_enabled:
            return []
        rows = self._conn.execute(
            f"SELECT {id_col}, embedding FROM {table} LIMIT ?", (limit,)
        ).fetchall()
        result: list[tuple[int, list[float]]] = []
        for row in rows:
            rid = row[0]
            blob = row[1]
            n = len(blob) // 4
            floats = list(struct.unpack(f"{n}f", blob))
            result.append((rid, floats))
        return result

    # ── Vector search ─────────────────────────────────────────────────────────

    def vector_search(
        self,
        embedding: list[float],
        table: str = "memory_vectors",
        id_col: str = "memory_id",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Approximate nearest-neighbour search using sqlite-vec.

        Args:
            embedding: Query vector.
            table:     Virtual table to search.
            id_col:    Name of the ID column to return.
            limit:     Maximum number of results.

        Returns:
            List of ``{"id": int, "distance": float}`` dicts ordered by
            ascending distance (closest first).  Returns an empty list when
            sqlite-vec is unavailable.
        """
        if not self._vec_enabled:
            logger.debug("vector_search skipped — sqlite-vec not available.")
            return []
        blob = _serialize_embedding(embedding)
        rows = self._conn.execute(
            f"""
            SELECT {id_col} AS id, distance
              FROM {table}
             WHERE embedding MATCH ?
               AND k = ?
            ORDER BY distance
            """,
            (blob, limit),
        ).fetchall()
        return [{"id": row["id"], "distance": row["distance"]} for row in rows]

    def vector_search_procedures(
        self, embedding: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Convenience wrapper for procedure vector search.

        Args:
            embedding: Query vector.
            limit:     Maximum number of results.

        Returns:
            List of ``{"id": int, "distance": float}`` dicts.
        """
        return self.vector_search(
            embedding,
            table="procedure_vectors",
            id_col="procedure_id",
            limit=limit,
        )

    # ── Full-text search ──────────────────────────────────────────────────────

    def fts_search(
        self, query: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """BM25 full-text search over the ``memory_fts`` FTS5 table.

        Args:
            query: FTS5 query string (plain text or FTS5 syntax).
            limit: Maximum number of results.

        Returns:
            List of ``{"id": int, "rank": float}`` dicts ordered by
            relevance (most relevant first).
        """
        rows = self._conn.execute(
            """
            SELECT rowid AS id, rank
              FROM memory_fts
             WHERE memory_fts MATCH ?
             ORDER BY rank
             LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [{"id": row["id"], "rank": row["rank"]} for row in rows]

    # ── Entities ──────────────────────────────────────────────────────────────

    def insert_entity(
        self,
        name: str,
        entity_type: str = "concept",
        description: Optional[str] = None,
        properties: Optional[dict] = None,
        importance: float = 0.5,
        source: str = "inferred",
    ) -> int:
        """Insert a new knowledge-graph entity.

        Args:
            name:        Human-readable entity name.
            entity_type: Ontology label (e.g. ``"person"``, ``"concept"``).
            description: Free-text description.
            properties:  Arbitrary key-value properties stored as JSON.
            importance:  Salience score in ``[0.0, 1.0]``.
            source:      Provenance label.

        Returns:
            The ``id`` of the newly inserted row.
        """
        props_json = json.dumps(properties) if properties else None
        now = _now()
        cursor = self._conn.execute(
            """
            INSERT INTO entities
                (name, entity_type, description, properties,
                 importance, source, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, entity_type, description, props_json, importance, source, now, now),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "concept",
        description: Optional[str] = None,
        properties: Optional[dict] = None,
        importance: float = 0.5,
        source: str = "inferred",
    ) -> int:
        """Insert a new entity or update the existing one with the same
        ``(name, entity_type)`` unique key.

        Existing ``access_count``, ``created_at``, and ``importance`` are
        preserved; description, properties, and source are overwritten.

        Args:
            name:        Entity name (part of unique key).
            entity_type: Entity type (part of unique key).
            description: Updated description (``None`` leaves existing value).
            properties:  Updated properties dict (``None`` leaves existing value).
            importance:  Updated importance score.
            source:      Updated provenance label.

        Returns:
            The ``id`` of the inserted or updated row.
        """
        existing = self.get_entity(name=name)
        if existing and existing.get("entity_type") == entity_type:
            eid = existing["id"]
            props_json = (
                json.dumps(properties) if properties is not None
                else existing.get("properties")
            )
            desc = description if description is not None else existing.get("description")
            self._conn.execute(
                """
                UPDATE entities
                   SET description   = ?,
                       properties    = ?,
                       importance    = ?,
                       source        = ?,
                       last_accessed = ?
                 WHERE id = ?
                """,
                (desc, props_json, importance, source, _now(), eid),
            )
            return eid
        return self.insert_entity(
            name, entity_type, description, properties, importance, source
        )

    def get_entity(
        self,
        entity_id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Fetch an entity by ID or name.

        At least one of *entity_id* or *name* must be provided.

        Args:
            entity_id: Primary key lookup.
            name:      Name lookup (returns the first match).

        Returns:
            Dict representation or ``None`` if not found.
        """
        if entity_id is not None:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
        elif name is not None:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE name = ?", (name,)
            ).fetchone()
        else:
            raise ValueError("entity_id or name must be provided")

        if row is None:
            return None
        result = _row_to_dict(row)
        if result.get("properties"):
            try:
                result["properties"] = json.loads(result["properties"])
            except (json.JSONDecodeError, TypeError):
                pass
        return result

    # ── Relations ─────────────────────────────────────────────────────────────

    def insert_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        properties: Optional[dict] = None,
        confidence: float = 1.0,
        source: str = "inferred",
    ) -> int:
        """Insert a directed relation between two entities.

        Args:
            source_id:     ID of the source entity.
            target_id:     ID of the target entity.
            relation_type: Relation label (e.g. ``"works_at"``).
            properties:    Extra key-value data stored as JSON.
            confidence:    Confidence score in ``[0.0, 1.0]``.
            source:        Provenance label.

        Returns:
            The ``id`` of the newly inserted row.
        """
        props_json = json.dumps(properties) if properties else None
        cursor = self._conn.execute(
            """
            INSERT INTO relations
                (source_id, target_id, relation_type, properties,
                 confidence, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_id, target_id, relation_type, props_json,
             confidence, source, _now()),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_relations(
        self, entity_id: int, max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Return all relations where *entity_id* is the source or target.

        Args:
            entity_id:   Entity to pivot on.
            max_results: Upper bound on returned rows.

        Returns:
            List of relation dicts, each including resolved entity names.
        """
        rows = self._conn.execute(
            """
            SELECT r.*,
                   es.name AS source_name,
                   et.name AS target_name
              FROM relations r
              JOIN entities es ON es.id = r.source_id
              JOIN entities et ON et.id = r.target_id
             WHERE r.source_id = ? OR r.target_id = ?
             ORDER BY r.confidence DESC, r.created_at DESC
             LIMIT ?
            """,
            (entity_id, entity_id, max_results),
        ).fetchall()
        result = []
        for row in rows:
            d = _row_to_dict(row)
            if d.get("properties"):
                try:
                    d["properties"] = json.loads(d["properties"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result

    # ── Conversations ────────────────────────────────────────────────────────

    def store_conversation(
        self, session_id: str, role: str, content: str, metadata: Optional[dict] = None
    ) -> int:
        """Store a conversation message.

        Args:
            session_id: Session identifier.
            role:       Message role (``"user"``, ``"assistant"``, ``"system"``).
            content:    Message text.
            metadata:   Optional extra data stored as JSON.

        Returns:
            The ``id`` of the newly inserted row.
        """
        meta_json = json.dumps(metadata) if metadata else None
        cursor = self._conn.execute(
            """
            INSERT INTO conversations (session_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, meta_json, _now()),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_recent_messages(
        self, session_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Fetch recent conversation messages for a session.

        Args:
            session_id: Session identifier.
            limit:      Maximum number of messages to return.

        Returns:
            List of message dicts ordered oldest-first.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM conversations
             WHERE session_id = ?
             ORDER BY created_at DESC
             LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        result = [_row_to_dict(r) for r in reversed(rows)]
        return result

    # ── Usefulness ───────────────────────────────────────────────────────────

    def update_usefulness(self, memory_id: int, delta: float) -> None:
        """Adjust the usefulness_score of a memory by *delta*, clamped to [0, 1].

        Args:
            memory_id: Target memory row ID.
            delta:     Amount to add (positive) or subtract (negative).
        """
        self._conn.execute(
            """
            UPDATE memories
               SET usefulness_score = MAX(0.0, MIN(1.0, usefulness_score + ?))
             WHERE id = ?
            """,
            (delta, memory_id),
        )

    # ── Count ────────────────────────────────────────────────────────────────

    def count_memories(self) -> int:
        """Return the total number of rows in the memories table."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        return int(row["n"]) if row else 0
