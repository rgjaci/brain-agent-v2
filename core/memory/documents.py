"""Brain Agent v2 — Document Ingestion Pipeline.

Implements the ``brain ingest <file>`` command, turning files and directory
trees into atomic :class:`DocumentChunk` objects that are then handed off to
the memory writer for embedding and storage.

Key features:

* **Format-aware chunking** — Python files split on ``def``/``class``
  boundaries; Markdown files split on headings; all others split on paragraph
  breaks with a character-count fallback.
* **Overlap** — consecutive chunks share :data:`CHUNK_OVERLAP` characters to
  preserve cross-boundary context.
* **Deduplication** — SHA-256 content hashing prevents re-ingesting unchanged
  files.
* **Size guard** — files larger than 500 KB are skipped to avoid memory
  pressure.
* **Directory walking** — :meth:`DocumentIngester.ingest_directory` recurses
  through a directory tree, automatically skipping hidden files and common
  non-source directories (``node_modules``, ``__pycache__``, ``.venv``,
  ``.git``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: set[str] = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".sh", ".bash", ".html", ".css",
    ".rst", ".csv",
}

MAX_CHUNK_SIZE: int = 1000    # maximum characters per chunk
CHUNK_OVERLAP: int = 100      # overlap characters between consecutive chunks
MAX_FILE_SIZE: int = 500_000  # 500 KB

# Directories to skip when walking a directory tree
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build",
})

# Config-table key prefix used to track ingested file hashes
_HASH_KEY_PREFIX = "ingested_file:"


# ──────────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single chunk of an ingested document.

    Attributes:
        content:      The chunk's text content.
        source_path:  Absolute filesystem path of the source file.
        chunk_index:  Zero-based index of this chunk within the document.
        total_chunks: Total number of chunks in the document.
        metadata:     Extra information (e.g. language, heading, byte range).
    """

    content: str
    source_path: str
    chunk_index: int
    total_chunks: int
    metadata: dict


# ──────────────────────────────────────────────────────────────────────────────
# DocumentIngester
# ──────────────────────────────────────────────────────────────────────────────

class DocumentIngester:
    """Ingest files and directories into the agent's memory store.

    Args:
        db:     Initialised :class:`~.database.MemoryDatabase` (used for
                the config table that tracks ingested-file hashes).
        writer: :class:`~.writer.MemoryWriter` instance whose
                ``process_document_chunk`` method is called for each chunk.
    """

    def __init__(self, db, writer) -> None:
        self.db = db
        self.writer = writer
        self._ensure_config_table()

    # ── Config table ──────────────────────────────────────────────────────────

    def _ensure_config_table(self) -> None:
        """Create the ``config`` table if it does not yet exist."""
        try:
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS config (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
        except Exception as exc:
            logger.debug("_ensure_config_table: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    async def ingest_file(self, path: str, session_id: str) -> dict:
        """Ingest a single file into memory.

        Steps:

        1. Validate the file exists and its extension is supported.
        2. Read the file (up to :data:`MAX_FILE_SIZE` bytes; skip if larger).
        3. Compute SHA-256 hash; skip if the file was already ingested with the
           same content.
        4. Chunk the content with :meth:`smart_chunk`.
        5. Call ``writer.process_document_chunk(chunk, session_id)`` for each
           chunk.
        6. Record the file's hash in the config table.

        Args:
            path:       Filesystem path to the file.
            session_id: Current conversation session ID passed to the writer.

        Returns:
            A result dict with the keys:

            * ``"chunks"``  — number of chunks created
            * ``"facts"``   — number of facts extracted (currently equals chunks)
            * ``"status"``  — ``"ok"`` or ``"error"``
            * ``"message"`` — human-readable status message
        """
        file_path = Path(path).resolve()

        # ── Validation ────────────────────────────────────────────────────────
        if not file_path.exists():
            return {
                "chunks": 0, "facts": 0,
                "status": "error",
                "message": f"File not found: {path}",
            }

        if not file_path.is_file():
            return {
                "chunks": 0, "facts": 0,
                "status": "error",
                "message": f"Path is not a file: {path}",
            }

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return {
                "chunks": 0, "facts": 0,
                "status": "error",
                "message": (
                    f"Unsupported file type '{file_path.suffix}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            }

        # ── Size guard ────────────────────────────────────────────────────────
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "chunks": 0, "facts": 0,
                "status": "error",
                "message": (
                    f"File too large ({file_size:,} bytes > {MAX_FILE_SIZE:,} bytes "
                    "limit). Skipping."
                ),
            }

        # ── Read content ──────────────────────────────────────────────────────
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {
                "chunks": 0, "facts": 0,
                "status": "error",
                "message": f"Could not read file: {exc}",
            }

        # ── Deduplication ─────────────────────────────────────────────────────
        file_hash = self.compute_hash(content)
        if self.is_already_ingested(file_hash):
            logger.info("ingest_file: %s already ingested (hash=%s).", path, file_hash[:8])
            return {
                "chunks": 0, "facts": 0,
                "status": "ok",
                "message": f"File already ingested (no changes detected): {path}",
            }

        # ── Chunking ──────────────────────────────────────────────────────────
        chunks = self.smart_chunk(content, str(file_path))

        # ── Write chunks ──────────────────────────────────────────────────────
        facts_written = 0
        for chunk in chunks:
            try:
                await self.writer.process_document_chunk(chunk, session_id)
                facts_written += 1
            except Exception as exc:
                logger.warning(
                    "ingest_file: failed to write chunk %d/%d of %s — %s.",
                    chunk.chunk_index + 1, chunk.total_chunks, path, exc,
                )

        # ── Mark ingested ─────────────────────────────────────────────────────
        self.mark_ingested(str(file_path), file_hash)

        logger.info(
            "ingest_file: %s → %d chunks, %d facts written.",
            path, len(chunks), facts_written,
        )
        return {
            "chunks": len(chunks),
            "facts": facts_written,
            "status": "ok",
            "message": (
                f"Successfully ingested '{file_path.name}': "
                f"{len(chunks)} chunks, {facts_written} facts stored."
            ),
        }

    async def ingest_directory(
        self, path: str, session_id: str, recursive: bool = True
    ) -> dict:
        """Ingest all supported files in a directory.

        Skips: hidden files/directories (names starting with ``.``), and the
        directories listed in :data:`_SKIP_DIRS`.

        Args:
            path:       Filesystem path to the directory.
            session_id: Current conversation session ID.
            recursive:  When ``True`` (default) recurse into sub-directories.

        Returns:
            Aggregated result dict with the keys ``"files"``, ``"chunks"``,
            ``"facts"``, ``"skipped"``, ``"errors"``, ``"status"``, and
            ``"message"``.
        """
        dir_path = Path(path).resolve()

        if not dir_path.exists():
            return {
                "files": 0, "chunks": 0, "facts": 0,
                "skipped": 0, "errors": 0,
                "status": "error",
                "message": f"Directory not found: {path}",
            }

        if not dir_path.is_dir():
            return {
                "files": 0, "chunks": 0, "facts": 0,
                "skipped": 0, "errors": 0,
                "status": "error",
                "message": f"Path is not a directory: {path}",
            }

        total_files = 0
        total_chunks = 0
        total_facts = 0
        total_skipped = 0
        total_errors = 0

        # Collect candidate files
        if recursive:
            candidate_files = []
            for root, dirs, files in os.walk(dir_path):
                root_path = Path(root)

                # Prune skipped directories in-place so os.walk doesn't recurse
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".")
                    and d not in _SKIP_DIRS
                ]

                for filename in files:
                    if filename.startswith("."):
                        continue
                    candidate_files.append(root_path / filename)
        else:
            candidate_files = [
                f for f in dir_path.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]

        for file_path in candidate_files:
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                total_skipped += 1
                continue

            result = await self.ingest_file(str(file_path), session_id)
            total_files += 1

            if result["status"] == "error":
                total_errors += 1
                logger.warning(
                    "ingest_directory: error on %s — %s",
                    file_path, result["message"],
                )
            else:
                total_chunks += result["chunks"]
                total_facts += result["facts"]

        logger.info(
            "ingest_directory: %s — files=%d, chunks=%d, facts=%d, "
            "skipped=%d, errors=%d.",
            path, total_files, total_chunks, total_facts,
            total_skipped, total_errors,
        )
        return {
            "files": total_files,
            "chunks": total_chunks,
            "facts": total_facts,
            "skipped": total_skipped,
            "errors": total_errors,
            "status": "ok" if total_errors == 0 else "partial",
            "message": (
                f"Ingested {total_files} files: {total_chunks} chunks, "
                f"{total_facts} facts. "
                f"Skipped {total_skipped} unsupported files, "
                f"{total_errors} errors."
            ),
        }

    # ── Chunking ──────────────────────────────────────────────────────────────

    def smart_chunk(self, content: str, source_path: str) -> list[DocumentChunk]:
        """Split document content into overlapping chunks.

        Chunking strategy is chosen based on the file extension:

        * **Python** (``.py``) — splits on ``def `` and ``class `` boundaries,
          keeping each function/class as its own chunk (with overflow splitting
          for very large definitions).
        * **Markdown** (``.md``, ``.rst``) — splits on level-1 or level-2
          heading lines (``# …`` / ``## …``).
        * **All others** — splits on blank lines (``\\n\\n``), with a
          character-count fallback for paragraphs that still exceed
          :data:`MAX_CHUNK_SIZE`.

        All strategies respect :data:`MAX_CHUNK_SIZE` and prepend
        :data:`CHUNK_OVERLAP` characters from the previous chunk to maintain
        cross-boundary context.

        Args:
            content:     Full file content as a string.
            source_path: Absolute path to the source file (used to pick the
                         chunking strategy and populate chunk metadata).

        Returns:
            List of :class:`DocumentChunk` objects; at least one per non-empty
            file.
        """
        if not content.strip():
            return []

        ext = Path(source_path).suffix.lower()
        language = ext.lstrip(".")

        if ext == ".py":
            raw_chunks = self._chunk_python(content)
        elif ext in {".md", ".rst"}:
            raw_chunks = self._chunk_markdown(content)
        else:
            raw_chunks = self._chunk_generic(content)

        # Post-process: enforce MAX_CHUNK_SIZE and add overlap
        processed: list[str] = []
        prev_tail = ""

        for raw in raw_chunks:
            if not raw.strip():
                continue

            # Prepend overlap from previous chunk
            chunk_text = (prev_tail + raw) if prev_tail else raw

            # Split oversized chunks
            if len(chunk_text) > MAX_CHUNK_SIZE:
                sub_chunks = self._split_by_size(chunk_text)
                for _i, sub in enumerate(sub_chunks):
                    processed.append(sub)
                prev_tail = chunk_text[-CHUNK_OVERLAP:] if len(chunk_text) > CHUNK_OVERLAP else ""
            else:
                processed.append(chunk_text)
                prev_tail = chunk_text[-CHUNK_OVERLAP:] if len(chunk_text) > CHUNK_OVERLAP else ""

        if not processed:
            # Fallback: treat entire file as one chunk (truncated)
            processed = [content[:MAX_CHUNK_SIZE]]

        total = len(processed)
        return [
            DocumentChunk(
                content=text,
                source_path=source_path,
                chunk_index=idx,
                total_chunks=total,
                metadata={
                    "language": language,
                    "extension": ext,
                    "file_name": Path(source_path).name,
                },
            )
            for idx, text in enumerate(processed)
        ]

    # ── Language-specific splitters ───────────────────────────────────────────

    def _chunk_python(self, content: str) -> list[str]:
        """Split Python source on ``def `` and ``class `` boundaries."""
        # Split on lines that start a new top-level definition
        pattern = re.compile(r"(?=^(?:def |class |\S))", re.MULTILINE)
        parts = pattern.split(content)
        # Reassemble adjacent short parts
        return self._coalesce(parts)

    def _chunk_markdown(self, content: str) -> list[str]:
        """Split Markdown/RST on level-1 and level-2 headings."""
        # Split just before any line that starts with # or ##
        pattern = re.compile(r"(?=^#{1,2} )", re.MULTILINE)
        parts = pattern.split(content)
        return self._coalesce(parts)

    def _chunk_generic(self, content: str) -> list[str]:
        """Split on blank lines (paragraph breaks)."""
        parts = re.split(r"\n\s*\n", content)
        return self._coalesce(parts)

    def _coalesce(self, parts: list[str]) -> list[str]:
        """Merge consecutive short parts until each part is large enough.

        Tries to fill chunks up to MAX_CHUNK_SIZE by concatenating small
        adjacent parts, then returns the result.

        Args:
            parts: Raw split parts (may be very short).

        Returns:
            Coalesced list of strings (still may exceed MAX_CHUNK_SIZE for a
            single very long definition — handled by :meth:`smart_chunk`).
        """
        result: list[str] = []
        current = ""

        for part in parts:
            if not part.strip():
                continue
            if current and len(current) + len(part) > MAX_CHUNK_SIZE:
                result.append(current)
                current = part
            else:
                current = (current + "\n\n" + part) if current else part

        if current.strip():
            result.append(current)

        return result

    def _split_by_size(self, text: str) -> list[str]:
        """Hard-split a string into :data:`MAX_CHUNK_SIZE` character windows
        with :data:`CHUNK_OVERLAP` character overlap.

        Args:
            text: Input string (may be arbitrarily long).

        Returns:
            List of fixed-size windows.
        """
        chunks: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + MAX_CHUNK_SIZE, length)
            chunks.append(text[start:end])
            if end == length:
                break
            start = end - CHUNK_OVERLAP

        return chunks

    # ── Hashing & deduplication ───────────────────────────────────────────────

    def compute_hash(self, content: str) -> str:
        """Return the SHA-256 hex digest of *content*.

        Args:
            content: File content string (encoded as UTF-8 for hashing).

        Returns:
            64-character lowercase hex string.
        """
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    def is_already_ingested(self, file_hash: str) -> bool:
        """Check whether *file_hash* has been recorded in the config table.

        Args:
            file_hash: SHA-256 hex digest from :meth:`compute_hash`.

        Returns:
            ``True`` if the hash is found in the config table.
        """
        try:
            rows = self.db.execute(
                "SELECT key FROM config WHERE key = ?",
                (f"{_HASH_KEY_PREFIX}{file_hash}",),
            )
            return len(rows) > 0
        except Exception as exc:
            logger.debug("is_already_ingested: config query failed — %s", exc)
            return False

    def mark_ingested(self, path: str, file_hash: str) -> None:
        """Store the file hash and path in the config table.

        Uses ``INSERT OR REPLACE`` so that re-ingesting a changed file
        overwrites the previous entry.

        Args:
            path:      Absolute filesystem path of the ingested file.
            file_hash: SHA-256 hex digest from :meth:`compute_hash`.
        """
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                (f"{_HASH_KEY_PREFIX}{file_hash}", json.dumps({"path": path})),
            )
        except Exception as exc:
            logger.warning(
                "mark_ingested: could not write hash for %s — %s.", path, exc
            )
