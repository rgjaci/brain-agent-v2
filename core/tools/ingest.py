"""
Ingest tool — trigger document ingestion from the tool system.

Wraps DocumentIngester to make it available as an agent tool.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    success: bool
    path: str
    chunks: int = 0
    documents: int = 0
    error: str = ""


class IngestTool:
    """Trigger document ingestion as a tool call."""

    def __init__(self, ingester=None):
        self.ingester = ingester

    async def execute(self, path: str, recursive: bool = False) -> IngestResult:
        """Ingest a file or directory into memory."""
        if not self.ingester:
            return IngestResult(
                success=False,
                path=path,
                error="Document ingester not configured",
            )

        if not path or not path.strip():
            return IngestResult(success=False, path="", error="Empty path")

        resolved = Path(path).expanduser().resolve()

        if not resolved.exists():
            return IngestResult(
                success=False,
                path=str(resolved),
                error=f"Path not found: {resolved}",
            )

        try:
            if resolved.is_dir():
                if not recursive:
                    return IngestResult(
                        success=False,
                        path=str(resolved),
                        error=(
                            "Path is a directory. Pass recursive=True to ingest "
                            "all documents in it."
                        ),
                    )
                total_chunks = 0
                total_docs = 0
                for file_path in sorted(resolved.rglob("*")):
                    if file_path.is_file() and file_path.suffix.lower() in (
                        ".txt", ".md", ".rst", ".py", ".js", ".ts", ".json",
                        ".yaml", ".yml", ".toml", ".html", ".css", ".csv",
                    ):
                        try:
                            await self.ingester.ingest(
                                str(file_path), session_id="tool"
                            )
                            total_docs += 1
                            # Rough chunk estimate
                            size = file_path.stat().st_size
                            total_chunks += max(1, size // 2048)
                        except Exception as e:
                            logger.warning(f"Failed to ingest {file_path}: {e}")

                return IngestResult(
                    success=True,
                    path=str(resolved),
                    chunks=total_chunks,
                    documents=total_docs,
                )
            else:
                # Single file
                await self.ingester.ingest(str(resolved), session_id="tool")
                size = resolved.stat().st_size
                chunks = max(1, size // 2048)
                return IngestResult(
                    success=True,
                    path=str(resolved),
                    chunks=chunks,
                    documents=1,
                )
        except Exception as e:
            logger.exception(f"Ingest failed for {path}")
            return IngestResult(success=False, path=path, error=str(e))

    def format_result(self, result: IngestResult) -> str:
        """Format for LLM/user output."""
        if not result.success:
            return f"[INGEST ERROR] {result.error}"

        if result.documents > 1:
            return (
                f"Ingested {result.documents} documents from {result.path} "
                f"({result.chunks} chunks stored in memory)"
            )
        return (
            f"Ingested: {result.path} (~{result.chunks} chunks stored in memory)"
        )
