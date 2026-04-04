"""Tests for memory/documents.py — DocumentIngester."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.documents import (
    CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    DocumentIngester,
)


@pytest.fixture
def mock_writer():
    w = MagicMock()
    w.process_document_chunk = AsyncMock()
    return w


@pytest.fixture
def ingester(db, mock_writer):
    return DocumentIngester(db, mock_writer)


# ── smart_chunk ──────────────────────────────────────────────────────────────

def test_smart_chunk_empty(ingester):
    assert ingester.smart_chunk("", "/tmp/test.txt") == []


def test_smart_chunk_small_file(ingester):
    chunks = ingester.smart_chunk("Hello world", "/tmp/test.txt")
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world"
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1


def test_smart_chunk_python(ingester):
    code = "def foo():\n    pass\n\ndef bar():\n    return 42\n\nclass Baz:\n    pass\n"
    chunks = ingester.smart_chunk(code, "/tmp/test.py")
    assert len(chunks) >= 1
    assert chunks[0].metadata["language"] == "py"


def test_smart_chunk_markdown(ingester):
    md = "# Heading 1\nSome content\n\n## Heading 2\nMore content\n"
    chunks = ingester.smart_chunk(md, "/tmp/readme.md")
    assert len(chunks) >= 1
    assert chunks[0].metadata["extension"] == ".md"


def test_smart_chunk_large_file(ingester):
    """Large content should be split into multiple chunks."""
    content = "word " * 2000  # ~10000 chars
    chunks = ingester.smart_chunk(content, "/tmp/big.txt")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= MAX_CHUNK_SIZE + CHUNK_OVERLAP + 50


# ── compute_hash ─────────────────────────────────────────────────────────────

def test_compute_hash_deterministic(ingester):
    h1 = ingester.compute_hash("hello")
    h2 = ingester.compute_hash("hello")
    assert h1 == h2
    assert len(h1) == 64


def test_compute_hash_different(ingester):
    h1 = ingester.compute_hash("hello")
    h2 = ingester.compute_hash("world")
    assert h1 != h2


# ── dedup (is_already_ingested / mark_ingested) ─────────────────────────────

def test_dedup_not_ingested(ingester):
    assert ingester.is_already_ingested("abc123") is False


def test_dedup_after_mark(ingester):
    ingester.mark_ingested("/tmp/test.txt", "abc123")
    assert ingester.is_already_ingested("abc123") is True


# ── ingest_file ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_file_not_found(ingester):
    result = await ingester.ingest_file("/nonexistent/file.txt", "sess1")
    assert result["status"] == "error"
    assert result["chunks"] == 0


@pytest.mark.asyncio
async def test_ingest_file_unsupported(ingester):
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"data")
        f.flush()
        result = await ingester.ingest_file(f.name, "sess1")
    os.unlink(f.name)
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_ingest_file_success(ingester, mock_writer):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("This is test content for ingestion.")
        f.flush()
        result = await ingester.ingest_file(f.name, "sess1")
    os.unlink(f.name)
    assert result["status"] == "ok"
    assert result["chunks"] >= 1
    assert mock_writer.process_document_chunk.called


@pytest.mark.asyncio
async def test_ingest_file_dedup(ingester, mock_writer):
    """Ingesting the same file twice should skip the second time."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("Dedup test content")
        f.flush()
        r1 = await ingester.ingest_file(f.name, "sess1")
        r2 = await ingester.ingest_file(f.name, "sess1")
    os.unlink(f.name)
    assert r1["chunks"] >= 1
    assert r2["chunks"] == 0  # skipped


# ── ingest_directory ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_directory_not_found(ingester):
    result = await ingester.ingest_directory("/nonexistent/dir", "sess1")
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_ingest_directory_success(ingester, mock_writer):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a supported file
        p = os.path.join(tmpdir, "test.txt")
        with open(p, "w") as f:
            f.write("Directory ingestion test")
        result = await ingester.ingest_directory(tmpdir, "sess1")
    assert result["files"] >= 1
    assert result["status"] == "ok"
