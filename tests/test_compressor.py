"""Tests for context/compressor.py — HistoryCompressor."""
from __future__ import annotations

import pytest

from core.context.compressor import HistoryCompressor


@pytest.fixture
def compressor():
    return HistoryCompressor(llm=None)


# ── count_tokens ─────────────────────────────────────────────────────────────

def test_count_tokens_string(compressor):
    assert compressor.count_tokens("hello world") >= 1


def test_count_tokens_empty(compressor):
    assert compressor.count_tokens("") >= 1


def test_count_tokens_list(compressor):
    msgs = [{"role": "user", "content": "hello " * 100}]
    assert compressor.count_tokens(msgs) > 10


def test_count_tokens_non_string(compressor):
    assert compressor.count_tokens(42) == 1


# ── compress ─────────────────────────────────────────────────────────────────

def test_compress_empty(compressor):
    assert compressor.compress([], 1000) == []


def test_compress_fits_budget(compressor):
    msgs = [{"role": "user", "content": "short"}]
    result = compressor.compress(msgs, 10000)
    assert result == msgs


def test_compress_over_budget(compressor):
    msgs = [{"role": "user", "content": "x" * 400} for _ in range(20)]
    result = compressor.compress(msgs, 500)
    assert len(result) < len(msgs)


# ── truncate_to_budget ───────────────────────────────────────────────────────

def test_truncate_keeps_at_least_one(compressor):
    msgs = [{"role": "user", "content": "x" * 10000}]
    result = compressor.truncate_to_budget(msgs, 10)
    assert len(result) == 1


def test_truncate_empty(compressor):
    assert compressor.truncate_to_budget([], 100) == []


# ── summarize_batch ──────────────────────────────────────────────────────────

def test_summarize_batch_fallback(compressor):
    """Without LLM, should produce a concatenated snippet summary."""
    msgs = [
        {"role": "user", "content": "Tell me about Python"},
        {"role": "assistant", "content": "Python is a programming language..."},
    ]
    summary = compressor.summarize_batch(msgs)
    assert "user:" in summary.lower() or "assistant:" in summary.lower()


# ── split_into_batches ───────────────────────────────────────────────────────

def test_split_into_batches_exact(compressor):
    items = list(range(10))
    batches = compressor.split_into_batches(items, 5)
    assert len(batches) == 2
    assert batches[0] == [0, 1, 2, 3, 4]


def test_split_into_batches_remainder(compressor):
    items = list(range(7))
    batches = compressor.split_into_batches(items, 3)
    assert len(batches) == 3
    assert batches[-1] == [6]
