"""Tests for memory/feedback.py — RetrievalFeedbackCollector."""
from __future__ import annotations
import time
import pytest
from core.memory.feedback import RetrievalFeedbackCollector, RetrievalEvent


@pytest.fixture
def collector(db):
    return RetrievalFeedbackCollector(db)


# ── record_retrieval ─────────────────────────────────────────────────────────

def test_record_retrieval_creates_events(collector):
    mems = [
        {"id": 1, "rrf_score": 0.9, "access_count": 5, "importance": 0.7,
         "category": "fact", "created_at": time.time()},
        {"id": 2, "rrf_score": 0.5, "access_count": 1, "importance": 0.3,
         "category": "general", "created_at": time.time()},
    ]
    collector.record_retrieval("test query", mems, "sess1")
    assert len(collector._buffer["sess1"]) == 2


def test_record_retrieval_skips_no_id(collector):
    mems = [{"rrf_score": 0.5}]  # no "id" key
    collector.record_retrieval("query", mems, "sess1")
    assert len(collector._buffer.get("sess1", [])) == 0


def test_record_retrieval_multiple_sessions(collector):
    collector.record_retrieval("q1", [{"id": 1, "rrf_score": 0.5}], "s1")
    collector.record_retrieval("q2", [{"id": 2, "rrf_score": 0.5}], "s2")
    assert len(collector._buffer) == 2


# ── extract_features ─────────────────────────────────────────────────────────

def test_extract_features_keys(collector):
    mem = {"id": 1, "rrf_score": 0.8, "access_count": 3, "importance": 0.6,
           "category": "fact", "created_at": time.time()}
    features = collector.extract_features(mem, "test query about fact", 0)
    expected_keys = {"rank_normalized", "rrf_score", "access_count_log",
                     "importance", "age_days", "category_match", "query_len"}
    assert set(features.keys()) == expected_keys


def test_extract_features_category_match(collector):
    mem = {"id": 1, "rrf_score": 0.5, "category": "fact", "created_at": time.time()}
    features = collector.extract_features(mem, "some fact here", 0)
    assert features["category_match"] == 1.0


def test_extract_features_no_category_match(collector):
    mem = {"id": 1, "rrf_score": 0.5, "category": "procedure", "created_at": time.time()}
    features = collector.extract_features(mem, "hello world", 0)
    assert features["category_match"] == 0.0


def test_extract_features_rank_normalized(collector):
    mem = {"id": 1, "rrf_score": 0.5, "created_at": time.time()}
    features = collector.extract_features(mem, "query", 10)
    assert features["rank_normalized"] == 0.5  # 10/20


# ── record_reference ─────────────────────────────────────────────────────────

def test_record_reference_marks_events(collector):
    mems = [{"id": 1, "rrf_score": 0.9}, {"id": 2, "rrf_score": 0.5}]
    collector.record_retrieval("q", mems, "s1")
    collector.record_reference("s1", [1])
    events = collector._buffer["s1"]
    assert events[0].was_referenced is True
    assert events[1].was_referenced is False


# ── get_training_data ────────────────────────────────────────────────────────

def test_get_training_data_basic(collector):
    mems = [
        {"id": 1, "rrf_score": 0.9, "access_count": 5, "importance": 0.7,
         "category": "fact", "created_at": time.time()},
    ]
    collector.record_retrieval("q", mems, "s1")
    collector.record_reference("s1", [1])
    X, y = collector.get_training_data()
    assert len(X) == 1
    assert y == [1]


def test_get_training_data_clears_buffer(collector):
    mems = [{"id": 1, "rrf_score": 0.5, "created_at": time.time()}]
    collector.record_retrieval("q", mems, "s1")
    collector.get_training_data()
    assert len(collector._buffer) == 0


# ── train_logistic_regression ────────────────────────────────────────────────

def test_train_lr_insufficient_data(collector):
    """Training with < 10 samples returns None."""
    mems = [{"id": i, "rrf_score": 0.5, "access_count": i, "importance": 0.5,
             "category": "fact", "created_at": time.time()} for i in range(5)]
    collector.record_retrieval("q", mems, "s1")
    collector.record_reference("s1", [0, 1])
    result = collector.train_logistic_regression()
    assert result is None


def test_train_lr_sufficient_data(collector):
    """Training with >= 10 samples returns weights dict."""
    for s in range(3):
        mems = [{"id": s * 10 + i, "rrf_score": 0.1 * i, "access_count": i,
                 "importance": 0.5, "category": "fact", "created_at": time.time()}
                for i in range(5)]
        collector.record_retrieval(f"query {s}", mems, f"session-{s}")
        collector.record_reference(f"session-{s}", [s * 10, s * 10 + 1])
    result = collector.train_logistic_regression()
    assert result is not None
    assert "weights" in result
    assert "bias" in result
    assert result["n_samples"] == 15


# ── persist / load weights ───────────────────────────────────────────────────

def test_persist_and_load_weights(collector):
    weights = {"weights": {"rrf_score": 0.5}, "bias": 0.1, "loss": 0.3, "n_samples": 100}
    collector.persist_weights(weights)
    loaded = collector.load_weights()
    assert loaded is not None
    assert loaded["bias"] == 0.1


def test_load_weights_empty(collector):
    result = collector.load_weights()
    assert result is None
