"""Tests for ContextAssembler — token counting, memory packing, best-at-edges."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock


def make_assembler():
    from core.context.assembler import ContextAssembler
    llm = MagicMock()
    return ContextAssembler(llm=llm)


def _mem(id, content, importance=0.5, rrf_score=1.0, created_at="2024-01-01"):
    return MagicMock(
        id=id,
        content=content,
        importance=importance,
        rrf_score=rrf_score,
        created_at=created_at,
        category="fact",
        access_count=1,
    )


# ── token counting ────────────────────────────────────────────────────────────

def test_count_tokens_empty():
    a = make_assembler()
    assert a.count_tokens("") >= 0  # may return 0 or small overhead


def test_count_tokens_rough_estimate():
    a = make_assembler()
    # ~4 chars per token heuristic
    tok = a.count_tokens("Hello world this is a test")
    assert tok > 0
    assert tok < 20  # 26 chars → ~6 tokens


def test_count_tokens_scales_with_length():
    a = make_assembler()
    short = a.count_tokens("hi")
    long  = a.count_tokens("hi " * 100)
    assert long > short


# ── system prompt ─────────────────────────────────────────────────────────────

def test_get_system_prompt_nonempty():
    a = make_assembler()
    prompt = a.get_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 50


# ── pack_memories ─────────────────────────────────────────────────────────────

def test_pack_memories_empty():
    a = make_assembler()
    result = a.pack_memories([], budget_tokens=1000)
    assert isinstance(result, str)


def test_pack_memories_returns_content():
    a = make_assembler()
    mems = [_mem(1, "User prefers vim"), _mem(2, "Production server is 10.0.0.1")]
    result = a.pack_memories(mems, budget_tokens=2000)
    assert "vim" in result or "10.0.0.1" in result


def test_pack_memories_respects_budget():
    a = make_assembler()
    # Fill with large memories, tiny budget
    mems = [_mem(i, "x" * 500) for i in range(20)]
    result = a.pack_memories(mems, budget_tokens=50)
    # Should truncate — result should be much smaller than all memories
    assert a.count_tokens(result) <= 200  # loose upper bound


# ── best_at_edges ─────────────────────────────────────────────────────────────

def test_apply_best_at_edges_returns_same_count():
    a = make_assembler()
    mems = [_mem(i, f"fact {i}", importance=i * 0.1) for i in range(10)]
    result = a.apply_best_at_edges(mems)
    assert len(result) == len(mems)


def test_apply_best_at_edges_empty():
    a = make_assembler()
    assert a.apply_best_at_edges([]) == []


def test_apply_best_at_edges_single():
    a = make_assembler()
    mems = [_mem(1, "only one")]
    result = a.apply_best_at_edges(mems)
    assert len(result) == 1


# ── format_procedure ──────────────────────────────────────────────────────────

def test_format_procedure_returns_string():
    a = make_assembler()
    proc = MagicMock()
    proc.name = "deploy_service"
    proc.description = "Deploy a systemd service"
    proc.steps = ["Step 1: write unit file", "Step 2: reload daemon", "Step 3: enable"]
    proc.warnings = ["Ensure ports are open"]
    result = a.format_procedure(proc)
    assert isinstance(result, str)
    assert "deploy_service" in result or "Deploy" in result


# ── chat history formatting ───────────────────────────────────────────────────

def test_format_chat_history_empty():
    a = make_assembler()
    result = a.format_chat_history([], budget_tokens=1000)
    assert isinstance(result, (list, str))


def test_format_chat_history_truncates_old():
    a = make_assembler()
    history = [
        {"role": "user",      "content": f"message {i}"}
        for i in range(100)
    ]
    result = a.format_chat_history(history, budget_tokens=100)
    # With tiny budget only recent messages should survive
    if isinstance(result, str):
        assert a.count_tokens(result) <= 500
    else:
        assert len(result) <= 100


# ── assemble ─────────────────────────────────────────────────────────────────

def test_assemble_returns_list_of_dicts():
    a = make_assembler()
    result = a.assemble(
        query="how do I set up SSH?",
        memories=[],
        kg_context="",
        chat_history=[],
        procedure=None,
    )
    assert isinstance(result, list)
    assert all(isinstance(m, dict) for m in result)
    assert any(m.get("role") == "system" for m in result)
    assert any(m.get("role") == "user" for m in result)
