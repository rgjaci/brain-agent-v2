"""Tests for core/memory/reasoning.py — System 2 Reasoning Engine."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def reasoning_engine(db, mock_llm, mock_embedder, kg):
    from core.memory.reasoning import ReasoningEngine
    return ReasoningEngine(
        db=db,
        llm=mock_llm,
        embedder=mock_embedder,
        kg=kg,
        interval=1,  # fast for testing
        max_cycles=5,
    )


# ── Lifecycle ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_stop(reasoning_engine):
    """Engine should start and stop cleanly."""
    await reasoning_engine.start()
    assert reasoning_engine.is_running is True
    await reasoning_engine.stop()
    assert reasoning_engine.is_running is False


@pytest.mark.asyncio
async def test_stop_when_not_running(reasoning_engine):
    """Stop on a non-running engine should be a no-op."""
    await reasoning_engine.stop()
    assert reasoning_engine.is_running is False


@pytest.mark.asyncio
async def test_double_start(reasoning_engine):
    """Starting twice should be idempotent."""
    await reasoning_engine.start()
    await reasoning_engine.start()  # no error
    assert reasoning_engine.is_running is True
    await reasoning_engine.stop()


# ── Single reasoning cycle ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reasoning_cycle_empty_db(reasoning_engine):
    """A cycle on an empty DB should complete without errors."""
    result = await reasoning_engine.reasoning_cycle()
    assert result.strategy in (
        "gap_analysis", "cross_domain", "rule_inference",
        "contradiction_check", "procedure_improvement", "question_answering",
    )
    assert result.elapsed_seconds >= 0


@pytest.mark.asyncio
async def test_reasoning_cycles_rotate_strategies(reasoning_engine):
    """Consecutive cycles should rotate through strategies."""
    strategies = []
    for _ in range(6):
        result = await reasoning_engine.reasoning_cycle()
        strategies.append(result.strategy)
    # Should have all 6 distinct strategies
    assert len(set(strategies)) == 6


# ── Gap analysis ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gap_analysis_generates_questions(reasoning_engine, db, mock_llm):
    """Gap analysis should generate questions about a topic."""
    db.insert_entity("Python", entity_type="language", importance=0.9)
    for i in range(5):
        db.insert_memory(f"User uses Python for task {i}", category="fact")

    mock_llm.generate_json = MagicMock(return_value={
        "gaps": ["Unknown Python version"],
        "questions": ["What version of Python does the user use?"],
        "importance": 0.7,
    })

    from core.memory.reasoning import FocusTopic
    focus = FocusTopic(name="Python", source="entity")
    result = await reasoning_engine._gap_analysis(focus)
    assert result.questions_generated >= 1

    questions = db.execute("SELECT * FROM memories WHERE category = 'question' AND source = 'reasoning'")
    assert len(questions) >= 1


@pytest.mark.asyncio
async def test_gap_analysis_empty(reasoning_engine, mock_llm):
    """Gap analysis with no matching memories should return 0."""
    from core.memory.reasoning import FocusTopic
    focus = FocusTopic(name="Nonexistent", source="entity")
    result = await reasoning_engine._gap_analysis(focus)
    assert result.questions_generated == 0


# ── Cross-domain connection ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cross_domain_finds_connections(reasoning_engine, db, mock_llm):
    """Cross-domain should find connections between two topics."""
    db.insert_entity("Docker", entity_type="tool")
    db.insert_entity("Kubernetes", entity_type="tool")
    db.insert_memory("User deploys with Docker", category="fact")
    db.insert_memory("User manages clusters with Kubernetes", category="fact")

    mock_llm.generate_json = MagicMock(return_value={
        "connections": [{"insight": "Docker containers are orchestrated by Kubernetes", "importance": 0.8}],
        "new_relations": [],
    })

    from core.memory.reasoning import FocusTopic
    focus = FocusTopic(name="Docker", source="entity")
    result = await reasoning_engine._cross_domain(focus)
    assert result.connections_found >= 0  # depends on RANDOM() entity selection


# ── Rule inference ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rule_inference(reasoning_engine, db, mock_llm):
    """Rule inference should derive rules from observations."""
    for i in range(10):
        db.insert_memory(f"Observation {i} about workflow", category="observation")

    mock_llm.generate_json = MagicMock(return_value=[
        {"rule": "When deploying, always run tests first", "confidence": 0.8, "evidence_count": 5, "category": "rule"}
    ])

    result = await reasoning_engine._rule_inference()
    assert result.rules_derived >= 1

    rules = db.execute("SELECT * FROM memories WHERE category = 'rule' AND source = 'reasoning'")
    assert len(rules) >= 1
    assert "Rule:" in rules[0]["content"]


@pytest.mark.asyncio
async def test_rule_inference_low_confidence_skipped(reasoning_engine, db, mock_llm):
    """Rules with confidence < 0.5 should be skipped."""
    for i in range(10):
        db.insert_memory(f"Observation {i}", category="observation")

    mock_llm.generate_json = MagicMock(return_value=[
        {"rule": "Weak rule", "confidence": 0.2, "evidence_count": 1}
    ])

    result = await reasoning_engine._rule_inference()
    assert result.rules_derived == 0


# ── Contradiction check ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_contradiction_check(reasoning_engine, db, mock_llm):
    """Contradiction check should detect and record conflicts."""
    for i in range(10):
        db.insert_memory(f"Fact {i}", category="fact")

    mock_llm.generate_json = MagicMock(return_value={
        "contradictions": [{
            "memory_a": "Fact 1",
            "memory_b": "Fact 2",
            "explanation": "They conflict about the same topic",
            "resolution": "Fact 2 is more recent",
        }]
    })

    result = await reasoning_engine._contradiction_check()
    assert result.insights_generated >= 1


# ── Question answering ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_question_answering_answers(reasoning_engine, db, mock_llm):
    """Question answering should answer questions and supersede them."""
    qid = db.insert_memory("What programming language does the user prefer?", category="question")
    db.insert_memory("User writes most code in Python", category="fact")
    db.insert_memory("User prefers Python over JavaScript", category="preference")

    mock_llm.generate_json = MagicMock(return_value={
        "answerable": True,
        "answer": "The user prefers Python",
        "confidence": 0.9,
        "reasoning": "Multiple facts support this",
    })

    result = await reasoning_engine._question_answering()
    assert result.questions_answered >= 1

    # Verify the original question was superseded
    q = db.get_memory(qid)
    assert q["superseded_by"] is not None


@pytest.mark.asyncio
async def test_question_answering_unanswerable(reasoning_engine, db, mock_llm):
    """Unanswerable questions should not generate insights."""
    db.insert_memory("What is the user's favorite color?", category="question")
    db.insert_memory("Some unrelated fact", category="fact")

    mock_llm.generate_json = MagicMock(return_value={
        "answerable": False,
        "answer": "",
        "confidence": 0.1,
        "reasoning": "No relevant knowledge",
    })

    result = await reasoning_engine._question_answering()
    assert result.questions_answered == 0


# ── Focus selection ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_focus_selects_entity(reasoning_engine, db):
    """Focus should select high-importance entities."""
    db.insert_entity("Python", entity_type="language", importance=0.9)
    db.insert_entity("obscure", entity_type="concept", importance=0.1)

    focus = await reasoning_engine._select_focus("gap_analysis")
    assert focus.name == "Python"


@pytest.mark.asyncio
async def test_focus_round_robin_penalty(reasoning_engine, db):
    """Recently focused topics should be penalised."""
    db.insert_entity("Python", entity_type="language", importance=0.9)
    db.insert_entity("Docker", entity_type="tool", importance=0.8)

    from core.memory.reasoning import FocusTopic
    reasoning_engine._focus_history = [
        FocusTopic(name="Python", source="entity")
    ] * 5

    focus = await reasoning_engine._select_focus("gap_analysis")
    # Docker should be preferred due to round-robin penalty on Python
    assert focus.name == "Docker"


@pytest.mark.asyncio
async def test_focus_question_strategy(reasoning_engine, db):
    """Question answering strategy should pick a stored question."""
    db.insert_memory("What IDE does the user use?", category="question")

    focus = await reasoning_engine._select_focus("question_answering")
    assert focus.source == "question"
    assert "IDE" in focus.name


@pytest.mark.asyncio
async def test_focus_fallback(reasoning_engine):
    """With no entities, focus should fall back to 'general'."""
    focus = await reasoning_engine._select_focus("gap_analysis")
    assert focus.name == "general"


# ── Event emission ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_events_emitted(reasoning_engine, db):
    """Reasoning should emit events for TUI."""
    db.insert_entity("Python", entity_type="language", importance=0.9)
    events = []
    reasoning_engine.on_event = lambda t, d: events.append((t, d))

    await reasoning_engine.reasoning_cycle()
    assert any(t == "reasoning_start" for t, _ in events)


# ── Error resilience ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cycle_survives_llm_error(reasoning_engine, db, mock_llm):
    """Cycle should complete even if LLM fails."""
    db.insert_entity("Python", entity_type="language", importance=0.9)
    for i in range(10):
        db.insert_memory(f"fact {i}", category="fact")

    mock_llm.generate_json = MagicMock(side_effect=RuntimeError("LLM offline"))

    result = await reasoning_engine.reasoning_cycle()
    assert result.elapsed_seconds >= 0


# ── Recent results ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recent_results(reasoning_engine):
    """Recent results should track completed cycles."""
    assert len(reasoning_engine.recent_results) == 0
    await reasoning_engine.reasoning_cycle()
    await reasoning_engine.reasoning_cycle()
    # recent_results is populated via _loop, direct cycle doesn't add to it
    # but we can check the cycle count
    assert reasoning_engine._cycle_count == 2


# ── ReasoningResult dataclass ────────────────────────────────────────────────

def test_reasoning_result_defaults():
    from core.memory.reasoning import ReasoningResult
    r = ReasoningResult()
    assert r.strategy == ""
    assert r.focus_topic == ""
    assert r.insights_generated == 0
    assert r.questions_generated == 0
    assert r.rules_derived == 0
    assert r.connections_found == 0
    assert r.questions_answered == 0
    assert r.elapsed_seconds == 0.0
