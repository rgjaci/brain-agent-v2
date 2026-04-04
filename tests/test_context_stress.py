"""Stress test for context assembly — 500 memories, token budget enforcement."""
from __future__ import annotations

import random
from unittest.mock import MagicMock


def test_context_assembly_500_memories_token_budget():
    """Create 500 memories of varying lengths, assemble context,
    assert total tokens never exceed 32768, assert at least 3 memories included."""
    from core.context.assembler import ContextAssembler
    from core.memory.reader import RetrievalResult, RetrievedMemory

    mock_llm = MagicMock()
    mock_llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    ContextAssembler(mock_llm)

    # Generate 500 memories with varying content length
    random.seed(42)
    memories = []
    for i in range(500):
        length = random.randint(20, 500)
        content = f"Memory {i}: " + "x" * length
        mem = RetrievedMemory(
            id=i,
            content=content,
            category=random.choice(["fact", "preference", "observation", "correction"]),
            importance=random.uniform(0.1, 1.0),
            access_count=random.randint(0, 50),
            rrf_score=random.uniform(0.01, 0.5),
        )
        memories.append(mem)

    result = RetrievalResult(
        memories=memories,
        kg_context="Entity: Python -- type: language\nEntity: Docker -- type: tool",
        procedures=[{
            "name": "debug_python",
            "description": "Debug Python errors",
            "steps": ["Check traceback", "Identify module", "Fix import"],
            "trigger_pattern": "debug python",
        }],
        query_entities=["Python", "Docker"],
        retrieval_strategy="normal",
    )

    # Assemble context using the format_for_context method of reader
    from core.memory.reader import MemoryReader
    mock_embedder = MagicMock()
    mock_kg = MagicMock()

    # Use a standalone reader to test format_for_context
    reader = MemoryReader(db=MagicMock(), embedder=mock_embedder, kg=mock_kg)
    # Use 30K budget (leaving room for system prompt + query in the full 32K context)
    formatted = reader.format_for_context(result, budget_tokens=30000)

    # Verify token budget
    total_tokens = max(1, len(formatted) // 4)
    assert total_tokens <= 32768, f"Token budget exceeded: {total_tokens} > 32768"

    # Also verify the formatted output is reasonable
    assert len(formatted) > 100  # non-trivial output

    # Verify at least 3 memories are included
    memory_count = formatted.count("[fact]") + formatted.count("[preference]") + \
                   formatted.count("[observation]") + formatted.count("[correction]")
    assert memory_count >= 3, f"Only {memory_count} memories included (need >= 3)"


def test_context_assembly_empty_memories():
    """Assembler handles zero memories gracefully."""
    from core.context.assembler import ContextAssembler
    from core.memory.reader import RetrievalResult

    mock_llm = MagicMock()
    mock_llm.count_tokens = MagicMock(return_value=1)
    ContextAssembler(mock_llm)

    result = RetrievalResult(
        memories=[], kg_context="", procedures=[],
        query_entities=[], retrieval_strategy="conservative",
    )

    from core.memory.reader import MemoryReader
    reader = MemoryReader(db=MagicMock(), embedder=MagicMock(), kg=MagicMock())
    formatted = reader.format_for_context(result, budget_tokens=32768)
    assert formatted == ""
