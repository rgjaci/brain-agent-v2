# Testing Guide

Brain Agent v2 has a comprehensive test suite with **98 tests** covering all major components.

## Running Tests

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_memory_read.py -v

# Run with coverage
pytest --cov=core --cov-report=term-missing

# Run with timeout (default: 30s per test)
pytest --timeout=30

# Run async tests
pytest --asyncio-mode=auto
```

## Test Structure

| Test File | Tests | Coverage |
|---|---|---|
| `test_tool_parser.py` | 13 | XML parsing, Pydantic validation, edge cases |
| `test_context_assembly.py` | 13 | Token budgeting, best-at-edges, message formatting |
| `test_kg.py` | 10 | Entity/relation CRUD, BFS traversal |
| `test_procedures.py` | 12 | Procedure CRUD, UCB1 scoring, relevance searches |
| `test_reranker.py` | 13 | Heuristic scoring, LR blending, best-at-edges |
| `test_memory_read.py` | 10 | Hybrid retrieval, RRF fusion, adaptive strategy |
| `test_memory_write.py` | 9 | Fact extraction, dedup, KG updates |
| `test_adaptive_retrieval.py` | 13 | Query classification, retrieval planning |
| `test_consolidation.py` | — | Duplicate merging, decay, promotion |
| `test_dream.py` | — | Abstraction, contradiction, pattern detection |
| `test_reasoning.py` | — | Gap analysis, cross-domain, rule inference |
| `test_documents.py` | — | Format-aware chunking, dedup |
| `test_feedback.py` | — | Feedback collection, LR training |
| `test_integration.py` | — | End-to-end agent turns |
| `test_mcp_server.py` | — | MCP tool exposure |
| `test_tui.py` | — | TUI composition |
| `test_context_stress.py` | — | Token budget edge cases |
| `test_database_extras.py` | — | DB schema, triggers |
| `test_memory_links.py` | — | Memory relationship tracking |
| `test_phase3.py` | — | Phase 3 feature integration |

## Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
timeout = 30
```

## Fixtures

Shared fixtures are in `tests/conftest.py`:

```python
# Common fixtures include:
# - db: In-memory MemoryDatabase
# - kg: KnowledgeGraph with test data
# - embedder: Mock embedding provider
# - llm: Mock LLM provider
# - agent: BrainAgent with mocked dependencies
```

## Writing Tests

### Database Tests

Use the in-memory database fixture:

```python
def test_insert_memory(db):
    db.execute(
        "INSERT INTO memories (content, category, source, importance, created_at) VALUES (?, ?, ?, ?, ?)",
        ("Test content", "fact", "test", 0.8, time.time()),
    )
    rows = db.execute("SELECT * FROM memories WHERE content = ?", ("Test content",))
    assert len(rows) == 1
    assert rows[0]["category"] == "fact"
```

### Knowledge Graph Tests

```python
def test_entity_crud(kg):
    entity = Entity(name="Python", entity_type="language", description="Programming language")
    kg.upsert_entity(entity)
    
    retrieved = kg.get_entity("Python")
    assert retrieved is not None
    assert retrieved.entity_type == "language"
```

### Reranker Tests

```python
def test_heuristic_scoring(reranker):
    memory = {
        "id": 1,
        "content": "Test",
        "category": "fact",
        "importance": 0.8,
        "access_count": 5,
        "created_at": time.time(),
        "rrf_score": 0.9,
    }
    score = reranker._heuristic_score(memory, 0)
    assert score > 0.9  # Should be boosted by multipliers
```

### Procedure Tests

```python
def test_procedure_ucb(store):
    # Insert procedures
    proc = Procedure(
        id=None,
        name="test_proc",
        description="Test procedure",
        trigger_pattern="test",
        preconditions=[],
        steps=["Step 1"],
        warnings=[],
        context="",
        success_count=5,
        attempt_count=6,
    )
    store.save(proc)
    
    # Retrieve and check UCB scoring
    results = store.find_relevant("test")
    assert len(results) > 0
    assert results[0].name == "test_proc"
```

## Test Categories

### Unit Tests

Test individual components in isolation:
- `test_tool_parser.py` — Tool call parsing
- `test_reranker.py` — Reranking logic
- `test_kg.py` — Knowledge graph operations
- `test_procedures.py` — Procedure store

### Integration Tests

Test component interactions:
- `test_memory_read.py` — Full retrieval pipeline
- `test_memory_write.py` — Extraction and storage
- `test_integration.py` — End-to-end agent turns

### Stress Tests

Test edge cases and limits:
- `test_context_stress.py` — Token budget boundaries
- `test_database_extras.py` — Schema validation

## Mocking Strategy

The test suite uses extensive mocking to avoid external dependencies:

```python
from unittest.mock import MagicMock, AsyncMock

# Mock LLM provider
llm = MagicMock()
llm.generate.return_value = "Test response"

# Mock embedding provider
embedder = MagicMock()
embedder.embed.return_value = [0.1] * 768  # 768-dim vector

# Mock async methods
llm.generate_async = AsyncMock(return_value="Test response")
```

## Known API Signatures

From the COMPLETION.md, these are the correct API signatures:

| Method | Correct Signature | Notes |
|---|---|---|
| `MemoryDatabase.insert_memory()` | No `embedding` or `session_id` kwarg | Embedding stored separately via `insert_embedding()` |
| `Procedure.__init__()` | Requires `id`, `preconditions`, `context` | All are required fields |
| `RetrievedMemory.__init__()` | No `created_at` kwarg | Recency stored in `metadata["created_at"]` as float |
| `KnowledgeGraph.traverse()` | Parameter is `max_facts` | Not `max_nodes` |
| `KnowledgeGraph.upsert_relation()` | 4th arg is `properties` (dict) | Use `detail=` keyword, not positional |

## Adding New Tests

1. Create a new file in `tests/` following the naming convention `test_<module>.py`
2. Use fixtures from `conftest.py` where possible
3. Mock external dependencies (LLM, embeddings, network)
4. Test both happy path and edge cases
5. Run `pytest tests/test_<module>.py -v` to verify

## CI Integration

Tests can be run in CI:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e '.[dev]'
      - run: pytest -v
```
