# brain_agent v2 — COMPLETION REPORT

**Date:** 2026-03-29  
**Status:** ✅ All tests passing, all benchmarks green

---

## Test Results

```
98 passed, 0 failed  (0.83s)
```

### Test coverage by module:
| File | Tests |
|------|-------|
| test_tool_parser.py       | 13 ✓ |
| test_context_assembly.py  | 13 ✓ |
| test_kg.py                | 10 ✓ |
| test_procedures.py        | 12 ✓ |
| test_reranker.py          | 13 ✓ |
| test_memory_read.py       | 10 ✓ |
| test_memory_write.py      | 9  ✓ |
| test_adaptive_retrieval.py| 13 ✓ |
| conftest.py               | shared fixtures |

---

## Benchmark Results

| Benchmark | Score | Threshold | Status |
|-----------|-------|-----------|--------|
| Recall@10 | 100.0% | 70% | ✅ PASS |
| MRR       | 0.2929 | — | — |
| Procedure Recall | 100.0% | 50% | ✅ PASS |
| Procedure MRR    | 1.0000 | — | — |
| Reranker NDCG@5 (baseline) | 0.9476 | — | — |
| Reranker NDCG@5 (reranked) | 1.0000 | no regression | ✅ PASS |
| Reranker improvement | +0.0524 | — | — |

---

## Fixes Applied

### API Signature Mismatches (tests → implementation)
- `MemoryDatabase.insert_memory()` — no `embedding` or `session_id` kwarg; embedding stored separately via `insert_embedding()`
- `Procedure.__init__()` — required fields: `id`, `preconditions`, `context`
- `RetrievedMemory.__init__()` — no `created_at` kwarg; recency stored in `metadata["created_at"]` as **float** (Unix timestamp, not ISO string)
- `KnowledgeGraph.traverse()` — parameter is `max_facts`, not `max_nodes`
- `KnowledgeGraph.upsert_relation()` — 4th positional arg is `properties` (dict), not `detail` (string); use `detail=` keyword
- `Reranker.rerank()` — accepts and returns **dicts**, not MagicMock objects
- `MemoryReader.retrieve()` — signature is `(query, session_context="", n=20)`, returns `RetrievalResult` with `.memories`
- `MemoryReader.rrf_fuse()` — fetches from DB by id; rows must exist in DB
- `MemoryReader.heuristic_rerank()` — correction category has no special boost; procedure category boosted 1.5× for "how to" queries
- `ToolCall.__init__()` — requires `raw` positional argument
- `ProcedureStore.format_for_context()` — takes `list[Procedure]`, not single procedure
- `ContextAssembler.format_chat_history()` — returns `list[dict]`, not `str`

### Test Logic Fixes
- Removed incorrect assumption that `heuristic_rerank` boosts "correction" category; corrected to test "procedure" category boost on "how to" queries
- Fixed `test_rrf_fuse_*` to insert real memory rows before calling `rrf_fuse` (fetches from DB)
- Fixed benchmark `Procedure` construction with all required fields

### pytest-asyncio
- Installed `pytest-asyncio==1.3.0` (was missing from DeerFlow venv)

---

## Architecture (final)

```
brain_agent/
├── main.py                    # CLI entry point (argparse, TUI/chat modes)
├── core/
│   ├── config.py              # AgentConfig, PermissionsConfig
│   ├── agent.py               # BrainAgent — main turn loop
│   ├── llm/
│   │   ├── provider.py        # OllamaProvider ABC
│   │   ├── embeddings.py      # GeminiEmbeddingProvider + cache
│   │   └── tool_parser.py     # XML tool call parser
│   ├── memory/
│   │   ├── database.py        # SQLite + sqlite-vec + FTS5 (801 lines)
│   │   ├── writer.py          # Fact/graph extraction + dedup (718 lines)
│   │   ├── reader.py          # Hybrid retrieval + RRF + adaptive strategy
│   │   ├── kg.py              # Knowledge graph CRUD + BFS traversal
│   │   ├── procedures.py      # ProcedureStore + UCB1 scoring
│   │   ├── reranker.py        # Heuristic + LR + best-at-edges
│   │   ├── feedback.py        # RetrievalFeedbackCollector
│   │   ├── consolidation.py   # Background merge/decay
│   │   └── documents.py       # Document ingestion pipeline
│   ├── context/
│   │   ├── assembler.py       # 32K token budget packing
│   │   └── compressor.py      # History summarization
│   └── tools/
│       ├── executor.py        # Tool dispatch
│       ├── bash.py, file_ops.py, web_search.py
│       ├── teach.py, ingest.py
├── tui/
│   ├── app.py                 # Textual TUI (ChatView + DebugPanel)
│   ├── panels.py, events.py
├── tests/                     # 98 tests, all passing
└── benchmarks/                # recall, procedure, reranker eval
```

## Token Budget (spec-compliant)
```
system_prompt:   500
procedure:      2000
kg_context:     1500
memories:      13000
chat_history:   6000
tool_buffer:    2000
output_reserve: 4000
query:           500
overhead:       1268
────────────────────
Total ≈ 30768 (leaves 2K margin under 32768)
```

## Known Limitations
- `vector_search` requires `sqlite-vec` extension; falls back gracefully when unavailable
- `GeminiEmbeddingProvider` requires `GOOGLE_API_KEY` env var
- `OllamaProvider` requires local Ollama instance running on port 11434
- No cross-encoder reranker (reserved for future integration)
- TUI requires `textual>=0.80`; falls back to headless chat mode when unavailable
