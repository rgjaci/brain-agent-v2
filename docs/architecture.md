# Architecture Overview

Brain Agent v2 is a **memory-first AI agent** built on the thesis that a small model (4B parameters) with an exceptional memory system can match frontier models on tasks where accumulated knowledge matters.

> **The LLM is a processor. Memory is the intelligence.**

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐ │
│  │   CLI    │  │   TUI    │  │  MCP     │  │  Direct API         │ │
│  │ main.py  │  │ tui/app  │  │ server/  │  │  BrainAgent class   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬────────────┘ │
└───────┼─────────────┼─────────────┼──────────────────┼──────────────┘
        │             │             │                  │
        ▼             ▼             ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        BrainAgent (core/agent.py)                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Main Turn Loop                             │   │
│  │                                                               │   │
│  │  1. Store user message in conversation history                │   │
│  │  2. Adaptive Retrieval (MemoryReader)                         │   │
│  │  3. Context Assembly (32K budget, best-at-edges)              │   │
│  │  4. LLM Generation + Tool Loop (max 10 iterations)            │   │
│  │  5. Store assistant response                                   │   │
│  │  6. Async: Extract knowledge (facts + entities + procedures)  │   │
│  │  7. Retrieval feedback + procedure success tracking           │   │
│  │  8. Consolidation (every 10 turns, idle >300s)                │   │
│  │  9. AutoDream (every 50 turns)                                │   │
│  │ 10. System 2 Reasoning (background cycle)                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
        │             │             │                  │
        ▼             ▼             ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Context     │ │  LLM Layer   │ │  Memory      │ │  Tools       │
│  System      │ │              │ │  System      │ │              │
│              │ │  provider.py │ │  database.py │ │  executor.py │
│  assembler.py│ │  embeddings  │ │  reader.py   │ │  bash.py     │
│  compressor  │ │  tool_parser │ │  writer.py   │ │  file_ops.py │
└──────────────┘ └──────────────┘ │  kg.py       │ │  web_search  │
                                  │  procedures  │ │  teach.py    │
                                  │  reranker.py │ │  ingest.py   │
                                  │  feedback.py │ └──────────────┘
                                  │  consolid.   │
                                  │  dream.py    │
                                  │  reasoning.py│
                                  │  documents.py│
                                  └──────────────┘
```

## Component Details

### Entry Points

| File | Description |
|---|---|
| `main.py` | CLI entry point with argparse. Supports multiple modes: TUI, headless chat, bootstrap, teach, recall, ingest, stats, and MCP server. Gracefully degrades when LLM/embedder unavailable. |
| `server/mcp_server.py` | Exposes the memory system via Model Context Protocol (MCP) for integration with external agents like Claude Code. Provides tools: `memory_store`, `memory_recall`, `kg_traverse`, `dream_trigger`. |

### Core Agent (`core/agent.py`)

The `BrainAgent` class is the central orchestrator. It manages the full turn lifecycle:

1. **Message Storage** — Stores user input in the conversation history
2. **Adaptive Retrieval** — Decides retrieval strategy (skip/conservative/normal/aggressive) based on query type
3. **Context Assembly** — Packs retrieved memories into the 32K token window with best-at-edges ordering
4. **LLM + Tool Loop** — Generates response, parses tool calls, executes tools, loops up to 10 iterations
5. **Response Storage** — Stores the final assistant response
6. **Async Knowledge Extraction** — Fire-and-forget extraction of facts, entities, relations, and procedures
7. **Feedback Collection** — Tracks retrieval effectiveness for reranker training
8. **Consolidation** — Periodic background cleanup (duplicate merging, importance decay)
9. **AutoDream** — LLM-powered abstraction and pattern detection
10. **System 2 Reasoning** — Background deliberate thinking cycle

All components are **optional** — the agent gracefully degrades if any subsystem is unavailable (no LLM, no embeddings, no tools).

### Configuration (`core/config.py`)

`AgentConfig` loads from YAML with environment variable overrides. Key sections:

- **Model**: LLM provider settings (Ollama/OpenRouter)
- **Embeddings**: Embedding provider configuration
- **Retrieval**: Top-K, RRF parameters, adaptive strategy thresholds
- **Token Budgets**: Context window allocation
- **Permissions**: Read/write/bash/network access controls
- **AutoDream**: Trigger conditions
- **System 2 Reasoning**: Background cycle settings

### Context System

| Component | File | Description |
|---|---|---|
| **ContextAssembler** | `core/context/assembler.py` | Packs the 32K token window with strict budget allocation. Implements "best-at-edges" ordering to mitigate lost-in-the-middle phenomenon. |
| **HistoryCompressor** | `core/context/compressor.py` | Progressive history compression. Keeps recent 5 messages verbatim, summarizes older in batches of 10. Falls back to truncation if no LLM available. |

### LLM Layer

| Component | File | Description |
|---|---|---|
| **LLMProvider** | `core/llm/provider.py` | Abstract base class with `OllamaProvider` (local) and `OpenRouterProvider` (cloud) implementations. Ollama supports JSON mode, token counting via tiktoken, and model pulling. |
| **EmbeddingProvider** | `core/llm/embeddings.py` | `GeminiEmbeddingProvider` (free tier, batch-aware, rate-limited) with `EmbeddingCache` (thread-safe LRU, SHA-256 keyed). Also includes `LocalEmbeddingProvider` fallback. |
| **ToolCallParser** | `core/llm/tool_parser.py` | Regex-based XML parser for tool calls. Pydantic validation for 8 tools: bash, read_file, write_file, edit_file, web_search, teach, recall, ingest. |

### Memory System

The heart of the project. See [Memory System](memory-system.md) for detailed documentation.

| Component | File | Description |
|---|---|---|
| **MemoryDatabase** | `core/memory/database.py` | Single SQLite with sqlite-vec (768-dim embeddings), FTS5 (BM25), and triggers. ~801 lines. |
| **MemoryReader** | `core/memory/reader.py` | Full retrieval pipeline: adaptive strategy, dense search, sparse search, RRF fusion, heuristic reranking, KG traversal, procedure retrieval. |
| **MemoryWriter** | `core/memory/writer.py` | LLM-powered extraction of facts, KG entities/relations, and procedures. Runs async (fire-and-forget). ~718 lines. |
| **KnowledgeGraph** | `core/memory/kg.py` | Entity/relation CRUD + BFS traversal. 9 entity types, 12 relation types. |
| **ProcedureStore** | `core/memory/procedures.py` | CRUD + UCB1-scored retrieval with success/failure tracking. |
| **Reranker** | `core/memory/reranker.py` | 2-stage pipeline: heuristic scoring, logistic regression blending. Cross-encoder stage reserved for Phase 4+. |
| **FeedbackCollector** | `core/memory/feedback.py` | Implicit feedback collection + pure-Python logistic regression training. |
| **ConsolidationEngine** | `core/memory/consolidation.py` | Periodic background job: near-duplicate merging, contradiction resolution, importance decay/promotion. |
| **DreamEngine** | `core/memory/dream.py` | LLM-powered "REM sleep" consolidation: abstraction, contradiction resolution, pattern detection. |
| **ReasoningEngine** | `core/memory/reasoning.py` | System 2 deliberate thinking: gap analysis, cross-domain connections, rule inference. |
| **DocumentIngester** | `core/memory/documents.py` | Format-aware chunking, SHA-256 dedup, 500KB size guard. |

### Tools

| Component | File | Description |
|---|---|---|
| **ToolExecutor** | `core/tools/executor.py` | Central dispatcher for 8 tools. Never raises, always returns `ToolResult`. |
| **BashTool** | `core/tools/bash.py` | Shell command execution with timeout, workdir, permission checks. |
| **FileOpsTool** | `core/tools/file_ops.py` | Read/write/edit files with permission enforcement. |
| **WebSearchTool** | `core/tools/web_search.py` | DuckDuckGo search with optional fact extraction. |
| **TeachTool** | `core/tools/teach.py` | Direct fact storage into memory. |
| **IngestTool** | `core/tools/ingest.py` | Document ingestion wrapper. |

### TUI (Terminal UI)

| Component | File | Description |
|---|---|---|
| **BrainAgentApp** | `tui/app.py` | 4-panel Textual TUI: Conversation (RichLog), Debug (RichLog), Token Budget (Static), Knowledge Base stats (Static). |
| **Panels** | `tui/panels.py` | Reusable panel widgets: `ConversationPanel`, `ActivityPanel`, `TokenBudgetPanel`. |
| **EventBus** | `tui/events.py` | Lightweight pub/sub with 200-event history. `AgentEvent` dataclass with `format_event()` for display. |

### Tests

98 tests passing across 20 test files covering all major components. See [Testing](testing.md) for details.

## Data Flow

### Retrieval Pipeline

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Classification│  (simple/complex/lookup/procedural)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Adaptive Strategy  │  (skip/conservative/normal/aggressive)
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────┐
│ Dense │ │  Sparse  │  (sqlite-vec ANN)  (FTS5 BM25)
│Search │ │  Search  │
└───┬───┘ └────┬─────┘
    │          │
    └────┬─────┘
         ▼
┌─────────────────────┐
│   RRF Fusion        │  (Reciprocal Rank Fusion)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Heuristic Rerank   │  (recency, importance, access freq, category)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  KG Traversal       │  (BFS 2 hops from extracted entities)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Procedure Retrieval│  (UCB1 scoring)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Context Assembly   │  (32K budget, best-at-edges)
└─────────────────────┘
```

### Async Knowledge Extraction

```
Completed Turn
    │
    ▼ (fire-and-forget)
┌─────────────────────┐
│  Fact Extraction    │  (LLM extracts standalone facts)
└────────┬────────────┘
         │
    ┌────┴────────────┐
    ▼                 ▼
┌──────────┐   ┌──────────────┐
│ KG Update│   │ Procedure    │
│ (entities│   │ Extraction   │
│ + rels)  │   │ (if multi-   │
└──────────┘   │ step success)│
               └──────────────┘
```

### Consolidation Cycle (every 10 turns)

```
┌─────────────────────┐
│ Near-Duplicate Merge│  (cosine similarity > 0.95)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Contradiction Check │  (resolve conflicting facts)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Importance Decay    │  (30+ days old → reduced weight)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Importance Promotion│  (frequently accessed → boosted)
└─────────────────────┘
```

### AutoDream Cycle (every 50 turns)

```
┌─────────────────────┐
│ Abstraction Creation│  (generalize specific facts)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Contradiction Res.  │  (LLM resolves conflicts)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Pattern Detection   │  (find recurring themes)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Connection Inference│  (link disparate domains)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Question Generation │  (identify knowledge gaps)
└─────────────────────┘
```

## Design Principles

1. **Graceful Degradation** — Every component is optional. The agent works without LLM, embeddings, or tools.
2. **Async by Default** — Knowledge extraction, consolidation, and reasoning run in the background.
3. **Single SQLite Database** — All memory stored in one file with vector search (sqlite-vec), full-text search (FTS5), and triggers.
4. **Best-at-Edges Ordering** — Most relevant memories placed at the beginning and end of context to mitigate lost-in-the-middle.
5. **Adaptive Retrieval** — Query classification determines how aggressively to search, balancing speed vs. thoroughness.
