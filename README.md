# Brain Agent v2

A memory-first AI agent that proves a 4B parameter model with an exceptional memory system can match frontier models on tasks where accumulated knowledge matters.

The LLM is a processor. Memory is the intelligence.

## Documentation

📚 **[Full Documentation](docs/README.md)** — Architecture, setup, extension guides, troubleshooting, and more.

| Guide | Description |
|---|---|
| [Architecture](docs/architecture.md) | System architecture, component details, data flow diagrams |
| [Setup](docs/setup.md) | Installation, configuration, deployment |
| [Memory System](docs/memory-system.md) | Database schema, knowledge graph, consolidation, dream engine |
| [Retrieval Pipeline](docs/retrieval-pipeline.md) | 8-stage retrieval pipeline with RRF fusion and reranking |
| [Extension Guide](docs/extension-guide.md) | Add tools, providers, categories, reasoning strategies |
| [Token Budget](docs/token-budget.md) | Context window allocation and tuning |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Testing](docs/testing.md) | Test suite overview and writing tests |
| [Benchmarks](docs/benchmarks.md) | Benchmark methodology and results |

## Quick Start

```bash
pip install -e '.[dev]'

# Set environment variables
export GEMINI_API_KEY=your-key-here          # Required for embeddings
export OLLAMA_MODEL=qwen3.5:4b-nothink      # Optional (default)
export OLLAMA_BASE_URL=http://localhost:11434 # Optional (default)

# Or use OpenRouter instead of Ollama:
export OPENROUTER_API_KEY=your-key-here

# Run the agent
python main.py chat          # headless terminal chat
python main.py               # TUI (requires textual)
python main.py bootstrap     # scan environment, store facts
python main.py teach "fact"  # store a fact directly
python main.py recall "query" # search memories
python main.py ingest path   # ingest file/directory
python main.py stats         # print memory statistics
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | - | Google Gemini API key for embeddings |
| `OLLAMA_MODEL` | No | `qwen3.5:4b-nothink` | Ollama model name |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OPENROUTER_API_KEY` | No | - | OpenRouter API key (overrides Ollama) |
| `OPENROUTER_MODEL` | No | `qwen/qwen-2.5-7b-instruct` | OpenRouter model |

## Architecture

```
User Input
  -> Adaptive Retrieval (decide: skip / conservative / normal / aggressive)
  -> Hybrid Search (FTS5 BM25 + sqlite-vec ANN)
  -> RRF Fusion + Heuristic Reranking
  -> KG Traversal (entity extraction -> BFS 2 hops)
  -> Procedure Retrieval (UCB1 selection)
  -> Context Assembly (32K token budget, best-at-edges ordering)
  -> LLM Generation + Tool Loop (max 10 iterations)
  -> Async Knowledge Extraction (facts + entities + relations + procedures)
  -> Consolidation (decay, promote, dedup, contradiction resolution)
```

**Storage:** Single SQLite database with sqlite-vec (768-dim embeddings) and FTS5 (BM25).

**5 Core Techniques:**

1. **Retrieval-Specialized Context Usage** -- structured prompts that teach the model to follow procedures and cite memories
2. **Adaptive Retrieval** -- confidence-based triggering (skip / conservative / normal / aggressive)
3. **Hierarchical Procedural Memory** -- UCB1-scored procedures extracted from successful multi-tool interactions
4. **Hybrid Search + Reranking** -- FTS5 + vector search + RRF fusion + heuristic/LR reranking (cross-encoder deferred to Phase 4+)
5. **Graph-Augmented Memory** -- lightweight SQLite KG with BFS traversal

## CLI Usage

```bash
# Interactive chat
python main.py chat

# Teach facts
python main.py teach "User prefers ed25519 SSH keys"
python main.py teach "Production server is at 10.0.0.1"

# Search memories
python main.py recall "SSH key preference"

# Ingest documentation
python main.py ingest ./docs/

# View stats
python main.py stats
```

## Running Tests

```bash
pip install -e '.[dev]'
pytest tests/ -q
```

## Benchmarks

```bash
# Recall@K benchmark
python benchmarks/recall_test.py

# Procedure matching
python benchmarks/procedure_test.py

# Reranker evaluation
python benchmarks/reranker_eval.py

# Export RAFT fine-tuning data
python benchmarks/export_raft_data.py --db ~/.brain_agent/memory.db --output raft_data.jsonl
```
