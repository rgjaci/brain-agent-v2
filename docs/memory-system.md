# Memory System

The memory system is the core of Brain Agent v2. It implements a multi-layered memory architecture inspired by human cognition: episodic memory, semantic knowledge graphs, procedural memory, and consolidation mechanisms.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory System                              │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  Episodic   │  │  Semantic    │  │   Procedural        │ │
│  │  Memory     │  │  (KG)        │  │   Memory            │ │
│  │             │  │              │  │                     │ │
│  │ memories    │  │ entities     │  │ procedures          │ │
│  │ table       │  │ + relations  │  │ table               │ │
│  │             │  │              │  │                     │ │
│  │ + vectors   │  │ + BFS        │  │ + UCB1 scoring      │ │
│  │   (sqlite)  │  │   traversal  │  │ + success tracking  │ │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬───────────┘ │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │   MemoryDatabase      │                       │
│              │   (SQLite + sqlite-vec│                       │
│              │    + FTS5 + triggers) │                       │
│              └───────────────────────┘                       │
│                          ▲                                   │
│         ┌────────────────┼─────────────────────┐             │
│         │                │                     │             │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────────┴──────────┐ │
│  │  Memory     │  │  Memory     │  │  Consolidation &    │ │
│  │  Reader     │  │  Writer     │  │  Dream Engine       │ │
│  │             │  │             │  │                     │ │
│  │  Retrieval  │  │  LLM-based  │  │  Background         │ │
│  │  Pipeline   │  │  Extraction │  │  Maintenance        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Database Schema

The entire memory system runs on a single SQLite database with three key extensions:

1. **sqlite-vec** — Vector similarity search (768-dimensional embeddings)
2. **FTS5** — Full-text search with BM25 ranking
3. **Triggers** — Automatic FTS index maintenance

### Tables

#### `memories` — Episodic Memory

Stores individual facts, observations, and experiences.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `content` | TEXT | The memory text |
| `category` | TEXT | Category label (fact, task, preference, observation, etc.) |
| `source` | TEXT | Origin (user, inferred, extracted, taught, document) |
| `importance` | REAL | Importance score [0.0, 1.0] |
| `confidence` | REAL | Confidence score [0.0, 1.0] |
| `access_count` | INTEGER | Number of times retrieved |
| `created_at` | REAL | Unix timestamp |
| `last_accessed` | REAL | Unix timestamp of last retrieval |
| `usefulness_score` | REAL | Computed usefulness [0.0, 1.0] |
| `superseded_by` | INTEGER | FK to newer version of this memory |
| `metadata` | TEXT | JSON metadata |

**Virtual Tables:**
- `memory_vectors` — sqlite-vec virtual table for 768-dim embedding search
- `memory_fts` — FTS5 virtual table for full-text search

#### `entities` — Knowledge Graph Nodes

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `name` | TEXT | Entity name |
| `entity_type` | TEXT | Type (person, tool, concept, project, file, service, language, config, other) |
| `description` | TEXT | Entity description |
| `properties` | TEXT | JSON properties |
| `importance` | REAL | Importance score |
| `source` | TEXT | Origin |
| `access_count` | INTEGER | Access count |
| `created_at` | REAL | Unix timestamp |
| `last_accessed` | REAL | Last access timestamp |

**Virtual Tables:**
- `entity_vectors` — sqlite-vec for entity embedding search

#### `relations` — Knowledge Graph Edges

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `source_id` | INTEGER | FK to entities.id |
| `target_id` | INTEGER | FK to entities.id |
| `relation_type` | TEXT | Relation type (uses, prefers, part_of, causes, depends_on, contradicts, instance_of, located_in, created_by, configured_with, works_with, manages, belongs_to) |
| `properties` | TEXT | JSON properties |
| `created_at` | REAL | Unix timestamp |

#### `procedures` — Procedural Memory

| Column | Type | Description |
|---|---|---|
| `id` | TEXT | UUID |
| `name` | TEXT | Procedure name |
| `description` | TEXT | What this procedure does |
| `steps` | TEXT | JSON array of steps |
| `preconditions` | TEXT | JSON array of preconditions |
| `context` | TEXT | JSON context information |
| `success_count` | INTEGER | Times successfully used |
| `failure_count` | INTEGER | Times failed |
| `last_used` | REAL | Unix timestamp |
| `created_at` | REAL | Unix timestamp |

#### `documents` — Document Storage

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `path` | TEXT | File path |
| `content` | TEXT | Document content |
| `chunk_hash` | TEXT | SHA-256 hash for dedup |
| `parent_id` | INTEGER | Parent document ID (for hierarchies) |
| `created_at` | REAL | Unix timestamp |

#### `conversations` — Conversation History

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `role` | TEXT | user / assistant / system |
| `content` | TEXT | Message content |
| `created_at` | REAL | Unix timestamp |

#### `retrieval_feedback` — Feedback Log

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `query` | TEXT | The query |
| `retrieved_ids` | TEXT | JSON array of retrieved memory IDs |
| `referenced_ids` | TEXT | JSON array of actually referenced IDs |
| `created_at` | REAL | Unix timestamp |

#### `config` — Configuration Storage

| Column | Type | Description |
|---|---|---|
| `key` | TEXT | Configuration key |
| `value` | TEXT | JSON value |

Used for storing reranker weights and other runtime configuration.

## Memory Categories

Memories are categorized to enable category-specific retrieval bonuses:

| Category | Description | Example |
|---|---|---|
| `fact` | Standalone factual statement | "Python 3.11+ is required" |
| `preference` | User preference | "User prefers dark mode" |
| `task` | Task-related information | "Deploy to production on Fridays" |
| `observation` | Observed behavior | "The API responds in ~200ms" |
| `correction` | User correction | "Don't use pip, use pipx" |
| `context` | Situational context | "Working on the auth module" |
| `general` | Default category | — |

## Knowledge Graph

### Entity Types

| Type | Description | Example |
|---|---|---|
| `person` | People | "Alice", "Bob" |
| `tool` | Software tools | "Docker", "Git" |
| `concept` | Abstract concepts | "Authentication", "Caching" |
| `project` | Projects | "brain-agent-v2" |
| `file` | Files/directories | "config.yaml" |
| `service` | Services | "PostgreSQL", "Redis" |
| `language` | Programming languages | "Python", "Rust" |
| `config` | Configuration items | "OLLAMA_MODEL" |
| `other` | Unclassified | — |

### Relation Types

| Type | Description | Example |
|---|---|---|
| `uses` | Entity uses another | "Project uses Docker" |
| `prefers` | Preference relation | "User prefers Python" |
| `part_of` | Composition | "Auth is part of API" |
| `causes` | Causal relationship | "Bug causes crash" |
| `depends_on` | Dependency | "API depends on DB" |
| `contradicts` | Contradiction | "Fact A contradicts Fact B" |
| `instance_of` | Type membership | "Redis is instance of Cache" |
| `located_in` | Location | "File located in src/" |
| `created_by` | Creation | "File created by User" |
| `configured_with` | Configuration | "Server configured with SSL" |
| `works_with` | Compatibility | "Tool A works with Tool B" |
| `manages` | Management | "Docker manages containers" |
| `belongs_to` | Ownership | "Module belongs to Project" |

## Retrieval Pipeline

See [Retrieval Pipeline](retrieval-pipeline.md) for detailed documentation.

## Memory Writer

The `MemoryWriter` (`core/memory/writer.py`) performs async knowledge extraction from completed interactions:

1. **Fact Extraction** — LLM extracts standalone facts from the conversation
2. **KG Entity/Relation Extraction** — LLM identifies entities and their relationships
3. **Procedure Extraction** — LLM identifies multi-step procedures from successful interactions

All three run as fire-and-forget async tasks — they don't block the main turn loop.

### Prompt Templates

The writer uses three LLM prompt templates:

1. **Fact Extraction** — "Extract standalone facts from this conversation..."
2. **Graph Extraction** — "Extract entities and relationships from this conversation..."
3. **Procedure Extraction** — "Extract multi-step procedures from this successful interaction..."

## Consolidation Engine

The `ConsolidationEngine` (`core/memory/consolidation.py`) runs every 10 turns (or after 300s idle):

1. **Near-Duplicate Merging** — Merges memories with cosine similarity > 0.95
2. **Contradiction Resolution** — Identifies and resolves conflicting facts
3. **Importance Decay** — Reduces importance of memories older than 30 days
4. **Importance Promotion** — Boosts frequently accessed memories

## Dream Engine

The `DreamEngine` (`core/memory/dream.py`) runs every 50 turns (or after 600s idle):

1. **Abstraction Creation** — Generalizes specific facts into higher-level concepts
2. **Contradiction Resolution** — LLM resolves conflicts between memories
3. **Pattern Detection** — Finds recurring themes across memories
4. **Connection Inference** — Links disparate domains
5. **Question Generation** — Identifies knowledge gaps

## Reasoning Engine

The `ReasoningEngine` (`core/memory/reasoning.py`) implements System 2 deliberate thinking:

1. **Gap Analysis** — Identifies missing information
2. **Cross-Domain Connections** — Finds relationships between different knowledge areas
3. **Rule Inference** — Derives general rules from specific observations
4. **Contradiction Checks** — Validates consistency of stored knowledge
5. **Procedure Improvement** — Suggests improvements to stored procedures
6. **Question Answering** — Attempts to answer stored knowledge gaps

## Document Ingestion

The `DocumentIngester` (`core/memory/documents.py`) handles permanent document storage:

- **Format-Aware Chunking** — Python: splits on `def`/`class`; Markdown: splits on headings; Others: splits on paragraphs
- **SHA-256 Dedup** — Prevents duplicate storage
- **500KB Size Guard** — Skips oversized files
- **Directory Walking** — Recursively ingests directories with skip list for common non-source directories

## Bootstrap Flow

The bootstrap process (`BrainAgent.bootstrap()` in `core/agent.py`) runs automatically on first launch to build initial knowledge about the user's environment.

### When It Runs

- Only when the database is **empty** (0 memories)
- Skipped if any memories already exist
- Can be triggered manually via `python main.py bootstrap`

### What It Scans

The bootstrap runs 9 shell commands (5-second timeout each) and stores the raw output as `observation` memories:

| Scan Name | Command | What It Captures |
|---|---|---|
| Shell config | `cat ~/.zshrc || cat ~/.bashrc` | Shell aliases, PATH, environment setup |
| Git config | `git config --global --list` | User name, email, default editor, aliases |
| Python | `python3 --version && which python3` | Python version and location |
| Node | `node --version && which node` | Node.js version and location |
| Docker | `docker --version` | Docker availability and version |
| Ollama models | `ollama list` | Available local LLM models |
| OS info | `uname -a` | Operating system and kernel version |
| Disk usage | `df -h / \| tail -1` | Available disk space |
| Recent projects | `find ~/projects -maxdepth 1 -type d \| head -20` | Project directories |

### How It Works

```python
BOOTSTRAP_SCANS = [
    ("Shell config", "cat ~/.zshrc 2>/dev/null || cat ~/.bashrc 2>/dev/null"),
    ("Git config", "git config --global --list 2>/dev/null"),
    ("Python", "python3 --version 2>/dev/null && which python3"),
    ("Node", "node --version 2>/dev/null && which node 2>/dev/null"),
    ("Docker", "docker --version 2>/dev/null"),
    ("Ollama models", "ollama list 2>/dev/null"),
    ("OS info", "uname -a"),
    ("Disk usage", "df -h / | tail -1"),
    ("Recent projects", "find ~/projects -maxdepth 1 -type d 2>/dev/null | head -20"),
]

for name, cmd in BOOTSTRAP_SCANS:
    result = await tool_executor.execute("bash", {"command": cmd, "timeout": 5})
    if result.success and result.output.strip():
        # Store raw scan output — no LLM call needed
        content = f"[{name}]\n{result.output.strip()}"
        db.insert_memory(
            content=content,
            category="observation",
            source=f"scan:{name}",
            importance=0.6,
            confidence=0.9,
        )
```

### Design Decisions

1. **No LLM involved** — Raw output is stored directly, keeping bootstrap fast
2. **Graceful failures** — Individual scan failures are logged but don't stop the process
3. **Events emitted** — `bootstrap_start`, `bootstrap_scan`, `bootstrap_done` events for TUI display
4. **Idempotent** — Only runs once (checks `db.count_memories() > 0`)

### Example Stored Memory

```
[Git config]
user.name=John Doe
user.email=john@example.com
core.editor=vim
alias.st=status
alias.co=checkout
```

This raw data becomes available for retrieval in subsequent conversations. The LLM can reason over it when relevant (e.g., "what's my git setup?").
