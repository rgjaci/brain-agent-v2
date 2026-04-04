# Retrieval Pipeline

The retrieval pipeline is the agent's "search engine" — it finds relevant memories, knowledge graph context, and procedures for every user query.

## Pipeline Stages

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 0: Query Classification          │
│  Classifies query as:                   │
│  - Simple (greeting, acknowledgment)    │
│  - Complex (reasoning, multi-part)      │
│  - Lookup (factual recall)              │
│  - Procedural (how-to, steps)           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 1: Adaptive Strategy Selection   │
│  Chooses retrieval intensity:           │
│  - Skip: no retrieval needed            │
│  - Conservative: minimal search         │
│  - Normal: standard hybrid search       │
│  - Aggressive: full search + KG + procs │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐  ┌──────────────┐
│ Stage 2a:     │  │ Stage 2b:    │
│ Dense Search  │  │ Sparse Search│
│ (sqlite-vec)  │  │ (FTS5 BM25)  │
│               │  │              │
│ Embeds query  │  │ Keyword match│
│ via Gemini    │  │ against FTS5 │
│ ANN search    │  │ index        │
└───────┬───────┘  └──────┬───────┘
        │                 │
        └────────┬────────┘
                 ▼
┌─────────────────────────────────────────┐
│  Stage 3: RRF Fusion                    │
│  Reciprocal Rank Fusion merges both     │
│  ranked lists into a single score:      │
│                                         │
│  RRF(d) = Σ 1 / (k + rank_i(d))        │
│                                         │
│  Default k=60                           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 4: Heuristic Reranking           │
│  Applies multipliers to RRF score:      │
│                                         │
│  - Recency: ×(1 + 0.3×e^(-age/30))     │
│  - Access freq: ×(1 + 0.1×min(n,10))   │
│  - Importance: ×(0.5 + importance)     │
│  - Correction bonus: ×1.5              │
│  - Preference bonus: ×1.3              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 5: Logistic Regression Blend     │
│  (when trained weights available)       │
│                                         │
│  final = 0.6 × heuristic + 0.4 × LR    │
│                                         │
│  LR trained on implicit feedback data   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 6: Knowledge Graph Traversal     │
│  - Extract entities from query          │
│  - BFS 2 hops from each entity          │
│  - Build supplementary context string   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 7: Procedure Retrieval           │
│  - BM25 search against procedures       │
│  - UCB1 scoring (success rate +         │
│    exploration bonus)                   │
│  - Format for context injection         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 8: Best-at-Edges Reordering      │
│  Highest-scoring memories placed at     │
│  both beginning and end of context      │
│  to mitigate lost-in-the-middle effect  │
└─────────────────────────────────────────┘
```

## Stage 0: Query Classification

The query classifier examines the user's input to determine the type of query:

```python
# Keywords that trigger aggressive retrieval
_AGGRESSIVE_KEYWORDS = {
    "remember", "memory", "memorize", "recall", "search", "find", "lookup",
    "what did", "when did", "have i", "did i", "told you", "mentioned",
    "stored", "saved", "know about",
}

# Keywords that trigger procedure retrieval
_PROCEDURE_KEYWORDS = {
    "how to", "how do", "steps to", "process for", "procedure for",
    "way to", "guide", "tutorial", "instructions", "help me",
}

# Keywords that trigger fact retrieval
_FACT_KEYWORDS = {
    "who is", "what is", "what are", "who are", "define", "explain",
    "describe", "tell me about", "what does",
}
```

## Stage 1: Adaptive Strategy

Based on query classification, the agent selects a retrieval strategy:

| Strategy | When | What It Does |
|---|---|---|
| **Skip** | Greetings, acknowledgments, simple responses | No retrieval, direct LLM response |
| **Conservative** | Follow-up questions, clarifications | Dense search only, top-5 |
| **Normal** | Standard queries | Full hybrid search (dense + sparse), top-10 |
| **Aggressive** | Explicit memory queries, complex tasks | Full search + KG traversal + procedure retrieval, top-15+ |

## Stage 2: Hybrid Search

### Dense Search (sqlite-vec ANN)

1. Embed the query using Gemini's `text-embedding-004` model (768 dimensions)
2. Search the `memory_vectors` virtual table using sqlite-vec's approximate nearest neighbor (ANN)
3. Return top-K results ranked by cosine similarity

```python
# Embedding is cached via SHA-256 keyed LRU cache
embedding = embedder.embed(query)  # Returns 768-dim vector

# sqlite-vec ANN search
results = db.execute("""
    SELECT memory_id, distance
    FROM memory_vectors
    ORDER BY distance ASC
    LIMIT ?
""", (top_k,))
```

### Sparse Search (FTS5 BM25)

1. Search the `memory_fts` virtual table using SQLite's FTS5 extension
2. Results ranked by BM25 algorithm
3. Returns top-K results ranked by relevance

```python
# FTS5 BM25 search
results = db.execute("""
    SELECT id, rank
    FROM memory_fts
    WHERE memory_fts MATCH ?
    ORDER BY rank
    LIMIT ?
""", (query, top_k))
```

## Stage 3: RRF Fusion

Reciprocal Rank Fusion (RRF) combines the two ranked lists:

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

Where:
- `d` is a document (memory)
- `k` is a constant (default: 60)
- `rank_i(d)` is the rank of document `d` in list `i`

The constant `k=60` is empirically chosen — it gives more weight to top-ranked items while still allowing items that appear in both lists to rise.

**Example:**

| Memory | Dense Rank | Sparse Rank | RRF Score |
|---|---|---|---|
| M1 | 1 | 3 | 1/61 + 1/63 = 0.0323 |
| M2 | 2 | 1 | 1/62 + 1/61 = 0.0325 |
| M3 | 5 | 2 | 1/65 + 1/62 = 0.0315 |

## Stage 4: Heuristic Reranking

Each memory's RRF score is multiplied by several factors:

### Recency Multiplier

```
recency = 1 + 0.3 × e^(-age_days / 30)
```

- Brand new memory (0 days): ×1.3
- 30 days old: ×1.11
- 90 days old: ×1.015
- Very old: approaches ×1.0

### Access Frequency Multiplier

```
frequency = 1 + 0.1 × min(access_count, 10)
```

- Never accessed: ×1.0
- Accessed 5 times: ×1.5
- Accessed 10+ times: ×2.0

### Importance Multiplier

```
importance = 0.5 + importance_score
```

- Low importance (0.0): ×0.5
- Medium importance (0.5): ×1.0
- High importance (1.0): ×1.5

### Category Bonuses

| Category | Multiplier | Rationale |
|---|---|---|
| `correction` | ×1.5 | User corrections are highly valuable |
| `preference` | ×1.3 | User preferences should be prioritized |
| Other | ×1.0 | No bonus |

## Stage 5: Logistic Regression Blend (Cross-Encoder Deferred)

When the reranker has been trained on feedback data, the heuristic score is blended with a learned logistic regression score:

```
final_score = 0.6 × heuristic_score + 0.4 × lr_score
```

The LR model is a pure-Python logistic regression trained on implicit feedback:
- **Features**: recency, access count, importance, category, query-memory similarity, original rank
- **Labels**: Whether a memory was actually referenced in the response (implicit positive)
- **Training**: Triggered by `RetrievalFeedbackCollector.train()` after sufficient data collected

### Cross-Encoder Reranker (Deferred — Phase 4+)

The spec describes a third reranking stage using a neural cross-encoder model (`ms-marco-MiniLM-L6-v2`, ~30MB, +33-48% retrieval quality). This is **not yet implemented** and is deferred for the following reasons:

1. **Heavy dependency** — `sentence-transformers` pulls in PyTorch (~2GB)
2. **Latency cost** — Adds ~100-200ms per retrieval call
3. **Current pipeline is effective** — Heuristic + LR already achieves NDCG@5 = 1.0 on benchmarks
4. **Spec guidance** — Listed as "Phase 4+ (after ~1000+ interactions)"

The placeholder exists in `core/memory/reranker.py` as `_cross_encoder_model = None` and the `cross_encoder_rerank()` method returns top-k by order as a fallback. To implement it later:

```python
# Install: pip install sentence-transformers
from sentence_transformers import CrossEncoder

class Reranker:
    _cross_encoder_model = None

    def _init_cross_encoder(self):
        if self._cross_encoder_model is None:
            self._cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def cross_encoder_rerank(self, query, candidates, top_k=10):
        self._init_cross_encoder()
        pairs = [(query, c["content"]) for c in candidates]
        scores = self._cross_encoder_model.predict(pairs)
        # ... sort and return top_k
```

## Stage 6: Knowledge Graph Traversal

1. **Entity Extraction** — Extract named entities from the query (filtering stop words and known tools)
2. **BFS Traversal** — For each entity, traverse the knowledge graph up to 2 hops
3. **Context Building** — Format the traversal results as a supplementary context string

```python
# Entity extraction from query
entities = extract_entities(query)  # Filters stop words, known tools

# BFS traversal (2 hops)
kg_context = kg.traverse(entities, max_facts=20)

# Result format:
# "Knowledge Graph Context:
#  - Python (language) → uses → Docker (tool)
#  - Docker (tool) → manages → Containers (concept)
#  ..."
```

## Stage 7: Procedure Retrieval

Procedures are retrieved using BM25-style text search and ranked with UCB1:

### UCB1 Scoring

```
UCB1 = success_rate + √(2 × log(total_attempts + 1) / (attempt_count + 1))
```

Where:
- `success_rate = success_count / attempt_count`
- `total_attempts` = sum of all procedure attempts (exploration term)
- The exploration bonus encourages trying less-used procedures

### Text Relevance

A simple keyword-overlap score is computed:

```python
q_tokens = set(query.lower().split())
proc_text = " ".join([proc.name, proc.description] + proc.steps)
p_tokens = set(proc_text.lower().split())
relevance = len(q_tokens & p_tokens) / len(q_tokens)
```

## Stage 8: Best-at-Edges Reordering

After all scoring is complete, memories are reordered so the highest-scoring items appear at both the beginning and end of the context window:

```python
def apply_best_at_edges(memories):
    """Interleave best memories at front and back of list."""
    sorted_memories = sorted(memories, key=lambda m: m["_rerank_score"], reverse=True)
    result = []
    left, right = 0, len(sorted_memories) - 1
    while left <= right:
        if left == right:
            result.append(sorted_memories[left])
        else:
            result.append(sorted_memories[left])
            result.append(sorted_memories[right])
        left += 1
        right -= 1
    return result
```

**Example:**

| Original Order | Score | Best-at-Edges Order |
|---|---|---|
| M1 | 0.3 | M1 (0.9) ← best at front |
| M2 | 0.5 | M5 (0.2) ← worst at back |
| M3 | 0.9 | M2 (0.5) |
| M4 | 0.2 | M4 (0.3) |
| M5 | 0.1 | M3 (0.9) ← best at back |

This ordering exploits the "lost in the middle" phenomenon — LLMs pay more attention to items at the extremes of long prompts.

## Feedback Loop

The retrieval system learns from implicit feedback:

1. **Collection** — After each turn, the system logs which memories were retrieved vs. which were actually referenced in the response
2. **Training** — Periodically, the `RetrievalFeedbackCollector` trains a logistic regression model on this data
3. **Application** — The trained weights are loaded by the `Reranker` and used in Stage 5

```python
# Collect feedback
collector.log_feedback(
    query="how to deploy",
    retrieved_ids=[1, 2, 3, 4, 5],
    referenced_ids=[2, 4],  # Only these were actually used
)

# Train when enough data collected
if collector.enough_data():
    weights = collector.train()
    collector.save_weights(weights)
```
