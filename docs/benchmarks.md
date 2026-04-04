# Benchmark Methodology

Brain Agent v2 includes a suite of benchmarks to evaluate the memory system's retrieval quality, procedure matching, and reranker effectiveness.

## Running Benchmarks

Benchmarks are separate from the test suite and use synthetic data:

```bash
# Recall benchmark
python benchmarks/recall_test.py

# Procedure benchmark
python benchmarks/procedure_test.py

# Reranker evaluation
python benchmarks/reranker_eval.py

# Export Raft data (for external evaluation)
python benchmarks/export_raft_data.py
```

All benchmarks use synthetic test data and run against in-memory databases — they don't touch the production database.

## Benchmark 1: Recall Test

**File:** `benchmarks/recall_test.py`

### Purpose

Measures how often the correct memory surfaces in the top-K retrieval results.

### Methodology

1. **Corpus** — 10 synthetic memories with known content, categories, and importance scores
2. **Queries** — Each memory has 3 associated queries (30 total query-memory pairs)
3. **Embeddings** — Uses a mock embedder with pre-computed embeddings for exact matching
4. **Retrieval** — Runs the full retrieval pipeline (dense + sparse + RRF + reranking)
5. **Metrics**:
   - **Recall@K** — Percentage of queries where the correct memory appears in top-K results
   - **MRR (Mean Reciprocal Rank)** — Average of 1/rank for each query (higher = better ranking)

### Test Corpus

The corpus includes diverse memory types:

| Memory | Category | Importance | Example Queries |
|---|---|---|---|
| SSH key preference | preference | 0.85 | "SSH key type", "what SSH key does the user prefer?" |
| Production server IP | fact | 0.9 | "production server IP", "where is the server?" |
| Deploy preference | preference | 0.8 | "how to deploy files", "rsync vs scp" |
| Python version | fact | 0.75 | "Python version on server", "what python is on prod?" |
| tmux config | preference | 0.7 | "tmux prefix key", "tmux config" |
| PostgreSQL port | fact | 0.88 | "PostgreSQL port", "database port" |
| nginx config correction | correction | 0.9 | "nginx config location", "where is nginx config?" |
| Backup schedule | fact | 0.8 | "backup schedule", "when does backup run?" |
| Python formatter | preference | 0.72 | "python formatter", "code style" |
| Tailscale config | fact | 0.85 | "Tailscale config", "VPN setup" |

### Results

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Recall@10 | 100.0% | 70% | ✅ PASS |
| MRR | 0.2929 | — | — |

### Interpretation

- **Recall@10 = 100%** — All correct memories appear in the top 10 results. This is expected with synthetic data and exact embedding matching.
- **MRR = 0.2929** — The average reciprocal rank is moderate. This means correct memories aren't always ranked #1, but they're consistently in the top results.

## Benchmark 2: Procedure Test

**File:** `benchmarks/procedure_test.py`

### Purpose

Measures how well the procedure retrieval system matches queries to relevant procedures using UCB1 scoring.

### Methodology

1. **Procedures** — 5 synthetic procedures (nginx deploy, SSH setup, PostgreSQL backup, Tailscale setup, systemd service)
2. **Queries** — Each procedure has 3 triggering queries (15 total)
3. **Retrieval** — BM25 text search against procedures table + UCB1 scoring
4. **Metrics**:
   - **Procedure Recall** — Percentage of queries where the correct procedure appears in results
   - **Procedure MRR** — Mean Reciprocal Rank for procedure matching

### Test Procedures

| Procedure | Steps | Trigger Keywords |
|---|---|---|
| deploy_nginx | 5 steps | "nginx deploy reverse proxy web server" |
| setup_ssh_keys | 4 steps | "ssh key setup authorized authentication" |
| postgres_backup | 3 steps | "postgres backup dump database export" |
| setup_tailscale | 3 steps | "tailscale vpn mesh network private" |
| systemd_service | 5 steps | "systemd service unit daemon autostart" |

### Results

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Procedure Recall | 100.0% | 50% | ✅ PASS |
| Procedure MRR | 1.0000 | — | — |

### Interpretation

- **Procedure Recall = 100%** — All correct procedures are found for their triggering queries.
- **Procedure MRR = 1.0** — Perfect ranking. The correct procedure is always ranked #1. This is expected with synthetic data and clear keyword separation.

## Benchmark 3: Reranker Evaluation

**File:** `benchmarks/reranker_eval.py`

### Purpose

Measures the effectiveness of the heuristic reranker in improving result quality compared to the baseline RRF ranking.

### Methodology

1. **Test Cases** — 3 queries with 5 memories each, with ground-truth relevance labels (0-3 scale)
2. **Baseline** — RRF scores only (no reranking)
3. **Reranked** — Heuristic reranking applied
4. **Metrics**:
   - **NDCG@5 (Normalized Discounted Cumulative Gain)** — Measures ranking quality with graded relevance
   - **Improvement** — Difference between reranked and baseline NDCG

### NDCG Calculation

```
DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
IDCG@K = DCG@K for ideal ranking
NDCG@K = DCG@K / IDCG@K
```

Where `rel_i` is the relevance label (0-3) of the item at position `i`.

### Test Cases

| Query | Memories | Relevance Distribution |
|---|---|---|
| "SSH key setup" | 5 memories | 2×high (3), 1×medium (2), 2×low (0) |
| "database backup" | 5 memories | 2×high (3), 1×medium (2), 1×medium (2), 1×low (0) |
| "nginx configuration" | 5 memories | 1×high (3), 1×medium (2), 1×low (1), 2×low (0) |

### Results

| Metric | Baseline | Reranked | Improvement | Status |
|---|---|---|---|---|
| NDCG@5 | 0.9476 | 1.0000 | +0.0524 | ✅ PASS |

### Interpretation

- **Baseline NDCG@5 = 0.9476** — RRF alone produces good rankings.
- **Reranked NDCG@5 = 1.0000** — Perfect ranking after heuristic reranking.
- **Improvement = +0.0524** — The reranker provides a modest but consistent improvement.

The reranker's main contributions:
- **Recency bonus** — Recent memories are boosted
- **Category bonuses** — Corrections and preferences get higher scores
- **Access frequency** — Frequently accessed memories are prioritized
- **Importance weighting** — High-importance memories rise in ranking

## Benchmark 4: Raft Data Export

**File:** `benchmarks/export_raft_data.py`

### Purpose

Exports retrieval data in Raft format for external evaluation and analysis.

### Usage

```bash
python benchmarks/export_raft_data.py --output raft_data.json
```

### Output Format

```json
{
  "queries": [
    {
      "query": "SSH key type",
      "retrieved": [
        {"id": 1, "content": "...", "score": 0.95},
        ...
      ],
      "expected": [1]
    }
  ]
}
```

## Running All Benchmarks

```bash
# Run all benchmarks
python benchmarks/recall_test.py --verbose
python benchmarks/procedure_test.py --verbose
python benchmarks/reranker_eval.py --verbose

# Or run as modules
python -m benchmarks.recall_test --verbose
python -m benchmarks.procedure_test --verbose
python -m benchmarks.reranker_eval --verbose
```

## Benchmark vs Test Suite

| Aspect | Test Suite (`tests/`) | Benchmarks (`benchmarks/`) |
|---|---|---|
| **Purpose** | Verify correctness | Measure quality |
| **Data** | Mock/fake objects | Synthetic test corpus |
| **Metrics** | Pass/fail | Recall, MRR, NDCG |
| **Speed** | Fast (0.83s) | Moderate |
| **Database** | In-memory | In-memory |
| **CI Integration** | Yes (pytest) | Manual |

## Improving Benchmark Realism

The current benchmarks use synthetic data with clear keyword separation. For more realistic evaluation:

1. **Use real conversation data** — Export actual queries and memories from the production database
2. **Add noise** — Include typos, paraphrases, and ambiguous queries
3. **Cross-validation** — Split data into train/test sets for reranker evaluation
4. **A/B testing** — Compare different retrieval strategies on the same queries
5. **Human evaluation** — Have humans rate retrieval quality for a subset of queries
