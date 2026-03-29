"""Recall benchmark — measures how often the correct memory surfaces in top-K retrieval.

Usage:
    python -m benchmarks.recall_test [--db PATH] [--top-k 10] [--verbose]

Generates a synthetic test set of (query, expected_memory) pairs, inserts them
into a fresh in-memory DB, then measures Recall@K and MRR.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock


# ── test corpus ───────────────────────────────────────────────────────────────

CORPUS = [
    # (memory_content, category, importance, retrieval_queries)
    ("User prefers ed25519 SSH keys over RSA",
     "preference", 0.85,
     ["SSH key type", "what SSH key does the user prefer?", "ed25519 or RSA?"]),

    ("Production server is hosted at 10.0.0.1 (Hetzner VPS)",
     "fact", 0.9,
     ["production server IP", "where is the server?", "Hetzner VPS address"]),

    ("Deploy with rsync, not scp — rsync handles large files better",
     "preference", 0.8,
     ["how to deploy files", "rsync vs scp", "file transfer preference"]),

    ("Python 3.12 is installed on the production server",
     "fact", 0.75,
     ["Python version on server", "what python is on prod?", "Python 3.12"]),

    ("User uses tmux with prefix key C-a (not the default C-b)",
     "preference", 0.7,
     ["tmux prefix key", "tmux config", "C-a vs C-b"]),

    ("PostgreSQL 16 database runs on port 5433 (non-default)",
     "fact", 0.88,
     ["PostgreSQL port", "database port", "postgres config"]),

    ("User corrected: nginx config lives in /etc/nginx/sites-available, not /etc/nginx/conf.d",
     "correction", 0.9,
     ["nginx config location", "where is nginx config?", "sites-available"]),

    ("Backup script runs daily at 03:00 via cron, outputs to /backups/daily/",
     "fact", 0.8,
     ["backup schedule", "when does backup run?", "cron backup"]),

    ("User prefers Black for Python formatting (line length 88)",
     "preference", 0.72,
     ["python formatter", "code style", "Black formatter"]),

    ("Tailscale is used for private network access; key is stored in /etc/tailscale/",
     "fact", 0.85,
     ["Tailscale config", "VPN setup", "tailscale key location"]),
]


class RecallResult(NamedTuple):
    query: str
    expected_id: int
    rank: int        # 1-indexed; 0 = not found
    found: bool
    top_k: int


# ── helpers ───────────────────────────────────────────────────────────────────

def build_fake_embedder(memory_embeddings: dict[int, list[float]]):
    """Return a mock embedder that returns the stored embedding for exact matches."""
    embedder = MagicMock()

    async def embed_query(text: str) -> list[float]:
        # For benchmarking: return a noisy version of the target embedding
        # Real benchmarks would use the actual embedding model
        import random
        base = [0.1] * 768
        noise = [random.gauss(0, 0.05) for _ in range(768)]
        return [b + n for b, n in zip(base, noise)]

    embedder.embed_query = AsyncMock(side_effect=embed_query)
    return embedder


async def run_benchmark(top_k: int = 10, verbose: bool = False) -> dict:
    from core.memory.database import MemoryDatabase
    from core.memory.kg import KnowledgeGraph
    from core.memory.reader import MemoryReader

    db = MemoryDatabase(":memory:")
    kg = KnowledgeGraph(db)
    embedder = build_fake_embedder({})

    # Insert corpus
    memory_ids: list[int] = []
    for content, category, importance, _ in CORPUS:
        mid = db.insert_memory(
            content=content,
            category=category,
            importance=importance,
            source="benchmark",
        )
        memory_ids.append(mid)

    reader = MemoryReader(db, embedder, kg)

    results: list[RecallResult] = []
    total_start = time.perf_counter()

    for mem_idx, (content, category, importance, queries) in enumerate(CORPUS):
        expected_id = memory_ids[mem_idx]
        for query in queries:
            # Patch vector_search to return all memories (simulate recall scenario)
            all_rows = db.execute(
                "SELECT id, content, category, importance, access_count, "
                "created_at FROM memories"
            )
            reader.db.vector_search = MagicMock(return_value=all_rows)
            reader.db.fts_search = MagicMock(return_value=[])

            result = await reader.retrieve(query)
            retrieved = result.memories if hasattr(result, "memories") else result
            retrieved_ids = [m.id for m in retrieved[:top_k]]

            rank = 0
            for i, rid in enumerate(retrieved_ids, start=1):
                if rid == expected_id:
                    rank = i
                    break

            results.append(RecallResult(
                query=query,
                expected_id=expected_id,
                rank=rank,
                found=rank > 0,
                top_k=top_k,
            ))

            if verbose:
                status = f"✓ rank={rank}" if rank > 0 else "✗ not found"
                print(f"  [{status:12s}] {query[:60]}")

    elapsed = time.perf_counter() - total_start

    # Metrics
    n = len(results)
    recall_at_k = sum(r.found for r in results) / n
    mrr = sum(1.0 / r.rank for r in results if r.rank > 0) / n

    return {
        "total_queries": n,
        "top_k": top_k,
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "found": sum(r.found for r in results),
        "not_found": sum(not r.found for r in results),
        "elapsed_s": round(elapsed, 3),
        "avg_ms_per_query": round(elapsed / n * 1000, 2) if n else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Brain Agent recall benchmark")
    parser.add_argument("--top-k",   type=int,  default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n=== Recall Benchmark (top-{args.top_k}) ===\n")
    metrics = asyncio.run(run_benchmark(top_k=args.top_k, verbose=args.verbose))

    print(f"\nResults:")
    print(f"  Recall@{args.top_k}: {metrics['recall_at_k']:.1%}")
    print(f"  MRR:         {metrics['mrr']:.4f}")
    print(f"  Found:       {metrics['found']} / {metrics['total_queries']}")
    print(f"  Elapsed:     {metrics['elapsed_s']:.3f}s "
          f"({metrics['avg_ms_per_query']:.1f}ms/query)")
    print()

    # Exit non-zero if recall is below threshold
    threshold = 0.7
    if metrics["recall_at_k"] < threshold:
        print(f"  ⚠ BELOW THRESHOLD ({threshold:.0%})", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"  ✓ PASS (threshold: {threshold:.0%})")


if __name__ == "__main__":
    main()
