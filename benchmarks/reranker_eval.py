"""Reranker evaluation benchmark — measures NDCG and rank improvement vs baseline.

Usage:
    python -m benchmarks.reranker_eval [--verbose]

Creates a set of (query, memories with relevance labels) tuples, runs the
heuristic reranker, and measures NDCG@K vs the unranked (RRF-score) baseline.
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys
import time

# ── evaluation dataset ────────────────────────────────────────────────────────

def make_test_case(query: str, memories: list[dict]) -> dict:
    """Each memory has: content, category, importance, access_count,
    created_at, rrf_score, relevance (ground truth 0-3)."""
    return {"query": query, "memories": memories}


def _days_ago(n: int) -> str:
    d = datetime.datetime.now() - datetime.timedelta(days=n)
    return d.isoformat()


TEST_CASES = [
    make_test_case("SSH key setup", [
        {"content": "User prefers ed25519 SSH keys",
         "category": "preference", "importance": 0.85, "access_count": 8,
         "created_at": _days_ago(3),  "rrf_score": 0.9, "relevance": 3},
        {"content": "User corrected: use ssh-copy-id not manual copy",
         "category": "correction", "importance": 0.8, "access_count": 2,
         "created_at": _days_ago(1),  "rrf_score": 0.6, "relevance": 3},
        {"content": "Python 3.12 installed on server",
         "category": "fact",       "importance": 0.7, "access_count": 1,
         "created_at": _days_ago(7),  "rrf_score": 0.4, "relevance": 0},
        {"content": "SSH config stored in ~/.ssh/config",
         "category": "fact",       "importance": 0.75, "access_count": 4,
         "created_at": _days_ago(14), "rrf_score": 0.55, "relevance": 2},
        {"content": "tmux prefix key is C-a",
         "category": "preference", "importance": 0.6, "access_count": 3,
         "created_at": _days_ago(5),  "rrf_score": 0.3, "relevance": 0},
    ]),

    make_test_case("database backup", [
        {"content": "PostgreSQL backup runs daily at 03:00",
         "category": "fact",       "importance": 0.9, "access_count": 5,
         "created_at": _days_ago(2),  "rrf_score": 0.85, "relevance": 3},
        {"content": "Backup output goes to /backups/daily/",
         "category": "fact",       "importance": 0.8, "access_count": 3,
         "created_at": _days_ago(2),  "rrf_score": 0.75, "relevance": 3},
        {"content": "User corrected: compress backups with gzip not bzip2",
         "category": "correction", "importance": 0.85, "access_count": 1,
         "created_at": _days_ago(0),  "rrf_score": 0.5, "relevance": 2},
        {"content": "Nginx config lives in /etc/nginx/sites-available",
         "category": "correction", "importance": 0.9, "access_count": 2,
         "created_at": _days_ago(4),  "rrf_score": 0.3, "relevance": 0},
        {"content": "PostgreSQL port is 5433 (non-default)",
         "category": "fact",       "importance": 0.85, "access_count": 6,
         "created_at": _days_ago(10), "rrf_score": 0.6, "relevance": 2},
    ]),

    make_test_case("nginx configuration", [
        {"content": "User corrected: nginx config in sites-available, not conf.d",
         "category": "correction", "importance": 0.95, "access_count": 4,
         "created_at": _days_ago(1),  "rrf_score": 0.7, "relevance": 3},
        {"content": "Nginx serves the app on port 8080",
         "category": "fact",       "importance": 0.8, "access_count": 2,
         "created_at": _days_ago(3),  "rrf_score": 0.8, "relevance": 2},
        {"content": "SSL certs managed by certbot, auto-renewed",
         "category": "fact",       "importance": 0.75, "access_count": 1,
         "created_at": _days_ago(20), "rrf_score": 0.5, "relevance": 1},
        {"content": "tmux prefix is C-a",
         "category": "preference", "importance": 0.6, "access_count": 5,
         "created_at": _days_ago(5),  "rrf_score": 0.2, "relevance": 0},
        {"content": "Python 3.12 is on the server",
         "category": "fact",       "importance": 0.7, "access_count": 1,
         "created_at": _days_ago(8),  "rrf_score": 0.1, "relevance": 0},
    ]),
]


# ── metrics ───────────────────────────────────────────────────────────────────

def dcg(relevances: list[int], k: int) -> float:
    score = 0.0
    for i, r in enumerate(relevances[:k], start=1):
        score += (2**r - 1) / math.log2(i + 1)
    return score


def ndcg(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def _make_mock_mem(row: dict) -> dict:
    return {
        "id": id(row),
        "content": row["content"],
        "category": row["category"],
        "importance": row["importance"],
        "access_count": row["access_count"],
        "created_at": row["created_at"],
        "rrf_score": row["rrf_score"],
        "_relevance": row["relevance"],
    }


# ── benchmark runner ──────────────────────────────────────────────────────────

def run_eval(verbose: bool = False) -> dict:
    from core.memory.reranker import Reranker

    reranker = Reranker(feedback_collector=None)
    K = 5

    baseline_ndcgs = []
    reranked_ndcgs = []

    start = time.perf_counter()

    for case in TEST_CASES:
        query = case["query"]
        mems_data = case["memories"]
        mocks = [_make_mock_mem(r) for r in mems_data]

        # Baseline: sort by rrf_score descending
        baseline = sorted(mocks, key=lambda m: m["rrf_score"], reverse=True)
        baseline_rels = [m["_relevance"] for m in baseline]
        b_ndcg = ndcg(baseline_rels, K)
        baseline_ndcgs.append(b_ndcg)

        # Reranked
        reranked = reranker.rerank(mocks, query=query)
        reranked_rels = [m.get("_relevance", 0) for m in reranked]
        r_ndcg = ndcg(reranked_rels, K)
        reranked_ndcgs.append(r_ndcg)

        if verbose:
            delta = r_ndcg - b_ndcg
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
            print(f"  {arrow} [{query[:35]:35s}]  "
                  f"baseline={b_ndcg:.3f}  reranked={r_ndcg:.3f}  Δ={delta:+.3f}")

    elapsed = time.perf_counter() - start
    n = len(TEST_CASES)
    avg_baseline = sum(baseline_ndcgs) / n
    avg_reranked = sum(reranked_ndcgs) / n

    return {
        "k": K,
        "n_cases": n,
        "avg_baseline_ndcg": round(avg_baseline, 4),
        "avg_reranked_ndcg": round(avg_reranked, 4),
        "improvement": round(avg_reranked - avg_baseline, 4),
        "elapsed_s": round(elapsed, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Reranker NDCG evaluation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n=== Reranker Eval (NDCG@K) ===\n")
    metrics = run_eval(verbose=args.verbose)

    print("\nResults:")
    print(f"  Baseline NDCG@{metrics['k']}:  {metrics['avg_baseline_ndcg']:.4f}")
    print(f"  Reranked NDCG@{metrics['k']}:  {metrics['avg_reranked_ndcg']:.4f}")
    delta = metrics["improvement"]
    arrow = "↑" if delta >= 0 else "↓"
    print(f"  Improvement:       {arrow} {abs(delta):.4f}")
    print(f"  Elapsed:           {metrics['elapsed_s']:.4f}s")
    print()

    # Pass if reranker is at least as good as baseline
    if metrics["avg_reranked_ndcg"] < metrics["avg_baseline_ndcg"] - 0.05:
        print("  ⚠ REGRESSION: reranker significantly worse than baseline",
              file=sys.stderr)
        sys.exit(1)
    else:
        print("  ✓ PASS (no regression vs baseline)")


if __name__ == "__main__":
    main()
