#!/usr/bin/env python3
"""Export RAFT fine-tuning data from the memory database.

Queries retrieval_log + conversations tables and exports
(query, retrieved_docs, answer, was_useful) as JSONL with
distractor documents mixed in for Qwen/Llama fine-tuning.

Usage:
    python benchmarks/export_raft_data.py --db ~/.brain_agent/memory.db --output raft_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def export_raft_data(db_path: str, output_path: str, max_distractors: int = 3) -> int:
    """Export RAFT training data from the database.

    Args:
        db_path:         Path to the SQLite database.
        output_path:     Output JSONL file path.
        max_distractors: Number of distractor documents to mix in per example.

    Returns:
        Number of examples exported.
    """
    from core.memory.database import MemoryDatabase

    db = MemoryDatabase(db_path)

    # Fetch all retrieval log entries with useful signal
    log_rows = db.execute(
        """
        SELECT rl.session_id, rl.query_text, rl.memory_id, rl.was_useful,
               rl.retrieval_rank, rl.retrieval_score
          FROM retrieval_log rl
         WHERE rl.was_useful IS NOT NULL
         ORDER BY rl.session_id, rl.created_at
        """
    )

    if not log_rows:
        print("No retrieval log data found.", file=sys.stderr)
        return 0

    # Group by (session_id, query_text)
    from collections import defaultdict
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in log_rows:
        key = (row["session_id"], row["query_text"])
        groups[key].append(row)

    # Fetch all memory contents for distractor pool
    all_memories = db.execute(
        "SELECT id, content FROM memories WHERE superseded_by IS NULL LIMIT 5000"
    )
    memory_map = {m["id"]: m["content"] for m in all_memories}
    all_memory_ids = list(memory_map.keys())

    # Find corresponding answers from conversations
    examples = []
    for (session_id, query_text), retrievals in groups.items():
        # Get the assistant response that followed this query
        conv_rows = db.execute(
            """
            SELECT content FROM conversations
             WHERE session_id = ? AND role = 'assistant'
             ORDER BY created_at
             LIMIT 1
            """,
            (session_id,),
        )
        if not conv_rows:
            continue

        answer = conv_rows[0]["content"]

        # Collect retrieved docs
        retrieved_docs = []
        for r in retrievals:
            mid = r["memory_id"]
            content = memory_map.get(mid, "")
            if content:
                retrieved_docs.append({
                    "content": content,
                    "was_useful": bool(r["was_useful"]),
                    "rank": r["retrieval_rank"],
                })

        if not retrieved_docs:
            continue

        # Add distractors — random memories not in the retrieved set
        retrieved_ids = {r["memory_id"] for r in retrievals}
        distractor_pool = [mid for mid in all_memory_ids if mid not in retrieved_ids]
        n_distractors = min(max_distractors, len(distractor_pool))
        distractor_ids = random.sample(distractor_pool, n_distractors) if distractor_pool else []
        distractors = [
            {"content": memory_map[did], "was_useful": False, "rank": -1}
            for did in distractor_ids
            if did in memory_map
        ]

        # Interleave retrieved docs and distractors
        all_docs = retrieved_docs + distractors
        random.shuffle(all_docs)

        example = {
            "query": query_text,
            "documents": all_docs,
            "answer": answer,
            "session_id": session_id,
            "n_relevant": sum(1 for d in all_docs if d["was_useful"]),
            "n_distractors": len(distractors),
        }
        examples.append(example)

    # Write JSONL
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Exported {len(examples)} RAFT training examples to {output_path}")
    return len(examples)


def main():
    parser = argparse.ArgumentParser(description="Export RAFT fine-tuning data")
    parser.add_argument("--db", default="~/.brain_agent/memory.db",
                        help="Path to SQLite database")
    parser.add_argument("--output", default="raft_data.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--distractors", type=int, default=3,
                        help="Max distractor documents per example")
    args = parser.parse_args()

    db_path = str(Path(args.db).expanduser())
    export_raft_data(db_path, args.output, args.distractors)


if __name__ == "__main__":
    main()
