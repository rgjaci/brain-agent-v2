"""Procedure retrieval benchmark — measures UCB ranking and match quality.

Usage:
    python -m benchmarks.procedure_test [--verbose]

Inserts a synthetic set of procedures and tests whether the correct one
surfaces for various triggering queries.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import NamedTuple


PROCEDURES = [
    {
        "name": "deploy_nginx",
        "description": "Deploy and configure Nginx as a reverse proxy",
        "trigger_pattern": "nginx deploy reverse proxy web server",
        "steps": [
            "Install nginx: apt-get install -y nginx",
            "Write config to /etc/nginx/sites-available/app",
            "Create symlink: ln -s sites-available/app sites-enabled/",
            "Test config: nginx -t",
            "Reload: systemctl reload nginx",
        ],
        "warnings": ["Backup existing config first"],
    },
    {
        "name": "setup_ssh_keys",
        "description": "Generate and deploy SSH keys for passwordless authentication",
        "trigger_pattern": "ssh key setup authorized authentication",
        "steps": [
            "Generate key: ssh-keygen -t ed25519 -C 'user@host'",
            "Copy to server: ssh-copy-id user@server",
            "Test: ssh -o PasswordAuthentication=no user@server",
            "Disable password auth in /etc/ssh/sshd_config",
        ],
        "warnings": ["Keep private key secure", "Test before disabling password auth"],
    },
    {
        "name": "postgres_backup",
        "description": "Backup PostgreSQL database using pg_dump",
        "trigger_pattern": "postgres backup dump database export",
        "steps": [
            "pg_dump -U postgres dbname > /backups/dbname_$(date +%Y%m%d).sql",
            "Verify: ls -lh /backups/",
            "Optionally compress: gzip /backups/dbname_*.sql",
        ],
        "warnings": ["Ensure /backups/ has enough space"],
    },
    {
        "name": "setup_tailscale",
        "description": "Install and configure Tailscale VPN",
        "trigger_pattern": "tailscale vpn mesh network private",
        "steps": [
            "curl -fsSL https://tailscale.com/install.sh | sh",
            "tailscale up --authkey=<key>",
            "Verify: tailscale status",
        ],
        "warnings": ["Auth key expires — use reusable key for automation"],
    },
    {
        "name": "systemd_service",
        "description": "Create and enable a systemd service unit",
        "trigger_pattern": "systemd service unit daemon autostart",
        "steps": [
            "Write /etc/systemd/system/myapp.service",
            "systemctl daemon-reload",
            "systemctl enable myapp",
            "systemctl start myapp",
            "systemctl status myapp",
        ],
        "warnings": ["Check logs: journalctl -u myapp -f"],
    },
]

QUERIES = [
    # (query, expected_procedure_name)
    ("how do I set up nginx as a reverse proxy?",        "deploy_nginx"),
    ("configure nginx web server",                        "deploy_nginx"),
    ("generate SSH keys for authentication",             "setup_ssh_keys"),
    ("how to do passwordless SSH login?",                "setup_ssh_keys"),
    ("backup the postgres database",                     "postgres_backup"),
    ("pg_dump database export",                          "postgres_backup"),
    ("install Tailscale VPN on the server",              "setup_tailscale"),
    ("set up private mesh network with tailscale",       "setup_tailscale"),
    ("create a systemd service to autostart my app",     "systemd_service"),
    ("make my app start on boot with systemd",           "systemd_service"),
]


class ProcResult(NamedTuple):
    query: str
    expected: str
    got: str | None
    rank: int
    found: bool


def run_benchmark(verbose: bool = False) -> dict:
    from core.memory.database import MemoryDatabase
    from core.memory.procedures import ProcedureStore, Procedure

    db = MemoryDatabase(":memory:")
    store = ProcedureStore(db)

    # Insert procedures
    proc_ids: dict[str, int] = {}
    for p in PROCEDURES:
        proc = Procedure(
            id=None,
            name=p["name"],
            description=p["description"],
            trigger_pattern=p["trigger_pattern"],
            preconditions=[],
            steps=p["steps"],
            warnings=p.get("warnings", []),
            context="",
        )
        pid = store.store(proc)
        proc_ids[p["name"]] = pid
        # Simulate some usage history: deploy_nginx and setup_ssh_keys are popular
        if p["name"] in ("deploy_nginx", "setup_ssh_keys"):
            for _ in range(5):
                store.record_success(pid)

    results: list[ProcResult] = []
    start = time.perf_counter()

    for query, expected_name in QUERIES:
        retrieved = store.find_relevant(query, max_results=5)
        names = [p.name for p in retrieved]

        rank = 0
        for i, name in enumerate(names, start=1):
            if name == expected_name:
                rank = i
                break

        found = rank > 0
        got = names[0] if names else None
        results.append(ProcResult(query=query, expected=expected_name,
                                   got=got, rank=rank, found=found))

        if verbose:
            status = f"✓ rank={rank}" if found else f"✗ got={got}"
            print(f"  [{status:14s}] {query[:55]}")

    elapsed = time.perf_counter() - start
    n = len(results)
    recall = sum(r.found for r in results) / n
    mrr = sum(1.0 / r.rank for r in results if r.rank > 0) / n

    return {
        "total_queries": n,
        "recall": round(recall, 4),
        "mrr": round(mrr, 4),
        "found": sum(r.found for r in results),
        "elapsed_s": round(elapsed, 3),
        "avg_ms_per_query": round(elapsed / n * 1000, 2) if n else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Brain Agent procedure benchmark")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n=== Procedure Retrieval Benchmark ===\n")
    metrics = run_benchmark(verbose=args.verbose)

    print(f"\nResults:")
    print(f"  Recall:  {metrics['recall']:.1%}")
    print(f"  MRR:     {metrics['mrr']:.4f}")
    print(f"  Found:   {metrics['found']} / {metrics['total_queries']}")
    print(f"  Elapsed: {metrics['elapsed_s']:.3f}s "
          f"({metrics['avg_ms_per_query']:.1f}ms/query)")
    print()

    threshold = 0.5
    if metrics["recall"] < threshold:
        print(f"  ⚠ BELOW THRESHOLD ({threshold:.0%})", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"  ✓ PASS (threshold: {threshold:.0%})")


if __name__ == "__main__":
    main()
