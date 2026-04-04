#!/usr/bin/env python3
"""brain_agent — CLI entry point.

Usage:
    python main.py              # start Textual TUI
    python main.py chat         # headless terminal chat
    python main.py bootstrap    # scan environment and store facts
    python main.py ingest PATH  # ingest file / directory
    python main.py teach TEXT   # teach a fact to the agent
    python main.py recall QUERY # search memories
    python main.py stats        # print memory statistics
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def build_agent(args=None):
    """Wire all components and return (BrainAgent, AgentConfig).

    Gracefully degrades when LLM or embedder are unavailable.
    """
    from core.agent import BrainAgent
    from core.config import AgentConfig
    from core.context.assembler import ContextAssembler
    from core.memory.consolidation import ConsolidationEngine
    from core.memory.database import MemoryDatabase
    from core.memory.feedback import RetrievalFeedbackCollector
    from core.memory.kg import KnowledgeGraph

    cfg = AgentConfig.load()
    if hasattr(args, "model") and getattr(args, "model", None):
        cfg.model = args.model

    debug = getattr(args, "debug", False) if args else False
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
    )

    db_path = Path(getattr(args, "db", "~/.brain_agent/memory.db") if args else "~/.brain_agent/memory.db").expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = MemoryDatabase(str(db_path))

    # LLM — graceful degradation
    llm = None
    try:
        from core.llm.provider import OllamaProvider
        llm = OllamaProvider.from_env()
    except Exception as exc:
        logging.warning("LLM unavailable (%s) — running without LLM.", exc)

    # Embedder — graceful degradation with local fallback
    embedder = None
    try:
        from core.llm.embeddings import GeminiEmbeddingProvider
        embedder = GeminiEmbeddingProvider(api_key=cfg.gemini_api_key)
        logging.info("Using Gemini embedding provider.")
    except Exception as exc:
        logging.warning("Gemini embedder unavailable (%s).", exc)
        try:
            from core.llm.embeddings import LocalEmbeddingProvider
            embedder = LocalEmbeddingProvider()
            logging.info("Falling back to local embedding provider.")
        except Exception as exc2:
            logging.warning(
                "Local embedder also unavailable (%s) — running without embeddings.", exc2
            )

    kg = KnowledgeGraph(db)
    feedback = RetrievalFeedbackCollector(db)

    # Writer — needs LLM + embedder
    writer = None
    if llm and embedder:
        from core.memory.writer import MemoryWriter
        writer = MemoryWriter(llm, embedder, db, kg)

    # Reader — needs embedder
    reader = None
    if embedder:
        from core.memory.reader import MemoryReader
        reader = MemoryReader(db, embedder, kg, llm)

    consolidator = ConsolidationEngine(db, llm, embedder)

    # AutoDream engine
    dream_engine = None
    if llm and embedder and cfg.dream_enabled:
        from core.memory.dream import DreamEngine
        dream_engine = DreamEngine(db, llm, embedder, consolidator, kg)

    # System 2 Reasoning engine
    reasoning_engine = None
    if llm and embedder and cfg.reasoning_enabled:
        from core.memory.reasoning import ReasoningEngine
        reasoning_engine = ReasoningEngine(
            db=db, llm=llm, embedder=embedder, kg=kg,
            interval=cfg.reasoning_interval,
            max_cycles=cfg.reasoning_max_cycles_per_session,
        )

    assembler = None
    if llm:
        assembler = ContextAssembler(llm)

    # Tool executor — uses actual constructors
    executor = None
    try:
        from core.tools.executor import ToolExecutor
        executor = ToolExecutor(
            permissions=None,
            db=db,
            embedder=embedder,
            writer=writer,
        )
    except Exception as exc:
        logging.warning("Tool executor init failed: %s", exc)

    agent = BrainAgent(
        llm=llm,
        embedder=embedder,
        db=db,
        reader=reader,
        writer=writer,
        assembler=assembler,
        tool_executor=executor,
        feedback=feedback,
        dream_engine=dream_engine,
        reasoning_engine=reasoning_engine,
    )
    return agent, cfg


async def _print_stats(agent) -> None:
    """Print comprehensive memory statistics."""
    try:
        db = agent.db
        def _n(q): return db.execute(q)[0]["n"]
        def _sum(q):
            r = db.execute(q)
            return r[0]["total"] if r and r[0]["total"] else 0

        print("\n" + "=" * 50)
        print("  Brain Agent — Memory Statistics")
        print("=" * 50)

        # Core counts
        print("\n  Core:")
        print(f"    memories   = {_n('SELECT COUNT(*) n FROM memories'):,}")
        print(f"    entities   = {_n('SELECT COUNT(*) n FROM entities'):,}")
        print(f"    relations  = {_n('SELECT COUNT(*) n FROM relations'):,}")
        print(f"    procedures = {_n('SELECT COUNT(*) n FROM procedures'):,}")
        print(f"    documents  = {_n('SELECT COUNT(*) n FROM documents'):,}")

        # Memory breakdown by category
        print("\n  Memories by category:")
        cats = db.execute(
            "SELECT category, COUNT(*) n FROM memories GROUP BY category ORDER BY n DESC"
        )
        for cat in cats:
            print(f"    {cat['category']:20s} = {cat['n']:,}")

        # Memory age distribution
        print("\n  Memory age distribution:")
        ages = db.execute(
            """
            SELECT
                CASE
                    WHEN created_at > strftime('%s','now') - 86400 THEN 'Last 24h'
                    WHEN created_at > strftime('%s','now') - 604800 THEN 'Last 7 days'
                    WHEN created_at > strftime('%s','now') - 2592000 THEN 'Last 30 days'
                    ELSE 'Older'
                END as age_group,
                COUNT(*) n
            FROM memories
            GROUP BY age_group
            ORDER BY age_group
            """
        )
        for age in ages:
            print(f"    {age['age_group']:20s} = {age['n']:,}")

        # Access patterns
        print("\n  Access patterns:")
        print(f"    Total accesses    = {_sum('SELECT SUM(access_count) total FROM memories'):,}")
        print(f"    Avg accesses/mem  = {_sum('SELECT AVG(access_count) total FROM memories'):.1f}")
        print(f"    Most accessed     = {_n('SELECT MAX(access_count) n FROM memories'):,}")

        # Procedure stats
        print("\n  Procedures:")
        procs = db.execute(
            "SELECT success_count, failure_count FROM procedures"
        )
        total_success = sum(p["success_count"] for p in procs)
        total_failure = sum(p["failure_count"] for p in procs)
        total_attempts = total_success + total_failure
        success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
        print(f"    Total successes   = {total_success:,}")
        print(f"    Total failures    = {total_failure:,}")
        print(f"    Success rate      = {success_rate:.1f}%")

        # Database size
        import os
        db_path = getattr(db, 'db_path', None)
        if db_path and os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"\n  Database size: {size_mb:.1f} MB")

        print()
    except Exception as exc:
        print(f"  (stats unavailable: {exc})")


async def cmd_chat(agent, args) -> None:
    print("Brain Agent  |  ctrl+c or /quit to exit")
    print("Commands:  /stats  /clear  /bootstrap  /good  /bad  /quit")
    print("Feedback:  /good (thumbs up)  /bad (thumbs down)")
    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/stats":
            await _print_stats(agent)
            continue
        if line == "/clear":
            agent.new_session()
            print("Session cleared.")
            continue
        if line == "/bootstrap":
            print("Bootstrapping...")
            await agent.bootstrap()
            print("Done.")
            continue
        if line == "/good":
            agent.record_feedback(accepted=True)
            print("👍 Feedback recorded — response was helpful")
            continue
        if line == "/bad":
            agent.record_feedback(accepted=False)
            print("👎 Feedback recorded — response was not helpful")
            continue
        try:
            result = await agent.process(line)
            print(f"\nAgent: {result.response}")
            if getattr(args, "debug", False):
                tools_used = [t.get("name") for t in (result.tool_calls or [])]
                print(f"  [mem={result.memories_used}  "
                      f"tok={result.tokens_used}"
                      f"{'  tools=' + str(tools_used) if tools_used else ''}]")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            if getattr(args, "debug", False):
                import traceback
                traceback.print_exc()


async def cmd_run(agent, cfg, args) -> None:
    try:
        from tui.app import BrainAgentApp
        app = BrainAgentApp(agent=agent, config=cfg)
        await app.run_async()
    except ImportError as exc:
        print(f"TUI unavailable ({exc}) -- falling back to chat mode\n")
        await cmd_chat(agent, args)


async def cmd_bootstrap(agent, args) -> None:
    print("Running environment scan...")
    await agent.bootstrap()
    print("Bootstrap complete.")


async def cmd_ingest(agent, args) -> None:
    from core.memory.documents import DocumentIngester
    ingester = DocumentIngester(agent.db, agent.writer)
    print(f"Ingesting: {args.path}")
    result = await ingester.ingest_file(args.path, agent.session_id)
    print(f"  chunks={result.get('chunks', 0)}  "
          f"status={result.get('status', '?')}")
    if result.get("message"):
        print(f"  {result['message']}")


async def cmd_teach(agent, args) -> None:
    """Store a fact directly into memory."""
    content = args.text
    if not content:
        print("Error: empty text")
        return
    mid = agent.db.insert_memory(
        content=content,
        category="fact",
        source="teach",
        importance=0.7,
    )
    print(f"Stored memory #{mid}: {content}")


async def cmd_recall(agent, args) -> None:
    """Search memories by query with optional filtering and export."""
    query = args.query
    if not query:
        print("Error: empty query")
        return

    category = getattr(args, "category", None)
    limit = getattr(args, "limit", 10)
    offset = getattr(args, "offset", 0)
    verbose = getattr(args, "verbose", False)
    export = getattr(args, "export", None)

    import json
    import re
    safe = re.sub(r'[^\w\s]', ' ', query).strip()
    fts_query = " OR ".join(safe.split()) if safe else ""

    # Build query with optional category filter
    if category:
        results = agent.db.execute(
            """
            SELECT id, content, category, source, importance, confidence,
                   access_count, created_at, last_accessed, usefulness_score
              FROM memories
             WHERE category = ?
             ORDER BY importance DESC, created_at DESC
             LIMIT ? OFFSET ?
            """,
            (category, limit, offset),
        )
    elif fts_query:
        # FTS search with category filter
        raw_results = agent.db.fts_search(fts_query, limit=limit + offset)
        results = []
        for i, hit in enumerate(raw_results):
            if i < offset:
                continue
            mem = agent.db.get_memory(hit["id"])
            if mem:
                results.append(mem)
    else:
        results = agent.db.execute(
            """
            SELECT id, content, category, source, importance, confidence,
                   access_count, created_at, last_accessed, usefulness_score
              FROM memories
             ORDER BY importance DESC, created_at DESC
             LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

    if not results:
        print(f'No memories found for: "{query}"')
        return

    if export:
        # Export as JSON
        export_data = []
        for mem in results:
            export_data.append({
                "id": mem.get("id"),
                "content": mem.get("content", ""),
                "category": mem.get("category", ""),
                "source": mem.get("source", ""),
                "importance": mem.get("importance", 0),
                "confidence": mem.get("confidence", 0),
                "access_count": mem.get("access_count", 0),
                "created_at": mem.get("created_at", 0),
            })
        with open(export, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported {len(export_data)} memories to {export}")
        return

    # Display results
    if verbose:
        print(f'\nMemory search results for "{query}" ({len(results)} shown, offset={offset}):')
        print("-" * 80)
        for i, mem in enumerate(results, offset + 1):
            content = mem.get("content", "")
            category = mem.get("category", "?")
            importance = mem.get("importance", 0)
            confidence = mem.get("confidence", 0)
            access_count = mem.get("access_count", 0)
            source = mem.get("source", "")
            created = mem.get("created_at", 0)
            print(f"\n{i}. [{category}] (importance={importance:.2f}, "
                  f"confidence={confidence:.2f}, accessed={access_count}x)")
            print(f"   Source: {source}")
            print(f"   Content: {content}")
            if created:
                from datetime import datetime
                print(f"   Created: {datetime.fromtimestamp(created).isoformat()}")
    else:
        print(f'\nMemory search results for "{query}":')
        for i, mem in enumerate(results, offset + 1):
            content = mem.get("content", "")
            category = mem.get("category", "?")
            print(f"  {i}. [{category}] {content}")

        if len(results) >= limit:
            print(f"\n  (showing {limit} of many — use --offset {offset + limit} for more)")


async def cmd_dream(agent, args) -> None:
    """Trigger an AutoDream cycle manually."""
    if not agent.dream_engine:
        print("AutoDream is not available (requires LLM + embedder).")
        return
    print("Starting AutoDream cycle (LLM-powered memory consolidation)...")
    report = await agent.dream_engine.dream()
    print(f"  abstractions created:   {report.abstractions_created}")
    print(f"  contradictions resolved: {report.contradictions_resolved}")
    print(f"  patterns detected:       {report.patterns_detected}")
    print(f"  connections added:       {report.connections_added}")
    print(f"  questions generated:     {report.questions_generated}")
    print(f"  elapsed: {report.elapsed_seconds:.1f}s")
    if report.errors:
        print(f"  warnings: {len(report.errors)}")


async def cmd_serve(agent, cfg, args) -> None:
    """Start the MCP server for external agent integration."""
    print("Starting Brain Agent MCP server...")
    print(f"  transport: {cfg.mcp_transport}")
    if cfg.mcp_transport == "sse":
        print(f"  host: {cfg.mcp_host}:{cfg.mcp_port}")
    from server.mcp_server import mcp as mcp_server
    mcp_server.run()


async def cmd_stats(agent, args) -> None:
    await _print_stats(agent)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brain_agent",
        description="Memory-augmented AI agent -- 4B model + exceptional memory",
    )
    p.add_argument("--model", default="qwen3.5:4b",
                   help="Ollama model name (default: qwen3.5:4b)")
    p.add_argument("--db",    default="~/.brain_agent/memory.db",
                   help="Path to SQLite database")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose debug logging")

    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("chat",      help="Headless terminal chat (no TUI)")
    sub.add_parser("bootstrap", help="Scan environment and store facts")
    sub.add_parser("stats",     help="Print memory statistics")

    ing = sub.add_parser("ingest", help="Ingest file or directory into memory")
    ing.add_argument("path", help="File or directory to ingest")

    teach_p = sub.add_parser("teach", help="Teach a fact to the agent")
    teach_p.add_argument("text", help="Fact to store")

    recall_p = sub.add_parser("recall", help="Search memories")
    recall_p.add_argument("query", help="Search query")
    recall_p.add_argument("--category", "-c", help="Filter by category (fact, preference, etc.)")
    recall_p.add_argument("--limit", "-n", type=int, default=10, help="Max results (default: 10)")
    recall_p.add_argument("--offset", "-o", type=int, default=0, help="Skip N results")
    recall_p.add_argument("--verbose", "-v", action="store_true", help="Show full details")
    recall_p.add_argument("--export", "-e", help="Export results to JSON file")

    sub.add_parser("dream", help="Trigger AutoDream (LLM-powered memory consolidation)")

    serve_p = sub.add_parser("serve", help="Start MCP server for external agent integration")
    serve_p.add_argument("--transport", default="stdio",
                         choices=["stdio", "sse"],
                         help="MCP transport (default: stdio)")
    serve_p.add_argument("--port", type=int, default=8765,
                         help="Port for SSE transport (default: 8765)")

    return p


def main() -> None:
    args = make_parser().parse_args()
    agent, cfg = build_agent(args)
    cmd = args.cmd or "run"
    handlers = {
        "run":       lambda: cmd_run(agent, cfg, args),
        "chat":      lambda: cmd_chat(agent, args),
        "bootstrap": lambda: cmd_bootstrap(agent, args),
        "ingest":    lambda: cmd_ingest(agent, args),
        "teach":     lambda: cmd_teach(agent, args),
        "recall":    lambda: cmd_recall(agent, args),
        "stats":     lambda: cmd_stats(agent, args),
        "dream":     lambda: cmd_dream(agent, args),
        "serve":     lambda: cmd_serve(agent, cfg, args),
    }
    try:
        asyncio.run(handlers[cmd]())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
