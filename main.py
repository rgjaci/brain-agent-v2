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
    from core.config import AgentConfig
    from core.memory.database import MemoryDatabase
    from core.memory.kg import KnowledgeGraph
    from core.memory.feedback import RetrievalFeedbackCollector
    from core.memory.consolidation import ConsolidationEngine
    from core.context.assembler import ContextAssembler
    from core.context.compressor import HistoryCompressor
    from core.agent import BrainAgent

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
    )
    return agent, cfg


async def _print_stats(agent) -> None:
    try:
        db = agent.db
        def _n(q): return db.execute(q)[0]["n"]
        print(
            f"  memories   = {_n('SELECT COUNT(*) n FROM memories'):,}\n"
            f"  entities   = {_n('SELECT COUNT(*) n FROM entities'):,}\n"
            f"  relations  = {_n('SELECT COUNT(*) n FROM relations'):,}\n"
            f"  procedures = {_n('SELECT COUNT(*) n FROM procedures'):,}"
        )
    except Exception as exc:
        print(f"  (stats unavailable: {exc})")


async def cmd_chat(agent, args) -> None:
    print("Brain Agent  |  ctrl+c or /quit to exit")
    print("Commands:  /stats  /clear  /bootstrap  /quit")
    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
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
                import traceback; traceback.print_exc()


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
    """Search memories by query."""
    query = args.query
    if not query:
        print("Error: empty query")
        return
    import re
    safe = re.sub(r'[^\w\s]', ' ', query).strip()
    # Use OR so any matching term returns results
    fts_query = " OR ".join(safe.split()) if safe else ""
    results = agent.db.fts_search(fts_query, limit=10) if fts_query else []
    if not results:
        print(f'No memories found for: "{query}"')
        return
    for i, hit in enumerate(results, 1):
        mem = agent.db.get_memory(hit["id"])
        if mem:
            print(f"  {i}. [{mem.get('category', '?')}] {mem.get('content', '')}")


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
    }
    try:
        asyncio.run(handlers[cmd]())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
