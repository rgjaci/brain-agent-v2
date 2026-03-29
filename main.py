#!/usr/bin/env python3
"""brain_agent — CLI entry point.

Usage:
    python main.py              # start Textual TUI
    python main.py chat         # headless terminal chat
    python main.py bootstrap    # scan environment and store facts
    python main.py ingest PATH  # ingest file / directory
    python main.py stats        # print memory statistics
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def build_agent(args):
    """Wire all components and return (BrainAgent, AgentConfig)."""
    from core.config import AgentConfig
    from core.llm.provider import OllamaProvider
    from core.llm.embeddings import GeminiEmbeddingProvider
    from core.memory.database import MemoryDatabase
    from core.memory.kg import KnowledgeGraph
    from core.memory.writer import MemoryWriter
    from core.memory.reader import MemoryReader
    from core.memory.procedures import ProcedureStore
    from core.memory.reranker import Reranker
    from core.memory.feedback import RetrievalFeedbackCollector
    from core.memory.consolidation import ConsolidationEngine
    from core.context.assembler import ContextAssembler
    from core.context.compressor import HistoryCompressor
    from core.tools.executor import ToolExecutor
    from core.tools.bash import BashTool
    from core.tools.file_ops import FileOpsTool
    from core.tools.web_search import WebSearchTool
    from core.tools.teach import TeachTool
    from core.tools.ingest import IngestTool
    from core.agent import BrainAgent

    cfg = AgentConfig(
        model=getattr(args, "model", "qwen3.5:4b-nothink"),
        db_path=getattr(args, "db", "~/.brain_agent/memory.db"),
        debug=getattr(args, "debug", False),
    )

    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
    )

    db_path = Path(cfg.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db       = MemoryDatabase(str(db_path))
    llm      = OllamaProvider(model=cfg.model)
    embedder = GeminiEmbeddingProvider()
    kg       = KnowledgeGraph(db)

    feedback    = RetrievalFeedbackCollector(db)
    writer      = MemoryWriter(llm, embedder, db, kg)
    reader      = MemoryReader(db, embedder, kg, llm)
    procedures  = ProcedureStore(db)
    reranker    = Reranker(feedback_collector=feedback)
    consolidate = ConsolidationEngine(db, llm, embedder)
    assembler   = ContextAssembler(llm)
    compressor  = HistoryCompressor(llm)

    executor = ToolExecutor(tools={
        "bash":       BashTool(cfg),
        "read_file":  FileOpsTool(cfg),
        "web_search": WebSearchTool(db, writer),
        "teach":      TeachTool(db, writer),
        "ingest":     IngestTool(db, writer),
    })

    agent = BrainAgent(
        llm=llm,
        embedder=embedder,
        db=db,
        reader=reader,
        writer=writer,
        assembler=assembler,
        tool_executor=executor,
        config=cfg,
        consolidation=consolidate,
        procedures=procedures,
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
            print("Bootstrapping…")
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
        print(f"TUI unavailable ({exc}) — falling back to chat mode\n")
        await cmd_chat(agent, args)


async def cmd_bootstrap(agent, args) -> None:
    print("Running environment scan…")
    await agent.bootstrap()
    print("Bootstrap complete.")


async def cmd_ingest(agent, args) -> None:
    from core.memory.documents import DocumentIngester
    ingester = DocumentIngester(agent.db, agent.writer)
    print(f"Ingesting: {args.path}")
    result = await ingester.ingest_file(args.path, agent.session())
    print(f"  chunks={result.get('chunks', 0)}  "
          f"status={result.get('status', '?')}")
    if result.get("message"):
        print(f"  {result['message']}")


async def cmd_stats(agent, args) -> None:
    await _print_stats(agent)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brain_agent",
        description="Memory-augmented AI agent — 4B model + exceptional memory",
    )
    p.add_argument("--model", default="qwen3.5:4b-nothink",
                   help="Ollama model name (default: qwen3.5:4b-nothink)")
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
        "stats":     lambda: cmd_stats(agent, args),
    }
    try:
        asyncio.run(handlers[cmd]())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
