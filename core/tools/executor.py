"""
Tool executor — central dispatcher that routes tool calls to the right tool.

Supports:
  bash, read_file, write_file, edit_file, web_search, teach, recall, ingest
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: str = ""


class ToolExecutor:
    """Route tool calls to the correct tool implementation."""

    def __init__(
        self,
        permissions: Optional[dict] = None,
        db=None,
        embedder=None,
        writer=None,
        ingester=None,
    ):
        self.db = db
        self.embedder = embedder

        # Import here to avoid circular issues at module load time
        from .bash import BashTool
        from .file_ops import FileOpsTool
        from .web_search import WebSearchTool
        from .teach import TeachTool
        from .ingest import IngestTool

        self.bash_tool = BashTool(permissions=permissions)
        self.file_ops = FileOpsTool(permissions=permissions)
        self.web_search_tool = WebSearchTool(db=db, writer=writer, permissions=permissions)
        self.teach_tool = TeachTool(db=db, embedder=embedder) if db else None
        self.ingest_tool = IngestTool(ingester=ingester)

        # Tool registry: name → handler method
        self._handlers = {
            "bash": self._run_bash,
            "read_file": self._run_read_file,
            "write_file": self._run_write_file,
            "edit_file": self._run_edit_file,
            "web_search": self._run_web_search,
            "teach": self._run_teach,
            "recall": self._run_recall,
            "ingest": self._run_ingest,
        }

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        """Dispatch a tool call. Always returns ToolResult (never raises)."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: {tool_name!r}. Available: {self.get_available_tools()}",
            )

        try:
            output = await handler(params)
            return ToolResult(tool_name=tool_name, success=True, output=output)
        except Exception as e:
            logger.exception(f"Tool {tool_name!r} raised an exception")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=str(e),
            )

    # ------------------------------------------------------------------ #
    #  Handlers                                                            #
    # ------------------------------------------------------------------ #

    async def _run_bash(self, params: dict) -> str:
        command = params.get("command", "").strip()
        timeout = params.get("timeout", None)
        workdir = params.get("workdir", None)
        if timeout is not None:
            timeout = int(timeout)
        result = await self.bash_tool.execute(command, timeout=timeout, workdir=workdir)
        return self.bash_tool.format_result(result)

    async def _run_read_file(self, params: dict) -> str:
        path = params.get("path", "").strip()
        start = params.get("start_line")
        end = params.get("end_line")
        if start is not None:
            start = int(start)
        if end is not None:
            end = int(end)
        result = self.file_ops.read_file(path, start_line=start, end_line=end)
        return self.file_ops.format_result(result)

    async def _run_write_file(self, params: dict) -> str:
        path = params.get("path", "").strip()
        content = params.get("content", "")
        append = bool(params.get("append", False))
        result = self.file_ops.write_file(path, content, append=append)
        return self.file_ops.format_result(result)

    async def _run_edit_file(self, params: dict) -> str:
        path = params.get("path", "").strip()
        old_str = params.get("old_str", "")
        new_str = params.get("new_str", "")
        replace_all = bool(params.get("replace_all", False))
        result = self.file_ops.edit_file(path, old_str, new_str, replace_all=replace_all)
        return self.file_ops.format_result(result)

    async def _run_web_search(self, params: dict) -> str:
        query = params.get("query", "").strip()
        max_results = int(params.get("max_results", 5))
        result = await self.web_search_tool.execute(query, max_results=max_results)
        return self.web_search_tool.format_result(result)

    async def _run_teach(self, params: dict) -> str:
        if self.teach_tool is None:
            return "[TEACH ERROR] Database not configured"
        content = params.get("content", "").strip()
        category = params.get("category", "fact")
        result = await self.teach_tool.execute(content, category=category)
        return self.teach_tool.format_result(result)

    async def _run_recall(self, params: dict) -> str:
        """Explicit memory search — hybrid FTS5 + vector search with RRF merge."""
        if self.db is None:
            return "[RECALL ERROR] Database not configured"
        query = params.get("query", "").strip()
        if not query:
            return "[RECALL ERROR] Empty query"
        try:
            import re
            safe_query = re.sub(r'[^\w\s]', ' ', query).strip()

            # Sparse (FTS5) search
            sparse_hits = self.db.fts_search(safe_query, limit=20) if safe_query else []

            # Dense (vector) search — if embedder available
            dense_hits = []
            if self.embedder is not None:
                try:
                    embedding = self.embedder.embed_query(query)
                    dense_hits = self.db.vector_search(
                        embedding=embedding,
                        table="memory_vectors",
                        id_col="memory_id",
                        limit=20,
                    )
                except Exception:
                    pass  # graceful degradation

            # RRF fusion
            k = 60
            scores: dict[int, float] = {}
            for rank, item in enumerate(sparse_hits):
                mid = item["id"]
                scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
            for rank, item in enumerate(dense_hits):
                mid = item["id"]
                scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)

            if not scores:
                return f'No memories found for: "{query}"'

            # Sort by RRF score descending, take top 10
            sorted_ids = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)[:10]

            lines = [f'Memory search results for "{query}":']
            for i, mid in enumerate(sorted_ids, 1):
                mem = self.db.get_memory(mid)
                if not mem:
                    continue
                content = mem.get("content", "")
                category = mem.get("category", "fact")
                lines.append(f"{i}. [{category}] {content}")
            return "\n".join(lines)
        except Exception as e:
            return f"[RECALL ERROR] {e}"

    def _check_network_permission(self, tool_name: str) -> bool:
        """Check whether a tool is allowed to make network requests.

        Args:
            tool_name: Name of the tool requesting network access.

        Returns:
            True if network access is permitted for this tool.
        """
        # Tools that inherently require network access
        network_tools = {"web_search", "ingest"}
        return tool_name in network_tools

    async def _run_ingest(self, params: dict) -> str:
        path = params.get("path", "").strip()
        recursive = bool(params.get("recursive", False))
        result = await self.ingest_tool.execute(path, recursive=recursive)
        return self.ingest_tool.format_result(result)

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def get_available_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._handlers.keys())

    def format_tool_result(self, tool_name: str, output: str) -> str:
        """Wrap output in XML tool_result tags for LLM context injection."""
        return f'<tool_result name="{tool_name}">\n{output}\n</tool_result>'
