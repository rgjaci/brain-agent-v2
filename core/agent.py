"""
BrainAgent — main agent loop.

Flow per turn:
  user_input
    → store conversation
    → adaptive retrieval (memory + KG + procedures)
    → context assembly (32K budget)
    → LLM call + tool loop (max 10 iterations)
    → store response
    → async: extract knowledge (facts + entities + relations + procedures)
    → async: log retrieval outcome
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Bootstrap scan commands
BOOTSTRAP_SCANS = [
    ("Shell config", "cat ~/.zshrc 2>/dev/null || cat ~/.bashrc 2>/dev/null"),
    ("Git config", "git config --global --list 2>/dev/null"),
    ("Python", "python3 --version 2>/dev/null && which python3"),
    ("Node", "node --version 2>/dev/null && which node 2>/dev/null"),
    ("Docker", "docker --version 2>/dev/null"),
    ("Ollama models", "ollama list 2>/dev/null"),
    ("OS info", "uname -a"),
    ("Disk usage", "df -h / | tail -1"),
    ("Recent projects", "find ~/projects -maxdepth 1 -type d 2>/dev/null | head -20"),
]


@dataclass
class TurnResult:
    """Result of one agent turn."""
    response: str
    tool_calls: list[dict] = field(default_factory=list)
    memories_used: int = 0
    procedure_used: str | None = None
    tokens_used: int = 0
    error: str | None = None


class BrainAgent:
    """
    Main agent loop.

    Components (all optional to allow partial initialization):
      - llm: OllamaProvider (or any LLMProvider)
      - embedder: GeminiEmbeddingProvider (or any EmbeddingProvider)
      - db: MemoryDatabase
      - reader: MemoryReader
      - writer: MemoryWriter
      - assembler: ContextAssembler
      - tool_executor: ToolExecutor
      - feedback: RetrievalFeedbackCollector
    """

    def __init__(
        self,
        llm=None,
        embedder=None,
        db=None,
        reader=None,
        writer=None,
        assembler=None,
        tool_executor=None,
        feedback=None,
        consolidator=None,
        dream_engine=None,
        reasoning_engine=None,
        session_id: str | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.db = db
        self.reader = reader
        self.writer = writer
        self.assembler = assembler
        self.tool_executor = tool_executor
        self.feedback = feedback
        self.consolidator = consolidator
        self.dream_engine = dream_engine
        self.reasoning_engine = reasoning_engine
        self.session_id = session_id or str(uuid.uuid4())
        self.on_event = on_event  # TUI event hook: (event_type, data) -> None

        # State for current turn
        self._current_tool_calls: list[dict] = []
        self._turn_count: int = 0
        self._last_active: float = 0.0
        self._write_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    async def process(self, user_input: str) -> TurnResult:
        """Process one user message. Returns TurnResult."""
        # Let background writes complete asynchronously — don't block the
        # next conversation turn.  Facts will be available for retrieval
        # once the write finishes (typically 15-30s later).
        # Cancel only if it's been running for > 2 minutes (likely stuck).
        if self._write_task and not self._write_task.done():
            # Check if task has been running too long (> 120s)
            try:
                task_age = asyncio.get_event_loop().time() - getattr(
                    self._write_task, '_start_time', 0
                )
            except Exception:
                task_age = 0
            if task_age > 120:
                self._write_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._write_task
        self._current_tool_calls = []
        self._emit("turn_start", {"input": user_input})

        # 1. Store user message
        if self.db:
            self.db.store_conversation(self.session_id, "user", user_input)

        # 2. Retrieve relevant memory
        context = None
        memories_used = 0
        procedure_name = None

        # Build session context from recent history (not session_id)
        session_context_text = ""
        if self.db:
            try:
                recent = self.db.get_recent_messages(self.session_id, limit=5)
                session_context_text = " ".join(
                    m.get("content", "") for m in recent if m.get("role") == "user"
                )
            except Exception:
                pass

        procedure_id = None
        if self.reader:
            self._emit("retrieval_start", {"query": user_input})
            try:
                context = await self.reader.retrieve(user_input, session_context_text)
                memories_used = len(context.memories) if context else 0
                if context and context.procedures:
                    proc = context.procedures[0] if context.procedures else None
                    if proc:
                        procedure_name = (
                            proc.get("name") if isinstance(proc, dict)
                            else getattr(proc, "name", None)
                        )
                        procedure_id = (
                            proc.get("id") if isinstance(proc, dict)
                            else getattr(proc, "id", None)
                        )
                self._emit("retrieval_done", {
                    "memories": memories_used,
                    "procedure": procedure_name,
                    "kg_facts": len(context.kg_context.split("\n")) if (context and context.kg_context) else 0,
                })
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")

        # 3. Get recent chat history
        history = []
        if self.db:
            try:
                raw = self.db.get_recent_messages(self.session_id, limit=20)
                # Exclude the message we just stored (last user turn)
                history = raw[:-1] if raw and raw[-1].get("role") == "user" else raw
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

        # 4. Assemble full context
        messages = []
        if self.assembler:
            messages = self.assembler.assemble(
                procedure=context.procedures[0] if (context and context.procedures) else None,
                memories=context.memories if context else [],
                kg_context=context.kg_context if context else "",
                chat_history=history,
                query=user_input,
            )
            token_info = self.assembler.get_token_usage(messages)
            self._emit("context_assembled", token_info)
        else:
            # Minimal fallback
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input},
            ]

        # 5. LLM + tool loop
        if not self.llm:
            response = "[No LLM configured]"
            tokens_used = 0
        else:
            try:
                response, tokens_used = await self.tool_loop(messages, max_iterations=10)
            except Exception as e:
                logger.exception("LLM/tool loop failed")
                response = f"[Error: {e}]"
                tokens_used = 0

        # 6. Store response
        if self.db:
            self.db.store_conversation(self.session_id, "assistant", response)

        self._emit("response_ready", {"response": response[:200]})

        # 7. Async: extract and store new knowledge (fire-and-forget)
        if self.writer:
            self._write_task = asyncio.create_task(self._async_write(user_input, response))
            self._write_task._start_time = asyncio.get_event_loop().time()

        # 8. Retrieval feedback + procedure tracking
        if self.feedback and context and context.memories:
            try:
                # Record which memories were retrieved
                memory_dicts = []
                for m in context.memories:
                    if hasattr(m, '__dataclass_fields__'):
                        memory_dicts.append({
                            "id": m.id, "content": m.content,
                            "category": m.category, "importance": m.importance,
                            "access_count": m.access_count, "rrf_score": m.rrf_score,
                        })
                    elif isinstance(m, dict):
                        memory_dicts.append(m)
                self.feedback.record_retrieval(user_input, memory_dicts, self.session_id)
                # Implicit feedback: top 5 memories assumed referenced
                referenced_ids = [m["id"] for m in memory_dicts[:5]]
                self.feedback.record_reference(self.session_id, referenced_ids)
                # Flush to DB
                self.feedback.persist_retrieval_log(self.session_id)
            except Exception as e:
                logger.warning(f"Feedback recording failed: {e}")

        # 8b. Procedure success/failure tracking
        if procedure_id is not None and self.db:
            try:
                from core.memory.procedures import ProcedureStore
                proc_store = ProcedureStore(self.db)
                agent_lower = response.lower()
                failure_indicators = (
                    "error:", "failed:", "i couldn't", "i'm unable",
                    "unable to", "could not", "cannot complete", "exception:",
                    "traceback", "permission denied", "not found", "timed out",
                )
                if any(ind in agent_lower for ind in failure_indicators):
                    proc_store.record_failure(procedure_id)
                else:
                    proc_store.record_success(procedure_id)
            except Exception as e:
                logger.warning(f"Procedure tracking failed: {e}")

        # 9. Consolidation — run if idle > 300s
        import time
        self._turn_count += 1
        now = time.time()
        if self.consolidator and (now - self._last_active > 300 or self._last_active == 0):
            try:
                await self.consolidator.maybe_consolidate(self._turn_count)
            except Exception as e:
                logger.warning(f"Consolidation failed: {e}")
            # Auto-train feedback reranker every consolidation cycle
            if self.feedback and self._turn_count % 50 == 0:
                try:
                    self.feedback.maybe_auto_train(threshold=100)
                except Exception as e:
                    logger.warning(f"Auto-train failed: {e}")

        # 10. AutoDream — LLM-powered memory consolidation
        if self.dream_engine:
            try:
                dream_report = await self.dream_engine.maybe_dream(self._turn_count)
                if dream_report:
                    self._emit("dream_done", {
                        "abstractions": dream_report.abstractions_created,
                        "contradictions": dream_report.contradictions_resolved,
                        "patterns": dream_report.patterns_detected,
                        "connections": dream_report.connections_added,
                        "questions": dream_report.questions_generated,
                    })
            except Exception as e:
                logger.warning(f"AutoDream failed: {e}")

        self._last_active = now

        return TurnResult(
            response=response,
            tool_calls=list(self._current_tool_calls),
            memories_used=memories_used,
            procedure_used=procedure_name,
            tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------ #
    #  Tool loop                                                           #
    # ------------------------------------------------------------------ #

    async def tool_loop(
        self, messages: list[dict], max_iterations: int = 10
    ) -> tuple[str, int]:
        """Execute LLM + tool calls until done or max_iterations."""
        from .llm.tool_parser import ToolCallParser

        parser = ToolCallParser()
        total_tokens = 0

        for i in range(max_iterations):
            self._emit("llm_call", {"iteration": i, "messages": len(messages)})

            response = self.llm.generate(messages)
            total_tokens += len(response) // 4  # rough estimate

            # Parse any tool calls
            tool_calls = parser.parse(response)

            if not tool_calls:
                self._emit("llm_done", {"iterations": i + 1, "tokens": total_tokens})
                return response, total_tokens

            # Execute tool calls
            for call in tool_calls:
                self._emit("tool_call", {"tool": call.name, "params": call.params})

                if self.tool_executor:
                    result = await self.tool_executor.execute(call.name, call.params)
                    self._current_tool_calls.append({
                        "name": call.name,
                        "params": call.params,
                        "success": result.success,
                    })
                    tool_output = result.output if result.success else f"[TOOL ERROR] {result.error}"
                else:
                    tool_output = "[Tool executor not configured]"
                    self._current_tool_calls.append({"name": call.name, "params": call.params})

                self._emit("tool_result", {"tool": call.name, "output_length": len(tool_output)})

                # Inject result back into conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": self.tool_executor.format_tool_result(call.name, tool_output)
                    if self.tool_executor
                    else f"<tool_result name=\"{call.name}\">\n{tool_output}\n</tool_result>",
                })

        logger.warning(f"Tool loop reached max_iterations={max_iterations}")
        return response, total_tokens

    # ------------------------------------------------------------------ #
    #  Background tasks                                                    #
    # ------------------------------------------------------------------ #

    async def _async_write(self, user_msg: str, agent_msg: str):
        """Extract and store knowledge from the just-completed exchange."""
        try:
            await self.writer.process_interaction(
                user_msg=user_msg,
                agent_msg=agent_msg,
                tool_calls=self._current_tool_calls,
                session_id=self.session_id,
            )
            self._emit("write_done", {"session": self.session_id})
        except Exception as e:
            logger.warning(f"Async knowledge extraction failed: {e}")

    # ------------------------------------------------------------------ #
    #  Bootstrap                                                           #
    # ------------------------------------------------------------------ #

    async def bootstrap(self):
        """
        Scan environment on first run to build initial knowledge.
        Only runs if DB is empty.
        """
        if not self.db or not self.tool_executor:
            return

        try:
            count = self.db.count_memories()
        except Exception:
            count = 0

        if count > 0:
            logger.info(f"Skipping bootstrap — {count} memories already exist")
            return

        logger.info("Bootstrap: scanning environment...")
        self._emit("bootstrap_start", {})

        for name, cmd in BOOTSTRAP_SCANS:
            try:
                result = await self.tool_executor.execute("bash", {
                    "command": cmd,
                    "timeout": 5,
                })
                if result.success and result.output.strip():
                    if self.db:
                        # Store raw scan output directly as a fact — no LLM call needed
                        # This keeps bootstrap fast; the LLM can reason over it later
                        content = f"[{name}]\n{result.output.strip()}"
                        self.db.insert_memory(
                            content=content,
                            category="observation",
                            source=f"scan:{name}",
                            importance=0.6,
                            confidence=0.9,
                        )
                    self._emit("bootstrap_scan", {"name": name, "lines": result.output.count("\n")})
                    print(f"  ✓ {name}")
            except Exception as e:
                logger.debug(f"Bootstrap scan failed for {name!r}: {e}")

        self._emit("bootstrap_done", {})
        logger.info("Bootstrap complete")

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _emit(self, event_type: str, data: dict):
        """Fire event to TUI or other listeners."""
        if self.on_event:
            with contextlib.suppress(Exception):
                self.on_event(event_type, data)

    def new_session(self):
        """Start a fresh conversation session."""
        self.session_id = str(uuid.uuid4())
        self._current_tool_calls = []
        return self.session_id

    @property
    def session(self) -> str:
        return self.session_id

    # ------------------------------------------------------------------ #
    #  Feedback                                                            #
    # ------------------------------------------------------------------ #

    def record_feedback(self, accepted: bool) -> None:
        """Record explicit user feedback for the last turn.

        Updates the retrieval feedback collector with whether the user
        found the agent's response helpful. This trains the reranker
        over time.

        Args:
            accepted: True if the response was helpful, False otherwise.
        """
        if self.feedback is None:
            logger.debug("record_feedback: no feedback collector available.")
            return

        try:
            self.feedback.record_outcome(self.session_id, accepted)
            self._emit("feedback", {"accepted": accepted, "session_id": self.session_id})
            logger.info(
                "record_feedback: %s feedback for session %s",
                "positive" if accepted else "negative",
                self.session_id,
            )
        except Exception as exc:
            logger.warning("record_feedback failed: %s", exc)
