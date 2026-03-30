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
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
    procedure_used: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None


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
        session_id: Optional[str] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
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
        self.session_id = session_id or str(uuid.uuid4())
        self.on_event = on_event  # TUI event hook: (event_type, data) -> None

        # State for current turn
        self._current_tool_calls: list[dict] = []
        self._turn_count: int = 0
        self._last_active: float = 0.0

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    async def process(self, user_input: str) -> TurnResult:
        """Process one user message. Returns TurnResult."""
        self._current_tool_calls = []
        self._emit("turn_start", {"input": user_input})

        # 1. Store user message
        if self.db:
            self.db.store_conversation(self.session_id, "user", user_input)

        # 2. Retrieve relevant memory
        context = None
        memories_used = 0
        procedure_name = None

        if self.reader:
            self._emit("retrieval_start", {"query": user_input})
            try:
                context = await self.reader.retrieve(user_input, self.session_id)
                memories_used = len(context.memories) if context else 0
                if context and context.procedures:
                    proc = context.procedures[0] if context.procedures else None
                    if proc:
                        procedure_name = (
                            proc.get("name") if isinstance(proc, dict)
                            else getattr(proc, "name", None)
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

        # 7. Async: extract and store new knowledge
        if self.writer:
            asyncio.create_task(self._async_write(user_input, response))

        # 8. Log retrieval outcome (assume accepted)
        if self.feedback:
            try:
                self.feedback.record_outcome(self.session_id, user_accepted=True)
            except AttributeError:
                pass  # record_outcome may not exist
            except Exception as e:
                logger.warning(f"Feedback logging failed: {e}")

        # 9. Consolidation — run if idle > 300s
        import time
        self._turn_count += 1
        now = time.time()
        if self.consolidator and (now - self._last_active > 300 or self._last_active == 0):
            try:
                await self.consolidator.maybe_consolidate(self._turn_count)
            except Exception as e:
                logger.warning(f"Consolidation failed: {e}")
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
        from .llm.tool_parser import ToolParser

        parser = ToolParser()
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
                    if self.writer:
                        # Store directly as a fact memory
                        await self.writer.extract_from_scan(name, result.output)
                    self._emit("bootstrap_scan", {"name": name, "lines": result.output.count("\n")})
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
            try:
                self.on_event(event_type, data)
            except Exception:
                pass

    def new_session(self):
        """Start a fresh conversation session."""
        self.session_id = str(uuid.uuid4())
        self._current_tool_calls = []
        return self.session_id

    @property
    def session(self) -> str:
        return self.session_id
