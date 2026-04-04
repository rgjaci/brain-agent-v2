"""
Context assembler — packs the 32K token window optimally.

TOKEN_BUDGET:
    system_prompt:   500   # Fixed system identity + instructions
    procedure:      2000   # Matched procedure (if any)
    kg_context:     1500   # KG entity/relation facts
    memories:      13000   # Retrieved memories
    chat_history:   6000   # Recent conversation turns
    tool_buffer:    2000   # Tool call/result space
    output_reserve: 4000   # Model response space
    query:           500   # Current user message
    overhead:       1268   # Formatting + tokenizer variance
    # Total ≈ 30768        # Leaves 2K margin under 32768
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

TOKEN_BUDGET = {
    "system_prompt": 500,
    "procedure": 2000,
    "kg_context": 1500,
    "memories": 13000,
    "chat_history": 6000,
    "tool_buffer": 2000,
    "output_reserve": 4000,
    "query": 500,
    "overhead": 1268,
}

MAX_CONTEXT_TOKENS = 32768

SYSTEM_PROMPT = """You are an AI assistant with access to a persistent memory system.

PROCEDURE (if provided): Follow these steps exactly. They are a proven approach to this type of problem. Adapt specifics to the current situation. Skip steps that don't apply and explain why.

MEMORIES: These are facts and context from previous interactions. Treat them as ground truth unless they contradict each other (prefer newer ones).

KG CONTEXT: These are entity relationships. Use them to understand how things connect.

CHAT HISTORY: Recent conversation for continuity.

When answering:
1. If a PROCEDURE matches, follow it step-by-step
2. Cite specific memories when making claims ("Based on what you told me earlier...")
3. If you're unsure, say so — don't hallucinate
4. If you need more information, use tools to get it
5. Be concise unless asked for detail

Available tools (use XML format):
<tool name="bash"><param name="command">command here</param></tool>
<tool name="read_file"><param name="path">/path/to/file</param></tool>
<tool name="write_file"><param name="path">/path</param><param name="content">text</param></tool>
<tool name="edit_file"><param name="path">/path</param><param name="old_str">old</param><param name="new_str">new</param></tool>
<tool name="web_search"><param name="query">search terms</param></tool>
<tool name="teach"><param name="content">fact or procedure to remember</param></tool>
<tool name="recall"><param name="query">what to search for</param></tool>
<tool name="ingest"><param name="path">/path/to/document</param></tool>"""


class ContextAssembler:
    """Pack the 32K context window optimally."""

    _encoder = None  # shared class-level cache

    def __init__(self, llm=None):
        self.llm = llm
        self.budget = TOKEN_BUDGET.copy()
        # Lazy-init tiktoken encoder once
        if ContextAssembler._encoder is None:
            try:
                import tiktoken
                ContextAssembler._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                ContextAssembler._encoder = False  # mark unavailable

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken when available, else heuristic."""
        if ContextAssembler._encoder:
            return max(1, len(ContextAssembler._encoder.encode(text)))
        return max(1, len(text) // 4)

    def assemble(
        self,
        procedure=None,
        memories: list | None = None,
        kg_context: str = "",
        chat_history: list | None = None,
        query: str = "",
        budget: dict | None = None,
    ) -> list[dict]:
        """
        Assemble messages list for LLM call.

        Returns OpenAI-style messages:
            [{"role": "system", "content": "..."}, ..., {"role": "user", "content": "..."}]
        """
        if budget is None:
            budget = self.budget

        memories = memories or []
        chat_history = chat_history or []

        # --- Build system message content ---
        system_content = SYSTEM_PROMPT

        # Procedure goes right after system prompt (highest priority signal)
        if procedure:
            proc_text = self.format_procedure(procedure)
            system_content += f"\n\n{proc_text}"

        # KG context
        if kg_context and kg_context.strip():
            system_content += f"\n\n## KNOWLEDGE GRAPH CONTEXT\n{kg_context.strip()}"

        # Memories — compute remaining budget
        fixed_tokens = (
            self.count_tokens(system_content)
            + budget.get("query", 500)
            + budget.get("tool_buffer", 2000)
            + budget.get("output_reserve", 4000)
            + budget.get("overhead", 1268)
        )
        history_tokens = budget.get("chat_history", 6000)
        memory_budget = max(1000, MAX_CONTEXT_TOKENS - fixed_tokens - history_tokens)
        memory_budget = min(memory_budget, budget.get("memories", 13000))

        if memories:
            ordered = self.apply_best_at_edges(memories)
            mem_text = self.pack_memories(ordered, memory_budget)
            if mem_text:
                system_content += f"\n\n## RELEVANT MEMORIES\n{mem_text}"

        messages: list[dict] = [{"role": "system", "content": system_content}]

        # Chat history — inject as alternating user/assistant messages
        if chat_history:
            history_budget = min(
                budget.get("chat_history", 6000),
                MAX_CONTEXT_TOKENS
                - self.count_tokens(system_content)
                - budget.get("query", 500)
                - budget.get("tool_buffer", 2000)
                - budget.get("output_reserve", 4000)
                - budget.get("overhead", 1268),
            )
            history_budget = max(500, history_budget)
            compressed = self.format_chat_history(chat_history, history_budget)
            messages.extend(compressed)

        # Final user query
        messages.append({"role": "user", "content": query})

        return messages

    def pack_memories(self, memories: list, budget_tokens: int = 13000) -> str:
        """Format and pack memories within token budget."""
        lines = []
        used = 0

        for mem in memories:
            # Support both dict and object
            if isinstance(mem, dict):
                content = mem.get("content", "")
                category = mem.get("category", "fact")
            else:
                content = getattr(mem, "content", str(mem))
                category = getattr(mem, "category", "fact")

            line = f"[{category}] {content}"
            tokens = self.count_tokens(line)

            if used + tokens > budget_tokens:
                break

            lines.append(line)
            used += tokens

        return "\n".join(lines)

    def apply_best_at_edges(self, memories: list) -> list:
        """
        Reorder so best memories are at top and bottom (Lost-in-the-Middle mitigation).

        Input already sorted best→worst.
        Output: best at position 0, second-best at position -1,
                third-best at position 1, etc.
        """
        if len(memories) <= 2:
            return memories

        top = []
        bottom = []

        for i, mem in enumerate(memories):
            if i % 2 == 0:
                top.append(mem)
            else:
                bottom.append(mem)

        # bottom reversed so best of bottom group is at the actual end
        return top + list(reversed(bottom))

    def format_procedure(self, procedure) -> str:
        """Format a procedure (dict or object) for context."""
        if isinstance(procedure, dict):
            name = procedure.get("name", "Procedure")
            description = procedure.get("description", "")
            steps = procedure.get("steps", [])
            preconditions = procedure.get("preconditions", [])
            warnings = procedure.get("warnings", [])
            context = procedure.get("context", "")
        else:
            name = getattr(procedure, "name", "Procedure")
            description = getattr(procedure, "description", "")
            steps = getattr(procedure, "steps", [])
            preconditions = getattr(procedure, "preconditions", [])
            warnings = getattr(procedure, "warnings", [])
            context = getattr(procedure, "context", "")

        # Deserialize if stored as JSON strings
        if isinstance(steps, str):
            import json
            try:
                steps = json.loads(steps)
            except Exception:
                steps = [steps]
        if isinstance(preconditions, str):
            import json
            try:
                preconditions = json.loads(preconditions)
            except Exception:
                preconditions = [preconditions]
        if isinstance(warnings, str):
            import json
            try:
                warnings = json.loads(warnings)
            except Exception:
                warnings = [warnings]

        lines = [f"## PROCEDURE: {name}"]
        if description:
            lines.append(description)

        if preconditions:
            if isinstance(preconditions, list):
                lines.append(f"\nPrerequisites: {', '.join(preconditions)}")
            else:
                lines.append(f"\nPrerequisites: {preconditions}")

        if steps:
            lines.append("\nSteps:")
            for i, step in enumerate(steps, 1):
                lines.append(f"  {i}. {step}")

        if warnings:
            if isinstance(warnings, list):
                lines.append(f"\nWarnings: {' | '.join(warnings)}")
            else:
                lines.append(f"\nWarnings: {warnings}")

        if context:
            lines.append(f"\nContext: {context}")

        return "\n".join(lines)

    def format_chat_history(
        self, messages: list[dict], budget_tokens: int = 6000
    ) -> list[dict]:
        """
        Return recent history messages within token budget.
        Keeps newest messages; drops oldest if over budget.
        """
        if not messages:
            return []

        # Work backwards from newest
        result = []
        used = 0

        for msg in reversed(messages):
            content = msg.get("content", "")
            tokens = self.count_tokens(content)
            if used + tokens > budget_tokens:
                break
            result.append(msg)
            used += tokens

        result.reverse()
        return result

    def get_token_usage(self, messages: list[dict]) -> dict:
        """Return token usage breakdown."""
        total = 0
        system_tokens = 0
        history_tokens = 0
        query_tokens = 0

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            tokens = self.count_tokens(content)
            total += tokens

            role = msg.get("role", "")
            if role == "system":
                system_tokens += tokens
            elif i == len(messages) - 1 and role == "user":
                query_tokens += tokens
            else:
                history_tokens += tokens

        return {
            "total": total,
            "system": system_tokens,
            "history": history_tokens,
            "query": query_tokens,
            "remaining": MAX_CONTEXT_TOKENS - total,
        }
