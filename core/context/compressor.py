"""
Progressive history compressor for Brain Agent v2.

Strategy:
- Recent 5 messages: always kept verbatim (most important for continuity)
- Older messages: summarized in batches of 10
- Very old: single compressed summary
- Budget enforced in tokens (estimate: len//4)
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

KEEP_RECENT = 5
BATCH_SIZE = 10


class HistoryCompressor:
    """
    Compress conversation history to fit a token budget.

    Falls back gracefully if no LLM is provided (truncation only).
    """

    def __init__(self, llm=None):
        self.llm = llm

    def count_tokens(self, text_or_messages) -> int:
        """Estimate token count."""
        if isinstance(text_or_messages, str):
            return max(1, len(text_or_messages) // 4)
        if isinstance(text_or_messages, list):
            total = 0
            for msg in text_or_messages:
                if isinstance(msg, dict):
                    total += max(1, len(msg.get("content", "")) // 4)
                else:
                    total += max(1, len(str(msg)) // 4)
            return total
        return 1

    def compress(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        """
        Main entry: compress history to fit within budget_tokens.

        Args:
            messages: Full conversation history (oldest → newest).
            budget_tokens: Max tokens allowed.

        Returns:
            Compressed list of message dicts.
        """
        if not messages:
            return []

        # Fast path: already fits
        if self.count_tokens(messages) <= budget_tokens:
            return messages

        # Always keep most recent KEEP_RECENT messages verbatim
        recent = messages[-KEEP_RECENT:]
        older = messages[:-KEEP_RECENT]

        recent_tokens = self.count_tokens(recent)

        # If even recent messages exceed budget, truncate them
        if recent_tokens >= budget_tokens:
            return self.truncate_to_budget(recent, budget_tokens)

        remaining_budget = budget_tokens - recent_tokens

        if not older:
            return recent

        # Summarize older messages in batches
        batches = self.split_into_batches(older, BATCH_SIZE)
        per_batch_budget = max(50, remaining_budget // max(len(batches), 1))

        summarized: list[dict] = []
        for batch in batches:
            summary = self.summarize_batch(batch, max_tokens=per_batch_budget)
            summarized.append({
                "role": "system",
                "content": f"[Earlier context: {summary}]"
            })

        combined = summarized + recent

        # Check if combined fits
        if self.count_tokens(combined) <= budget_tokens:
            return combined

        # Still too large: truncate summaries
        return self.truncate_to_budget(combined, budget_tokens)

    def summarize_batch(self, messages: list[dict], max_tokens: int = 200) -> str:
        """
        Summarize a batch of messages.

        Uses LLM if available, otherwise extracts key phrases.
        """
        formatted = self.format_messages_for_summary(messages)

        if self.llm is not None:
            prompt = (
                f"Summarize this conversation in {max_tokens} tokens or fewer. "
                f"Keep: decisions made, facts stated, preferences expressed, "
                f"unresolved questions. Be concise.\n\n{formatted}"
            )
            try:
                result = self.llm.generate(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                return result.strip()
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}, using fallback")

        # Fallback: extract first words from each message
        parts = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            snippet = content[:80].replace("\n", " ").strip()
            if len(content) > 80:
                snippet += "..."
            parts.append(f"{role}: {snippet}")
        return " | ".join(parts)

    def format_messages_for_summary(self, messages: list[dict]) -> str:
        """Format messages for summarization prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:500]  # Truncate each to 500 chars
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def split_into_batches(self, messages: list, batch_size: int = 10) -> list[list]:
        """Split list into fixed-size batches."""
        return [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]

    def truncate_to_budget(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        """
        Keep the newest messages that fit within budget.
        Always keeps at least 1 message.
        """
        if not messages:
            return []

        result = []
        used = 0

        for msg in reversed(messages):
            content = msg.get("content", "")
            tokens = max(1, len(content) // 4)
            if used + tokens > budget_tokens and result:
                break
            result.append(msg)
            used += tokens

        result.reverse()
        return result
