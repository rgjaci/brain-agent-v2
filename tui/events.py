"""Event bus for Brain Agent TUI real-time debug updates."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class AgentEvent:
    type: str
    data: dict
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Lightweight event bus. Stores recent events for TUI replay."""

    MAX_HISTORY = 200

    def __init__(self):
        self._subscribers: list[Callable[[AgentEvent], None]] = []
        self._history: deque[AgentEvent] = deque(maxlen=self.MAX_HISTORY)

    def emit(self, event_type: str, data: dict):
        event = AgentEvent(type=event_type, data=data)
        self._history.append(event)
        for sub in list(self._subscribers):
            try:
                sub(event)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[AgentEvent], None]):
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    def get_handler(self) -> Callable[[str, dict], None]:
        """Returns a function compatible with BrainAgent.on_event signature."""
        def handler(event_type: str, data: dict):
            self.emit(event_type, data)
        return handler

    def recent(self, n: int = 50) -> list[AgentEvent]:
        events = list(self._history)
        return events[-n:]

    def clear(self):
        self._history.clear()


# Human-readable labels for each event type
EVENT_LABELS = {
    "turn_start":        "▶ New turn",
    "retrieval_start":   "🔍 Retrieval",
    "retrieval_done":    "✓ Retrieved",
    "context_assembled": "📦 Context",
    "llm_call":          "🤖 LLM call",
    "llm_done":          "✓ LLM done",
    "tool_call":         "🔧 Tool",
    "tool_result":       "✓ Tool result",
    "response_ready":    "💬 Response",
    "bootstrap_start":   "🚀 Bootstrap",
    "bootstrap_scan":    "🔍 Scan",
    "bootstrap_done":    "✓ Bootstrap done",
    "write_done":        "💾 Write done",
}


def format_event(event: AgentEvent) -> str:
    """Format an event into a one-line string for the activity panel."""
    label = EVENT_LABELS.get(event.type, f"• {event.type}")
    d = event.data
    extra = ""

    if event.type == "retrieval_done":
        mems = d.get("memories", 0)
        kg = d.get("kg_facts", 0)
        extra = f"{mems} mem, {kg} KG facts"
    elif event.type == "llm_call":
        extra = f"iter {d.get('iteration', 0) + 1}"
    elif event.type == "llm_done":
        extra = f"{d.get('tokens', 0)} tok, {d.get('iterations', 1)} iter"
    elif event.type == "tool_call":
        extra = d.get("tool", "?")
    elif event.type == "tool_result":
        extra = f"{d.get('output_length', 0)} chars"
    elif event.type == "context_assembled":
        total = d.get("total_tokens", 0)
        budget = d.get("budget", 32000)
        extra = f"{total}/{budget} tok"
    elif event.type == "bootstrap_scan":
        extra = d.get("name", "")
    elif event.type == "turn_start":
        q = d.get("input", "")
        extra = f"{q[:40]}..." if len(q) > 40 else q
    elif event.type == "response_ready":
        r = d.get("response", "")
        extra = f"{r[:40]}..." if len(r) > 40 else r

    return f"{label}: {extra}" if extra else label
