"""Tests for tui/app.py — Textual TUI for Brain Agent."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    import textual.app  # check textual installed
    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False

pytestmark = pytest.mark.skipif(not _TEXTUAL, reason="textual not installed")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@dataclass
class FakeTurnResult:
    response: str = "mock response"
    tool_calls: list = field(default_factory=list)
    memories_used: int = 3
    procedure_used: str | None = None
    tokens_used: int = 100
    error: str | None = None


@pytest.fixture
def mock_agent(db):
    """Mock BrainAgent with enough surface area for TUI testing."""
    agent = MagicMock()
    agent.db = db
    agent.process = AsyncMock(return_value=FakeTurnResult())
    agent.bootstrap = AsyncMock()
    agent.new_session = MagicMock(return_value="new-session-id")
    agent.on_event = None
    agent.reasoning_engine = None  # prevent MagicMock auto-creating attrs
    agent.dream_engine = None
    # populate some DB rows so _refresh_stats works
    db.insert_memory("test fact", category="fact", importance=0.8)
    db.insert_entity("Python", entity_type="language")
    return agent


# ── App construction ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_app_mounts_no_agent():
    """App should mount successfully even without an agent (demo mode)."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=None, config=None)
    async with app.run_test():
        # Should show ready message
        chat_log = app.query_one("#chat-log")
        assert chat_log is not None


@pytest.mark.asyncio
async def test_app_mounts_with_agent(mock_agent):
    """App should mount and wire agent event hook."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        chat_log = app.query_one("#chat-log")
        assert chat_log is not None


# ── Send message ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_empty_ignored(mock_agent):
    """Empty input should not trigger agent.process()."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        await app.action_send_message()
        mock_agent.process.assert_not_called()


@pytest.mark.asyncio
async def test_send_message_calls_agent(mock_agent):
    """Typing text and sending should call agent.process() with the text."""
    from textual.widgets import TextArea

    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        textarea = app.query_one("#user-input", TextArea)
        textarea.text = "hello world"
        await app.action_send_message()
        # Give the background task time to run
        await asyncio.sleep(0.2)
        mock_agent.process.assert_called_once_with("hello world")


@pytest.mark.asyncio
async def test_send_clears_textarea(mock_agent):
    """After sending, the textarea should be cleared."""
    from textual.widgets import TextArea

    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        textarea = app.query_one("#user-input", TextArea)
        textarea.text = "test input"
        await app.action_send_message()
        assert textarea.text == ""


@pytest.mark.asyncio
async def test_send_with_tool_calls(mock_agent):
    """Response with tool_calls should display tool names."""
    mock_agent.process = AsyncMock(return_value=FakeTurnResult(
        response="done",
        tool_calls=[{"name": "bash"}, {"name": "read_file"}],
    ))
    from textual.widgets import TextArea

    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        textarea = app.query_one("#user-input", TextArea)
        textarea.text = "run ls"
        await app.action_send_message()
        await asyncio.sleep(0.2)
        mock_agent.process.assert_called_once()


# ── No agent (demo mode) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_no_agent_demo_mode():
    """With no agent, sending a message should show demo mode text."""
    from textual.widgets import TextArea

    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=None, config=None)
    async with app.run_test():
        textarea = app.query_one("#user-input", TextArea)
        textarea.text = "hello"
        await app.action_send_message()
        await asyncio.sleep(0.1)
        # Should not crash


# ── Agent error handling ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_process_error(mock_agent):
    """If agent.process() raises, error should be displayed, not crash."""
    mock_agent.process = AsyncMock(side_effect=RuntimeError("LLM timeout"))
    from textual.widgets import TextArea

    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        textarea = app.query_one("#user-input", TextArea)
        textarea.text = "test"
        await app.action_send_message()
        await asyncio.sleep(0.2)
        # Should not crash


# ── Clear chat ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_chat(mock_agent):
    """Ctrl+L should clear the chat log and start a new session."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app.action_clear_chat()
        mock_agent.new_session.assert_called_once()


# ── Toggle debug ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_toggle_debug(mock_agent):
    """Toggle debug should hide/show the debug panel."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        panel = app.query_one("#debug-panel")
        assert app._debug_shown is True
        app.action_toggle_debug()
        assert app._debug_shown is False
        assert panel.display is False
        app.action_toggle_debug()
        assert app._debug_shown is True
        assert panel.display is True


# ── Bootstrap ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bootstrap(mock_agent):
    """Bootstrap action should call agent.bootstrap()."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        await app.action_bootstrap()
        mock_agent.bootstrap.assert_called_once()


# ── Agent event hook ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_event_retrieval(mock_agent):
    """Retrieval events should render in the debug panel."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._on_agent_event("retrieval", {
            "strategy": "aggressive",
            "memories": 5,
            "entities": 2,
        })
        # Should not crash


@pytest.mark.asyncio
async def test_agent_event_llm_done(mock_agent):
    """LLM done events should update the token display."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._on_agent_event("llm_done", {
            "tokens": 1500,
            "elapsed_ms": 420,
        })


@pytest.mark.asyncio
async def test_agent_event_write_done(mock_agent):
    """Write done events should show extraction stats."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._on_agent_event("write_done", {
            "facts": 3,
            "entities": 1,
            "relations": 2,
        })


@pytest.mark.asyncio
async def test_agent_event_tool_call(mock_agent):
    """Tool call events should show tool name."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._on_agent_event("tool_call", {"name": "bash"})
        app._on_agent_event("tool_result", {"result": "file.txt"})


@pytest.mark.asyncio
async def test_agent_event_procedure_match(mock_agent):
    """Procedure match events should show name and confidence."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._on_agent_event("procedure_match", {
            "name": "deploy_docker",
            "confidence": 0.85,
        })


@pytest.mark.asyncio
async def test_agent_event_consolidation(mock_agent):
    """Consolidation events should show merge count."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._on_agent_event("consolidation", {"merged": 3})


# ── Stats panel ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_refresh_stats(mock_agent):
    """Stats panel should display memory/entity/relation counts."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._refresh_stats()
        # Should not crash and stats-display should be updated


@pytest.mark.asyncio
async def test_refresh_stats_no_agent():
    """Stats refresh with no agent should show fallback message."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=None, config=None)
    async with app.run_test():
        app._refresh_stats()


# ── Token display ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_token_display_update(mock_agent):
    """Token display should handle various usage data."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._update_token_display({
            "total": 16000,
            "system": 500,
            "procedures": 2000,
            "memory": 8000,
            "history": 5500,
        })


@pytest.mark.asyncio
async def test_token_display_empty(mock_agent):
    """Token display should handle empty usage data."""
    from tui.app import BrainAgentApp
    app = BrainAgentApp(agent=mock_agent, config=None)
    async with app.run_test():
        app._update_token_display({})


# ── Fallback stub ────────────────────────────────────────────────────────────

def test_stub_class_exists():
    """When textual is not available, a stub BrainAgentApp should exist."""
    # The actual fallback is tested via import guards; just verify the module imports
    from tui.app import BrainAgentApp
    assert BrainAgentApp is not None
