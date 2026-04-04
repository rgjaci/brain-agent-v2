"""Tests for ToolCallParser — XML parsing, multi-call, validation."""
from __future__ import annotations

from core.llm.tool_parser import ToolCall, ToolCallParser


def make_parser():
    return ToolCallParser()


# ── basic parsing ─────────────────────────────────────────────────────────────

def test_parse_single_bash_call():
    p = make_parser()
    xml = '<tool name="bash"><param name="command">ls -la</param></tool>'
    calls = p.parse(xml)
    assert len(calls) == 1
    assert calls[0].name == "bash"
    assert calls[0].params["command"] == "ls -la"


def test_parse_read_file_call():
    p = make_parser()
    xml = '<tool name="read_file"><param name="path">/etc/hosts</param></tool>'
    calls = p.parse(xml)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].params["path"] == "/etc/hosts"


def test_parse_multiple_calls():
    p = make_parser()
    xml = (
        '<tool name="bash"><param name="command">echo hello</param></tool>'
        ' some text '
        '<tool name="read_file"><param name="path">/tmp/x</param></tool>'
    )
    calls = p.parse(xml)
    assert len(calls) == 2
    assert calls[0].name == "bash"
    assert calls[1].name == "read_file"


def test_parse_empty_returns_empty():
    p = make_parser()
    assert p.parse("") == []
    assert p.parse("no tool calls here") == []


def test_parse_with_surrounding_text():
    p = make_parser()
    text = (
        "I'll run the command now.\n"
        '<tool name="bash"><param name="command">whoami</param></tool>\n'
        "Let me know if you need more."
    )
    calls = p.parse(text)
    assert len(calls) == 1
    assert calls[0].params["command"] == "whoami"


# ── multiline commands ────────────────────────────────────────────────────────

def test_parse_multiline_bash():
    p = make_parser()
    xml = (
        '<tool name="bash">'
        '<param name="command">apt-get update\napt-get install -y nginx</param>'
        '</tool>'
    )
    calls = p.parse(xml)
    assert len(calls) == 1
    assert "apt-get" in calls[0].params["command"]


# ── optional parameters ───────────────────────────────────────────────────────

def test_parse_bash_with_timeout():
    p = make_parser()
    xml = (
        '<tool name="bash">'
        '<param name="command">sleep 5</param>'
        '<param name="timeout">10</param>'
        '</tool>'
    )
    calls = p.parse(xml)
    assert len(calls) == 1
    assert int(calls[0].params.get("timeout", 30)) == 10


def test_parse_bash_default_timeout():
    p = make_parser()
    xml = '<tool name="bash"><param name="command">ls</param></tool>'
    calls = p.parse(xml)
    # timeout defaults to 30 if not specified
    assert calls[0].params.get("timeout", 30) == 30 or "timeout" not in calls[0].params


# ── web_search tool ───────────────────────────────────────────────────────────

def test_parse_web_search():
    p = make_parser()
    xml = '<tool name="web_search"><param name="query">python asyncio tutorial</param></tool>'
    calls = p.parse(xml)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].params["query"] == "python asyncio tutorial"


# ── teach tool ────────────────────────────────────────────────────────────────

def test_parse_teach():
    p = make_parser()
    xml = (
        '<tool name="teach">'
        '<param name="fact">User prefers tmux over screen</param>'
        '<param name="category">preference</param>'
        '</tool>'
    )
    calls = p.parse(xml)
    assert len(calls) == 1
    assert calls[0].name == "teach"
    assert "tmux" in calls[0].params["fact"]


# ── validation ────────────────────────────────────────────────────────────────

def test_unknown_tool_is_skipped_or_included():
    p = make_parser()
    xml = '<tool name="nonexistent_tool"><param name="x">y</param></tool>'
    calls = p.parse(xml)
    # Either raises or returns empty — should not crash
    assert isinstance(calls, list)


def test_missing_required_param_handled():
    p = make_parser()
    # bash requires "command"
    xml = '<tool name="bash"><param name="timeout">5</param></tool>'
    # Should not raise — either skips or includes with missing param
    calls = p.parse(xml)
    assert isinstance(calls, list)


# ── ToolCall dataclass ────────────────────────────────────────────────────────

def test_tool_call_has_name_and_params():
    tc = ToolCall(name="bash", params={"command": "ls"},
                  raw='<tool name="bash"><param name="command">ls</param></tool>')
    assert tc.name == "bash"
    assert tc.params["command"] == "ls"
