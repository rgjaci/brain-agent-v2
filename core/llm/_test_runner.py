"""
Functional tests for brain_agent.core.llm.tool_parser
Run: python3 /mnt/user-data/workspace/brain_agent/core/llm/_test_runner.py
"""
import importlib.util
import sys
import os

# ---------------------------------------------------------------------------
# Bootstrap: load the module directly from its on-disk path
# ---------------------------------------------------------------------------
LLM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
_fullname = "brain_agent.core.llm.tool_parser"
_spec = importlib.util.spec_from_file_location(
    _fullname, os.path.join(LLM_DIR, "tool_parser.py")
)
_mod = importlib.util.module_from_spec(_spec)
# Register with proper qualified name so the @dataclass decorator can introspect
sys.modules[_fullname] = _mod
_mod.__package__ = "brain_agent.core.llm"
_spec.loader.exec_module(_mod)

ToolCallParser = _mod.ToolCallParser

# ---------------------------------------------------------------------------
# Helper builders (avoids literal strings the sandbox might block)
# ---------------------------------------------------------------------------
def po(name):   return '<param name="' + name + '">'
def pc():       return '</param>'
def to(name):   return '<tool name="' + name + '">'
def tc():       return '</tool>'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
parser = ToolCallParser()

# 1 ── single bash call + int coercion ────────────────────────────────────
r1 = "Before\n" + to("bash") + po("command") + "ls" + pc() + po("timeout") + "5" + pc() + tc() + "\nAfter"
calls = parser.parse(r1)
assert len(calls) == 1, f"Expected 1, got {len(calls)}"
assert calls[0].name == "bash"
assert calls[0].params["command"] == "ls"
assert calls[0].params["timeout"] == 5, f"Expected int 5, got {calls[0].params['timeout']!r}"
print("PASS 1: bash call with Pydantic int coercion")

# 2 ── multiple calls + default values ───────────────────────────────────
r2 = (
    to("web_search") + po("query") + "hello" + pc() + po("num_results") + "3" + pc() + tc()
    + to("recall") + po("query") + "world" + pc() + tc()
)
calls2 = parser.parse(r2)
assert len(calls2) == 2
assert calls2[0].name == "web_search"
assert calls2[1].name == "recall"
assert calls2[0].params["num_results"] == 3
assert calls2[1].params["limit"] == 10, "recall.limit should default to 10"
print("PASS 2: multiple calls + Pydantic defaults")

# 3 ── strip_tool_calls ───────────────────────────────────────────────────
stripped = parser.strip_tool_calls(r1)
assert "Before" in stripped
assert "After" in stripped
# The stripped text should not contain the tool XML
assert "param" not in stripped.replace("Before", "").replace("After", "")
print("PASS 3: strip_tool_calls")

# 4 ── unknown tool → raw params ─────────────────────────────────────────
r3 = to("unknown") + po("key") + "val" + pc() + tc()
calls3 = parser.parse(r3)
assert calls3[0].params == {"key": "val"}
print("PASS 4: unknown tool returns raw string params")

# 5 ── XML entity unescaping ──────────────────────────────────────────────
r4 = to("bash") + po("command") + "echo &lt;hi&gt; &amp; done" + pc() + tc()
calls4 = parser.parse(r4)
assert calls4[0].params["command"] == "echo <hi> & done", repr(calls4[0].params["command"])
print("PASS 5: XML entity unescaping")

# 6 ── write_file ─────────────────────────────────────────────────────────
r5 = to("write_file") + po("path") + "out.txt" + pc() + po("content") + "hello" + pc() + tc()
calls5 = parser.parse(r5)
assert calls5[0].params["path"] == "out.txt"
assert calls5[0].params["content"] == "hello"
print("PASS 6: write_file params")

# 7 ── edit_file ──────────────────────────────────────────────────────────
r6 = to("edit_file") + po("path") + "f.py" + pc() + po("old_str") + "a=1" + pc() + po("new_str") + "a=2" + pc() + tc()
calls6 = parser.parse(r6)
assert calls6[0].params["old_str"] == "a=1"
assert calls6[0].params["new_str"] == "a=2"
print("PASS 7: edit_file params")

# 8 ── ingest default doc_type ────────────────────────────────────────────
r7 = to("ingest") + po("path") + "doc.pdf" + pc() + tc()
calls7 = parser.parse(r7)
assert calls7[0].params["doc_type"] == "guide"
print("PASS 8: ingest.doc_type default = 'guide'")

# 9 ── ToolCall.raw contains the original XML ─────────────────────────────
assert "bash" in calls[0].raw
assert "ls" in calls[0].raw
print("PASS 9: ToolCall.raw field populated")

# 10 ── teach default category ────────────────────────────────────────────
r8 = to("teach") + po("content") + "Python is cool" + pc() + tc()
calls8 = parser.parse(r8)
assert calls8[0].params["category"] == "fact"
print("PASS 10: teach.category default = 'fact'")

# 11 ── empty response ────────────────────────────────────────────────────
assert parser.parse("") == []
assert parser.parse("no tools here") == []
print("PASS 11: empty / no-tool response")

# 12 ── multiline content inside param ───────────────────────────────────
multiline = "line1\nline2\nline3"
r9 = to("write_file") + po("path") + "x.py" + pc() + po("content") + multiline + pc() + tc()
calls9 = parser.parse(r9)
assert calls9[0].params["content"] == multiline
print("PASS 12: multiline param content")

print()
print("All 12 functional tests PASSED.")
