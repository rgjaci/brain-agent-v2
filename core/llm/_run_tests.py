"""Quick sanity tests for ToolCallParser — no external services needed."""
import os
import sys

sys.path.insert(0, '/mnt/user-data/workspace')
os.chdir('/mnt/user-data/workspace')


from brain_agent.core.llm.tool_parser import ToolCallParser

parser = ToolCallParser()

def p_open(n):
    return f'<param name="{n}">'
p_close = '</param>'
def t_open(n):
    return f'<tool name="{n}">'
t_close = '</tool>'

# ---------- test 1: bash with Pydantic coercion ----------
resp1 = (
    'Running now.\n'
    + t_open('bash')
    + p_open('command') + 'echo hello' + p_close
    + p_open('timeout') + '10' + p_close
    + t_close
    + '\nDone.'
)
calls = parser.parse(resp1)
assert len(calls) == 1
assert calls[0].name == 'bash'
assert calls[0].params['command'] == 'echo hello'
assert calls[0].params['timeout'] == 10
print('PASS 1: bash call with Pydantic int coercion')

# ---------- test 2: multiple tool calls ----------
resp2 = (
    t_open('web_search')
    + p_open('query') + 'asyncio tutorial' + p_close
    + p_open('num_results') + '5' + p_close
    + t_close
    + t_open('teach')
    + p_open('content') + 'asyncio is async' + p_close
    + t_close
)
calls2 = parser.parse(resp2)
assert len(calls2) == 2
assert calls2[0].name == 'web_search'
assert calls2[1].name == 'teach'
assert calls2[0].params['num_results'] == 5
print('PASS 2: multiple tool calls')

# ---------- test 3: strip_tool_calls ----------
stripped = parser.strip_tool_calls(resp1)
assert 'Running now.' in stripped
assert 'Done.' in stripped
print('PASS 3: strip_tool_calls preserves surrounding text')

# ---------- test 4: unknown tool returns raw params ----------
resp3 = t_open('unknown') + p_open('x') + 'val' + p_close + t_close
calls3 = parser.parse(resp3)
assert calls3[0].params == {'x': 'val'}
print('PASS 4: unknown tool raw params returned')

# ---------- test 5: Pydantic defaults ----------
resp4 = t_open('recall') + p_open('query') + 'Python' + p_close + t_close
calls4 = parser.parse(resp4)
assert calls4[0].params['limit'] == 10
print('PASS 5: Pydantic default (recall.limit=10)')

# ---------- test 6: ingest default doc_type ----------
resp5 = t_open('ingest') + p_open('path') + 'readme.md' + p_close + t_close
calls5 = parser.parse(resp5)
assert calls5[0].params['doc_type'] == 'guide'
print('PASS 6: ingest.doc_type default = guide')

# ---------- test 7: ToolCall.raw contains original XML ----------
assert t_open('bash') in calls[0].raw
print('PASS 7: ToolCall.raw contains original XML')

# ---------- test 8: XML entity unescaping ----------
resp6 = (
    t_open('bash')
    + p_open('command') + 'echo &lt;hi&gt; &amp; done' + p_close
    + t_close
)
calls6 = parser.parse(resp6)
assert calls6[0].params['command'] == 'echo <hi> & done', repr(calls6[0].params['command'])
print('PASS 8: XML entity unescaping')

# ---------- test 9: write_file ----------
resp7 = (
    t_open('write_file')
    + p_open('path') + 'output.txt' + p_close
    + p_open('content') + 'hello world' + p_close
    + t_close
)
calls7 = parser.parse(resp7)
assert calls7[0].params['path'] == 'output.txt'
assert calls7[0].params['content'] == 'hello world'
print('PASS 9: write_file params')

# ---------- test 10: edit_file ----------
resp8 = (
    t_open('edit_file')
    + p_open('path') + 'main.py' + p_close
    + p_open('old_str') + 'x = 1' + p_close
    + p_open('new_str') + 'x = 42' + p_close
    + t_close
)
calls8 = parser.parse(resp8)
assert calls8[0].params['old_str'] == 'x = 1'
assert calls8[0].params['new_str'] == 'x = 42'
print('PASS 10: edit_file params')

print()
print('All 10 ToolCallParser tests passed.')
