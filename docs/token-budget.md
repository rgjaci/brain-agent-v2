# Token Budget Explanation

The agent uses a 32K token context window with strict budget allocation for each component.

## Budget Allocation

| Component | Tokens | Percentage | Purpose |
|---|---|---|---|
| **System Prompt** | 500 | 1.5% | Core instructions and behavior |
| **Procedure Context** | 2,000 | 6.1% | Retrieved procedures with steps |
| **Knowledge Graph** | 1,500 | 4.6% | Entity relationships from KG traversal |
| **Memories** | 13,000 | 39.7% | Retrieved memories (main content) |
| **Conversation History** | 6,000 | 18.3% | Recent conversation messages |
| **Tool Output** | 2,000 | 6.1% | Results from tool executions |
| **Output Buffer** | 4,000 | 12.2% | Reserved for LLM response |
| **Query** | 500 | 1.5% | Current user query |
| **Overhead** | 1,268 | 3.9% | Formatting, separators, XML tags |
| **Total** | **32,768** | **100%** | — |

## Rationale

### System Prompt (500 tokens)
The system prompt contains core instructions for the LLM:
- How to use tools
- Response formatting guidelines
- Behavioral constraints
- Memory usage instructions

500 tokens is sufficient for a concise but complete system prompt. Larger prompts waste context that could be used for memories.

### Procedure Context (2,000 tokens)
Procedures contain step-by-step instructions that the agent may need to follow. 2,000 tokens allows for:
- 2-3 procedures with ~10 steps each
- Preconditions and warnings
- Context notes

Procedures are typically shorter than memories because they're structured lists rather than prose.

### Knowledge Graph (1,500 tokens)
The KG context contains entity relationships in a structured format:
```
- Python (language) → uses → Docker (tool)
- Docker (tool) → manages → Containers (concept)
```

1,500 tokens allows for ~20-30 relationship triples, which is sufficient for most queries.

### Memories (13,000 tokens) — The Largest Budget
This is the core of the memory-first approach. 13,000 tokens allows for:
- ~50-100 memories (at ~130-260 tokens each)
- The most important component of the context

This is intentionally the largest budget because the agent's thesis is that memory quality matters more than model size.

### Conversation History (6,000 tokens)
Recent conversation context for continuity:
- Last 5 messages kept verbatim
- Older messages compressed via summarization
- 6,000 tokens allows for ~15-20 messages of history

### Tool Output (2,000 tokens)
Results from tool executions (bash output, file contents, search results):
- Command output can be verbose
- 2,000 tokens is a guard against excessive output
- Tool outputs are truncated if they exceed this budget

### Output Buffer (4,000 tokens)
Reserved for the LLM's response:
- Ensures the model has room to generate a complete answer
- 4,000 tokens allows for ~3,000 words of output
- Prevents truncation of long responses

### Query (500 tokens)
The current user query:
- Most queries are short (<100 tokens)
- 500 tokens is a safe upper bound

### Overhead (1,268 tokens)
Formatting, XML tags, section headers, and separators:
```
<memories>
  <memory id="1" category="fact" score="0.95">
    Content here...
  </memory>
</memories>
```

This is a fixed cost that doesn't scale with content.

## Tuning the Budget

### When to Increase Memory Budget

If the agent frequently says "I don't have enough context" or misses relevant information:

```yaml
token_budget:
  memories: 15000  # Increase from 13000
  history: 5000    # Decrease to compensate
```

### When to Decrease Memory Budget

If the agent's responses are unfocused or include irrelevant information:

```yaml
token_budget:
  memories: 10000  # Decrease from 13000
  retrieval:
    top_k: 5       # Also reduce retrieval count
```

### When to Increase Tool Output Budget

If tool outputs are frequently truncated:

```yaml
token_budget:
  tool: 4000       # Increase from 2000
  memories: 11000  # Decrease to compensate
```

### When to Increase Output Buffer

If the agent's responses are frequently cut off:

```yaml
token_budget:
  output: 6000     # Increase from 4000
  memories: 11000  # Decrease to compensate
```

## Token Counting

The agent uses `tiktoken` for token counting (OpenAI's tokenizer):

```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")
token_count = len(encoder.encode(text))
```

This is an approximation — actual token counts may vary slightly depending on the model.

## Best-at-Edges and Token Budget

The "best-at-edges" reordering works within the memory budget:

1. Memories are scored and sorted by relevance
2. The highest-scoring memories are placed at the beginning and end of the memory section
3. Lower-scoring memories fill the middle
4. The total memory content stays within the 13,000 token budget

This ensures the most relevant memories are in the positions where the LLM pays the most attention.

## History Compression

The `HistoryCompressor` manages the 6,000 token history budget:

1. **Recent 5 messages** — Kept verbatim (no compression)
2. **Older messages** — Summarized in batches of 10
3. **Budget enforcement** — If compressed history still exceeds budget, oldest batches are dropped
4. **Fallback** — If no LLM is available, history is truncated to fit

```python
# Compression flow
messages = [msg1, msg2, ..., msg20]
recent = messages[-5:]  # Keep verbatim
older = messages[:-5]   # Compress in batches of 10

for batch in chunks(older, 10):
    summary = llm.summarize(batch)
    compressed.append(summary)

# Enforce budget
while token_count(compressed) > 6000:
    compressed.pop(0)  # Drop oldest
```
