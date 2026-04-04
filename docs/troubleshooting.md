# Troubleshooting Guide

Common issues and their solutions.

## sqlite-vec Issues

### Problem: `sqlite-vec` fails to load

```
ImportError: cannot import name 'sqlite_vec' from 'sqlite_vec'
```

**Solutions:**

1. **Reinstall the package:**
   ```bash
   pip uninstall sqlite-vec
   pip install sqlite-vec>=0.1.6
   ```

2. **Check if it loads:**
   ```bash
   python -c "import sqlite_vec; print(sqlite_vec.version())"
   ```

3. **On Linux, install SQLite development headers:**
   ```bash
   sudo apt-get install libsqlite3-dev  # Debian/Ubuntu
   sudo dnf install sqlite-devel        # Fedora
   sudo pacman -S sqlite                # Arch
   ```

4. **Check Python version compatibility:**
   ```bash
   python --version  # Must be 3.11+
   ```

### Problem: Vector dimension mismatch

```
sqlite3.OperationalError: dimension mismatch: expected 768, got X
```

**Solution:** Ensure your embedding provider produces 768-dimensional vectors. The Gemini `text-embedding-004` model produces 768-dim embeddings by default.

```python
# Check embedding dimension
from core.llm.embeddings import GeminiEmbeddingProvider
embedder = GeminiEmbeddingProvider(api_key="...")
embedding = embedder.embed("test")
print(len(embedding))  # Should be 768
```

## Ollama Issues

### Problem: Cannot connect to Ollama

```
ConnectionError: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded
```

**Solutions:**

1. **Check if Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Verify the model is pulled:**
   ```bash
   ollama list
   ollama pull qwen3.5:4b-nothink
   ```

4. **Check the base URL:**
   ```bash
   export OLLAMA_BASE_URL=http://localhost:11434
   ```

5. **If running in Docker, use host networking:**
   ```bash
   export OLLAMA_BASE_URL=http://host.docker.internal:11434
   ```

### Problem: Ollama model not found

```
Error: model "qwen3.5:4b-nothink" not found
```

**Solution:**
```bash
ollama pull qwen3.5:4b-nothink
# Or use a different model:
export OLLAMA_MODEL=llama3.2:3b
```

## Gemini API Issues

### Problem: Gemini API key not set

```
ValueError: GEMINI_API_KEY environment variable not set
```

**Solution:**
```bash
export GEMINI_API_KEY=your-key-here
# Or add to ~/.bashrc or ~/.zshrc:
echo 'export GEMINI_API_KEY=your-key-here' >> ~/.bashrc
source ~/.bashrc
```

### Problem: Gemini rate limit exceeded

```
google.api_core.exceptions.ResourceExhausted: 429 Too Many Requests
```

**Solutions:**

1. **The `EmbeddingCache` should reduce API calls.** Verify it's working:
   ```python
   # The cache uses SHA-256 keyed LRU with 10,000 entries
   # Duplicate queries should be served from cache
   ```

2. **Wait and retry** — Gemini free tier has rate limits that reset.

3. **Consider upgrading** to a paid tier if you hit limits frequently.

4. **Batch embeddings** — The provider supports batch embedding calls which are more efficient.

## Database Issues

### Problem: Database file not found

```
sqlite3.OperationalError: unable to open database file
```

**Solutions:**

1. **Check the database path:**
   ```bash
   # Default location
   ls -la ~/.brain_agent/memory.db
   ```

2. **Create the directory:**
   ```bash
   mkdir -p ~/.brain_agent
   ```

3. **Custom path in config:**
   ```yaml
   database:
     path: /custom/path/memory.db
   ```

### Problem: Database corruption

```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions:**

1. **Try to recover:**
   ```bash
   sqlite3 ~/.brain_agent/memory.db ".recover" | sqlite3 ~/.brain_agent/memory_recovered.db
   mv ~/.brain_agent/memory_recovered.db ~/.brain_agent/memory.db
   ```

2. **Restore from backup:**
   ```bash
   cp ~/.brain_agent/memory.db.backup.20260329 ~/.brain_agent/memory.db
   ```

3. **Start fresh** (loses all data):
   ```bash
   rm ~/.brain_agent/memory.db
   # The agent will recreate it on next run
   ```

## LLM Issues

### Problem: LLM not available (graceful degradation)

The agent works without an LLM but with limited functionality:

- ✅ Retrieval still works
- ✅ Context assembly still works
- ❌ No generation (can't produce responses)
- ❌ No knowledge extraction
- ❌ No history compression

**Solution:** Set up an LLM provider (Ollama or OpenRouter) as described in the [Setup Guide](setup.md).

### Problem: Tool call parsing fails

```
ToolCallParser: Failed to parse tool call
```

**Solutions:**

1. **Check the XML format** — Tool calls must follow this format:
   ```xml
   <tool name="bash">
     <param name="command">ls -la</param>
   </tool>
   ```

2. **Verify the tool is registered** in `core/llm/tool_parser.py`:
   ```python
   TOOL_SCHEMAS = {
       "bash": BashParams,
       "read_file": ReadFileParams,
       # ...
   }
   ```

3. **Check the model supports tool calling** — Some smaller models struggle with XML formatting. Try a larger model.

## TUI Issues

### Problem: TUI doesn't start

```
ModuleNotFoundError: No module named 'textual'
```

**Solution:**
```bash
pip install textual>=0.80
# Or install with dev dependencies:
pip install -e '.[dev]'
```

### Problem: TUI rendering issues

**Solutions:**

1. **Ensure your terminal supports Unicode and colors**
2. **Try a different terminal emulator**
3. **Use headless mode instead:**
   ```bash
   python main.py chat
   ```

## Performance Issues

### Problem: Retrieval is slow

**Solutions:**

1. **Check the number of memories:**
   ```bash
   python main.py stats
   ```

2. **Run consolidation to clean up:**
   ```python
   # In the agent, consolidation runs automatically every 10 turns
   # You can also trigger it manually
   agent.consolidation_engine.run_consolidation()
   ```

3. **Reduce top_k in config:**
   ```yaml
   retrieval:
     top_k: 5  # Default is 10
   ```

4. **Disable KG traversal for faster retrieval:**
   ```yaml
   retrieval:
     enable_kg_traversal: false
   ```

### Problem: High memory usage

**Solutions:**

1. **Check database size:**
   ```bash
   du -sh ~/.brain_agent/memory.db
   ```

2. **Run consolidation to merge duplicates:**
   ```python
   agent.consolidation_engine.merge_near_duplicates()
   ```

3. **Vacuum the database:**
   ```bash
   sqlite3 ~/.brain_agent/memory.db "VACUUM;"
   ```

## Common Error Messages

| Error | Cause | Solution |
|---|---|---|
| `GEMINI_API_KEY not set` | Missing API key | `export GEMINI_API_KEY=...` |
| `Connection refused: 11434` | Ollama not running | `ollama serve` |
| `model not found` | Model not pulled | `ollama pull <model>` |
| `dimension mismatch` | Wrong embedding size | Use 768-dim model |
| `no module named textual` | Missing TUI dependency | `pip install textual` |
| `database disk image is malformed` | DB corruption | Recover or restore backup |
| `Permission denied: bash` | Bash disabled in config | Set `permissions.bash: true` |
| `ToolCallParser: Failed to parse` | Malformed XML | Check XML format |
| `429 Too Many Requests` | Gemini rate limit | Wait or upgrade tier |
| `ImportError: sqlite_vec` | sqlite-vec not installed | `pip install sqlite-vec` |

## Debug Mode

Enable debug logging to see detailed information:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Or run with verbose output
python main.py chat --verbose
```

In the TUI, press `Ctrl+D` to toggle the debug panel.

## Getting Help

1. **Check the logs** — Most issues are logged with helpful context
2. **Run the health check:**
   ```bash
   python health_check.py
   ```
3. **Check test results:**
   ```bash
   pytest -v
   ```
4. **Review the [Architecture docs](architecture.md)** for understanding how components interact
