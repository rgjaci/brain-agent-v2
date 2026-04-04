# Setup & Deployment Guide

## Prerequisites

- **Python 3.11+** — The agent requires Python 3.11 or later
- **Git** — For cloning the repository

## Quick Setup

### 1. Install Dependencies

```bash
# Clone and install
cd brain-agent-v2
pip install -e '.[dev]'

# Optional: install ML dependencies for reranker training
pip install -e '.[ml]'

# Optional: install MCP server dependencies
pip install -e '.[server]'
```

### 2. Set Up Embeddings (Required)

The agent uses Google's Gemini API for embeddings (free tier available).

```bash
# Get a free API key from https://aistudio.google.com/
export GEMINI_API_KEY=your-key-here
```

### 3. Set Up LLM Provider (Optional but Recommended)

The agent works without an LLM (retrieval-only mode), but for full functionality you need one of:

#### Option A: Ollama (Local, Recommended)

```bash
# Install Ollama: https://ollama.com/
curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model
ollama pull qwen3.5:4b-nothink

# Start Ollama server (runs automatically after install)
# Default: http://localhost:11434

# Configure (optional — these are the defaults)
export OLLAMA_MODEL=qwen3.5:4b-nothink
export OLLAMA_BASE_URL=http://localhost:11434
```

#### Option B: OpenRouter (Cloud)

```bash
# Get an API key from https://openrouter.ai/
export OPENROUTER_API_KEY=your-key-here
export OPENROUTER_MODEL=qwen/qwen-2.5-7b-instruct  # or any model you prefer
```

### 4. Run the Agent

```bash
# TUI mode (recommended for interactive use)
python main.py

# Headless chat mode
python main.py chat

# Bootstrap — scan environment and store initial facts
python main.py bootstrap

# Teach a fact directly
python main.py teach "The project uses Python 3.11+"

# Recall/search memories
python main.py recall "what model does the agent use"

# Ingest a file or directory
python main.py ingest ./docs

# View memory statistics
python main.py stats

# Start MCP server for external agent integration
python main.py mcp
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | **Yes** | — | Google Gemini API key for embeddings |
| `OLLAMA_MODEL` | No | `qwen3.5:4b-nothink` | Ollama model name |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OPENROUTER_API_KEY` | No | — | OpenRouter API key (overrides Ollama) |
| `OPENROUTER_MODEL` | No | `qwen/qwen-2.5-7b-instruct` | OpenRouter model name |

## Configuration File

A sample configuration file is provided at `config.example.yaml` in the project root.

To use it:

```bash
# Create the config directory
mkdir -p ~/.brain_agent

# Copy and customize the example
cp config.example.yaml ~/.brain_agent/config.yaml
```

The agent loads configuration from `~/.brain_agent/config.yaml` by default (or a path specified via `--config`). Example:

```yaml
model:
  provider: ollama
  model: qwen3.5:4b-nothink
  base_url: http://localhost:11434

embeddings:
  provider: gemini
  model: text-embedding-004
  dimension: 768

retrieval:
  top_k: 10
  rrf_k: 60
  adaptive_threshold: 0.5

token_budget:
  total: 32768
  system: 500
  procedure: 2000
  kg: 1500
  memories: 13000
  history: 6000
  tool: 2000
  output: 4000
  query: 500
  overhead: 1268

permissions:
  read: true
  write: true
  bash: true
  network: true

autodream:
  enabled: true
  turn_interval: 50
  idle_seconds: 600

reasoning:
  enabled: true
  cycle_seconds: 120
```

## Database Location

The SQLite database is stored at `~/.brain_agent/memory.db` by default. You can change this in the config:

```yaml
database:
  path: /custom/path/memory.db
```

## sqlite-vec Installation Notes

The agent uses `sqlite-vec` for vector similarity search. It's included as a dependency and should install automatically via pip:

```bash
pip install sqlite-vec>=0.1.6
```

If you encounter issues:

```bash
# Check if sqlite-vec loads correctly
python -c "import sqlite_vec; print(sqlite_vec.version())"

# On some Linux systems, you may need sqlite development headers
sudo apt-get install libsqlite3-dev  # Debian/Ubuntu
sudo dnf install sqlite-devel        # Fedora
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_memory_read.py -v

# Run with coverage (requires pytest-cov)
pytest --cov=core --cov-report=term-missing
```

## Running Benchmarks

Benchmarks are separate from the test suite and require actual data:

```bash
# Recall benchmark
python benchmarks/recall_test.py

# Procedure benchmark
python benchmarks/procedure_test.py

# Reranker evaluation
python benchmarks/reranker_eval.py

# Export Raft data
python benchmarks/export_raft_data.py
```

## Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e '.[dev,ml,server]'

ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

CMD ["python", "main.py", "chat"]
```

Build and run:

```bash
docker build -t brain-agent .
docker run -it \
  -e GEMINI_API_KEY=your-key \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  brain-agent
```

## Production Considerations

1. **Database Backups** — The SQLite database is a single file. Regular backups are recommended:
   ```bash
   cp ~/.brain_agent/memory.db ~/.brain_agent/memory.db.backup.$(date +%Y%m%d)
   ```

2. **Rate Limiting** — The Gemini embedding provider has rate limits. The `EmbeddingCache` reduces API calls via SHA-256 keyed LRU caching.

3. **Memory Growth** — The database grows with use. The consolidation engine merges duplicates and applies importance decay, but monitor disk usage periodically:
   ```bash
   python main.py stats
   ```

4. **Permissions** — In production, consider restricting bash and network permissions:
   ```yaml
   permissions:
     read: true
     write: false
     bash: false
     network: false
   ```
