# Brain Agent v2 Documentation

## Getting Started

- **[Setup & Deployment](setup.md)** — Installation, configuration, and deployment guide
- **[Quick Start](../README.md#quick-start)** — Get up and running in 5 minutes

## Understanding the System

- **[Architecture Overview](architecture.md)** — System architecture, component details, and data flow diagrams
- **[Memory System](memory-system.md)** — Database schema, knowledge graph, consolidation, and dream engine
- **[Retrieval Pipeline](retrieval-pipeline.md)** — How the agent finds relevant memories (8-stage pipeline)
- **[Token Budget](token-budget.md)** — Context window allocation and tuning guide

## Development

- **[Extension Guide](extension-guide.md)** — How to add tools, providers, categories, and reasoning strategies
- **[Testing Guide](testing.md)** — Test suite overview and how to write tests
- **[Benchmarks](benchmarks.md)** — Benchmark methodology and results

## Operations

- **[Troubleshooting](troubleshooting.md)** — Common issues and solutions

## Reference

- **[README](../README.md)** — Project overview and quick start
- **[COMPLETION.md](../COMPLETION.md)** — Test results and benchmark scores
- **[spec_2.md](../spec_2.md)** — Original specification
- **[config.example.yaml](../config.example.yaml)** — Sample configuration file with all options documented

## Quick Links

| Topic | Document |
|---|---|
| Install dependencies | [Setup Guide](setup.md) |
| Configure API keys | [Setup Guide](setup.md#2-set-up-embeddings-required) |
| Understand how retrieval works | [Retrieval Pipeline](retrieval-pipeline.md) |
| Add a new tool | [Extension Guide](extension-guide.md#adding-a-new-tool) |
| Add a new LLM provider | [Extension Guide](extension-guide.md#adding-a-new-llm-provider) |
| Fix sqlite-vec errors | [Troubleshooting](troubleshooting.md#sqlite-vec-issues) |
| Fix Ollama connection errors | [Troubleshooting](troubleshooting.md#ollama-issues) |
| Tune token budgets | [Token Budget](token-budget.md#tuning-the-budget) |
| Run tests | [Testing Guide](testing.md) |
| Run benchmarks | [Benchmarks](benchmarks.md) |
