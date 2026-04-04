# Extension Guide

This guide explains how to extend Brain Agent v2 with new tools, LLM providers, embedding providers, memory categories, and reasoning strategies.

## Adding a New Tool

Tools are the agent's way of interacting with the external world. Each tool is a class that implements a `run` method.

### Step 1: Create the Tool Class

Create a new file in `core/tools/`, e.g., `core/tools/my_tool.py`:

```python
"""My custom tool."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standard result type for all tools."""
    success: bool
    output: str
    error: str = ""


class MyTool:
    """Description of what your tool does.

    Args:
        permissions: Permission configuration dict.
    """

    def __init__(self, permissions: dict | None = None) -> None:
        self.permissions = permissions or {}

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            ToolResult with success status and output/error message.
        """
        try:
            # Check permissions
            if not self.permissions.get("my_tool", False):
                return ToolResult(
                    success=False,
                    output="",
                    error="Permission denied: my_tool is not enabled.",
                )

            # Your tool logic here
            param1 = kwargs.get("param1", "")
            result = f"Processed: {param1}"

            logger.info("MyTool executed successfully.")
            return ToolResult(success=True, output=result)

        except Exception as e:
            logger.error("MyTool failed: %s", e)
            return ToolResult(success=False, output="", error=str(e))
```

### Step 2: Register the Tool in the Executor

Edit `core/tools/executor.py` to add your tool to the dispatcher:

```python
from .my_tool import MyTool

class ToolExecutor:
    def __init__(self, config: AgentConfig) -> None:
        # ...existing tools...
        self.tools = {
            "bash": BashTool(config.permissions),
            "read_file": FileOpsTool(config.permissions),
            # ...existing tools...
            "my_tool": MyTool(config.permissions),  # Add your tool
        }
```

### Step 3: Add Pydantic Validation

Edit `core/llm/tool_parser.py` to add parameter validation:

```python
from pydantic import BaseModel, Field

class MyToolParams(BaseModel):
    """Parameters for the my_tool tool."""
    param1: str = Field(..., description="Description of param1")
    param2: int = Field(default=0, description="Optional param2")
```

Add it to the tool registry:

```python
TOOL_SCHEMAS = {
    "bash": BashParams,
    "read_file": ReadFileParams,
    # ...existing tools...
    "my_tool": MyToolParams,  # Add your tool
}
```

### Step 4: Update the Tool Call Parser

Ensure the XML parser recognizes your tool. The parser uses regex patterns:

```python
# In ToolCallParser, your tool will be automatically recognized
# if it's in TOOL_SCHEMAS. The XML format is:
#
# <tool name="my_tool">
#   <param name="param1">value</param>
#   <param name="param2">42</param>
# </tool>
```

### Step 5: Add Tests

Create `tests/test_my_tool.py`:

```python
import pytest
from core.tools.my_tool import MyTool, ToolResult


class TestMyTool:
    def test_basic_execution(self):
        tool = MyTool(permissions={"my_tool": True})
        result = tool.run(param1="test")
        assert result.success
        assert "Processed: test" in result.output

    def test_permission_denied(self):
        tool = MyTool(permissions={"my_tool": False})
        result = tool.run(param1="test")
        assert not result.success
        assert "Permission denied" in result.error
```

## Adding a New LLM Provider

### Step 1: Create the Provider Class

Create a new file in `core/llm/`, e.g., `core/llm/anthropic_provider.py`:

```python
"""Anthropic LLM provider."""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from .provider import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider.

    Args:
        api_key: Anthropic API key.
        model: Model name (e.g., "claude-3-sonnet-20240229").
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    @classmethod
    def from_env(cls) -> AnthropicProvider:
        """Create provider from environment variables."""
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        return cls(api_key=api_key, model=model)

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the model."""
        # Implement Anthropic API call
        # ...
        pass

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from the model."""
        # Implement streaming
        # ...
        pass

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"
```

### Step 2: Register in Agent Config

Edit `core/config.py` to support the new provider:

```python
class ModelConfig(BaseModel):
    provider: str = "ollama"  # "ollama", "openrouter", "anthropic"
    model: str = "qwen3.5:4b-nothink"
    # ...existing fields...
```

### Step 3: Wire in the Agent

Edit `core/agent.py` to instantiate the provider:

```python
from .llm.anthropic_provider import AnthropicProvider

class BrainAgent:
    def _init_llm(self) -> None:
        provider = self.config.model.provider
        if provider == "anthropic":
            self.llm = AnthropicProvider.from_env()
        elif provider == "ollama":
            self.llm = OllamaProvider.from_env()
        # ...existing providers...
```

## Adding a New Embedding Provider

### Step 1: Create the Provider Class

```python
"""OpenAI embedding provider."""
from __future__ import annotations

import logging
from typing import Optional

from .embeddings import EmbeddingCache

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider:
    """OpenAI text-embedding-3-small provider.

    Args:
        api_key: OpenAI API key.
        model: Model name.
        dimension: Embedding dimension.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self._cache = EmbeddingCache(maxsize=10000)

    def embed(self, text: str) -> Optional[list[float]]:
        """Embed a single text string."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        # Implement OpenAI API call
        # ...
        embedding = [...]  # 1536-dim vector
        self._cache.set(text, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Embed multiple texts in one API call."""
        # Implement batch embedding
        # ...
        pass
```

## Adding a New Memory Category

Memory categories are used in the retrieval pipeline for category-specific bonuses.

### Step 1: Define the Category

Categories are free-form strings, but you should document them. Add to the list in `core/memory/database.py` or simply use the category when inserting:

```python
db.execute(
    "INSERT INTO memories (content, category, source, importance, created_at) VALUES (?, ?, ?, ?, ?)",
    (content, "my_category", "extracted", 0.7, time.time()),
)
```

### Step 2: Add Category Bonus (Optional)

Edit `core/memory/reranker.py` to add a bonus for your category:

```python
def _heuristic_score(self, memory: dict, rank: int) -> float:
    # ...existing code...

    # Add your category bonus
    category_mult = 1.0
    if category == "correction":
        category_mult = 1.5
    elif category == "preference":
        category_mult = 1.3
    elif category == "my_category":
        category_mult = 1.2  # Your bonus
    # ...existing code...
```

## Adding a New Entity Type to the Knowledge Graph

### Step 1: Update Valid Entity Types

Edit `core/memory/kg.py`:

```python
VALID_ENTITY_TYPES = {
    "person",
    "tool",
    "concept",
    "project",
    "file",
    "service",
    "language",
    "config",
    "other",
    "my_entity_type",  # Add your type
}
```

## Adding a New Relation Type to the Knowledge Graph

Edit `core/memory/kg.py`:

```python
VALID_RELATION_TYPES = {
    "uses",
    "prefers",
    "part_of",
    # ...existing types...
    "my_relation_type",  # Add your type
}
```

## Adding a New Reasoning Strategy

### Step 1: Extend the Reasoning Engine

Edit `core/memory/reasoning.py`:

```python
class ReasoningEngine:
    def __init__(self, db, llm, kg) -> None:
        # ...existing code...
        pass

    def my_reasoning_strategy(self) -> list[str]:
        """Implement your custom reasoning strategy.

        Returns:
            List of insights or conclusions.
        """
        # Your reasoning logic here
        insights = []
        # ...
        return insights

    def run_cycle(self) -> dict:
        """Run a full reasoning cycle."""
        results = {
            "gap_analysis": self.analyze_gaps(),
            "cross_domain": self.find_cross_domain_connections(),
            "rule_inference": self.infer_rules(),
            "contradictions": self.check_contradictions(),
            "procedure_improvement": self.improve_procedures(),
            "my_reasoning": self.my_reasoning_strategy(),  # Add your strategy
        }
        return results
```

## Adding a New Consolidation Strategy

Edit `core/memory/consolidation.py`:

```python
class ConsolidationEngine:
    def my_consolidation_strategy(self) -> dict:
        """Implement your custom consolidation strategy.

        Returns:
            Dict with statistics about what was consolidated.
        """
        stats = {"merged": 0, "promoted": 0, "decayed": 0}
        # Your consolidation logic here
        return stats

    def run_consolidation(self) -> dict:
        """Run the full consolidation cycle."""
        results = {
            "duplicate_merge": self.merge_near_duplicates(),
            "contradiction_resolution": self.resolve_contradictions(),
            "importance_decay": self.apply_importance_decay(),
            "importance_promotion": self.apply_importance_promotion(),
            "my_consolidation": self.my_consolidation_strategy(),  # Add yours
        }
        return results
```

## Configuration Extension

To add new configuration options, edit `core/config.py`:

```python
from pydantic import BaseModel, Field

class MyFeatureConfig(BaseModel):
    """Configuration for my feature."""
    enabled: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_items: int = Field(default=100, ge=1)

class AgentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    # ...existing config sections...
    my_feature: MyFeatureConfig = Field(default_factory=MyFeatureConfig)  # Add yours
```

Then use it in your code:

```python
if self.config.my_feature.enabled:
    threshold = self.config.my_feature.threshold
    # ...
```
