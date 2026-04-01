"""Brain Agent v2 — Configuration management.

Loads agent configuration from YAML with environment variable overrides.
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_DB_PATH = Path.home() / ".brain_agent" / "memory.db"
DEFAULT_CONFIG_PATH = Path.home() / ".brain_agent" / "config.yaml"
DEFAULT_WORKSPACE = Path.home() / ".brain_agent" / "workspace"


@dataclass
class PermissionsConfig:
    """Fine-grained permissions for file I/O, bash, and network."""

    read_allowed: list[str] = field(
        default_factory=lambda: [
            "~/.brain_agent/**",
            "${CWD}/**",
            "~/projects/**",
        ]
    )
    read_blocked: list[str] = field(
        default_factory=lambda: [
            "~/.ssh/id_*",
            "~/.aws/**",
            "**/.env",
        ]
    )
    write_allowed: list[str] = field(
        default_factory=lambda: [
            "~/.brain_agent/workspace/**",
            "${CWD}/**",
        ]
    )
    write_blocked: list[str] = field(
        default_factory=lambda: [
            "~/.bashrc",
            "~/.zshrc",
            "~/.brain_agent/config.yaml",
        ]
    )
    bash_blocked_patterns: list[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "sudo rm",
            "chmod 777",
            "> /dev/sd",
        ]
    )
    bash_timeout_default: int = 30
    bash_timeout_max: int = 300
    network_allowed: list[str] = field(
        default_factory=lambda: [
            "generativelanguage.googleapis.com",
            "api.duckduckgo.com",
            "html.duckduckgo.com",
        ]
    )
    network_default: str = "prompt_user"


@dataclass
class AgentConfig:
    """Complete configuration for Brain Agent v2.

    Supports loading from YAML and environment variable overrides.
    Hierarchy: defaults → YAML file → environment variables.
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    model: str = "qwen3.5:4b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 2000

    # ── Embeddings ────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dims: int = 768

    # ── Memory ────────────────────────────────────────────────────────────────
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    workspace_path: Path = field(default_factory=lambda: DEFAULT_WORKSPACE)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    token_budget_total: int = 32768

    # ── Permissions ───────────────────────────────────────────────────────────
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)

    # ── AutoDream ─────────────────────────────────────────────────────────────
    dream_enabled: bool = True
    dream_interval_turns: int = 50
    dream_idle_threshold: int = 600

    # ── System 2 Reasoning ───────────────────────────────────────────────────
    reasoning_enabled: bool = True
    reasoning_interval: int = 180  # seconds between cycles
    reasoning_max_cycles_per_session: int = 100

    # ── MCP Server ───────────────────────────────────────────────────────────
    mcp_transport: str = "stdio"
    mcp_host: str = "localhost"
    mcp_port: int = 8765

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug_mode: bool = False
    debug_file: Optional[Path] = None

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "AgentConfig":
        """Load config from YAML file, with environment variable overrides.

        Args:
            path: Path to the YAML config file. Defaults to
                  ``~/.brain_agent/config.yaml``.

        Returns:
            A fully populated :class:`AgentConfig` instance.
        """
        config = cls()

        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}

            for key, value in data.items():
                if key == "permissions" and isinstance(value, dict):
                    config.permissions = PermissionsConfig(**value)
                elif hasattr(config, key):
                    setattr(config, key, value)

        # ── Environment variable overrides ────────────────────────────────────
        if api_key := os.environ.get("GEMINI_API_KEY"):
            config.gemini_api_key = api_key
        if model := os.environ.get("BRAIN_AGENT_MODEL"):
            config.model = model
        if db_path := os.environ.get("BRAIN_AGENT_DB"):
            config.db_path = Path(db_path)
        if debug := os.environ.get("BRAIN_AGENT_DEBUG"):
            config.debug_mode = debug.lower() in ("1", "true", "yes")
        if ollama_url := os.environ.get("OLLAMA_BASE_URL"):
            config.ollama_base_url = ollama_url

        # ── Path normalisation ────────────────────────────────────────────────
        config.db_path = Path(config.db_path).expanduser()
        config.workspace_path = Path(config.workspace_path).expanduser()
        if config.debug_file is not None:
            config.debug_file = Path(config.debug_file).expanduser()

        return config

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Persist current configuration to a YAML file.

        Args:
            path: Destination path. Parent directories are created if absent.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "model": self.model,
            "ollama_base_url": self.ollama_base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "gemini_api_key": self.gemini_api_key,
            "embedding_model": self.embedding_model,
            "embedding_dims": self.embedding_dims,
            "db_path": str(self.db_path),
            "workspace_path": str(self.workspace_path),
            "retrieval_top_k": self.retrieval_top_k,
            "token_budget_total": self.token_budget_total,
            "debug_mode": self.debug_mode,
            "debug_file": str(self.debug_file) if self.debug_file else None,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def ensure_dirs(self) -> None:
        """Create necessary directories if they do not already exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        if self.debug_file is not None:
            self.debug_file.parent.mkdir(parents=True, exist_ok=True)

    def is_valid(self) -> tuple[bool, list[str]]:
        """Validate config for minimum required settings.

        Returns:
            A ``(ok, errors)`` tuple where *ok* is ``True`` when the config
            passes all validation checks and *errors* is a list of human-
            readable problem descriptions.
        """
        errors: list[str] = []

        if not self.model:
            errors.append("model must not be empty")
        if self.temperature < 0.0 or self.temperature > 2.0:
            errors.append(f"temperature {self.temperature} is outside [0.0, 2.0]")
        if self.max_tokens < 1:
            errors.append("max_tokens must be >= 1")
        if self.retrieval_top_k < 1:
            errors.append("retrieval_top_k must be >= 1")
        if self.embedding_dims not in (256, 512, 768, 1024, 1536, 3072):
            errors.append(
                f"embedding_dims {self.embedding_dims} is an unusual value — "
                "expected one of 256, 512, 768, 1024, 1536, 3072"
            )

        return (len(errors) == 0, errors)

    def __repr__(self) -> str:  # pragma: no cover
        masked_key = (
            f"{self.gemini_api_key[:4]}…" if len(self.gemini_api_key) > 4 else "***"
        )
        return (
            f"AgentConfig(model={self.model!r}, "
            f"db_path={self.db_path}, "
            f"gemini_api_key={masked_key!r}, "
            f"debug_mode={self.debug_mode})"
        )
