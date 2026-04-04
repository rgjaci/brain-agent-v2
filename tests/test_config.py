"""Tests for AgentConfig — loading, env overrides, validation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.config import AgentConfig, PermissionsConfig


class TestPermissionsConfig:
    def test_default_permissions(self):
        perms = PermissionsConfig()
        assert len(perms.read_allowed) > 0
        assert len(perms.write_allowed) > 0
        assert len(perms.read_blocked) > 0
        assert len(perms.write_blocked) > 0

    def test_blocked_includes_sensitive_paths(self):
        perms = PermissionsConfig()
        blocked = " ".join(perms.read_blocked + perms.write_blocked)
        assert ".ssh" in blocked or ".env" in blocked


class TestAgentConfigDefaults:
    def test_default_model(self):
        config = AgentConfig()
        assert config.model == "qwen3.5:4b"

    def test_default_ollama_url(self):
        config = AgentConfig()
        assert config.ollama_base_url == "http://localhost:11434"

    def test_default_temperature(self):
        config = AgentConfig()
        assert config.temperature == 0.3

    def test_default_dream_enabled(self):
        config = AgentConfig()
        assert config.dream_enabled is True

    def test_default_reasoning_enabled(self):
        config = AgentConfig()
        assert config.reasoning_enabled is True

    def test_default_retrieval_top_k(self):
        config = AgentConfig()
        assert config.retrieval_top_k == 5


class TestAgentConfigLoad:
    def test_load_from_nonexistent_file(self):
        """Loading from a non-existent file should return defaults."""
        config = AgentConfig.load(path=Path("/nonexistent/config.yaml"))
        assert config.model == "qwen3.5:4b"

    def test_load_from_yaml(self):
        """Loading from a valid YAML file should override defaults."""
        yaml_content = """
model: "llama3.2:3b"
temperature: 0.7
dream_enabled: false
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = AgentConfig.load(path=Path(f.name))
            assert config.model == "llama3.2:3b"
            assert config.temperature == 0.7
            assert config.dream_enabled is False
            os.unlink(f.name)

    def test_load_with_permissions(self):
        """Loading YAML with permissions section should work."""
        yaml_content = """
permissions:
  read_allowed:
    - "/custom/**"
  bash_timeout_default: 60
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = AgentConfig.load(path=Path(f.name))
            assert "/custom/**" in config.permissions.read_allowed
            assert config.permissions.bash_timeout_default == 60
            os.unlink(f.name)


class TestAgentConfigEnvOverrides:
    def test_gemini_api_key_override(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            config = AgentConfig.load(path=Path("/nonexistent"))
            assert config.gemini_api_key == "test-key-123"

    def test_model_override(self):
        with patch.dict(os.environ, {"BRAIN_AGENT_MODEL": "custom-model"}):
            config = AgentConfig.load(path=Path("/nonexistent"))
            assert config.model == "custom-model"

    def test_db_path_override(self):
        with patch.dict(os.environ, {"BRAIN_AGENT_DB": "/tmp/test.db"}):
            config = AgentConfig.load(path=Path("/nonexistent"))
            assert config.db_path == Path("/tmp/test.db")

    def test_debug_override(self):
        with patch.dict(os.environ, {"BRAIN_AGENT_DEBUG": "true"}):
            config = AgentConfig.load(path=Path("/nonexistent"))
            assert config.debug_mode is True

    def test_ollama_url_override(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:11434"}):
            config = AgentConfig.load(path=Path("/nonexistent"))
            assert config.ollama_base_url == "http://custom:11434"


class TestAgentConfigPathNormalization:
    def test_db_path_expansion(self):
        config = AgentConfig()
        config.db_path = Path("~/.brain_agent/memory.db")
        # After load, paths should be expanded
        config = AgentConfig.load(path=Path("/nonexistent"))
        assert not str(config.db_path).startswith("~")
