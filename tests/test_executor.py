"""Tests for ToolExecutor — dispatch, error handling, and tool results."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.tools.executor import ToolExecutor, ToolResult


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    return MagicMock()


@pytest.fixture
def mock_writer():
    return MagicMock()


@pytest.fixture
def mock_ingester():
    return MagicMock()


@pytest.fixture
def executor(mock_db, mock_embedder, mock_writer, mock_ingester):
    permissions = {
        "read_allowed": ["~/**"],
        "write_allowed": ["~/**"],
        "bash_allowed": True,
    }
    return ToolExecutor(
        permissions=permissions,
        db=mock_db,
        embedder=mock_embedder,
        writer=mock_writer,
        ingester=mock_ingester,
    )


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, executor):
        result = await executor.execute("nonexistent_tool", {})
        assert not result.success
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_never_raises(self, executor):
        """ToolExecutor.execute should never raise an exception."""
        # Even with bad params, it should return a ToolResult
        result = await executor.execute("bash", {"command": None})
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_bash_tool_execution(self, executor):
        with patch.object(executor.bash_tool, 'execute', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = MagicMock(success=True, output="hello", error="")
            with patch.object(executor.bash_tool, 'format_result', return_value="hello"):
                result = await executor.execute("bash", {"command": "echo hello"})
                assert result.success
                assert result.tool_name == "bash"

    @pytest.mark.asyncio
    async def test_bash_tool_timeout(self, executor):
        with patch.object(executor.bash_tool, 'execute', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = MagicMock(success=False, output="", error="timeout")
            with patch.object(executor.bash_tool, 'format_result', return_value="timeout"):
                result = await executor.execute("bash", {"command": "sleep 100", "timeout": 1})
                assert result.tool_name == "bash"

    @pytest.mark.asyncio
    async def test_teach_tool_requires_db(self, mock_embedder):
        """TeachTool should be None when db is not provided."""
        executor = ToolExecutor(db=None, embedder=mock_embedder)
        result = await executor.execute("teach", {"content": "test fact"})
        # Should handle missing teach_tool gracefully
        assert not result.success or result.tool_name == "teach"

    @pytest.mark.asyncio
    async def test_ingest_tool_requires_ingester(self):
        """IngestTool should handle missing ingester gracefully."""
        executor = ToolExecutor(ingester=None)
        result = await executor.execute("ingest", {"path": "/tmp/test.txt"})
        assert result.tool_name == "ingest"

    def test_available_tools(self, executor):
        tools = executor.get_available_tools()
        assert isinstance(tools, list)
        assert "bash" in tools
        assert "read_file" in tools
        assert "write_file" in tools


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(tool_name="test", success=True, output="done")
        assert result.success
        assert result.output == "done"
        assert result.error == ""

    def test_error_result(self):
        result = ToolResult(tool_name="test", success=False, output="", error="failed")
        assert not result.success
        assert result.error == "failed"
