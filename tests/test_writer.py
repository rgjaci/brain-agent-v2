"""Tests for MemoryWriter — fact, graph, and procedure extraction."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.memory.writer import MemoryWriter


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate_json.return_value = []
    return llm


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    return embedder


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.execute.return_value = []
    return db


@pytest.fixture
def mock_kg():
    kg = MagicMock()
    return kg


@pytest.fixture
def writer(mock_llm, mock_embedder, mock_db, mock_kg):
    return MemoryWriter(mock_llm, mock_embedder, mock_db, mock_kg)


class TestFactExtraction:
    @pytest.mark.asyncio
    async def test_extract_facts_empty_response(self, writer, mock_llm):
        mock_llm.generate_json.return_value = []
        facts = await writer.extract_facts("User: Hi", "Agent: Hello")
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_single_fact(self, writer, mock_llm):
        mock_llm.generate_json.return_value = [
            {"content": "User's name is Sarah", "category": "fact", "importance": 0.9}
        ]
        facts = await writer.extract_facts("I'm Sarah", "Nice to meet you, Sarah!")
        assert len(facts) == 1
        assert facts[0].content == "User's name is Sarah"
        assert facts[0].category == "fact"
        assert facts[0].importance == 0.9

    @pytest.mark.asyncio
    async def test_extract_facts_multiple_facts(self, writer, mock_llm):
        mock_llm.generate_json.return_value = [
            {"content": "User uses Neovim", "category": "preference", "importance": 0.7},
            {"content": "User works at Acme Corp", "category": "fact", "importance": 0.8},
        ]
        facts = await writer.extract_facts(
            "I use Neovim and work at Acme Corp",
            "Got it, noted!",
        )
        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_extract_facts_invalid_json(self, writer, mock_llm):
        mock_llm.generate_json.side_effect = Exception("JSON parse error")
        facts = await writer.extract_facts("test", "test")
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_clamps_importance(self, writer, mock_llm):
        mock_llm.generate_json.return_value = [
            {"content": "test", "category": "fact", "importance": 1.5}
        ]
        facts = await writer.extract_facts("test", "test")
        assert len(facts) == 1
        assert facts[0].importance <= 1.0

    @pytest.mark.asyncio
    async def test_extract_facts_skips_empty_content(self, writer, mock_llm):
        mock_llm.generate_json.return_value = [
            {"content": "", "category": "fact", "importance": 0.5}
        ]
        facts = await writer.extract_facts("test", "test")
        assert facts == []


class TestGraphExtraction:
    @pytest.mark.asyncio
    async def test_extract_graph_empty(self, writer, mock_llm):
        mock_llm.generate_json.return_value = {"entities": [], "relations": []}
        entities, relations = await writer.extract_graph("Hi", "Hello")
        assert entities == []
        assert relations == []

    @pytest.mark.asyncio
    async def test_extract_graph_with_entities(self, writer, mock_llm):
        mock_llm.generate_json.return_value = {
            "entities": [
                {"name": "Python", "type": "language", "description": "Programming language"}
            ],
            "relations": []
        }
        entities, _relations = await writer.extract_graph("I use Python", "Great choice!")
        assert len(entities) == 1
        assert entities[0].name == "Python"

    @pytest.mark.asyncio
    async def test_extract_graph_with_relations(self, writer, mock_llm):
        mock_llm.generate_json.return_value = {
            "entities": [
                {"name": "User", "type": "person", "description": "The user"},
                {"name": "Docker", "type": "tool", "description": "Container platform"}
            ],
            "relations": [
                {"source": "User", "target": "Docker", "type": "uses", "detail": "for deployment"}
            ]
        }
        entities, relations = await writer.extract_graph("I deploy with Docker", "Noted!")
        assert len(entities) == 2
        assert len(relations) == 1


class TestProcedureExtraction:
    @pytest.mark.asyncio
    async def test_extract_procedure_empty(self, writer, mock_llm):
        mock_llm.generate_json.return_value = {}
        procedure = await writer.extract_procedure(
            "Deploy the app",
            "Done! I pushed to production.",
            [{"name": "bash", "params": {"command": "git push"}}],
        )
        assert procedure is None

    @pytest.mark.asyncio
    async def test_extract_procedure_valid(self, writer, mock_llm):
        mock_llm.generate_json.return_value = {
            "name": "deploy_app",
            "description": "Deploy application to production",
            "trigger_pattern": "deploy push production",
            "preconditions": ["git repo is clean"],
            "steps": ["git push", "ssh to server", "restart service"],
            "warnings": ["Check logs after restart"],
            "context": "Production server at 10.0.0.1"
        }
        procedure = await writer.extract_procedure(
            "Deploy the app",
            "Done!",
            [{"name": "bash", "params": {"command": "git push"}}],
        )
        assert procedure is not None
        assert procedure.name == "deploy_app"


class TestProcessInteraction:
    @pytest.mark.asyncio
    async def test_process_interaction_no_llm(self, mock_embedder, mock_db, mock_kg):
        """Writer should handle missing LLM gracefully."""
        writer = MemoryWriter(None, mock_embedder, mock_db, mock_kg)
        # Should not raise
        await writer.process_interaction("test", "test", [], "session-1")

    @pytest.mark.asyncio
    async def test_process_interaction_no_embedder(self, mock_llm, mock_db, mock_kg):
        """Writer should handle missing embedder gracefully."""
        writer = MemoryWriter(mock_llm, None, mock_db, mock_kg)
        # Should not raise
        await writer.process_interaction("test", "test", [], "session-1")

    @pytest.mark.asyncio
    async def test_process_interaction_never_crashes_agent(self, writer):
        """Writer must never raise — exceptions are caught and logged."""
        # Even with malformed input, should not raise
        await writer.process_interaction("", "", [], "")
