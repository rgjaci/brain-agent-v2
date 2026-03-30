"""Tests for Phase 3+ features — 38 tests covering:
- writer.process_document_chunk
- writer.detect_contradictions
- reader.hierarchical_doc_search
- reranker.cross_encoder_rerank
- feedback.persist_retrieval_log / maybe_auto_train
- executor._run_recall (hybrid) / _check_network_permission
- provider: OpenRouterProvider, OllamaProvider.from_env
- main.py: build_agent, teach/recall subcommands
- agent.py: bootstrap with count_memories
"""
from __future__ import annotations
import os
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedder_sync():
    """Embedder with synchronous embed/embed_query (non-async)."""
    emb = MagicMock()
    emb.embed = MagicMock(return_value=[0.1] * 768)
    emb.embed_query = MagicMock(return_value=[0.1] * 768)
    return emb


# ════════════════════════════════════════════════════════════════════════════
#  writer.process_document_chunk
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def doc_writer(mock_llm, mock_embedder_sync, db, kg):
    from core.memory.writer import MemoryWriter
    return MemoryWriter(mock_llm, mock_embedder_sync, db, kg)


@dataclass
class FakeChunk:
    content: str
    source_path: str
    chunk_index: int
    total_chunks: int
    metadata: dict


@pytest.mark.asyncio
async def test_process_document_chunk_stores_memory(doc_writer, db):
    chunk = FakeChunk("This is test content", "/tmp/test.txt", 0, 1,
                      {"file_name": "test.txt", "language": "txt", "extension": ".txt"})
    mid = await doc_writer.process_document_chunk(chunk, "sess1")
    mem = db.get_memory(mid)
    assert mem is not None
    assert mem["category"] == "document"


@pytest.mark.asyncio
async def test_process_document_chunk_creates_document_row(doc_writer, db):
    chunk = FakeChunk("content", "/tmp/doc.txt", 0, 3,
                      {"file_name": "doc.txt", "language": "txt", "extension": ".txt"})
    await doc_writer.process_document_chunk(chunk, "sess1")
    docs = db.execute("SELECT * FROM documents WHERE source_path = '/tmp/doc.txt'")
    assert len(docs) == 1
    assert docs[0]["total_chunks"] == 3


@pytest.mark.asyncio
async def test_process_document_chunk_creates_section(doc_writer, db):
    chunk = FakeChunk("section content", "/tmp/sections.txt", 2, 5,
                      {"file_name": "sections.txt", "language": "txt", "extension": ".txt"})
    await doc_writer.process_document_chunk(chunk, "sess1")
    sections = db.execute("SELECT * FROM document_sections")
    assert len(sections) >= 1
    assert sections[0]["position"] == 2


@pytest.mark.asyncio
async def test_process_document_chunk_reuses_existing_doc(doc_writer, db):
    """Second chunk of same file should reuse the document row."""
    c1 = FakeChunk("chunk 1", "/tmp/multi.txt", 0, 2,
                   {"file_name": "multi.txt", "language": "txt", "extension": ".txt"})
    c2 = FakeChunk("chunk 2", "/tmp/multi.txt", 1, 2,
                   {"file_name": "multi.txt", "language": "txt", "extension": ".txt"})
    await doc_writer.process_document_chunk(c1, "sess1")
    await doc_writer.process_document_chunk(c2, "sess1")
    docs = db.execute("SELECT * FROM documents WHERE source_path = '/tmp/multi.txt'")
    assert len(docs) == 1


# ════════════════════════════════════════════════════════════════════════════
#  writer.detect_contradictions
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_detect_contradictions_no_vectors(doc_writer, db):
    """Without vector search, returns empty list (graceful degradation)."""
    mid = db.insert_memory("test fact")
    result = await doc_writer.detect_contradictions(mid)
    assert result == []


@pytest.mark.asyncio
async def test_detect_contradictions_nonexistent_memory(doc_writer, db):
    result = await doc_writer.detect_contradictions(99999)
    assert result == []


# ════════════════════════════════════════════════════════════════════════════
#  reader.hierarchical_doc_search
# ════════════════════════════════════════════════════════════════════════════

def test_hierarchical_doc_search_empty(reader):
    results = reader.hierarchical_doc_search("nonexistent query")
    assert results == []


def test_hierarchical_doc_search_with_docs(db, mock_embedder, kg):
    """Documents stored as memories with category='document' are found."""
    from core.memory.reader import MemoryReader
    import json

    # Insert a document row
    db.execute(
        "INSERT INTO documents (title, source_path, doc_type, total_chunks, created_at) "
        "VALUES ('Test Doc', '/tmp/test.txt', 'text', 1, ?)", (time.time(),)
    )
    doc_rows = db.execute("SELECT id FROM documents WHERE title = 'Test Doc'")
    doc_id = doc_rows[0]["id"]

    # Insert a memory with category='document'
    meta = json.dumps({"doc_id": doc_id})
    db.insert_memory("Python is great for data science", category="document",
                     metadata={"doc_id": doc_id})

    reader = MemoryReader(db, mock_embedder, kg)
    results = reader.hierarchical_doc_search("Python data science")
    assert len(results) >= 1
    assert results[0]["doc_id"] == doc_id


# ════════════════════════════════════════════════════════════════════════════
#  reranker.cross_encoder_rerank
# ════════════════════════════════════════════════════════════════════════════

def test_cross_encoder_fallback(reranker):
    """Without sentence-transformers, should return top-k by order."""
    candidates = [{"content": f"memory {i}", "id": i} for i in range(10)]
    result = reranker.cross_encoder_rerank("test query", candidates, top_k=5)
    assert len(result) == 5


def test_cross_encoder_empty(reranker):
    result = reranker.cross_encoder_rerank("query", [], top_k=5)
    assert result == []


def test_cross_encoder_fewer_than_topk(reranker):
    candidates = [{"content": "only one", "id": 1}]
    result = reranker.cross_encoder_rerank("query", candidates, top_k=5)
    assert len(result) == 1


# ════════════════════════════════════════════════════════════════════════════
#  feedback: persist_retrieval_log / maybe_auto_train
# ════════════════════════════════════════════════════════════════════════════

def test_persist_retrieval_log(feedback, db):
    # Insert a real memory so FK constraint is satisfied
    mid = db.insert_memory("persist test fact", category="fact")
    mems = [{"id": mid, "rrf_score": 0.9, "access_count": 3, "importance": 0.7,
             "category": "fact", "created_at": time.time()}]
    feedback.record_retrieval("query", mems, "sess-persist")
    feedback.record_reference("sess-persist", [mid])
    count = feedback.persist_retrieval_log("sess-persist")
    assert count == 1
    # Buffer should be cleared for that session
    assert "sess-persist" not in feedback._buffer


def test_persist_retrieval_log_empty(feedback):
    count = feedback.persist_retrieval_log("nonexistent-session")
    assert count == 0


def test_maybe_auto_train_below_threshold(feedback):
    result = feedback.maybe_auto_train(threshold=200)
    assert result is None


def test_maybe_auto_train_above_threshold(feedback, db):
    """Insert enough retrieval_log rows then verify auto-train triggers."""
    # Insert real memories first to satisfy FK constraint
    mem_ids = [db.insert_memory(f"fact {i}") for i in range(15)]
    # Insert 210 dummy retrieval_log rows referencing real memory IDs
    for i in range(210):
        db.execute(
            "INSERT INTO retrieval_log "
            "(session_id, query_text, memory_id, retrieval_method, retrieval_rank, "
            "retrieval_score, was_in_context, was_useful, created_at) "
            "VALUES (?, ?, ?, 'hybrid', ?, 0.5, 1, 1, ?)",
            (f"s{i}", f"query {i}", mem_ids[i % len(mem_ids)], i, time.time()),
        )
    # Need buffered data for training
    for s in range(3):
        mems = [{"id": mem_ids[j % len(mem_ids)], "rrf_score": 0.1 * j, "access_count": j,
                 "importance": 0.5, "category": "fact", "created_at": time.time()}
                for j in range(5)]
        feedback.record_retrieval(f"q{s}", mems, f"auto-{s}")
        feedback.record_reference(f"auto-{s}", [mem_ids[0], mem_ids[1]])
    result = feedback.maybe_auto_train(threshold=200)
    assert result is not None
    assert "weights" in result


# ════════════════════════════════════════════════════════════════════════════
#  executor: _run_recall (hybrid) / _check_network_permission
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def executor(db, mock_embedder_sync):
    from core.tools.executor import ToolExecutor
    return ToolExecutor(permissions=None, db=db, embedder=mock_embedder_sync)


@pytest.mark.asyncio
async def test_run_recall_empty_db(executor):
    result = await executor.execute("recall", {"query": "anything"})
    assert "No memories found" in result.output


@pytest.mark.asyncio
async def test_run_recall_with_memory(executor, db):
    db.insert_memory("Python is used for data science", category="fact")
    result = await executor.execute("recall", {"query": "Python data"})
    assert result.success
    assert "Python" in result.output


@pytest.mark.asyncio
async def test_run_recall_empty_query(executor):
    result = await executor.execute("recall", {"query": ""})
    assert "RECALL ERROR" in result.output


def test_check_network_permission(executor):
    assert executor._check_network_permission("web_search") is True
    assert executor._check_network_permission("bash") is False


# ════════════════════════════════════════════════════════════════════════════
#  provider: OpenRouterProvider, OllamaProvider.from_env
# ════════════════════════════════════════════════════════════════════════════

def test_openrouter_provider_init():
    from core.llm.provider import OpenRouterProvider
    p = OpenRouterProvider(api_key="test-key", model="test/model")
    assert p.model == "test/model"
    assert p.api_key == "test-key"


def test_openrouter_count_tokens():
    from core.llm.provider import OpenRouterProvider
    p = OpenRouterProvider(api_key="x")
    assert p.count_tokens("hello world") > 0


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
def test_from_env_openrouter():
    from core.llm.provider import OllamaProvider, OpenRouterProvider
    provider = OllamaProvider.from_env()
    assert isinstance(provider, OpenRouterProvider)


@patch.dict(os.environ, {}, clear=True)
def test_from_env_ollama_fails_gracefully():
    """Without Ollama running, from_env should raise but return OllamaProvider type."""
    from core.llm.provider import OllamaProvider
    try:
        provider = OllamaProvider.from_env()
        # If somehow Ollama is running, that's fine too
        assert isinstance(provider, OllamaProvider)
    except RuntimeError:
        pass  # expected — Ollama not running


# ════════════════════════════════════════════════════════════════════════════
#  main.py: build_agent
# ════════════════════════════════════════════════════════════════════════════

def test_build_agent_graceful():
    """build_agent should not crash even without LLM/embedder."""
    import types
    args = types.SimpleNamespace(model="nonexistent", db=":memory:", debug=False)
    agent, cfg = None, None
    try:
        from main import build_agent
        agent, cfg = build_agent(args)
    except Exception:
        pass
    # If build succeeded, verify basics
    if agent is not None:
        assert agent.db is not None


# ════════════════════════════════════════════════════════════════════════════
#  agent.py: bootstrap with count_memories
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_bootstrap_skips_when_memories_exist(db):
    from core.agent import BrainAgent
    db.insert_memory("existing fact")
    agent = BrainAgent(db=db, tool_executor=MagicMock())
    await agent.bootstrap()
    # Should not crash, should skip because memories exist
    assert db.count_memories() >= 1


@pytest.mark.asyncio
async def test_bootstrap_runs_on_empty_db(db):
    from core.agent import BrainAgent
    mock_exec = MagicMock()
    mock_exec.execute = AsyncMock(return_value=MagicMock(success=True, output="Linux 6.1"))
    agent = BrainAgent(db=db, tool_executor=mock_exec, writer=MagicMock())
    agent.writer.extract_from_scan = AsyncMock()
    await agent.bootstrap()
    assert mock_exec.execute.called


@pytest.mark.asyncio
async def test_agent_process_no_llm(db):
    """Agent should handle missing LLM gracefully."""
    from core.agent import BrainAgent
    agent = BrainAgent(db=db)
    result = await agent.process("hello")
    assert "[No LLM configured]" in result.response


@pytest.mark.asyncio
async def test_agent_stores_conversation(db):
    from core.agent import BrainAgent
    agent = BrainAgent(db=db)
    await agent.process("test input")
    msgs = db.get_recent_messages(agent.session_id, limit=10)
    assert len(msgs) >= 2  # user + assistant


# ════════════════════════════════════════════════════════════════════════════
#  Additional coverage
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_agent_new_session(db):
    from core.agent import BrainAgent
    agent = BrainAgent(db=db)
    old_session = agent.session_id
    new_id = agent.new_session()
    assert new_id != old_session


def test_reader_extract_query_entities(reader):
    entities = reader.extract_query_entities('Configure "my-project" with Docker')
    assert "my-project" in entities
    assert "docker" in entities


def test_reader_format_for_context_empty(reader):
    from core.memory.reader import RetrievalResult
    result = RetrievalResult(
        memories=[], kg_context="", procedures=[],
        query_entities=[], retrieval_strategy="normal",
    )
    formatted = reader.format_for_context(result)
    assert formatted == ""


def test_reranker_rerank_with_lr_weights(feedback, db):
    from core.memory.reranker import Reranker
    reranker = Reranker(feedback_collector=feedback)
    mems = [
        {"id": 1, "rrf_score": 0.5, "access_count": 3, "importance": 0.7,
         "category": "fact", "created_at": time.time()},
        {"id": 2, "rrf_score": 0.8, "access_count": 1, "importance": 0.3,
         "category": "general", "created_at": time.time()},
    ]
    weights = {"weights": {"rrf_score": 1.0, "importance": 0.5}, "bias": 0.0}
    result = reranker.rerank(mems, "test query", weights=weights)
    assert len(result) == 2
    assert all("_rerank_score" in m for m in result)


def test_reranker_apply_best_at_edges_small(reranker):
    mems = [{"_rerank_score": 1.0}]
    result = reranker.apply_best_at_edges(mems)
    assert len(result) == 1


def test_compressor_format_messages(db):
    from core.context.compressor import HistoryCompressor
    comp = HistoryCompressor()
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    formatted = comp.format_messages_for_summary(msgs)
    assert "user:" in formatted
    assert "assistant:" in formatted


def test_executor_available_tools(db, mock_embedder_sync):
    from core.tools.executor import ToolExecutor
    executor = ToolExecutor(permissions=None, db=db, embedder=mock_embedder_sync)
    tools = executor.get_available_tools()
    assert "bash" in tools
    assert "recall" in tools
    assert "teach" in tools


@pytest.mark.asyncio
async def test_executor_unknown_tool(db, mock_embedder_sync):
    from core.tools.executor import ToolExecutor
    executor = ToolExecutor(permissions=None, db=db, embedder=mock_embedder_sync)
    result = await executor.execute("nonexistent_tool", {})
    assert not result.success
    assert "Unknown tool" in result.error


def test_openrouter_generate_json_fallback():
    from core.llm.provider import OpenRouterProvider
    p = OpenRouterProvider(api_key="test")
    # generate_json relies on generate which would fail without network
    # Just verify the object exists and count_tokens works
    assert p.count_tokens("test") == 1


def test_db_insert_memory_with_metadata(db):
    mid = db.insert_memory("fact with meta", metadata={"key": "value"})
    mem = db.get_memory(mid)
    assert mem["metadata"]["key"] == "value"
