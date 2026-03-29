"""Tests for ProcedureStore — storage, retrieval, record success/failure."""
from __future__ import annotations
import pytest
import time
from core.memory.database import MemoryDatabase
from core.memory.procedures import ProcedureStore, Procedure


def make_store():
    db = MemoryDatabase(":memory:")
    return ProcedureStore(db), db


def _proc(name, description, trigger=None, steps=None):
    return Procedure(
        id=None,
        name=name,
        description=description,
        trigger_pattern=trigger or name,
        preconditions=[],
        steps=steps or ["Step 1", "Step 2"],
        warnings=[],
        context="",
    )


# ── store ─────────────────────────────────────────────────────────────────────

def test_store_procedure_returns_id():
    store, _ = make_store()
    pid = store.store(_proc("deploy", "Deploy service to production"))
    assert isinstance(pid, int)
    assert pid > 0


def test_store_procedure_deduplicates_by_name():
    store, _ = make_store()
    store.store(_proc("deploy", "Deploy service"))
    store.store(_proc("deploy", "Deploy service (updated)"))
    all_procs = store.get_all()
    names = [p.name for p in all_procs]
    assert names.count("deploy") == 1


def test_store_multiple_procedures():
    store, _ = make_store()
    store.store(_proc("setup_ssh",    "Configure SSH keys"))
    store.store(_proc("deploy_nginx", "Deploy Nginx config"))
    store.store(_proc("backup_db",    "Backup PostgreSQL database"))
    all_procs = store.get_all()
    assert len(all_procs) >= 3


# ── find_relevant ─────────────────────────────────────────────────────────────

def test_find_relevant_returns_list():
    store, _ = make_store()
    store.store(_proc("setup_ssh", "Configure SSH keys and authorized_keys"))
    result = store.find_relevant("setup SSH keys")
    assert isinstance(result, list)


def test_find_relevant_no_procedures():
    store, _ = make_store()
    result = store.find_relevant("anything")
    assert result == []


def test_find_relevant_max_results():
    store, _ = make_store()
    for i in range(10):
        store.store(_proc(f"proc_{i}", f"Procedure number {i} for testing retrieval"))
    result = store.find_relevant("procedure testing", max_results=3)
    assert len(result) <= 3


def test_find_relevant_empty_query():
    store, _ = make_store()
    store.store(_proc("setup_ssh", "Configure SSH keys"))
    result = store.find_relevant("")
    assert isinstance(result, list)


# ── record_success / record_failure ───────────────────────────────────────────

def test_record_success_increments():
    store, _ = make_store()
    pid = store.store(_proc("deploy", "Deploy service"))
    store.record_success(pid)
    store.record_success(pid)
    proc = store.get_all()[0]
    assert proc.success_count >= 2


def test_record_failure_increments():
    store, _ = make_store()
    pid = store.store(_proc("deploy", "Deploy service"))
    store.record_failure(pid)
    proc = store.get_all()[0]
    assert proc.attempt_count >= 2  # attempt_count = success + failure + 1


def test_record_success_doesnt_crash():
    store, _ = make_store()
    pid = store.store(_proc("deploy", "Deploy service"))
    store.record_success(pid)  # should not raise


# ── format_for_context ────────────────────────────────────────────────────────

def test_format_for_context_returns_string():
    store, _ = make_store()
    proc = _proc("setup_ssh", "Configure SSH",
                 steps=["Generate key", "Copy to server", "Test connection"])
    store.store(proc)
    stored = store.get_all()
    result = store.format_for_context(stored)
    assert isinstance(result, str)
    assert len(result) > 10


def test_format_for_context_empty_list():
    store, _ = make_store()
    result = store.format_for_context([])
    assert isinstance(result, str)
