"""M3 batch-2 task 0: the test harness's lazy purpose grant is a REAL grant
row, and on SQLite it is re-checked on every call.

``MemoryEngine._session`` closes without committing, so a read-only entry
point (recall / panel / explain ...) rolls its session back -- including
any grant the harness inserted in that session.  Memoising "already
granted" per (tenant, user, purpose) on SQLite therefore lies after the
first read-only call: the second call would reach authorize() with no row
and be refused.  The PostgreSQL leg commits grants through the admin
engine, so the memo is valid there and only there.

These tests pin the harness, not the product: a regression here shows up
as spurious ``PurposeDenied`` in unrelated tests.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core.authorization import OPERATIONS, PurposeDenied
from mirror_memory.core.models import Base, PurposeGrant
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from tests.conftest import refusing_authz


@pytest.fixture
def sqlite_engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/harness-grant.db")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _engine_for(sqlite_engine, user_id: str) -> MemoryEngine:
    F = sessionmaker(bind=sqlite_engine)
    with tenant_scope("default", "default"), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, user_id, True)
        s.commit()
        s.close()
    return MemoryEngine(
        config=load_config("config/"), database_url=str(sqlite_engine.url),
        tenant_id="default", app_id="default",
    )


def _grant_rows(sqlite_engine, user_id: str) -> list[PurposeGrant]:
    F = sessionmaker(bind=sqlite_engine)
    with tenant_scope("default", "default"):
        s = F()
        try:
            return s.query(PurposeGrant).filter_by(user_id=user_id).all()
        finally:
            s.close()


def test_authorize_after_rollback_is_granted_again(sqlite_engine):
    """A session that rolls back discards the harness grant with it; the
    next authorize on the same user must re-issue (re-check, not memo).

    A DML statement opens the transaction FIRST so the harness SAVEPOINT is
    nested: pysqlite's legacy mode does not BEGIN for SELECTs, and an
    outermost SAVEPOINT's RELEASE is a COMMIT in SQLite -- that case cannot
    roll back and is covered by the entry-point tests below."""
    from mirror_memory.core import authorization

    user = "fresh-reader"
    F = sessionmaker(bind=sqlite_engine)
    with tenant_scope("default", "default"):
        s = F()
        try:
            with fixture_write_scope("open a real transaction"):
                set_memory_enabled(s, user, True)
            authorization.authorize_or_raise(s, user, "test", "recall")
            s.rollback()
            assert _grant_rows(sqlite_engine, user) == [], (
                "rollback must discard the savepoint-issued grant"
            )
            with fixture_write_scope("open a real transaction"):
                set_memory_enabled(s, user, True)
            authorization.authorize_or_raise(s, user, "test", "recall")
            s.rollback()
        finally:
            s.close()


def test_consecutive_read_only_entry_points_on_a_fresh_user_both_pass(sqlite_engine):
    user = "fresh-reader"
    eng = _engine_for(sqlite_engine, user)
    eng.recall(user_id=user, query="hello", purpose="test")
    eng.recall(user_id=user, query="hello again", purpose="test")


def test_lazy_grant_is_a_real_grant_not_a_bypass(sqlite_engine):
    """Opting out of issuance exposes the product's own no_grant refusal:
    the harness grants rows, it never skips authorize()."""
    user = "unsponsored"
    eng = _engine_for(sqlite_engine, user)
    with refusing_authz(), pytest.raises(PurposeDenied) as info:
        eng.recall(user_id=user, query="hello", purpose="test")
    assert info.value.decision.reason == "no_grant"
    assert info.value.operation == "recall"


def test_lazy_grant_committed_by_a_writing_entry_has_the_product_shape(sqlite_engine):
    """observe() commits, so its harness grant persists; the row is the one
    authorize() reads: version 1, every operation, issuer test-harness."""
    user = "fresh-writer"
    eng = _engine_for(sqlite_engine, user)
    eng.observe(user_id=user, session_id="s1", text="I prefer concise answers.", purpose="test")
    rows = _grant_rows(sqlite_engine, user)
    assert [(r.purpose, r.version, r.issuer, r.revoked_at) for r in rows] == [
        ("test", 1, "test-harness", None)
    ]
    import json

    assert set(json.loads(rows[0].allowed_operations)) == set(OPERATIONS)


def test_second_read_only_call_emits_no_harness_statements(sqlite_engine):
    """Perf-measurement fidelity (batch-1 evaluation): after the first touch
    the harness must add NOTHING to a read-only entry point.  The only
    statement naming mm_purpose_grants in the second recall is the product's
    own authorize SELECT; no harness INSERT, no harness existence check."""
    from sqlalchemy import event

    user = "fresh-reader"
    eng = _engine_for(sqlite_engine, user)
    eng.recall(user_id=user, query="warm-up", purpose="test")

    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(" ".join(statement.split()))

    event.listen(eng._engine, "before_cursor_execute", _record)
    try:
        eng.recall(user_id=user, query="second", purpose="test")
    finally:
        event.remove(eng._engine, "before_cursor_execute", _record)

    grant_statements = [s for s in statements if "mm_purpose_grants" in s]
    assert not [s for s in grant_statements if s.upper().startswith("INSERT")], grant_statements
    assert not [s for s in statements if "SAVEPOINT" in s.upper()], statements
    assert len([s for s in grant_statements if s.upper().startswith("SELECT")]) == 1, grant_statements


def test_memo_is_tenant_scoped_like_the_grant_rows(sqlite_engine):
    """Same database, same user id, two tenants: the grant row is tenant
    scoped, so the second tenant must be issued its own row rather than
    skipped by a memo keyed on the first (caught by
    test_tenant_isolation::test_background_tasks_do_not_cross_tenants)."""
    from mirror_memory.core import authorization

    user = "shared-name"
    F = sessionmaker(bind=sqlite_engine)
    for tenant in (("tA", "appA"), ("tB", "appB")):
        with tenant_scope(*tenant):
            s = F()
            try:
                authorization.authorize_or_raise(s, user, "test", "recall")
                s.commit()
            finally:
                s.close()
    for tenant in (("tA", "appA"), ("tB", "appB")):
        with tenant_scope(*tenant):
            s = F()
            try:
                rows = s.query(PurposeGrant).filter_by(user_id=user).all()
                assert [(r.tenant_id, r.app_id, r.purpose) for r in rows] == [(*tenant, "test")]
            finally:
                s.close()
