"""M3 batch-2 task 2: the access audit.

Every entry that returns memory writes one ``mm_access_audit`` row in the
same transaction as the read -- who, under which purpose and grant
version, through which entry, how many memories and which ids (ids only,
never content).  Refused reads write a ``denied`` row.  The audit is
fail-closed: if it cannot be written the entry returns nothing.  Rows
survive ``delete_memories``; retention is a config value enforced only by
an explicit, tenant-scoped purge.

The inventory test is driven by ENTRY_REGISTRY: every entry flagged
``reads_memory`` must have a driver here and must leave a row named after
it, so a new memory-reading entry cannot ship un-audited.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import access_audit
from mirror_memory.core.authorization import ENTRY_REGISTRY, PurposeDenied
from mirror_memory.core.models import AccessAudit, Base, Belief
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.server import create_app, register_tenant_key
from tests.conftest import refusing_authz

USER = "u-audit"
TENANT = ("default", "default")
PURPOSE = "api"


@pytest.fixture
def db(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/audit.db")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _seed(db, user_id: str = USER, tenant=TENANT, beliefs: int = 3) -> None:
    F = sessionmaker(bind=db)
    with tenant_scope(*tenant), fixture_write_scope("seed beliefs"):
        s = F()
        set_memory_enabled(s, user_id, True)
        for i in range(beliefs):
            s.add(Belief(user_id=user_id, dimension="topic", key=f"k{i}",
                         claim_text=f"CLAIM-{i}", confidence=0.9, layer="L2"))
        s.commit()
        s.close()


class _FakeLLM:
    """Enough of an LLM for get_verification_candidates to take its real
    path (without a client it returns [] before authorizing)."""

    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


def _engine(db, **kw) -> MemoryEngine:
    cfg = kw.pop("config", None) or load_config("config/")
    kw.setdefault("llm_client", _FakeLLM())
    return MemoryEngine(config=cfg, database_url=str(db.url),
                        tenant_id=TENANT[0], app_id=TENANT[1], **kw)


def _rows(db, tenant=TENANT) -> list[AccessAudit]:
    F = sessionmaker(bind=db)
    with tenant_scope(*tenant):
        s = F()
        try:
            return s.execute(select(AccessAudit).order_by(AccessAudit.id)).scalars().all()
        finally:
            s.close()


def _belief_ids(db) -> list[int]:
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            return [b.id for b in s.execute(select(Belief).order_by(Belief.id)).scalars()]
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Inventory: every memory-reading entry leaves a row named after it
# ---------------------------------------------------------------------------


def _library_drivers(eng: MemoryEngine, belief_id: int) -> dict[str, object]:
    return {
        "mirror_memory.api.MemoryEngine.recall": lambda: eng.recall(user_id=USER, query="claim", purpose=PURPOSE),
        "mirror_memory.api.MemoryEngine.panel": lambda: eng.panel(user_id=USER, purpose=PURPOSE),
        "mirror_memory.api.MemoryEngine.get_beliefs": lambda: eng.get_beliefs(user_id=USER, purpose=PURPOSE),
        "mirror_memory.api.MemoryEngine.get_verification_candidates": lambda: eng.get_verification_candidates(
            user_id=USER, purpose=PURPOSE),
        "mirror_memory.api.MemoryEngine.belief_history": lambda: eng.belief_history(
            user_id=USER, belief_id=belief_id, purpose=PURPOSE),
        "mirror_memory.api.MemoryEngine.explain": lambda: eng.explain(
            user_id=USER, belief_id=belief_id, purpose=PURPOSE),
    }


def _server_drivers(client: TestClient, belief_id: int) -> dict[str, object]:
    h = {"Authorization": "Bearer key-a", "X-MM-Purpose": PURPOSE}
    return {
        "mirror_memory.server.recall": lambda: client.post(
            "/recall", json={"user_id": USER, "query": "claim"}, headers=h),
        "mirror_memory.server.panel": lambda: client.get("/panel", params={"user_id": USER}, headers=h),
        "mirror_memory.server.beliefs": lambda: client.get("/beliefs", params={"user_id": USER}, headers=h),
        "mirror_memory.server.belief_by_id": lambda: client.get(
            f"/belief/{belief_id}", params={"user_id": USER}, headers=h),
        "mirror_memory.server.stats": lambda: client.get("/stats", params={"user_id": USER}, headers=h),
    }


def test_every_memory_reading_entry_in_the_registry_is_audited(db):
    """Driven by ENTRY_REGISTRY: the set of entries flagged reads_memory
    must equal the set of drivers below (a new flagged entry without a
    driver fails), and each driver must leave exactly one allowed row
    carrying that entry's registry key."""
    _seed(db)
    belief_id = _belief_ids(db)[0]
    eng = _engine(db)
    from mirror_memory import server as server_mod

    server_mod._tenant_key_registry.clear()
    register_tenant_key("key-a", *TENANT, purposes={PURPOSE})
    app = create_app(config=load_config("config/"), database_url=str(db.url))
    try:
        with TestClient(app) as client:
            drivers = {**_library_drivers(eng, belief_id), **_server_drivers(client, belief_id)}
            flagged = {k for k, v in ENTRY_REGISTRY.items() if v.get("reads_memory")}
            assert set(drivers) == flagged, (
                "memory-reading entries without an audit driver: "
                f"{sorted(flagged - set(drivers))}; drivers for unflagged entries: "
                f"{sorted(set(drivers) - flagged)}"
            )
            for key, run in drivers.items():
                before = len(_rows(db))
                result = run()
                if hasattr(result, "status_code"):
                    assert result.status_code == 200, (key, result.status_code, result.text)
                rows = _rows(db)
                assert len(rows) == before + 1, (key, [r.entry for r in rows[before:]])
                row = rows[-1]
                assert row.entry == key
                assert row.decision == "allowed"
                assert row.purpose == PURPOSE
                assert row.user_id == USER
                assert isinstance(row.grant_version, int)
                assert row.operation == ENTRY_REGISTRY[key]["operation"]
                ids = json.loads(row.memory_ids_json)
                assert row.memory_count == len(ids), key
                assert not row.truncated
    finally:
        server_mod._tenant_key_registry.clear()


def test_rows_carry_the_ids_the_entry_returned(db):
    _seed(db, beliefs=3)
    ids = _belief_ids(db)
    eng = _engine(db)
    beliefs = eng.get_beliefs(user_id=USER, purpose=PURPOSE)
    assert len(beliefs) == 3
    row = _rows(db)[-1]
    assert sorted(json.loads(row.memory_ids_json)) == ids and row.memory_count == 3
    eng.explain(user_id=USER, belief_id=ids[1], purpose=PURPOSE)
    assert json.loads(_rows(db)[-1].memory_ids_json) == [ids[1]]
    # a miss is an access attempt too: recorded with no ids
    assert eng.explain(user_id=USER, belief_id=999_999, purpose=PURPOSE) is None
    row = _rows(db)[-1]
    assert (row.memory_count, json.loads(row.memory_ids_json)) == (0, [])


def test_recall_records_exactly_the_beliefs_that_reached_the_prompt(db):
    _seed(db, beliefs=3)
    eng = _engine(db)
    block = eng.recall(user_id=USER, query="CLAIM", purpose=PURPOSE)
    row = _rows(db)[-1]
    ids = json.loads(row.memory_ids_json)
    if block:
        assert ids and set(ids) <= set(_belief_ids(db))
    else:
        assert ids == []
    assert row.memory_count == len(ids)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_refused_read_writes_a_committed_denied_row(db):
    _seed(db)
    eng = _engine(db)
    with refusing_authz(), pytest.raises(PurposeDenied):
        eng.recall(user_id=USER, query="x", purpose="never-granted")
    rows = _rows(db)
    assert [(r.decision, r.reason, r.purpose, r.entry) for r in rows] == [
        ("denied", "no_grant", "never-granted", "mirror_memory.api.MemoryEngine.recall")
    ]
    assert rows[0].grant_version is None
    assert (rows[0].memory_count, rows[0].memory_ids_json) == (0, "[]")


def test_revoked_and_wrong_operation_refusals_are_audited_with_their_reason(db):
    _seed(db)
    eng = _engine(db)
    eng.grant_purpose(user_id=USER, purpose="narrow", operations=["observe"], issuer="t")
    with pytest.raises(PurposeDenied):
        eng.get_beliefs(user_id=USER, purpose="narrow")
    eng.grant_purpose(user_id=USER, purpose="gone", operations=["explain"], issuer="t")
    eng.revoke_purpose(user_id=USER, purpose="gone", actor="t", reason="test")
    with pytest.raises(PurposeDenied):
        eng.get_beliefs(user_id=USER, purpose="gone")
    reasons = [(r.purpose, r.decision, r.reason) for r in _rows(db)]
    assert reasons == [("narrow", "denied", "operation_not_allowed"), ("gone", "denied", "revoked")]


def test_missing_purpose_is_refused_in_python_and_not_audited(db):
    """The batch-1 invariant stands: no purpose -> refused before any SQL,
    so there is nothing to attribute and no row."""
    _seed(db)
    eng = _engine(db, default_purpose=None)
    statements: list[str] = []
    eng._ensure_db()
    event.listen(eng._engine, "before_cursor_execute",
                 lambda conn, cursor, statement, parameters, context, executemany:
                 statements.append(statement))
    with pytest.raises(PurposeDenied, match="missing_purpose"):
        eng.recall(user_id=USER, query="x")
    assert statements == []
    assert _rows(db) == []


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


def test_audit_write_failure_fails_the_read_and_returns_nothing(db):
    """The table is gone: the INSERT fails, the entry raises instead of
    returning the already-rendered memory."""
    _seed(db)
    eng = _engine(db)
    assert eng.get_beliefs(user_id=USER, purpose=PURPOSE)
    with db.begin() as conn:
        conn.exec_driver_sql("DROP TABLE mm_access_audit")
    with pytest.raises(Exception) as info:
        eng.get_beliefs(user_id=USER, purpose=PURPOSE)
    assert "mm_access_audit" in str(info.value)
    with pytest.raises(Exception):
        eng.recall(user_id=USER, query="claim", purpose=PURPOSE)


def test_audit_recorder_failure_fails_the_read(db, monkeypatch):
    _seed(db)
    eng = _engine(db)

    def boom(*a, **kw):
        raise RuntimeError("audit sink down")

    monkeypatch.setattr(access_audit, "record_access", boom)
    with pytest.raises(RuntimeError, match="audit sink down"):
        eng.get_beliefs(user_id=USER, purpose=PURPOSE)
    with pytest.raises(RuntimeError):
        eng.panel(user_id=USER, purpose=PURPOSE)
    monkeypatch.undo()
    assert _rows(db) == []


def test_verification_candidates_audit_is_not_fail_open(db, monkeypatch):
    """The candidate selection swallows its own errors (empty list); the
    audit row is written anyway, and if IT fails the entry fails."""
    _seed(db)
    eng = _engine(db)
    from mirror_memory.extraction import verification

    monkeypatch.setattr(verification, "get_question_candidates",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("selection broke")))
    assert eng.get_verification_candidates(user_id=USER, purpose=PURPOSE) == []
    row = _rows(db)[-1]
    assert (row.entry, row.memory_count) == (
        "mirror_memory.api.MemoryEngine.get_verification_candidates", 0)


# ---------------------------------------------------------------------------
# Cap, deletion, retention
# ---------------------------------------------------------------------------


def test_ids_beyond_the_cap_are_cut_and_the_row_marked_truncated(db):
    _seed(db, beliefs=5)
    cfg = load_config("config/")
    cfg.access_audit.max_ids = 2
    eng = _engine(db, config=cfg)
    assert len(eng.get_beliefs(user_id=USER, purpose=PURPOSE)) == 5
    row = _rows(db)[-1]
    assert row.memory_count == 5
    assert len(json.loads(row.memory_ids_json)) == 2
    assert row.truncated is True


def test_record_access_validates_its_inputs(db):
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            with pytest.raises(ValueError):
                access_audit.record_access(s, user_id="u", purpose="p", operation="recall",
                                           entry="e", decision="maybe", max_ids=10)
            with pytest.raises(ValueError):
                access_audit.record_access(s, user_id="u", purpose="p", operation="recall",
                                           entry="e", decision="allowed", max_ids=0)
            with pytest.raises(ValueError):
                access_audit.record_access(s, user_id="u", purpose="p", operation="recall",
                                           entry="e", decision="allowed",
                                           max_ids=access_audit.MAX_IDS_HARD_CAP + 1)
        finally:
            s.rollback()
            s.close()


def test_delete_memories_keeps_the_audit_rows(db):
    _seed(db)
    eng = _engine(db)
    eng.get_beliefs(user_id=USER, purpose=PURPOSE)
    eng.recall(user_id=USER, query="claim", purpose=PURPOSE)
    assert len(_rows(db)) == 2
    eng.delete_memories(user_id=USER, purpose=PURPOSE)
    assert _belief_ids(db) == []
    assert len(_rows(db)) == 2, "ids only, no content: the audit is not memory"


def _backdate(db, row_id: int, days: int, tenant=TENANT) -> None:
    with db.begin() as conn:
        conn.execute(text(
            "UPDATE mm_access_audit SET created_at = :t WHERE id = :id "
            "AND tenant_id = :tenant AND app_id = :app"
        ), {"t": datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days),
            "id": row_id, "tenant": tenant[0], "app": tenant[1]})


def test_purge_is_explicit_retention_bound_and_tenant_scoped(db):
    _seed(db)
    _seed(db, tenant=("tB", "appB"))
    eng = _engine(db)
    for _ in range(3):
        eng.get_beliefs(user_id=USER, purpose=PURPOSE)
    other = MemoryEngine(config=load_config("config/"), database_url=str(db.url),
                         tenant_id="tB", app_id="appB")
    other.get_beliefs(user_id=USER, purpose=PURPOSE)
    mine = _rows(db)
    theirs = _rows(db, ("tB", "appB"))
    assert (len(mine), len(theirs)) == (3, 1)
    # two of mine and the other tenant's row are 100 days old (retention 90)
    _backdate(db, mine[0].id, 100)
    _backdate(db, mine[1].id, 100)
    _backdate(db, theirs[0].id, 100, tenant=("tB", "appB"))
    # nothing is purged by reading
    eng.get_beliefs(user_id=USER, purpose=PURPOSE)
    assert len(_rows(db)) == 4
    removed = eng.purge_access_audit()
    assert removed == 2
    assert len(_rows(db)) == 2
    assert len(_rows(db, ("tB", "appB"))) == 1, "the other tenant's rows are not mine to purge"
    # an explicit instant works too, and the purge is idempotent
    assert eng.purge_access_audit(older_than=datetime.now(UTC) - timedelta(days=100)) == 0


def test_retention_cutoff_and_default():
    cfg = load_config("config/")
    assert cfg.access_audit.retention_days == 90
    now = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
    assert access_audit.retention_cutoff(90, now=now) == datetime(2026, 7, 30, 12, 0)
    with pytest.raises(ValueError):
        access_audit.retention_cutoff(0)


# ---------------------------------------------------------------------------
# Cost: statements per read, and the SQLite write lock
# ---------------------------------------------------------------------------


def test_each_allowed_read_adds_exactly_one_audit_insert(db):
    """The audit's product cost: one INSERT per read entry (the commit on
    the formerly read-only paths is the transaction's, not a statement)."""
    _seed(db)
    eng = _engine(db)
    eng._ensure_db()
    belief_id = _belief_ids(db)[0]
    statements: list[str] = []
    event.listen(eng._engine, "before_cursor_execute",
                 lambda conn, cursor, statement, parameters, context, executemany:
                 statements.append(" ".join(statement.split())))
    for key, run in _library_drivers(eng, belief_id).items():
        statements.clear()
        run()
        audit = [s for s in statements if "mm_access_audit" in s]
        assert len(audit) == 1 and audit[0].upper().startswith("INSERT"), (key, audit)


def test_sqlite_concurrent_reads_complete_under_the_write_lock(tmp_path):
    """On SQLite every audited read is now a write transaction: concurrent
    readers serialise on the file's write lock instead of running in
    parallel.  With the default busy timeout they all complete; this pins
    that no reader fails with 'database is locked' and records the cost."""
    url = f"sqlite:///{tmp_path}/audit-lock.db"
    eng0 = create_engine(url)
    Base.metadata.create_all(eng0)
    eng0.dispose()
    db = create_engine(url)
    try:
        _seed(db, beliefs=3)
    finally:
        db.dispose()
    readers, per_reader = 8, 10
    errors: list[BaseException] = []
    barrier = threading.Barrier(readers + 2, timeout=30)  # readers + writer + main

    def read_loop():
        eng = MemoryEngine(config=load_config("config/"), database_url=url,
                           tenant_id=TENANT[0], app_id=TENANT[1])
        barrier.wait()
        try:
            for _ in range(per_reader):
                eng.recall(user_id=USER, query="claim", purpose=PURPOSE)
        except BaseException as exc:  # noqa: BLE001 -- recorded, asserted below
            errors.append(exc)

    def write_loop():
        eng = MemoryEngine(config=load_config("config/"), database_url=url,
                           tenant_id=TENANT[0], app_id=TENANT[1])
        barrier.wait()
        try:
            for i in range(per_reader):
                eng.observe(user_id=USER, session_id="s-w", text=f"I like item {i}.",
                            turn_count=i, purpose=PURPOSE)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=read_loop) for _ in range(readers)]
    threads.append(threading.Thread(target=write_loop))
    for t in threads:
        t.start()
    started = time.monotonic()
    barrier.wait()
    for t in threads:
        t.join(timeout=120)
    elapsed = time.monotonic() - started
    assert not errors, [repr(e) for e in errors]
    check = create_engine(url)
    try:
        rows = _rows(check)
    finally:
        check.dispose()
    recalls = [r for r in rows if r.entry.endswith(".recall")]
    assert len(recalls) == readers * per_reader
    print(f"\nsqlite write-lock cost: {readers} readers x {per_reader} recalls + "
          f"{per_reader} observes in {elapsed:.2f}s "
          f"({elapsed / (readers * per_reader) * 1000:.1f} ms per recall wall-clock)")
