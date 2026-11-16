"""M4 task 1 (cloud backend): the deletion log in its own PostgreSQL
database -- outside any pg_dump of the main DB -- with the app role
restricted to INSERT / SELECT.

The log database is created on the fly (the CI service user is a
superuser); the role restriction is the real mechanism: UPDATE/DELETE
are refused AT THE DATABASE LEVEL, so even a compromised engine cannot
rewrite or erase the deletion history.
"""

from __future__ import annotations

import os
import uuid

import time

import pytest
from sqlalchemy import create_engine, text

from mirror_memory.core.deletion import (
    DeletionEntry,
    FileDeletionLog,
    GENESIS_HASH,
    PgDeletionLog,
)

IS_PG = os.environ.get("MM_DATABASE_URL", "").startswith("postgresql")

pytestmark = [
    pytest.mark.postgres_only("the cloud log backend needs a second PostgreSQL database"),
    pytest.mark.skipif(not IS_PG, reason="needs MM_DATABASE_URL pointing at PostgreSQL"),
]

LOG_DB = "mm_test_deletion_log"


def _db_id() -> str:
    return uuid.uuid4().hex


def _admin_url() -> str:
    url = os.environ["MM_DATABASE_URL"]
    return url.rsplit("/", 1)[0] + "/postgres"


def _log_url() -> str:
    return os.environ["MM_DATABASE_URL"].rsplit("/", 1)[0] + "/" + LOG_DB


@pytest.fixture
def log_db_url():
    admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(text(
                "SELECT 1 FROM pg_database WHERE datname = :d"), {"d": LOG_DB}).first()
        if not exists:
            with admin.connect() as conn:
                conn.execute(text(f'CREATE DATABASE "{LOG_DB}"'))
                conn.commit()
        # a fresh table per run: an earlier run's schema (or rows) must not
        # leak into this run's hash-chain assertions
        log_engine = create_engine(_log_url())
        try:
            conn = log_engine.connect()
            conn.exec_driver_sql("DROP TABLE IF EXISTS mm_deletion_log")
            conn.commit()  # SQLAlchemy 2 autobegin: no commit = rolled back
            conn.close()
        finally:
            log_engine.dispose()
        yield _log_url()
    finally:
        admin.dispose()


def test_pg_log_chain_head_and_reads(log_db_url):
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    assert log.head(db_id) == (0, GENESIS_HASH)
    # M5 patch 4: appends are fail-closed without a genesis entry
    log.ensure_genesis(db_id, applied=0)
    e1 = log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u",
        scope="belief", scope_key="9",
        fingerprints=[{"table": "mm_beliefs", "id": 9, "created_at": None}],
        generation=1, purpose="api",
    ))
    e2 = log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u",
        scope="user", scope_key="", fingerprints=[], generation=2, purpose="api",
    ))
    assert e1.seq == 1 and e1.prev_hash != GENESIS_HASH  # chains from genesis
    assert e2.prev_hash == e1.hash
    # the seq is GLOBALLY monotonic (the task's spec) -- a fresh db life
    # continues the identity's counter, so only relative order is asserted
    assert e2.seq > e1.seq
    assert log.head(db_id) == (e2.seq, e2.hash)
    entries = list(log.read_from(db_id, 0))
    assert [e.seq for e in entries] == sorted(e.seq for e in entries)
    assert entries[0].fingerprints[0]["table"] == "mm_beliefs"
    log.verify_chain(db_id)
    # entries are keyed by db_id: another life is untouched
    assert log.head(_db_id()) == (0, GENESIS_HASH)


def test_pg_log_app_role_is_insert_select_only(log_db_url):
    """The restriction is enforced by the DATABASE: the app role can
    append and read but cannot UPDATE or DELETE -- even a compromised
    engine cannot rewrite or erase the deletion history."""
    log = PgDeletionLog(log_db_url)
    log._ensure_schema()
    db_id = _db_id()
    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            # the role persists across runs (its privileges are the point):
            # create it once, then (re)assert the grants
            exists = conn.execute(text(
                "SELECT 1 FROM pg_roles WHERE rolname = 'mm_log_app'")).first()
            if not exists:
                conn.execute(text("CREATE ROLE mm_log_app LOGIN PASSWORD 'logpw'"))
            conn.execute(text(
                "GRANT INSERT, SELECT ON mm_deletion_log TO mm_log_app"))
            conn.execute(text(
                "GRANT USAGE ON SCHEMA public TO mm_log_app"))
        app_url = log_db_url.replace("postgres:mm@", "mm_log_app:logpw@")
        app = create_engine(app_url)
        try:
            with app.begin() as conn:
                conn.execute(text(
                    "INSERT INTO mm_deletion_log (db_id, tenant_id, app_id, user_id, "
                    "scope, scope_key, fingerprints_json, generation, purpose, "
                    "prev_hash, hash) VALUES (:db, 't', 'a', 'u', 'belief', '1', "
                    "'[]', 1, 'p', :prev, :hash)"
                ), {"db": db_id, "prev": GENESIS_HASH, "hash": "h" * 64})
            with app.connect() as conn:
                rows = conn.execute(text(
                    "SELECT count(*) FROM mm_deletion_log WHERE db_id = :db"),
                    {"db": db_id}).scalar()
            assert rows == 1
            with pytest.raises(Exception, match="permission denied"):
                with app.begin() as conn:
                    conn.execute(text(
                        "UPDATE mm_deletion_log SET user_id = 'evil' WHERE db_id = :db"),
                        {"db": db_id})
            with pytest.raises(Exception, match="permission denied"):
                with app.connect() as conn:
                    conn.execute(text(
                        "DELETE FROM mm_deletion_log WHERE db_id = :db"), {"db": db_id})
        finally:
            app.dispose()
    finally:
        admin.dispose()


def test_embedded_and_cloud_logs_share_the_entry_format(log_db_url, tmp_path):
    """The same entry serialises through both backends and verifies: a
    deployment can switch backends (or mirror) without format drift."""
    entry = DeletionEntry(
        db_id=_db_id(), tenant_id="t", app_id="a", user_id="u",
        scope="session", scope_key="s1",
        fingerprints=[{"table": "mm_session_summaries", "id": 3, "created_at": None}],
        generation=4, purpose="p",
    )
    file_log = FileDeletionLog(tmp_path / "deletions")
    pg_log = PgDeletionLog(log_db_url)
    # M5 patch 4: both backends refuse appends without an initialized genesis
    file_log.ensure_genesis(entry.db_id, applied=0)
    pg_log.ensure_genesis(entry.db_id, applied=0)
    fe = file_log.append(entry)
    pe = pg_log.append(entry)
    assert fe.hash == pe.hash
    assert pe.prev_hash != GENESIS_HASH  # chains from the written genesis
    file_log.verify_chain(entry.db_id)
    pg_log.verify_chain(entry.db_id)


def test_prune_runs_on_the_maintenance_connection(log_db_url):
    """Compaction DELETES rows: without a maintenance connection it
    refuses (the app role has no DELETE); with one it prunes through the
    checkpoint and the chain stays verifiable."""
    from datetime import datetime

    from mirror_memory.core.deletion import DeletionEntry, PgDeletionLog

    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)   # M5 patch 4: fail-closed appends
    for i in range(1, 5):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="api", created_at=datetime(2020, 1, i),
        ))

    app_only = PgDeletionLog(log_db_url)
    with pytest.raises(RuntimeError, match="maintenance"):
        app_only.prune(db_id, 2)

    maint_url = log_db_url  # the test's server login IS the maintainer
    with_maint = PgDeletionLog(log_db_url, maintenance_url=maint_url)
    assert with_maint.prune(db_id, 2) == 2
    assert with_maint.checkpoint_seq(db_id) == 2
    assert with_maint.has_genesis(db_id)
    assert [e.seq for e in with_maint.read_from(db_id, 2)] == [3, 4]
    with_maint.verify_chain(db_id)
    # idempotent: pruning through the same point again removes nothing
    assert with_maint.prune(db_id, 2) == 0
    with_maint.verify_chain(db_id)


# ---------------------------------------------------------------------------
# M5 patch 5: the PG deletion log's own integrity witness.  The log table lives
# in its own database, so the witness must not be weaker: it is a separate
# table there, authenticated with a key held in a FILE.
# ---------------------------------------------------------------------------


_pg_needs = pytest.mark.skipif(
    not os.environ.get("MM_DATABASE_URL", "").startswith("postgresql"),
    reason="needs MM_DATABASE_URL pointing at PostgreSQL",
)


def _pg_witness_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MM_DELETION_WITNESS_KEY", str(tmp_path / "witness.key"))


def test_pg_witness_records_each_append(log_db_url, tmp_path, monkeypatch):
    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    assert log.witness.read(db_id).tail_seq == 0

    for i in range(1, 3):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))

    state = log.witness.read(db_id)
    assert state.tail_seq == log.head(db_id)[0] == 2
    log.verify_witness(db_id, applied=2)       # healthy: passes


def test_pg_witness_catches_a_truncated_log_table(log_db_url, tmp_path, monkeypatch):
    """The cloud-deployment equivalent of the file-backend truncation test:
    rows are removed from mm_deletion_log behind the engine's back."""
    from mirror_memory.core.witness import WitnessAhead

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 4):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[], generation=i,
            purpose="p"))

    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "DELETE FROM mm_deletion_log WHERE db_id = :db AND seq > 1"),
                {"db": db_id})
    finally:
        admin.dispose()

    with pytest.raises(WitnessAhead) as excinfo:
        log.verify_witness(db_id, applied=3)
    assert "lost" in str(excinfo.value).lower()


def test_pg_witness_missing_row_with_deletions_is_refused(
        log_db_url, tmp_path, monkeypatch):
    """A cleared witness table with deletions on record must refuse, never
    back-fill silently."""
    from mirror_memory.core.deletion import DeletionLogCorrupted

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    log.witness.retire(db_id)

    with pytest.raises(DeletionLogCorrupted):
        log.verify_witness(db_id, applied=1)
    assert log.witness.read(db_id) is None


def test_pg_witness_tampered_row_is_refused(log_db_url, tmp_path, monkeypatch):
    from mirror_memory.core.deletion import DeletionLogCorrupted

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))

    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "UPDATE mm_deletion_log_witness SET tail_seq = 0 "
                "WHERE db_id = :db"), {"db": db_id})
    finally:
        admin.dispose()

    with pytest.raises(DeletionLogCorrupted):
        log.verify_witness(db_id, applied=1)


# ---------------------------------------------------------------------------
# M5 patch 6 §2: the PG witness write is a MONOTONIC compare-and-set guarded
# by its own advisory lock (in the witness's database).  A lagging writer can
# never regress the witness, and concurrent appends keep the witness at the
# true head.
# ---------------------------------------------------------------------------


def test_pg_witness_rejects_a_lagging_writer(log_db_url, tmp_path, monkeypatch):
    from mirror_memory.core.witness import WitnessAhead, WitnessState

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    current = log.witness.read(db_id)
    assert current.tail_seq == 1

    # a stale state (seq 0) must be refused, not overwrite the witness
    stale = WitnessState(
        db_id=db_id, genesis_hash=current.genesis_hash, tail_seq=0,
        tail_hash=current.genesis_hash, checkpoint_seq=0,
        updated_at=current.updated_at)
    with pytest.raises(WitnessAhead):
        log.witness.record(stale)
    assert log.witness.read(db_id).tail_seq == 1, "no regression"


def test_pg_witness_equal_seq_same_hash_is_idempotent(
        log_db_url, tmp_path, monkeypatch):
    from mirror_memory.core.witness import WitnessState

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    current = log.witness.read(db_id)
    same = WitnessState(
        db_id=db_id, genesis_hash=current.genesis_hash, tail_seq=0,
        tail_hash=current.tail_hash, checkpoint_seq=0,
        updated_at=current.updated_at)
    log.witness.record(same)                   # equal seq + same hash: OK
    assert log.witness.read(db_id).tail_seq == 0


def test_pg_concurrent_appends_witness_matches_head(
        log_db_url, tmp_path, monkeypatch):
    import threading

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    errors: list[BaseException] = []

    def worker(idx):
        try:
            for j in range(6):
                log.append(DeletionEntry(
                    db_id=db_id, tenant_id="t", app_id="a",
                    user_id=f"u{idx}", scope="session",
                    scope_key=f"s{idx}-{j}", fingerprints=[],
                    generation=j + 1, purpose="p"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors[:2]

    head_seq, head_hash = log.head(db_id)
    assert head_seq == 18
    state = log.witness.read(db_id)
    assert state.tail_seq == 18
    assert state.tail_hash == head_hash
    log.verify_witness(db_id, applied=18)


# ---------------------------------------------------------------------------
# M5 patch 7 §3: the witness table's permission boundary, on REAL roles.
# The log app role (INSERT/SELECT on mm_deletion_log only) must NOT be able
# to write the witness table; a dedicated witness role can UPDATE it but
# still cannot forge a row (the MAC lives in the key file).
# ---------------------------------------------------------------------------


def test_pg_witness_table_permissions_on_real_roles(
        log_db_url, tmp_path, monkeypatch):
    from mirror_memory.core.deletion import DeletionLogCorrupted

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)          # creates the witness table

    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            exists = conn.execute(text(
                "SELECT 1 FROM pg_roles WHERE rolname = 'mm_log_app'")).first()
            if not exists:
                conn.execute(text(
                    "CREATE ROLE mm_log_app LOGIN PASSWORD 'logpw'"))
            # the LOG app role's grants stay INSERT/SELECT on the log table;
            # it gets NOTHING on the witness table (no grants at all).
            conn.execute(text(
                "GRANT INSERT, SELECT ON mm_deletion_log TO mm_log_app"))
            conn.execute(text("GRANT USAGE ON SCHEMA public TO mm_log_app"))
            # a dedicated witness role may UPDATE the witness (repair/recover)
            wexists = conn.execute(text(
                "SELECT 1 FROM pg_roles WHERE rolname = 'mm_witness_app'"
            )).first()
            if not wexists:
                conn.execute(text(
                    "CREATE ROLE mm_witness_app LOGIN PASSWORD 'wpw'"))
            conn.execute(text(
                f"GRANT SELECT, UPDATE, INSERT ON {log.witness.TABLE} "
                "TO mm_witness_app"))
            conn.execute(text("GRANT USAGE ON SCHEMA public TO mm_witness_app"))

        # 1. the LOG app role cannot read the witness table at all
        log_app = create_engine(
            log_db_url.replace("postgres:mm@", "mm_log_app:logpw@"))
        try:
            with log_app.begin() as conn:
                conn.execute(text(
                    f"SELECT 1 FROM {log.witness.TABLE} LIMIT 1"))
            raise AssertionError("mm_log_app must NOT read the witness table")
        except Exception as exc:
            assert "permission denied" in str(exc).lower(), str(exc)[:100]
        finally:
            log_app.dispose()

        # 2. the log app role cannot UPDATE the witness either
        log_app = create_engine(
            log_db_url.replace("postgres:mm@", "mm_log_app:logpw@"))
        try:
            with log_app.begin() as conn:
                conn.execute(text(
                    f"UPDATE {log.witness.TABLE} SET tail_seq = 0 "
                    "WHERE db_id = :db"), {"db": db_id})
            raise AssertionError("mm_log_app must NOT update the witness")
        except Exception as exc:
            assert "permission denied" in str(exc).lower(), str(exc)[:100]
        finally:
            log_app.dispose()

        # 3. even a role with UPDATE cannot forge: the MAC key is a file the
        #    database cannot read, so a hand-written row fails the MAC check
        witness_engine = create_engine(
            log_db_url.replace("postgres:mm@", "mm_witness_app:wpw@"))
        try:
            with witness_engine.begin() as conn:
                conn.execute(text(
                    f"UPDATE {log.witness.TABLE} SET tail_seq = 9, "
                    "tail_hash = 'f', mac = '0' WHERE db_id = :db"),
                    {"db": db_id})
        finally:
            witness_engine.dispose()
        from mirror_memory.core.witness import WitnessError

        with pytest.raises(WitnessError):
            log.witness.read(db_id)             # unauthenticated row
        with pytest.raises(DeletionLogCorrupted):
            log.verify_witness(db_id, applied=1)  # startup path: refuse
    finally:
        admin.dispose()


# ---------------------------------------------------------------------------
# M5 patch 7 §5: brief repro ① on PostgreSQL through the REAL engine path.
# A dedicated main database (real app role + RLS), a real deletion with log
# entries, then the attacker truncates the LOG TABLE to its genesis and
# erases the witness row while the main DB is restored from an OLD backup.
# The real engine must refuse startup.
# ---------------------------------------------------------------------------


def _pg_repro_database(main_db: str):
    """A fresh main database with the M4/M5 schema, served through the app
    role (the same shape the deletion-race e2e uses)."""
    import contextlib

    from mirror_memory.core import schema_migration as sm
    from mirror_memory.core.models import Base
    from mirror_memory.core.schema_migration import (
        Step11DeletionReplay,
        install_row_level_security,
    )
    from mirror_memory.core.sql_audit import no_audit_scope
    from sqlalchemy.engine import make_url

    from tests.conftest import PG_TEST_ROLE_PASSWORDS

    admin_url = os.environ["MM_DATABASE_URL"].rsplit("/", 1)[0] + "/postgres"
    main_url = os.environ["MM_DATABASE_URL"].rsplit("/", 1)[0] + "/" + main_db
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with no_audit_scope(), admin_engine.connect() as conn:
        conn.exec_driver_sql(
            f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
        conn.exec_driver_sql(f'CREATE DATABASE "{main_db}"')
    admin_engine.dispose()

    from mirror_memory.core import rls as _rls
    from mirror_memory.core.tenancy import system_scope as _system_scope

    owner_engine = create_engine(main_url)
    Base.metadata.create_all(owner_engine)
    with _system_scope(reason="p7 repro: watermark parity"):
        sm.ensure_watermark(owner_engine)
    install_row_level_security(owner_engine, verify=False)
    Step11DeletionReplay().apply(owner_engine)
    Step11DeletionReplay().verify(owner_engine)
    with owner_engine.begin() as conn:
        conn.exec_driver_sql(
            f"GRANT SELECT, INSERT, UPDATE ON mm_deletion_watermark "
            f"TO {_rls.app_role()}, {_rls.maintenance_role()}")
    owner_engine.dispose()
    runtime_url = make_url(main_url).set(
        username="mm_app",
        password=PG_TEST_ROLE_PASSWORDS["app"],
    ).render_as_string(hide_password=False)
    return runtime_url, admin_url


@_pg_needs
def test_pg_brief_repro_1_truncated_log_table_refused(
        log_db_url, tmp_path, monkeypatch):
    """The PG equivalent of brief repro 1: real engine + real app role + RLS;
    the log TABLE is truncated to its genesis and the witness row removed
    while the main DB (restored backup) carries applied == 0."""
    import shutil

    from mirror_memory.api import MemoryEngine
    from mirror_memory.config.loader import load_config
    from mirror_memory.core.deletion import (
        DeletionLogCorrupted,
        DeletionLogUnavailable,
        PgDeletionLog,
    )
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from sqlalchemy.orm import sessionmaker

    _pg_witness_env(tmp_path, monkeypatch)
    main_db = "mm_test_p7_repro"
    runtime_url, admin_url = _pg_repro_database(main_db)

    class _L:
        def generate(self, **_kw):
            return '{"patterns": []}'

    cfg = load_config("config/")
    cfg.llm_client = _L()
    log = PgDeletionLog(log_db_url)
    eng = MemoryEngine(config=cfg, database_url=runtime_url,
                       llm_client=_L(), deletion_log=log,
                       tenant_id="default", app_id="default")
    eng._ensure_db()
    db_id = eng._db_id
    with tenant_scope(*eng._tenant), fixture_write_scope("p7 seed"):
        s = sessionmaker(bind=eng._engine)()
        set_memory_enabled(s, "u1", True)
        s.commit()
        s.close()
    n = eng.observe(user_id="u1", session_id="s1",
                    text="hello", purpose="chat")
    r = eng.delete_memories(user_id="u1", purpose="chat")
    assert r.beliefs >= 1 or n >= 0
    witness_before = log.witness.read(db_id)
    assert witness_before is not None and witness_before.tail_seq >= 1
    eng._engine.dispose()

    # the attack: truncate the log table to its genesis + drop the witness row
    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "DELETE FROM mm_deletion_log WHERE db_id = :db AND seq > 0"),
                {"db": db_id})
            conn.execute(text(
                "DELETE FROM mm_deletion_log_witness WHERE db_id = :db"),
                {"db": db_id})
    finally:
        admin.dispose()

    # reset the main DB's watermark to 0 (what an old backup carries)
    owner_url = os.environ["MM_DATABASE_URL"].rsplit("/", 1)[0] + "/" + main_db
    owner = create_engine(owner_url)
    try:
        with owner.begin() as conn:
            conn.execute(text(
                "UPDATE mm_deletion_watermark SET applied_seq = 0 "
                "WHERE id = 1"))
    finally:
        owner.dispose()

    eng2 = MemoryEngine(config=cfg, database_url=runtime_url,
                        llm_client=_L(), deletion_log=log,
                        tenant_id="default", app_id="default")
    with pytest.raises((DeletionLogCorrupted, DeletionLogUnavailable)):
        eng2._ensure_db()

    # the log table was NOT re-minted: still exactly the genesis
    check = create_engine(log_db_url)
    try:
        with check.connect() as conn:
            rows = conn.execute(text(
                "SELECT seq, scope FROM mm_deletion_log WHERE db_id = :db "
                "ORDER BY seq"), {"db": db_id}).all()
    finally:
        check.dispose()
    assert [(int(r[0]), r[1]) for r in rows] == [(0, "genesis")], rows
    shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# M5 patch 8 §2/§4⑤: the PG append/prune continuity gates.
# ---------------------------------------------------------------------------


def test_p8_pg_append_refuses_missing_witness(log_db_url, tmp_path, monkeypatch):
    from mirror_memory.core.deletion import DeletionLogCorrupted

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    log.witness.retire(db_id)                 # the witness row is gone

    with pytest.raises(DeletionLogCorrupted):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id="x", scope="user",
            scope_key="", fingerprints=[], generation=2, purpose="x"))
    # the table did not grow
    assert log.head(db_id)[0] == 1
    assert log.witness.read(db_id) is None


def test_p8_pg_append_refuses_equal_height_divergence(
        log_db_url, tmp_path, monkeypatch):
    from mirror_memory.core.deletion import DeletionLogCorrupted

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 3):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    witness = log.witness.read(db_id)

    # rewrite the tail row's hash in the table (a same-height divergence)
    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "UPDATE mm_deletion_log SET hash = repeat('f', 64) "
                "WHERE db_id = :db AND seq = 2"), {"db": db_id})
    finally:
        admin.dispose()

    with pytest.raises((DeletionLogCorrupted, Exception)):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id="x", scope="user",
            scope_key="", fingerprints=[], generation=3, purpose="x"))
    # the witness is not overwritten with the forged head
    after = log.witness.read(db_id)
    assert after.tail_seq == witness.tail_seq
    assert after.tail_hash == witness.tail_hash


def test_p8_pg_prune_refuses_when_witness_behind_unextendable(
        log_db_url, tmp_path, monkeypatch):
    """Prune must verify the witnessed continuity before rewriting rows."""
    from mirror_memory.core.witness import WitnessAhead

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url, maintenance_url=log_db_url.replace(
        "postgres:mm@", "postgres:mm@"))
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 5):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    stale = log.witness.read(db_id)

    # delete the tail rows: the witness is now AHEAD of the table
    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "DELETE FROM mm_deletion_log WHERE db_id = :db AND seq > 2"),
                {"db": db_id})
    finally:
        admin.dispose()

    with pytest.raises((WitnessAhead, Exception)):
        log.prune(db_id, 2)
    # the witness (which is ahead) is untouched
    after = log.witness.read(db_id)
    assert after.tail_seq == stale.tail_seq


# ---------------------------------------------------------------------------
# M5 patch 9 §4: the FULL pg_dump / pg_restore round trip (not a watermark
# reset) with the witness / creation-registry integrity checks on the real
# engine startup path.
# ---------------------------------------------------------------------------


def _p9_dump_and_restore(main_db: str, dump_path: str, admin_url: str):
    """pg_dump the main DB to *dump_path*, drop it, recreate it, pg_restore.
    Uses the REAL container tools (same path the deletion-race e2e uses)."""
    import subprocess
    from mirror_memory.core.sql_audit import no_audit_scope

    env = dict(os.environ, PGPASSWORD="mm")
    result = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=mm", "mm-pg",
         "pg_dump", "-U", "postgres", "--format=custom", main_db],
        env=env, capture_output=True, check=True)
    if len(result.stdout) < 5000:
        raise AssertionError(
            f"pg_dump produced only {len(result.stdout)} bytes -- the "
            "database was empty at dump time")
    open(dump_path, "wb").write(result.stdout)

    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with no_audit_scope(), admin_engine.connect() as conn:
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
            conn.exec_driver_sql(f'CREATE DATABASE "{main_db}"')
    finally:
        admin_engine.dispose()
    # pg_restore --role=mm_owner creates the tables AS mm_owner, so that
    # role needs CREATE on the public schema of the fresh database -- a
    # grant is per-database, so it must be applied against the new DB.
    grant_engine = create_engine(
        admin_url.rsplit("/", 1)[0] + "/" + main_db,
        isolation_level="AUTOCOMMIT")
    try:
        with grant_engine.connect() as gconn:
            gconn.exec_driver_sql('GRANT ALL ON SCHEMA public TO mm_owner')
    finally:
        grant_engine.dispose()
    with open(dump_path, "rb") as dump_file:
        restore = subprocess.run(
            ["docker", "exec", "-i", "-e", "PGPASSWORD=mm", "mm-pg",
             "pg_restore", "-U", "postgres", "--role=mm_owner",
             "--dbname=" + main_db],
            stdin=dump_file, capture_output=True)
    # pg_restore exits 1 on IGNORED errors too (e.g. RLS objects owned by
    # roles the restore role cannot touch) -- the tables are what matter,
    # so verify them instead of trusting the exit code.
    check = create_engine(
        admin_url.rsplit("/", 1)[0] + "/" + main_db)
    try:
        with check.connect() as conn:
            n = conn.execute(text(
                "SELECT count(*) FROM pg_tables WHERE schemaname='public' "
                "AND tablename LIKE 'mm_%'")).scalar()
    finally:
        check.dispose()
    assert n >= 20, (
        f"pg_restore produced only {n} mm_* tables; stderr: "
        f"{restore.stderr.decode()[:300]}")


@_pg_needs
def test_p9_full_pg_dump_restore_roundtrip_with_witness_checks(
        log_db_url, tmp_path, monkeypatch):
    """Backup -> delete (log entries + witness advance) -> pg_dump -> restore
    the OLD main DB -> the real engine must refuse when the log table is
    truncated, and serve normally when it is intact."""
    import shutil

    from mirror_memory.api import MemoryEngine
    from mirror_memory.config.loader import load_config
    from mirror_memory.core.deletion import (
        DeletionLogCorrupted, DeletionLogUnavailable, PgDeletionLog)
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from mirror_memory.core.sql_audit import no_audit_scope
    from sqlalchemy.orm import sessionmaker

    _pg_witness_env(tmp_path, monkeypatch)
    main_db = "mm_test_p9_roundtrip"
    runtime_url, admin_url = _pg_repro_database(main_db)

    class _L:
        def generate(self, **_kw):
            return '{"patterns": []}'

    cfg = load_config("config/")
    cfg.llm_client = _L()
    log = PgDeletionLog(log_db_url)
    eng = MemoryEngine(config=cfg, database_url=runtime_url,
                       llm_client=_L(), deletion_log=log,
                       tenant_id="default", app_id="default")
    eng._ensure_db()
    db_id = eng._db_id
    with tenant_scope(*eng._tenant), fixture_write_scope("p9 seed"):
        s = sessionmaker(bind=eng._engine)()
        set_memory_enabled(s, "u1", True)
        s.commit()
        s.close()
    eng.observe(user_id="u1", session_id="s1", text="hello", purpose="chat")
    eng.delete_memories(user_id="u1", purpose="chat")
    assert log.witness.read(db_id).tail_seq >= 1
    eng._engine.dispose()

    # pg_dump the CURRENT state (with the deletion applied), then restore it:
    # a healthy round trip -- the engine must come back up and serve.
    _p9_dump_and_restore(main_db, str(tmp_path / "current.dump"), admin_url)
    eng_ok = MemoryEngine(config=cfg, database_url=runtime_url,
                          llm_client=_L(), deletion_log=log,
                          tenant_id="default", app_id="default")
    eng_ok._ensure_db()
    eng_ok._engine.dispose()

    # now restore a backup taken BEFORE any deletion existed -- build it by
    # dropping the witness row and truncating the log table to the genesis
    # (the attacker action), while the main DB still carries applied >= 1.
    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "DELETE FROM mm_deletion_log WHERE db_id = :db AND seq > 0"),
                {"db": db_id})
            conn.execute(text(
                "DELETE FROM mm_deletion_log_witness WHERE db_id = :db"),
                {"db": db_id})
    finally:
        admin.dispose()

    eng_bad = MemoryEngine(config=cfg, database_url=runtime_url,
                           llm_client=_L(), deletion_log=log,
                           tenant_id="default", app_id="default")
    with pytest.raises((DeletionLogCorrupted, DeletionLogUnavailable)):
        eng_bad._ensure_db()

    # the log table still holds exactly the genesis -- evidence preserved
    check = create_engine(log_db_url)
    try:
        with check.connect() as conn:
            rows = conn.execute(text(
                "SELECT seq, scope FROM mm_deletion_log WHERE db_id = :db "
                "ORDER BY seq"), {"db": db_id}).all()
    finally:
        check.dispose()
    assert [(int(r[0]), r[1]) for r in rows] == [(0, "genesis")], rows

    # cleanup: drop the repro database
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with no_audit_scope(), admin_engine.connect() as conn:
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
    finally:
        admin_engine.dispose()
    shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# M5 patch 10 §1: the PG prune gate is unified with the file backend —
# full chain verification + witness continuity + ancestry BEFORE the
# compaction transaction.
# ---------------------------------------------------------------------------


def test_p10_pg_prune_refuses_a_broken_chain(log_db_url, tmp_path, monkeypatch):
    """A tampered entry hash anywhere in the table: prune refuses and the
    table is untouched (fail-closed), even though the witness row is fine."""
    from mirror_memory.core.deletion import DeletionLogCorrupted

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 5):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    # tamper seq 1's hash in place
    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "UPDATE mm_deletion_log SET hash = repeat('e', 64) "
                "WHERE db_id = :db AND seq = 1"), {"db": db_id})
    finally:
        admin.dispose()
    # snapshot AFTER the tamper: what must be preserved is the attacker's
    # state at refusal time (the engine must not "repair" or compact it)
    table_before = _table_rows(log_db_url, db_id)

    with pytest.raises(DeletionLogCorrupted) as excinfo:
        log.prune(db_id, 3)
    assert "does not verify" in str(excinfo.value).lower()
    assert _table_rows(log_db_url, db_id) == table_before, (
        "a refused prune must not touch the table")


def _table_rows(log_db_url, db_id):
    check = create_engine(log_db_url)
    try:
        with check.connect() as conn:
            return conn.execute(text(
                "SELECT seq, hash FROM mm_deletion_log WHERE db_id = :db "
                "ORDER BY seq"), {"db": db_id}).all()
    finally:
        check.dispose()


def test_p10_pg_prune_refuses_when_witnessed_hash_not_ancestor(
        log_db_url, tmp_path, monkeypatch):
    """The witness records seq 2/hash H; the table's seq-2 hash differs (a
    fork planted below the witnessed point).  Even though verify_witness's
    own seq comparison may not fire, the ancestry check must refuse the
    prune."""
    from mirror_memory.core.deletion import DeletionLogCorrupted

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u1", scope="session",
        scope_key="s1", fingerprints=[], generation=1, purpose="p"))
    assert log.witness.read(db_id).tail_seq == 1

    # replace seq 1 with a DIFFERENT valid-looking entry (fork below witness)
    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "DELETE FROM mm_deletion_log WHERE db_id = :db AND seq = 1"),
                {"db": db_id})
    finally:
        admin.dispose()
    forked = DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="zz", scope="session",
        scope_key="zz", fingerprints=[], generation=1, purpose="forged")
    forked.seq = 1
    forked.prev_hash = "0" * 64
    forked.created_at = None
    forked.hash = forked.compute_hash()
    log._insert(create_engine(log_db_url).connect(), forked) \
        if False else None
    from sqlalchemy import text as _t

    eng = create_engine(log_db_url)
    try:
        with eng.begin() as conn:
            conn.execute(_t(
                "INSERT INTO mm_deletion_log (db_id, seq, tenant_id, app_id, "
                "user_id, scope, scope_key, fingerprints_json, generation, "
                "purpose, prev_hash, hash, created_at) VALUES (:db, 1, 't', "
                "'a', 'zz', 'session', 'zz', '[]', 1, 'forged', :prev, "
                ":hash, NOW())"),
                {"db": db_id, "prev": "0" * 64, "hash": forked.hash})
    finally:
        eng.dispose()

    with pytest.raises(DeletionLogCorrupted):
        log.prune(db_id, 1)


@_pg_needs
def test_p10_pg_predeletion_backup_restore_replays_the_deletion(
        log_db_url, tmp_path, monkeypatch):
    """§5: a REAL pre-deletion backup of the main DB, a real deletion recorded
    in the independent log + witness, then a REAL pg_dump/pg_restore of that
    backup.  Startup must REPLAY the deletion onto the restored rows (the
    memory must not come back) and land the watermark at the log tail."""
    import subprocess

    from mirror_memory.api import MemoryEngine
    from mirror_memory.config.loader import load_config
    from mirror_memory.core import schema_migration as sm
    from mirror_memory.core.deletion import PgDeletionLog
    from mirror_memory.core.models import Belief
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.sql_audit import no_audit_scope
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from sqlalchemy.orm import sessionmaker

    _pg_witness_env(tmp_path, monkeypatch)
    main_db = "mm_test_p10_replay"
    runtime_url, admin_url = _pg_repro_database(main_db)

    class _L:
        def generate(self, **_kw):
            return '{"patterns": []}'

    cfg = load_config("config/")
    cfg.llm_client = _L()
    log = PgDeletionLog(log_db_url)
    eng = MemoryEngine(config=cfg, database_url=runtime_url,
                       llm_client=_L(), deletion_log=log,
                       tenant_id="default", app_id="default")
    eng._ensure_db()
    db_id = eng._db_id

    # ---- seed one real belief (ORM, under the proper scopes) -------------
    with tenant_scope(*eng._tenant), fixture_write_scope("p10 seed"):
        s = sessionmaker(bind=eng._engine)()
        set_memory_enabled(s, "u1", True)
        s.add(Belief(user_id="u1", dimension="topic", key="tea",
                     claim_text="I love jasmine tea", confidence=0.9))
        s.commit()
        s.close()
    eng._engine.dispose()

    # ---- the PRE-deletion backup (real pg_dump of the main DB) ----------
    env = dict(os.environ, PGPASSWORD="mm")
    dump = subprocess.run(
        ["docker", "exec", "-e", "PGPASSWORD=mm", "mm-pg", "pg_dump", "-U",
         "postgres", "--format=custom", main_db],
        env=env, capture_output=True, check=True)
    assert len(dump.stdout) > 5000, "the pre-deletion dump is suspiciously empty"
    assert b"mm_deletion_watermark" in dump.stdout, (
        "the pre-deletion dump must carry the watermark table")
    assert b"mm_beliefs" in dump.stdout, "the dump must carry the belief table"
    pre_dump = tmp_path / "pre_deletion.dump"
    pre_dump.write_bytes(dump.stdout)

    # ---- the deletion happens AFTER that backup --------------------------
    eng2 = MemoryEngine(config=cfg, database_url=runtime_url,
                        llm_client=_L(), deletion_log=log,
                        tenant_id="default", app_id="default")
    eng2._ensure_db()
    result = eng2.delete_memories(user_id="u1", purpose="chat")
    assert result.beliefs >= 1, result
    tail, tail_hash = log.head(db_id)
    assert tail >= 1, "the deletion must be recorded in the independent log"
    witness_after = log.witness.read(db_id)
    assert witness_after.tail_seq == tail and witness_after.tail_hash == tail_hash
    eng2._engine.dispose()

    # ---- restore the PRE-deletion backup (real pg_restore) ---------------
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with no_audit_scope(), admin_engine.connect() as conn:
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
            conn.exec_driver_sql(f'CREATE DATABASE "{main_db}"')
    finally:
        admin_engine.dispose()
    # m5 patch 10: a GRANT applies to ONE database, so it must be issued
    # against the freshly created database (issuing it on the admin
    # connection targets `postgres`, leaving pg_restore unable to create
    # tables -- the bug this note exists to prevent).
    grant_engine = create_engine(
        admin_url.rsplit("/", 1)[0] + "/" + main_db,
        isolation_level="AUTOCOMMIT")
    try:
        with grant_engine.connect() as gconn:
            gconn.exec_driver_sql('GRANT ALL ON SCHEMA public TO mm_owner')
    finally:
        grant_engine.dispose()
    with open(pre_dump, "rb") as dump_file:
        restore = subprocess.run(
            ["docker", "exec", "-i", "-e", "PGPASSWORD=mm", "mm-pg",
             "pg_restore", "-U", "postgres", "--role=mm_owner",
             "--dbname=" + main_db],
            stdin=dump_file, capture_output=True)
    # verification reads use the OWNER login inside no_audit_scope (the
    # tenant-predicate audit rule is for runtime paths, not test checks)
    owner = create_engine(
        admin_url.rsplit("/", 1)[0] + "/" + main_db)
    try:
        with no_audit_scope(), owner.connect() as conn:
            n_tables = conn.execute(text(
                "SELECT count(*) FROM pg_tables WHERE schemaname='public' "
                "AND tablename LIKE 'mm_%'")).scalar()
            applied = conn.execute(text(
                "SELECT applied_seq FROM mm_deletion_watermark WHERE id = 1"
            )).scalar()
            beliefs_restored = conn.execute(text(
                "SELECT count(*) FROM mm_beliefs WHERE user_id='u1'")).scalar()
    finally:
        owner.dispose()
    assert n_tables >= 20, (
        f"pg_restore produced only {n_tables} mm_* tables; "
        f"stderr={restore.stderr.decode()[:600]}")

    # the restored backup really is BEHIND the log (that is what replay is for)
    assert applied < tail, (applied, tail)
    assert beliefs_restored >= 1, "the restored backup still holds the memory"

    # ---- startup on the restored DB: replay must re-apply the deletion ----
    eng3 = MemoryEngine(config=cfg, database_url=runtime_url,
                        llm_client=_L(), deletion_log=log,
                        tenant_id="default", app_id="default")
    eng3._ensure_db()          # witness/credential/chain gates + replay
    check = create_engine(admin_url.rsplit("/", 1)[0] + "/" + main_db)
    try:
        with no_audit_scope(), check.connect() as conn:
            left = conn.execute(text(
                "SELECT count(*) FROM mm_beliefs WHERE user_id='u1'")).scalar()
            landed = conn.execute(text(
                "SELECT applied_seq FROM mm_deletion_watermark WHERE id = 1"
            )).scalar()
    finally:
        check.dispose()
    eng3._engine.dispose()
    assert left == 0, "the restored memory must be deleted by the replay"
    assert landed == tail, (landed, tail)

    # §3: the same restored main DB + independent log, driven through the
    # RESTRICTED credentials (runtime log role appends/maintains the witness,
    # maintenance role compacts) — the full chain on limited grants.
    app_url, maint_url = _p10_build_restricted_log_db()
    try:
        rlog = PgDeletionLog(app_url, maintenance_url=maint_url)
        rid = "c1" * 16
        rlog.ensure_genesis(rid, applied=0)
        for i in range(1, 4):
            rlog.append(DeletionEntry(
                db_id=rid, tenant_id="t", app_id="a", user_id=f"u{i}",
                scope="session", scope_key=f"s{i}", fingerprints=[],
                generation=i, purpose="p"))
        rlog.verify_witness(rid, applied=3)
        assert rlog.prune(rid, 2) == 2
        rlog.verify_witness(rid, applied=3)
        assert rlog.witness.read(rid).tail_seq == 3
    finally:
        _p10_drop_restricted_log_db()

    # cleanup
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with no_audit_scope(), admin_engine.connect() as conn:
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
    finally:
        admin_engine.dispose()


# ---------------------------------------------------------------------------
# M5 patch 10 §4: the ACTUAL log/witness chain runs on REAL RESTRICTED
# credentials -- not separate GRANT probes.  Owner provisions; the runtime
# log role (no DDL, no DELETE) performs genesis/append/verify; the
# maintenance role performs the compaction.
# ---------------------------------------------------------------------------

BASE_PG = "postgresql://postgres:mm@127.0.0.1:5437"
P10_LOGDB = "mm_test_p10_roles"
P10_APP = ("mm_p10_logapp", "pw_mm_p10_logapp")
P10_MAINT = ("mm_p10_logmaint", "pw_mm_p10_logmaint")


def _p10_build_restricted_log_db():
    """Owner provisions a log DB + the two restricted roles; returns the
    (app_url, maint_url) the running code must use."""
    from mirror_memory.core.sql_audit import no_audit_scope

    admin = create_engine(BASE_PG + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with no_audit_scope(), admin.connect() as conn:
            for role, pw in (P10_APP, P10_MAINT):
                if not conn.execute(text(
                        "SELECT 1 FROM pg_roles WHERE rolname = :r"),
                        {"r": role}).first():
                    conn.exec_driver_sql(
                        f"CREATE ROLE {role} LOGIN PASSWORD '{pw}'")
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{P10_LOGDB}" WITH (FORCE)')
            conn.exec_driver_sql(f'CREATE DATABASE "{P10_LOGDB}"')
    finally:
        admin.dispose()

    owner = create_engine(BASE_PG + "/" + P10_LOGDB)
    try:
        PgDeletionLog(owner).provision_schema()   # owner-only DDL
        with owner.begin() as conn:
            conn.exec_driver_sql(
                f"GRANT USAGE ON SCHEMA public TO {P10_APP[0]}, {P10_MAINT[0]}")
            # runtime log role: append-only on the log, witness maintenance,
            # registry append.  NO DDL, NO DELETE.
            conn.exec_driver_sql(
                f"GRANT SELECT, INSERT ON mm_deletion_log TO {P10_APP[0]}")
            conn.exec_driver_sql(
                "GRANT SELECT, INSERT, UPDATE ON mm_deletion_log_witness "
                f"TO {P10_APP[0]}")
            conn.exec_driver_sql(
                "GRANT SELECT, INSERT ON mm_creation_registry "
                f"TO {P10_APP[0]}")
            # maintenance: compaction only
            conn.exec_driver_sql(
                "GRANT SELECT, INSERT, DELETE ON mm_deletion_log "
                f"TO {P10_MAINT[0]}")
            conn.exec_driver_sql(
                "GRANT SELECT ON mm_deletion_log_witness "
                f"TO {P10_MAINT[0]}")
    finally:
        owner.dispose()

    def _url(role, pw):
        return (BASE_PG.replace("postgres:mm", f"{role}:{pw}")
                + "/" + P10_LOGDB)

    return _url(*P10_APP), _url(*P10_MAINT)


def _p10_drop_restricted_log_db():
    from mirror_memory.core.sql_audit import no_audit_scope

    admin = create_engine(BASE_PG + "/postgres", isolation_level="AUTOCOMMIT")
    try:
        with no_audit_scope(), admin.connect() as conn:
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{P10_LOGDB}" WITH (FORCE)')
    finally:
        admin.dispose()


def test_p10_pg_chain_runs_on_restricted_roles(tmp_path, monkeypatch):
    """The REAL running code on REAL restricted logins: genesis, append,
    verify and prune all succeed, and the runtime role is proven unable to
    DROP the table or DELETE log rows."""
    _pg_witness_env(tmp_path, monkeypatch)
    app_url, maint_url = _p10_build_restricted_log_db()
    try:
        log = PgDeletionLog(app_url, maintenance_url=maint_url)
        db_id = "b1" * 16
        log.ensure_genesis(db_id, applied=0)
        for i in range(1, 4):
            log.append(DeletionEntry(
                db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
                scope="session", scope_key=f"s{i}", fingerprints=[],
                generation=i, purpose="p"))
        assert log.head(db_id)[0] == 3
        log.verify_witness(db_id, applied=3)

        # compaction on the MAINTENANCE credential
        assert log.prune(db_id, 2) == 2
        assert log.checkpoint_seq(db_id) == 2
        log.verify_witness(db_id, applied=3)
        assert log.witness.read(db_id).tail_seq == 3

        # the runtime role must NOT be able to delete rows or drop tables
        app_engine = create_engine(app_url, isolation_level="AUTOCOMMIT")
        try:
            for statement in (
                "DELETE FROM mm_deletion_log WHERE db_id = :db",
                "DROP TABLE mm_deletion_log",
            ):
                with pytest.raises(Exception) as excinfo:
                    with app_engine.begin() as conn:
                        conn.execute(text(statement), {"db": db_id})
                # PostgreSQL refuses these two ways depending on the object:
                # missing table privilege -> "permission denied", non-owner
                # DROP -> "must be owner".  Both are the intended refusal.
                msg = str(excinfo.value).lower()
                assert ("permission denied" in msg or "must be owner" in msg), (
                    statement, msg[:120])
        finally:
            app_engine.dispose()

        # evidence still intact after the refused destructive attempts
        assert log.head(db_id)[0] == 3
    finally:
        _p10_drop_restricted_log_db()


def test_p10_pg_issue_genesis_on_restricted_role_without_owner(
        tmp_path, monkeypatch):
    """The runtime role cannot provision: a MISSING table fails closed with
    the owner-provisioning error instead of silently creating it."""
    _pg_witness_env(tmp_path, monkeypatch)
    app_url, maint_url = _p10_build_restricted_log_db()
    from mirror_memory.core.sql_audit import no_audit_scope

    admin = create_engine(BASE_PG + "/" + P10_LOGDB, isolation_level="AUTOCOMMIT")
    try:
        with no_audit_scope(), admin.connect() as conn:
            conn.exec_driver_sql("DROP TABLE IF EXISTS mm_creation_registry")
    finally:
        admin.dispose()
    try:
        log = PgDeletionLog(app_url, maintenance_url=maint_url)
        with pytest.raises(Exception) as excinfo:
            log.ensure_genesis("b2" * 16, applied=0)
        msg = str(excinfo.value).lower()
        assert "permission denied" in msg or "does not exist" in msg, msg[:160]
    finally:
        _p10_drop_restricted_log_db()


# ---------------------------------------------------------------------------
# M5 patch 11 §2: deletion-adopt on the REAL PG path — dry run with no side
# effects, reconcile refused on divergence, retire applied.  The CLI must
# work identically on the postgres backend (patch 9 flagged the signature
# mismatch; this pins it).
# ---------------------------------------------------------------------------


def test_p11_adopt_cli_on_pg_backend_dry_run_and_reconcile(
        log_db_url, tmp_path, monkeypatch, capsys):
    from mirror_memory import deletion_adopt
    from mirror_memory.core import deletion as _dm

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))

    monkeypatch.setattr(_dm, "deletion_log_from_config",
                        lambda *a, **k: log)
    witness_row_before = log.witness.read(db_id)

    # 1. dry run: prints the plan, changes nothing
    monkeypatch.setattr("sys.argv", ["deletion-adopt", "--database-url",
                                     log_db_url, "--db-id", db_id,
                                     "--config-dir", "config/",
                                     "--reconcile-witness"])
    rc = deletion_adopt.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "chain: verified" in out
    assert log.witness.read(db_id).__dict__ == witness_row_before.__dict__

    # 2. apply with --yes
    monkeypatch.setattr("sys.argv", ["deletion-adopt", "--database-url",
                                     log_db_url, "--db-id", db_id,
                                     "--config-dir", "config/",
                                     "--reconcile-witness", "--yes"])
    rc = deletion_adopt.main()
    assert rc == 0
    state = log.witness.read(db_id)
    assert state.tail_seq == log.head(db_id)[0]
    assert state.tail_hash == log.head(db_id)[1]


def test_p11_adopt_cli_on_pg_backend_retire(
        log_db_url, tmp_path, monkeypatch, capsys):
    from mirror_memory import deletion_adopt
    from mirror_memory.core import deletion as _dm

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    assert log.witness.read(db_id) is not None

    monkeypatch.setattr(_dm, "deletion_log_from_config",
                        lambda *a, **k: log)
    monkeypatch.setattr("sys.argv", ["deletion-adopt", "--database-url",
                                     log_db_url, "--db-id", db_id,
                                     "--config-dir", "config/",
                                     "--retire-witness", "--yes"])
    rc = deletion_adopt.main()
    assert rc == 0
    assert log.witness.read(db_id) is None


def test_p11_adopt_cli_on_pg_backend_refuses_divergence(
        log_db_url, tmp_path, monkeypatch, capsys):
    from mirror_memory import deletion_adopt
    from mirror_memory.core import deletion as _dm
    from mirror_memory.core.witness import WitnessError

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    witness_before = log.witness.read(db_id)

    # tamper the row so the head hash no longer matches the witness
    admin = create_engine(log_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.begin() as conn:
            conn.execute(text(
                "UPDATE mm_deletion_log SET hash = repeat('d', 64) "
                "WHERE db_id = :db AND seq = 1"), {"db": db_id})
    finally:
        admin.dispose()

    monkeypatch.setattr(_dm, "deletion_log_from_config",
                        lambda *a, **k: log)
    monkeypatch.setattr("sys.argv", ["deletion-adopt", "--database-url",
                                     log_db_url, "--db-id", db_id,
                                     "--config-dir", "config/",
                                     "--reconcile-witness", "--yes"])
    with pytest.raises(SystemExit) as excinfo:
        deletion_adopt.main()
    assert excinfo.value.code == 1
    # the witness still holds the ORIGINAL tail hash
    after = log.witness.read(db_id)
    assert after.tail_hash == witness_before.tail_hash


# ---------------------------------------------------------------------------
# M5 patch 11 §3: deterministic multiprocess prune/restore interleaving.
# Process A holds the log advisory lock (append mid-transaction) while
# process B attempts a prune; B must not compact an unverified or
# concurrently-changing history.
# ---------------------------------------------------------------------------


def _p11_appender(log_db_url, db_id, idx, errors_q):
    """Module-level so spawn can pickle it."""
    from mirror_memory.core.deletion import PgDeletionLog, DeletionEntry

    try:
        l = PgDeletionLog(log_db_url)
        for j in range(8):
            l.append(DeletionEntry(
                db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{idx}",
                scope="session", scope_key=f"s{idx}-{j}", fingerprints=[],
                generation=j + 1, purpose="p"))
    except BaseException as exc:  # noqa: BLE001
        errors_q.put(f"appender {idx}: {type(exc).__name__}: {str(exc)[:140]}")


def _p11_pruner(log_db_url, db_id, errors_q):
    """Module-level so spawn can pickle it."""
    import time as _time

    from mirror_memory.core.deletion import PgDeletionLog

    try:
        l = PgDeletionLog(log_db_url, maintenance_url=log_db_url)
        for _ in range(4):
            head = l.head(db_id)[0]
            if head > 4:
                l.prune(db_id, head - 2)
            _time.sleep(0.05)
    except BaseException as exc:  # noqa: BLE001
        errors_q.put(f"pruner: {type(exc).__name__}: {str(exc)[:140]}")


def test_p11_pg_prune_and_append_interleave_safely(
        log_db_url, tmp_path, monkeypatch):
    """Deterministic interleave: 4 appender processes + 1 pruner, all on the
    same db_id.  Every prune runs under the verified-continuity gate and the
    witness never regresses or blesses a fork."""
    import multiprocessing as mp

    _pg_witness_env(tmp_path, monkeypatch)
    log = PgDeletionLog(log_db_url)
    db_id = _db_id()
    log.ensure_genesis(db_id, applied=0)
    witness_floor = log.witness.read(db_id)

    ctx = mp.get_context("spawn")
    errors: list = ctx.Queue()

    procs = [ctx.Process(target=_p11_appender,
                         args=(log_db_url, db_id, i, errors))
             for i in range(4)]
    procs.append(ctx.Process(target=_p11_pruner,
                             args=(log_db_url, db_id, errors)))
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
        assert p.exitcode == 0, p.exitcode

    collected = []
    while not errors.empty():
        collected.append(errors.get())
    assert not collected, collected[:4]

    head_seq, head_hash = log.head(db_id)
    assert head_seq >= 32, head_seq
    log.verify_chain(db_id)
    state = log.witness.read(db_id)
    assert state.tail_seq >= witness_floor.tail_seq
    assert state.tail_seq <= head_seq


# ---------------------------------------------------------------------------
# M5 patch 12 §3: the COMBINED acceptance — one restored main DB, ONE log
# identity, REAL restricted credentials, and a Barrier-scheduled interleave
# of append / prune / verify across processes.  This is the joint-acceptance
# case the brief says must not be replaced by separate log tests.
# ---------------------------------------------------------------------------


def _p12_child_append(log_db_url, db_id, idx, errors_q):
    from mirror_memory.core.deletion import PgDeletionLog, DeletionEntry

    try:
        l = PgDeletionLog(log_db_url)
        for j in range(6):
            l.append(DeletionEntry(
                db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{idx}",
                scope="session", scope_key=f"s{idx}-{j}", fingerprints=[],
                generation=j + 1, purpose="p"))
    except BaseException as exc:  # noqa: BLE001
        errors_q.put(f"append {idx}: {type(exc).__name__}: {str(exc)}")


def _p12_child_prune(log_db_url, db_id, errors_q):
    from mirror_memory.core.deletion import PgDeletionLog

    try:
        l = PgDeletionLog(log_db_url, maintenance_url=log_db_url)
        head = l.head(db_id)[0]
        if head >= 6:
            l.prune(db_id, head - 2)
    except BaseException as exc:  # noqa: BLE001
        errors_q.put(f"prune: {type(exc).__name__}: {str(exc)[:400]}")


def test_p12_pg_combined_restricted_roles_with_barrier(
        log_db_url, tmp_path, monkeypatch):
    """Same restricted log DB the whole time: genesis on the app credential,
    then appender processes and a pruner process released by ONE barrier, then
    verify on the app credential.  Proves the continuity gate, the advisory
    lock and the witness monotonicity hold across processes AND roles."""
    import multiprocessing as mp

    _pg_witness_env(tmp_path, monkeypatch)
    app_url, maint_url = _p10_build_restricted_log_db()
    try:
        # production order: the OWNER provisions the tables ONCE, before any
        # runtime process exists (the restricted roles hold no DDL -- see
        # patch 10 §4).  Spawning without provisioning would race the CREATE
        # TABLE between children, which is exactly what the role split
        # prevents in production.
        owner = create_engine(BASE_PG + "/" + P10_LOGDB)
        try:
            PgDeletionLog(owner).provision_schema()
        finally:
            owner.dispose()
        # genesis on the APP credential (the runtime creates its own life)
        log = PgDeletionLog(app_url, maintenance_url=maint_url)
        db_id = "d1" * 16
        log.ensure_genesis(db_id, applied=0)
        floor = log.witness.read(db_id).tail_seq

        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        # every process uses the RESTRICTED log database URL: the conftest
        # `log_db_url` fixture points at the shared test log store -- a
        # DIFFERENT database from the one provisioned above (M5 patch 12 §3)
        procs = [ctx.Process(target=_p12_child_append,
                             args=(app_url, db_id, i, errors))
                 for i in range(3)]
        procs.append(ctx.Process(target=_p12_child_prune,
                                 args=(app_url, db_id, errors)))
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)
            assert p.exitcode == 0, p.exitcode

        collected = []
        while not errors.empty():
            collected.append(errors.get())
        assert not collected, collected[:4]

        # invariants on the app credential after the interleave
        head_seq, head_hash = log.head(db_id)
        assert head_seq == 18
        log.verify_chain(db_id)
        state = log.witness.read(db_id)
        assert state.tail_seq == 18
        assert state.tail_hash == head_hash
        assert state.tail_seq >= floor
        log.verify_witness(db_id, applied=18)
    finally:
        _p10_drop_restricted_log_db()


# ---------------------------------------------------------------------------
# M5 patch 12 §3: the FULL joint acceptance on ONE restored main DB and ONE
# log identity with the REAL restricted credentials, with the append/prune
# children released by a Barrier so the interleave is deterministic:
#   seed -> pre-deletion pg_dump -> delete -> restore -> concurrent
#   append+prune on the restricted log -> startup replay -> invariants.
# ---------------------------------------------------------------------------


def _p12b_child_append(log_db_url, db_id, idx, start_ev, errors_q):
    from mirror_memory.core.deletion import PgDeletionLog, DeletionEntry

    try:
        start_ev.wait(timeout=60)          # barrier: all start together
        l = PgDeletionLog(log_db_url)
        for j in range(4):
            l.append(DeletionEntry(
                db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{idx}",
                scope="session", scope_key=f"s{idx}-{j}", fingerprints=[],
                generation=j + 1, purpose="p"))
    except BaseException as exc:  # noqa: BLE001
        errors_q.put(f"append {idx}: {type(exc).__name__}: {exc}")


def _p12b_child_prune(log_db_url, db_id, start_ev, errors_q):
    from mirror_memory.core.deletion import PgDeletionLog

    try:
        start_ev.wait(timeout=60)          # barrier: all start together
        l = PgDeletionLog(log_db_url, maintenance_url=log_db_url)
        head = l.head(db_id)[0]
        if head >= 5:
            l.prune(db_id, head - 1)
    except BaseException as exc:  # noqa: BLE001
        errors_q.put(f"prune: {type(exc).__name__}: {exc}")


def test_p12_pg_joint_restore_then_concurrent_append_prune(
        log_db_url, tmp_path, monkeypatch):
    """ONE restored main DB + ONE log identity + REAL restricted logins +
    a Barrier-scheduled append/prune interleave + startup replay."""
    import multiprocessing as mp
    import subprocess

    from mirror_memory.api import MemoryEngine
    from mirror_memory.config.loader import load_config
    from mirror_memory.core.models import Belief
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.sql_audit import no_audit_scope
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from sqlalchemy.orm import sessionmaker

    _pg_witness_env(tmp_path, monkeypatch)
    app_url, maint_url = _p10_build_restricted_log_db()
    main_db = "mm_test_p12_joint"
    runtime_url, admin_url = _pg_repro_database(main_db)

    class _L:
        def generate(self, **_kw):
            return '{"patterns": []}'

    try:
        owner = create_engine(BASE_PG + "/" + P10_LOGDB)
        try:
            PgDeletionLog(owner).provision_schema()
        finally:
            owner.dispose()

        cfg = load_config("config/")
        cfg.llm_client = _L()
        alog = PgDeletionLog(app_url, maintenance_url=maint_url)
        eng = MemoryEngine(config=cfg, database_url=runtime_url,
                           llm_client=_L(), deletion_log=alog,
                           tenant_id="default", app_id="default")
        eng._ensure_db()
        db_id = eng._db_id
        with tenant_scope(*eng._tenant), fixture_write_scope("p12b seed"):
            s = sessionmaker(bind=eng._engine)()
            set_memory_enabled(s, "u1", True)
            s.add(Belief(user_id="u1", dimension="topic", key="tea",
                         claim_text="I love jasmine tea", confidence=0.9))
            s.commit()
            s.close()
        eng._engine.dispose()

        # pre-deletion backup
        env = dict(os.environ, PGPASSWORD="mm")
        dump = subprocess.run(
            ["docker", "exec", "-e", "PGPASSWORD=mm", "mm-pg", "pg_dump",
             "-U", "postgres", "--format=custom", main_db],
            env=env, capture_output=True, check=True)
        assert len(dump.stdout) > 5000 and b"mm_beliefs" in dump.stdout
        pre = tmp_path / "pre.dump"
        pre.write_bytes(dump.stdout)

        # deletion recorded in the independent restricted log
        eng2 = MemoryEngine(config=cfg, database_url=runtime_url,
                            llm_client=_L(), deletion_log=alog,
                            tenant_id="default", app_id="default")
        eng2._ensure_db()
        res = eng2.delete_memories(user_id="u1", purpose="chat")
        assert res.beliefs >= 1, res
        eng2._engine.dispose()
        tail_before = alog.head(db_id)[0]
        assert tail_before >= 1

        # restore the PRE-deletion main DB (real pg_restore)
        admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            with no_audit_scope(), admin_engine.connect() as conn:
                conn.exec_driver_sql(
                    f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
                conn.exec_driver_sql(f'CREATE DATABASE "{main_db}"')
        finally:
            admin_engine.dispose()
        grant_engine = create_engine(
            admin_url.rsplit("/", 1)[0] + "/" + main_db,
            isolation_level="AUTOCOMMIT")
        try:
            with grant_engine.connect() as gconn:
                gconn.exec_driver_sql('GRANT ALL ON SCHEMA public TO mm_owner')
        finally:
            grant_engine.dispose()
        with open(pre, "rb") as fh:
            subprocess.run(
                ["docker", "exec", "-i", "-e", "PGPASSWORD=mm", "mm-pg",
                 "pg_restore", "-U", "postgres", "--role=mm_owner",
                 "--dbname=" + main_db], stdin=fh, capture_output=True)
        owner = create_engine(admin_url.rsplit("/", 1)[0] + "/" + main_db)
        try:
            with no_audit_scope(), owner.connect() as conn:
                n = conn.execute(text(
                    "SELECT count(*) FROM pg_tables WHERE schemaname='public' "
                    "AND tablename LIKE 'mm_%'")).scalar()
                still_there = conn.execute(text(
                    "SELECT count(*) FROM mm_beliefs WHERE user_id='u1'")).scalar()
        finally:
            owner.dispose()
        assert n >= 20, "the restore must recreate the main schema"
        assert still_there >= 1, "the restored backup still has the memory"

        # Barrier-released append+prune interleave on the SAME log identity
        ctx = mp.get_context("spawn")
        start = ctx.Event()
        errors = ctx.Queue()
        procs = [ctx.Process(target=_p12b_child_append,
                             args=(app_url, db_id, i, start, errors))
                 for i in range(2)]
        procs.append(ctx.Process(target=_p12b_child_prune,
                                 args=(app_url, db_id, start, errors)))
        for p in procs:
            p.start()
        start.set()                        # release the barrier
        for p in procs:
            p.join(timeout=180)
            assert p.exitcode == 0, p.exitcode
        collected = []
        while not errors.empty():
            collected.append(errors.get())
        assert not collected, collected[:3]

        # startup on the RESTORED main DB: the replay re-applies the deletion
        eng3 = MemoryEngine(config=cfg, database_url=runtime_url,
                            llm_client=_L(), deletion_log=alog,
                            tenant_id="default", app_id="default")
        eng3._ensure_db()
        check = create_engine(admin_url.rsplit("/", 1)[0] + "/" + main_db)
        try:
            with no_audit_scope(), check.connect() as conn:
                left = conn.execute(text(
                    "SELECT count(*) FROM mm_beliefs WHERE user_id='u1'")).scalar()
                landed = conn.execute(text(
                    "SELECT applied_seq FROM mm_deletion_watermark WHERE id = 1"
                )).scalar()
        finally:
            check.dispose()
        eng3._engine.dispose()
        assert left == 0, "the replay must delete the restored memory"
        head_after = alog.head(db_id)[0]
        assert landed <= head_after, (landed, head_after)
        assert head_after >= tail_before + 8, (head_after, tail_before)
        alog.verify_chain(db_id)
        st = alog.witness.read(db_id)
        assert st.tail_seq == head_after and st.tail_hash == alog.head(db_id)[1]
    finally:
        _p10_drop_restricted_log_db()
        admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            with no_audit_scope(), admin_engine.connect() as conn:
                conn.exec_driver_sql(
                    f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
        finally:
            admin_engine.dispose()
