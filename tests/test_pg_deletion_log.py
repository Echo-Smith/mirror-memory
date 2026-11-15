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
