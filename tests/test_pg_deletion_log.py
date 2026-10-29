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
        exists = admin.connect().execute(text(
            "SELECT 1 FROM pg_database WHERE datname = :d"), {"d": LOG_DB}).first()
        if not exists:
            admin.connect().execute(text(f'CREATE DATABASE "{LOG_DB}"'))
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
    assert e1.prev_hash == GENESIS_HASH and e2.prev_hash == e1.hash
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
            rows = app.connect().execute(text(
                "SELECT count(*) FROM mm_deletion_log WHERE db_id = :db"), {"db": db_id}).scalar()
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
    fe = file_log.append(entry)
    pe = pg_log.append(entry)
    assert fe.hash == pe.hash
    assert pe.prev_hash == GENESIS_HASH  # a fresh db life starts the chain
    file_log.verify_chain(entry.db_id)
    pg_log.verify_chain(entry.db_id)
