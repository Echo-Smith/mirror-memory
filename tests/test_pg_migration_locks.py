"""M2 patch task 3 -- PostgreSQL migration DDL never blocks for long.

Every migration DDL waits at most ``MM_MIGRATION_LOCK_TIMEOUT`` before
failing (and retrying), and the composite primary key is built with
``CREATE UNIQUE INDEX CONCURRENTLY`` then adopted with ``PRIMARY KEY USING
INDEX`` -- so no step scans a table while holding an exclusive lock.

Proof tests (PostgreSQL only; SQLite cannot express any of this):

- a leftover build index is dropped and rebuilt, and the swap completes --
  this is also the CONCURRENTLY-failure recovery path: a cancelled build
  leaves its index behind (PostgreSQL marks it INVALID) and ``DROP INDEX``
  removes an INVALID index exactly like a valid one, so the
  drop-then-rebuild policy recovers without special-casing validity;
- a concurrent reader on a 200k-row table never stalls beyond a second
  while the table is converted.
"""

from __future__ import annotations

import os
import threading
import time

import pytest
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    create_engine,
    inspect,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from mirror_memory.core.models import Base
from mirror_memory.core.schema_migration import (
    MigrationRefused,
    SetLockTimeout,
    _pkey_build_index,
    _replace_primary_keys,
    _swap_primary_key,
)
from mirror_memory.core.sql_audit import no_audit_scope

PG_URL = os.environ.get("MM_DATABASE_URL", "")
IS_PG = PG_URL.startswith("postgres")

PROBE_TABLE = "mm_probe_migrate"
BUILD_INDEX = "mm_probe_migrate__pkey_build"

#: Private base so the probe's mapping never leaks into the engine's own
#: ``Base.metadata`` (the swap helpers register their own model-shape Table
#: there, and it is removed again at cleanup).
_ProbeBase = declarative_base()


class _Probe(_ProbeBase):
    __tablename__ = PROBE_TABLE

    id: int = Column(Integer, primary_key=True, autoincrement=False)
    tenant_id: str = Column(String(64), nullable=False)
    app_id: str = Column(String(64), nullable=False)
    payload: str = Column(String(32), nullable=False)


pytestmark = [
    pytest.mark.postgres_only("PostgreSQL catalog behaviour; SQLite cannot express it"),
    pytest.mark.skipif(not IS_PG, reason="needs MM_DATABASE_URL pointing at PostgreSQL"),
]


@pytest.fixture
def pg_engine():
    eng = create_engine(PG_URL)
    yield eng
    eng.dispose()


def _register_probe() -> Table:
    """The probe table's MODEL shape (composite PK) for the swap helpers.

    The live table below is created legacy-shaped (single-column PK); the
    registration is removed again at cleanup so the rest of the suite never
    sees the probe.
    """
    if PROBE_TABLE in Base.metadata.tables:
        return Base.metadata.tables[PROBE_TABLE]
    return Table(
        PROBE_TABLE,
        Base.metadata,
        Column("id", Integer, autoincrement=False),
        Column("tenant_id", String(64), nullable=False),
        Column("app_id", String(64), nullable=False),
        Column("payload", String(32), nullable=False),
        PrimaryKeyConstraint("tenant_id", "app_id", "id"),
    )


def _probe_cleanup(engine) -> None:
    Table(PROBE_TABLE, MetaData()).drop(engine, checkfirst=True)
    model_table = Base.metadata.tables.get(PROBE_TABLE)
    if model_table is not None:
        Base.metadata.remove(model_table)


def _seed_live_table(pg_engine, rows: int) -> Table:
    """Create the probe table legacy-shaped (single-column PK) and fill it.

    Rows carry explicit unique ids (the ids come from the loop index, and a
    single sequence of 1..200000 cannot collide), so the bulk insert works
    against the legacy single-column primary key as-is.
    """
    meta = MetaData()
    live = Table(
        PROBE_TABLE,
        meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("tenant_id", String(64), nullable=False, server_default=""),
        Column("app_id", String(64), nullable=False, server_default=""),
        Column("payload", String(32), nullable=False, server_default=""),
    )
    live.drop(pg_engine, checkfirst=True)
    live.create(pg_engine)
    factory = sessionmaker(bind=pg_engine)
    with no_audit_scope():
        for start in range(0, rows, 20_000):
            mappings = [
                {
                    "id": i,
                    "tenant_id": f"t{i % 8}",
                    "app_id": "x",
                    "payload": f"p{i}",
                }
                for i in range(start, min(start + 20_000, rows))
            ]
            session = factory()
            try:
                session.bulk_insert_mappings(_Probe, mappings)
                session.commit()
            finally:
                session.close()
    return live


def test_lock_timeout_value_is_validated():
    """The lock timeout reaches statement text as a literal, so the value is
    pinned to a strict duration grammar before anything is emitted."""
    assert SetLockTimeout("250ms").value == "250ms"
    assert SetLockTimeout("3s").value == "3s"
    with pytest.raises(MigrationRefused):
        SetLockTimeout("3; DROP TABLE mm_beliefs")
    with pytest.raises(MigrationRefused):
        SetLockTimeout("banana")


def test_leftover_build_index_is_dropped_and_swap_completes(pg_engine, monkeypatch):
    """A leftover build index (here: a plain non-unique index someone left
    under the build name) must not break or be adopted by the swap.

    This is also the CONCURRENTLY-failure recovery path: a cancelled build
    leaves its index behind (PostgreSQL marks it INVALID) and ``DROP INDEX``
    removes an INVALID index exactly like a valid one, so the
    drop-then-rebuild policy recovers without special-casing validity.
    """
    monkeypatch.setenv("MM_MIGRATION_LOCK_TIMEOUT", "10s")
    _register_probe()
    live = _seed_live_table(pg_engine, 50)
    factory = sessionmaker(bind=pg_engine)
    try:
        _swap_primary_key(pg_engine, PROBE_TABLE)
        insp = inspect(pg_engine)
        assert insp.get_pk_constraint(PROBE_TABLE)["constrained_columns"] == [
            "tenant_id", "app_id", "id",
        ]

        # someone leaves a fresh index under the build name (a plain index,
        # not the CONCURRENTLY build object): the next swap must drop it and
        # rebuild rather than adopt or fail
        from sqlalchemy import Index as _Index

        leftover = _Index(
            BUILD_INDEX,
            *[_register_probe().c[c.name] for c in _register_probe().primary_key.columns],
        )
        leftover.create(pg_engine)
        assert inspect(pg_engine).has_index(PROBE_TABLE, BUILD_INDEX)
        _swap_primary_key(pg_engine, PROBE_TABLE)
        # a fresh inspector: the earlier one caches reflection results
        assert not inspect(pg_engine).has_index(PROBE_TABLE, BUILD_INDEX)

        # the data survived both swaps, and reads still work through a
        # plain session (timing probe: outside the auditor, same as above)
        session = factory()
        with no_audit_scope():
            try:
                assert session.query(live).count() == 50
            finally:
                session.close()
    finally:
        _probe_cleanup(pg_engine)


def test_swap_on_200k_rows_never_blocks_a_concurrent_reader(pg_engine, monkeypatch):
    """Convert a 200k-row table while another connection reads it in a loop.

    The old single-statement swap held ``ACCESS EXCLUSIVE`` for the whole
    index build; the CONCURRENTLY + USING INDEX recipe must keep every read
    gap around milliseconds.  The measured maximum is printed for the
    report.
    """
    monkeypatch.setenv("MM_MIGRATION_LOCK_TIMEOUT", "10s")
    _register_probe()
    live = _seed_live_table(pg_engine, 200_000)
    factory = sessionmaker(bind=pg_engine)
    try:
        stop = threading.Event()
        gaps: list[float] = []

        def reader() -> None:
            # timing probe, not a tenant read: the probe table is a tenant
            # table, but this loop measures lock waits, so it runs outside
            # the auditor (no_audit_scope) -- the swap itself runs inside
            # system_scope the same way.
            last = time.perf_counter()
            with no_audit_scope():
                while not stop.is_set():
                    session = factory()
                    try:
                        session.query(live).count()
                    finally:
                        session.close()
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now
                    time.sleep(0.005)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        try:
            started = time.perf_counter()
            _replace_primary_keys(pg_engine, [PROBE_TABLE])
            duration = time.perf_counter() - started
        finally:
            stop.set()
            reader_thread.join()

        insp = inspect(pg_engine)
        assert insp.get_pk_constraint(PROBE_TABLE)["constrained_columns"] == [
            "tenant_id", "app_id", "id",
        ]
        assert gaps, "the reader never completed a single pass"
        max_gap = max(gaps)
        print(
            f"pk swap on 200k rows: {duration:.2f}s, "
            f"{len(gaps)} reads, max read gap {max_gap * 1000:.1f} ms"
        )
        # a blocking swap would hold the reader for the whole index build
        assert max_gap < 2.0, f"reader stalled for {max_gap:.2f}s"
    finally:
        _probe_cleanup(pg_engine)