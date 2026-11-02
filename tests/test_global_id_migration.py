"""M4 batch 2 task 0: ``mm_beliefs.global_id`` -- the belief's cross-device
identity, migration step 12.

- a fresh belief is minted a global_id at creation (model default);
- a step-11 database upgrades to 12: every row backfilled with a UUID
  (database-minted, one per row), NOT NULL on PostgreSQL, the unique
  (tenant, app, global_id) index valid, schema parity intact;
- step 11's lesson: an interrupted run RESUMES -- needed() looks at the
  full set of artifacts (column, unfilled rows, NOT NULL, index), so a
  half-applied step is never mistaken for a finished one;
- both databases: the SQLite leg here, the PostgreSQL leg through the
  owner-side pg_admin_engine fixture.
"""

from __future__ import annotations

import re

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.core import schema_migration as sm
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, system_scope, tenant_scope
from tests.conftest import strip_to_step

TENANT = ("default", "default")
USER = "u-gid"
HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _migration_cfg(tmp_path):
    from mirror_memory.config.loader import load_config

    cfg = load_config("config/")
    cfg.deletion_log.directory = str(tmp_path / "deletions")
    return cfg


def _seed(engine, n: int = 7) -> None:
    F = sessionmaker(bind=engine)
    with tenant_scope(*TENANT), fixture_write_scope("seed beliefs"):
        s = F()
        set_memory_enabled(s, USER, True)
        for i in range(n):
            record_claim(s, USER, dimension="topic", key=f"k{i}",
                         claim_text=f"claim {i}", confidence=0.8,
                         session_id="s1")
        s.commit()
        s.close()


def _global_ids(engine) -> list[str]:
    with system_scope(reason="test: read every belief's global_id"):
        with engine.connect() as conn:
            return [r[0] for r in conn.execute(
                text("SELECT global_id FROM mm_beliefs ORDER BY id")).all()]


def _upgrade(engine, tmp_path):
    return sm.upgrade_database(
        engine, default_tenant=TENANT[0], default_app=TENANT[1],
        config=_migration_cfg(tmp_path), confirm_backup=True)


def test_new_beliefs_are_minted_a_global_id(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    Base.metadata.create_all(eng)
    sm.ensure_watermark(eng)
    _seed(eng, 3)
    ids = _global_ids(eng)
    assert len(ids) == 3 and all(HEX32.match(g) for g in ids)
    assert len(set(ids)) == 3
    eng.dispose()


def test_step12_upgrades_a_step11_database_sqlite(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'v11.db'}")
    Base.metadata.create_all(eng)
    sm.ensure_watermark(eng)
    _seed(eng)
    strip_to_step(eng, 11)
    assert sm.current_version(eng) == 11
    step = sm.Step12BeliefGlobalId()
    assert step.needed(eng)

    _upgrade(eng, tmp_path)

    assert sm.current_version(eng) == sm.LATEST_VERSION
    ids = _global_ids(eng)
    assert len(ids) == 7 and all(HEX32.match(g) for g in ids)
    assert len(set(ids)) == 7, "one UUID per row, not one per batch"
    assert sm._index_state(eng, "mm_beliefs", sm.GLOBAL_ID_INDEX) == "valid"
    assert not step.needed(eng)
    assert sm.verify_schema_parity(eng) == []
    eng.dispose()


def test_step12_resumes_after_an_interruption_sqlite(tmp_path):
    """Half-applied shapes must all read as NEEDED and finish cleanly:
    (a) column present, half the rows backfilled, no index;
    (b) every row backfilled but the index missing;
    (c) a version row would not be recorded by an interrupted run, so
        the runner re-enters the step -- and apply is re-entrant."""
    eng = create_engine(f"sqlite:///{tmp_path / 'half.db'}")
    Base.metadata.create_all(eng)
    sm.ensure_watermark(eng)
    _seed(eng, 6)
    strip_to_step(eng, 11)
    step = sm.Step12BeliefGlobalId()
    assert "global_id" not in {
        c["name"] for c in inspect(eng).get_columns("mm_beliefs")}, "strip left the column"

    # (a) column added, half backfilled, no index (the run died mid-backfill)
    with eng.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE mm_beliefs ADD COLUMN global_id VARCHAR(32) NOT NULL DEFAULT ''")
    with system_scope(reason="test: simulate a half-applied step 12"):
        with eng.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE mm_beliefs SET global_id = lower(hex(randomblob(16))) "
                "WHERE id IN (SELECT id FROM mm_beliefs ORDER BY id LIMIT 3)")
    assert step.needed(eng), "unfilled rows -> needed"
    half = _global_ids(eng)
    assert sum(1 for g in half if g) == 3

    _upgrade(eng, tmp_path)
    after = _global_ids(eng)
    assert all(HEX32.match(g) for g in after) and len(set(after)) == 6
    # the already-filled rows kept their ids (re-entry does not re-mint)
    assert [g for g in half if g] == after[:3]
    assert not step.needed(eng)

    # (b) index missing with every row filled
    with eng.begin() as conn:
        conn.exec_driver_sql("DROP INDEX IF EXISTS uq_mm_beliefs_global_id")
        conn.execute(text("DELETE FROM mm_schema_version WHERE version >= 12"))
    assert step.needed(eng), "a missing index -> needed, not 'already done'"
    _upgrade(eng, tmp_path)
    assert sm._index_state(eng, "mm_beliefs", sm.GLOBAL_ID_INDEX) == "valid"
    assert _global_ids(eng) == after, "re-entry never re-mints"
    eng.dispose()


def test_step12_upgrades_a_step11_database_pg(pg_admin_engine, tmp_path):
    """PostgreSQL: the real column drop (strip), ADD COLUMN nullable ->
    batched gen_random_uuid backfill -> NOT NULL proven without a
    blocking scan -> CREATE UNIQUE INDEX CONCURRENTLY; then an
    interruption (column added, half backfilled, nullable, no index)
    resumes."""
    eng = pg_admin_engine
    _seed(eng, 9)
    strip_to_step(eng, 11)
    assert "global_id" not in {c["name"] for c in inspect(eng).get_columns("mm_beliefs")}
    step = sm.Step12BeliefGlobalId()
    assert step.needed(eng)

    _upgrade(eng, tmp_path)

    col = next(c for c in inspect(eng).get_columns("mm_beliefs") if c["name"] == "global_id")
    assert col["nullable"] is False
    ids = _global_ids(eng)
    assert len(ids) == 9 and all(HEX32.match(g) for g in ids) and len(set(ids)) == 9
    assert sm._index_state(eng, "mm_beliefs", sm.GLOBAL_ID_INDEX) == "valid"
    assert not step.needed(eng)
    assert sm.verify_schema_parity(eng) == []

    # interruption: back to "column added nullable, half filled, no index"
    strip_to_step(eng, 11)
    with eng.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE mm_beliefs ADD COLUMN global_id VARCHAR(32)")
    with system_scope(reason="test: simulate a half-applied step 12"):
        with eng.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE mm_beliefs SET global_id = replace(gen_random_uuid()::text, '-', '') "
                "WHERE id IN (SELECT id FROM mm_beliefs ORDER BY id LIMIT 4)")
    assert step.needed(eng)
    _upgrade(eng, tmp_path)
    col = next(c for c in inspect(eng).get_columns("mm_beliefs") if c["name"] == "global_id")
    assert col["nullable"] is False
    ids2 = _global_ids(eng)
    assert all(HEX32.match(g) for g in ids2) and len(set(ids2)) == 9
    assert sm._index_state(eng, "mm_beliefs", sm.GLOBAL_ID_INDEX) == "valid"
    assert not step.needed(eng)
