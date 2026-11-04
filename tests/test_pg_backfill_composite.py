"""M3 batch-2 task 1 finding: step 6 could not run on a real PostgreSQL
version-5 database.

After step 2 every tenant table on PostgreSQL has the physical composite
primary key ``(tenant_id, app_id, id)``, and ``_backfill_column`` refused
anything but a single-column key -- so ``Step6PurposeAuthorization.apply``
raised ``MigrationError`` on the ``mm_evolution_jobs.purpose`` backfill the
moment it ran against a database that had actually been through steps
1-5.  The SQLite leg never saw it: its step-6 path is ``ADD COLUMN ... NOT
NULL DEFAULT`` with no backfill, and the migration fixtures build the
legacy database on SQLite.  These tests run the helper and the whole of
steps 6 and 7 against the PostgreSQL schema the suite builds (composite
keys, RLS on).
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import MetaData, create_engine, inspect, text

from mirror_memory.core import schema_migration as sm
from mirror_memory.core.schema_migration import upgrade_database
from mirror_memory.core.sql_audit import no_audit_scope


def _migration_cfg():
    """A deletion-log config for step-11 migrations that don't exercise
    the log itself (M4 patch 2: the migration refuses to mint a
    data-bearing database without a log to consult)."""
    import tempfile

    from mirror_memory.config.loader import load_config

    cfg = load_config("config/")
    cfg.deletion_log.directory = tempfile.mkdtemp(prefix="mm-migration-log-")
    return cfg


IS_PG = os.environ.get("MM_DATABASE_URL", "").startswith("postgresql")

pytestmark = [
    pytest.mark.postgres_only("physical composite primary keys only exist on PostgreSQL"),
    pytest.mark.skipif(not IS_PG, reason="needs MM_DATABASE_URL pointing at PostgreSQL"),
]


def _insert_job(conn, job_id: str, purpose: str, tenant: str = "t", app: str = "a") -> None:
    conn.execute(
        text(
            "INSERT INTO mm_evolution_jobs (id, user_id, evidence_watermark, "
            "idempotency_key, status, purpose, tenant_id, app_id, lease_token, "
            "prompt_version, attempt_count, error_code, claimed_revision, "
            "created_at) VALUES (:id, 'u1', '{}', :ik, 'pending', :purpose, "
            ":tenant, :app, 0, '', 0, '', 0, now())"
        ),
        {"id": job_id, "ik": "ik-" + job_id, "purpose": purpose, "tenant": tenant, "app": app},
    )


def _jobs(engine):
    with no_audit_scope(), engine.connect() as conn:
        return conn.execute(
            text("SELECT id, purpose FROM mm_evolution_jobs ORDER BY id")
        ).all()


def test_backfill_column_on_a_composite_pk_table(pg_admin_engine):
    live = sm._live_table(pg_admin_engine, "mm_evolution_jobs")
    assert [c.name for c in live.primary_key.columns] == ["tenant_id", "app_id", "id"]
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        _insert_job(conn, "j0", "")
        _insert_job(conn, "j1", "", tenant="t2")
        _insert_job(conn, "j2", "keep")
    with no_audit_scope():
        stamped = sm._backfill_column(
            pg_admin_engine, "mm_evolution_jobs", {"purpose": "legacy-app"},
            ["purpose"], 1, marker="",
        )
    assert stamped == 2
    assert _jobs(pg_admin_engine) == [("j0", "legacy-app"), ("j1", "legacy-app"), ("j2", "keep")]


def test_step6_runs_against_a_version5_postgres_database(pg_admin_engine):
    """Strip the step-6 artifacts from the suite's schema (which is what a
    database that stopped at version 5 looks like), seed a pre-M3 job, and
    run the step: ADD COLUMN -> batched backfill -> NOT NULL recipe -> RLS
    on the new tables, all on composite-key tables."""
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE mm_purpose_grant_events")
        conn.exec_driver_sql("DROP TABLE mm_purpose_grants")
        conn.exec_driver_sql("ALTER TABLE mm_evolution_jobs DROP COLUMN purpose")
    sm._ensure_version_table(pg_admin_engine)
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        vt = sm._version_table(MetaData())
        conn.execute(vt.delete())
        conn.execute(vt.insert().values(version=5, note="simulated v5"))
        conn.execute(text(
            "INSERT INTO mm_evolution_jobs (id, user_id, evidence_watermark, "
            "idempotency_key, status, tenant_id, app_id, lease_token, "
            "prompt_version, attempt_count, error_code, claimed_revision, "
            "created_at) VALUES ('legacy-job', 'u1', '{}', 'ik-legacy', "
            "'pending', 't', 'a', 0, '', 0, '', 0, now())"
        ))

    step = sm.Step6PurposeAuthorization(legacy_job_purpose="legacy-app")
    assert step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)
    step.verify(pg_admin_engine)

    cols = {c["name"]: c for c in inspect(pg_admin_engine).get_columns("mm_evolution_jobs")}
    assert cols["purpose"]["nullable"] is False
    assert _jobs(pg_admin_engine) == [("legacy-job", "legacy-app")]
    assert step.legacy_jobs_to_cancel == 0
    assert not step.needed(pg_admin_engine)


def test_step7_runs_against_a_version6_postgres_database(pg_admin_engine):
    """M3 batch-2 task 1: ADD COLUMN -> batched backfill (composite keys)
    -> NOT NULL recipe -> SET DEFAULT '' on the three provenance tables;
    with the legacy purpose the pre-M3 rows are labelled and later rows
    still take the marker."""
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        for table in sm.SOURCE_PURPOSE_TABLES:
            conn.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN source_purpose")
        conn.execute(text(
            "INSERT INTO mm_evidence (user_id, session_id, ref, content, source_type, "
            "extraction_method, authority, observed_at, created_at, retention_state, "
            "tenant_id, app_id) VALUES ('u1', 's1', 's1:1', 'c', 'message', '', 'user', "
            "now(), now(), 'hot', 't', 'a')"
        ))
        conn.execute(text(
            "INSERT INTO mm_belief_events (belief_id, user_id, event_type, evidence_json, "
            "detail_json, created_at, tenant_id, app_id) VALUES (1, 'u1', 'created', '[]', "
            "'{}', now(), 't', 'a')"
        ))
        conn.execute(text(
            "INSERT INTO mm_snapshots (id, user_id, version, status, content_json, "
            "support_policy_json, evidence_watermark, prompt_version, created_at, "
            "tenant_id, app_id) VALUES ('snap1', 'u1', 1, 'active', '{}', '{}', '{}', "
            "'v1', now(), 't', 'a')"
        ))
    sm._ensure_version_table(pg_admin_engine)
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        vt = sm._version_table(MetaData())
        conn.execute(vt.delete())
        conn.execute(vt.insert().values(version=6, note="simulated v6"))

    step = sm.Step7SourcePurpose(legacy_job_purpose="legacy-app")
    assert step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)
    step.verify(pg_admin_engine)
    assert step.rows_labelled == {"mm_evidence": 1, "mm_belief_events": 1, "mm_snapshots": 1}

    insp = inspect(pg_admin_engine)
    for table in sm.SOURCE_PURPOSE_TABLES:
        col = {c["name"]: c for c in insp.get_columns(table)}["source_purpose"]
        assert col["nullable"] is False, table
        assert "''" in str(col["default"]), (table, col["default"])
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO mm_snapshots (id, user_id, version, status, content_json, "
            "support_policy_json, evidence_watermark, prompt_version, created_at, "
            "tenant_id, app_id) VALUES ('snap2', 'u1', 2, 'active', '{}', '{}', '{}', "
            "'v1', now(), 't', 'a')"
        ))
        rows = conn.execute(text(
            "SELECT id, source_purpose FROM mm_snapshots ORDER BY id"
        )).all()
    assert rows == [("snap1", "legacy-app"), ("snap2", "")]
    assert not step.needed(pg_admin_engine)


def test_step8_creates_the_audit_table_with_rls_on_postgres(pg_admin_engine):
    """M3 batch-2 task 2: the new table gets the composite key and the RLS
    posture (ENABLE + FORCE, both policies), and the app role sees only
    its tenant's rows."""
    from sqlalchemy.orm import Session

    from mirror_memory.core import rls
    from mirror_memory.core.models import AccessAudit
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from tests.conftest import _pg_runtime_url

    with no_audit_scope(), pg_admin_engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE mm_access_audit")
    sm._ensure_version_table(pg_admin_engine)
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        vt = sm._version_table(MetaData())
        conn.execute(vt.delete())
        conn.execute(vt.insert().values(version=7, note="simulated v7"))
    step = sm.Step8AccessAudit()
    assert step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)
    step.verify(pg_admin_engine)
    rls.verify_posture(pg_admin_engine, tables=["mm_access_audit"])
    live = sm._live_table(pg_admin_engine, "mm_access_audit")
    assert [c.name for c in live.primary_key.columns] == ["tenant_id", "app_id", "id"]

    runtime = create_engine(_pg_runtime_url())
    try:
        for tenant in (("tA", "app"), ("tB", "app")):
            with tenant_scope(*tenant), fixture_write_scope("seed audit rows"):
                s = Session(runtime)
                s.add(AccessAudit(user_id="u1", purpose="p", operation="recall",
                                  entry="e", decision="allowed"))
                s.commit()
                s.close()
        with tenant_scope("tA", "app"):
            s = Session(runtime)
            rows = s.query(AccessAudit).all()
            assert [(r.tenant_id, r.app_id) for r in rows] == [("tA", "app")]
            s.close()
    finally:
        runtime.dispose()


def test_step9_swaps_belief_indexes_concurrently_and_stamps_summaries(pg_admin_engine):
    """M3 batch 3 step 9 on PostgreSQL: CONCURRENTLY create the
    tenant-prefixed belief indexes, drop the bare ones, and give
    mm_session_summaries its provenance column (legacy labels via the
    shared parameter; later rows take the marker default)."""
    from sqlalchemy import text

    from mirror_memory.core import schema_migration as sm
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from mirror_memory.core.models import SessionSummary
    from sqlalchemy.orm import Session
    from tests.conftest import _pg_runtime_url

    # simulate v8: drop the model's composite indexes (they exist from
    # create_all), create the legacy bare ones, drop the summary column
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        for table, composite in sm.TENANT_BELIEF_INDEXES.items():
            conn.exec_driver_sql(f"DROP INDEX {composite}")
            conn.exec_driver_sql(
                f"CREATE INDEX {sm.LEGACY_BELIEF_INDEXES[table]} ON {table} (belief_id)")
        conn.exec_driver_sql("ALTER TABLE mm_session_summaries DROP COLUMN source_purpose")
        conn.execute(text(
            "INSERT INTO mm_session_summaries (user_id, session_id, summary_text, "
            "turn_count, created_at, tenant_id, app_id) VALUES ('u1', 's-old', 'old', 1, "
            "now(), 'tA', 'app')"
        ))
    sm._ensure_version_table(pg_admin_engine)
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        vt = sm._version_table(MetaData())
        conn.execute(vt.delete())
        conn.execute(vt.insert().values(version=8, note="simulated v8"))

    step = sm.Step9RecallVisibility(legacy_job_purpose="legacy-app")
    assert step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)  # CONCURRENTLY: outside transactions
    step.verify(pg_admin_engine)

    insp = inspect(pg_admin_engine)
    for table, composite in sm.TENANT_BELIEF_INDEXES.items():
        names = {ix["name"] for ix in insp.get_indexes(table)}
        assert composite in names and sm.LEGACY_BELIEF_INDEXES[table] not in names
    assert "source_purpose" in {c["name"] for c in insp.get_columns("mm_session_summaries")}

    runtime = create_engine(_pg_runtime_url())
    try:
        # a row written after the labelled migration carries the marker
        step2 = sm.Step9RecallVisibility()
        assert not step2.needed(pg_admin_engine)
        step2.apply(pg_admin_engine)
        with tenant_scope("tA", "app"), fixture_write_scope("seed one more summary"):
            s = Session(runtime)
            s.add(SessionSummary(user_id="u1", session_id="s-new", summary_text="new"))
            s.commit()
            s.close()
        # read through the owner connection (the maintenance policy lets it
        # see every tenant's rows; the app role would need the GUC)
        with no_audit_scope(), pg_admin_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT session_id, source_purpose FROM mm_session_summaries ORDER BY session_id")).all()
        assert rows == [("s-new", ""), ("s-old", "legacy-app")]
    finally:
        runtime.dispose()


def test_v5_database_upgrades_to_latest_with_interrupted_step(pg_admin_engine):
    """Task 4: a REAL version-5 PostgreSQL database (composite-key tables
    with pre-M3 data) upgrades to the latest version; an interruption
    between steps leaves the recorded version at the last completed step
    and the re-run finishes the job without relabelling."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from mirror_memory.core.models import SessionSummary
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from tests.conftest import _pg_runtime_url

    with no_audit_scope(), pg_admin_engine.begin() as conn:
        # strip everything steps 6-9 created (what a v5 database lacks)
        conn.exec_driver_sql("DROP TABLE mm_purpose_grant_events")
        conn.exec_driver_sql("DROP TABLE mm_purpose_grants")
        conn.exec_driver_sql("DROP TABLE mm_access_audit")
        conn.exec_driver_sql("ALTER TABLE mm_evolution_jobs DROP COLUMN purpose")
        for table in sm.SOURCE_PURPOSE_TABLES:
            conn.exec_driver_sql(f"ALTER TABLE {table} DROP COLUMN source_purpose")
        for table, composite in sm.TENANT_BELIEF_INDEXES.items():
            conn.exec_driver_sql(f"DROP INDEX {composite}")
            conn.exec_driver_sql(
                f"CREATE INDEX {sm.LEGACY_BELIEF_INDEXES[table]} ON {table} (belief_id)")
        conn.exec_driver_sql("ALTER TABLE mm_session_summaries DROP COLUMN source_purpose")
        # pre-M3 data on composite-key tables
        conn.execute(text(
            "INSERT INTO mm_evolution_jobs (id, user_id, evidence_watermark, "
            "idempotency_key, status, tenant_id, app_id, lease_token, "
            "prompt_version, attempt_count, error_code, claimed_revision, created_at) "
            "VALUES ('j1', 'u1', '{}', 'ik1', 'pending', 'tA', 'app', 0, '', 0, '', 0, now())"
        ))
        conn.execute(text(
            "INSERT INTO mm_evidence (user_id, session_id, ref, content, source_type, "
            "extraction_method, authority, observed_at, created_at, retention_state, "
            "tenant_id, app_id) VALUES ('u1', 's1', 's1:1', 'coffee', 'message', '', "
            "'user', now(), now(), 'hot', 'tA', 'app')"
        ))
        conn.execute(text(
            "INSERT INTO mm_belief_events (belief_id, user_id, event_type, evidence_json, "
            "detail_json, created_at, tenant_id, app_id) VALUES (1, 'u1', 'created', '[]', "
            "'{}', now(), 'tA', 'app')"
        ))
        conn.execute(text(
            "INSERT INTO mm_session_summaries (user_id, session_id, summary_text, "
            "turn_count, created_at, tenant_id, app_id) VALUES ('u1', 's-old', 'old', 1, "
            "now(), 'tA', 'app')"
        ))
    sm._ensure_version_table(pg_admin_engine)
    with no_audit_scope(), pg_admin_engine.begin() as conn:
        vt = sm._version_table(MetaData())
        conn.execute(vt.delete())
        conn.execute(vt.insert().values(version=5, note="pre-M3 production shape"))

    # interrupt: run only through step 6 (as if the process died after it)
    upgrade_database(pg_admin_engine, default_tenant="tA", default_app="app",
                     steps=sm.DEFAULT_STEPS[:6], confirm_backup=True)
    assert sm.current_version(pg_admin_engine) == 6
    # the job carries the legacy label; provenance tables still lack the column
    with no_audit_scope(), pg_admin_engine.connect() as conn:
        purpose = conn.execute(text(
            "SELECT purpose FROM mm_evolution_jobs WHERE id = 'j1'")).scalar_one()
    assert purpose == "", "step 6 ran without a legacy purpose: the pre-M3 job keeps the marker"

    # the interrupted re-run: everything from 7 to latest, WITH the shared
    # legacy choice (explicit list -- swapping steps[-1] would drop step 10
    # and duplicate version 9)
    steps = list(sm.DEFAULT_STEPS[6:])
    steps[0] = sm.Step7SourcePurpose(legacy_job_purpose="legacy-app")
    steps[2] = sm.Step9RecallVisibility(legacy_job_purpose="legacy-app")
    upgrade_database(pg_admin_engine, default_tenant="tA", default_app="app",
                     steps=steps, confirm_backup=True)
    assert sm.current_version(pg_admin_engine) == sm.LATEST_VERSION

    with no_audit_scope(), pg_admin_engine.connect() as conn:
        evidence_purpose, event_purpose, summary_missing = conn.execute(text(
            "SELECT (SELECT source_purpose FROM mm_evidence), "
            "(SELECT source_purpose FROM mm_belief_events), "
            "(SELECT count(*) FROM information_schema.columns "
            " WHERE table_name = 'mm_session_summaries' AND column_name = 'source_purpose')"
        )).one()
    assert (evidence_purpose, event_purpose, summary_missing) == ("legacy-app", "legacy-app", 1)

    # the upgraded database works under RLS: write + read through the app role
    runtime = create_engine(_pg_runtime_url())
    try:
        with tenant_scope("tA", "app"), fixture_write_scope("seed through app role"):
            s = Session(runtime)
            s.add(SessionSummary(user_id="u1", session_id="s-new", summary_text="after"))
            s.commit()
            s.close()
        with no_audit_scope(), pg_admin_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT session_id, source_purpose FROM mm_session_summaries ORDER BY session_id")).all()
        # the pre-M3 summary row got the legacy label; the post-migration
        # one carries the marker
        assert rows == [("s-new", ""), ("s-old", "legacy-app")]
    finally:
        runtime.dispose()
