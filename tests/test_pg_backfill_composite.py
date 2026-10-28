"""M3 batch-2 task 1 finding: step 6 could not run on a real PostgreSQL
version-5 database.

After step 2 every tenant table on PostgreSQL has the physical composite
primary key ``(tenant_id, app_id, id)``, and ``_backfill_column`` refused
anything but a single-column key -- so ``Step6PurposeAuthorization.apply``
raised ``MigrationError`` on the ``mm_evolution_jobs.purpose`` backfill the
moment it ran against a database that had actually been through steps
1-5.  The SQLite leg never saw it: its step-6 path is ``ADD COLUMN ... NOT
NULL DEFAULT`` with no backfill, and the migration fixtures build the
legacy database on SQLite.  These tests run the helper and the whole step
against the PostgreSQL schema the suite builds (composite keys, RLS on).
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import MetaData, inspect, text

from mirror_memory.core import schema_migration as sm
from mirror_memory.core.sql_audit import no_audit_scope

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
