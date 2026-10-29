"""M2 batch-2 task 1: PostgreSQL row-level security -- the second defence.

The six groups the task brief requires:

1. read injection off (``MM_TENANT_READ_FILTER=rls``): RLS alone hides
   other tenants' reads and blocks cross-tenant writes;
2. no tenant setting: reads return 0 rows, writes fail (fail closed);
3. pooled-connection reuse: tenant A's per-transaction setting cannot
   survive into tenant B's transaction on the same physical connection;
4. one Session across commit / rollback: the setting is re-applied to
   every new transaction;
5. the ``system_scope`` call sites work on the maintenance role, the app
   role cannot switch the defence off, the maintenance role is a policy
   (not BYPASSRLS), and the owner is subject to FORCE like everyone else;
6. the auditor / write-guard / isolation suites pass with RLS on: the
   whole PostgreSQL leg of the suite runs through the app role against an
   RLS-protected schema (conftest), so the dual-db run is that proof.

The DB-layer probes deliberately bypass the application's read injection
and run under ``no_audit_scope()``: the auditor checks the application's
statements, these tests check what the database itself admits.  The
auditor's own new rules (hard violations for re-pointing the tenant
setting) are tested here as well; they are dialect-independent.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, delete, func, insert, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.schema import CreateIndex, DropIndex
from sqlalchemy.orm import Session, sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Belief, EvolutionJob
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.sql_audit import AuditViolation, _dialect_var, audit_sql, no_audit_scope
from mirror_memory.core.tenancy import (
    TenantContextMissing,
    TenantError,
    fixture_write_scope,
    no_tenant_scope,
    read_filter_mode,
    read_filter_override,
    reset_read_filter_cache,
    system_scope,
    tenant_scope,
)
from mirror_memory.worker.evolution import enqueue_job, process_pending


def _worker_config():
    class FakeLLM:
        def generate(self, **kw):
            return '{"patterns": [], "support_policy": {}}'

    cfg = load_config("config/")
    cfg.llm_client = FakeLLM()
    return cfg


@pytest.fixture
def worker_config():
    return _worker_config()


def _seed_tenant(app_url, tenant, user, config, *, jobs: bool = True) -> None:
    """Seed one belief (and optionally an evolution job) for one tenant
    through the app role -- exactly the runtime path."""
    engine = create_engine(app_url)
    with tenant_scope(*tenant):
        session = Session(engine)
        try:
            set_memory_enabled(session, user, True)
            record_claim(
                session,
                user,
                dimension="topic",
                key="sleep",
                claim_text=f"{tenant[0]} belief",
                confidence=0.8,
                session_id="s1",
            )
            if jobs:
                assert enqueue_job(session, user, config, purpose="test") is not None
            session.commit()
        finally:
            session.close()
    engine.dispose()


@pytest.fixture
def two_tenants(pg_app_url, worker_config):
    """One user with a belief and a pending job in each of tA and tB."""
    _seed_tenant(pg_app_url, ("tA", "app"), "uA", worker_config)
    _seed_tenant(pg_app_url, ("tB", "app"), "uB", worker_config)
    return pg_app_url


def _count_all(conn) -> int:
    return conn.execute(select(func.count()).select_from(Belief.__table__)).scalar()


def _setting(conn) -> str | None:
    return conn.execute(text("select current_setting('mm.tenant_id', true)")).scalar()


# ---------------------------------------------------------------------------
# 1. Read injection off: RLS alone
# ---------------------------------------------------------------------------


def test_with_read_injection_off_rls_hides_and_blocks_other_tenants(two_tenants):
    engine = create_engine(two_tenants)
    beliefs = Belief.__table__
    try:
        with read_filter_override("rls"), no_audit_scope(), tenant_scope("tB", "app"):
            session = Session(engine)
            # ORM read with the injection skipped: only tB's row comes back,
            # including the Query.count() shape (column-only top-level
            # SELECT over a subquery -- the M1 blind spot).
            assert [r.tenant_id for r in session.query(Belief).all()] == ["tB"]
            assert session.query(Belief).count() == 1
            # Core read addressed at the other tenant: 0 rows.
            assert session.execute(
                select(func.count()).select_from(beliefs).where(beliefs.c.tenant_id == "tA")
            ).scalar() == 0
            # Cross-tenant UPDATE / DELETE affect 0 rows.
            assert session.execute(
                update(beliefs).where(beliefs.c.tenant_id == "tA").values(confidence=0.1)
            ).rowcount == 0
            assert session.execute(
                delete(beliefs).where(beliefs.c.tenant_id == "tA")
            ).rowcount == 0
            # Cross-tenant INSERT is refused outright (WITH CHECK).
            with pytest.raises(DBAPIError, match="row-level security"):
                session.execute(
                    insert(beliefs).values(
                        tenant_id="tA", app_id="app", user_id="uA", dimension="topic",
                        key="sneak", claim_text="planted", confidence=0.9,
                    )
                )
            session.rollback()
            session.close()
        # tA's data is intact, read back through the normal app path.
        with tenant_scope("tA", "app"):
            session = Session(engine)
            assert [b.claim_text for b in session.query(Belief).all()] == ["tA belief"]
            assert session.query(Belief).one().confidence == 0.8
            session.close()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# 2. No tenant setting at all
# ---------------------------------------------------------------------------


def test_no_tenant_setting_reads_zero_rows_and_writes_fail(two_tenants):
    engine = create_engine(two_tenants)
    beliefs = Belief.__table__
    try:
        with no_tenant_scope(), no_audit_scope():
            with engine.connect() as conn:
                assert _setting(conn) in ("", None)
                assert _count_all(conn) == 0
                assert conn.execute(
                    select(beliefs.c.id).where(beliefs.c.tenant_id == "tA")
                ).first() is None
                with pytest.raises(DBAPIError, match="row-level security"):
                    conn.execute(
                        insert(beliefs).values(
                            tenant_id="tA", app_id="app", user_id="uA",
                            dimension="topic", key="k", claim_text="x",
                        )
                    )
                conn.rollback()
                assert conn.execute(
                    update(beliefs).where(beliefs.c.tenant_id == "tA").values(confidence=0.1)
                ).rowcount == 0
                assert conn.execute(
                    delete(beliefs).where(beliefs.c.tenant_id == "tA")
                ).rowcount == 0
        # The application layer still refuses tenant statements without a
        # context at all (unchanged by RLS) -- two independent defences.
        with no_tenant_scope():
            session = Session(engine)
            with pytest.raises(TenantContextMissing):
                session.query(Belief).all()
            session.close()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# 3. Pool reuse -- one physical connection, two tenants in sequence
# ---------------------------------------------------------------------------


def test_pooled_connection_reuse_cannot_leak_a_tenant_setting(two_tenants):
    engine = create_engine(two_tenants, pool_size=1, max_overflow=0)
    pids = []
    try:
        with read_filter_override("rls"), no_audit_scope():
            with tenant_scope("tA", "app"):
                session = Session(engine)
                pids.append(session.execute(text("select pg_backend_pid()")).scalar())
                assert session.query(Belief).count() == 1
                assert _setting(session) == "tA"
                session.commit()
                session.close()
            with tenant_scope("tB", "app"):
                session = Session(engine)
                pids.append(session.execute(text("select pg_backend_pid()")).scalar())
                assert pids[0] == pids[1]  # the same physical connection
                assert _setting(session) == "tB"
                assert [r.tenant_id for r in session.query(Belief).all()] == ["tB"]
                session.close()
            # Checked in again with no tenant: the setting did not survive
            # (transaction-local; PostgreSQL discards it, not Python).
            with no_tenant_scope():
                with engine.connect() as conn:
                    assert conn.execute(text("select pg_backend_pid()")).scalar() == pids[0]
                    assert _setting(conn) in ("", None)
                    assert _count_all(conn) == 0
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# 4. One Session across commit / rollback
# ---------------------------------------------------------------------------


def test_session_keeps_the_tenant_setting_across_commit_and_rollback(two_tenants):
    engine = create_engine(two_tenants)
    try:
        with read_filter_override("rls"), no_audit_scope(), tenant_scope("tA", "app"):
            session = Session(engine)
            assert session.query(Belief).count() == 1
            session.commit()
            assert session.query(Belief).count() == 1  # new transaction, re-set
            assert _setting(session) == "tA"
            session.rollback()
            assert session.query(Belief).count() == 1
            with fixture_write_scope("rls test: second belief after commit"):
                session.add(Belief(user_id="uA", dimension="topic", key="second",
                                   claim_text="tA second", confidence=0.5))
            session.commit()
            assert session.query(Belief).count() == 2
            session.close()
        with tenant_scope("tB", "app"):
            session = Session(engine)
            assert session.query(Belief).count() == 1  # tB saw none of it
            session.close()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# 5. system_scope call sites on the maintenance role; nobody can switch off
# ---------------------------------------------------------------------------


def test_worker_scan_sees_every_tenant_via_the_maintenance_role(two_tenants, worker_config):
    engine = create_engine(two_tenants)
    factory = sessionmaker(bind=engine)
    try:
        process_pending(factory, worker_config)
        # Both tenants' jobs were scanned, claimed in their own tenant and
        # completed.  This final read runs under system_scope too, which
        # routes the fresh session to the maintenance engine.
        with system_scope("rls test: read final job states across tenants"):
            session = factory()
            try:
                assert session.get_bind() is not engine  # routed, not the app engine
                jobs = session.query(EvolutionJob).all()
                assert {(j.tenant_id, j.user_id, j.status) for j in jobs} == {
                    ("tA", "uA", "completed"),
                    ("tB", "uB", "completed"),
                }
            finally:
                session.close()
    finally:
        engine.dispose()


def test_maintenance_role_is_a_policy_not_bypassrls(pg_app_url, pg_maintenance_url):
    _seed_tenant(pg_app_url, ("tA", "app"), "uA", _worker_config(), jobs=False)
    _seed_tenant(pg_app_url, ("tB", "app"), "uB", _worker_config(), jobs=False)
    engine = create_engine(pg_maintenance_url)
    try:
        with system_scope("rls test: maintenance full-access probe"), no_audit_scope():
            with engine.connect() as conn:
                assert _count_all(conn) == 2
                attrs = conn.execute(text(
                    "SELECT rolsuper, rolbypassrls FROM pg_catalog.pg_roles "
                    "WHERE rolname = current_user"
                )).first()
                assert tuple(attrs) == (False, False)
    finally:
        engine.dispose()


def test_owner_is_subject_to_force_and_admitted_only_by_the_policy(pg_admin_engine, pg_app_url):
    """FORCE ROW LEVEL SECURITY applies to the table owner: with the
    maintenance policy dropped the owner sees nothing; with it back, all."""
    from psycopg import sql

    from mirror_memory.core import rls
    from mirror_memory.core.schema_migration import install_row_level_security

    _seed_tenant(pg_app_url, ("tA", "app"), "uA", _worker_config(), jobs=False)
    _seed_tenant(pg_app_url, ("tB", "app"), "uB", _worker_config(), jobs=False)
    owner = rls.owner_role()
    with no_audit_scope():
        with pg_admin_engine.begin() as conn:
            conn.execute(rls.DropPolicy("mm_beliefs", rls.MAINTENANCE_POLICY))
        # SET LOCAL ROLE is transaction-scoped: the rollback below restores
        # the superuser identity before the connection returns to the pool.
        with pg_admin_engine.connect() as conn:
            conn.connection.driver_connection.execute(
                sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(owner))
            )
            assert conn.execute(text("select current_user")).scalar() == owner
            assert _count_all(conn) == 0  # subject to the policies, none admits it
            conn.rollback()
        install_row_level_security(pg_admin_engine)  # re-entrant repair
        with pg_admin_engine.connect() as conn:
            conn.connection.driver_connection.execute(
                sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(owner))
            )
            assert conn.execute(text("select current_user")).scalar() == owner
            assert _count_all(conn) == 2
            conn.rollback()


def test_single_tenant_startup_refuses_a_multi_tenant_database(pg_app_url):
    """The startup probe runs on the maintenance engine: the app role alone
    would see one tenant and pass vacuously."""
    from mirror_memory.api import MemoryEngine

    _seed_tenant(pg_app_url, ("tA", "app"), "uA", _worker_config(), jobs=False)
    _seed_tenant(pg_app_url, ("tB", "app"), "uB", _worker_config(), jobs=False)
    engine = MemoryEngine(config=load_config("config/"), database_url=pg_app_url,
                          tenant_id="tA", app_id="app", single_tenant_mode=True)
    with pytest.raises(RuntimeError, match="multiple tenants"):
        engine._ensure_db()


def test_single_tenant_startup_without_maintenance_registration_fails_closed(
    pg_admin_engine, pg_app_url, pg_maintenance_url, monkeypatch
):
    from mirror_memory.api import MemoryEngine
    from mirror_memory.core import rls

    _seed_tenant(pg_app_url, ("tA", "app"), "uA", _worker_config(), jobs=False)
    monkeypatch.delenv("MM_MAINTENANCE_DATABASE_URL", raising=False)
    rls.clear_maintenance_registry()
    try:
        engine = MemoryEngine(config=load_config("config/"), database_url=pg_app_url,
                              tenant_id="tA", app_id="app", single_tenant_mode=True)
        with pytest.raises(rls.RowSecurityError, match="maintenance"):
            engine._ensure_db()
    finally:
        rls.register_maintenance_url(pg_app_url, pg_maintenance_url)
        rls.register_maintenance_url(
            pg_admin_engine.url.render_as_string(hide_password=False), pg_maintenance_url
        )


def test_runtime_engine_refuses_the_superuser_login(pg_admin_engine):
    """A superuser bypasses every policy silently; an engine on that login
    refuses to serve rather than run with the second defence inert."""
    from mirror_memory.api import MemoryEngine
    from mirror_memory.core import rls

    engine = MemoryEngine(
        config=load_config("config/"),
        database_url=pg_admin_engine.url.render_as_string(hide_password=False),
        tenant_id="tA", app_id="app",
    )
    with pytest.raises(rls.RowSecurityError, match="superuser"):
        engine._ensure_db()


def test_app_role_cannot_switch_row_level_security_off(pg_app_url):
    engine = create_engine(pg_app_url)
    try:
        with engine.connect() as conn:
            for ddl in (
                "ALTER TABLE mm_beliefs DISABLE ROW LEVEL SECURITY",
                "ALTER TABLE mm_beliefs NO FORCE ROW LEVEL SECURITY",
                "DROP POLICY mm_tenant_rows ON mm_beliefs",
                "DROP POLICY mm_maintenance_rows ON mm_beliefs",
            ):
                with pytest.raises(DBAPIError, match="must be owner"):
                    conn.execute(text(ddl))
                conn.rollback()
    finally:
        engine.dispose()


def test_migration_runs_as_owner_over_every_tenant(pg_admin_engine, pg_app_url):
    """The five migration call sites are whole-table operations on the
    migration login; with RLS installed they still see every row and the
    plan records the latest version."""
    from mirror_memory.core import rls
    from mirror_memory.core.schema_migration import (
        LATEST_VERSION,
        current_version,
        upgrade_database,
    )

    _seed_tenant(pg_app_url, ("tA", "app"), "uA", _worker_config(), jobs=False)
    _seed_tenant(pg_app_url, ("tB", "app"), "uB", _worker_config(), jobs=False)
    before = upgrade_database(
        pg_admin_engine, default_tenant="legacy", default_app="app", confirm_backup=True
    )
    assert before["mm_beliefs"] == 2  # _count_rows under system_scope saw both tenants
    assert current_version(pg_admin_engine) == LATEST_VERSION
    rls.verify_posture(pg_admin_engine)


# ---------------------------------------------------------------------------
# Migration step 3: needed / apply / verify, re-entrant
# ---------------------------------------------------------------------------


def test_step3_needed_flips_and_apply_is_reentrant(pg_admin_engine):
    from mirror_memory.core import rls
    from mirror_memory.core.schema_migration import Step3RowLevelSecurity

    step = Step3RowLevelSecurity()
    assert not step.needed(pg_admin_engine)  # the fixture already installed it
    with pg_admin_engine.begin() as conn:
        conn.execute(text("ALTER TABLE mm_beliefs NO FORCE ROW LEVEL SECURITY"))
        conn.execute(rls.DropPolicy("mm_evidence", rls.TENANT_POLICY))
    assert step.needed(pg_admin_engine)
    with pytest.raises(rls.RowSecurityError, match="mm_beliefs"):
        step.verify(pg_admin_engine)
    step.apply(pg_admin_engine)
    step.verify(pg_admin_engine)
    assert not step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)  # a second run is a no-op repair
    step.verify(pg_admin_engine)


def test_posture_verification_rejects_a_bypassrls_app_role(pg_admin_engine):
    import psycopg
    from psycopg import sql

    from mirror_memory.core import rls

    app = rls.app_role()
    conninfo = pg_admin_engine.url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    with psycopg.connect(conninfo, autocommit=True) as admin:
        try:
            admin.execute(sql.SQL("ALTER ROLE {} BYPASSRLS").format(sql.Identifier(app)))
            with pytest.raises(rls.RowSecurityError, match="bypassrls"):
                rls.verify_posture(pg_admin_engine)
        finally:
            admin.execute(sql.SQL("ALTER ROLE {} NOBYPASSRLS").format(sql.Identifier(app)))
    rls.verify_posture(pg_admin_engine)


# ---------------------------------------------------------------------------
# Dialect-independent: auditor hard violations, read-filter modes
# ---------------------------------------------------------------------------


def _audit_as(sql: str, dialect: str = "postgresql"):
    token = _dialect_var.set(dialect)
    try:
        audit_sql(sql)
        return None
    except AuditViolation as exc:
        return exc
    finally:
        _dialect_var.reset(token)


def test_auditor_exempts_only_the_engines_own_tenant_setting():
    from mirror_memory.core.rls import SET_TENANT_SQL

    assert _audit_as(SET_TENANT_SQL) is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT set_config('mm.tenant_id', 'tB', false)",
        "SELECT set_config('mm.tenant_id', %(x)s, true), set_config('mm.app_id', %(y)s, true), 1",
        "SELECT set_config('mm.app_id', 'other', true)",
        "SET mm.tenant_id = 'tB'",
        "SET LOCAL mm.tenant_id = 'tB'",
        "SET SESSION mm.app_id = 'x'",
        "SET ROLE mm_maintenance",
        "SET LOCAL ROLE mm_maintenance",
        "RESET ROLE",
        "SET SESSION AUTHORIZATION postgres",
        "SET row_security = off",
        "SET LOCAL row_security TO off",
        "SELECT set_config('row_security', 'off', true)",
        "SELECT 1; SET ROLE mm_maintenance",
        "SELECT 1; SET LOCAL mm.tenant_id = 'tB'",
    ],
)
def test_auditor_hard_violation_for_rls_bypass_vectors(sql):
    exc = _audit_as(sql)
    assert exc is not None and "row-level security" in str(exc)


@pytest.mark.parametrize(
    "sql",
    [
        'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE "mm_beliefs" TO "mm_app", "mm_maintenance"',
        'GRANT USAGE ON SCHEMA "public" TO "mm_app"',
        'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA "public" TO "mm_app"',
        'CREATE POLICY "mm_tenant_rows" ON "mm_beliefs" FOR ALL TO "mm_app" USING (tenant_id = current_setting(\'mm.tenant_id\', true)) WITH CHECK (true)',
        'DROP POLICY IF EXISTS "mm_tenant_rows" ON "mm_beliefs"',
        'CREATE ROLE "mm_app" LOGIN',
        'ALTER TABLE "mm_beliefs" OWNER TO "mm_owner"',
        'ALTER TABLE "mm_beliefs" ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY',
    ],
)
def test_auditor_exempts_single_statement_posture_ddl(sql):
    assert _audit_as(sql) is None


def test_auditor_still_rejects_smuggling_behind_posture_ddl():
    assert _audit_as('GRANT SELECT ON TABLE "mm_beliefs" TO "mm_app"; SELECT * FROM mm_beliefs') is not None
    assert _audit_as('DROP POLICY "p" ON "mm_beliefs"; SELECT * FROM mm_beliefs') is not None


def test_read_filter_rls_mode_keeps_injection_on_sqlite(db_session):
    """RLS cannot carry the read filter on SQLite, so the rls mode falls back
    to the application-layer injection there (the stronger filter), instead
    of silently reading unfiltered or refusing to run."""
    if db_session.get_bind().dialect.name != "sqlite":
        pytest.skip("the fallback is for binds without row-level security")
    with fixture_write_scope("rls fallback seed"):
        db_session.add(Belief(user_id="u", dimension="d", key="mine", claim_text="default's", confidence=0.5))
        db_session.commit()
        with tenant_scope("tOther", "app"):
            other = Session(db_session.get_bind())
            other.add(Belief(user_id="u", dimension="d", key="theirs", claim_text="other's", confidence=0.5))
            other.commit()
            other.close()
    with read_filter_override("rls"):
        assert [b.key for b in db_session.query(Belief).all()] == ["mine"]
        assert db_session.query(Belief).count() == 1


def test_read_filter_none_is_measurement_only(monkeypatch):
    try:
        monkeypatch.setenv("MM_TENANT_READ_FILTER", "none")
        monkeypatch.delenv("MM_ALLOW_UNFILTERED_MEASUREMENT", raising=False)
        reset_read_filter_cache()
        with pytest.raises(TenantError, match="measurement"):
            read_filter_mode()
        monkeypatch.setenv("MM_ALLOW_UNFILTERED_MEASUREMENT", "1")
        reset_read_filter_cache()
        assert read_filter_mode() == "none"
        monkeypatch.setenv("MM_TENANT_READ_FILTER", "bogus")
        reset_read_filter_cache()
        with pytest.raises(TenantError, match="MM_TENANT_READ_FILTER"):
            read_filter_mode()
    finally:
        reset_read_filter_cache()


def test_system_engine_for_is_identity_without_postgres():
    from mirror_memory.core.rls import system_engine_for

    engine = create_engine("sqlite://")
    assert system_engine_for(engine) is engine
    engine.dispose()


def test_server_refuses_to_start_without_a_tenant_read_filter(tmp_path):
    from fastapi.testclient import TestClient

    from mirror_memory.server import create_app

    app = create_app(config=load_config("config/"), database_url=f"sqlite:///{tmp_path}/s.db")
    with read_filter_override("none"), pytest.raises(RuntimeError, match="measurement"):
        with TestClient(app):
            pass


# ---------------------------------------------------------------------------
# Migration step 4 on PostgreSQL: CONCURRENTLY build, re-entrant
# ---------------------------------------------------------------------------


def test_step4_builds_the_pending_index_concurrently_and_reentrantly(pg_admin_engine):
    from sqlalchemy.schema import DropIndex

    from mirror_memory.core import schema_migration as sm

    step = sm.Step4PendingJobsIndex()
    assert sm._index_state(pg_admin_engine, "mm_evolution_jobs", sm.PENDING_JOBS_INDEX) == "valid"
    assert not step.needed(pg_admin_engine)
    # a database from before the index: drop it, the step restores it
    with pg_admin_engine.begin() as conn:
        conn.execute(DropIndex(sm._pending_jobs_index(concurrently=False), if_exists=True))
    assert sm._index_state(pg_admin_engine, "mm_evolution_jobs", sm.PENDING_JOBS_INDEX) is None
    assert step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)
    step.verify(pg_admin_engine)
    assert not step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)  # second run: no-op repair, still valid
    step.verify(pg_admin_engine)
    with pg_admin_engine.connect() as conn:
        definition = conn.execute(text(
            "SELECT pg_catalog.pg_get_indexdef(i.indexrelid) FROM pg_catalog.pg_index AS i "
            "JOIN pg_catalog.pg_class AS c ON c.oid = i.indexrelid WHERE c.relname = :name"
        ), {"name": sm.PENDING_JOBS_INDEX}).scalar_one()
    assert "WHERE" in definition and "pending" in definition and "created_at" in definition


# ---------------------------------------------------------------------------
# M2 batch-2 patch task 2: the migration is not bound by statement_timeout
# ---------------------------------------------------------------------------


def test_migration_runs_with_statement_timeout_zero_and_restores_it(pg_admin_engine, monkeypatch):
    """The owner-side engine carries the application's statement_timeout
    (1 s here, via build_engine).  Inside upgrade_database every connection
    the migration checks out runs with statement_timeout = 0 -- session
    level, so CONCURRENTLY builds outside a transaction are covered -- and
    the setting is reset when the connection returns to the pool."""
    from mirror_memory.core.pg_engine import build_engine
    from mirror_memory.core.schema_migration import upgrade_database

    monkeypatch.setenv("MM_PG_STATEMENT_TIMEOUT_MS", "1000")
    engine = build_engine(pg_admin_engine.url.render_as_string(hide_password=False))
    seen: dict[str, str] = {}

    class ProbeStep:
        version = 99
        name = "probe: statement_timeout inside the migration"

        def needed(self, eng):
            return True

        def apply(self, eng):
            with eng.connect() as conn:
                seen["during"] = conn.execute(text("SHOW statement_timeout")).scalar()
                # longer than the application's 1 s limit: must not be cancelled
                conn.execute(text("SELECT pg_sleep(1.3)"))
            # the CONCURRENTLY path runs on an autocommit connection
            with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                seen["autocommit"] = conn.execute(text("SHOW statement_timeout")).scalar()
                seen["lock_timeout_default"] = conn.execute(text("SHOW lock_timeout")).scalar()

        def verify(self, eng):
            pass

    try:
        with engine.connect() as conn:
            assert conn.execute(text("SHOW statement_timeout")).scalar() == "1s"
        upgrade_database(
            engine, default_tenant="t", default_app="a", confirm_backup=True, steps=[ProbeStep()]
        )
        assert seen["during"] == "0" and seen["autocommit"] == "0", seen
        # lock_timeout stays whatever the migration's own DDL sets; the
        # session default (5 s from build_engine) is untouched here
        assert seen["lock_timeout_default"] == "5s", seen
        # every connection the migration used is back in the pool with the
        # application's timeout restored
        with engine.connect() as conn:
            assert conn.execute(text("SHOW statement_timeout")).scalar() == "1s"
        with pytest.raises(DBAPIError, match="statement timeout"):
            with engine.connect() as conn:
                conn.execute(text("SELECT pg_sleep(1.3)"))
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# M2 batch-2 patch task 3: step 5 swaps the user_id indexes CONCURRENTLY
# ---------------------------------------------------------------------------


def test_step5_swaps_user_indexes_concurrently_and_reentrantly(pg_admin_engine):
    from mirror_memory.core import schema_migration as sm
    from mirror_memory.core.models import TENANT_USER_INDEXED_TABLES
    from mirror_memory.core.tenancy import legacy_user_index_name, tenant_user_index_name

    step = sm.Step5TenantUserIndexes()
    assert not step.needed(pg_admin_engine)  # fresh schema already has the composite indexes
    # an older database: single-column index present, composite missing
    new, old = sm._tenant_user_index_pair("mm_belief_events", concurrently=False)
    with pg_admin_engine.begin() as conn:
        conn.execute(DropIndex(new, if_exists=True))
        conn.execute(CreateIndex(old))
    assert step.needed(pg_admin_engine)
    with pytest.raises(sm.MigrationError, match="mm_belief_events"):
        step.verify(pg_admin_engine)
    step.apply(pg_admin_engine)
    step.verify(pg_admin_engine)
    assert sm._index_state(pg_admin_engine, "mm_belief_events", tenant_user_index_name("mm_belief_events")) == "valid"
    assert sm._index_state(pg_admin_engine, "mm_belief_events", legacy_user_index_name("mm_belief_events")) is None
    # an INVALID leftover of a failed CONCURRENTLY build is rebuilt
    with pg_admin_engine.begin() as conn:
        conn.execute(text(
            "UPDATE pg_catalog.pg_index SET indisvalid = false "
            "WHERE indexrelid = (SELECT oid FROM pg_catalog.pg_class WHERE relname = :name)"
        ), {"name": tenant_user_index_name("mm_evidence")})
    assert sm._index_state(pg_admin_engine, "mm_evidence", tenant_user_index_name("mm_evidence")) == "invalid"
    assert step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)
    step.verify(pg_admin_engine)
    assert not step.needed(pg_admin_engine)
    step.apply(pg_admin_engine)  # no-op repair
    step.verify(pg_admin_engine)
    # a replacement, not an addition: the composite index is there, the
    # single-column one is gone, on every converted table
    with pg_admin_engine.connect() as conn:
        for name in TENANT_USER_INDEXED_TABLES:
            rows = set(conn.execute(text(
                "SELECT indexname FROM pg_catalog.pg_indexes WHERE schemaname = current_schema() "
                "AND tablename = :t AND indexname LIKE 'ix\\_%user%'"
            ), {"t": name}).scalars().all())
            assert rows == {tenant_user_index_name(name)}, (name, rows)



# ---------------------------------------------------------------------------
# M2 batch-2 patch task 4: the tenant setting follows the Session's binding
# ---------------------------------------------------------------------------


def _recording_engine(url):
    from sqlalchemy import event

    engine = create_engine(url)
    recorded: list[tuple[str, object]] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _rec(conn, cursor, statement, parameters, context, executemany):
        recorded.append((statement, parameters))

    return engine, recorded


def _tenant_settings(recorded) -> list[str]:
    return [
        p["tenant_id"] for s, p in recorded
        if "set_config('mm.tenant_id'" in s and isinstance(p, dict) and "tenant_id" in p
    ]


def test_session_bound_to_a_refuses_to_run_under_b_without_any_sql(two_tenants):
    """A Session bound to tenant A, driven inside a tenant-B context: every
    way of starting a transaction on it must raise, and no statement may
    run with B's tenant setting -- not the ORM query (already covered by
    the binding check), and not raw text or an explicit connection()."""
    from mirror_memory.core.tenancy import TenantMismatchError

    engine, recorded = _recording_engine(two_tenants)
    try:
        with tenant_scope("tA", "app"):
            session = Session(engine)  # bound to tA at creation
        with tenant_scope("tB", "app"):
            with pytest.raises(TenantMismatchError):
                session.execute(text("select 1"))
            session.rollback()
            with pytest.raises(TenantMismatchError):
                session.connection()
            session.rollback()
            with pytest.raises(TenantMismatchError):
                session.query(Belief).all()
            session.rollback()
        session.close()
        assert "tB" not in _tenant_settings(recorded), _tenant_settings(recorded)
        # and under its own tenant the same session works, with A's setting
        with tenant_scope("tA", "app"):
            assert session.execute(text("select 1")).scalar() == 1
            assert session.query(Belief).count() == 1
            session.close()
        assert _tenant_settings(recorded) and set(_tenant_settings(recorded)) == {"tA"}
    finally:
        engine.dispose()


def test_unbound_session_under_a_tenant_refuses_raw_sql_too(two_tenants):
    from mirror_memory.core.tenancy import TenantMismatchError

    engine, recorded = _recording_engine(two_tenants)
    try:
        with no_tenant_scope():
            session = Session(engine)  # bound to None
        with tenant_scope("tB", "app"):
            with pytest.raises(TenantMismatchError, match="no bound tenant"):
                session.execute(text("select 1"))
            session.rollback()
        session.close()
        assert _tenant_settings(recorded) == []
    finally:
        engine.dispose()


def test_raw_connection_still_takes_the_ambient_tenant(two_tenants):
    """engine.connect() has no Session binding: the ContextVar is the only
    source, as before."""
    engine, recorded = _recording_engine(two_tenants)
    try:
        with tenant_scope("tB", "app"), no_audit_scope():
            with engine.connect() as conn:
                assert _count_all(conn) == 1
        assert _tenant_settings(recorded) == ["tB"]
    finally:
        engine.dispose()
