"""Shared test fixtures for mirror-memory."""

import contextlib
import contextvars

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.models import Base


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_write_fixture_scope: run WITHOUT the blanket test write scope "
        "(used by the write-guard canary tests)",
    )
    config.addinivalue_line(
        "markers",
        "fixture_writes(reason): the test body itself writes protected tables "
        "directly (seeding/tuning); reason is mandatory",
    )
    config.addinivalue_line(
        "markers",
        "sqlite_only(reason): the test exercises embedded-SQLite behaviour "
        "(the file write lock) that PostgreSQL does not have; mandatory "
        "reason, mirrored",
    )
    config.addinivalue_line(
        "markers",
        "postgres_only(reason): the test exercises PostgreSQL catalog "
        "behaviour that SQLite cannot express; mandatory reason, mirrored "
        "on the sqlite_only side by dialect-conditional skips",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    """Open the fixture write scope around FIXTURE SETUP only (closeout).

    Fixtures seed data by definition, so they run under
    fixture_write_scope.  The test BODY no longer does: writes from a test
    body need @pytest.mark.fixture_writes(reason=...), and engine-code
    writes remain governed by the classified scopes (guard active in test
    bodies -- that is the point of the narrowing).

    Internal autouse fixtures (underscore-prefixed conftest machinery)
    are skipped: their contexts must survive into the test body, and this
    wrapper's token-reset on setup-exit would clobber a nested scope that
    a suspended yield-fixture still holds open.
    """
    if fixturedef.argname.startswith("_"):
        yield
        return
    from mirror_memory.core.tenancy import fixture_write_scope

    with fixture_write_scope("fixture setup: " + fixturedef.argname):
        yield


def _marker_reason(request) -> str:
    """Extract the mandatory reason from a fixture_writes marker."""
    writes_marker = request.node.get_closest_marker("fixture_writes")
    reason = (writes_marker.kwargs or {}).get("reason") or (
        writes_marker.args[0] if writes_marker.args else ""
    )
    if not reason:
        raise pytest.UsageError(
            "@pytest.mark.fixture_writes requires a reason=... argument"
        )
    return reason


@pytest.fixture(autouse=True)
def _body_write_scope(request):
    """Write scope for the TEST BODY (closeout patch, task 1).

    Default: NO blanket scope -- the write guard is live in test bodies.
    A test that must write protected tables directly declares
    @pytest.mark.fixture_writes(reason="...") and runs under
    fixture_write_scope for its body.  Canary tests
    (@pytest.mark.no_write_fixture_scope) keep running bare either way.
    """
    if request.node.get_closest_marker("no_write_fixture_scope"):
        yield
        return
    writes_marker = request.node.get_closest_marker("fixture_writes")
    if writes_marker is None:
        yield
        return
    from mirror_memory.core.tenancy import fixture_write_scope

    with fixture_write_scope("test body (declared): " + _marker_reason(request)):
        yield


@pytest.fixture(autouse=True)
def _default_tenant_context():
    """Test entry point: every test runs under the explicit default tenant.

    Mirrors what a MemoryEngine constructor does in production.  Tests that
    exercise other tenants wrap themselves in tenant_scope(); the fail-closed
    tests clear the context explicitly with no_tenant_scope().
    """
    from mirror_memory.core.tenancy import tenant_scope

    with tenant_scope("default", "default"):
        yield


@pytest.fixture(autouse=True)
def _scope_strict():
    """Task 1b: unregistered engine system_scope call sites fail tests."""
    import os

    old = os.environ.get("MM_SCOPE_STRICT")
    os.environ["MM_SCOPE_STRICT"] = "1"
    yield
    if old is None:
        os.environ.pop("MM_SCOPE_STRICT", None)
    else:
        os.environ["MM_SCOPE_STRICT"] = old


@pytest.fixture(autouse=True)
def _default_test_purpose(monkeypatch):
    """M3: every MemoryEngine the tests build gets the explicit default
    purpose "test" (the engine parameter is the sanctioned way to default
    a purpose; nothing implicit in engine code).  A test that needs an
    engine WITHOUT a default (the purpose-gate tests) passes
    default_purpose=None explicitly, which setdefault respects."""
    from mirror_memory.api import MemoryEngine

    original = MemoryEngine.__init__

    def with_default_purpose(self, *args, **kwargs):
        kwargs.setdefault("default_purpose", "test")
        original(self, *args, **kwargs)

    monkeypatch.setattr(MemoryEngine, "__init__", with_default_purpose)
    yield


#: True inside the granting wrapper; refusal tests flip it off for their
#: duration (conftest-owned, deliberately NOT read by src -- an env-var
#: auto-grant in engine code was removed as an authorization hole,
#: M3 batch-1 patch task 1).
_test_grants_active = contextvars.ContextVar("mm_test_grants_active", default=True)


@pytest.fixture(autouse=True)
def _grant_on_authorize(monkeypatch):
    """M3: the purpose authorization layer is on for every engine call in
    the suite, and the harness issues REAL grant rows for the identities a
    test touches (src has no trace of this; tools/check_no_autogrant.py
    keeps it that way).

    Why not a fixture-prep pre-issuance (M3 batch-2 task 0): the user ids,
    tenants and purposes a test uses only appear inside its body, so a
    grant is issued on first touch of each (tenant, user, purpose) and
    then REMEMBERED -- read-only paths add no statement after that, which
    is what keeps the SQLite perf measurement free of harness cost.

    Per dialect:
    - PostgreSQL: the grant is committed through a dedicated admin engine
      (own one-connection pool -- it never borrows from the engine under
      test, so a pool-exhaustion storm cannot deadlock on it) and memoised
      for the test.
    - SQLite: an admin engine would be a different database (in-memory is
      per-engine), so the row is inserted in the caller's session inside a
      SAVEPOINT released at once.  It is memoised only when that savepoint
      was outermost (RELEASE == COMMIT, durable); see _grant_in_savepoint.

    Still lazy, by necessity (the list the task asked for): every first
    touch of a new (tenant, user, purpose); and on SQLite every authorize
    that runs inside a transaction some earlier DML opened (the worker's
    publish re-check, bench seeding under fixture_write_scope) -- those
    re-check with one SELECT because a later rollback could discard the
    row.

    Refusal tests opt out with the ``refusing_authz`` context manager."""
    import json as _json

    from sqlalchemy import create_engine as _create_engine
    from sqlalchemy.orm import sessionmaker as _sessionmaker

    from mirror_memory.core import authorization as _authorization
    from mirror_memory.core.models import PurposeGrant as _PurposeGrant
    from mirror_memory.core.tenancy import peek_tenant as _peek_tenant

    original = _authorization.authorize_or_raise
    admin: dict = {"engine": None}
    granted: set[tuple[str, str, str, str]] = set()
    sqlite_granted: dict[tuple[int, str, str, str, str], object] = {}

    def _grant_now(bind, tenant_id, app_id, user_id, purpose):
        """PostgreSQL only: idempotent issuance in an independent, COMMITTED
        transaction on the admin engine (own pool of one: no contention
        with the engine under test, no borrowing from its pool)."""
        if admin["engine"] is None:
            # the URL OBJECT (not str()): str() masks the password, and an
            # engine built on the masked URL fails authentication
            admin["engine"] = _create_engine(bind.url, pool_size=1, max_overflow=0)
        Session = _sessionmaker(bind=admin["engine"])
        gs = Session()
        try:
            existing = (
                gs.query(_PurposeGrant)
                .filter_by(user_id=user_id, purpose=purpose)
                .first()
            )
            if existing is None:
                _authorization.grant_purpose(
                    gs, user_id, purpose,
                    operations=list(_authorization.OPERATIONS),
                    issuer="test-harness",
                    reason="harness lazy grant (pg admin engine)",
                )
            gs.commit()
        except Exception:
            gs.rollback()
            raise
        finally:
            gs.close()
        granted.add((tenant_id, app_id, user_id, purpose))

    def _grant_in_savepoint(bind, session, user_id, purpose):
        """SQLite: an admin engine would see a DIFFERENT database (in-memory
        is per-engine), so the grant row is inserted directly in the
        caller's session inside a SAVEPOINT released immediately -- no CAS,
        no event row.  A collision with a grant issued meanwhile (e.g. the
        bench's own grant_bench_purpose) is tolerated by rolling the
        savepoint back and carrying on.

        Memoised only when the row is known to be durable: pysqlite's
        legacy mode does not BEGIN for SELECTs, so when authorize is the
        session's first statement the SAVEPOINT is outermost and its
        RELEASE *is* a COMMIT (SQLite semantics) -- the row survives any
        later rollback and read-only paths pay nothing further.  When some
        earlier DML already opened the transaction (worker publish
        re-check, bench seeding inside fixture_write_scope) the savepoint is
        nested and a later rollback would discard the row, so those calls
        re-check with one SELECT each time."""
        from mirror_memory.core.tenancy import fixture_write_scope as _fws

        # grant rows are tenant-scoped: the same user under another tenant
        # on the same database has no row yet
        tenant = _peek_tenant() or ("", "")
        key = (id(bind), tenant[0], tenant[1], user_id, purpose)
        if key in sqlite_granted:
            return
        existing = (
            session.query(_PurposeGrant)
            .filter_by(user_id=user_id, purpose=purpose)
            .first()
        )
        if existing is not None:
            return
        dbapi = session.connection().connection.dbapi_connection
        outermost = not getattr(dbapi, "in_transaction", True)
        nested = session.begin_nested()
        try:
            with _fws("harness lazy grant (sqlite savepoint)"):
                session.add(_PurposeGrant(
                    user_id=user_id,
                    purpose=purpose,
                    version=1,
                    allowed_operations=_json.dumps(list(_authorization.OPERATIONS)),
                    issuer="test-harness",
                ))
            nested.commit()
        except Exception:
            nested.rollback()
            return
        if outermost:
            sqlite_granted[key] = bind  # the reference pins id(bind) for the test

    def granting(session, user_id, purpose, operation, *, lock=False):
        if not _test_grants_active.get():
            return original(session, user_id, purpose, operation, lock=lock)
        resolved = (purpose or "").strip()
        if not resolved:
            return original(session, user_id, purpose, operation, lock=lock)
        bind = session.get_bind()
        if bind.url.get_backend_name() == "postgresql":
            tenant = _peek_tenant() or ("", "")
            key = (tenant[0], tenant[1], user_id, resolved)
            if key not in granted:
                _grant_now(bind, tenant[0], tenant[1], user_id, resolved)
        else:
            _grant_in_savepoint(bind, session, user_id, resolved)
        return original(session, user_id, purpose, operation, lock=lock)

    monkeypatch.setattr(_authorization, "authorize_or_raise", granting)
    yield

    if admin["engine"] is not None:
        admin["engine"].dispose()


@contextlib.contextmanager
def refusing_authz():
    """Opt a code block out of the harness's lazy test-grant issuance:
    authorize_or_raise delegates straight through, so missing grants
    produce real no_grant refusals.  Import from tests.conftest."""
    token = _test_grants_active.set(False)
    try:
        yield
    finally:
        _test_grants_active.reset(token)


@pytest.fixture(autouse=True)
def _sql_auditor():
    """M1: the SQL auditor is always on during tests (MM_SQL_AUDIT=0 to
    disable for performance measurement runs)."""
    import os

    from mirror_memory.core.sql_audit import install_sql_auditor

    if os.environ.get("MM_SQL_AUDIT", "1") != "0":
        install_sql_auditor()
    yield


def pytest_sessionfinish(session, exitstatus):
    """Report the auditor verdict-cache statistics for the whole run
    (task 1a: numbers must come from the full suite, not single runs)."""
    import os

    if os.environ.get("MM_SQL_AUDIT", "1") == "0":
        return
    try:
        from mirror_memory.core.sql_audit import auditor_cache_stats
    except Exception:  # noqa: BLE001
        return
    distinct, hits, misses = auditor_cache_stats()
    total = hits + misses
    rate = f"{hits / total * 100:.1f}%" if total else "n/a"
    print(
        f"\nsql-auditor cache: distinct_sql_texts={distinct} "
        f"hits={hits} misses={misses} hit_rate={rate}"
    )


def _pg_url() -> str | None:
    """PostgreSQL ADMIN URL for this test run (task 0), or None for SQLite.

    This is the superuser / owner-side login: it creates schemas, applies
    migrations and installs the RLS posture.  Tenant work in tests goes
    through the APP role (``_pg_role_url("app")``) so that row-level
    security actually applies -- a superuser bypasses every policy.
    """
    import os

    url = os.environ.get("MM_DATABASE_URL", "")
    return url if url.startswith("postgres") else None


#: Test-only credentials for the two login roles (created by
#: ``_pg_rls_roles`` once per session; the owner role has no login).
PG_TEST_ROLE_PASSWORDS = {"app": "mm_app_test", "maintenance": "mm_maintenance_test"}


def _pg_runtime_url() -> str | None:
    """The URL tenant work runs on: the app role under RLS; the admin login
    when RLS is switched off (``MM_PG_RLS=0``, the measurement-only
    baseline C -- no policies, no per-transaction setting, superuser
    connection)."""
    from mirror_memory.core import rls

    return _pg_role_url("app") if rls.rls_enabled() else _pg_url()


def _pg_role_url(kind: str) -> str | None:
    """The admin URL re-pointed at the app / maintenance login."""
    admin = _pg_url()
    if admin is None:
        return None
    from sqlalchemy.engine import make_url

    from mirror_memory.core import rls

    role = rls.app_role() if kind == "app" else rls.maintenance_role()
    return (
        make_url(admin)
        .set(username=role, password=PG_TEST_ROLE_PASSWORDS[kind])
        .render_as_string(hide_password=False)
    )


@pytest.fixture(scope="session", autouse=True)
def _pg_rls_roles():
    """PostgreSQL only: make sure the owner / app / maintenance roles exist
    with the test credentials, and register the maintenance URL (M2
    batch-2 task 1).

    Roles are cluster-level and persist across tests; the per-test schema
    rebuild (``db_session``) re-applies ownership, grants and policies.
    """
    import os

    admin = _pg_url()
    if admin is None:
        yield
        return
    import psycopg
    from psycopg import sql
    from sqlalchemy.engine import make_url

    from mirror_memory.core import rls

    conninfo = make_url(admin).set(drivername="postgresql").render_as_string(hide_password=False)
    roles = (
        (rls.owner_role(), False, None),
        (rls.app_role(), True, PG_TEST_ROLE_PASSWORDS["app"]),
        (rls.maintenance_role(), True, PG_TEST_ROLE_PASSWORDS["maintenance"]),
    )
    with psycopg.connect(conninfo, autocommit=True) as conn:
        for name, login, password in roles:
            exists = conn.execute(
                "SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s", (name,)
            ).fetchone()
            if exists is None:
                if login:
                    conn.execute(
                        sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                            sql.Identifier(name), sql.Literal(password)
                        )
                    )
                else:
                    conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(name)))
            elif login:
                conn.execute(
                    sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                        sql.Identifier(name), sql.Literal(password)
                    )
                )
    maintenance_url = _pg_role_url("maintenance")
    old = os.environ.get("MM_MAINTENANCE_DATABASE_URL")
    os.environ["MM_MAINTENANCE_DATABASE_URL"] = maintenance_url
    # Both the admin and the app database identities route system_scope
    # sessions to the maintenance engine.
    rls.register_maintenance_url(admin, maintenance_url)
    rls.register_maintenance_url(_pg_role_url("app"), maintenance_url)
    try:
        yield
    finally:
        rls.clear_maintenance_registry()
        if old is None:
            os.environ.pop("MM_MAINTENANCE_DATABASE_URL", None)
        else:
            os.environ["MM_MAINTENANCE_DATABASE_URL"] = old


def _pg_fresh_schema(admin_engine) -> None:
    """Rebuild the schema from the model and install the RLS posture on it.

    Every table ends up owned by the owner role with ENABLE + FORCE row
    level security and both policies, exactly what step 3 of the migration
    leaves behind; verification is skipped here (the dedicated RLS tests
    verify it) to keep the per-test cost down.

    Leaked sessions: a test that fails while a Session still holds an
    open transaction keeps that connection alive (the failing frame
    references it, and ``Engine.dispose`` cannot close a checked-out
    connection), and the next test's ``DROP TABLE`` would then wait on
    its locks forever.  The drop runs under a 2 s transaction-local
    ``lock_timeout``; on timeout the harness terminates every other
    backend on this test database, warns loudly (so the leaking test is
    visible), and retries.  The run keeps going; the failure that caused
    the leak is still reported as such.
    """
    import warnings

    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from mirror_memory.core.schema_migration import install_row_level_security

    def drop_bounded() -> None:
        from sqlalchemy import MetaData

        from mirror_memory.core.schema_migration import (
            WATERMARK_TABLE,
            _version_table,
        )

        with admin_engine.connect() as conn:
            conn.execute(text("SET LOCAL lock_timeout = '2s'"))
            Base.metadata.drop_all(conn)  # clean slate (previous test leftovers)
            # the version table is not in Base.metadata; a fresh schema has
            # no migration history either (a leftover row would make the
            # next upgrade_database skip every step)
            _version_table(MetaData()).drop(conn, checkfirst=True)
            # the deletion watermark is not in Base.metadata either; a
            # leftover row pairs the fresh schema with a stale (db_id,
            # applied_seq) the new test's log cannot match (M4)
            conn.execute(text(
                f"DROP TABLE IF EXISTS {WATERMARK_TABLE}"))
            conn.commit()

    try:
        drop_bounded()
    except OperationalError as exc:
        # 55P03 lock timeout and 40P01 deadlock both mean someone else's
        # locks are in the way -- the self-healing below applies to both
        if "55P03" not in str(exc) and "40P01" not in str(exc) and "lock timeout" not in str(exc).lower() and "deadlock" not in str(exc).lower():
            raise
        with admin_engine.connect() as conn:
            conn.rollback()
            leaked = conn.execute(text(
                "SELECT pid, state, left(query, 120) FROM pg_catalog.pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_catalog.pg_backend_pid() "
                "AND backend_type = 'client backend'"
            )).all()
            for pid, _state, _query in leaked:
                conn.execute(text("SELECT pg_catalog.pg_terminate_backend(:pid)"), {"pid": pid})
            conn.commit()
        warnings.warn(
            "pg harness: schema rebuild waited >2s for locks held by leaked "
            f"session(s); terminated {len(leaked)} backend(s): "
            + "; ".join(f"pid={p} {s}: {q!r}" for p, s, q in leaked),
            stacklevel=2,
        )
        drop_bounded()
    Base.metadata.create_all(admin_engine)
    from mirror_memory.core import rls

    if rls.rls_enabled():
        install_row_level_security(admin_engine, verify=False)

    # M4: the deletion watermark lives outside the model metadata -- the
    # runtime (app role) reads and advances it, so it needs the grants too
    from mirror_memory.core import rls as _rls, schema_migration as _sm
    from mirror_memory.core.tenancy import system_scope as _system_scope

    with _system_scope(reason="test harness: watermark parity with step 11"):
        _sm.ensure_watermark(admin_engine)
    with admin_engine.begin() as conn:
        conn.exec_driver_sql(
            f"GRANT SELECT, INSERT, UPDATE ON mm_deletion_watermark "
            f"TO {_rls.app_role()}, {_rls.maintenance_role()}")



@pytest.fixture
def pg_admin_engine():
    """PostgreSQL only: a fresh RLS-protected schema; yields the admin
    (owner-side) engine.  Skips on SQLite."""
    pg = _pg_url()
    if pg is None:
        pytest.skip("needs MM_DATABASE_URL pointing at PostgreSQL")
    engine = create_engine(pg, echo=False)
    _pg_fresh_schema(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def pg_app_url(pg_admin_engine):
    """The runtime URL for the schema ``pg_admin_engine`` just built: the
    app role under RLS, the admin login when RLS is off (mode C)."""
    return _pg_runtime_url()


@pytest.fixture
def pg_maintenance_url(pg_admin_engine):
    return _pg_role_url("maintenance")


@pytest.fixture(autouse=True)
def _pg_engine_reaper():
    """PostgreSQL only: close every engine pool opened during a test.

    Several test modules build their own engines (helper fixtures, worker
    factories, migration tests).  Under SQLite those are in-memory and
    cheap; under PostgreSQL each keeps pooled server connections alive, and
    a full-suite run exhausts the server's connection slots (CI run
    40338513001: 271 cascading 'too many clients' errors from test_worker
    onward).  Two sweeps:

    - ``sqlalchemy.create_engine`` is patched so engines created through
      the module attribute are tracked and disposed directly;
    - modules that imported the name at their own import time bypass that,
      so teardown ALSO walks ``gc`` for every live ``Engine`` and disposes
      it -- refcount delays and cycles would otherwise hold server
      connections until long after the test finished.
    """
    if _pg_url() is None:
        yield
        return
    import gc

    import sqlalchemy
    from sqlalchemy.engine import Engine

    created: list = []
    original = sqlalchemy.create_engine

    def recording_create_engine(*args, **kwargs):
        engine = original(*args, **kwargs)
        created.append(engine)
        return engine

    def _dispose_all() -> None:
        seen = set()
        for engine in created:
            seen.add(id(engine))
            try:
                engine.dispose()
            except Exception:  # noqa: BLE001 -- cleanup must not fail a test
                pass
        for obj in gc.get_objects():
            if isinstance(obj, Engine) and id(obj) not in seen:
                try:
                    obj.dispose()
                except Exception:  # noqa: BLE001
                    pass

    sqlalchemy.create_engine = recording_create_engine
    try:
        yield
    finally:
        sqlalchemy.create_engine = original
        _dispose_all()


@pytest.fixture
def db_session():
    """Per-test database session.

    SQLite (default): in-memory; tables dropped after each test.

    PostgreSQL (``MM_DATABASE_URL``): the test runs against the shared
    database with the schema rebuilt around it (``drop_all`` +
    ``create_all`` from ``Base.metadata``) by the ADMIN login, the RLS
    posture installed on it, and the session opened through the APP role
    engine -- so the whole suite exercises row-level security as the
    second defence (M2 batch-2 task 1), not just the dedicated RLS tests.
    Rebuild chosen over transaction-rollback because much of the suite
    manages its own transactions and opens several sessions per test -- a
    rollback harness would fight both.  Chosen over schema/database-per-
    test because it needs no identifier construction at all: everything
    goes through metadata DDL, so there is no SQL-text assembly here.
    """
    pg = _pg_url()
    if pg is None:
        engine = create_engine("sqlite://", echo=False)
        Base.metadata.create_all(engine)
        # M4: deploy-time shape -- the watermark row exists before any
        # content is seeded (matching the PG branch / Step 11)
        from mirror_memory.core import schema_migration as _sm

        _sm.ensure_watermark(engine)
        SessionFactory = sessionmaker(bind=engine)
        session = SessionFactory()
        yield session
        session.rollback()
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
        return
    admin_engine = create_engine(pg, echo=False)
    _pg_fresh_schema(admin_engine)
    engine = create_engine(_pg_runtime_url(), echo=False)
    SessionFactory = sessionmaker(bind=engine)
    session = SessionFactory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()
        admin_engine.dispose()


@pytest.fixture
def config():
    """Minimal MemoryConfig for testing."""
    return load_config("config/")


class MockLLM:
    """Mock LLM client that returns canned responses."""

    def __init__(self, response='{"claims": []}'):
        self._response = response
        self.calls = []

    def generate(self, *, system_prompt, payload_text, fallback=None):
        self.calls.append({"system_prompt": system_prompt, "payload_text": payload_text})
        return self._response


@pytest.fixture
def mock_llm():
    """Default mock LLM returning empty claims."""
    return MockLLM()


# ---------------------------------------------------------------------------
# M4 batch-1 patch task 2: on the PostgreSQL leg the default `file`
# deletion-log backend needs an explicit directory for server URLs (the
# URL-derived default is embedded-SQLite only).  The wrapper runs at
# conftest import time, so every test module's `from ... import
# load_config` binds it; tests that exercise the deletion log set their
# own backend/directory after load_config and override this default.
# ---------------------------------------------------------------------------
import functools as _functools

import mirror_memory.config.loader as _cfg_loader

_original_load_config = _cfg_loader.load_config


@_functools.wraps(_original_load_config)
def _load_config_with_log_directory(*args, **kwargs):
    cfg = _original_load_config(*args, **kwargs)
    import os as _os

    if _os.environ.get("MM_DATABASE_URL", "").startswith("postgres"):
        cfg.deletion_log.directory = (
            cfg.deletion_log.directory
            or f"/tmp/mm-test-deletions-{_os.getpid()}")
    return cfg


_cfg_loader.load_config = _load_config_with_log_directory


def strip_to_step10(engine) -> None:
    """Roll a v11 database back to EXACTLY the step-10 shape (M4 patch 2
    tests): drop every step-11 artifact and the version rows above 10.
    The result is what a real pre-M4 backup looks like: tenant data, no
    watermark, no generations table, no snapshot synthesis generation.
    Run it owner-side on PostgreSQL."""
    from sqlalchemy import MetaData, inspect, text

    from mirror_memory.core.schema_migration import (
        WATERMARK_TABLE,
        _version_table,
    )

    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {WATERMARK_TABLE}")
        conn.exec_driver_sql("DROP TABLE IF EXISTS mm_deletion_generations")
        if "source_generation" in {
                c["name"] for c in inspect(conn).get_columns("mm_snapshots")}:
            conn.exec_driver_sql(
                "ALTER TABLE mm_snapshots DROP COLUMN source_generation")
        _version_table(MetaData()).create(engine, checkfirst=True)
        conn.execute(text(
            "DELETE FROM mm_schema_version WHERE version > 10"))
        if conn.execute(text(
                "SELECT max(version) FROM mm_schema_version")).scalar() is None:
            for v, note in [(1, "tenancy"), (5, "pending-jobs index"),
                            (10, "derived provenance")]:
                conn.execute(text(
                    "INSERT INTO mm_schema_version (version, note) "
                    "VALUES (:v, :n)"), {"v": v, "n": note})
