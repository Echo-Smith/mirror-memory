"""Shared test fixtures for mirror-memory."""

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
        with admin_engine.connect() as conn:
            conn.execute(text("SET LOCAL lock_timeout = '2s'"))
            Base.metadata.drop_all(conn)  # clean slate (previous test leftovers)
            conn.commit()

    try:
        drop_bounded()
    except OperationalError as exc:
        if "55P03" not in str(exc) and "lock timeout" not in str(exc).lower():
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
    install_row_level_security(admin_engine, verify=False)


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
    """The app-role URL for the schema ``pg_admin_engine`` just built."""
    return _pg_role_url("app")


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
    engine = create_engine(_pg_role_url("app"), echo=False)
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
