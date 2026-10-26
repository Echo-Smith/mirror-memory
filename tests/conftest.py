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
    """PostgreSQL URL for this test run (task 0), or None for SQLite."""
    import os

    url = os.environ.get("MM_DATABASE_URL", "")
    return url if url.startswith("postgres") else None


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
    ``create_all`` from ``Base.metadata``).  Chosen over
    transaction-rollback because much of the suite manages its own
    transactions and opens several sessions per test -- a rollback
    harness would fight both.  Chosen over schema/database-per-test
    because it needs no identifier construction at all: everything goes
    through metadata DDL, so there is no SQL-text assembly here.  Cost
    measured locally: ~0.15 s per rebuild.
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
    engine = create_engine(pg, echo=False)
    Base.metadata.drop_all(engine)  # clean slate (previous test leftovers)
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)
    session = SessionFactory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


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
