"""Shared test fixtures for mirror-memory."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.models import Base


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


@pytest.fixture
def db_session():
    """In-memory SQLite session with all tables, rolled back after each test."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)
    session = SessionFactory()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(engine)


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
