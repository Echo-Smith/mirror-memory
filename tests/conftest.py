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
def _sql_auditor():
    """M1: the SQL auditor is always on during tests (MM_SQL_AUDIT=0 to
    disable for performance measurement runs)."""
    import os

    from mirror_memory.core.sql_audit import install_sql_auditor

    if os.environ.get("MM_SQL_AUDIT", "1") != "0":
        install_sql_auditor()
    yield


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
