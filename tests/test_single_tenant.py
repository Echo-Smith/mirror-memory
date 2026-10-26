"""M1 batch-2 task 6 -- end-side single-tenant mode.

Startup refusals (fail-closed):
1. database holds more than one tenant
2. configured tenant differs from the stored one
3. the server tenant middleware is active at the same time

Plus: the mode actually works (observe/recall; tenant columns stamped,
verified through ORM reads) and the read injection is really skipped
(statements no longer carry the loader-criteria bind parameters).  The
write guard and flush tenant checks stay active.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.tenancy import (
    TenantMismatchError,
    WriteGuardViolation,
    fixture_write_scope,
    set_single_tenant_mode,
    tenant_scope,
)


# canary file: guard/flush assertions must run WITHOUT the blanket test
# write scope, like the write-guard suite
pytestmark = pytest.mark.no_write_fixture_scope


class _SilentLLM:
    def generate(self, *, system_prompt, payload_text, fallback=None):
        return '{"claims": []}'


def make_engine(url, tenant=("tA", "app"), **kw):
    return MemoryEngine(
        config=load_config("config/"),
        database_url=url,
        llm_client=_SilentLLM(),
        tenant_id=tenant[0],
        app_id=tenant[1],
        **kw,
    )


def seed_belief(url, tenant, key):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with tenant_scope(*tenant), fixture_write_scope("seed tenant row"):
        s = factory()
        s.add(Belief(user_id="u1", dimension="topic", key=key,
                     claim_text="ROW", confidence=0.9))
        s.commit()
        s.close()
    engine.dispose()


# 1 ------------------------------------------------------------------------


def test_refuses_multi_tenant_database(tmp_path):
    url = f"sqlite:///{tmp_path}/multi.db"
    make_engine(url)._ensure_db()
    seed_belief(url, ("tA", "app"), "a")
    seed_belief(url, ("tB", "app"), "b")
    with pytest.raises(RuntimeError, match="multiple"):
        make_engine(url, single_tenant_mode=True)._ensure_db()


# 2 ------------------------------------------------------------------------


def test_refuses_tenant_mismatch(tmp_path):
    url = f"sqlite:///{tmp_path}/mismatch.db"
    make_engine(url)._ensure_db()
    seed_belief(url, ("tA", "app"), "stored")  # the stored tenant is tA
    # configuring tB must refuse
    with pytest.raises(RuntimeError, match="configured"):
        make_engine(url, tenant=("tB", "app"), single_tenant_mode=True)._ensure_db()


# 3 ------------------------------------------------------------------------


def test_refuses_with_server_middleware(tmp_path):
    from mirror_memory import server as server_mod

    url = f"sqlite:///{tmp_path}/mutual.db"
    server_mod._tenant_key_registry.clear()
    server_mod.register_tenant_key("some-key", "tA", "app")
    try:
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            make_engine(url, single_tenant_mode=True)._ensure_db()
    finally:
        server_mod._tenant_key_registry.clear()


# happy path ---------------------------------------------------------------


def test_single_tenant_mode_works_and_skips_injection(tmp_path):
    url = f"sqlite:///{tmp_path}/single.db"
    engine = make_engine(url, single_tenant_mode=True)
    engine.observe(user_id="u1", session_id="s1", text="I love coffee")

    raw = create_engine(url)
    Base.metadata.create_all(raw)
    factory = sessionmaker(bind=raw)

    # tenant columns still stamped (ORM read under the engine's tenant)
    with tenant_scope("tA", "app"):
        s = factory()
        row = s.query(Belief).one()
        assert (row.tenant_id, row.app_id) == ("tA", "app")
        s.close()
    assert engine.recall(user_id="u1", query="coffee") is not None

    # read injection really skipped: an ORM SELECT in this mode must NOT
    # carry the loader-criteria bind parameters anymore
    captured = []
    event.listen(raw, "before_cursor_execute",
                 lambda c, cur, s_, p, ctx, m: captured.append(s_))
    set_single_tenant_mode(True)
    with tenant_scope("tA", "app"):
        s = factory()
        s.query(Belief).all()
        s.close()
    set_single_tenant_mode(False)
    assert captured, "no SQL captured"
    assert not any("mm_tenant_id" in sql for sql in captured), (
        "loader-criteria parameters present although injection is skipped"
    )


def test_flush_checks_still_active_in_single_tenant(tmp_path):
    url = f"sqlite:///{tmp_path}/guard.db"
    make_engine(url, single_tenant_mode=True)._ensure_db()
    raw = create_engine(url)
    Base.metadata.create_all(raw)
    factory = sessionmaker(bind=raw)

    set_single_tenant_mode(True)
    with tenant_scope("tA", "app"):
        s = factory()
        # guard: direct belief write outside scopes still refused
        with pytest.raises(WriteGuardViolation):
            s.add(Belief(user_id="u1", dimension="d", key="k", claim_text="x"))
        # flush check: carrying a foreign tenant still refused
        with fixture_write_scope("seed attempt"):
            s.add(Belief(user_id="u1", dimension="d", key="k2",
                         claim_text="x", tenant_id="tB", app_id="app"))
        with pytest.raises(TenantMismatchError):
            s.commit()
        s.rollback()
        s.close()
    set_single_tenant_mode(False)
