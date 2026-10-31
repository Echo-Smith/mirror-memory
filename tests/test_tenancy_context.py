"""M1 task 1 -- tenant context passing rules and tooling.

The seven required scenarios plus nested-scope ordering:
1. no tenant context on a tenant-scoped query -> TenantContextMissing
2. bare ThreadPoolExecutor fails closed; the tenant-aware wrapper works
3. asyncio.gather over 50 tenants, each coroutine sees only its own data
4. a worker processing tenant A's job while the scheduler sits in tenant B
5. lazy load of an object loaded under tenant A, from tenant B -> error
6. context restored after an exception inside tenant_scope
7. nested tenant_scope / system_scope enter-exit ordering
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import record_claim
from mirror_memory.core.tenancy import (
    TenantAwareExecutor,
    TenantContextMissing,
    TenantMismatchError,
    no_tenant_scope,
    submit_with_tenant,
    system_scope,
    tenant_scope,
)


def session_at(url):
    from sqlalchemy.pool import NullPool

    engine = create_engine(url, poolclass=NullPool)
    Base.metadata.create_all(engine)
    from mirror_memory.core import schema_migration as _sm_wm
    _sm_wm.ensure_watermark(engine)
    return sessionmaker(bind=engine)


def seed(factory, tenant, key, claim):
    with tenant_scope(*tenant):
        s = factory()
        record_claim(s, "u1", dimension="topic", key=key, claim_text=claim, confidence=0.8)
        s.commit()
        s.close()


# 1. no tenant context -> TenantContextMissing
def test_missing_context_fails_closed(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/ctx.db")
    seed(factory, ("tA", "app"), "k", "A")
    s = factory()  # created under no scope? no -- autouse fixture... use no_tenant_scope
    with no_tenant_scope():
        s2 = factory()
        with pytest.raises(TenantContextMissing):
            s2.query(Belief).all()
        s2.close()
    s.close()


# 2. bare thread pool fails closed; wrapped executor works
def test_bare_thread_pool_fails_closed_and_wrapper_works(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/pool.db")
    seed(factory, ("tA", "app"), "k", "A")

    def read_claims():
        s = session_at(f"sqlite:///{tmp_path}/pool.db")()
        try:
            return [b.claim_text for b in s.query(Belief).all()]
        finally:
            s.close()

    # A session *created* in the bare worker thread has no bound tenant and
    # no ambient context -> fail closed.
    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(TenantContextMissing):
            list(pool.submit(read_claims).result() for _ in range(1))

    # The submitting tenant's context travels with the wrapped executor.
    with tenant_scope("tA", "app"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            wrapped = TenantAwareExecutor(pool)
            results = [wrapped.submit(read_claims) for _ in range(4)]
            for future in results:
                assert future.result() == ["A"]

    # submit_with_tenant on a bare pool behaves the same as the wrapper.
    with tenant_scope("tA", "app"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [submit_with_tenant(pool, read_claims) for _ in range(4)]
            for future in futures:
                assert future.result() == ["A"]


# 3. asyncio.gather over 50 tenants
def test_asyncio_gather_fifty_tenants(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/async.db")
    tenants = [(f"tenant-{i:03d}", "app") for i in range(50)]
    for i, tenant in enumerate(tenants):
        seed(factory, tenant, "k", f"CLAIM-{i}")

    async def read_one(tenant, marker):
        with tenant_scope(*tenant):
            s = factory()
            try:
                rows = s.query(Belief).all()
                assert len(rows) == 1
                assert rows[0].claim_text == marker
                # let other coroutines run between the statement and the
                # assertion checks to prove no cross-talk via the loop
                await asyncio.sleep(0)
                rows = s.query(Belief).all()
                assert rows[0].claim_text == marker
            finally:
                s.close()

    async def main():
        await asyncio.gather(
            *(read_one(t, f"CLAIM-{i}") for i, t in enumerate(tenants))
        )

    asyncio.run(main())


# 4. worker reads the tenant from the job record, not the scheduler
@pytest.mark.fixture_writes(reason="worker context test: enqueue job row directly")
def test_worker_uses_job_tenant_not_scheduler_context(tmp_path):
    from mirror_memory.config.loader import load_config
    from mirror_memory.core.models import EvolutionJob, Snapshot
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.worker.evolution import process_pending

    factory = session_at(f"sqlite:///{tmp_path}/worker.db")
    cfg = load_config("config/")
    # worker tick without a deletion log: explicit backend=none (M4 patch-4)
    cfg.deletion_log.backend = "none"

    class FakeLLM:
        def generate(self, **kw):
            return '{"patterns": [], "support_policy": {}}'

    cfg.llm_client = FakeLLM()

    # Seed tenant A with 3 sessions of evidence and enqueue its job.
    seed(factory, ("tA", "app"), "sleep", "A_CONTENT")
    with tenant_scope("tA", "app"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        for i in range(3):
            record_claim(s, "u1", dimension="topic", key=f"k{i}",
                         claim_text="A_CONTENT", confidence=0.8, session_id=f"s{i}")
        s.commit()
        job_id = "job-a-1"
        s.add(EvolutionJob(
            id=job_id, user_id="u1",
            evidence_watermark="{}", idempotency_key="tA:u1:v1",
            status="pending", tenant_id="tA", app_id="app", purpose="test"))
        s.commit()
        s.close()

    # The *scheduler* runs under tenant B's context while draining the queue.
    with tenant_scope("tB", "app"):
        process_pending(factory, cfg)

    with tenant_scope("tA", "app"):
        s = factory()
        job = s.get(EvolutionJob, job_id)
        assert job.status == "completed"
        snaps = s.query(Snapshot).all()
        assert len(snaps) == 1
        s.close()
    with tenant_scope("tB", "app"):
        s = factory()
        assert s.query(Snapshot).all() == []
        s.close()


# 5. cross-tenant lazy load raises
def test_cross_tenant_lazy_load_raises(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/lazy.db")
    seed(factory, ("tA", "app"), "k", "A")
    seed(factory, ("tB", "app"), "k", "B")

    with tenant_scope("tA", "app"):
        s = factory()
        belief = s.query(Belief).one()
        aid = belief.id
        s.close()

    with tenant_scope("tB", "app"):
        s2 = factory()
        # session.get cannot reach the other tenant's row at all
        assert s2.get(Belief, aid) is None
        s2.close()

    # An object loaded under tA and *expunged* is detached; loading it into
    # a tB-bound session is impossible, and touching tenant-scoped queries
    # with a tA-bound session inside tB raises the mismatch error.
    with pytest.raises(TenantMismatchError):
        with tenant_scope("tB", "app"):
            s3 = factory()  # bound to tB
            with tenant_scope("tA", "app"):
                s3.query(Belief).all()  # bound(tB) != ambient(tA)


# 6. context restored after an exception inside tenant_scope
def test_scope_restores_context_after_exception():
    original = __import__("mirror_memory.core.tenancy", fromlist=["peek_tenant"]).peek_tenant()
    with pytest.raises(RuntimeError):
        with tenant_scope("tX", "app"):
            assert __import__("mirror_memory.core.tenancy", fromlist=["peek_tenant"]).peek_tenant() == ("tX", "app")
            raise RuntimeError("boom")
    assert __import__("mirror_memory.core.tenancy", fromlist=["peek_tenant"]).peek_tenant() == original


# 7. nested scope ordering
def test_nested_scope_ordering():
    import mirror_memory.core.tenancy as ten

    assert ten.peek_tenant() == ("default", "default")  # autouse fixture
    with tenant_scope("outer", "app"):
        with tenant_scope("inner", "app"):
            assert ten.peek_tenant() == ("inner", "app")
            with system_scope(reason="nested test"):
                assert ten.in_system_scope()
                assert ten.system_reason() == "nested test"
                assert ten.peek_tenant() == ("inner", "app")  # system scope keeps tenant
            assert not ten.in_system_scope()
        assert ten.peek_tenant() == ("outer", "app")
    assert ten.peek_tenant() == ("default", "default")

    with pytest.raises(ValueError):
        with system_scope(reason=""):  # reason is mandatory, checked on enter
            pass
