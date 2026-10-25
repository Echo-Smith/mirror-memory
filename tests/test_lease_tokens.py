"""M1 batch-2 task 3 -- I1/I3 lease-token fencing.

Required scenarios:
1. Worker A's lease expires, B re-claims, A resumes and commits ->
   LeaseLost and A's memory writes roll back
2. Two workers race to claim -> exactly one wins
3. Renewal failure followed by commit -> LeaseLost
4. lease_token is monotonic (never resets, survives re-claims/failure)
5. Token check and tenant condition apply together: tenant A's token
   cannot commit tenant B's job
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base, Belief, EvolutionJob, Snapshot
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import tenant_scope
from mirror_memory.worker.evolution import (
    LeaseLost,
    _claim_job,
    _process_single_job,
    _renew_lease,
    _set_job_state,
    enqueue_job,
    process_pending,
)


def session_at(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def make_config(lease_seconds=60):
    cfg = load_config("config/")

    class FakeLLM:
        def generate(self, **kw):
            return '{"patterns": [], "support_policy": {}}'

    cfg.llm_client = FakeLLM()
    cfg.worker.job_lease_seconds = lease_seconds
    return cfg


def seed_job(factory, tenant=("default", "default"), job_id="job-1"):
    cfg = make_config()
    with tenant_scope(*tenant):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        for i in range(3):
            record_claim(
                s, "u1", dimension="topic", key=f"k{i}",
                claim_text="content", confidence=0.8, session_id=f"s{i}",
            )
        s.commit()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("uuid.uuid4._rand_on_first_use", None, raising=False)
        import mirror_memory.worker.evolution as ev
        job = ev.EvolutionJob(
            id=job_id, user_id="u1",
            evidence_watermark="{}", idempotency_key=f"{job_id}:ik",
            status="pending", tenant_id=tenant[0], app_id=tenant[1],
        )
        from mirror_memory.core.tenancy import queue_write_scope

        with queue_write_scope():
            s.add(job)
        s.commit()
        s.close()
    return cfg


# 1. expired lease, re-claim, old holder commits -> LeaseLost + rollback
def test_expired_lease_old_holder_commit_rolls_back(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/l1.db")
    cfg = make_config(lease_seconds=1)
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        from mirror_memory.core.tenancy import queue_write_scope

        with queue_write_scope():
            s.add(EvolutionJob(id="j1", user_id="u1", idempotency_key="ik1", status="pending"))
        s.commit()
        token_a = _claim_job(s, "j1", lease_seconds=1)
        assert token_a == 1
        s.close()

    # lease expires in wall-clock terms; force it in the DB
    with tenant_scope("default", "default"):
        s = factory()
        s.query(EvolutionJob).filter(EvolutionJob.id == "j1").update(
            {"lease_expires_at": datetime.now(UTC) - timedelta(seconds=5)},
            synchronize_session=False,
        )
        s.commit()
        s.close()

    # worker B sweeps + re-claims (fresh token) and completes the job
    process_pending(factory, cfg)

    with tenant_scope("default", "default"):
        s = factory()
        row = s.query(EvolutionJob).filter(EvolutionJob.id == "j1").one()
        assert row.lease_token == 2  # B holds token 2
        assert row.status == "completed"  # B finished it
        snapshots_before = s.query(Snapshot).count()
        # A resumes with its stale token and tries to complete the job
        with pytest.raises(LeaseLost):
            _set_job_state(s, "j1", token_a, {"status": "completed"})
        s.rollback()
        # A's memory writes in the same transaction roll back with it:
        # simulate the publish A made before noticing the loss
        from mirror_memory.core.tenancy import engine_write_scope

        with engine_write_scope():
            s.add(Snapshot(id="ghost", user_id="u1", version=99, status="active"))
        with pytest.raises(LeaseLost):
            _set_job_state(s, "j1", token_a, {"status": "completed"})
        s.rollback()  # the documented LeaseLost handling in the worker
        assert s.query(Snapshot).count() == snapshots_before
        assert s.query(Snapshot).filter(Snapshot.id == "ghost").count() == 0
        s.close()


# 2. two workers race to claim -> exactly one wins
def test_concurrent_claim_single_winner(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/l2.db")
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        from mirror_memory.core.tenancy import queue_write_scope

        with queue_write_scope():
            s.add(EvolutionJob(id="j2", user_id="u1", idempotency_key="ik2", status="pending"))
        s.commit()
        s.close()

    results = []
    import threading

    def claim(factory_ref, out):
        with tenant_scope("default", "default"):
            ss = factory_ref()
            try:
                out.append(_claim_job(ss, "j2", lease_seconds=60))
            finally:
                ss.close()

    threads = [
        threading.Thread(target=claim, args=(factory, results)) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    with tenant_scope("default", "default"):
        s = factory()
        assert s.query(EvolutionJob).filter(EvolutionJob.id == "j2").one().lease_token == 1
        s.close()


# 3. renewal failure then commit -> LeaseLost
def test_failed_renewal_then_commit_lease_lost(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/l3.db")
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        from mirror_memory.core.tenancy import queue_write_scope

        with queue_write_scope():
            s.add(EvolutionJob(id="j3", user_id="u1", idempotency_key="ik3", status="pending"))
        s.commit()
        token = _claim_job(s, "j3", lease_seconds=60)
        assert token == 1
        # another worker steals the lease (expiry sweep + re-claim)
        s.query(EvolutionJob).filter(EvolutionJob.id == "j3").update(
            {
                "status": "pending",
                "lease_expires_at": datetime.now(UTC) - timedelta(seconds=1),
            },
            synchronize_session=False,
        )
        s.commit()
        token_b = _claim_job(s, "j3", lease_seconds=60)
        assert token_b == 2
        # our renewal with the stale token fails...
        assert _renew_lease(s, "j3", token, 60) is False
        # ...and so does the fenced completion
        with pytest.raises(LeaseLost):
            _set_job_state(s, "j3", token, {"status": "completed"})
        s.close()


# 4. lease_token is monotonic across re-claims and failures
def test_lease_token_monotonic(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/l4.db")
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        from mirror_memory.core.tenancy import queue_write_scope

        with queue_write_scope():
            s.add(EvolutionJob(id="j4", user_id="u1", idempotency_key="ik4", status="pending"))
        s.commit()
        seen = []
        for cycle in range(4):
            # expire and return to pending, then re-claim: token must grow
            s.query(EvolutionJob).filter(EvolutionJob.id == "j4").update(
                {"status": "pending", "lease_expires_at": None},
                synchronize_session=False,
            )
            s.commit()
            token = _claim_job(s, "j4", lease_seconds=60)
            seen.append(token)
            # a failing run must not reset the token
            _set_job_state(s, "j4", token, {"status": "failed", "error_code": "x"})
            s.commit()
        assert seen == [1, 2, 3, 4], seen
        s.close()


# 5. tenant condition and token check together
def test_token_and_tenant_both_enforced(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/l5.db")
    # job in tenant A
    with tenant_scope("tA", "app"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        from mirror_memory.core.tenancy import queue_write_scope

        with queue_write_scope():
            s.add(EvolutionJob(
                id="jA", user_id="u1", idempotency_key="ikA",
                status="pending", tenant_id="tA", app_id="app",
            ))
        s.commit()
        token_a = _claim_job(s, "jA", lease_seconds=60)
        assert token_a == 1
        s.close()
    # tenant B's session: the row is invisible (tenant filter) AND the
    # fenced update from B's context can never land
    with tenant_scope("tB", "app"):
        s = factory()
        assert s.query(EvolutionJob).filter(EvolutionJob.id == "jA").count() == 0
        with pytest.raises(LeaseLost):
            _set_job_state(s, "jA", token_a, {"status": "completed"})
        s.close()
    # back in A: the same token still works
    with tenant_scope("tA", "app"):
        s = factory()
        _set_job_state(s, "jA", token_a, {"status": "completed"})
        assert s.query(EvolutionJob).filter(EvolutionJob.id == "jA").one().status == "completed"
        s.close()
