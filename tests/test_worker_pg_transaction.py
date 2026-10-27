"""M2 batch-2 patch task 1: the worker must not hold a database transaction
while the LLM runs.

Nodes 1-5 are compute over a snapshot; only node 6 writes.  A worker that
keeps the snapshot's read transaction open across the LLM calls sits
"idle in transaction" for the whole compute, and PostgreSQL's
``idle_in_transaction_session_timeout`` (60 s by default in
``core.pg_engine``, 2 s here) terminates the backend -- the publish then
fails with a dead connection.  The fix ends the read transaction after the
snapshot is taken and opens a fresh one for the publish; the lease
renewal, revision check and consent re-check keep their "the snapshot may
be stale" semantics.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Belief, EvolutionJob
from mirror_memory.core.pg_engine import build_engine
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import queue_write_scope, tenant_scope
from mirror_memory.worker.evolution import _claim_job, _process_single_job

pytestmark = pytest.mark.skipif(
    "postgresql" not in os.environ.get("MM_DATABASE_URL", ""),
    reason="idle_in_transaction_session_timeout is a PostgreSQL setting",
)

USER = "u1"
TENANT = ("default", "default")


class SleepyLLM:
    """Sleeps in generate() -- the shape of a slow model call -- and records
    how many connections the engine had checked out while it 'thought'."""

    def __init__(self, seconds: float, engine=None) -> None:
        self.seconds = seconds
        self.engine = engine
        self.calls = 0
        self.checked_out_during_call: list[int] = []

    def generate(self, **kw):
        self.calls += 1
        if self.engine is not None:
            self.checked_out_during_call.append(self.engine.pool.checkedout())
        time.sleep(self.seconds)
        return '{"patterns": [], "support_policy": {}}'


def _config(llm):
    cfg = load_config("config/")
    cfg.llm_client = llm
    cfg.worker.job_lease_seconds = 60
    return cfg


def _seed(session, job_id: str) -> None:
    set_memory_enabled(session, USER, True)
    for i in range(3):  # >= 2 active beliefs so K3 synthesis makes its LLM call too
        record_claim(
            session, USER, dimension="topic", key=f"k{i}",
            claim_text=f"claim {i}", confidence=0.8, session_id=f"s{i}",
        )
    with queue_write_scope():
        session.add(EvolutionJob(
            id=job_id, user_id=USER, evidence_watermark="{}",
            idempotency_key=f"{job_id}:ik", status="pending",
        ))
    session.commit()


def test_compute_longer_than_idle_in_transaction_timeout_still_completes(pg_app_url, monkeypatch):
    monkeypatch.setenv("MM_PG_IDLE_IN_TRANSACTION_TIMEOUT_MS", "2000")
    engine = build_engine(pg_app_url)
    llm = SleepyLLM(3.0, engine)
    cfg = _config(llm)
    factory = sessionmaker(bind=engine)
    try:
        with tenant_scope(*TENANT):
            session = factory()
            try:
                _seed(session, "idle-1")
                assert session.execute(text(
                    "SELECT setting FROM pg_catalog.pg_settings "
                    "WHERE name = 'idle_in_transaction_session_timeout'"
                )).scalar() == "2000"
                session.commit()
                token = _claim_job(session, "idle-1", lease_seconds=60)
                assert token == 1
                job_row = session.get(EvolutionJob, (*TENANT, "idle-1"))
                _process_single_job(session, job_row, cfg, lease_token=token)
                row = session.get(EvolutionJob, (*TENANT, "idle-1"))
                assert (row.status, row.error_code) == ("completed", ""), (row.status, row.error_code)
                assert llm.calls >= 1
                # no connection was held while the model was thinking
                assert llm.checked_out_during_call and max(llm.checked_out_during_call) == 0, (
                    llm.checked_out_during_call
                )
                # the publish landed (synthesis writes a snapshot / version) and
                # the seeded beliefs are intact
                assert session.query(Belief).filter(Belief.user_id == USER).count() >= 3
            finally:
                session.close()
    finally:
        engine.dispose()


def test_snapshot_objects_do_not_lazy_load_during_compute(pg_app_url):
    """Ending the read transaction must not leave the job row (or anything
    else the publish phase touches) expired: an attribute access during
    compute would re-open a transaction behind the model call."""
    engine = build_engine(pg_app_url)
    statements: list[str] = []
    in_llm = threading.Event()

    from sqlalchemy import event

    @event.listens_for(engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):
        if in_llm.is_set():
            statements.append(statement)

    class ProbingLLM(SleepyLLM):
        def generate(self, **kw):
            in_llm.set()
            try:
                return super().generate(**kw)
            finally:
                in_llm.clear()

    llm = ProbingLLM(0.05, engine)
    cfg = _config(llm)
    factory = sessionmaker(bind=engine)
    try:
        with tenant_scope(*TENANT):
            session = factory()
            try:
                _seed(session, "idle-2")
                token = _claim_job(session, "idle-2", lease_seconds=60)
                job_row = session.get(EvolutionJob, (*TENANT, "idle-2"))
                _process_single_job(session, job_row, cfg, lease_token=token)
                assert session.get(EvolutionJob, (*TENANT, "idle-2")).status == "completed"
            finally:
                session.close()
        assert statements == [], statements  # no SQL at all while the model ran
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# M3 batch-1 task 0: the lease is renewed again at the publish transaction
# ---------------------------------------------------------------------------


def test_compute_longer_than_lease_still_completes_via_publish_renewal(pg_app_url, monkeypatch):
    """A compute that outlives the lease, with nobody re-claiming: the
    renewal at the publish transaction extends the lease, so the job
    completes with a lease that was valid at publish time."""
    engine = build_engine(pg_app_url)
    llm = SleepyLLM(2.5, engine)  # compute 2.5 s > lease 1 s
    cfg = _config(llm)
    cfg.worker.job_lease_seconds = 1
    factory = sessionmaker(bind=engine)
    try:
        with tenant_scope(*TENANT):
            session = factory()
            try:
                _seed(session, "renew-1")
                session.commit()
                token = _claim_job(session, "renew-1", lease_seconds=1)
                assert token == 1
                job_row = session.get(EvolutionJob, (*TENANT, "renew-1"))
                _process_single_job(session, job_row, cfg, lease_token=token)
                row = session.get(EvolutionJob, (*TENANT, "renew-1"))
                assert row.status == "completed", row.error_code
                # the lease on record outlives the original 1 s window: the
                # publish-start renewal extended it (compute alone ran 2.5 s)
                assert row.lease_expires_at is not None
                assert row.completed_at is not None
                assert row.lease_expires_at > row.completed_at  # still held at completion
                session.close()
            finally:
                session.rollback()
    finally:
        engine.dispose()


def test_reclaim_during_compute_fails_the_publish_renewal(pg_app_url):
    """The lease is re-claimed by another worker DURING the compute: the
    pre-compute renewal succeeded (token still held then), the publish-start
    renewal finds a moved token and the whole transaction rolls back --
    zero memory writes."""
    engine = build_engine(pg_app_url)
    cfg = _config(None)
    cfg.worker.job_lease_seconds = 1
    factory = sessionmaker(bind=engine)

    state = {"reclaimed": False}

    class ReclaimingLLM:
        """Performs the hostile re-claim from 'inside the compute'."""

        def generate(self, **kw):
            with tenant_scope(*TENANT):
                s = factory()
                try:
                    s.query(EvolutionJob).filter(EvolutionJob.id == "renew-2").update(
                        {"lease_expires_at": datetime.now(UTC) - timedelta(seconds=5),
                         "status": "pending"},
                        synchronize_session=False,
                    )
                    s.commit()
                    tok = _claim_job(s, "renew-2", lease_seconds=60)
                    # the claim bumps the token: 1 (claim) -> 2 (A's
                    # pre-compute renewal) -> 3 (the hostile re-claim)
                    assert tok == 3
                    state["reclaimed"] = True
                finally:
                    s.close()
            return '{"patterns": [], "support_policy": {}}'

    cfg.llm_client = ReclaimingLLM()
    try:
        with tenant_scope(*TENANT):
            session = factory()
            try:
                _seed(session, "renew-2")
                session.commit()
                beliefs_before = session.query(Belief).filter(Belief.user_id == USER).count()
                token = _claim_job(session, "renew-2", lease_seconds=1)
                assert token == 1
                job_row = session.get(EvolutionJob, (*TENANT, "renew-2"))
                _process_single_job(session, job_row, cfg, lease_token=token)
                assert state["reclaimed"]
                s2 = factory()
                row = s2.get(EvolutionJob, (*TENANT, "renew-2"))
                assert row.lease_token == 3  # the hostile worker holds it
                beliefs_after = s2.query(Belief).filter(Belief.user_id == USER).count()
                assert beliefs_after == beliefs_before  # zero writes by the displaced worker
                s2.close()
                session.close()
            finally:
                session.rollback()
    finally:
        engine.dispose()
