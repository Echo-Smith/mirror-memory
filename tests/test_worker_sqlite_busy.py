"""M3 batch-2 patch task 0: SQLite lock collisions do not burn attempts.

A second connection holding SQLite's write lock across several worker
sweeps must NOT push the job toward failed: a busy database is transient.
The worker's contract is rollback -> back to pending with the attempt
count restored -> deferred to the end of the queue (created_at bumped),
and error_code = database_is_locked.  Runs on SQLite (the end-side
shape); skipped on the PostgreSQL leg where lock waits are bounded by
lock_timeout instead.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base, Belief, EvolutionJob
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.worker.evolution import _claim_job, _process_single_job

TENANT = ("default", "default")
USER = "u1"


class FastLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


def _hold_write_lock(url, stop: threading.Event):
    """Cycle a write lock: acquire (BEGIN IMMEDIATE + INSERT), hold ~0.7 s,
    release -- so the worker's BUSY bookkeeping retries land once the lock
    frees, while the processing writes keep colliding."""
    import time as _time

    blocker = create_engine(url, connect_args={"timeout": 0.05}).connect()
    try:
        while not stop.is_set():
            try:
                blocker.execute(text("BEGIN IMMEDIATE"))
                blocker.execute(text(
                    "INSERT INTO mm_memory_preferences (tenant_id, app_id, "
                    "user_id, enabled, state_revision, updated_at) VALUES "
                    "('default', 'default', 'blocker', 1, 1, CURRENT_TIMESTAMP)"
                ))
                stop.wait(0.7)
                blocker.rollback()
                _time.sleep(0.05)
            except Exception:  # noqa: BLE001 -- BUSY: back off and retry
                try:
                    blocker.rollback()
                except Exception:  # noqa: BLE001
                    pass
                _time.sleep(0.05)
    finally:
        blocker.close()


def test_consecutive_sqlite_lock_hits_keep_the_job_pending(tmp_path):
    cfg = load_config("config/")
    cfg.llm_client = FastLLM()
    cfg.worker.job_lease_seconds = 60

    url = f"sqlite:///{tmp_path / 'busy.db'}"
    # tiny busy timeout: contended statements surface BUSY immediately
    eng = create_engine(url, connect_args={"timeout": 0.05})
    Base.metadata.create_all(eng)
    factory = sessionmaker(bind=eng)

    with tenant_scope(*TENANT):
        s = factory()
        with fixture_write_scope("busy test seed"):
            set_memory_enabled(s, USER, True)
            record_claim(s, USER, dimension="topic", key="k0",
                         claim_text="seed", confidence=0.8, session_id="s1")
            s.add(EvolutionJob(
                id="busy-1", user_id=USER, purpose="test",
                evidence_watermark="{}", idempotency_key="busy:ik", status="pending"))
        s.commit()
        s.close()

    with tenant_scope(*TENANT):
        s = factory()
        token = _claim_job(s, "busy-1", lease_seconds=60)
        assert token == 1
        job_row = s.get(EvolutionJob, (*TENANT, "busy-1"))
        s.close()

    stop = threading.Event()
    holder_thread = threading.Thread(target=_hold_write_lock, args=(url, stop), daemon=True)
    holder_thread.start()
    try:
        with tenant_scope(*TENANT):
            s = factory()
            for _ in range(3):
                _process_single_job(s, job_row, cfg, lease_token=token)
            s.close()
    finally:
        stop.set()
        holder_thread.join(timeout=10)

    with tenant_scope(*TENANT):
        s = factory()
        row = s.get(EvolutionJob, (*TENANT, "busy-1"))
        assert (row.status, row.error_code) == ("pending", "database_is_locked"), (
            row.status, row.error_code
        )
        # the claim's +1 is not burned: each BUSY restore puts the count
        # back to the pre-claim value, so repeated lock collisions never
        # accumulate toward failed (the old behaviour drifted 1 -> 0 -> ...
        # but would hit max_attempts on a mixed busy/work load)
        assert row.attempt_count == 0, (
            f"attempt_count drifted to {row.attempt_count}: BUSY burned attempts"
        )
        s.close()
    eng.dispose()
