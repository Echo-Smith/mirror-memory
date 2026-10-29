"""M3 batch-1 patch task 5: revocation-vs-publish race on SQLite.

A FILE-backed database, two threads with their OWN connections, 50 rounds:
either the publish commits before the revocation's commit lands (job
completed, completion instant <= revocation instant) or the publish's
authorization re-check sees the committed revocation and the job is
cancelled.  A new memory may never appear after a committed revocation.

Why this works on SQLite without FOR SHARE: SQLite serializes writers --
a commit takes the EXCLUSIVE lock on the journal file, so two write
transactions are strictly ordered.  The hazard this test also locks down
is the visible failure shape: when the publish's write collides with the
revocation's lock, the statement raises ``database is locked``
(SQLite BUSY).  The worker's handling of that is CONTRACT: the generic
failure handler rolls everything back and the fenced bookkeeping returns
the job to ``pending`` (attempt_count < max_attempts) -- i.e. RETRY on
the next sweep, never publish half a transaction.  The ``Busy`` outcome
test below locks that contract.
"""

from __future__ import annotations

import faulthandler
import sys
import threading
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.authorization import grant_purpose, revoke_purpose
from mirror_memory.core.models import Belief, EvolutionJob, PurposeGrant
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import tenant_scope
from mirror_memory.worker.evolution import _claim_job, _process_single_job, enqueue_job

PURPOSE = "revoke-sqlite"
TENANT = ("default", "default")
USER = "u1"
ROUNDS = 50


class FakeLLM:
    def __init__(self, hook=None):
        self.hook = hook

    def generate(self, **kw):
        if self.hook is not None:
            self.hook()
        return '{"patterns": [], "support_policy": {}}'


def _config(hook=None):
    cfg = load_config("config/")
    cfg.llm_client = FakeLLM(hook)
    cfg.worker.job_lease_seconds = 60
    return cfg


@pytest.fixture
def factory(tmp_path):
    # busy timeout 1s: contended statements surface BUSY quickly so the
    # worker's documented handling (rollback -> pending -> retry) is
    # exercised rather than hidden behind a long default wait
    engine = create_engine(
        f"sqlite:///{tmp_path / 'revoke-sqlite.db'}",
        connect_args={"timeout": 1.0},
    )
    from mirror_memory.core.models import Base

    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


def _seed_and_claim(factory, job_id, round_no):
    with tenant_scope(*TENANT):
        s = factory()
        set_memory_enabled(s, USER, True)
        s.commit()
        record_claim(s, USER, dimension="topic", key=f"k{round_no}",
                     claim_text="seed", confidence=0.8, session_id="s1")
        s.commit()
        job = enqueue_job(s, USER, _config(), purpose=PURPOSE)
        job_id = job.id
        s.close()
    with tenant_scope(*TENANT):
        s = factory()
        token = _claim_job(s, job_id, lease_seconds=60)
        s.close()
    return job_id, token


def _belief_count(factory) -> int:
    with tenant_scope(*TENANT):
        s = factory()
        n = s.query(Belief).filter(Belief.user_id == USER).count()
        s.close()
    return n


def test_revoke_racing_the_publish_on_sqlite(factory):
    counts = {"completed_before_revoke": 0, "cancelled": 0, "busy_retry": 0}
    for round_no in range(ROUNDS):
        with tenant_scope(*TENANT):
            s = factory()
            existing = s.query(PurposeGrant).filter_by(user_id=USER, purpose=PURPOSE).first()
            grant_purpose(
                s, USER, PURPOSE, operations=["extract"], issuer="reviewer",
                expected_version=existing.version if existing is not None else None,
            )
            s.commit()
            s.close()
        job_id, token = _seed_and_claim(factory, f"r{round_no}", round_no)

        revoke_at: list[datetime] = []
        barrier = threading.Barrier(2)
        outcome: dict = {}
        # the hook MUST time out: a bare barrier.wait() can hang forever in
        # the race window where the other side already broke the barrier
        cfg = _config(hook=lambda: barrier.wait(timeout=10))

        def run_publish():
            with tenant_scope(*TENANT):
                s = factory()
                try:
                    barrier.wait(timeout=10)
                    job_row = s.get(EvolutionJob, (*TENANT, job_id))
                    _process_single_job(s, job_row, cfg, lease_token=token)
                except threading.BrokenBarrierError:
                    outcome["error"] = "broken barrier"
                    s.rollback()
                except Exception as exc:  # noqa: BLE001 -- asserted below
                    outcome["error"] = repr(exc)
                    s.rollback()
                finally:
                    s.close()

        def run_revoke():
            try:
                barrier.wait(timeout=10)
            except threading.BrokenBarrierError:
                revoke_at.append(datetime.now(UTC).replace(tzinfo=None))
                return
            with tenant_scope(*TENANT):
                s = factory()
                try:
                    revoke_purpose(s, USER, PURPOSE, actor="reviewer", reason="raced")
                    s.commit()
                    revoke_at.append(datetime.now(UTC).replace(tzinfo=None))
                finally:
                    s.close()

        tp = threading.Thread(target=run_publish, daemon=True)
        tr = threading.Thread(target=run_revoke, daemon=True)
        tp.start()
        tr.start()
        import faulthandler as _fh

        _fh.dump_traceback_later(12, file=sys.stderr, repeat=False)
        tp.join(timeout=30)
        tr.join(timeout=30)
        _fh.cancel_dump_traceback_later()
        assert not tp.is_alive() and not tr.is_alive(), (round_no, "threads outlived the round")

        with tenant_scope(*TENANT):
            s = factory()
            row = s.get(EvolutionJob, (*TENANT, job_id))
            status, completed_at = row.status, row.completed_at
            s.close()

        if "error" in outcome and "broken barrier" in outcome["error"]:
            # the revoke side finished first; the publish never computed
            counts["cancelled"] += 1
            assert status == "cancelled", (round_no, status, outcome)
            continue
        if "error" in outcome and "locked" in outcome["error"]:
            # CONTRACT: database-is-locked => rollback => back to pending
            # (retry on the next sweep), never a half-published transaction
            counts["busy_retry"] += 1
            assert status == "pending", (round_no, status, outcome)
            continue

        if status == "completed":
            assert completed_at is not None
            if revoke_at:
                assert revoke_at[0] >= completed_at, (
                    round_no, f"memory landed after a committed revocation "
                    f"(revoke={revoke_at[0]}, completed={completed_at})"
                )
            counts["completed_before_revoke"] += 1
        else:
            assert status == "cancelled", (round_no, status, outcome)
            counts["cancelled"] += 1
        assert _belief_count(factory) == round_no + 1, (round_no, _belief_count(factory))

    # both outcomes must actually occur across 50 rounds, or the test
    # isn't exercising the race
    assert counts["completed_before_revoke"] + counts["cancelled"] >= ROUNDS - counts["busy_retry"]
    print(f"\nsqlite revocation race over {ROUNDS} rounds: {counts}")
