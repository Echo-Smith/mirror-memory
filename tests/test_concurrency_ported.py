"""M2 batch-2 task 3: the 15 v2 concurrency tests, ported to v1 semantics.

Source: mirror-memory-v2 ``tests/integration/test_concurrency.py``
(archived, read-only).  PostgreSQL only -- SQLite has no real concurrency
(the v2 suite skips there too).

Classification (task requirement; one row per v2 test):

| #  | v2 test                                          | decision        | reason |
|----|--------------------------------------------------|-----------------|--------|
| 1  | I1 test_two_threads_race_same_job                | 直接移植        | v1 `_claim_job` is the same conditional-UPDATE claim; 2 real threads, 50 rounds |
| 2  | I1 test_complete_with_stale_token_fails          | 按 v1 语义改写  | v1 re-claims through the sweep (running+expired -> pending) then `_claim_job`; completion is fenced on lease_token |
| 3  | I2 test_revoke_between_claim_and_publish         | xfail(M3)       | v2 revokes an AuthorizationSnapshot; purpose-bound consent is M3 |
| 4  | I2 test_delete_between_claim_and_publish         | 按 v1 语义改写  | v2 scope-delete -> v1 `delete_user_memories` (rows + job gone, tombstone kept); the holder's publish dies on the lease fence and writes nothing |
| 5  | I3 test_two_workers_complete_only_holder_wins    | 直接移植        | v1 fencing: only the current lease_token holder completes; the loser's memory writes roll back with it; 50 rounds |
| 6  | I5 test_service_rejects_wrong_purpose            | xfail(M3)       | purpose-bound consent is M3 |
| 7  | I5 test_worker_rejects_wrong_purpose             | xfail(M3)       | purpose-bound consent is M3 |
| 8  | PI test_revoke_between_claim_and_publish         | xfail(M3)       | as #3 (interleaved variant) |
| 9  | PI test_delete_between_claim_and_publish         | 按 v1 语义改写  | as #4, interleaved after the compute-side renewal |
| 10 | PI test_lease_reclaimed_between_claim_and_publish| 直接移植        | re-claim bumps lease_token; the old holder's publish path fails its renewal (LeaseLost, handled: rollback, no writes) |
| 11 | PI test_no_interference_publish_succeeds         | 按 v1 语义改写  | v2 publishes atoms -> v1 `_process_single_job` completes the job |
| 12 | PI test_concurrent_publish_only_one_succeeds     | 直接移植        | two real threads publish against one revision; the CAS lets exactly one land; 50 rounds |
| 13 | BG test_expired_lease_not_reclaimed_publish_fails| 按 v1 语义改写  | **semantic difference**: v1 fences by token, not by wall clock -- an expired lease nobody re-claimed still belongs to its holder, so the publish COMPLETES; v2 fails it.  Asserted as v1 behaves; flagged in the report |
| 14 | BG test_evidence_deleted_between_claim_and_publish| 按 v1 语义改写 | v2 forget(evidence_ids) -> v1 evidence rows deleted; the Publisher's authority gate refuses (`evidence_missing`) |
| 15 | BG test_no_memory_with_revocation_does_not_mark_ready | 按 v1 语义改写 | the revoke half is M3; the v1 core -- a refused publish leaves the job cancelled, never completed -- is kept via the consent gate |

v1 addition required by the task: the version CAS under REAL thread
concurrency (``test_cas_real_concurrency_only_one_side_wins``, 8 threads,
50 rounds).

Race tests loop 50x so a pass is not luck.  Threads do not inherit
ContextVars: every thread body opens its own ``tenant_scope`` (the RLS
begin-hook reads it per transaction).  Every session is closed in a
``finally`` and a winning thread commits before it exits -- a session left
open holds its row locks and would stall the loser (and the next test's
schema rebuild).
"""

from __future__ import annotations

import contextlib
import os
import threading
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.conftest import refusing_authz

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Belief, DeletionTombstone, Evidence, EvolutionJob
from mirror_memory.core.proposal import TRANSITION_SUPPORT, StateTransitionProposal
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    cas_bump_state_revision,
    delete_user_memories,
    get_state_revision,
    record_claim,
    record_evidence,
    set_memory_enabled,
)
from mirror_memory.core.tenancy import engine_write_scope, queue_write_scope, tenant_scope
from mirror_memory.worker.evolution import (
    LeaseLost,
    _claim_job,
    _process_single_job,
    _renew_lease,
    _set_job_state,
)

pytestmark = pytest.mark.skipif(
    "postgresql" not in os.environ.get("MM_DATABASE_URL", ""),
    reason="real-concurrency tests require PostgreSQL",
)

LEASE_SECONDS = 60
USER = "u1"
TENANT = ("default", "default")
ROUNDS = 50


def make_config(lease_seconds=LEASE_SECONDS):
    cfg = load_config("config/")

    class FakeLLM:
        def generate(self, **kw):
            return '{"patterns": [], "support_policy": {}}'

    cfg.llm_client = FakeLLM()
    cfg.worker.job_lease_seconds = lease_seconds
    return cfg


@pytest.fixture
def factory(pg_app_url):
    engine = create_engine(pg_app_url)
    yield sessionmaker(bind=engine)
    engine.dispose()


@contextlib.contextmanager
def opened(factory, **kw):
    session = factory(**kw)
    try:
        yield session
    finally:
        session.close()  # rolls back whatever was left open


def pk(job_id: str) -> tuple:
    return (TENANT[0], TENANT[1], job_id)


def seed_beliefs(session, count=1):
    set_memory_enabled(session, USER, True)
    for i in range(count):
        record_claim(
            session, USER, dimension="topic", key=f"k{i}",
            claim_text=f"claim {i}", confidence=0.8, session_id=f"s{i}",
        )
    session.commit()


def seed_job(session, job_id, purpose="test"):
    with queue_write_scope():
        session.add(EvolutionJob(
            id=job_id, user_id=USER, evidence_watermark="{}",
            idempotency_key=f"{job_id}:ik", status="pending", purpose=purpose))
    session.commit()


def expire_and_sweep(session, job_id):
    """What the worker's crash-recovery sweep does to an expired lease:
    the row goes back to pending so another worker can claim it (the
    lease_token is deliberately left alone -- it is the fencing
    generation)."""
    session.query(EvolutionJob).filter(EvolutionJob.id == job_id).update(
        {"lease_expires_at": datetime.now(UTC) - timedelta(seconds=5)},
        synchronize_session=False,
    )
    session.query(EvolutionJob).filter(
        EvolutionJob.id == job_id, EvolutionJob.status == "running"
    ).update({"status": "pending"}, synchronize_session=False)
    session.commit()


def expire_only(session, job_id):
    session.query(EvolutionJob).filter(EvolutionJob.id == job_id).update(
        {"lease_expires_at": datetime.now(UTC) - timedelta(seconds=5)},
        synchronize_session=False,
    )
    session.commit()


def belief_count(session) -> int:
    return session.query(Belief).filter(Belief.user_id == USER).count()


# ---------------------------------------------------------------------------
# 1. two threads race the same job -> exactly one claim
# ---------------------------------------------------------------------------


def test_two_threads_race_same_job(factory):
    for round_no in range(ROUNDS):
        job_id = f"race-{round_no}"
        with tenant_scope(*TENANT), opened(factory) as s:
            seed_beliefs(s)
            seed_job(s, job_id)
        results = []
        barrier = threading.Barrier(2)

        def try_claim(n):
            with tenant_scope(*TENANT), opened(factory) as session:
                barrier.wait(timeout=10)
                results.append((n, _claim_job(session, job_id, lease_seconds=LEASE_SECONDS)))

        threads = [threading.Thread(target=try_claim, args=(n,)) for n in (1, 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert len(results) == 2, (round_no, results)
        winners = [n for n, token in results if token is not None]
        assert len(winners) == 1, (round_no, results)
        with tenant_scope(*TENANT), opened(factory) as s:
            row = s.get(EvolutionJob, pk(job_id))
            assert (row.status, row.attempt_count, row.lease_token) == ("running", 1, 1)


# ---------------------------------------------------------------------------
# 2. completion with a stale token fails (sweep + re-claim, v1 shape)
# ---------------------------------------------------------------------------


def test_complete_with_stale_token_fails(factory):
    for round_no in range(ROUNDS):
        job_id = f"stale-{round_no}"
        with tenant_scope(*TENANT), opened(factory) as s:
            seed_beliefs(s)
            seed_job(s, job_id)
        with tenant_scope(*TENANT), opened(factory) as s1, opened(factory) as s2:
            assert _claim_job(s1, job_id, lease_seconds=1) == 1
            with opened(factory) as s_exp:
                expire_and_sweep(s_exp, job_id)
            assert _claim_job(s2, job_id, lease_seconds=LEASE_SECONDS) == 2
            with pytest.raises(LeaseLost):
                _set_job_state(s1, job_id, 1, {"status": "completed"})
            _set_job_state(s2, job_id, 2, {
                "status": "completed", "completed_at": datetime.now(UTC), "error_code": "",
            })
        with tenant_scope(*TENANT), opened(factory) as s:
            row = s.get(EvolutionJob, pk(job_id))
            assert (row.status, row.lease_token) == ("completed", 2)


# ---------------------------------------------------------------------------
# 4 / 9. delete between claim and publish (v1 deletion semantics)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("renew_first", [False, True], ids=["before-compute", "mid-compute"])
def test_delete_between_claim_and_publish_creates_nothing(factory, renew_first):
    cfg = make_config()
    job_id = "del-1"
    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s, count=2)
        seed_job(s, job_id)
    with tenant_scope(*TENANT), opened(factory, expire_on_commit=False) as holder:
        token = _claim_job(holder, job_id, lease_seconds=LEASE_SECONDS)
        assert token == 1
        job_row = holder.get(EvolutionJob, pk(job_id))
        holder.commit()
        if renew_first:
            assert _renew_lease(holder, job_id, token, LEASE_SECONDS)
        # interfere: the user's data is erased -- rows AND the job -- and
        # the content-free tombstone is written
        with opened(factory) as s_del:
            counts = delete_user_memories(s_del, USER)
            s_del.commit()
        assert counts["beliefs"] == 2 and counts["evolution_jobs"] == 1
        # the holder walks the publish path: its renewal finds no job row,
        # LeaseLost is handled inside (rollback), nothing is written
        _process_single_job(holder, job_row, cfg, lease_token=token)
        with opened(factory) as s_check:
            assert belief_count(s_check) == 0
            assert s_check.get(EvolutionJob, pk(job_id)) is None
            assert s_check.query(DeletionTombstone).filter(
                DeletionTombstone.user_id == USER
            ).count() == 1


# ---------------------------------------------------------------------------
# 5. two workers complete: only the current holder wins
# ---------------------------------------------------------------------------


def test_two_workers_complete_only_holder_wins(factory):
    for round_no in range(ROUNDS):
        job_id = f"fence-{round_no}"
        with tenant_scope(*TENANT), opened(factory) as s:
            seed_beliefs(s)
            seed_job(s, job_id)
        with tenant_scope(*TENANT), opened(factory) as s1, opened(factory) as s2:
            token1 = _claim_job(s1, job_id, lease_seconds=1)
            assert token1 == 1
            with opened(factory) as s_exp:
                expire_and_sweep(s_exp, job_id)
            token2 = _claim_job(s2, job_id, lease_seconds=LEASE_SECONDS)
            assert token2 == 2
            # W1 (stale) writes memory AND tries to complete in one
            # transaction: the fence fires and the ghost rolls back with it
            with engine_write_scope():
                s1.add(Belief(user_id=USER, dimension="topic", key=f"ghost-{round_no}",
                              claim_text="must not survive", confidence=0.9))
            with pytest.raises(LeaseLost):
                _set_job_state(s1, job_id, token1, {"status": "completed"})
            _set_job_state(s2, job_id, token2, {
                "status": "completed", "completed_at": datetime.now(UTC), "error_code": "",
            })
        with tenant_scope(*TENANT), opened(factory) as s:
            assert s.query(Belief).filter(Belief.key == f"ghost-{round_no}").count() == 0
            assert s.get(EvolutionJob, pk(job_id)).status == "completed"


# ---------------------------------------------------------------------------
# 10. lease reclaimed between claim and publish
# ---------------------------------------------------------------------------


def test_lease_reclaimed_between_claim_and_publish(factory):
    cfg = make_config()
    job_id = "reclaim-1"
    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s)
        seed_job(s, job_id)
    with tenant_scope(*TENANT), opened(factory, expire_on_commit=False) as holder:
        token1 = _claim_job(holder, job_id, lease_seconds=1)
        assert token1 == 1
        job_row = holder.get(EvolutionJob, pk(job_id))
        holder.commit()
        with opened(factory) as s_exp:
            expire_and_sweep(s_exp, job_id)
        with opened(factory) as s2:
            assert _claim_job(s2, job_id, lease_seconds=LEASE_SECONDS) == 2
        # the old holder walks the publish path: renewal fails (token moved),
        # LeaseLost is handled with a rollback, nothing is written
        _process_single_job(holder, job_row, cfg, lease_token=token1)
        with opened(factory) as s_check:
            assert belief_count(s_check) == 1  # only the seed
            row = s_check.get(EvolutionJob, pk(job_id))
            assert (row.status, row.lease_token) == ("running", 2)  # untouched by W1
        with opened(factory) as s3:
            _set_job_state(s3, job_id, 2, {
                "status": "completed", "completed_at": datetime.now(UTC), "error_code": "",
            })


# ---------------------------------------------------------------------------
# 11. no interference: publish completes
# ---------------------------------------------------------------------------


def test_no_interference_publish_succeeds(factory):
    cfg = make_config()
    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s)
        seed_job(s, "clean-1")
    with tenant_scope(*TENANT), opened(factory) as s:
        token = _claim_job(s, "clean-1", lease_seconds=LEASE_SECONDS)
        assert token == 1
        _process_single_job(s, s.get(EvolutionJob, pk("clean-1")), cfg, lease_token=token)
        row = s.get(EvolutionJob, pk("clean-1"))
        assert row.status == "completed", row.error_code


# ---------------------------------------------------------------------------
# 12. concurrent publish, real threads: exactly one lands
# ---------------------------------------------------------------------------


def test_concurrent_publish_only_one_succeeds(factory):
    cfg = make_config()
    for round_no in range(ROUNDS):
        with tenant_scope(*TENANT), opened(factory) as s:
            seed_beliefs(s)
            belief_id = s.query(Belief).filter(Belief.user_id == USER).first().id
            revision = get_state_revision(s, USER)
        results = []
        barrier = threading.Barrier(2)

        def publish(n):
            with tenant_scope(*TENANT), opened(factory) as session:
                barrier.wait(timeout=10)
                decision = Publisher(session, cfg).publish(
                    StateTransitionProposal(
                        transition=TRANSITION_SUPPORT, user_id=USER,
                        target_belief_id=belief_id, expected_revision=revision,
                        payload={"claim_text": f"writer {n}"},
                    )
                )
                # the winner commits at once: its CAS row lock is what the
                # loser's UPDATE is waiting on
                if decision.committed:
                    session.commit()
                else:
                    session.rollback()
                results.append((n, decision.committed, decision.reason))

        threads = [threading.Thread(target=publish, args=(n,)) for n in (1, 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        assert len(results) == 2, (round_no, results)
        committed = [r for r in results if r[1]]
        refused = [r for r in results if not r[1]]
        assert len(committed) == 1 and len(refused) == 1, (round_no, results)
        assert "stale_revision" in refused[0][2], refused
        with tenant_scope(*TENANT), opened(factory) as s:
            assert get_state_revision(s, USER) == revision + 1
            # reset for the next round's fresh belief set
            delete_user_memories(s, USER)
            s.commit()


# ---------------------------------------------------------------------------
# 13. expired lease, nobody reclaims (v1: the holder still owns it)
# ---------------------------------------------------------------------------


def test_expired_lease_not_reclaimed_holder_still_completes(factory):
    """v1 fences by lease_token, not by wall clock.  Until another worker
    re-claims (bumping the token) an expired lease still belongs to its
    holder, so the publish path completes.  v2 fails this publish; the
    difference is recorded in the module table and the report."""
    cfg = make_config()
    job_id = "expire-1"
    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s)
        seed_job(s, job_id)
    with tenant_scope(*TENANT), opened(factory, expire_on_commit=False) as holder:
        token = _claim_job(holder, job_id, lease_seconds=1)
        assert token == 1
        job_row = holder.get(EvolutionJob, pk(job_id))
        holder.commit()
        with opened(factory) as s_exp:
            expire_only(s_exp, job_id)
        _process_single_job(holder, job_row, cfg, lease_token=token)
        with opened(factory) as s_check:
            row = s_check.get(EvolutionJob, pk(job_id))
            assert (row.status, row.lease_token) == ("completed", 1), row.error_code


# ---------------------------------------------------------------------------
# 14. evidence deleted between claim and publish: authority gate refuses
# ---------------------------------------------------------------------------


def test_evidence_deleted_between_claim_and_publish(factory):
    with tenant_scope(*TENANT):
        with opened(factory) as s:
            set_memory_enabled(s, USER, True)
            belief, _ = record_claim(
                s, USER, dimension="topic", key="k", claim_text="c",
                confidence=0.8, session_id="s1",
            )
            evidence = record_evidence(s, USER, ref="e1", content="supporting")
            s.commit()
            belief_id, evidence_id = belief.id, evidence.id
            revision = get_state_revision(s, USER)
        with opened(factory) as s_del:  # interfere: the evidence is gone
            s_del.query(Evidence).filter(Evidence.user_id == USER).delete(synchronize_session=False)
            s_del.commit()
        with opened(factory) as s_pub:
            before = [(b.id, b.claim_text) for b in s_pub.query(Belief).filter(Belief.user_id == USER)]
            decision = Publisher(s_pub, make_config()).publish(
                StateTransitionProposal(
                    transition=TRANSITION_SUPPORT, user_id=USER,
                    target_belief_id=belief_id, expected_revision=revision,
                    evidence_ids=[evidence_id], payload={"claim_text": "updated"},
                )
            )
            assert not decision.committed
            assert decision.reason == "evidence_missing"
            s_pub.rollback()
            after = [(b.id, b.claim_text) for b in s_pub.query(Belief).filter(Belief.user_id == USER)]
            assert after == before


# ---------------------------------------------------------------------------
# 15. refused publish leaves the job cancelled, never completed
# ---------------------------------------------------------------------------


def test_refused_publish_does_not_mark_completed(factory):
    cfg = make_config()
    job_id = "refuse-1"
    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s)
        seed_job(s, job_id)
    with tenant_scope(*TENANT), opened(factory, expire_on_commit=False) as holder:
        token = _claim_job(holder, job_id, lease_seconds=LEASE_SECONDS)
        assert token == 1
        job_row = holder.get(EvolutionJob, pk(job_id))
        holder.commit()
        # consent withdrawn after the claim (v1's analogue of revocation
        # for collection; erasure is exempt, collection is not)
        with opened(factory) as s_consent:
            set_memory_enabled(s_consent, USER, False)
            s_consent.commit()
        _process_single_job(holder, job_row, cfg, lease_token=token)
        with opened(factory) as s_check:
            row = s_check.get(EvolutionJob, pk(job_id))
            assert (row.status, row.error_code) == ("cancelled", "memory_disabled")


# ---------------------------------------------------------------------------
# 3 / 8 / 6 / 7 (M3): purpose-bound consent, now implemented
# ---------------------------------------------------------------------------

from mirror_memory.core.authorization import PurposeDenied, grant_purpose, revoke_purpose  # noqa: E402

PURPOSE = "revoke-test"


def pg_url() -> str:
    return os.environ["MM_DATABASE_URL"]


def test_revoke_between_claim_and_publish(factory):
    """I2 (v2 shape, v1 semantics): the job is claimed, the purpose grant is
    revoked mid-compute, the publish must not land.  The publish re-check
    (FOR SHARE on the grant row) cancels the job with zero memory written."""
    cfg = make_config()
    job_id = "i2-revoke-1"
    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s)
        grant_purpose(s, USER, PURPOSE, operations=["extract"], issuer="reviewer")
        seed_job(s, job_id, purpose=PURPOSE)
    with tenant_scope(*TENANT), opened(factory, expire_on_commit=False) as holder:
        token = _claim_job(holder, job_id, lease_seconds=LEASE_SECONDS)
        assert token == 1
        job_row = holder.get(EvolutionJob, pk(job_id))
        holder.commit()
        # interfere: the grant is revoked while the worker "computes"
        with opened(factory) as s_rev:
            revoke_purpose(s_rev, USER, PURPOSE, actor="reviewer", reason="i2")
            s_rev.commit()
        _process_single_job(holder, job_row, cfg, lease_token=token)
        with opened(factory) as s_check:
            row = s_check.get(EvolutionJob, pk(job_id))
            assert row.status == "cancelled", (row.status, row.error_code)
            assert row.error_code in ("authorization_revoked", "revoked")
            assert belief_count(s_check) == 1  # the seed; nothing published


def test_revoke_between_claim_and_publish_interleaved(factory):
    """PI variant: the revocation lands after the compute-side renewal but
    before the publish; same outcome (the publish re-check), and the seed
    belief set is all that exists."""
    cfg = make_config()
    job_id = "pi-revoke-1"
    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s)
        grant_purpose(s, USER, PURPOSE, operations=["extract"], issuer="reviewer")
        seed_job(s, job_id, purpose=PURPOSE)
    with tenant_scope(*TENANT), opened(factory, expire_on_commit=False) as holder:
        token = _claim_job(holder, job_id, lease_seconds=LEASE_SECONDS)
        assert token == 1
        job_row = holder.get(EvolutionJob, pk(job_id))
        holder.commit()
        # the compute-side renewal succeeds (still the holder) ...
        with opened(factory) as s_mid:
            assert _renew_lease(s_mid, job_id, token, LEASE_SECONDS)
        # ... then the grant is revoked ...
        with opened(factory) as s_rev:
            revoke_purpose(s_rev, USER, PURPOSE, actor="reviewer", reason="pi")
            s_rev.commit()
        # ... and the publish must not land
        _process_single_job(holder, job_row, cfg, lease_token=token)
        with opened(factory) as s_check:
            row = s_check.get(EvolutionJob, pk(job_id))
            assert row.status == "cancelled", (row.status, row.error_code)
            assert row.error_code in ("authorization_revoked", "revoked")
            assert belief_count(s_check) == 1


def test_service_rejects_wrong_purpose(factory):
    """I5: the grant covers purpose X; calling with purpose Y is refused
    (no_grant), at the service layer, before any business statement."""
    from mirror_memory.core.authorization import PurposeDenied, grant_purpose
    from mirror_memory.api import MemoryEngine

    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s)
        grant_purpose(s, USER, "analytics", operations=["observe"], issuer="reviewer")
        s.commit()

    from tests.conftest import _pg_runtime_url

    eng = MemoryEngine(
        config=make_config(), database_url=_pg_runtime_url(),
        tenant_id=TENANT[0], app_id=TENANT[1], default_purpose=None,
    )
    try:
        # observe under the granted purpose works ...
        eng.observe(user_id=USER, session_id="s9", text="granted path", purpose="analytics")
        # ... and every operation under a purpose it does not carry is
        # refused.  The harness lazy-grant is OFF for the probes -- the
        # refusal is the point (I5).
        with refusing_authz():
            for call in (
                lambda p: eng.recall(user_id=USER, purpose=p),
                lambda p: eng.correct(user_id=USER, belief_id=1, new_claim_text="x", purpose=p),
                lambda p: eng.explain(user_id=USER, belief_id=1, purpose=p),
                lambda p: eng.observe(user_id=USER, session_id="s9", text="x", purpose=p),
            ):
                with pytest.raises(PurposeDenied, match="no_grant"):
                    call("memory_management")
    finally:
        eng._engine.dispose()


def test_worker_rejects_wrong_purpose(factory):
    """I5 at the worker: the job carries its own purpose; a worker running a
    job whose purpose has no grant is cancelled.  The harness lazy-grant is
    OFF here -- the refusal is the point."""
    cfg = make_config()
    job_id = "i5-worker-1"
    with tenant_scope(*TENANT), opened(factory) as s:
        seed_beliefs(s)
        # the grant covers only "worker-purpose" ...
        grant_purpose(s, USER, "worker-purpose", operations=["extract"], issuer="reviewer")
        s.commit()
        # ... but the job was enqueued under a DIFFERENT purpose
        seed_job(s, job_id, purpose="other-purpose")
    with tenant_scope(*TENANT), opened(factory, expire_on_commit=False) as holder:
        token = _claim_job(holder, job_id, lease_seconds=LEASE_SECONDS)
        assert token == 1
        job_row = holder.get(EvolutionJob, pk(job_id))
        holder.commit()
        with refusing_authz():
            _process_single_job(holder, job_row, cfg, lease_token=token)
        with opened(factory) as s_check:
            row = s_check.get(EvolutionJob, pk(job_id))
            assert (row.status, row.error_code) == ("cancelled", "no_grant")
            assert belief_count(s_check) == 1

