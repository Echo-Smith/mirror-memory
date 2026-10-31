"""M3 batch-1 task 4: revocation races the publish -- and loses, every time.

Two 50-round real-thread suites on PostgreSQL:

1. revoke DURING the compute (between claim and publish): the publish's
   FOR SHARE re-check sees the committed revocation and the job is
   cancelled with zero memory written.
2. revoke concurrent with the publish (barrier-synchronized): either the
   publish commits first (its FOR SHARE blocked the revocation's UPDATE
   until the transaction ended) or the revocation commits first and the
   re-check cancels the job.  The invariant each round asserts: never
   "job completed AND the revocation committed before the job's
   completion instant" -- i.e. no memory may land after a committed
   revocation.

Why FOR SHARE (PostgreSQL): the re-check SELECT takes a shared row lock
on the grant; the revocation's UPDATE needs the conflicting exclusive
lock.  Without it, the hole is this interleaving under READ COMMITTED:

    publish tx: re-check reads grant (version 1, not revoked) -> decision made
    revoke tx:  UPDATE ... SET revoked_at = now(); COMMIT   <-- between check and commit
    publish tx: Publisher writes the memory; COMMIT         <-- stale decision lands

The publish's LATER statements would even SEE the revocation (new
snapshot per statement), but the decision was already made -- the memory
lands after a committed revocation.  With FOR SHARE the revoke blocks on
the shared lock until the publish transaction ends: strictly ordered.
SQLite serializes writers by itself (rollback-journal EXCLUSIVE at
commit); a FOR SHARE there is a syntax error, so it is not emitted -- a
revocation that commits mid-publish makes the publish's own write fail
with BUSY and roll back, which upholds the same invariant.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.authorization import grant_purpose, revoke_purpose
from mirror_memory.core.models import Belief, EvolutionJob, PurposeGrantEvent, PurposeGrant
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import queue_write_scope, tenant_scope
from mirror_memory.worker.evolution import _claim_job, _process_single_job, enqueue_job

pytestmark = pytest.mark.skipif(
    "postgresql" not in __import__("os").environ.get("MM_DATABASE_URL", ""),
    reason="real-concurrency revocation tests require PostgreSQL",
)

PURPOSE = "revoke-test"
TENANT = ("default", "default")


class FakeLLM:
    def __init__(self, hook=None, seconds=0.0):
        self.hook = hook
        self.seconds = seconds

    def generate(self, **kw):
        if self.hook is not None:
            self.hook()
        if self.seconds:
            import time

            time.sleep(self.seconds)
        return '{"patterns": [], "support_policy": {}}'


def _config(hook=None, seconds=0.0):
    cfg = load_config("config/")
    cfg.llm_client = FakeLLM(hook, seconds)
    cfg.worker.job_lease_seconds = 60
    return cfg


@pytest.fixture
def factory(pg_app_url):
    # lock_timeout: the race's statements are ms-level; a contention wait
    # must error fast (and be retried) instead of hanging the round -- a
    # hung publisher outlives the join timeout and poisons the rebuild
    engine = create_engine(
        pg_app_url, connect_args={"options": "-c lock_timeout=5000"})
    yield sessionmaker(bind=engine)
    engine.dispose()


def _seed_user(factory):
    with tenant_scope(*TENANT):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        s.close()


def _grant_current(factory):
    """Idempotent per round: grant when missing, else CAS-update (which
    also clears a previous round's revocation)."""
    from mirror_memory.core.models import PurposeGrant

    with tenant_scope(*TENANT):
        s = factory()
        existing = (
            s.query(PurposeGrant)
            .filter_by(user_id="u1", purpose=PURPOSE)
            .first()
        )
        grant_purpose(
            s, "u1", PURPOSE, operations=["extract"], issuer="reviewer",
            expected_version=existing.version if existing is not None else None,
        )
        s.commit()
        s.close()


def _revoke(factory, done: list):
    with tenant_scope(*TENANT):
        s = factory()
        revoke_purpose(s, "u1", PURPOSE, actor="reviewer", reason="raced")
        s.commit()
        s.close()
    done.append(datetime.now(UTC).replace(tzinfo=None))


def _belief_count(factory) -> int:
    with tenant_scope(*TENANT):
        s = factory()
        n = s.query(Belief).filter(Belief.user_id == "u1").count()
        s.close()
    return n


# ---------------------------------------------------------------------------
# 1. revoke between claim and publish, 50 rounds
# ---------------------------------------------------------------------------


def test_revoke_during_compute_cancels_the_publish(factory):
    _seed_user(factory)
    for round_no in range(50):
        _grant_current(factory)
        with tenant_scope(*TENANT):
            s = factory()
            record_claim(s, "u1", dimension="topic", key=f"k{round_no}",
                         claim_text="seed", confidence=0.8, session_id="s1")
            s.commit()
            job = enqueue_job(s, "u1", _config(), purpose=PURPOSE)
            job_id = job.id  # read before the session closes (expiry)
            s.close()
        del job
        with tenant_scope(*TENANT):
            s = factory()
            token = _claim_job(s, job_id, lease_seconds=60)
            assert token is not None
            s.close()

        revoked_at = []
        cfg = _config(hook=lambda: _revoke(factory, revoked_at) if not revoked_at else None)

        with tenant_scope(*TENANT):
            s = factory()
            job_row = s.get(EvolutionJob, (*TENANT, job_id))
            _process_single_job(s, job_row, cfg, lease_token=token)
            s.close()

        with tenant_scope(*TENANT):
            s = factory()
            row = s.get(EvolutionJob, (*TENANT, job_id))
            assert row.status == "cancelled", (round_no, row.status, row.error_code)
            # the hook fires inside formulate, i.e. after the snapshot
            # authorize: the publish re-check refuses with "revoked"
            assert row.error_code == "revoked", (round_no, row.error_code)
            s.close()
        # the seed belief is all that exists; nothing published after revoke
        assert _belief_count(factory) == round_no + 1, (round_no, _belief_count(factory))


# ---------------------------------------------------------------------------
# 2. revoke racing the publish, 50 rounds
# ---------------------------------------------------------------------------


def test_revoke_racing_the_publish_never_lands_after_revocation(factory):
    _seed_user(factory)
    for round_no in range(50):
        _grant_current(factory)
        with tenant_scope(*TENANT):
            s = factory()
            record_claim(s, "u1", dimension="topic", key=f"k{round_no}",
                         claim_text="seed", confidence=0.8, session_id="s1")
            s.commit()
            job = enqueue_job(s, "u1", _config(), purpose=PURPOSE)
            job_id = job.id  # read before the session closes (expiry)
            s.close()
        del job
        with tenant_scope(*TENANT):
            s = factory()
            token = _claim_job(s, job_id, lease_seconds=60)
            assert token is not None
            s.close()

        revoke_done: list[datetime] = []
        barrier = threading.Barrier(2)
        # NO mid-compute barrier hook: the waits per round are 3 when the
        # publish reaches generate (init + hook) and 2 when it is
        # cancelled early -- an odd count shifts the reusable barrier's
        # pairing across rounds until a no-timeout wait strands forever
        # (CI: rounds 24 / 40 / 47, publish outlived the 120s join).  The
        # race is the round-start barrier plus the 20ms compute window;
        # the invariant asserted below does not need a second sync.
        cfg = _config(hook=None, seconds=0.02)

        outcome: dict = {}

        def run_publish():
            # a lock-timeout abort (CI load) rolls back ENTIRELY: retry the
            # round's publish instead of recording a half-finished attempt
            import time as _time

            from sqlalchemy.exc import OperationalError

            with tenant_scope(*TENANT):
                for attempt in range(5):
                    s = factory()
                    try:
                        barrier.wait(timeout=10)
                        job_row = s.get(EvolutionJob, (*TENANT, job_id))
                        _process_single_job(s, job_row, cfg, lease_token=token)
                        return
                    except OperationalError:
                        s.rollback()
                        if attempt == 4:
                            raise
                        _time.sleep(0.05 * (attempt + 1))
                    except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                        outcome["publish_error"] = repr(exc)
                        s.rollback()
                        return
                    finally:
                        s.close()

        def run_revoke():
            import time as _time

            from sqlalchemy.exc import OperationalError

            barrier.wait(timeout=10)
            for attempt in range(5):
                try:
                    _revoke(factory, revoke_done)
                    return
                except OperationalError:
                    if attempt == 4:
                        raise
                    _time.sleep(0.05 * (attempt + 1))

        tp = threading.Thread(target=run_publish, daemon=True)
        tr = threading.Thread(target=run_revoke, daemon=True)
        tp.start()
        tr.start()
        tp.join(timeout=120)
        tr.join(timeout=120)
        assert not tp.is_alive() and not tr.is_alive(), (
            round_no, "a racing thread outlived its round -- its locks would "
            "poison the next test's schema rebuild"
        )

        with tenant_scope(*TENANT):
            s = factory()
            row = s.get(EvolutionJob, (*TENANT, job_id))
            status = row.status
            completed_at = row.completed_at
            s.close()
        assert status in ("completed", "cancelled"), (round_no, status, outcome)
        assert "publish_error" not in outcome or "Busy" in outcome.get("publish_error", ""), (
            round_no, outcome
        )
        if status == "completed" and revoke_done:
            # the publish won: its transaction (and the memory it wrote)
            # must predate the revocation's commit
            assert completed_at is not None
            assert revoke_done[0] >= completed_at, (
                round_no, f"memory landed after a committed revocation "
                f"(revoke={revoke_done[0]}, completed={completed_at})"
            )
        if status == "cancelled":
            with tenant_scope(*TENANT):
                s = factory()
                code = s.get(EvolutionJob, (*TENANT, job_id)).error_code
                s.close()
            # the revocation may win before the snapshot authorize (reason
            # "revoked") or between the snapshot and the publish (the
            # publish re-check cancels with "authorization_revoked")
            assert code in ("authorization_revoked", "revoked"), (round_no, code)
        # whatever the order, the seed belief is all that exists
        assert _belief_count(factory) == round_no + 1, (round_no, _belief_count(factory))


def row_status_is(factory, job_id, error_code) -> bool:
    with tenant_scope(*TENANT):
        s = factory()
        row = s.get(EvolutionJob, (*TENANT, job_id))
        ok = row.error_code == error_code
        s.close()
    return ok


# ---------------------------------------------------------------------------
# Patch task 3: the publish re-check is CURRENT-ALLOWED only
# ---------------------------------------------------------------------------


def test_renewal_between_claim_and_publish_still_publishes(factory):
    """A renewal (or an added operation) during the compute is fine: the
    version moved but extract is still allowed -- the publish lands, and
    the audit trail records both versions."""
    _seed_user(factory)
    with tenant_scope(*TENANT):
        s = factory()
        grant_purpose(s, "u1", PURPOSE, operations=["extract"], issuer="reviewer")
        s.commit()
        record_claim(s, "u1", dimension="topic", key="k0",
                     claim_text="seed", confidence=0.8, session_id="s1")
        s.commit()
        job = enqueue_job(s, "u1", _config(), purpose=PURPOSE)
        job_id = job.id
        s.close()
    with tenant_scope(*TENANT):
        s = factory()
        token = _claim_job(s, job_id, lease_seconds=60)
        s.close()

    from mirror_memory.core.models import PurposeGrantEvent

    renewed: list[int] = []
    cfg = _config(hook=lambda: (
        (lambda: (
            (lambda s: (
                grant_purpose(s, "u1", PURPOSE,
                              operations=["extract", "recall"], issuer="reviewer",
                              expected_version=renewed[0] if renewed else 1),
                s.commit(),
                renewed.append(1),
            ))(factory()),
        ))() if not renewed else None,
    ))

    with tenant_scope(*TENANT):
        s = factory()
        job_row = s.get(EvolutionJob, (*TENANT, job_id))
        _process_single_job(s, job_row, cfg, lease_token=token)
        s.close()

    with tenant_scope(*TENANT):
        s = factory()
        row = s.get(EvolutionJob, (*TENANT, job_id))
        assert (row.status, row.error_code) == ("completed", ""), (row.status, row.error_code)
        grants = s.query(PurposeGrant).filter_by(user_id="u1", purpose=PURPOSE).first()
        snapshot_or_publish = s.query(PurposeGrantEvent).filter_by(
            user_id="u1", purpose=PURPOSE, action="invoked",
            operation="extract", decision="published",
        ).all()
        assert snapshot_or_publish, "the publish invocation was not audited"
        event = snapshot_or_publish[-1]
        assert event.reason.startswith("snapshot_version=1 publish_version="), event.reason
        assert grants.version >= 2  # renewed mid-compute
        s.close()


def test_extract_removed_between_claim_and_publish_cancels(factory):
    """The operation (not the whole grant) is removed mid-compute: the
    publish re-check refuses with the decision's reason and nothing lands."""
    _seed_user(factory)
    with tenant_scope(*TENANT):
        s = factory()
        grant_purpose(s, "u1", PURPOSE, operations=["extract", "recall"], issuer="reviewer")
        s.commit()
        record_claim(s, "u1", dimension="topic", key="k0",
                     claim_text="seed", confidence=0.8, session_id="s1")
        s.commit()
        job = enqueue_job(s, "u1", _config(), purpose=PURPOSE)
        job_id = job.id
        s.close()
    with tenant_scope(*TENANT):
        s = factory()
        token = _claim_job(s, job_id, lease_seconds=60)
        s.close()

    removed: list[bool] = []
    cfg = _config(hook=lambda: (
        (lambda: (
            (lambda s: (
                grant_purpose(s, "u1", PURPOSE, operations=["recall"],
                              issuer="reviewer", expected_version=1),
                s.commit(),
                removed.append(True),
            ))(factory()),
        ))() if not removed else None,
    ))

    with tenant_scope(*TENANT):
        s = factory()
        job_row = s.get(EvolutionJob, (*TENANT, job_id))
        _process_single_job(s, job_row, cfg, lease_token=token)
        s.close()

    with tenant_scope(*TENANT):
        s = factory()
        row = s.get(EvolutionJob, (*TENANT, job_id))
        assert (row.status, row.error_code) == ("cancelled", "operation_not_allowed"), (
            row.status, row.error_code
        )
        s.close()
