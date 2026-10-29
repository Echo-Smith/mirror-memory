"""Async evolution worker -- domain-agnostic.

Six-node pipeline:
1. load_evidence    -- read active beliefs, background, session count
2. formulate        -- one LLM call for candidate understanding
3. validate         -- deterministic checks (forbidden labels, third-party)
4. compile_policy   -- compile support policy from candidate
5. synthesize       -- K3 understanding synthesis
6. persist_snapshot  -- save versioned snapshot

All domain-specific content (forbidden labels, third-party markers,
system prompts) comes from ``config``.  Zero domain coupling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import OperationalError

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.authorization import Decision, PurposeDenied
from mirror_memory.core.constants import (
    FORMULATE_MIN_CONFIDENCE,
    JOB_BATCH_LIMIT,
    LOAD_EVIDENCE_LIMIT,
)
from mirror_memory.core import provenance
from mirror_memory.core.models import Belief, EvolutionJob
from mirror_memory.core.utils import truncate_id
from mirror_memory.worker.snapshot import promote_shadow_if_ready

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job enqueue
# ---------------------------------------------------------------------------


def enqueue_job(
    session: object,
    user_id: str,
    config: MemoryConfig,
    *,
    purpose: str,
) -> EvolutionJob | None:
    """Write or merge an evolution job for *user_id*.

    Idempotent: same user + same evidence watermark produces at most
    one pending job.  Returns the job row, or ``None`` if the user
    has no active beliefs.

    ``purpose`` is REQUIRED (M3 batch-1 task 2): the job carries the
    purpose of the observe that triggered it, and the worker re-checks
    the authorization with it (operation ``extract``) before compute and
    again before publish.  There is no default.
    """
    from mirror_memory.core.repository import is_memory_enabled

    if not purpose or not str(purpose).strip():
        raise PurposeDenied(
            Decision(False, "missing_purpose", None, ""), "enqueue"
        )
    purpose = str(purpose).strip()
    if not is_memory_enabled(session, user_id):
        return None

    watermark = _current_watermark(session, user_id)
    idempotency_key = f"{user_id}:{watermark}:{config.worker.prompt_version}"

    # Check for existing pending/running job.
    existing = (
        session.query(EvolutionJob)
        .filter(
            EvolutionJob.user_id == user_id,
            EvolutionJob.status.in_(("pending", "running")),
        )
        .first()
    )
    if existing:
        # Queue bookkeeping via explicit bulk UPDATE (tenant-filtered by
        # the hook) rather than identity-keyed attribute mutation.
        session.query(EvolutionJob).filter(
            EvolutionJob.id == existing.id,
            EvolutionJob.status.in_(("pending", "running")),
        ).update(
            {
                EvolutionJob.evidence_watermark: json.dumps({"latest_update": watermark}),
                EvolutionJob.idempotency_key: idempotency_key,
            },
            synchronize_session=False,
        )
        session.commit()
        session.refresh(existing)
        return existing

    # Promote shadow if ready.
    promote_shadow_if_ready(session, user_id)

    # Reuse stale job with same idempotency key.
    stale = (
        session.query(EvolutionJob)
        .filter(
            EvolutionJob.user_id == user_id,
            EvolutionJob.idempotency_key == idempotency_key,
            EvolutionJob.status.in_(("completed", "failed")),
        )
        .first()
    )
    if stale:
        session.query(EvolutionJob).filter(
            EvolutionJob.id == stale.id,
            EvolutionJob.status.in_(("completed", "failed")),
        ).update(
            {
                EvolutionJob.status: "pending",
                EvolutionJob.attempt_count: 0,
                EvolutionJob.error_code: "",
                EvolutionJob.started_at: None,
                EvolutionJob.completed_at: None,
                EvolutionJob.evidence_watermark: json.dumps({"latest_update": watermark}),
            },
            synchronize_session=False,
        )
        session.commit()
        session.refresh(stale)
        return stale

    from mirror_memory.core.tenancy import queue_write_scope

    with queue_write_scope():
        job = EvolutionJob(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            purpose=purpose,
            evidence_watermark=json.dumps({"latest_update": watermark}),
            prompt_version=config.worker.prompt_version,
            idempotency_key=idempotency_key,
            status="pending",
        )
        session.add(job)
        session.commit()
    return job


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_pending(session_factory: object, config: MemoryConfig) -> None:
    """Process all pending evolution jobs.

    Tenancy (M1): the job queue is shared across tenants, so the
    crash-recovery sweep and the queue scan run under the explicit
    ``system_scope`` bypass (the only one, with a recorded reason).  Each
    job is then processed in a *fresh session created inside the job's own
    tenant scope* -- sessions are single-tenant by design, so the scan
    session is never reused for tenant work.

    Fencing (task 3, I1/I3): the crash-recovery sweep expires *leases*
    (``lease_expires_at``), not just old ``started_at``; a re-claim bumps
    ``lease_token`` so the previous holder's commit matches zero rows.

    Parameters
    ----------
    session_factory:
        Callable returning a new SQLAlchemy session.
    config:
        The ``MemoryConfig``.
    """
    from mirror_memory.core.tenancy import system_scope

    session = None
    try:
        with system_scope(reason="evolution worker: crash-recovery sweep and pending-job scan"):
            # The scan session is created INSIDE the scope: on PostgreSQL with
            # row-level security that binds it to the maintenance engine
            # (core.tenancy routes system_scope sessions), where the
            # cross-tenant scan can see every tenant's jobs.
            session = session_factory() if callable(session_factory) else session_factory
            # M4 task 3: keep the main DB aligned with its deletion log.
            # A mismatch means a restore happened (or a replay is running
            # elsewhere): the worker attempts the replay itself and
            # otherwise skips this tick -- it must not process jobs while
            # the database is behind its log.
            from mirror_memory.core import replay as deletion_replay
            from mirror_memory.core.deletion import DeletionLogStale, ReplayPending

            log = _deletion_log_for(config)
            if log is not None:
                try:
                    deletion_replay.reconcile(session_factory, log)
                except ReplayPending:
                    logger.warning("deletion replay in progress elsewhere; skipping this tick")
                    return
                except DeletionLogStale as exc:
                    logger.error("deletion log is older than the main DB: %s", exc)
                    return

            # Crash recovery: expire leases whose holder is gone.  The
            # lease_token is deliberately NOT reset -- it is the monotonic
            # fencing generation (I3).
            now = datetime.now(UTC)
            session.query(EvolutionJob).filter(
                EvolutionJob.status == "running",
                EvolutionJob.lease_expires_at.isnot(None),
                EvolutionJob.lease_expires_at < now,
            ).update({"status": "pending"}, synchronize_session=False)
            session.commit()

            jobs = (
                session.query(EvolutionJob)
                .filter(EvolutionJob.status == "pending")
                .order_by(EvolutionJob.created_at)
                .limit(JOB_BATCH_LIMIT)
                .all()
            )
        if not jobs:
            return

        for job in jobs:
            _claim_and_process(session_factory, job, config)
    finally:
        if session is not None:
            session.close()


class LeaseLost(RuntimeError):
    """The job's lease was taken over by another worker (I3 fencing).

    Raised when a completion/failure/renewal UPDATE matches zero rows
    because ``lease_token`` moved on.  The caller must roll back the same
    transaction: any memory writes the old holder made are discarded."""


def _claim_job(session: object, job_id: str, lease_seconds: int) -> int | None:
    """Atomically claim a pending job; returns the new lease token.

    The conditional UPDATE is the claim (only one writer can flip
    ``pending -> running`` and bump the token); SQLite serialises writers,
    and PostgreSQL gets real row-level contention in M2's ported tests.
    """
    now = datetime.now(UTC)
    updated = (
        session.query(EvolutionJob)
        .filter(
            EvolutionJob.id == job_id,
            EvolutionJob.status == "pending",
        )
        .update(
            {
                EvolutionJob.status: "running",
                EvolutionJob.started_at: now,
                EvolutionJob.attempt_count: EvolutionJob.attempt_count + 1,
                EvolutionJob.lease_token: EvolutionJob.lease_token + 1,
                EvolutionJob.lease_expires_at: now + timedelta(seconds=lease_seconds),
            },
            synchronize_session=False,
        )
    )
    session.commit()
    if updated == 0:
        return None
    token = session.query(EvolutionJob.lease_token).filter(
        EvolutionJob.id == job_id
    ).scalar()
    return int(token)


def _renew_lease(session: object, job_id: str, token: int, lease_seconds: int) -> bool:
    """Extend the lease; False when the lease was lost."""
    now = datetime.now(UTC)
    updated = (
        session.query(EvolutionJob)
        .filter(
            EvolutionJob.id == job_id,
            EvolutionJob.status == "running",
            EvolutionJob.lease_token == token,
        )
        .update(
            {EvolutionJob.lease_expires_at: now + timedelta(seconds=lease_seconds)},
            synchronize_session=False,
        )
    )
    session.commit()
    return updated == 1


def _set_job_state(session: object, job_id: str, token: int, changes: dict) -> None:
    """Queue bookkeeping with fencing: the UPDATE only lands while the
    caller still holds *token*; otherwise :class:`LeaseLost`."""
    updated = (
        session.query(EvolutionJob)
        .filter(
            EvolutionJob.id == job_id,
            EvolutionJob.lease_token == token,
        )
        .update(changes, synchronize_session=False)
    )
    session.commit()
    if updated == 0:
        raise LeaseLost(
            f"job {job_id}: lease token {token} no longer held; "
            "the job was re-claimed by another worker"
        )


def _claim_and_process(session_factory: object, job: EvolutionJob, config: MemoryConfig) -> None:
    """Claim and process one job inside a session bound to the job's tenant."""
    from mirror_memory.core.tenancy import tenant_scope

    with tenant_scope(job.tenant_id, job.app_id):
        session = session_factory() if callable(session_factory) else session_factory
        try:
            token = _claim_job(session, job.id, config.worker.job_lease_seconds)
            if token is None:
                return  # claimed by another worker
            job_row = session.get(EvolutionJob, job.id)
            if job_row is None:
                return
            _process_single_job(session, job_row, config, token)
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Single job processing (6-node pipeline)
# ---------------------------------------------------------------------------


def _process_single_job(
    session: object,
    job: EvolutionJob,
    config: MemoryConfig,
    lease_token: int | None = None,
) -> None:
    """Process a single evolution job through the 6-node pipeline.

    Only Node 6 writes.  Nodes 1-5 are pure Compute over a snapshot of the
    evidence read at the start; the revision guard and the memory-enabled
    check sit immediately before the write, so any state change that happened
    during the compute aborts the publish and leaves the database untouched.

    Transactions (M2 batch-2 patch): the snapshot is read in one
    transaction which is then ENDED -- the connection goes back to the pool
    -- before any model call; nothing is held "idle in transaction" while
    the LLM thinks (PostgreSQL's ``idle_in_transaction_session_timeout``
    would otherwise terminate the backend mid-compute).  The publish opens
    a fresh transaction and re-reads: the Publisher's consent, revision and
    authority checks were designed for a stale snapshot and are unchanged.

    Fencing (task 3): with *lease_token*, the completion/failure UPDATE only
    lands while this worker still holds the lease; on :class:`LeaseLost` the
    transaction (including the publish's memory writes) rolls back.
    """
    token = lease_token
    # Plain copies of the job fields the phases below need: after the
    # snapshot transaction ends the ORM row is not touched again until the
    # publish transaction (see _end_snapshot_transaction).
    job_id, user_id = job.id, job.user_id
    attempt_count, watermark = job.attempt_count, job.evidence_watermark
    try:
        from mirror_memory.core.authorization import authorize_or_raise
        from mirror_memory.core.repository import get_state_revision, is_memory_enabled

        # Renew the lease before the compute-heavy nodes: the pre-compute
        # renewal must find the token still held; a lost lease aborts before
        # any compute.  A second renewal happens at the publish transaction
        # (below), so the publish itself never runs on an expired lease
        # either -- the lease only has to be held at two instants, not
        # across the whole compute.
        if token is not None and not _renew_lease(
            session, job_id, token, config.worker.job_lease_seconds
        ):
            raise LeaseLost(f"job {job_id}: lease lost before compute")

        # -- snapshot transaction ------------------------------------------
        # Consent gate first (its reason is the long-standing one): a
        # disabled user's beliefs must not even be read into a compute
        # payload, let alone published.
        if not is_memory_enabled(session, user_id):
            _block_job(session, job, "memory_disabled", token)
            return

        # Authorization gate (M3 batch-1 task 2): the job carries the
        # purpose of the observe that enqueued it; the worker re-checks it
        # for the extract operation.  A pre-M3 job (empty purpose) is
        # cancelled, never run ungated; a refusal cancels the job with the
        # decision's reason.
        if not (job.purpose or "").strip():
            from mirror_memory.core import metrics

            metrics.inc(metrics.MISSING_PURPOSE)
            _block_job(session, job, "missing_purpose", token)
            return
        try:
            grant_version = authorize_or_raise(
                session, user_id, job.purpose, "extract"
            ).grant_version
        except PurposeDenied as exc:
            _block_job(session, job, exc.decision.reason, token)
            return
        # The rows the publish writes (ledger events) are collected under
        # the job's purpose -- the purpose of the observe that enqueued it.
        provenance.bind(session, job.purpose)
        # Cross-purpose visibility (M3 batch 3): the job may only READ
        # memory collected under purposes that are still live (the A2
        # rule) -- a job under Q must not synthesize from P-only evidence
        # after P was revoked.  One rule per job, computed inside this
        # transaction.
        visibility_rule = provenance.visibility(session, user_id)

        # We intentionally ignore job.claimed_revision here (a snapshot from
        # enqueue time).  The revision may have been bumped by state mutations
        # between enqueue and compute, so we re-read the current value to
        # compare against the post-compute revision at Node 6.
        expected_revision = get_state_revision(session, user_id)
        # M4 task 4: the deletion generation at claim time -- the publish
        # re-checks it (with the revision) so a deletion -- or a backup
        # restore + replay -- that happened during the compute cancels the
        # job instead of writing deleted content back.
        from mirror_memory.core.repository import current_deletion_generation

        expected_generation = current_deletion_generation(session, user_id)

        # Node 1: load_evidence (plain data, no ORM instances retained),
        # filtered to memory the job's purpose may see
        evidence, visible_beliefs = _load_evidence(session, user_id, visibility=visibility_rule)

        # Node 5's read half (K3): the beliefs/summaries the synthesis prompt
        # needs, taken in the same snapshot.  Fail-open like the node itself.
        from mirror_memory.extraction.synthesis import Synthesizer

        synthesizer = Synthesizer(config)
        try:
            k3_prepared = synthesizer.prepare(session, user_id, config, visibility=visibility_rule)
        except Exception:
            logger.warning("evolution: K3 synthesis skipped in worker (job=%s)", truncate_id(job_id))
            k3_prepared = None

        _end_snapshot_transaction(session)
        # -- no connection held from here until the publish -----------------

        # Node 2: formulate (LLM)
        candidate = _formulate(session, user_id, evidence, config)

        # Node 3: validate
        validated = _validate(candidate, evidence, config)

        # Node 4: compile_policy
        policy = _compile_policy(validated)

        # Node 5: synthesize (K3 -- pure Compute, fail-open, writes nothing)
        understanding = None
        try:
            understanding = synthesizer.produce(k3_prepared, config)
        except Exception:
            logger.warning("evolution: K3 synthesis skipped in worker (job=%s)", truncate_id(job_id))

        # The published content is the validated candidate, with K3's
        # understanding folded in when it produced one.
        content = dict(validated)
        if understanding is not None:
            content["understanding"] = understanding

        # The snapshot is MERGED free text: its provenance is the UNION of
        # everything the synthesis read -- the visible beliefs' provenance,
        # the previous snapshots' unions, and the job's own purpose (M3
        # batch-3 patch task 2).  ONE batched probe for the whole set:
        # per-belief queries would multiply the publish transaction's
        # statements (and its lock-holding time) by the belief count.
        input_purposes: set[str] = {job.purpose}
        if visible_beliefs:
            link_purposes, event_purposes = provenance.provenance_purposes(
                session, user_id, [b.id for b in visible_beliefs],
            )
            for belief in visible_beliefs:
                input_purposes |= link_purposes.get(belief.id, set()) | event_purposes.get(belief.id, set())
        input_purposes |= _previous_snapshot_purposes(session, user_id)

        # -- publish transaction -------------------------------------------
        # Nothing read for the snapshot is trusted here: expire the identity
        # map so the Publisher's consent / revision / authority re-checks
        # read the database as it is now, in a fresh transaction.
        session.expire_all()

        # Renew once more now that the compute is done (M3 batch-1 task 0):
        # a compute longer than the lease must not publish on an expired
        # one.  The renewal is fenced on the token -- if another worker
        # re-claimed during the compute, this fails and nothing of ours is
        # written.  _renew_lease commits, so the publish's own statements
        # open the fresh transaction after it.
        if token is not None and not _renew_lease(
            session, job_id, token, config.worker.job_lease_seconds
        ):
            raise LeaseLost(f"job {job_id}: lease lost before publish")

        # Consent gate again (pre-existing semantics: its reason wins)
        if not is_memory_enabled(session, user_id):
            _block_job(session, job, "memory_disabled", token)
            return

        # Authorization re-check (M3 batch-1 task 4, patch task 3): only
        # "is extract still allowed right now" matters.  The version is
        # recorded, not compared -- a renewal or an added operation during
        # the compute publishes normally; a revocation, an expiry, or the
        # operation being removed cancels.  authorize(lock=True) takes FOR
        # SHARE on the grant row (PostgreSQL): a concurrent revocation's
        # UPDATE blocks until this transaction ends, so a committed
        # revocation can never slip between this check and the publish's
        # commit.  SQLite serializes writers by itself.  The refusal (or a
        # lost authorization) cancels the job with the decision's reason
        # before the Publisher writes, so nothing lands.  Both versions --
        # the snapshot's and the publish-time's -- go to the authorization
        # audit trail and the log.
        from mirror_memory.core.authorization import authorize_or_raise, record_purpose_invocation

        try:
            decision = authorize_or_raise(
                session, user_id, job.purpose, "extract", lock=True
            )
        except PurposeDenied as exc:
            _block_job(session, job, exc.decision.reason, token)
            return
        publish_grant_version = decision.grant_version
        record_purpose_invocation(
            session, user_id, job.purpose,
            operation="extract", decision="published",
            actor="worker",
            reason=f"snapshot_version={grant_version} publish_version={publish_grant_version}",
        )
        # M4 task 4: the deletion generation, re-checked WITH the revision
        # at the same point and under the same lock -- a deletion that
        # committed during the compute (or a restore + replay that moved
        # the generation) must cancel the job, never write through it.
        # This also fixes the revision ABA: a restore rewinds the revision
        # to a stale-but-matching value, but the replay raises the
        # generation past the claim-time one.
        from mirror_memory.core.repository import current_deletion_generation

        current_generation = current_deletion_generation(session, user_id)
        if current_generation != expected_generation:
            session.rollback()
            _block_job(session, job, "deletion_generation_changed", token)
            return

        # Node 6: publish.  The Publisher is the only component allowed to
        # write derived state, and it re-checks consent, revision, and
        # authority itself -- so a change that landed during the compute is
        # caught here rather than trusted.
        from mirror_memory.core.proposal import TRANSITION_SYNTHESIZE, StateTransitionProposal
        from mirror_memory.core.publisher import Publisher

        is_shadow = evidence.get("distinct_sessions", 1) < 2
        proposal = StateTransitionProposal(
            transition=TRANSITION_SYNTHESIZE,
            user_id=user_id,
            session_id=None,
            expected_revision=expected_revision,
            payload={
                "content": content,
                "policy": policy,
                "watermark": watermark,
                "shadow": is_shadow,
                "source_purposes": sorted(input_purposes),
            },
        )
        decision = Publisher(session, config).publish(proposal)
        if not decision.committed:
            logger.info(
                "evolution: publish refused (job=%s reason=%s)",
                truncate_id(job_id), decision.reason,
            )
            _block_job(session, job, decision.reason, token)
            return

        _set_job_state(session, job_id, token, {
            "status": "completed",
            "completed_at": datetime.now(UTC),
            "error_code": "",
        })
        logger.info("evolution: completed job=%s", truncate_id(job_id))

    except LeaseLost:
        # Another worker re-claimed the job after our lease expired: every
        # memory write in THIS transaction (the publish above included)
        # must not survive.  Roll back; the new owner manages the row.
        session.rollback()
        logger.warning(
            "evolution: lease lost (job=%s); rolled back this worker's writes",
            truncate_id(job_id),
        )
    except Exception as exc:
        # A failure surfaced OUTSIDE the publish's savepoint (e.g. the
        # authorization re-check) leaves the transaction aborted on
        # PostgreSQL: roll it back before any bookkeeping.  The fenced
        # completion UPDATE still guards the bookkeeping itself.
        session.rollback()
        busy = _is_sqlite_busy(exc)
        changes = {
            # A SQLite lock collision (database is locked / BUSY) is
            # TRANSIENT: it must not burn an attempt -- the job goes back to
            # pending with its attempt count restored, deferred to the end
            # of the queue (M3 patch task 0)
            "status": "pending" if busy else (
                "failed" if attempt_count >= config.worker.max_attempts else "pending"
            ),
            "error_code": type(exc).__name__ if not busy else "database_is_locked",
        }
        if busy:
            changes["attempt_count"] = max(0, attempt_count - 1)
            changes["created_at"] = datetime.now(UTC)  # defer: end of the queue
            # the contending writer may still hold the lock while this
            # bookkeeping runs: retry the fenced bookkeeping under a
            # bounded deadline (the sweep's next pass would only hit the
            # same lock).  PG failures never take this path (the text
            # check is SQLite-specific).
            deadline = time.monotonic() + _BUSY_BOOKKEEPING_TIMEOUT
            while True:
                try:
                    _set_job_state(session, job_id, token, changes)
                    logger.info(
                        "evolution: job=%s deferred (sqlite busy); attempt "
                        "count restored", truncate_id(job_id),
                    )
                    return
                except OperationalError as busy_exc:
                    if not _is_sqlite_busy(busy_exc) or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.4)
            # unreachable: the loop returns or raises
        try:
            _set_job_state(session, job_id, token, changes)
        except LeaseLost:
            session.rollback()
            logger.warning(
                "evolution: lease lost while failing (job=%s); writes rolled back",
                truncate_id(job_id),
            )
            return
        logger.warning("evolution: failed job=%s error=%s", truncate_id(job_id), type(exc).__name__)
    finally:
        # a session reused across jobs must not carry this job's purpose
        # into the next one
        provenance.unbind(session)


def _previous_snapshot_purposes(session: object, user_id: str) -> set[str]:
    """The provenance unions of the user's live snapshots (the synthesis
    reads the latest as context -- its inputs are inputs here too)."""
    from mirror_memory.core.models import Snapshot

    purposes: set[str] = set()
    for snap in (
        session.query(Snapshot)
        .filter(Snapshot.user_id == user_id, Snapshot.status.in_(("active", "shadow")))
        .all()
    ):
        try:
            listed = [p for p in json.loads(snap.source_purposes_json or "[]") if p]
        except ValueError:
            listed = []
        if listed:
            purposes |= set(listed)
        elif snap.source_purpose:
            purposes.add(snap.source_purpose)
    return purposes


#: How long the BUSY bookkeeping keeps retrying the fenced pending-update
#: before giving up and surfacing the lock error (the sweep retries anyway).
_BUSY_BOOKKEEPING_TIMEOUT = 6.0


def _deletion_log_for(config: MemoryConfig):
    """The deletion log for the worker's reconciliation (M4 task 3), built
    from the same configuration rules as the engine's."""
    cached = getattr(config, "_deletion_log_instance", None)
    if cached is not None or hasattr(config, "_deletion_log_instance"):
        return cached
    from mirror_memory.core.deletion import FileDeletionLog, PgDeletionLog

    settings = config.deletion_log
    instance = None
    if settings.backend == "postgres" and settings.url:
        instance = PgDeletionLog(settings.url)
    elif settings.backend == "file":
        directory = settings.directory or _default_deletion_directory(config)
        instance = FileDeletionLog(directory)
    config._deletion_log_instance = instance
    return instance


def _default_deletion_directory(config: MemoryConfig) -> str:
    from pathlib import Path

    url = getattr(config, "database_url", "") or ""
    if url.startswith("sqlite:///"):
        db_path = url[len("sqlite:///"):]
        return str((Path(db_path).parent / "deletions").resolve())
    import hashlib
    import tempfile

    digest = hashlib.sha1(url.encode()).hexdigest()[:12]
    return str(Path(tempfile.gettempdir()) / f"mm-deletions-{digest}")


def _is_sqlite_busy(exc: BaseException) -> bool:
    """True for a SQLite lock collision: the error text carries
    'database is locked' / 'database table is locked' (pysqlite raises
    OperationalError with those messages; there is no distinct SQLSTATE)."""
    text = str(exc)
    return "database is locked" in text or "database table is locked" in text


def _end_snapshot_transaction(session: object) -> None:
    """Close the snapshot's read transaction without expiring loaded state.

    ``commit()`` rather than ``rollback()``: a rollback always expires every
    instance in the session, and an expired attribute touched during the
    compute would lazy-load -- i.e. quietly open a new transaction behind
    the model call, which is exactly what this is avoiding.  Expiry on
    commit is suspended for this one commit only; the publish phase expires
    everything itself before it reads.
    """
    prior = session.expire_on_commit
    session.expire_on_commit = False
    try:
        session.commit()
    finally:
        session.expire_on_commit = prior


def _set_job_state(session: object, job_id: str, token: int | None, changes: dict) -> None:
    """Queue bookkeeping: job-state writes go through an explicit bulk
    UPDATE (tenant-filtered by the hook) rather than attribute mutation.
    With a lease token the UPDATE is fenced (I3); without one (legacy
    direct calls in tests) it behaves as an unfenced UPDATE."""
    if token is None:
        session.query(EvolutionJob).filter(
            EvolutionJob.id == job_id,
        ).update(changes, synchronize_session=False)
        session.commit()
        return
    updated = (
        session.query(EvolutionJob)
        .filter(
            EvolutionJob.id == job_id,
            EvolutionJob.lease_token == token,
        )
        .update(changes, synchronize_session=False)
    )
    if updated == 0:
        # Fence fired: commit NOTHING from this holder's transaction (the
        # publish and any other writes roll back with it) and surface the
        # loss.  The check precedes the commit on purpose -- committing
        # first would persist exactly the writes the fence exists to
        # discard.
        session.rollback()
        raise LeaseLost(
            f"job {job_id}: lease token {token} no longer held; "
            "the job was re-claimed by another worker"
        )
    session.commit()


def _block_job(session: object, job: EvolutionJob, reason: str, token: int | None = None) -> None:
    """Terminal-state a job whose publish was refused.

    The job is marked ``cancelled`` rather than ``completed``: nothing was
    written, and the next ``enqueue_job`` for the user must be free to run.
    """
    try:
        _set_job_state(session, job.id, token, {
            "status": "cancelled",
            "completed_at": datetime.now(UTC),
            "error_code": reason,
        })
    except LeaseLost:
        session.rollback()
        logger.warning(
            "evolution: lease lost while blocking (job=%s); new owner decides",
            truncate_id(job.id),
        )


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _current_watermark(session: object, user_id: str) -> str:
    """Get the latest belief update timestamp as watermark."""
    latest = (
        session.query(Belief.updated_at)
        .filter(Belief.user_id == user_id, Belief.status == "active")
        .order_by(Belief.updated_at.desc())
        .limit(1)
        .scalar()
    )
    return latest.isoformat() if latest else "none"


def _load_evidence(
    session: object, user_id: str, *, visibility: "VisibilityRule | None" = None,
) -> tuple[dict, list]:
    """Returns ``(evidence payload, the visible Belief rows)`` -- the rows
    are what the synthesis actually read, and the snapshot's provenance
    union is computed from them (M3 batch-3 patch)."""
    """Node 1: Read active beliefs, background, and session count.

    With a visibility rule only beliefs whose provenance passes it feed the
    synthesis -- and ``distinct_sessions`` counts the VISIBLE page only
    (the shadow gate asks whether the job's purpose has seen the user in
    more than one session, which is a question about what this job may
    use)."""
    beliefs = (
        session.query(Belief)
        .filter(Belief.user_id == user_id, Belief.status == "active")
        .order_by(Belief.last_evidence_at.desc())
        .limit(LOAD_EVIDENCE_LIMIT)
        .all()
    )
    if visibility is not None:
        beliefs = provenance.beliefs_visible(session, user_id, visibility, beliefs)
    belief_data = []
    for b in beliefs:
        belief_data.append({
            "dimension": b.dimension,
            "key": b.key,
            "claim_text": b.claim_text,
            "confidence": b.confidence,
            "layer": b.layer,
            "value": b.value_json,
        })

    distinct_sessions = len({
        b.origin_session_id for b in beliefs if b.origin_session_id
    })

    return {
        "beliefs": belief_data,
        "user_id": user_id,
        "distinct_sessions": distinct_sessions,
    }, beliefs


def _formulate(
    session: object,
    user_id: str,
    evidence: dict,
    config: MemoryConfig,
) -> dict:
    """Node 2: One LLM call for candidate understanding."""
    llm = config.llm_client
    if llm is None:
        return {"patterns": [], "support_policy": {}}

    system_prompt = config.prompts.formulate
    if not system_prompt:
        return {"patterns": [], "support_policy": {}}

    beliefs_text = "\n".join(
        f"- [{b['dimension']}] {b['key']} (confidence={b['confidence']:.2f}): {b['claim_text']}"
        for b in evidence["beliefs"]
        if b["confidence"] >= FORMULATE_MIN_CONFIDENCE
    )

    user_content = f"[Beliefs]\n{beliefs_text or '(none yet)'}"
    try:
        raw = llm.generate(
            system_prompt=system_prompt,
            payload_text=user_content,
        )
    except Exception:
        logger.warning("evolution: formulate LLM call failed")
        return {"patterns": [], "support_policy": {}}

    return _parse_formulate_json(raw)


def _parse_formulate_json(raw: str) -> dict:
    """Parse the formulate LLM output."""
    from mirror_memory.core.utils import parse_llm_json

    result = parse_llm_json(raw)
    if isinstance(result, dict):
        return result
    return {"patterns": [], "support_policy": {}}


def _validate(
    candidate: dict,
    evidence: dict | None,
    config: MemoryConfig,
) -> dict:
    """Node 3: Deterministic validation.

    Checks:
    - Forbidden labels (from config.worker.forbidden_labels)
    - Third-party attribution (from config.worker.third_party_markers)
    - Evidence existence
    - Contradiction forcing needs_verification
    """
    patterns = candidate.get("patterns", [])
    validated = []

    forbidden = {label.lower() for label in config.worker.forbidden_labels}
    third_party = {marker.lower() for marker in config.worker.third_party_markers}

    # Build evidence index.
    belief_keys: set[str] = set()
    contradicted_keys: set[str] = set()
    distinct_sessions = 1
    if evidence and "beliefs" in evidence:
        for b in evidence["beliefs"]:
            belief_keys.add(b.get("key", ""))
            # Counter-evidence: needs_clarification marks a contradicted belief.
            try:
                val = json.loads(b.get("value", "{}") or "{}")
                if val.get("clarification_status") == "needs_clarification":
                    contradicted_keys.add(b.get("key", ""))
            except (TypeError, ValueError):
                pass
        distinct_sessions = evidence.get("distinct_sessions", 1)

    for p in patterns:
        desc = p.get("description", "")
        desc_lower = desc.lower()

        # Forbidden labels
        if any(label in desc_lower for label in forbidden):
            continue
        # Empty description
        if not desc.strip():
            continue
        # Third-party attribution
        if any(marker in desc_lower for marker in third_party):
            continue
        # Evidence existence check
        p_keys = set(p.get("evidence_keys") or [])
        if p_keys and not p_keys.issubset(belief_keys):
            continue
        # Counter-evidence: any contradicted belief forces needs_verification
        if contradicted_keys:
            p["needs_verification"] = True
        # Single-session evidence forces needs_verification
        if distinct_sessions < 2:
            p["needs_verification"] = True
        validated.append(p)

    candidate["patterns"] = validated[:3]
    return candidate


_DEFAULT_POLICY = {
    "response_length": "normal",
    "pacing": "validate_before_suggestions",
    "question_budget": 2,
    "avoid_topics": [],
    "preferred_knowledge_paths": [],
}


def _compile_policy(candidate: dict) -> dict:
    """Node 4: Compile support policy from candidate."""
    raw_policy = candidate.get("support_policy", {})
    policy = dict(_DEFAULT_POLICY)
    if isinstance(raw_policy, dict):
        if raw_policy.get("response_length") in ("brief", "normal"):
            policy["response_length"] = raw_policy["response_length"]
        if raw_policy.get("pacing"):
            policy["pacing"] = raw_policy["pacing"]
    return policy


# ---------------------------------------------------------------------------
# Async worker entry point
# ---------------------------------------------------------------------------


async def evolution_worker(
    stop: asyncio.Event,
    config: MemoryConfig,
) -> None:
    """Persistent background worker that polls for pending jobs.

    Parameters
    ----------
    stop:
        An ``asyncio.Event`` that, when set, signals the worker to
        stop after the current poll cycle.
    config:
        The ``MemoryConfig`` providing session factory and worker
        parameters.
    """
    session_factory = config.session_factory
    poll_interval = config.worker.poll_interval_seconds
    # audit-retention maintenance runs on its own (much longer) cadence
    maintenance_state: dict[str, float] = {}
    from mirror_memory.worker.maintenance import next_due, purge_expired_audit

    while not stop.is_set():
        try:
            await asyncio.to_thread(process_pending, session_factory, config)
        except Exception:
            logger.warning("evolution: worker will retry on next tick", exc_info=True)
        if next_due(maintenance_state, config):
            try:
                await asyncio.to_thread(purge_expired_audit, session_factory, config)
            except Exception:
                logger.warning("audit maintenance: will retry next interval", exc_info=True)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_interval)
