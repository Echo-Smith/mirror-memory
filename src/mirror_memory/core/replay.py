"""Replay of the deletion log onto a restored (or lagging) main database
(M4 task 3).

The watermark row in the MAIN database ``(db_id, applied_seq)`` rolls
back together with every backup of it.  Comparing it to the deletion
log's tail detects a restore:

- applied == tail  -> normal start (nothing to do);
- applied <  tail  -> the main DB is BEHIND its log (it was restored from
  a backup taken before some deletions): replay the entries in seq
  order.  Each entry is applied in ONE transaction that deletes only
  fingerprint-matching rows, cancels the user's restored pending jobs,
  aligns the user's generation to the entry's (the LOG is the authority,
  so a restore can never roll a generation back) and advances the
  watermark -- an interruption resumes from the watermark, and the
  fingerprint matching makes every step idempotent;
- applied >  tail  -> the log is OLDER than the main database: refuse to
  start with operator guidance (the log's recovery point must never be
  older than the main DB).

Multi-instance: one replayer per database life.  PostgreSQL takes a
session-level advisory lock (its own dedicated connection, released
after the replay); the embedded backend takes a non-blocking flock on a
lock file next to the main DB.  Instances that find the lock held raise
:class:`ReplayPending` -- they keep refusing entries until the replay is
done and they can complete their startup check.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.engine import Engine

from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import (
    DeletionEntry,
    DeletionLog,
    DeletionLogStale,
    ReplayPending,
)

logger = logging.getLogger(__name__)


def _replay_lock(engine: Engine, db_id: str):
    """A non-blocking, process-level replay lock for one database life.
    PostgreSQL: a session-level advisory lock on a dedicated connection.
    SQLite: an flock next to the main DB file.  Returns a context manager
    that yields True when held, False when another instance holds it."""
    import os

    if engine.dialect.name == "postgresql":
        conn = engine.connect()
        got = conn.execute(text("SELECT pg_try_advisory_lock(hashtext(:k))"),
                           {"k": f"mm-replay:{db_id}"}).scalar()
        if not got:
            conn.close()
            return _FalseLock()
        holder = {"conn": conn}

        @contextlib.contextmanager
        def _pg():
            try:
                yield True
            finally:
                with contextlib.suppress(Exception):
                    holder["conn"].execute(
                        text("SELECT pg_advisory_unlock(hashtext(:k))"),
                        {"k": f"mm-replay:{db_id}"})
                holder["conn"].close()

        return _pg()

    url = str(engine.url)
    if url.startswith("sqlite:///"):
        db_path = url[len("sqlite:///"):]
    else:
        db_path = f"/tmp/mm-{abs(hash(url))}"
    lock_path = Path(db_path + ".replay.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return _FalseLock()

    @contextlib.contextmanager
    def _file():
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()

    return _file()


class _FalseLock:
    def __enter__(self):
        return False

    def __exit__(self, *exc):
        return False


def _fp_set(entry: DeletionEntry) -> set[str]:
    return {fp.get("_fp") or "" for fp in entry.fingerprints}


def _matching(session, model, criteria, wanted: set[str]) -> list:
    """The candidate rows whose content-free fingerprint was recorded in
    the entry -- ONLY those are deleted (a recreated row with a reused id
    has a different created_at and does not match)."""
    from mirror_memory.core.deletion import fingerprint

    rows = session.execute(select(model).where(*criteria)).scalars().all()
    matched = []
    for row in rows:
        if fingerprint(model.__tablename__, row.id, getattr(row, "created_at", None)) in wanted:
            matched.append(row)
    return matched


def _delete_matching(session, model, criteria, wanted: set[str]) -> int:
    matched = _matching(session, model, criteria, wanted)
    for row in matched:
        session.delete(row)
    if matched:
        session.flush()
    return len(matched)


def apply_entry(session, entry: DeletionEntry) -> dict[str, int]:
    """Re-apply ONE deletion entry inside the caller's transaction.

    Only fingerprint-matching rows are deleted, the user's restored
    pending jobs are cancelled, the user's generation is aligned to the
    entry's (the log is the authority), and the caller advances the
    watermark in the same transaction."""
    from mirror_memory.core.models import (
        Belief, BeliefEvent, BeliefEvidenceLink, BeliefIdentity, BeliefVersion,
        DeletionGeneration, Evidence, EvidenceDigest, EvolutionJob,
        ExtractionStats, InterventionEvent, MemoryTransition, ProposalLog,
        SessionSummary, Snapshot,
    )
    from mirror_memory.core.deletion import fingerprint
    from mirror_memory.core.tenancy import deletion_write_scope

    wanted = _fp_set(entry)
    tenant_id, app_id = entry.tenant_id, entry.app_id
    tenant_criteria = (Belief.tenant_id == tenant_id, Belief.app_id == app_id)
    counts: dict[str, int] = {}

    def candidates(model, *extra):
        criteria = [model.tenant_id == tenant_id, model.app_id == app_id,
                    model.user_id == entry.user_id, *extra]
        return _delete_matching(session, model, tuple(criteria), wanted)

    with deletion_write_scope():
        if entry.scope == "session":
            counts["summaries"] = candidates(SessionSummary,
                                             SessionSummary.session_id == entry.scope_key)
        else:
            if entry.scope == "belief" and entry.scope_key:
                counts["beliefs"] = _delete_matching(
                    session, Belief,
                    tuple(tenant_criteria) + (Belief.id == int(entry.scope_key),),
                    wanted)
            else:
                counts["beliefs"] = candidates(Belief)
            counts["events"] = candidates(BeliefEvent)
            counts["digests"] = candidates(EvidenceDigest)
            counts["transitions"] = candidates(MemoryTransition)
            counts["versions"] = candidates(BeliefVersion)
            counts["identities"] = candidates(BeliefIdentity)
            counts["proposals"] = candidates(ProposalLog)
            counts["snapshots"] = candidates(Snapshot)
            if entry.scope == "user":
                counts["evidence"] = candidates(Evidence)
                counts["stats"] = candidates(ExtractionStats)
                counts["interventions"] = candidates(InterventionEvent)
                counts["links"] = _delete_matching(
                    session, BeliefEvidenceLink,
                    (BeliefEvidenceLink.tenant_id == tenant_id,
                     BeliefEvidenceLink.app_id == app_id,
                     BeliefEvidenceLink.belief_id.in_(
                         select(Belief.id).where(
                             Belief.tenant_id == tenant_id,
                             Belief.app_id == app_id,
                             Belief.user_id == entry.user_id))),
                    wanted)
            # links for belief scope hang off the (deleted) belief's
            # evidence: match them among the user's remaining links
            if entry.scope == "belief":
                counts["links"] = _delete_matching(
                    session, BeliefEvidenceLink,
                    (BeliefEvidenceLink.tenant_id == tenant_id,
                     BeliefEvidenceLink.app_id == app_id,
                     BeliefEvidenceLink.evidence_id.in_(
                         select(Evidence.id).where(
                             Evidence.tenant_id == tenant_id,
                             Evidence.app_id == app_id,
                             Evidence.user_id == entry.user_id))),
                    wanted)
        # the restored pending jobs read memory the deletion removed
        cancelled = (
            session.query(EvolutionJob)
            .filter(
                EvolutionJob.tenant_id == tenant_id,
                EvolutionJob.app_id == app_id,
                EvolutionJob.user_id == entry.user_id,
                EvolutionJob.status.in_(("pending", "running")),
            )
            .update({"status": "cancelled", "error_code": "deletion_replayed"},
                    synchronize_session=False)
        )
        counts["jobs_cancelled"] = int(cancelled)
        # the LOG is the authority: align (never roll back) the generation
        exists = session.get(DeletionGeneration,
                             (tenant_id, app_id, entry.user_id, entry.generation))
        if exists is None:
            session.add(DeletionGeneration(
                user_id=entry.user_id, generation=entry.generation,
                scope=entry.scope))
    session.flush()
    return counts


def reconcile(session_factory, log: DeletionLog | None, *,
              tenant: tuple[str, str] | None = None) -> str:
    """Bring the main database back in line with its deletion log.

    Returns ``"ok"`` when the watermark equals the log tail, or
    ``"replayed"`` after applying the missed entries.  Raises
    :class:`ReplayPending` while another instance holds the replay lock
    and :class:`DeletionLogStale` when the log is older than the main DB.
    With no log configured there is nothing to compare (deletions are
    refused anyway); the check is skipped.  *tenant* scopes the whole
    reconciliation (the replayed deletions are tenant writes)."""
    if log is None:
        return "ok"
    if tenant is not None:
        from mirror_memory.core.tenancy import tenant_scope

        with tenant_scope(*tenant):
            return _reconcile_scoped(session_factory, log)
    return _reconcile_scoped(session_factory, log)


def _reconcile_scoped(session_factory, log: DeletionLog) -> str:
    session = session_factory() if callable(session_factory) else session_factory
    bind = session.get_bind()
    try:
        db_id, applied = sm.ensure_watermark(bind)
        log.verify_chain(db_id)
        tail_seq, _tail_hash = log.head(db_id)
        if applied > tail_seq:
            raise DeletionLogStale(
                f"the deletion log is OLDER than the main database "
                f"(watermark {applied} > log tail {tail_seq} for db {db_id[:8]}). "
                "Restore a newer log (or a backup of it) before starting; the "
                "log's recovery point must never be older than the main DB."
            )
        if applied == tail_seq:
            return "ok"
        lock = _replay_lock(bind, db_id)
        with lock as held:
            if not held:
                raise ReplayPending(
                    "another instance is replaying the deletion log; "
                    "entries are refused until it finishes"
                )
            # re-read under the lock: the winner may have finished already
            _db_id2, applied = sm.ensure_watermark(bind)
            if applied >= tail_seq:
                return "ok"
            watermark = sm._watermark_table(__import__("sqlalchemy").MetaData())
            for entry in log.read_from(db_id, applied):
                apply_entry(session, entry)
                # the watermark advances in the SAME transaction as the
                # applied entry: an interruption resumes from here
                session.execute(
                    watermark.update().where(watermark.c.id == 1)
                    .values(applied_seq=entry.seq))
                session.commit()
        logger.warning(
            "deletion replay: applied entries up to seq %d (watermark was %d)",
            tail_seq, applied)
        return "replayed"
    finally:
        session.close()
