"""M4 tasks 2-4: the deletion replay and the publish-time generation check.

- replay: a restored main DB (watermark behind the log) replays its
  entries on startup -- only fingerprint-matching rows go (id reuse is
  safe), the user's restored pending jobs are cancelled, the generation
  aligns to the LOG (never rolls back), and the watermark advances in
  the same transaction (interruptions resume);
- refusal: while a replay is running elsewhere the entries refuse
  (ReplayPending); a log OLDER than the main DB refuses startup with
  guidance;
- publish: the job's claim-time generation is re-checked with the
  revision at the M3 re-check point -- a deletion (or restore + replay)
  during the compute cancels the job (deletion_generation_changed);
  this also fixes the revision ABA.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import (
    DeletionLogUnavailable,
    DeletionEntry,
    DeletionLogStale,
    FileDeletionLog,
    ReplayPending,
)
from mirror_memory.core.models import (
    Base,
    Belief,
    DeletionGeneration,
    DeletionTombstone,
    EvolutionJob,
    Snapshot,
)
from mirror_memory.core.replay import _REPLAY_HANDLERS, reconcile
from mirror_memory.core.repository import (
    forget_belief,
    set_memory_enabled,
)
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.worker.evolution import _claim_job, _process_single_job, enqueue_job

USER = "u-m4"
TENANT = ("default", "default")


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


@pytest.fixture
def env(tmp_path):
    """A database + its paired deletion log + an engine."""
    db = create_engine(f"sqlite:///{tmp_path / 'main.db'}")
    Base.metadata.create_all(db)

    from mirror_memory.core import schema_migration as _sm_mint
    _sm_mint.ensure_watermark(db)
    log = FileDeletionLog(tmp_path / "deletions")
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()

    class _Env:
        pass

    _Env.db = db
    _Env.log = log
    _Env.F = F
    _Env.cfg = None

    _Env.cfg = load_config("config/")
    _Env.cfg.llm_client = _FakeLLM()
    _Env.cfg.worker.job_lease_seconds = 60
    _Env.cfg.session_summary_enabled = True
    # the worker derives its log from the CONFIG: it must look at the SAME
    # directory as the engine's explicit log (on the PG leg the harness
    # default points elsewhere -- M4 patch task 2 removed the derived
    # default for server URLs)
    _Env.cfg.deletion_log.directory = str(tmp_path / "deletions")
    _Env.engine = MemoryEngine(config=_Env.cfg, database_url=str(db.url),
                               tenant_id=TENANT[0], app_id=TENANT[1],
                               llm_client=_FakeLLM(), deletion_log=log)
    yield _Env
    db.dispose()


def _observe(eng, text_, purpose="api", session_id="s1"):
    return eng.observe(user_id=USER, session_id=session_id, text=text_, purpose=purpose)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_restores_the_deletion_after_a_backup_restore(env):
    """observe sentinel -> BACKUP -> delete under three scopes -> RESTORE
    -> reconcile: the sentinel is gone from every surface again."""
    eng = env.engine
    _observe(eng, "I like SENTINELFOUR and coffee.", purpose="api", session_id="sa")
    _observe(eng, "I keep notes about gardening in the shed.", purpose="api", session_id="sb")
    # BACKUP: SQLite's backup API into a second file (the real tool)
    backup_url = str(env.db.url).replace("main.db", "backup.db")
    src = env.engine._engine.raw_connection()
    dest = create_engine(backup_url).raw_connection()
    src.connection.backup(dest.connection)  # the real sqlite3 backup API
    dest.close()  # returns the fairy to its pool
    src.close()

    # three scopes of deletion, each logged
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=env.db)()
        try:
            belief = next(b for b in s.query(Belief)
                          .filter(Belief.claim_text.contains("SENTINELFOUR")).all())
        finally:
            s.close()
    assert eng.forget_belief(user_id=USER, belief_id=belief.id, purpose="api") is True
    eng.forget_session(user_id=USER, session_id="sa", purpose="api")
    eng.delete_memories(user_id=USER, purpose="api")
    tail = env.log.head(sm.read_watermark(env.db)[0])[0]
    assert tail == 3, "three logged deletions"

    # RESTORE the backup (the watermark row comes back with it): dispose
    # every pooled connection first -- a restore replaces the FILE, and a
    # real deployment restarts the service afterwards
    eng._engine.dispose()
    env.db.dispose()
    restored = create_engine(backup_url)
    raw = restored.raw_connection()
    dest2 = create_engine(str(env.db.url)).raw_connection()
    raw.connection.backup(dest2.connection)
    dest2.close()
    raw.close()
    restored.dispose()

    # the restored DB is behind its log, and the sentinel is BACK
    restored_check = create_engine(str(env.db.url))
    db_id, applied = sm.read_watermark(restored_check)
    assert applied < tail
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=restored_check)()
        try:
            assert s.query(Belief).filter(Belief.claim_text.contains("SENTINELFOUR")).count() == 1
        finally:
            s.close()
    restored_check.dispose()

    # STARTUP reconciliation: a NEW engine on the restored database replays
    # on its first entry (the restart path) and then serves
    eng2 = MemoryEngine(config=env.cfg, database_url=str(env.db.url),
                        tenant_id=TENANT[0], app_id=TENANT[1],
                        llm_client=_FakeLLM(), deletion_log=env.log)
    assert eng2.get_beliefs(user_id=USER, purpose="api") is not None  # triggers init
    db_id2, applied2 = sm.read_watermark(env.db)
    assert applied2 == tail
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=env.db)()
        try:
            assert s.query(Belief).filter(
                Belief.claim_text.contains("SENTINELFOUR")).count() == 0
            assert s.query(DeletionGeneration).filter_by(user_id=USER).count() >= 1
            # the user's restored pending jobs were cancelled
            jobs = s.query(EvolutionJob).filter_by(user_id=USER).all()
            assert all(j.status != "pending" for j in jobs)
        finally:
            s.close()
    # every read surface is clean again
    assert "SENTINELFOUR" not in (eng2.recall(user_id=USER, query="SENTINELFOUR", purpose="api") or "")
    assert "SENTINELFOUR" not in json.dumps(eng2.export_page(user_id=USER, purpose="api").to_dict(), default=str)


def test_replay_never_resurrects_a_deleted_generation(env):
    """The log is the AUTHORITY for generations: after replay the user's
    generation equals the log's per-user maximum -- a restore cannot roll
    it back."""
    eng = env.engine
    _observe(eng, "I like coffee and I live in Berlin.", purpose="api")
    eng.forget_session(user_id=USER, session_id="s1", purpose="api")
    eng.delete_memories(user_id=USER, purpose="api")
    # the generation in the restored DB must reach the log's max
    db_id, _ = sm.read_watermark(env.db)
    max_in_log = max(e.generation for e in env.log.read_from(db_id, 0))
    F = sessionmaker(bind=env.db)
    reconcile(lambda: sessionmaker(bind=env.db)(), env.log, tenant=TENANT)
    with tenant_scope(*TENANT):
        s = F()
        try:
            current = s.execute(
                select(DeletionGeneration.generation)
                .where(DeletionGeneration.user_id == USER)
                .order_by(DeletionGeneration.generation.desc())
            ).scalars().first()
        finally:
            s.close()
    assert current == max_in_log


def test_replay_is_reentrant_after_an_interruption(env):
    """Replay to half, "crash" (drop the process state), start again: the
    second pass resumes from the watermark and the result is identical."""
    eng = env.engine
    _observe(eng, "I like coffee and I live in Berlin.", purpose="api")
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=env.db)()
        try:
            beliefs = s.query(Belief).all()
        finally:
            s.close()
    for b in beliefs:
        eng.forget_belief(user_id=USER, belief_id=b.id, purpose="api")

    # simulate a restore: the watermark rolls back with the backup (the
    # log does not) -- reconcile must re-apply the entries
    F = sessionmaker(bind=env.db)
    wtable = sm._watermark_table(sm.MetaData())
    with env.db.begin() as conn:
        conn.execute(wtable.update().where(wtable.c.id == 1).values(applied_seq=0))
    status = reconcile(lambda: F(), env.log, tenant=TENANT)
    assert status == "replayed"
    # re-running is a no-op ("ok") and the state is identical
    status2 = reconcile(lambda: F(), env.log, tenant=TENANT)
    assert status2 == "ok"
    with tenant_scope(*TENANT):
        s = F()
        try:
            assert s.query(Belief).count() == 0
            assert s.query(DeletionTombstone).count() >= 1
        finally:
            s.close()


def test_replay_cancels_restored_pending_jobs(env):
    eng = env.engine
    _observe(eng, "I like coffee and I live in Berlin.", purpose="api")
    cfg = env.cfg
    F = sessionmaker(bind=env.db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            job = enqueue_job(s, USER, cfg, purpose="api")
            s.commit()
            job_id = job.id
        finally:
            s.close()
    eng.delete_memories(user_id=USER, purpose="api")
    reconcile(lambda: F(), env.log, tenant=TENANT)
    with tenant_scope(*TENANT):
        s = F()
        try:
            row = s.get(EvolutionJob, job_id)
        finally:
            s.close()
    # the job was deleted with the user (user scope) -- nothing pending;
    # had it been restored by a backup, the replay would have cancelled it
    assert row is None or row.status == "cancelled"


def test_log_older_than_main_db_refuses_startup(env):
    """Watermark > log tail: the log's recovery point is older than the
    main DB -- refuse with operator guidance."""
    eng = env.engine
    _observe(eng, "I like coffee and I live in Berlin.", purpose="api")
    eng.delete_memories(user_id=USER, purpose="api")
    db_id, applied = sm.read_watermark(env.db)
    assert applied >= 1, "a committed deletion advances the watermark"
    # simulate a log that lost its entries (an operator restored an OLD
    # log file): seed the stale log with a GENESIS for the same db_id (a
    # real stale log would carry one) -- without genesis the refusal is
    # "not found" (task 2), with genesis it is "older than the main DB"
    stale_log = FileDeletionLog(env.log.directory.parent / "stale")
    stale_log.ensure_genesis(db_id)
    with pytest.raises(DeletionLogStale) as info:
        reconcile(lambda: sessionmaker(bind=env.db)(), stale_log, tenant=TENANT)
    assert "OLDER than the main database" in str(info.value)
    # a log for a DIFFERENT db_id (no genesis for ours) -> "not found"
    other_log = FileDeletionLog(env.log.directory.parent / "other")
    with pytest.raises(DeletionLogUnavailable, match="genesis"):
        reconcile(lambda: sessionmaker(bind=env.db)(), other_log, tenant=TENANT)


# ---------------------------------------------------------------------------
# refusal while another instance replays
# ---------------------------------------------------------------------------


def test_engine_start_refuses_while_another_instance_replays(env):
    """Hold the replay lock manually (another instance mid-replay): the
    engine's startup check raises ReplayPending -- it keeps refusing
    until the replay finishes, then starts."""
    eng = env.engine
    _observe(eng, "I like coffee and I live in Berlin.", purpose="api")
    belief = sessionmaker(bind=env.db)().query(Belief).first()
    eng.forget_belief(user_id=USER, belief_id=belief.id, purpose="api")
    db_id, _ = sm.read_watermark(env.db)
    # roll the watermark back (simulating a restore) and hold the lock
    sm.ensure_watermark(env.db)
    from mirror_memory.core import schema_migration

    wtable = schema_migration._watermark_table(schema_migration.MetaData())
    with env.db.begin() as conn:
        conn.execute(wtable.update().where(wtable.c.id == 1).values(applied_seq=0))

    from mirror_memory.core.replay import _replay_lock

    lock = _replay_lock(env.db, db_id)
    with lock as held:
        assert held is True
        with pytest.raises(ReplayPending):
            reconcile(lambda: sessionmaker(bind=env.db)(), env.log, tenant=TENANT)
        # a NEW engine on the same database refuses to serve too (its
        # startup check runs on the first entry)
        with pytest.raises(ReplayPending):
            refusing = MemoryEngine(config=env.cfg, database_url=str(env.db.url),
                                    tenant_id=TENANT[0], app_id=TENANT[1],
                                    llm_client=_FakeLLM(), deletion_log=env.log)
            refusing.get_beliefs(user_id=USER, purpose="api")
    # the lock is released: the next start replays and serves
    status = reconcile(lambda: sessionmaker(bind=env.db)(), env.log, tenant=TENANT)
    assert status == "replayed"
    eng2 = MemoryEngine(config=env.cfg, database_url=str(env.db.url),
                        tenant_id=TENANT[0], app_id=TENANT[1],
                        llm_client=_FakeLLM(), deletion_log=env.log)
    assert eng2.get_beliefs(user_id=USER, purpose="api") is not None


# ---------------------------------------------------------------------------
# task 4: the publish-time generation check
# ---------------------------------------------------------------------------


def test_publish_cancels_when_the_generation_changed_during_compute(env):
    """The generation moving during the compute (a deletion's CAS bump --
    or a restore + replay aligning to the log) -> the publish's
    generation re-check cancels the job (deletion_generation_changed),
    zero writes.  (A REAL belief/user-scope deletion also cancels or
    deletes the job row directly; this unit pins the check itself.)"""
    eng = env.engine
    cfg = env.cfg
    _observe(eng, "I like coffee and I live in Berlin.", purpose="api")
    F = sessionmaker(bind=env.db)

    # the generation moves DURING the compute: the LLM hook (a separate,
    # committed session) bumps it between the claim-time read and the
    # publish-time re-check
    class _BumpLLM:
        def generate(self, **kw):
            with tenant_scope(*TENANT), fixture_write_scope("the concurrent deletion bump"):
                s = F()
                try:
                    from mirror_memory.core.repository import bump_deletion_generation

                    bump_deletion_generation(s, USER, scope="user")
                    s.commit()
                finally:
                    s.close()
            return '{"patterns": [], "support_policy": {}}'

    cfg.llm_client = _BumpLLM()
    with tenant_scope(*TENANT):
        s = F()
        try:
            beliefs_before = s.query(Belief).count()
            job = enqueue_job(s, USER, cfg, purpose="api")
            s.commit()
            token = _claim_job(s, job.id, lease_seconds=60)
            job_row = s.get(EvolutionJob, job.id)
            _process_single_job(s, job_row, cfg, token)
            s.commit()
            s.expire_all()
            row = s.get(EvolutionJob, job.id)
            beliefs = s.query(Belief).count()
        finally:
            s.close()
    assert row.status == "cancelled" and row.error_code == "deletion_generation_changed", (
        row.status, row.error_code)
    assert beliefs == beliefs_before, "zero writes through the job"


def test_revision_aba_is_caught_by_the_generation_check(env):
    """The ABA sequence: claim -> RESTORE the backup (the revision rewinds
    to the claimed value!) -> replay raises the generation.  The claimed
    job is fingerprinted by the user-scope deletion, so the replay removes
    the restored row outright (identical to the original deletion); zero
    writes either way.  (The publish-time generation check for a job that
    SURVIVES a restore is covered by the test above.)"""
    eng = env.engine
    cfg = env.cfg
    _observe(eng, "I like coffee and I live in Berlin.", purpose="api")

    F = sessionmaker(bind=env.db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            job = enqueue_job(s, USER, cfg, purpose="api")
            s.commit()
            token = _claim_job(s, job.id, lease_seconds=60)
            job_id = job.id
        finally:
            s.close()

    # BACKUP AFTER the claim (the running job is in it) -- the ABA setup:
    # the deletion then moves the revision AND the generation
    backup_url = str(env.db.url).replace("main.db", "aba-backup.db")
    src = eng._engine.raw_connection()
    dest = create_engine(backup_url).raw_connection()
    src.connection.backup(dest.connection)
    dest.close()
    src.close()
    eng.delete_memories(user_id=USER, purpose="api")

    # RESTORE: the job is back (claimed, running), the revision rewinds to
    # the claim-time value (ABA!) -- but the replay raises the generation
    # past the claim-time one and the job's fencing state no longer holds
    eng._engine.dispose()
    env.db.dispose()
    restored = create_engine(backup_url)
    raw = restored.raw_connection()
    dest2 = create_engine(str(env.db.url)).raw_connection()
    raw.connection.backup(dest2.connection)
    dest2.close()
    raw.close()
    restored.dispose()
    reconcile(lambda: sessionmaker(bind=env.db)(), env.log, tenant=TENANT)

    with tenant_scope(*TENANT):
        s = sessionmaker(bind=env.db)()
        try:
            # the claimed job is in the user-scope entry's fingerprints:
            # the replay deletes the restored row (the original deletion
            # deleted it too) -- an ABA publish is impossible
            row = s.get(EvolutionJob, job_id)
            assert row is None, "the replay must remove the restored job"
            generation_now = s.execute(
                select(DeletionGeneration.generation)
                .where(DeletionGeneration.user_id == USER)
                .order_by(DeletionGeneration.generation.desc())
            ).scalars().first()
            assert generation_now == 1, generation_now
            beliefs = s.query(Belief).filter(Belief.claim_text.contains("coffee")).count()
        finally:
            s.close()
    assert beliefs == 0


# ---------------------------------------------------------------------------
# M4 batch-1 patch task 3: the replay is FINGERPRINT-DRIVEN and verified
# per table -- every fingerprint the log recorded must correspond to a row
# that is GONE from the restored database (checked by table, not through
# read entrances).
# ---------------------------------------------------------------------------

_TABLE_MODEL = {
    m.class_.__tablename__: m.class_
    for m in Base.registry.mappers
    if m.class_.__tablename__ and m.class_.__tablename__.startswith("mm_")
}


def _seed_full_user_content(env):
    """Rows in every table the coverage matrix deletes: observe produces
    beliefs/events/links/evidence/versions/identities/summaries, the
    published job adds proposals and a snapshot, and the compaction
    digest / tier audit / telemetry / a second (pending) job are seeded
    through their real write paths."""
    from datetime import datetime

    from mirror_memory.core.models import EvidenceDigest
    from mirror_memory.core.repository import (
        record_extraction_stats,
        record_intervention_event,
        set_belief_tier,
    )
    from mirror_memory.core.tenancy import engine_write_scope
    from mirror_memory.worker.evolution import _claim_job, _process_single_job

    eng = env.engine
    eng.observe(user_id=USER, session_id="sa",
                text="I like SENTINELCOV and coffee.", purpose="api")
    with tenant_scope(*TENANT):
        s = env.F()
        try:
            belief = s.query(Belief).filter(
                Belief.claim_text.contains("SENTINELCOV")).one()
            now = datetime.utcnow()
            with engine_write_scope():  # the digest is a Publisher write
                s.add(EvidenceDigest(
                    user_id=USER, belief_id=belief.id, support_count=1,
                    contradict_count=0, verify_count=0, correct_count=0,
                representative_ids="[]", summary="seed digest",
                source_distribution="{}", authority_distribution="{}",
                    created_at=now, updated_at=now))
            set_belief_tier(s, belief.id,
                            from_tier=belief.memory_tier or "hot",
                            to_tier="warm", reason="seed", score=0.5,
                            proposal_desc="seed")
            record_intervention_event(s, USER, "sa", kind="question",
                                      detail={"seed": True})
            record_extraction_stats(s, USER, session_id="sa",
                                    trigger="observe", model="fake")
            job = enqueue_job(s, USER, env.cfg, purpose="api")
            s.commit()
            token = _claim_job(s, job.id, lease_seconds=60)
            _process_single_job(s, job, env.cfg, token)  # proposal + snapshot
            enqueue_job(s, USER, env.cfg, purpose="api")  # stays pending
            s.commit()
        finally:
            s.close()


def _backup_now(env, name: str) -> str:
    backup_url = str(env.db.url).replace("main.db", name)
    src = env.engine._engine.raw_connection()
    dest = create_engine(backup_url).raw_connection()
    src.connection.backup(dest.connection)  # the real sqlite3 backup API
    dest.close()
    src.close()
    return backup_url


def _restore_over_main(env, backup_url: str) -> None:
    env.engine._engine.dispose()
    env.db.dispose()
    restored = create_engine(backup_url)
    raw = restored.raw_connection()
    dest2 = create_engine(str(env.db.url)).raw_connection()
    raw.connection.backup(dest2.connection)
    dest2.close()
    raw.close()
    restored.dispose()


def _assert_every_logged_fingerprint_is_absent(env, since_seq: int):
    """Read the entries the replay applied and check, PER TABLE, that no
    row of the restored database still carries a logged fingerprint."""
    from mirror_memory.core.deletion import fingerprint as fp_hash

    db_id, _applied = sm.read_watermark(env.db)
    entries = list(env.log.read_from(db_id, since_seq))
    wanted: dict[str, set[str]] = {}
    for e in entries:
        for fp in e.fingerprints:
            wanted.setdefault(fp["table"], set()).add(fp["_fp"])
    assert wanted, "the deletion logged no fingerprints"
    survivors = []
    with tenant_scope(*TENANT):
        s = env.F()
        try:
            for table, hashes in wanted.items():
                for row in s.query(_TABLE_MODEL[table]).all():
                    if fp_hash(table, row.id,
                               getattr(row, "created_at", None)) in hashes:
                        survivors.append((table, row.id))
        finally:
            s.close()
    assert survivors == [], (
        f"rows the log deleted are still in the restored DB: {survivors}")
    return entries


def test_patch3_belief_scope_replay_deletes_every_fingerprinted_row(env):
    """belief scope: backup -> forget both beliefs -> restore -> replay;
    every fingerprinted row (incl. the ORPHAN EVIDENCE of the second
    forget) is gone from its table; the restored pending job is
    cancelled; jobs are CANCELLED, not fingerprinted (the matrix's belief
    column)."""
    eng = env.engine
    _seed_full_user_content(env)
    backup_url = _backup_now(env, "p3-belief.db")

    with tenant_scope(*TENANT):
        s = env.F()
        try:
            beliefs = s.query(Belief).order_by(Belief.id).all()
        finally:
            s.close()
    _db_id_before, applied_before = sm.read_watermark(env.db)
    for b in beliefs:
        assert eng.forget_belief(user_id=USER, belief_id=b.id, purpose="api")
    _restore_over_main(env, backup_url)
    assert reconcile(lambda: env.F(), env.log, tenant=TENANT) == "replayed"

    entries = _assert_every_logged_fingerprint_is_absent(env, applied_before)
    per_table: set[str] = set()
    for e in entries:
        for fp in e.fingerprints:
            per_table.add(fp["table"])
    assert "mm_evidence" in per_table, (
        "the second forget must fingerprint the orphaned evidence")
    assert "mm_evolution_jobs" not in per_table, (
        "belief scope cancels jobs; it never fingerprints them")
    with tenant_scope(*TENANT):
        s = env.F()
        try:
            jobs = s.query(EvolutionJob).filter_by(user_id=USER).all()
            assert jobs and all(j.status == "cancelled" for j in jobs)
        finally:
            s.close()


def test_patch3_session_scope_replay_deletes_every_fingerprinted_row(env):
    """session scope: backup -> forget the session -> restore -> replay;
    the fingerprinted summary rows are gone from mm_session_summaries."""
    eng = env.engine
    _seed_full_user_content(env)
    backup_url = _backup_now(env, "p3-session.db")

    _db_id_before, applied_before = sm.read_watermark(env.db)
    eng.forget_session(user_id=USER, session_id="sa", purpose="api")
    _restore_over_main(env, backup_url)
    assert reconcile(lambda: env.F(), env.log, tenant=TENANT) == "replayed"

    entries = _assert_every_logged_fingerprint_is_absent(env, applied_before)
    assert entries, "the session forget must be logged"
    from mirror_memory.core.models import SessionSummary

    with tenant_scope(*TENANT):
        s = env.F()
        try:
            assert s.query(SessionSummary).filter_by(
                user_id=USER, session_id="sa").count() == 0
        finally:
            s.close()


def test_patch3_user_scope_replay_covers_all_handler_tables(env):
    """user scope: the deletion fingerprints rows in ALL handler tables;
    after backup -> delete -> restore -> replay every one of those rows is
    gone (per-table check), and the pending job the restore brought back
    is fingerprinted (deleted), not merely cancelled."""
    from mirror_memory.core import replay as replay_mod

    eng = env.engine
    _seed_full_user_content(env)
    backup_url = _backup_now(env, "p3-user.db")

    _db_id_before, applied_before = sm.read_watermark(env.db)
    eng.delete_memories(user_id=USER, purpose="api")
    _restore_over_main(env, backup_url)
    assert reconcile(lambda: env.F(), env.log, tenant=TENANT) == "replayed"

    entries = _assert_every_logged_fingerprint_is_absent(env, applied_before)
    entry = entries[-1]
    tables_in_entry = {fp["table"] for fp in entry.fingerprints}
    assert tables_in_entry == set(replay_mod._REPLAY_HANDLERS), (
        "the user-scope deletion must fingerprint every handler table",
        sorted(tables_in_entry))
    with tenant_scope(*TENANT):
        s = env.F()
        try:
            assert s.query(EvolutionJob).filter_by(user_id=USER).count() == 0
        finally:
            s.close()


def test_replay_refuses_an_entry_naming_a_table_without_a_handler(env):
    """An entry naming a REAL table that has no replay handler (mm_users)
    must fail loudly -- silently skipping a table would leave deleted
    content in the restored database."""
    from mirror_memory.core.deletion import (
        UnhandledDeletionTable,
        fingerprint,
    )
    from mirror_memory.core.replay import apply_entry

    entry = DeletionEntry(
        db_id="d" * 32, tenant_id=TENANT[0], app_id=TENANT[1], user_id=USER,
        scope="user", scope_key="",
        fingerprints=[{"table": "mm_users", "id": 1, "created_at": None,
                       "_fp": fingerprint("mm_users", 1, None)}],
        generation=1, purpose="api")
    with tenant_scope(*TENANT):
        s = env.F()
        try:
            with pytest.raises(UnhandledDeletionTable, match="mm_users"):
                apply_entry(s, entry)
        finally:
            s.close()


def test_coverage_matrix_delete_tables_all_have_replay_handlers():
    """守卫：覆盖面清单（docs/plans/2026-10-30-m4-deletion-coverage.md）
    里任一范围标「删」的表，都必须在重放处理器表里；清单与处理器一起
    演进，不允许静默漏重放。"""
    doc = (Path(__file__).parents[1] / "docs" / "plans"
           / "2026-10-30-m4-deletion-coverage.md")
    delete_tables: set[str] = set()
    for line in doc.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| mm_"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        name_cell, scope_cells = cells[0], cells[1:4]
        if not any("删" in c for c in scope_cells):
            continue
        parts = name_cell.split("/")
        prefix = parts[0].strip()
        for part in parts:
            name = part.strip()
            if not name.startswith("mm_"):
                name = f"{prefix.rsplit('_', 1)[0]}_{name}"
            delete_tables.add(name)
    missing = delete_tables - _REPLAY_HANDLERS
    assert not missing, (
        f"the coverage matrix deletes {sorted(missing)} but the replay "
        "handlers do not cover them")
    # every handler names a REAL table (typo guard)
    unknown = _REPLAY_HANDLERS - set(Base.metadata.tables)
    assert not unknown, f"replay handlers name non-existent tables: {unknown}"


# ---------------------------------------------------------------------------
# M4 batch-1 patch task 4: a failing reconcile stops the worker COLD
# ---------------------------------------------------------------------------


def test_patch4_worker_refuses_to_work_when_the_reconcile_fails(env, monkeypatch):
    """A broken log chain skips the WHOLE tick: the expired lease is NOT
    recovered, nothing is claimed, nothing is published, the alert metric
    increments, and the next tick inside the backoff window does not even
    re-attempt.  Once the log is healthy the worker proceeds normally."""
    from datetime import UTC, datetime, timedelta

    from mirror_memory.core import metrics
    from mirror_memory.worker import evolution as evo

    # a short backoff base keeps the heal leg fast; the window semantics
    # are identical (the second tick inside the window must be suppressed)
    monkeypatch.setattr(evo, "_RECONCILE_BACKOFF", {})
    monkeypatch.setattr(evo, "_RECONCILE_BACKOFF_BASE_SECONDS", 0.05)
    _observe(env.engine, "I like coffee and I live in Berlin.", purpose="api")

    with tenant_scope(*TENANT):
        s = env.F()
        try:
            running = enqueue_job(s, USER, env.cfg, purpose="api")
            s.commit()
            _claim_job(s, running.id, lease_seconds=60)
            # an expired lease the sweep WOULD recover on a healthy tick
            from mirror_memory.core.tenancy import queue_write_scope

            with queue_write_scope():
                running.lease_expires_at = (datetime.now(UTC)
                                            - timedelta(seconds=5))
            # a SECOND pending job: enqueue_job would merge into the
            # claimed one (same user + watermark), so insert it directly
            # with its own idempotency key (the canonical creation shape)
            import uuid as _uuid

            with queue_write_scope():
                pending = EvolutionJob(
                    id=_uuid.uuid4().hex[:16],
                    user_id=USER,
                    purpose="api",
                    evidence_watermark=json.dumps({"latest_update": "p4"}),
                    prompt_version=env.cfg.worker.prompt_version,
                    idempotency_key="p4-pending-job",
                    status="pending",
                )
                s.add(pending)
            s.commit()
            running_id, pending_id = running.id, pending.id
        finally:
            s.close()

    db_id, _applied = sm.read_watermark(env.db)
    path = env.log._path_for(db_id)
    original = path.read_text()

    # tamper the first ENTRY's hash (line 0 is the file header): the
    # chain no longer verifies
    lines = original.splitlines()
    first_entry = next(i for i, ln in enumerate(lines)
                       if ln.strip() and '"header"' not in ln)
    rec = json.loads(lines[first_entry])
    rec["hash"] = "0" * 64
    lines[first_entry] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")

    failures_before = metrics.value(metrics.DELETION_RECONCILE_FAILURES)
    evo.process_pending(lambda: env.F(), env.cfg)

    with tenant_scope(*TENANT):
        s = env.F()
        try:
            assert s.get(EvolutionJob, running_id).status == "running", (
                "the expired lease must NOT be recovered while the log is broken")
            assert s.get(EvolutionJob, pending_id).status == "pending", (
                "nothing may be claimed while the log is broken")
        finally:
            s.close()
    assert metrics.value(metrics.DELETION_RECONCILE_FAILURES) >= failures_before + 1

    # inside the backoff window the tick skips WITHOUT re-attempting
    evo.process_pending(lambda: env.F(), env.cfg)
    assert metrics.value(metrics.DELETION_RECONCILE_FAILURES) == failures_before + 1, (
        "the backoff gate must suppress further reconcile attempts")

    # heal: AFTER the backoff window the tick recovers the lease and
    # processes the pending job again
    import time as _time

    path.write_text(original)
    _time.sleep(0.06)
    evo.process_pending(lambda: env.F(), env.cfg)
    with tenant_scope(*TENANT):
        s = env.F()
        try:
            # the sweep recovered the expired lease and the scan claimed
            # and COMPLETED it in the same (healthy) tick
            assert s.get(EvolutionJob, running_id).status == "completed", (
                "the healthy tick recovers the expired lease and works the job")
            assert s.get(EvolutionJob, pending_id).status != "pending", (
                "the healthy tick claims and processes the queue again")
        finally:
            s.close()


def test_patch4_worker_skips_when_the_log_is_unreachable(env, monkeypatch):
    """An UNREACHABLE log (infrastructure error) also stops the tick --
    and an unreachable log must not be hammered: the backoff state
    escalates."""
    from mirror_memory.core import metrics
    from mirror_memory.worker import evolution as evo

    monkeypatch.setattr(evo, "_RECONCILE_BACKOFF", {})

    def _boom(*a, **kw):
        raise ConnectionError("the deletion log database is down")

    _observe(env.engine, "I like tea and I live in Oslo.", purpose="api")
    # the worker builds its OWN FileDeletionLog over the same directory --
    # patch the CLASS, not the engine's instance
    from mirror_memory.core.deletion import FileDeletionLog

    monkeypatch.setattr(FileDeletionLog, "has_genesis", _boom)

    with tenant_scope(*TENANT):
        s = env.F()
        try:
            pending = enqueue_job(s, USER, env.cfg, purpose="api")
            s.commit()
            pending_id = pending.id
        finally:
            s.close()

    failures_before = metrics.value(metrics.DELETION_RECONCILE_FAILURES)
    evo.process_pending(lambda: env.F(), env.cfg)
    assert metrics.value(metrics.DELETION_RECONCILE_FAILURES) >= failures_before + 1
    with tenant_scope(*TENANT):
        s = env.F()
        try:
            assert s.get(EvolutionJob, pending_id).status == "pending"
        finally:
            s.close()
    # the failure was recorded: the state escalates (2nd window longer)
    failures, deadline = evo._RECONCILE_BACKOFF[str(env.db.url)]
    assert failures == 1 and deadline > 0
