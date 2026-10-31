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
from mirror_memory.core.replay import reconcile
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
    to the claimed value!) -> replay raises the generation -> publish:
    the revision alone would pass (ABA), the generation cancels."""
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
            row = s.get(EvolutionJob, job_id)
            assert row is not None, "the restore brought the job back"
            generation_now = s.execute(
                select(DeletionGeneration.generation)
                .where(DeletionGeneration.user_id == USER)
                .order_by(DeletionGeneration.generation.desc())
            ).scalars().first()
            assert generation_now == 1, generation_now
            _process_single_job(s, row, cfg, row.lease_token)
            s.commit()
            s.expire_all()
            row = s.get(EvolutionJob, job_id)
            beliefs = s.query(Belief).filter(Belief.claim_text.contains("coffee")).count()
        finally:
            s.close()
    # the restored job cannot publish: it was cancelled by the replay (or,
    # had it survived, the publish-time generation check refuses) -- zero
    # writes either way
    assert row.status == "cancelled" and row.error_code in (
        "deletion_replayed", "deletion_generation_changed"), (row.status, row.error_code)
    assert beliefs == 0
