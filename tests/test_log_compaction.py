"""M4 batch 2 task 4: deletion-log retention and compaction.

- ``prune(before_seq)`` removes the entries a retention period lets go
  and writes a CHECKPOINT entry that carries the chain hash at that
  point; the chain continues from the checkpoint, startup verification
  stays whole, and the genesis (the lineage proof) is never pruned;
- a main database whose watermark lies below the checkpoint is OLDER
  than the log's retention: startup refuses (DeletionLogPruned) --
  replaying only the remainder would resurrect deleted content;
- on the cloud the DELETE runs on the MAINTENANCE connection (the app
  role has no DELETE); on the embedded backend the log owner prunes;
- the background pass prunes exactly the entries past the retention;
- a sync peer whose cursor lies before the checkpoint cannot be served:
  export raises TombstoneCursorPruned (rebuild, don't silently skip);
- configuration: retention_days shorter than backup_retention_days is a
  startup configuration error.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import (
    DeletionEntry,
    DeletionLogStale,
    FileDeletionLog,
)
from mirror_memory.core.models import Base
from mirror_memory.core.replay import reconcile
from mirror_memory.core.tombstones import TombstoneCursorPruned
from mirror_memory.worker.maintenance import prune_deletion_log

TENANT = ("default", "default")
USER = "u-prune"


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


def _log_with_entries(tmp_path, n: int = 5) -> tuple[FileDeletionLog, str]:
    log = FileDeletionLog(tmp_path / "deletions")
    db_id = "d" * 32
    for i in range(1, n + 1):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="api",
        ))
    return log, db_id


def test_prune_writes_a_checkpoint_and_the_chain_stays_whole(tmp_path):
    log, db_id = _log_with_entries(tmp_path, 5)
    tail_before, hash_before = log.head(db_id)
    assert log.verify_chain(db_id) is None

    removed = log.prune(db_id, 3)
    assert removed == 3
    assert log.head(db_id) == (tail_before, hash_before), "the tail is untouched"
    assert log.checkpoint_seq(db_id) == 3
    # genesis survives (the lineage proof), the checkpoint carries the
    # chain forward, entries 4..5 are intact
    assert log.has_genesis(db_id)
    scopes = [(e.seq, e.scope) for e in log.read_from(db_id, 0)]
    assert (1, "checkpoint") in scopes or (3, "checkpoint") in scopes
    assert (3, "checkpoint") in scopes
    assert [e.seq for e in log.read_from(db_id, 3)] == [4, 5]
    log.verify_chain(db_id)  # the startup check still passes
    # pruning again through the same point is a no-op (the checkpoint IS
    # the record)
    assert log.prune(db_id, 3) == 0
    log.verify_chain(db_id)
    # a deeper prune re-checkpoints (the old checkpoint is subsumed)
    assert log.prune(db_id, 4) == 1
    assert log.checkpoint_seq(db_id) == 4
    log.verify_chain(db_id)


def test_prune_below_the_head_keeps_replay_working(tmp_path):
    """A restored database whose watermark is AT or ABOVE the checkpoint
    replays the remaining entries; BELOW the checkpoint refuses."""
    from mirror_memory.core.deletion import DeletionLogPruned

    log, db_id = _log_with_entries(tmp_path, 5)
    eng = create_engine(f"sqlite:///{tmp_path / 'main.db'}")
    Base.metadata.create_all(eng)
    sm.ensure_watermark(eng)
    # pair the main DB's watermark with THIS log's life (what a real
    # deployment's watermark row already is)
    wtable0 = sm._watermark_table(sm.MetaData())
    with eng.begin() as conn:
        conn.execute(wtable0.update().where(wtable0.c.id == 1).values(db_id=db_id))

    log.prune(db_id, 3)
    # watermark 4 (>= checkpoint): entries 4..5 replay; the checkpoint is
    # skipped as a non-deletion
    wtable = sm._watermark_table(sm.MetaData())
    with eng.begin() as conn:
        conn.execute(wtable.update().where(wtable.c.id == 1).values(applied_seq=4))
    status = reconcile(lambda: sessionmaker(bind=eng)(), log, tenant=TENANT)
    assert status == "replayed"
    assert sm.read_watermark(eng)[1] == 5
    assert log.verify_chain(db_id) is None

    # watermark 2 (< checkpoint 3): refuse -- never a partial replay
    with eng.begin() as conn:
        conn.execute(wtable.update().where(wtable.c.id == 1).values(applied_seq=2))
    from mirror_memory.core.deletion import DeletionLogPruned

    with pytest.raises(DeletionLogPruned) as info:
        reconcile(lambda: sessionmaker(bind=eng)(), log, tenant=TENANT)
    assert "compact" in str(info.value) and "checkpoint" in str(info.value)
    eng.dispose()


def test_config_refuses_retention_shorter_than_backup_retention(tmp_path):
    cfg = load_config("config/")
    cfg.deletion_log.retention_days = 7
    cfg.deletion_log.backup_retention_days = 30
    from mirror_memory.core.deletion import deletion_log_from_config

    with pytest.raises(ValueError, match="backup_retention_days"):
        deletion_log_from_config(cfg.deletion_log, f"sqlite:///{tmp_path / 'x.db'}")


def test_prune_pass_deletes_exactly_the_retired_entries(tmp_path):
    """The background pass prunes through the newest entry past the
    retention window -- and stops at the first entry still inside it."""
    # entries 1-2 are 40 days old, 3-4 are today (created_at is part of
    # the hashed payload, so the ages are stamped at append time)
    now = datetime.utcnow()
    log = FileDeletionLog(tmp_path / "deletions")
    db_id = "d" * 32
    for i in range(1, 5):
        created = now - timedelta(days=40) if i <= 2 else now
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="api", created_at=created))

    class _Cfg:
        class deletion_log:  # noqa: N801 - a bare namespace for the pass
            retention_days = 35

    assert prune_deletion_log(_Cfg(), log, db_id) == 2, "entries 1-2 retire"
    assert log.checkpoint_seq(db_id) == 2
    assert [e.seq for e in log.read_from(db_id, 2)] == [3, 4]
    log.verify_chain(db_id)


def test_export_refuses_a_cursor_before_the_checkpoint(tmp_path):
    """Compaction must not break the sync: a peer cursor before the
    checkpoint gets an explicit rebuild demand, never a silent skip."""
    from mirror_memory.core.tombstones import export_from_log

    log, db_id = _log_with_entries(tmp_path, 4)
    log.prune(db_id, 2)
    with pytest.raises(TombstoneCursorPruned) as info:
        export_from_log(log, db_id, 0, 100)
    assert "checkpoint" in str(info.value) and "rebuild" in str(info.value)
    # a cursor at or after the checkpoint exports normally
    batch, cursor, _skipped = export_from_log(log, db_id, 2, 100)
    assert [e.origin_seq for e in batch.entries] == [3, 4] and cursor == 4
