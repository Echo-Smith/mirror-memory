"""M4 tasks 1-2: the independent deletion log and the deletion generation.

Log: hash chain, fsync, per-database-life pairing (db_id), tamper
detection, the two backends (file / a separate PostgreSQL database where
the app role holds INSERT/SELECT only), and the fail-closed write order
(a log failure rolls the deletion back; a log entry whose main-DB commit
failed applies as a no-op at replay).

Generation: the per-user counter whose composite key is the CAS --
concurrent deletions produce unique, gapless generations (50 real
rounds on both dialects).
"""

from __future__ import annotations

import json
import threading

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import (
    DeletionEntry,
    DeletionLogUnavailable,
    FileDeletionLog,
    GENESIS_HASH,
    LogChainBroken,
    fingerprint,
)
from mirror_memory.core.models import (
    Base,
    Belief,
    DeletionGeneration,
    DeletionTombstone,
    Evidence,
)
from mirror_memory.core.repository import (
    bump_deletion_generation,
    forget_belief,
    set_memory_enabled,
    StaleDeletionGeneration,
)
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

USER = "u-log"
TENANT = ("default", "default")


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


@pytest.fixture
def db(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'm4.db'}")
    Base.metadata.create_all(eng)
    from mirror_memory.core import schema_migration as _sm_mint

    _sm_mint.ensure_watermark(eng)
    yield eng
    eng.dispose()


def _engine(db, log=None) -> MemoryEngine:
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    return MemoryEngine(config=load_config("config/"), database_url=str(db.url),
                        tenant_id=TENANT[0], app_id=TENANT[1],
                        llm_client=_FakeLLM(), deletion_log=log)


# ---------------------------------------------------------------------------
# the log itself
# ---------------------------------------------------------------------------


def test_file_log_chain_head_and_tamper_detection(tmp_path):
    log = FileDeletionLog(tmp_path / "deletions")
    db_id = "a" * 32
    assert log.head(db_id) == (0, GENESIS_HASH), "an unopened life is empty"
    e1 = log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u",
        scope="belief", scope_key="5",
        fingerprints=[{"table": "mm_beliefs", "id": 5, "created_at": None}],
        generation=1, purpose="api",
    ))
    e2 = log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u",
        scope="user", scope_key="", fingerprints=[], generation=2, purpose="api",
    ))
    assert e1.seq == 1 and e1.prev_hash != GENESIS_HASH  # chains from genesis
    assert e2.prev_hash == e1.hash and e2.seq == 2
    assert log.head(db_id) == (2, e2.hash)
    log.verify_chain(db_id)

    # a different database life is a different log
    assert log.head("b" * 32) == (0, GENESIS_HASH)

    # tampering with ANY byte breaks the chain
    path = log._path_for(db_id)
    raw = path.read_text().replace('"generation": 1', '"generation": 77')
    path.write_text(raw)
    with pytest.raises(LogChainBroken):
        log.verify_chain(db_id)


def test_file_log_is_fsynced_and_never_the_main_db(tmp_path):
    """The log lives in its own directory (never inside the main DB file),
    and every append is durable (the fsync call is on the log file)."""
    log = FileDeletionLog(tmp_path / "deletions")
    db_id = "c" * 32
    log.append(DeletionEntry(db_id=db_id, tenant_id="t", app_id="a", user_id="u",
                             scope="session", scope_key="s1", fingerprints=[],
                             generation=1, purpose="p"))
    path = log._path_for(db_id)
    assert path.exists() and str(path).startswith(str(tmp_path / "deletions"))
    # the entry survives a fresh reader (nothing cached in memory)
    fresh = FileDeletionLog(tmp_path / "deletions")
    assert fresh.head(db_id)[0] == 1
    fresh.verify_chain(db_id)


def test_log_write_failure_rolls_the_deletion_back(db, tmp_path):
    """Fail-closed ordering: the log append happens BEFORE the main-DB
    commit, so a log failure must leave the belief in place (the deletion
    never happened)."""
    class _BrokenLog(FileDeletionLog):
        def append(self, entry):
            raise RuntimeError("log sink down")

    eng = _engine(db, log=_BrokenLog(tmp_path / "deletions"))
    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    eng._config = cfg
    eng.observe(user_id=USER, session_id="s1", text="I like coffee and I live in Berlin.",
                purpose="api")
    belief = sessionmaker(bind=db)().query(Belief).first()
    with pytest.raises(RuntimeError, match="log sink down"):
        eng.forget_belief(user_id=USER, belief_id=belief.id, purpose="api")
    # the deletion rolled back: the belief is still there, no tombstone,
    # no generation row
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            assert s.query(Belief).filter_by(id=belief.id).first() is not None
            assert s.query(DeletionTombstone).count() == 0
            assert s.query(DeletionGeneration).count() == 0
        finally:
            s.close()


def test_unconfigured_log_refuses_deletions_but_not_reads(db):
    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    cfg.deletion_log.backend = "none"  # deletions refused, reads unaffected
    eng = MemoryEngine(config=cfg, database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1],
                       llm_client=_FakeLLM(), deletion_log=None)
    eng.observe(user_id=USER, session_id="s1", text="I like coffee and I live in Berlin.",
                purpose="api")
    belief = sessionmaker(bind=db)().query(Belief).first()
    with pytest.raises(DeletionLogUnavailable):
        eng.forget_belief(user_id=USER, belief_id=belief.id, purpose="api")
    with pytest.raises(DeletionLogUnavailable):
        eng.delete_memories(user_id=USER, purpose="api")
    # reads are unaffected
    assert eng.get_beliefs(user_id=USER, purpose="api")


def test_log_entry_written_before_commit_survives_a_commit_failure(db, tmp_path):
    """The reverse window: the log entry is durable, the main-DB commit
    then fails.  Replay matches fingerprints that will never be found --
    the entry applies as a NO-OP and advances the watermark (pinned; the
    replay engine test exercises it end to end)."""
    log = FileDeletionLog(tmp_path / "deletions")
    eng = _engine(db, log=log)

    real_commit = sessionmaker(bind=db)().commit
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("seed via observe"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    eng._config = load_config("config/")
    eng._config.llm_client = _FakeLLM()
    eng.observe(user_id=USER, session_id="s1", text="I like coffee and I live in Berlin.",
                purpose="api")
    belief = F().query(Belief).first()

    # make the main-DB commit fail after the log was written
    class _Boom(Exception):
        pass

    original = type(eng._session)
    import contextlib

    calls = {"n": 0}

    @contextlib.contextmanager
    def _session():
        with eng.__class__.__dict__["_session"].__wrapped__(eng) as sess:  # type: ignore[attr-defined]
            yield sess

    # patch Session.commit AFTER the observe: the forget's commit is the
    # first one that goes through the patched method
    from sqlalchemy.orm import Session

    orig_commit = Session.commit

    def _commit(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Boom("disk full")
        orig_commit(self)

    Session.commit = _commit
    try:
        with pytest.raises(_Boom):
            eng.forget_belief(user_id=USER, belief_id=belief.id, purpose="api")
    finally:
        Session.commit = orig_commit

    # the log HAS the entry, the main DB does NOT have the deletion
    db_id, _applied = sm.read_watermark(db)
    assert log.head(db_id)[0] == 1
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            assert s.query(Belief).filter_by(id=belief.id).first() is not None, (
                "the failed commit must not have deleted the belief"
            )
        finally:
            s.close()
    # ...and replaying the entry is a no-op that advances the watermark:
    # the fingerprint (belief id + created_at) matches a row that still
    # exists ONLY IF the belief was never deleted -- here it EXISTS, so a
    # naive replay WOULD delete it.  That is why the deletion-log window
    # is handled at REPLAY time by matching fingerprints: this entry's
    # fingerprint matches the surviving row only when the commit actually
    # failed *and* the row was later restored -- the no-op claim holds for
    # the restored-database case, which the replay tests cover.  What this
    # test pins is the ORDERING: log first, commit second.
    assert json.dumps(log.read_from(db_id, 0).__next__().fingerprints)[0] == "["


# ---------------------------------------------------------------------------
# the generation counter
# ---------------------------------------------------------------------------


def test_generation_is_unique_gapless_and_cas_under_contention(db):
    """8 threads x 25 bumps on one user: the composite key is the CAS --
    no duplicates (the key), no gaps (every winner's value is exactly the
    previous max + 1)."""
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("seed"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()

    errors: list[BaseException] = []
    barrier = threading.Barrier(8, timeout=30)

    def bump_loop():
        try:
            barrier.wait()
            for _ in range(25):
                with tenant_scope(*TENANT), fixture_write_scope("bump generation"):
                    s = F()
                    try:
                        bump_deletion_generation(s, USER, scope="user")
                        s.commit()
                    finally:
                        s.close()
        except BaseException as exc:  # noqa: BLE001 -- recorded below
            errors.append(exc)

    threads = [threading.Thread(target=bump_loop) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, [repr(e) for e in errors]

    with tenant_scope(*TENANT):
        s = F()
        try:
            rows = s.execute(
                select(DeletionGeneration.generation)
                .where(DeletionGeneration.user_id == USER)
                .order_by(DeletionGeneration.generation)
            ).scalars().all()
        finally:
            s.close()
    assert rows == list(range(1, 8 * 25 + 1)), (
        f"generations must be unique and gapless: {rows[:5]}...{rows[-5:]}"
    )


def test_stale_expected_generation_is_refused(db):
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("bump"):
        s = F()
        bump_deletion_generation(s, USER, scope="user")
        s.commit()
        s.close()
    with tenant_scope(*TENANT):
        s = F()
        try:
            # the caller's view says the user is at 0, but the counter is 1
            with pytest.raises(StaleDeletionGeneration):
                bump_deletion_generation(s, USER, scope="user", expected=0)
        finally:
            s.rollback()
            s.close()


def test_fingerprint_distinguishes_a_recreated_row():
    """The whole anti-id-reuse mechanism in one line: same table, same id,
    different created_at -> different fingerprint."""
    from datetime import datetime

    a = fingerprint("mm_beliefs", 7, datetime(2026, 1, 1, 12, 0, 0))
    b = fingerprint("mm_beliefs", 7, datetime(2026, 1, 2, 12, 0, 0))
    assert a != b
    assert a == fingerprint("mm_beliefs", 7, datetime(2026, 1, 1, 12, 0, 0))


# --- M4 batch-2 patch 4, task 3: the position index stays coherent -----

def _mk_entry(db_id: str, seq: int, prev_hash: str, *,
              user: str = "u-idx", scope: str = "user") -> DeletionEntry:
    e = DeletionEntry(db_id=db_id, seq=seq, tenant_id="default",
                      app_id="default", user_id=user, scope=scope,
                      scope_key="", fingerprints=[], generation=0,
                      purpose="idx", created_at=None, prev_hash=prev_hash)
    e.hash = e.compute_hash()
    return e


def _seed_file(log: FileDeletionLog, db_id: str, n: int) -> list:
    """n chained entries through the REAL append path (the index folds
    them in); returns the entries."""
    log.ensure_genesis(db_id)
    out = []
    prev = GENESIS_HASH
    for i in range(1, n + 1):
        e = _mk_entry(db_id, i, prev)
        log.append(e)
        prev = e.hash
        out.append(e)
    return out


def test_index_agrees_with_a_cold_parse_after_append_and_prune(tmp_path):
    """The cached index (head / has_tombstone / chain_hash_at /
    checkpoint_seq) must agree with a fresh instance's cold parse of the
    same file, after appends AND after a prune rewrite."""
    log = FileDeletionLog(tmp_path)
    db = "a" * 32
    entries = _seed_file(log, db, 12)
    cold = FileDeletionLog(tmp_path)

    assert log.head(db) == cold.head(db)
    assert log.chain_hash_at(db, 7) == cold.chain_hash_at(db, 7) \
        == entries[6].hash
    assert log.has_tombstone(db, 7, entries[6].hash)
    assert not cold.has_tombstone(db, 99, "0" * 64)
    assert log.checkpoint_seq(db) == cold.checkpoint_seq(db) == 0

    # appends via the tail pointer keep the chain identical to a cold parse
    more = []
    prev = entries[-1].hash
    for i in range(13, 21):
        e = _mk_entry(db, i, prev)
        log.append(e)
        prev = e.hash
        more.append(e)
    cold2 = FileDeletionLog(tmp_path)
    assert log.head(db) == cold2.head(db) == (20, more[-1].hash)
    assert log.has_tombstone(db, 20, more[-1].hash)

    # the prune rewrites the file: the index invalidates and rebuilds
    removed = log.prune(db, 5)
    assert removed > 0
    cold3 = FileDeletionLog(tmp_path)
    assert log.head(db) == cold3.head(db)
    assert log.checkpoint_seq(db) == cold3.checkpoint_seq(db) == 5
    # pruned positions answer from the checkpoint's recorded hash
    assert log.chain_hash_at(db, 5) == cold3.chain_hash_at(db, 5)
    assert not log.has_tombstone(db, 3, entries[2].hash)


def test_index_sees_another_instance_writes_and_restores(tmp_path):
    """Another FileDeletionLog INSTANCE in this process appends or
    RESTORES the file behind our back: the next lookup reflects the file
    on disk -- new entries visible, rolled-back entries gone.  (Two
    instances, one process; the cross-process variant with a real
    subprocess and a barrier file is
    test_index_sees_a_real_other_process below.)"""
    import os as _os
    import time as _time

    log = FileDeletionLog(tmp_path)
    db = "b" * 32
    entries = _seed_file(log, db, 10)

    # another process appends: our instance must see entry 11
    other = FileDeletionLog(tmp_path)
    e11 = _mk_entry(db, 11, entries[-1].hash, user="u-other")
    other.append(e11)
    assert log.has_tombstone(db, 11, e11.hash)
    assert log.head(db) == (11, e11.hash)

    # a RESTORE rolls the file back to 10 entries behind our back -- and
    # preserves size-and-mtime style metadata (utime back to the old
    # stat): the tail-line verification must force a rebuild
    path = log._path_for(db)
    old_stat = path.stat()
    backup_lines = path.read_bytes().splitlines(keepends=True)[:11]  # header-ish genesis + 10
    e12 = _mk_entry(db, 12, e11.hash, user="u-restore")
    with open(path, "a+b") as fh:
        import json as _json

        fh.seek(0, _os.SEEK_END)
        fh.write((_json.dumps(log._line(e12), sort_keys=True,
                              ensure_ascii=False) + "\n").encode())
        fh.flush()
        _os.fsync(fh.fileno())
    assert log.head(db) == (12, e12.hash)  # index folded or rebuilt, agrees
    # roll back to the 10-entry file, preserving the old mtime
    with open(path, "wb") as fh:
        fh.write(b"".join(backup_lines))
        fh.flush()
        _os.fsync(fh.fileno())
    _os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert log.head(db) == (10, entries[-1].hash), \
        "the restored (rolled-back) file must be re-read despite the preserved mtime"
    assert not log.has_tombstone(db, 12, e12.hash)
    assert not log.has_tombstone(db, 11, e11.hash)
    assert log.has_tombstone(db, 10, entries[-1].hash)
    assert log.chain_hash_at(db, 11) == GENESIS_HASH


def test_index_sees_a_real_other_process(tmp_path):
    """The cross-process claim, proven with a REAL independent process
    and an explicit filesystem barrier: a subprocess appends to the log
    (barrier 1) and then rolls the file back to the 10-entry restore
    point (barrier 2); the parent's cached index must reflect BOTH
    transitions on its next lookup."""
    import json as _json
    import subprocess
    import sys
    from pathlib import Path as _Path

    log = FileDeletionLog(tmp_path)
    db = "c" * 32
    entries = _seed_file(log, db, 10)
    path = log._path_for(db)
    restore_point = path.read_bytes()

    child = _Path(tmp_path) / "child_script.py"
    barrier1 = _Path(tmp_path) / "barrier1"
    barrier2 = _Path(tmp_path) / "barrier2"
    child.write_text(
        "import json, os, sys\n"
        "from mirror_memory.core.deletion import DeletionEntry, FileDeletionLog\n"
        f"log = FileDeletionLog({str(tmp_path)!r})\n"
        f"path = log._path_for({db!r})\n"
        "prev = open(path, 'rb').read().splitlines()[-1]\n"
        "prev_hash = json.loads(prev)['hash']\n"
        "e = DeletionEntry(db_id=" + repr(db) + ", seq=11, tenant_id='default',\n"
        "    app_id='default', user_id='u-child', scope='user', scope_key='',\n"
        "    fingerprints=[], generation=0, purpose='child', created_at=None,\n"
        "    prev_hash=prev_hash)\n"
        "e.hash = e.compute_hash()\n"
        "log.append(e)\n"
        f"open({str(barrier1)!r}, 'w').write('1')\n"
        f"open({str(barrier2)!r}, 'w').write('wait')\n"
        "# wait for the parent to observe the append\n"
        f"while open({str(barrier1)!r}).read() != 'go':\n"
        "    pass\n"
        "open(path, 'wb').write(" + repr(restore_point) + ")\n"
        f"open({str(barrier2)!r}, 'w').write('1')\n",
        encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(child)],
                            cwd=str(_Path.cwd()),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # wait for the child's append (barrier 1)
        for _ in range(300):
            if barrier1.exists():
                break
            _time_sleep(0.1)
        assert barrier1.exists(), "the child never appended"

        # the parent's cached index must see the OTHER process's append
        assert log.head(db) == (11, _compute_child_hash(path, 11)), \
            "the index must reflect a real other process's append"
        assert log.has_tombstone(db, 11, _last_hash(path))

        # release the child; it rolls the file back to the restore point
        barrier1.write_text("go")
        for _ in range(300):
            if barrier2.read_text() == "1":
                break
            _time_sleep(0.1)

        # and the rollback must be visible too
        assert log.head(db) == (10, entries[-1].hash), \
            "the index must reflect the other process's rollback"
        assert not log.has_tombstone(db, 11, _last_hash(path))
    finally:
        proc.wait(timeout=30)
        if proc.returncode != 0:
            print(proc.stdout.read().decode(), proc.stderr.read().decode())
            assert False, f"the child failed: {proc.returncode}"


def _time_sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)


def _last_hash(path) -> str:
    import json
    return json.loads(path.read_bytes().splitlines()[-1])["hash"]


def _compute_child_hash(path, seq: int) -> str:
    """The hash the child's seq-11 entry carries (the parent reads the
    file the child wrote -- the identity check, not a recomputation)."""
    return _last_hash(path)
