"""M5 patch 4 §1: deletion-log first creation vs. corruption (end-to-end).

The security property under test: **only a provable FIRST creation may
initialize a log**.  Once a database life has deletions, a missing, empty or
truncated log means history was LOST -- the engine must refuse to start
(fail-closed) and must never mint a fresh genesis, because a fresh genesis
would silently erase the record that makes a restore safe (deleted content
could come back).

Covered here, all against a REAL SQLite database and a REAL engine:
  * the full restore scenario: memory -> backup -> delete (log gets entries)
    -> truncate the log -> restore the old backup -> start/read MUST refuse
    and the deleted memory must not reappear;
  * watermark > 0 with the log missing / empty / partially written;
  * blank-only and partial-line logs;
  * two engines and multiple PROCESSES racing the first creation;
  * a creation interrupted between create and write;
  * a healthy log is never rewritten by a normal startup.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import (
    DeletionEntry,
    DeletionLogCorrupted,
    DeletionLogUnavailable,
    FileDeletionLog,
)
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope


class _FakeLLM:
    """A deterministic provider that yields ONE real pattern per turn, so
    observe() actually writes a belief (needed to have something to delete)."""

    def generate(self, **_kw):
        return (
            '{"patterns": [{"claim_text": "I love jasmine tea every morning", '
            '"predicate": "likes", "object_text": "jasmine tea", '
            '"polarity": "positive", "confidence": 0.9, '
            '"support_quote": "I love jasmine tea every morning"}]}'
        )


def _cfg():
    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    return cfg


def _new_engine(tmp_path: Path, log_dir: Path, db: Path | None = None):
    url = f"sqlite:///{db or (tmp_path / 'main.db')}"
    return MemoryEngine(config=_cfg(), database_url=url, llm_client=_FakeLLM(),
                        deletion_log=FileDeletionLog(log_dir))


def _enable(eng: MemoryEngine, user: str = "u1"):
    """Enable memory AND grant the purpose, under a fixture write scope."""
    from sqlalchemy.orm import Session

    from mirror_memory.core import authorization

    eng._ensure_db()
    with tenant_scope(*eng._tenant), fixture_write_scope("m5p4: enable"):
        s = Session(eng._engine)
        try:
            set_memory_enabled(s, user, True)
            authorization.grant_purpose(
                s, user, "chat",
                operations=list(authorization.OPERATIONS),
                issuer="m5p4-harness", reason="safety test grant",
            )
            s.commit()
        finally:
            s.close()


def _observe(eng, text_="sample memory", session="s1"):
    return eng.observe(user_id="u1", session_id=session, text=text_, purpose="chat")


def _delete(eng):
    """Delete the user's memories: this is what writes log entries and
    advances the watermark."""
    return eng.delete_memories(user_id="u1", purpose="chat")


def _backup(db: Path, dest: Path):
    src = sqlite3.connect(db)
    out = sqlite3.connect(dest)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()


def _log_path(log_dir: Path, db_id: str) -> Path:
    return log_dir / f"{db_id}.deletions.jsonl"


# ---------------------------------------------------------------------------
# 1. the full restore scenario (the headline security case)
# ---------------------------------------------------------------------------


def test_truncated_log_blocks_restore_and_memory_does_not_resurrect(tmp_path):
    """memory -> backup -> delete -> log gets entries -> truncate log ->
    restore old backup -> start MUST refuse; the deleted memory stays gone."""
    log_dir = tmp_path / "log"
    db = tmp_path / "main.db"
    eng = _new_engine(tmp_path, log_dir, db)
    _enable(eng)
    db_id = sm.ensure_watermark(eng._engine)[0]

    # 1. a real memory that will be deleted
    n = _observe(eng, "I love jasmine tea every morning")
    assert n >= 1
    eng._ensure_db()
    with tenant_scope(*eng._tenant):
        recalled = eng.recall(user_id="u1", query="tea", purpose="chat")
        assert recalled and "jasmine tea" in str(recalled), \
            "the memory must be readable before the deletion"

    # 2. backup taken BEFORE the deletion
    backup = tmp_path / "before.db"
    _backup(db, backup)

    # 3. delete: the log must gain real entries
    _delete(eng)
    log = FileDeletionLog(log_dir)
    tail, _h = log.head(db_id)
    assert tail >= 1, "the deletion must be recorded in the log"
    entries_before = list(log.read_from(db_id, 0))
    path = _log_path(log_dir, db_id)
    full_bytes = path.read_bytes()
    assert len(full_bytes) > 0

    # 4. truncate the log (simulating a lost/corrupted log)
    path.write_bytes(b"")

    # 5. restore the OLD backup: the watermark comes back at the pre-deletion
    #    value, i.e. it may be 0 -- this is exactly why "applied == 0" cannot
    #    be used alone to prove a first creation.
    eng._engine.dispose()
    db.write_bytes(backup.read_bytes())

    # 6. starting the engine over the restored DB with a truncated log must
    #    FAIL CLOSED -- no fresh genesis, no replay, no resurrected memory
    with pytest.raises(DeletionLogCorrupted) as excinfo:
        eng2 = _new_engine(tmp_path, log_dir, db)
        eng2._ensure_db()
    msg = str(excinfo.value).lower()
    assert "empty" in msg or "truncat" in msg or "lost" in msg

    # 7. the log was NOT re-initialized behind the operator's back
    assert path.read_bytes() == b"", "a corrupted log must never be silently rewritten"

    # 8. and the deleted memory must NOT come back through a raw read either:
    #    the refusal happened before any entry served.  Prove the backup does
    #    contain the pre-deletion belief (so the risk is real, not vacuous).
    raw = sqlite3.connect(db)
    try:
        rows = raw.execute(
            "SELECT claim_text FROM mm_beliefs WHERE user_id='u1'").fetchall()
    finally:
        raw.close()
    assert any("jasmine tea" in (r[0] or "") for r in rows), (
        "control: the restored backup really does contain the deleted memory"
    )


def test_missing_log_with_applied_watermark_fails_closed(tmp_path):
    """Watermark > 0 and the log file is GONE: refuse, never mint a new log."""
    log_dir = tmp_path / "log"
    db = tmp_path / "main.db"
    eng = _new_engine(tmp_path, log_dir, db)
    _enable(eng)
    db_id = sm.ensure_watermark(eng._engine)[0]
    _observe(eng)
    _delete(eng)
    _, applied = sm.ensure_watermark(eng._engine)
    assert applied >= 1
    eng._engine.dispose()

    _log_path(log_dir, db_id).unlink()

    log = FileDeletionLog(log_dir)
    with pytest.raises(DeletionLogCorrupted) as excinfo:
        log.ensure_genesis(db_id, applied=applied)
    assert "lost" in str(excinfo.value).lower()
    assert not _log_path(log_dir, db_id).exists(), "no new log may be minted"


def test_engine_startup_refuses_when_log_lost_after_deletions(tmp_path):
    """Through the REAL engine startup path (not just the log API)."""
    log_dir = tmp_path / "log"
    db = tmp_path / "main.db"
    eng = _new_engine(tmp_path, log_dir, db)
    _enable(eng)
    db_id = sm.ensure_watermark(eng._engine)[0]
    _observe(eng)
    _delete(eng)
    eng._engine.dispose()
    _log_path(log_dir, db_id).unlink()

    with pytest.raises(DeletionLogUnavailable):
        eng2 = _new_engine(tmp_path, log_dir, db)
        eng2._ensure_db()


def test_partial_lines_log_is_corruption(tmp_path):
    """A log truncated mid-line (partial last row) must be reported as
    corruption, not silently ignored or re-initialized."""
    log = FileDeletionLog(tmp_path / "log")
    db_id = "b" * 32
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(db_id=db_id, tenant_id="t", app_id="a",
                             user_id="u", scope="user", scope_key="",
                             fingerprints=[], generation=1, purpose="p"))
    path = _log_path(tmp_path / "log", db_id)
    data = path.read_bytes()
    path.write_bytes(data[:-7])          # tear the last line in half

    with pytest.raises(DeletionLogCorrupted):
        list(log.read_from(db_id, 0))


def test_blank_only_log_with_applied_is_corruption(tmp_path):
    """A whitespace-only file with deletions on record is not a new log."""
    log = FileDeletionLog(tmp_path / "log")
    db_id = "c" * 32
    path = _log_path(tmp_path / "log", db_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n\n   \n")

    with pytest.raises(DeletionLogCorrupted):
        log.ensure_genesis(db_id, applied=5)
    assert path.read_bytes() == b"\n\n   \n", "content must be left untouched"


def test_content_without_genesis_is_corruption(tmp_path):
    log = FileDeletionLog(tmp_path / "log")
    db_id = "d" * 32
    path = _log_path(tmp_path / "log", db_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b'{"seq": 1, "db_id": "' + db_id.encode() + b'", "tenant_id": "t", '
        b'"app_id": "a", "user_id": "u", "scope": "user", "scope_key": "", '
        b'"fingerprints": [], "generation": 1, "purpose": "p", '
        b'"created_at": null, "prev_hash": "' + b"0" * 64 + b'", '
        b'"hash": "' + b"0" * 64 + b'", "global_key": "", "origin": "", '
        b'"origin_seq": 0, "origin_hash": ""}\n')

    # even with applied == 0 this is not a provable first creation
    with pytest.raises(DeletionLogCorrupted):
        log.ensure_genesis(db_id, applied=0)


# ---------------------------------------------------------------------------
# 2. legitimate first creation (and crash recovery) still works
# ---------------------------------------------------------------------------


def test_first_creation_of_a_brand_new_life_succeeds(tmp_path):
    log = FileDeletionLog(tmp_path / "log")
    db_id = "e" * 32
    log.ensure_genesis(db_id, applied=0)
    assert log.has_genesis(db_id)
    path = _log_path(tmp_path / "log", db_id)
    assert path.read_bytes().count(b"\n") == 1


def test_interrupted_creation_is_recovered_when_nothing_was_deleted(tmp_path):
    """Crash between O_EXCL create and the genesis write leaves an EMPTY file.
    With applied == 0 nothing was ever deleted, so finishing the initialization
    is safe -- and it must produce exactly one genesis."""
    log = FileDeletionLog(tmp_path / "log")
    db_id = "f" * 32
    path = _log_path(tmp_path / "log", db_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # the interrupted state: the file was created, the genesis write had not
    # happened yet, and the crash predates the binding marker (which is the
    # ONLY way an empty file can still be a provable first creation)
    path.write_bytes(b"")
    assert not log.is_bound(db_id)

    log.ensure_genesis(db_id, applied=0)

    assert log.has_genesis(db_id)
    assert path.read_bytes().count(b"\n") == 1
    # idempotent: another startup does not add a second genesis
    log.ensure_genesis(db_id, applied=0)
    assert path.read_bytes().count(b"\n") == 1


def test_healthy_log_is_never_rewritten_by_startup(tmp_path):
    log = FileDeletionLog(tmp_path / "log")
    db_id = "1" * 32
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(db_id=db_id, tenant_id="t", app_id="a",
                             user_id="u", scope="user", scope_key="",
                             fingerprints=[], generation=1, purpose="p"))
    path = _log_path(tmp_path / "log", db_id)
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    log.ensure_genesis(db_id, applied=1)     # a normal restart

    assert path.read_bytes() == before, "a healthy log must not be rewritten"
    assert path.stat().st_mtime_ns == before_mtime
    log.verify_chain(db_id)                  # chain still whole


def test_two_engines_racing_first_creation(tmp_path):
    """Two engines on the same fresh database life end with ONE genesis."""
    log_dir = tmp_path / "log"
    db = tmp_path / "main.db"
    a = _new_engine(tmp_path, log_dir, db)
    b = _new_engine(tmp_path, log_dir, db)
    _enable(a)
    b._ensure_db()
    db_id = sm.ensure_watermark(a._engine)[0]
    path = _log_path(log_dir, db_id)
    assert path.read_bytes().count(b"\n") == 1, "exactly one genesis"


def _mp_create(log_dir: str, db_id: str) -> None:
    from mirror_memory.core.deletion import FileDeletionLog as FDL

    FDL(log_dir).ensure_genesis(db_id, applied=0)


def test_multiprocess_racing_first_creation(tmp_path):
    """Several PROCESSES race the first creation: exactly one genesis."""
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    db_id = "2" * 32
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_mp_create, args=(str(log_dir), db_id))
             for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"creation process failed: {p.exitcode}"

    path = _log_path(log_dir, db_id)
    assert path.read_bytes().count(b"\n") == 1
    FileDeletionLog(log_dir).verify_chain(db_id)


# ---------------------------------------------------------------------------
# 3. appends are fail-closed too
# ---------------------------------------------------------------------------


def test_append_without_genesis_fails_closed(tmp_path):
    log = FileDeletionLog(tmp_path / "log")
    db_id = "3" * 32
    with pytest.raises(DeletionLogUnavailable):
        log.append(DeletionEntry(db_id=db_id, tenant_id="t", app_id="a",
                                 user_id="u", scope="user", scope_key="",
                                 fingerprints=[], generation=1, purpose="p"))
    assert not _log_path(tmp_path / "log", db_id).exists(), (
        "a failed append must not leave a log behind"
    )


# ---------------------------------------------------------------------------
# 4. no full-file read on the append path (long-log scaling)
# ---------------------------------------------------------------------------


def test_append_does_not_rescan_the_whole_log(tmp_path, monkeypatch):
    """Appending to a long log must stay O(1) in file size: no per-append
    pass over the whole file (the index is maintained under the lock)."""
    log = FileDeletionLog(tmp_path / "log")
    db_id = "4" * 32
    log.ensure_genesis(db_id, applied=0)
    for i in range(50):
        log.append(DeletionEntry(db_id=db_id, tenant_id="t", app_id="a",
                                 user_id="u", scope="user", scope_key="",
                                 fingerprints=[], generation=i + 1,
                                 purpose="p"))

    reads = {"n": 0}
    original = FileDeletionLog._iter_entries

    def counting(self, path):
        reads["n"] += 1
        return original(self, path)

    monkeypatch.setattr(FileDeletionLog, "_iter_entries", counting)
    log.append(DeletionEntry(db_id=db_id, tenant_id="t", app_id="a",
                             user_id="u", scope="user", scope_key="",
                             fingerprints=[], generation=99, purpose="p"))
    assert reads["n"] == 0, (
        f"append parsed the log {reads['n']} time(s); appends must not rescan"
    )
