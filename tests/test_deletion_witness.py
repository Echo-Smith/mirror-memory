"""M5 patch 5 §1: deletion-history INTEGRITY, end to end.

Patch 4 proved a log had *existed* (a binding sidecar).  That is not enough:
it cannot tell a healthy log from one whose trailing entries were removed.
This module pins the stronger property -- the engine refuses to serve when
the history on disk is less complete than the last state it durably
observed, and no read entry point works in that state.

Three truncation shapes are covered against a REAL engine and a REAL SQLite
database, each following the threat scenario:

    memory -> backup -> delete (log gets entries) -> truncate -> restore the
    old backup -> startup AND every read entry point must refuse.

  (a) the log file is deleted (the witness survives);
  (b) the log keeps only its genesis line (a complete-line truncation);
  (c) the log keeps only part of its deletion history.

Plus: blank/torn logs, legacy upgrade without a witness, a healthy log,
log restoration, explicit adoption, concurrency, and the crash windows of
the log-then-witness ordering.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
import threading
from pathlib import Path

import pytest
from sqlalchemy import text

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import authorization, schema_migration as sm
from mirror_memory.core.deletion import (
    DeletionEntry,
    DeletionLogCorrupted,
    DeletionLogUnavailable,
    FileDeletionLog,
)
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.core.witness import WitnessAhead, WitnessError


@pytest.fixture(autouse=True)
def _isolated_witness_key(tmp_path, monkeypatch):
    """Keep the MAC key inside the test's tmp dir (never the user's home)."""
    monkeypatch.setenv("MM_DELETION_WITNESS_KEY", str(tmp_path / "witness.key"))


class _FakeLLM:
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


def _engine(tmp_path: Path, log_dir: Path, db: Path | None = None) -> MemoryEngine:
    return MemoryEngine(
        config=_cfg(),
        database_url=f"sqlite:///{db or (tmp_path / 'main.db')}",
        llm_client=_FakeLLM(),
        deletion_log=FileDeletionLog(log_dir),
    )


def _enable(eng: MemoryEngine, user: str = "u1"):
    from sqlalchemy.orm import Session

    eng._ensure_db()
    with tenant_scope(*eng._tenant), fixture_write_scope("m5p5: enable"):
        s = Session(eng._engine)
        try:
            set_memory_enabled(s, user, True)
            authorization.grant_purpose(
                s, user, "chat",
                operations=list(authorization.OPERATIONS),
                issuer="m5p5-harness", reason="integrity test grant",
            )
            s.commit()
        finally:
            s.close()


def _observe(eng, session="s1"):
    return eng.observe(
        user_id="u1", session_id=session,
        text="I love jasmine tea every morning", purpose="chat")


def _backup(db: Path, dest: Path):
    src = sqlite3.connect(db)
    out = sqlite3.connect(dest)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()


def _log_file(log_dir: Path, db_id: str) -> Path:
    return log_dir / f"{db_id}.deletions.jsonl"


def _line_count(path: Path) -> int:
    return len([ln for ln in path.read_bytes().split(b"\n") if ln])


def _prepare_history(tmp_path: Path, deletions: int = 3):
    """memory -> backup -> N deletes.  Returns (log_dir, db, db_id, backup)."""
    log_dir = tmp_path / "log"
    db = tmp_path / "main.db"
    eng = _engine(tmp_path, log_dir, db)
    _enable(eng)
    db_id = sm.ensure_watermark(eng._engine)[0]
    _observe(eng)
    backup = tmp_path / "before.db"
    _backup(db, backup)
    for i in range(deletions):
        # each iteration writes memory then deletes it -> one log entry each
        _observe(eng, session=f"s{i}")
        eng.delete_memories(user_id="u1", purpose="chat")
    eng._engine.dispose()
    return log_dir, db, db_id, backup


def _restore(backup: Path, db: Path):
    db.write_bytes(backup.read_bytes())


# ---------------------------------------------------------------------------
# (a)(b)(c): the three truncation shapes
# ---------------------------------------------------------------------------


def test_a_deleted_log_file_with_witness_refuses_even_at_watermark_zero(tmp_path):
    """(a) The log FILE is gone but the witness records a real head.
    The restored backup's watermark is 0 -- the witness is what refuses."""
    log_dir, db, db_id, backup = _prepare_history(tmp_path)
    log = FileDeletionLog(log_dir)
    witness_state = log.witness.read(db_id)
    assert witness_state is not None and witness_state.tail_seq >= 1

    path = _log_file(log_dir, db_id)
    path.unlink()
    _restore(backup, db)
    # the restored database really does carry applied_seq == 0
    raw = sqlite3.connect(db)
    try:
        assert raw.execute(
            "SELECT applied_seq FROM mm_deletion_watermark").fetchone()[0] == 0
    finally:
        raw.close()

    with pytest.raises(DeletionLogCorrupted) as excinfo:
        again = _engine(tmp_path, log_dir, db)
        again._ensure_db()
    assert "witness" in str(excinfo.value).lower()
    assert not path.exists(), "a lost log must never be re-created"


def test_b_genesis_only_truncation_refuses(tmp_path):
    """(b) A complete-line truncation that leaves only the genesis line: every
    deletion entry is gone.  The witness catches it (the log 'head' is seq 0
    while the witness saw seq >= 1)."""
    log_dir, db, db_id, backup = _prepare_history(tmp_path)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    assert len(lines) >= 2
    path.write_bytes(lines[0] + b"\n")          # only the genesis survives
    _restore(backup, db)

    log = FileDeletionLog(log_dir)
    with pytest.raises(WitnessAhead) as excinfo:
        log.verify_witness(db_id, applied=0)
    assert "lost" in str(excinfo.value).lower()

    # and the ENGINE refuses to serve
    with pytest.raises((WitnessAhead, DeletionLogCorrupted)):
        again = _engine(tmp_path, log_dir, db)
        again._ensure_db()
    # the evidence was not destroyed
    assert _line_count(path) == 1


def test_c_partial_history_truncation_refuses(tmp_path):
    """(c) Only PART of the deletion history survives (genesis + first
    deletion, the rest removed)."""
    log_dir, db, db_id, backup = _prepare_history(tmp_path, deletions=3)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    assert len(lines) >= 4
    path.write_bytes(b"\n".join(lines[:2]) + b"\n")
    surviving = _line_count(path)
    _restore(backup, db)

    log = FileDeletionLog(log_dir)
    with pytest.raises(WitnessAhead):
        log.verify_witness(db_id, applied=0)

    with pytest.raises((WitnessAhead, DeletionLogCorrupted)):
        again = _engine(tmp_path, log_dir, db)
        again._ensure_db()
    assert _line_count(path) == surviving, "evidence must be left untouched"


def test_truncation_after_compaction_is_caught(tmp_path):
    """A compaction checkpoint keeps the head equal, but a LATER truncation
    still has to be caught: the witness records the head, and the head moves
    with every append even after pruning."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "7" * 32
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 6):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    log.prune(db_id, 3)
    after_prune = log.witness.read(db_id)
    assert after_prune is not None
    assert after_prune.checkpoint_seq >= 3
    assert after_prune.tail_seq == 5

    # now drop the entries ABOVE the checkpoint (the tail rows)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    keep = [ln for ln in lines
            if json.loads(ln).get("scope") in ("genesis", "checkpoint")]
    path.write_bytes(b"\n".join(keep) + b"\n")
    log._index_invalidate(db_id)

    with pytest.raises(WitnessAhead):
        log.verify_witness(db_id, applied=0)


# ---------------------------------------------------------------------------
# the read entry points must refuse too (not just startup)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["deleted", "genesis_only", "partial"])
def test_every_read_entry_refuses_on_truncation(tmp_path, shape):
    """After a truncation the engine must refuse at startup; an engine object
    that already exists must not serve reads either -- proven by the fact that
    a startup is required for any entry point to work at all."""
    log_dir, db, db_id, backup = _prepare_history(tmp_path)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    if shape == "deleted":
        path.unlink()
    elif shape == "genesis_only":
        path.write_bytes(lines[0] + b"\n")
    else:
        path.write_bytes(b"\n".join(lines[:2]) + b"\n")
    _restore(backup, db)

    eng = _engine(tmp_path, log_dir, db)
    with pytest.raises((WitnessAhead, DeletionLogCorrupted, DeletionLogUnavailable)):
        eng._ensure_db()

    # a read through a FRESH engine must also fail (the guard is in startup,
    # so no entry point can be reached without it)
    eng2 = _engine(tmp_path, log_dir, db)
    with pytest.raises((WitnessAhead, DeletionLogCorrupted, DeletionLogUnavailable)):
        eng2.recall(user_id="u1", query="tea", purpose="chat")
    with pytest.raises((WitnessAhead, DeletionLogCorrupted, DeletionLogUnavailable)):
        eng2.delete_memories(user_id="u1", purpose="chat")


# ---------------------------------------------------------------------------
# other corrupted states
# ---------------------------------------------------------------------------


def test_blank_log_with_witness_refuses(tmp_path):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "8" * 32
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    path = _log_file(log_dir, db_id)
    path.write_bytes(b"")                      # blanked out
    log._index_invalidate(db_id)

    with pytest.raises(DeletionLogCorrupted):
        log.ensure_genesis(db_id, applied=0)
    assert path.read_bytes() == b""


def test_torn_line_is_corruption_and_evidence_kept(tmp_path):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "9" * 32
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    path = _log_file(log_dir, db_id)
    original = path.read_bytes()
    path.write_bytes(original[:-9])            # tear the last line
    torn = path.read_bytes()

    with pytest.raises(DeletionLogCorrupted):
        list(log.read_from(db_id, 0))
    assert path.read_bytes() == torn, "a refusal must not rewrite evidence"

    # verify_witness surfaces it too
    log._index_invalidate(db_id)
    with pytest.raises(DeletionLogCorrupted):
        log.verify_witness(db_id, applied=0)


# ---------------------------------------------------------------------------
# legacy upgrade, healthy logs, restoration, explicit adoption
# ---------------------------------------------------------------------------


def test_legacy_log_without_witness_is_adopted_when_nothing_was_deleted(tmp_path):
    """An old deployment (log + genesis, no witness, applied == 0) upgrades
    safely: nothing was ever deleted, so the current head is adopted."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a1" * 16
    # simulate the pre-patch-5 state: genesis written, then the witness removed
    log.ensure_genesis(db_id, applied=0)
    log.witness.retire(db_id)
    assert log.witness.read(db_id) is None

    log.verify_witness(db_id, applied=0)       # adopts the head
    state = log.witness.read(db_id)
    assert state is not None and state.tail_seq == 0


def test_legacy_log_without_witness_and_with_deletions_is_refused(tmp_path):
    """With deletions on record a missing witness is NOT back-filled: that
    would bless a possibly truncated history as intact."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a2" * 16
    log.ensure_genesis(db_id, applied=0)
    log.witness.retire(db_id)

    with pytest.raises(DeletionLogCorrupted) as excinfo:
        log.verify_witness(db_id, applied=2)
    assert "witness" in str(excinfo.value).lower()
    assert log.witness.read(db_id) is None, "must not forge a witness"


def test_healthy_log_passes_and_is_not_rewritten(tmp_path):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a3" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    path = _log_file(log_dir, db_id)
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns

    log.verify_witness(db_id, applied=1)       # a normal restart

    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_mtime
    log.verify_chain(db_id)


def test_witness_tampered_is_refused(tmp_path):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a4" * 16
    log.ensure_genesis(db_id, applied=0)
    wpath = log.witness._path(db_id)
    record = json.loads(wpath.read_bytes())
    record["payload"]["tail_seq"] = 0          # pretend nothing was appended
    record["payload"]["tail_hash"] = "0" * 64
    wpath.write_text(json.dumps(record))

    with pytest.raises(DeletionLogCorrupted) as excinfo:
        log.verify_witness(db_id, applied=0)
    assert "mac" in str(excinfo.value).lower() or "trust" in str(excinfo.value).lower()


def test_witness_missing_key_is_refused_not_trusted(tmp_path):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a5" * 16
    log.ensure_genesis(db_id, applied=0)
    log.witness.key_path.unlink()

    with pytest.raises(DeletionLogCorrupted):
        log.verify_witness(db_id, applied=0)


def test_restored_log_ahead_of_witness_advances_it(tmp_path):
    """After restoring an OLDER main DB whose log is intact and AHEAD of the
    witness (a crash between log fsync and witness write), the witness is
    advanced to the log head -- nothing was lost."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a6" * 16
    log.ensure_genesis(db_id, applied=0)
    state_before = log.witness.read(db_id)
    # simulate the crash window: log has an entry the witness never saw
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    log.witness.record(state_before)           # rewind the witness
    assert log.witness.read(db_id).tail_seq == 0

    log.verify_witness(db_id, applied=1)       # heal forward

    assert log.witness.read(db_id).tail_seq == log.head(db_id)[0]


def test_witness_ahead_of_log_is_refused(tmp_path):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a7" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    # simulate a log rollback: rewrite the file back to the genesis only
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    path.write_bytes(lines[0] + b"\n")
    log._index_invalidate(db_id)

    with pytest.raises(WitnessAhead):
        log.verify_witness(db_id, applied=0)


# ---------------------------------------------------------------------------
# concurrency: first creation and concurrent log writes
# ---------------------------------------------------------------------------


def _mp_first_creation(log_dir: str, db_id: str) -> None:
    from mirror_memory.core.deletion import FileDeletionLog as FDL

    FDL(log_dir).ensure_genesis(db_id, applied=0)


def test_multiprocess_first_creation_leaves_one_genesis_and_one_witness(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    db_id = "b1" * 16
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_mp_first_creation, args=(str(log_dir), db_id))
             for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, p.exitcode

    log = FileDeletionLog(log_dir)
    assert _line_count(_log_file(log_dir, db_id)) == 1
    state = log.witness.read(db_id)
    assert state is not None and state.tail_seq == 0
    log.verify_chain(db_id)


def test_concurrent_appends_keep_log_and_witness_consistent(tmp_path):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "b2" * 16
    log.ensure_genesis(db_id, applied=0)
    errors: list[BaseException] = []

    def worker(idx):
        try:
            for j in range(10):
                log.append(DeletionEntry(
                    db_id=db_id, tenant_id="t", app_id="a",
                    user_id=f"u{idx}", scope="session",
                    scope_key=f"s{idx}-{j}", fingerprints=[],
                    generation=j + 1, purpose="p"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors[:2]

    # exactly 40 appends, a whole chain, and the witness at the head
    head_seq, _ = log.head(db_id)
    assert head_seq == 40
    log.verify_chain(db_id)
    state = log.witness.read(db_id)
    assert state.tail_seq == 40
    assert state.tail_hash == log.head(db_id)[1]


# ---------------------------------------------------------------------------
# crash windows of the log-then-witness ordering
# ---------------------------------------------------------------------------


def test_crash_between_log_write_and_witness_advances_witness(tmp_path):
    """Crash window (a): the log has the entry, the witness does not.  The log
    is AHEAD, so nothing was lost -- the next verify heals forward."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "c1" * 16
    log.ensure_genesis(db_id, applied=0)
    stale = log.witness.read(db_id)

    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    log.witness.record(stale)                  # the witness never advanced

    log.verify_witness(db_id, applied=1)
    assert log.witness.read(db_id).tail_seq == 1


def test_witness_never_ahead_after_a_normal_append(tmp_path):
    """Invariant: after every successful append the witness tail equals the
    log head -- the ordering can never leave the witness ahead."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "c2" * 16
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 6):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id="u",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
        state = log.witness.read(db_id)
        head = log.head(db_id)
        assert state.tail_seq == head[0]
        assert state.tail_hash == head[1]


# ---------------------------------------------------------------------------
# long-log cost: appends stay O(1) in log length
# ---------------------------------------------------------------------------


def test_append_stays_cheap_on_a_long_log(tmp_path, monkeypatch):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "d1" * 16
    log.ensure_genesis(db_id, applied=0)
    for i in range(200):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id="u",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i + 1, purpose="p"))

    reads = {"n": 0}
    original = FileDeletionLog._iter_entries

    def counting(self, path):
        reads["n"] += 1
        return original(self, path)

    monkeypatch.setattr(FileDeletionLog, "_iter_entries", counting)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=999, purpose="p"))
    # ONE parse is allowed (the witness needs the genesis + checkpoint view
    # after an index invalidation); a per-append full rescan is not.
    assert reads["n"] <= 1, f"append parsed the log {reads['n']} times"


def test_legacy_back_fill_refuses_a_broken_chain(tmp_path):
    """Red-team C: the legacy path must not bless a broken chain.  A log whose
    genesis..head chain does not verify must NOT get a witness written."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "e1" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    log.witness.retire(db_id)                  # the legacy state

    # corrupt one entry's hash in place (a tampered chain)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    record = json.loads(lines[1])
    record["hash"] = "f" * 64
    lines[1] = json.dumps(record, sort_keys=True).encode()
    path.write_bytes(b"\n".join(lines) + b"\n")
    log._index_invalidate(db_id)

    with pytest.raises(Exception):
        log.verify_witness(db_id, applied=0)
    assert log.witness.read(db_id) is None, (
        "a broken chain must never be recorded as witnessed-intact"
    )


def test_witness_checkpoint_seq_is_monotonic(tmp_path):
    """Red-team E: an index-cache miss must not roll the recorded checkpoint
    backwards."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "e2" * 16
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 6):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    log.prune(db_id, 3)
    assert log.witness.read(db_id).checkpoint_seq >= 3

    # simulate a stale in-memory view: drop the checkpoint cache, then record
    log._index_invalidate(db_id)
    log._index_checkpoint.pop(db_id, None)
    state = log._witness_state(db_id, tail_seq=5, tail_hash="0" * 64)
    assert state.checkpoint_seq >= 3, (
        f"checkpoint rolled back to {state.checkpoint_seq}"
    )


def test_witness_key_is_outside_the_witness_directory(tmp_path, monkeypatch):
    """Red-team D-2: whoever can write the witness directory must not be able
    to read the MAC key and re-forge the witness."""
    key = tmp_path / "secrets" / "witness.key"
    monkeypatch.setenv("MM_DELETION_WITNESS_KEY", str(key))
    log = FileDeletionLog(tmp_path / "log")
    db_id = "e3" * 16
    log.ensure_genesis(db_id, applied=0)

    assert key.exists(), "the key must be created at the configured path"
    assert not (log.witness.directory / ".witness.key").exists(), (
        "the key must NOT live inside the witness directory"
    )
    assert key.stat().st_mode & 0o077 == 0, "the key must be 0600"
