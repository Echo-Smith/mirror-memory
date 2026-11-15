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


def test_credentialed_log_without_witness_always_refused(
        tmp_path):
    """M5 patch 7 §1: a log with a creation credential but no witness is
    refused in EVERY shape.  Patch 6 exempted the "log == exactly the
    registered genesis, applied == 0" case as a provable first-creation
    interruption -- but that shape is indistinguishable from "deletions
    happened, then the witness was erased and the log truncated to the
    genesis" (brief repro 1).  Fail closed in both; the operator reconciles."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a1" * 16
    log.ensure_genesis(db_id, applied=0)
    log.witness.retire(db_id)

    # shape 1: log holds only the genesis -> STILL refused (indistinguishable)
    with pytest.raises(DeletionLogCorrupted) as excinfo:
        log.verify_witness(db_id, applied=0)
    assert "indistinguishable" in str(excinfo.value).lower()
    assert log.witness.read(db_id) is None, "no witness may be forged"

    # shape 2: the same erasure AFTER a deletion, now against append() too
    # (M5 patch 8: append no longer skips its checks when the witness is
    # absent -- before, a removed witness let the log be extended and the
    # witness re-minted from the forged head)
    with pytest.raises(DeletionLogCorrupted):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id="u",
            scope="user", scope_key="", fingerprints=[], generation=1,
            purpose="p"))
    assert log.witness.read(db_id) is None, "no witness may be forged"


def test_legacy_log_without_witness_or_credential_at_zero_adopts(tmp_path):
    """A log that this store NEVER registered (no credential, no witness) and
    an empty watermark is the one provable-brand-new shape: adopt it."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a1" * 16
    log.ensure_genesis(db_id, applied=0)
    log.witness.retire(db_id)
    # remove the creation credential too: the pre-patch-6 state has neither
    (log.directory / ".creation-registry.jsonl").unlink()

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
    """A log AHEAD of its witness (a crash between log fsync and witness
    write) heals forward: the witness is advanced to the log head."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "a6" * 16
    log.ensure_genesis(db_id, applied=0)
    state_before = log.witness.read(db_id)
    # keep the registry out of the way: the credential was recorded at
    # creation with the GENESIS hash, which has not changed -- only the
    # witness was rewound, which is the crash-window shape being tested
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


# ---------------------------------------------------------------------------
# M5 patch 6 §1: the creation registry — the answer to "was this life EVER
# initialised?" that neither applied==0, an empty file, nor a self-consistent
# chain can give.
# ---------------------------------------------------------------------------


def test_forged_selfconsistent_log_is_refused_by_credential(tmp_path):
    """The headline attack: erase the witness AND replace the log with a fresh
    self-consistent one (new genesis + valid chain).  applied == 0.  The
    credential's genesis hash exposes the replacement."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "f1" * 16
    log.ensure_genesis(db_id, applied=0)
    original_genesis = log.witness.read(db_id).genesis_hash
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))

    # the attack: forge a brand-new, internally consistent history.  The
    # forge itself is refused at ensure_genesis (the credential proves the
    # life already existed); the deepest check -- verify_witness on the
    # planted file -- must ALSO refuse via the genesis-hash mismatch.
    # Part 1: the guarded creation refuses the forge outright (the registry
    # proves the life already existed).
    path = _log_file(log_dir, db_id)
    saved_log = path.read_bytes()
    saved_witness = log.witness.read(db_id)
    path.unlink()
    with pytest.raises(DeletionLogCorrupted):
        FileDeletionLog(log_dir).ensure_genesis(db_id, applied=0)
    # restore the state so Part 2 exercises the verify layer
    path.write_bytes(saved_log)
    log.witness.record(saved_witness)
    log._index_invalidate(db_id)

    # Part 2: even when a forged file IS planted while the witness is intact,
    # verify_witness rejects it on the genesis-hash mismatch (the witness at
    # the original genesis can never match a foreign chain).
    forged = FileDeletionLog(tmp_path / "staging")
    forged.ensure_genesis(db_id, applied=0)
    forged.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    import shutil as _sh
    _sh.copy2(_log_file(tmp_path / "staging", db_id), path)
    log._index_invalidate(db_id)

    # the planted foreign chain is refused -- by the divergence check or the
    # genesis-mismatch check, whichever fires first (both are correct).
    with pytest.raises(DeletionLogCorrupted):
        log.verify_witness(db_id, applied=0)
    # and the ORIGINAL genesis hash was preserved by the registry
    cred = FileDeletionLog(log_dir).registry.find(db_id)
    assert cred is not None
    assert cred.genesis_hash == original_genesis


def test_witness_and_log_erased_but_credential_kept_refuses(tmp_path):
    """Both witness and log erased; only the credential survives.  Whether
    entries were lost cannot be proven -> refuse, do not re-initialise."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "f2" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))

    log.witness.retire(db_id)
    _log_file(log_dir, db_id).unlink()

    with pytest.raises(DeletionLogCorrupted) as excinfo:
        log.ensure_genesis(db_id, applied=0)
    assert "credential" in str(excinfo.value).lower()
    assert not _log_file(log_dir, db_id).exists(), "no new log may be minted"


def test_registry_tampering_is_refused(tmp_path):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "f3" * 16
    log.ensure_genesis(db_id, applied=0)

    reg_path = log.directory / ".creation-registry.jsonl"
    lines = reg_path.read_text().splitlines()
    record = json.loads(lines[0])
    record["payload"]["genesis_hash"] = "0" * 64
    lines[0] = json.dumps(record, sort_keys=True)
    reg_path.write_text("\n".join(lines) + "\n")

    from mirror_memory.core.creation_registry import RegistryError
    with pytest.raises(RegistryError):
        log.registry.find(db_id)


def test_brand_new_life_registers_and_then_refuses_erasures(tmp_path):
    """A genuinely new life registers itself; from then on, deleting the log
    (even with the witness) is a refusal, not a re-creation."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "f4" * 16
    log.ensure_genesis(db_id, applied=0)
    assert log.registry.find(db_id) is not None, "creation was registered"

    _log_file(log_dir, db_id).unlink()
    log.witness.retire(db_id)                  # the whole evidence erased

    with pytest.raises(DeletionLogCorrupted):
        log.ensure_genesis(db_id, applied=0), log.verify_witness(db_id, applied=0)


def test_multiprocess_creation_writes_exactly_one_credential(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    db_id = "f5" * 16
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_mp_first_creation, args=(str(log_dir), db_id))
             for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, p.exitcode

    log = FileDeletionLog(log_dir)
    records = [r for r in log.registry.created_lives() if r == db_id]
    assert records == [db_id], "exactly one credential"
    assert _line_count(_log_file(log_dir, db_id)) == 1


# ---------------------------------------------------------------------------
# M5 patch 6 §2: the heal-forward path proves ANCESTRY, not just seq order.
# A rollback + re-append that lands on a higher seq must NOT be healed.
# ---------------------------------------------------------------------------


def test_rollback_with_reappend_to_higher_seq_is_refused(tmp_path):
    """Roll the log back to seq 1, keep the witness at seq 3, then append two
    NEW entries so the log reaches seq 3 again with DIFFERENT hashes.  The
    ancestry check must refuse the heal."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "g1" * 16
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 4):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    assert log.witness.read(db_id).tail_seq == 3

    # roll back: keep genesis + first entry only
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    path.write_bytes(b"\n".join(lines[:2]) + b"\n")
    log._index_invalidate(db_id)

    # re-appending onto the rolled-back log must be REFUSED at the write
    # (M5 patch 6 §2: the append validates the witness-vs-head under the
    # lock) -- the attack can never launder its forged head into the witness.
    for i in range(1, 3):
        with pytest.raises((WitnessAhead, DeletionLogCorrupted)):
            log.append(DeletionEntry(
                db_id=db_id, tenant_id="t", app_id="a", user_id=f"x{i}",
                scope="session", scope_key=f"x{i}", fingerprints=[],
                generation=100 + i, purpose="other"))
    # the log still holds exactly the surviving entries
    assert log.head(db_id)[0] == 1
    # the witness was NOT overwritten by the forged head
    assert log.witness.read(db_id).tail_seq == 3

    # and verify_witness still refuses the truncated state
    with pytest.raises(WitnessAhead):
        log.verify_witness(db_id, applied=3)


def test_rollback_with_reappend_still_refused_even_at_lower_head(tmp_path):
    """The attacker does not even need to reach the witnessed seq: a rollback
    to seq 1 with the witness at 3 must refuse outright (WitnessAhead)."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "g2" * 16
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 4):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    path.write_bytes(b"\n".join(lines[:2]) + b"\n")
    log._index_invalidate(db_id)

    with pytest.raises(WitnessAhead):
        log.verify_witness(db_id, applied=3)
    # the witness is untouched
    assert log.witness.read(db_id).tail_seq == 3


def test_crash_heal_still_works_when_the_chain_is_genuine(tmp_path):
    """The legitimate crash window (log ahead, witness behind, SAME chain)
    still heals -- the ancestry proof accepts the real history."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "g3" * 16
    log.ensure_genesis(db_id, applied=0)
    stale = log.witness.read(db_id)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=2, purpose="p"))
    log.witness.record(stale)                  # simulate the crash window
    assert log.witness.read(db_id).tail_seq == 0

    log.verify_witness(db_id, applied=2)       # heal forward
    assert log.witness.read(db_id).tail_seq == 2
    assert log.witness.read(db_id).tail_hash == log.head(db_id)[1]


# ---------------------------------------------------------------------------
# M5 patch 6 §3/§4: the operator CLI — dry-run by default, --yes to apply,
# failures change nothing, and the reconcile refuses to bless a SHORTER
# history than the witness already records.
# ---------------------------------------------------------------------------




def _patch_adopt_log(monkeypatch, log):
    """Make the CLI operate on the SAME FileDeletionLog object the test built
    (the real config loader does not carry the test's directory)."""
    from mirror_memory import deletion_adopt

    from mirror_memory.core import deletion as _deletion_mod

    monkeypatch.setattr(
        _deletion_mod, "deletion_log_from_config", lambda *a, **k: log)


def _run_adopt(args, monkeypatch, capsys):
    from mirror_memory import deletion_adopt

    monkeypatch.setattr("sys.argv", ["deletion-adopt", *args])
    return deletion_adopt.main()


def test_adopt_witness_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "h1" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    _patch_adopt_log(monkeypatch, log)
    before = log.witness.read(db_id)

    rc = _run_adopt(
        ["--database-url", f"sqlite:///{tmp_path/'main.db'}",
         "--db-id", db_id, "--config-dir", "config/",
         "--reconcile-witness"], monkeypatch, capsys)
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "nothing changed" in out
    # the witness is untouched
    assert log.witness.read(db_id).tail_seq == before.tail_seq


def test_adopt_witness_reconcile_refuses_a_shorter_history(
        tmp_path, monkeypatch, capsys):
    """The witness says seq 3; the log only reaches seq 2.  A reconcile would
    bless the shorter history -- refused, and nothing changes."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "h2" * 16
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 4):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    path.write_bytes(b"\n".join(lines[:3]) + b"\n")   # head drops to 2
    log._index_invalidate(db_id)
    _patch_adopt_log(monkeypatch, log)
    before = log.witness.read(db_id)

    with pytest.raises(SystemExit) as excinfo:
        _run_adopt(
            ["--database-url", f"sqlite:///{tmp_path/'main.db'}",
             "--db-id", db_id, "--config-dir", "config/",
             "--reconcile-witness", "--yes"], monkeypatch, capsys)
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out
    assert log.witness.read(db_id).tail_seq == before.tail_seq


def test_adopt_witness_reconcile_applies_with_yes(
        tmp_path, monkeypatch, capsys):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "h3" * 16
    log.ensure_genesis(db_id, applied=0)
    log.witness.retire(db_id)
    head_seq, head_hash = log.head(db_id)
    _patch_adopt_log(monkeypatch, log)

    rc = _run_adopt(
        ["--database-url", f"sqlite:///{tmp_path/'main.db'}",
         "--db-id", db_id, "--config-dir", "config/",
         "--reconcile-witness", "--yes"], monkeypatch, capsys)
    assert rc == 0
    state = log.witness.read(db_id)
    assert state.tail_seq == head_seq
    assert state.tail_hash == head_hash


def test_adopt_witness_retire_removes_only_the_witness(
        tmp_path, monkeypatch, capsys):
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "h4" * 16
    log.ensure_genesis(db_id, applied=0)
    path = _log_file(log_dir, db_id)
    log_bytes = path.read_bytes()
    registry_bytes = (log_dir / ".creation-registry.jsonl").read_bytes()
    _patch_adopt_log(monkeypatch, log)

    rc = _run_adopt(
        ["--database-url", f"sqlite:///{tmp_path/'main.db'}",
         "--db-id", db_id, "--config-dir", "config/",
         "--retire-witness", "--yes"], monkeypatch, capsys)
    assert rc == 0
    assert log.witness.read(db_id) is None
    # log and registry untouched
    assert path.read_bytes() == log_bytes
    assert (log_dir / ".creation-registry.jsonl").read_bytes() == registry_bytes


# ---------------------------------------------------------------------------
# M5 patch 7 §5: the two log-level reproductions required by the brief.
# Both must be REFUSED and both must leave the witness untouched.
# ---------------------------------------------------------------------------


def test_brief_repro_1_credential_kept_witness_removed_log_truncated(
        tmp_path):
    """Brief repro ①: keep the binding AND the creation credential, remove
    the witness after a real deletion, and truncate the log to the original
    genesis.  applied == 0 (a restored old backup).  The credential makes the
    interrupted-first-creation shape LOOK plausible -- it must still be
    refused, and the witness must not be re-minted."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "p1" * 16
    log.ensure_genesis(db_id, applied=0)     # binding + credential + witness
    credential = log.registry.find(db_id)
    assert credential is not None
    original_genesis = credential.genesis_hash

    # a real deletion happened
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    assert log.witness.read(db_id).tail_seq == 1

    # the attack: erase the witness, truncate the log to the original genesis
    log.witness.retire(db_id)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    path.write_bytes(lines[0] + b"\n")       # only the genesis survives
    log._index_invalidate(db_id)

    # every route must refuse
    with pytest.raises(DeletionLogCorrupted):
        log.verify_witness(db_id, applied=0)
    with pytest.raises(DeletionLogCorrupted):
        log.ensure_genesis(db_id, applied=0)

    # the witness stays GONE: no silent re-mint over the truncated history
    assert log.witness.read(db_id) is None
    # and the truncated evidence is untouched
    assert _line_count(path) == 1
    # the credential still names the ORIGINAL genesis
    assert log.registry.find(db_id).genesis_hash == original_genesis


def test_brief_repro_2_divergent_longer_chain_at_witness_point(tmp_path):
    """Brief repro ②: keep the OLD witness and the original genesis, but
    replace the log beyond the witness point with a LONGER, internally valid
    forked chain, then append.  The witness records seq N with hash H; the
    forged chain also reaches seq >= N but via different hashes.  Must be
    refused (equal-height different-hash or ancestry failure) with the
    witness unchanged."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "p2" * 16
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 4):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    witness = log.witness.read(db_id)
    assert witness.tail_seq == 3
    original_tail_hash = witness.tail_hash
    genesis_line = [ln for ln in path_lines(log_dir, db_id)
                    if json.loads(ln).get("scope") == "genesis"][0]

    # forge: keep genesis, then build a DIFFERENT 5-entry chain from it
    forged_entries = []
    prev = json.loads(genesis_line)["hash"]
    for i in range(1, 6):
        entry = DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"f{i}",
            scope="session", scope_key=f"z{i}", fingerprints=[],
            generation=50 + i, purpose="forged")
        entry.seq = i
        entry.prev_hash = prev
        entry.created_at = None
        entry.hash = entry.compute_hash()
        forged_entries.append(entry)
        prev = entry.hash
    path = _log_file(log_dir, db_id)
    path.write_bytes(genesis_line.encode() + b"\n" + b"".join(
        (json.dumps(log._line(e), sort_keys=True) + "\n").encode()
        for e in forged_entries))
    log._index_invalidate(db_id)
    assert log.head(db_id)[0] == 5           # the fork is LONGER than seq 3

    # append on top of the forged chain must be refused
    with pytest.raises((WitnessAhead, DeletionLogCorrupted)):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id="x", scope="user",
            scope_key="", fingerprints=[], generation=99, purpose="x"))

    # verify must refuse (equal-ish height, different hashes => diverged)
    with pytest.raises((WitnessAhead, DeletionLogCorrupted)):
        log.verify_witness(db_id, applied=3)

    # the witness is untouched: still the ORIGINAL seq-3 tail
    after = log.witness.read(db_id)
    assert after.tail_seq == 3
    assert after.tail_hash == original_tail_hash


def path_lines(log_dir: Path, db_id: str) -> list[str]:
    path = log_dir / f"{db_id}.deletions.jsonl"
    return [ln for ln in path.read_bytes().decode().split("\n") if ln]


# ---------------------------------------------------------------------------
# M5 patch 7 §4: the CLI must refuse every forged shape and leave the log,
# the witness AND the registry byte-identical on failure.
# ---------------------------------------------------------------------------


def _cli_state(log_dir: Path, db_id: str) -> dict:
    log = FileDeletionLog(log_dir)
    return {
        "log": (log_dir / f"{db_id}.deletions.jsonl").read_bytes()
        if (log_dir / f"{db_id}.deletions.jsonl").exists() else None,
        "registry": (log_dir / ".creation-registry.jsonl").read_bytes()
        if (log_dir / ".creation-registry.jsonl").exists() else None,
        "witness": (lambda s: (
            (log.witness.directory / f"{db_id}.witness.json").read_bytes()
            if (log.witness.directory / f"{db_id}.witness.json").exists()
            else None))(log),
    }


def test_cli_reconcile_refuses_a_diverged_fork_unchanged(
        tmp_path, monkeypatch, capsys):
    """A forked chain with the same height but different hashes: reconcile
    must refuse and every artifact stays byte-identical."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "q1" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    witness_before = log.witness.read(db_id)

    # forge: same height (seq 1), different content
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    entry = DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="zz", scope="user",
        scope_key="", fingerprints=[], generation=77, purpose="forged")
    entry.seq = 1
    entry.prev_hash = json.loads(lines[0])["hash"]
    entry.created_at = None
    entry.hash = entry.compute_hash()
    path.write_bytes(lines[0] + b"\n" +
                     (json.dumps(log._line(entry), sort_keys=True) + "\n").encode())
    log._index_invalidate(db_id)
    before = _cli_state(log_dir, db_id)

    with pytest.raises(SystemExit) as excinfo:
        _patch_adopt_log(monkeypatch, log)
        _run_adopt(
            ["--database-url", f"sqlite:///{tmp_path/'main.db'}",
             "--db-id", db_id, "--config-dir", "config/",
             "--reconcile-witness", "--yes"], monkeypatch, capsys)
    assert excinfo.value.code == 1
    after = _cli_state(log_dir, db_id)
    assert after == before, "a refused CLI run must change NOTHING"

    # the witness still holds the ORIGINAL tail hash
    assert log.witness.read(db_id).tail_hash == witness_before.tail_hash


# ---------------------------------------------------------------------------
# M5 patch 7 §5: brief repro ① AND ② through the REAL ENGINE startup path
# (not a direct verify_witness call), with backup/restore.
# ---------------------------------------------------------------------------


def _prepared_engine(tmp_path: Path, deletions: int = 2):
    log_dir = tmp_path / "log"
    db = tmp_path / "main.db"
    eng = _engine(tmp_path, log_dir, db)
    _enable(eng)
    _observe(eng)
    backup = tmp_path / "before.db"
    _backup(db, backup)
    for i in range(deletions):
        _observe(eng, session=f"d{i}")
        eng.delete_memories(user_id="u1", purpose="chat")
    eng._engine.dispose()
    return log_dir, db, backup


def test_brief_repro_1_through_real_engine_startup(tmp_path):
    """Repro ① end to end: real engine creates the history, the attacker
    truncates the log to genesis + erases the witness, the old main DB is
    restored (applied == 0), and a REAL MemoryEngine must refuse startup."""
    log_dir, db, backup = _prepared_engine(tmp_path)
    log = FileDeletionLog(log_dir)
    db_id = log.created_lives()[0] if hasattr(log, "created_lives") else None
    # find the single life
    lives = log.registry.created_lives()
    assert len(lives) == 1
    db_id = lives[0]

    log.witness.retire(db_id)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    path.write_bytes(lines[0] + b"\n")          # truncate to genesis
    log._index_invalidate(db_id)
    _restore(backup, db)

    eng = _engine(tmp_path, log_dir, db)
    with pytest.raises((DeletionLogCorrupted, DeletionLogUnavailable)):
        eng._ensure_db()
    assert log.witness.read(db_id) is None, "witness must stay gone"
    assert _line_count(path) == 1, "truncated evidence must be preserved"


def test_brief_repro_2_through_real_engine_append(tmp_path):
    """Repro ② end to end: real history, the witness kept, the log replaced
    by a longer forked chain; a REAL engine must refuse startup, and an
    already-running engine must refuse observe."""
    log_dir, db, _backup_used = _prepared_engine(tmp_path)
    log = FileDeletionLog(log_dir)
    db_id = log.registry.created_lives()[0]
    witness_before = log.witness.read(db_id)
    genesis_line = [ln for ln in path_lines(log_dir, db_id)
                    if json.loads(ln).get("scope") == "genesis"][0]

    prev = json.loads(genesis_line)["hash"]
    forged = []
    for i in range(1, 6):
        entry = DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"f{i}",
            scope="session", scope_key=f"z{i}", fingerprints=[],
            generation=50 + i, purpose="forged")
        entry.seq = i
        entry.prev_hash = prev
        entry.created_at = None
        entry.hash = entry.compute_hash()
        forged.append(entry)
        prev = entry.hash
    path = _log_file(log_dir, db_id)
    path.write_bytes(genesis_line.encode() + b"\n" + b"".join(
        (json.dumps(log._line(e), sort_keys=True) + "\n").encode()
        for e in forged))
    log._index_invalidate(db_id)

    eng = _engine(tmp_path, log_dir, db)
    with pytest.raises((WitnessAhead, DeletionLogCorrupted)):
        eng._ensure_db()
    with pytest.raises((WitnessAhead, DeletionLogCorrupted,
                        DeletionLogUnavailable)):
        eng.observe(user_id="u1", session_id="sx", text="x", purpose="chat")
    # witness unchanged
    after = log.witness.read(db_id)
    assert after.tail_seq == witness_before.tail_seq
    assert after.tail_hash == witness_before.tail_hash


# ---------------------------------------------------------------------------
# M5 patch 8 §4: the required runtime regressions.
# ---------------------------------------------------------------------------


def test_p8_runtime_witness_removed_and_log_truncated_then_append(tmp_path):
    """① An ALREADY-RUNNING history: the witness is removed and the log is
    truncated to its genesis, then append() is attempted.  It must be
    refused, and neither the log nor the (absent) witness may change."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "r1" * 16
    log.ensure_genesis(db_id, applied=0)
    for i in range(1, 4):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))

    # the attack, against the RUNNING log object (not a restart)
    log.witness.retire(db_id)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    path.write_bytes(lines[0] + b"\n")
    log._index_invalidate(db_id)
    before_log = path.read_bytes()

    with pytest.raises((DeletionLogCorrupted, DeletionLogUnavailable)):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id="x",
            scope="user", scope_key="", fingerprints=[], generation=99,
            purpose="x"))

    assert path.read_bytes() == before_log, "the log must not change"
    assert log.witness.read(db_id) is None, "no witness may be minted"


def test_p8_witness_behind_and_tail_tampered_startup_does_not_heal(
        tmp_path):
    """③ The witness is BEHIND (log longer) and the extra entries are
    tampered.  Startup must NOT heal-forward the witness before the chain
    has been verified -- verify_chain runs first (M5 patch 7/8 §1)."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "r2" * 16
    log.ensure_genesis(db_id, applied=0)
    stale = log.witness.read(db_id)
    for i in range(1, 4):
        log.append(DeletionEntry(
            db_id=db_id, tenant_id="t", app_id="a", user_id=f"u{i}",
            scope="session", scope_key=f"s{i}", fingerprints=[],
            generation=i, purpose="p"))
    log.witness.record(stale)                 # the witness is behind
    assert log.witness.read(db_id).tail_seq == 0

    # tamper the LAST entry's hash (chain broken beyond the witness)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    record = json.loads(lines[-1])
    record["hash"] = "f" * 64
    lines[-1] = json.dumps(record, sort_keys=True).encode()
    path.write_bytes(b"\n".join(lines) + b"\n")
    log._index_invalidate(db_id)
    witness_before = log.witness.read(db_id)

    # verify_witness itself must run the chain check BEFORE healing
    with pytest.raises(Exception):
        log.verify_witness(db_id, applied=0)

    after = log.witness.read(db_id)
    assert after.tail_seq == witness_before.tail_seq, (
        "the witness must NOT be healed forward over a broken chain")
    assert after.tail_hash == witness_before.tail_hash


def test_p8_bound_and_key_present_but_all_evidence_gone(tmp_path):
    """④ Binding and key survive, but log, witness and registry are all
    missing and the old watermark is 0.  Must NOT mint."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "r3" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    key_path = log.witness.key_path

    # erase everything except the binding marker and the key
    log.witness.retire(db_id)
    (log_dir / ".creation-registry.jsonl").unlink()
    _log_file(log_dir, db_id).unlink()
    assert log.is_bound(db_id), "the binding marker survived"
    assert key_path.exists()

    with pytest.raises(DeletionLogCorrupted) as excinfo:
        log.ensure_genesis(db_id, applied=0)
    assert "bound" in str(excinfo.value).lower()
    assert not _log_file(log_dir, db_id).exists(), "no new log may be minted"
    assert log.witness.read(db_id) is None


def test_p8_bound_and_key_only_through_real_engine(tmp_path):
    """④ through the REAL engine: only the binding marker and the key
    survive; log, witness and registry are gone and the main DB watermark is
    0.  Startup must refuse and nothing may be created."""
    log_dir = tmp_path / "log"
    db = tmp_path / "main.db"
    eng = _engine(tmp_path, log_dir, db)
    _enable(eng)
    _observe(eng)
    eng.delete_memories(user_id="u1", purpose="chat")
    db_id = eng._db_id
    eng._engine.dispose()

    log = FileDeletionLog(log_dir)
    log.witness.retire(db_id)
    (log_dir / ".creation-registry.jsonl").unlink()
    _log_file(log_dir, db_id).unlink()
    assert log.is_bound(db_id) and log.witness.key_path.exists()

    eng2 = _engine(tmp_path, log_dir, db)
    with pytest.raises((DeletionLogCorrupted, DeletionLogUnavailable)):
        eng2._ensure_db()
    assert not _log_file(log_dir, db_id).exists(), "no log may be minted"


def test_p8_cli_dry_run_has_no_persistent_side_effects(
        tmp_path, monkeypatch, capsys):
    """§3: a dry run must not change the log, the witness OR the registry --
    and it must report the chain/credential state."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "d8" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    _patch_adopt_log(monkeypatch, log)
    before = _cli_state(log_dir, db_id)

    rc = _run_adopt(
        ["--database-url", f"sqlite:///{tmp_path/'main.db'}",
         "--db-id", db_id, "--config-dir", "config/",
         "--reconcile-witness"], monkeypatch, capsys)
    assert rc == 0
    out = capsys.readouterr().out
    assert "chain: verified" in out
    assert "dry run" in out
    assert _cli_state(log_dir, db_id) == before, "a dry run changed state"


def test_p8_cli_refuses_when_chain_is_broken(tmp_path, monkeypatch, capsys):
    """A reconcile over a BROKEN chain must refuse (and change nothing)."""
    log_dir = tmp_path / "log"
    log = FileDeletionLog(log_dir)
    db_id = "d9" * 16
    log.ensure_genesis(db_id, applied=0)
    log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u", scope="user",
        scope_key="", fingerprints=[], generation=1, purpose="p"))
    log.witness.retire(db_id)
    path = _log_file(log_dir, db_id)
    lines = [ln for ln in path.read_bytes().split(b"\n") if ln]
    record = json.loads(lines[-1])
    record["hash"] = "e" * 64
    lines[-1] = json.dumps(record, sort_keys=True).encode()
    path.write_bytes(b"\n".join(lines) + b"\n")
    log._index_invalidate(db_id)
    _patch_adopt_log(monkeypatch, log)
    before = _cli_state(log_dir, db_id)

    with pytest.raises(SystemExit) as excinfo:
        _run_adopt(
            ["--database-url", f"sqlite:///{tmp_path/'main.db'}",
             "--db-id", db_id, "--config-dir", "config/",
             "--reconcile-witness", "--yes"], monkeypatch, capsys)
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out
    assert _cli_state(log_dir, db_id) == before
