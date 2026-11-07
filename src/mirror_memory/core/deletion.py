"""The independent deletion log (M4 task 1).

Every deletion is recorded OUTSIDE the main database -- the record must
survive a restore of the main DB, which is exactly what makes replay
after a backup-restore possible (I2 extended to backups: deleted memory
cannot come back).

Pairing.  The log is paired to one *life* of the main database: the
main DB's watermark row (``mm_deletion_watermark``, single-row, rolls
back WITH the main DB) carries a ``db_id``; log entries are keyed by the
same ``db_id``.

- restore a backup  -> the watermark row (db_id + applied seq) comes back
  -> the log for that db_id is found again -> the entries applied after
  the backup are replayed onto the restored rows.
- create a fresh DB -> a new db_id -> a fresh, empty log (an old log is
  never replayed onto an unrelated database).
- log older than the main DB (applied seq > log tail) -> the log lost
  entries the main DB already applied -> refuse to start (the log's
  recovery point must not be older than the main DB).

Entries carry NO content: row fingerprints are hashes of
``table:id:created_at`` -- enough for replay to match "the row that was
deleted" (a recreated row with a reused id has a different created_at
and does not match) without the log itself becoming a memory copy.

Fail-closed ordering.  A deletion writes the log BEFORE committing the
main transaction: a log failure rolls the deletion back (no unlogged
deletion); the reverse window -- log written, main commit fails -- is
safe because replay matches fingerprints that will never be found (the
entry applies as a no-op and advances the watermark).
"""

from __future__ import annotations

import contextlib
import threading
import fcntl
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

GENESIS_HASH = "0" * 64


class DeletionLogUnavailable(RuntimeError):
    """No deletion log is configured: deletions are refused (fail-closed)
    -- an unlogged deletion cannot be replayed after a restore."""


class DeletionLogCorrupted(DeletionLogUnavailable):
    """The log store exists but cannot be trusted as this database's
    history: a pre-existing EMPTY or TRUNCATED log while the watermark
    records applied deletions, malformed lines, or a file without its
    genesis.  Fail-closed -- recovery/adoption is an explicit operator
    flow (deletion-adopt / restore the log), never an automatic
    re-initialization."""


class LogChainBroken(RuntimeError):
    """The log's hash chain does not verify: entries were lost, altered
    or written out of order.  Startup is refused."""


class DeletionLogStale(RuntimeError):
    """The main DB applied more log entries than the log still holds:
    the log's recovery point is OLDER than the main database.  Restore
    the log (or a newer backup of it) before starting."""


class UnhandledDeletionTable(RuntimeError):
    """A deletion-log entry names a table with no replay handler: the
    coverage matrix and core.replay must be extended together.  Never
    silently ignored -- the restored database would keep deleted
    content."""


class DeletionLogPruned(RuntimeError):
    """The main database's watermark lies below the deletion log's
    compaction checkpoint: the restored backup is OLDER than the log's
    retention -- the pruned entries can no longer be replayed, and
    replaying only the remainder would resurrect deleted content.
    Restore a newer backup (one taken after that checkpoint) and adopt it,
    or retire the database's history deliberately (new-lineage)."""


class ReplayPending(RuntimeError):
    """Another instance is replaying the log onto this database; entries
    are refused until it finishes."""


def fingerprint(table: str, row_id: object, created_at: datetime | None) -> str:
    """The content-free identity of one deleted row: hash of
    ``table:id:created_at``.  A recreated row with a reused id has a
    different created_at, so replay never deletes it by mistake."""
    raw = f"{table}:{row_id}:{created_at.isoformat() if created_at else ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class DeletionEntry:
    """One committed-or-pending deletion, content-free."""

    db_id: str
    tenant_id: str
    app_id: str
    user_id: str
    scope: str  # "belief" / "session" / "user"
    scope_key: str  # belief_id / session_id / ""
    fingerprints: list[dict] = field(default_factory=list)  # [{table,id,created_at}]
    generation: int = 0
    purpose: str = ""
    seq: int = 0  # assigned by the log on append
    prev_hash: str = GENESIS_HASH
    hash: str = ""
    created_at: datetime | None = None
    # M4 batch 2 (tombstone sync): the cross-device scope key -- a
    # belief's global_id / the session_id / "" -- and, for a deletion
    # produced by APPLYING a remote tombstone, where that tombstone came
    # from (the export never echoes it back to its origin)
    global_key: str = ""
    origin: str = ""
    origin_seq: int = 0
    # M4 batch-2 patch 2: for an entry produced by APPLYING a remote
    # tombstone, the origin's log-entry hash it was exported with -- the
    # stable identity the applied table and the echo check key on
    origin_hash: str = ""

    def payload(self) -> dict:
        """The canonical, hashable form (everything except seq/hash).  The
        batch-2 fields enter the payload only when set, so entries written
        before them hash exactly as they did (the chain stays valid)."""
        payload = {
            "db_id": self.db_id,
            "tenant_id": self.tenant_id,
            "app_id": self.app_id,
            "user_id": self.user_id,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "fingerprints": self.fingerprints,
            "generation": self.generation,
            "purpose": self.purpose,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "prev_hash": self.prev_hash,
        }
        if self.global_key:
            payload["global_key"] = self.global_key
        if self.origin:
            payload["origin"] = self.origin
            payload["origin_seq"] = self.origin_seq
        if self.origin_hash:
            payload["origin_hash"] = self.origin_hash
        return payload

    def compute_hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _genesis_entry(db_id: str) -> DeletionEntry:
    """The birth certificate of one database life: seq 0, no deletion
    content, hash-chained like every entry.  Its presence proves THIS log
    belongs to THAT database -- a log without it was created elsewhere."""
    return DeletionEntry(db_id=db_id, tenant_id="", app_id="", user_id="",
                         scope="genesis", scope_key="", fingerprints=[],
                         generation=0, purpose="",
                         created_at=datetime.utcnow())


class DeletionLog:
    """The log interface: append / read_from / head / genesis, per
    database life."""

    def append(self, entry: DeletionEntry) -> DeletionEntry:  # pragma: no cover - interface
        raise NotImplementedError

    def read_from(self, db_id: str, after_seq: int) -> Iterator[DeletionEntry]:  # pragma: no cover
        raise NotImplementedError

    def head(self, db_id: str) -> tuple[int, str]:  # pragma: no cover
        """(last seq, last hash); (0, GENESIS) for an empty log."""
        raise NotImplementedError

    def _has_genesis_indexed(self, db_id: str) -> bool:
        """Genesis presence via the stat-guarded index -- O(1) for an
        unchanged file, one parse only when the file actually changed.

        The append path must never rescan the whole log (M5 patch 4); the
        index already holds every seq, and seq 0 is the genesis.

        A COMPACTED log legitimately no longer carries the seq-0 line: prune
        replaces the retired prefix with a CHECKPOINT entry that records the
        chain hash and proves the lineage continues from the pruned genesis.
        So a checkpoint is accepted as evidence of an initialized log, which
        is exactly what it is."""
        path = self._path_for(db_id)
        try:
            if path.stat().st_size == 0:
                return False
        except FileNotFoundError:
            return False
        idx = self._index_for(db_id)
        for seq, entry in idx.items():
            if entry[2] == "genesis":
                return True
        # compacted: a checkpoint carries the lineage of the pruned genesis
        return self._index_checkpoint.get(db_id, 0) > 0

    def has_genesis(self, db_id: str) -> bool:  # pragma: no cover - interface
        """Does this log carry the genesis entry (seq 0) for *db_id*?  Its
        absence means the log was created for a DIFFERENT database."""
        raise NotImplementedError

    def chain_hash_at(self, db_id: str, seq: int) -> str:  # pragma: no cover - interface
        """The chain hash stored AT *seq* (GENESIS_HASH for seq 0 or when
        no entry exists at that position) -- the sync channel's continuity
        anchor (M4 batch-2 patch 2)."""
        raise NotImplementedError

    def has_tombstone(self, db_id: str, seq: int, entry_hash: str) -> bool:  # pragma: no cover - interface
        """True when the log still holds the entry the tombstone names:
        (seq, entry_hash) -- the sender's log position and entry hash the
        tombstone was exported from.  Absent means the deletion was lost
        in a restore and must re-apply."""
        raise NotImplementedError

    def checkpoint_seq(self, db_id: str) -> int:  # pragma: no cover - interface
        """The seq of the log's compaction checkpoint (0 = none): entries
        at or below it have been pruned; the entry at that seq lives on
        only as the checkpoint (its payload is the recorded chain hash)."""
        raise NotImplementedError

    def genesis_entries(self) -> list[tuple[str, object]]:  # pragma: no cover - interface
        """Every genesis entry in the log store as (db_id, created_at) --
        the migration's lineage check (M4 patch 2): a log that holds ANY
        genesis has already served a database life."""
        raise NotImplementedError

    def ensure_genesis(self, db_id: str, *, applied: int = 0) -> None:  # pragma: no cover - interface
        """Write the genesis entry if this log has none for *db_id*.

        *applied* is the main database's watermark (applied deletions) and
        is the FIRST-CREATION proof: a brand-new database life has applied
        == 0, so minting a genesis is safe -- nothing was ever deleted.
        With applied > 0 a missing/empty log means history was LOST
        (truncated, deleted, interrupted mid-life) and must never be
        silently re-initialized: fail closed instead."""
        raise NotImplementedError

    def verify_chain(self, db_id: str) -> None:
        """Walk the whole chain INCLUDING the genesis entry (after_seq = -1
        reaches seq 0); raise LogChainBroken on any mismatch.  A
        checkpoint (compaction) entry carries the hash the chain had at
        the pruned-through point and is verified as a recorded fact, not
        recomputed -- the chain continues from its hash."""
        prev = GENESIS_HASH
        for entry in self.read_from(db_id, -1):
            if entry.scope == "checkpoint":
                # the link into the kept chain must hold; the checkpoint's
                # own hash is a recorded fact (its payload was pruned)
                if entry.prev_hash != prev:
                    raise LogChainBroken(
                        f"seq {entry.seq}: checkpoint does not continue the chain")
                prev = entry.hash
                continue
            if entry.prev_hash != prev:
                raise LogChainBroken(f"seq {entry.seq}: prev_hash mismatch")
            if entry.hash != entry.compute_hash():
                raise LogChainBroken(f"seq {entry.seq}: entry hash mismatch")
            prev = entry.hash


# ---------------------------------------------------------------------------
# Embedded backend: an append-only file next to (never inside) the main DB
# ---------------------------------------------------------------------------


class FileDeletionLog(DeletionLog):
    """JSON-lines file per database life: line 1 is the ``db_id`` header,
    every further line one entry.  Each append is flushed AND fsynced; a
    flock serialises appends so the hash chain cannot interleave.  The
    chain is verified on first open (startup) and refuses to run broken."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._verified: set[str] = set()
        # Per-db_id position index: seq -> (hash, origin_hash, scope).
        # Built on first use and refreshed whenever the file's
        # (size, mtime_ns) changes, so the per-entry lookups
        # (chain_hash_at / has_tombstone / head / checkpoint_seq /
        # append's tail) stop rescanning the whole file (batch-2 patch 3,
        # task 2).  Any external rewrite -- a device restore, a prune in
        # another process -- changes size or mtime, which invalidates the
        # cache; our own writes refresh it under the append lock.
        self._index: dict[str, dict[int, tuple[str, str, str]]] = {}
        self._index_stat: dict[str, tuple[int, int, int]] = {}
        # (last_seq, last_hash) per db_id -- head() and append's tail
        # without a max() scan
        self._index_tail: dict[str, tuple[int, str]] = {}
        # the log's highest checkpoint seq per db_id (checkpoint_seq
        # without an index walk)
        self._index_checkpoint: dict[str, int] = {}

    def _index_for(self, db_id: str) -> dict[int, tuple[str, str, str]]:
        """The db_id's position index, rebuilt when the file changed on
        disk (a restore rewrites the file behind our back).

        The freshness guard is the (inode, size, mtime_ns) stat key: a
        restore changes the inode (copy) or the size/mtime (rewrite), so
        the next lookup rebuilds.  HONEST LIMIT (batch-2 patch 4, task 3):
        an in-place rewrite preserving ALL THREE is not caught here -- an
        in-place same-stat tamper is what the startup/reconcile chain
        verification catches (verify_chain), and every process start
        rebuilds the index from the file anyway.  Our own append folds
        into the index under the append lock; our own prune invalidates
        explicitly."""
        path = self._path_for(db_id)
        try:
            st = path.stat()
        except FileNotFoundError:
            self._index_invalidate(db_id)
            return {}
        stat_key = (st.st_ino, st.st_size, st.st_mtime_ns)
        cached = self._index.get(db_id)
        if cached is not None and self._index_stat.get(db_id) == stat_key:
            return cached
        idx: dict[int, tuple[str, str, str]] = {}
        checkpoint = 0
        for e in self._iter_entries(path):
            idx[e.seq] = (e.hash, e.origin_hash, e.scope)
            if e.scope == "checkpoint" and e.seq > checkpoint:
                checkpoint = e.seq
        self._index[db_id] = idx
        self._index_stat[db_id] = stat_key
        self._index_checkpoint[db_id] = checkpoint
        if idx:
            last = max(idx)
            self._index_tail[db_id] = (last, idx[last][0])
        else:
            self._index_tail.pop(db_id, None)
            self._index_checkpoint.pop(db_id, None)
        return idx

    def _index_record_write(self, db_id: str, seq: int, entry_hash: str,
                            origin_hash: str, scope: str) -> None:
        """Fold OUR OWN append into the index and refresh the cached stat
        -- called while the append lock is still held, so the stat cannot
        race another writer."""
        path = self._path_for(db_id)
        idx = self._index.setdefault(db_id, {})
        idx[seq] = (entry_hash, origin_hash, scope)
        self._index_tail[db_id] = (seq, entry_hash)
        try:
            st = path.stat()
        except FileNotFoundError:
            self._index_invalidate(db_id)
            return
        self._index_stat[db_id] = (st.st_ino, st.st_size, st.st_mtime_ns)

    def _index_invalidate(self, db_id: str) -> None:
        self._index.pop(db_id, None)
        self._index_stat.pop(db_id, None)
        self._index_tail.pop(db_id, None)
        self._index_checkpoint.pop(db_id, None)

    def _path_for(self, db_id: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"{db_id}.deletions.jsonl"

    @contextlib.contextmanager
    def _locked(self, path: Path):
        with open(path, "a+b") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield fh
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _binding_path(self, db_id: str) -> Path:
        """A tiny sidecar that records "a log file existed for this db_id".

        Why it is needed (M5 patch 4, the headline scenario): a restored
        backup brings its OLD watermark back -- typically ``applied == 0`` --
        so the watermark alone can never distinguish "brand-new database"
        from "restored backup whose log was truncated".  The sidecar is
        written once, in the log store, *before* the log file; it survives a
        truncation of the log file itself, so an empty log file next to an
        existing binding is provably NOT a first creation."""
        return self.directory / f"{db_id}.bound"

    def _mark_bound(self, db_id: str) -> None:
        """Durably record that this log store owns *db_id*.

        Idempotent and race-safe across THREADS and PROCESSES: the marker is
        written to a uniquely named temp file, fsynced, then published with
        an exclusive ``os.link`` (which fails with FileExistsError for the
        loser instead of clobbering the winner).  A publish that already
        happened is a success, not an error."""
        path = self._binding_path(db_id)
        if path.exists():
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / f".{db_id}.bound.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(db_id + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(tmp, path)        # atomic publish, fails if present
            except FileExistsError:
                pass                      # another racer published first
            except OSError:
                # filesystems without hard links: best-effort atomic rename
                if not path.exists():
                    os.replace(tmp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)

    def is_bound(self, db_id: str) -> bool:
        """True when this log store has EVER held a log for *db_id*."""
        return self._binding_path(db_id).exists()

    def _write_genesis(self, fh, db_id: str) -> None:
        """Write the genesis entry (seq 0) to a freshly created log file at
        the current position; flush + fsync so the birth certificate is
        durable before anything else can observe the file."""
        genesis = _genesis_entry(db_id)
        genesis.seq = 0
        genesis.prev_hash = GENESIS_HASH
        genesis.hash = genesis.compute_hash()
        fh.seek(0)
        fh.write((json.dumps(self._line(genesis), sort_keys=True,
                             ensure_ascii=False) + "\n").encode())
        fh.flush()
        os.fsync(fh.fileno())

    def ensure_genesis(self, db_id: str, *, applied: int = 0) -> None:
        """Bind the log to this database life (M5 patch 4, task 1).

        Creation protocol -- only a PROVABLE first creation initializes:

        * the file does not exist AND the watermark is ``applied == 0``
          (a brand-new database life that never deleted anything): create
          it with O_CREAT|O_EXCL -- the winner of the exclusive create is
          the provable creator -- and write the genesis under the lock;
        * the file exists but is EMPTY (a concurrent creation that was
          interrupted between create and write, i.e. the patch-3 race
          window) and ``applied == 0``: re-initialize -- nothing was ever
          deleted, so the empty residue can hold only a genesis;
        * ANY other state is NOT initialized:
            - file missing with ``applied > 0``: the log was lost after
              deletions happened -- fail closed (``DeletionLogCorrupted``);
            - file EMPTY with ``applied > 0``: the log was truncated --
              fail closed, never mint new history over it;
            - file has content but no genesis: corrupted/foreign -- fail
              closed;
            - file has content with genesis: a live log -- untouched (a
              subsequent reconcile verifies the chain).

        Multi-process coordination is the O_EXCL create plus the per-file
        ``flock``: racers serialise, exactly one genesis ever lands, and a
        crash mid-creation is recovered from the ``applied == 0`` proof on
        the next start.  Every initialization write is fsynced."""
        path = self._path_for(db_id)
        if applied > 0 and not path.exists():
            raise DeletionLogCorrupted(
                f"no deletion log file exists for database {db_id[:8]}... "
                f"but the watermark records {applied} applied deletion(s): "
                "the log was lost or never written for this database life.  "
                "Refusing to mint new history -- restore the log (or a "
                "backup of it) and adopt the database, or retire this "
                "history deliberately with a new deployment line.")
        was_bound = self.is_bound(db_id)
        # the binding is written before the file: if a crash happens after
        # this point, the next start sees "bound but no/empty log" and refuses
        # instead of minting a fresh history over a lost one.
        self._mark_bound(db_id)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            pass
        else:
            # exclusive create won: we are the provable creator
            with os.fdopen(fd, "a+b") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    self._write_genesis(fh, db_id)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            return
        # the file pre-exists: decide on CONTENT + applied under the lock
        with self._locked(path) as fh:
            size = os.fstat(fh.fileno()).st_size
            if size == 0:
                if applied > 0 or was_bound:
                    # bound + empty (or applied > 0): history was lost.  A
                    # restored backup carries its OLD watermark (often 0), so
                    # the binding -- not the watermark -- is the proof.
                    raise DeletionLogCorrupted(
                        f"the deletion log for database {db_id[:8]}... is "
                        f"EMPTY (0 bytes) but this log store has already "
                        f"been bound to that database and its watermark is "
                        f"{applied}: the log was truncated, or a restore "
                        "brought back a database whose log is gone.  "
                        "Refusing to treat an empty log as a new one -- "
                        "restore the log and adopt, or retire this history "
                        "deliberately.")
                # crash between create and write, nothing ever deleted:
                # safe to (re-)initialize the genesis
                self._write_genesis(fh, db_id)
                return
            if not self.has_genesis(db_id):
                raise DeletionLogCorrupted(
                    f"the deletion log for database {db_id[:8]}... has "
                    "content but no genesis entry: it is corrupted or was "
                    "not created by this engine.  Refusing to start -- "
                    "restore/adopt the log explicitly.")
            # a live log: untouched here; reconcile verifies the chain
        return

    def _ensure_file(self, db_id: str) -> Path:
        """DEPRECATED creation helper kept for the adopt/backfill paths that
        legitimately mint a brand-new log store (applied == 0 proof is the
        caller's responsibility there).  O(1): decides on the stat size,
        never reads the file."""
        path = self._path_for(db_id)
        with self._locked(path) as fh:
            if os.fstat(fh.fileno()).st_size == 0:
                self._write_genesis(fh, db_id)
        return path

    def append(self, entry: DeletionEntry) -> DeletionEntry:
        # M5 patch 4: an append must never (re-)create a log for a database
        # life it has no genesis for -- startup reconciliation guarantees the
        # genesis exists; a vanished log is corruption, not a fresh start.
        # The check uses the stat-guarded INDEX (segment 0), never a full parse
        # of the file, so appends stay O(1) in log length.
        if not self._has_genesis_indexed(entry.db_id):
            raise DeletionLogUnavailable(
                f"cannot append: no deletion log with a genesis entry "
                f"exists for database {entry.db_id[:8]}...  The log was "
                "lost or never initialized -- reconcile/adopt first; "
                "deletions stay refused (fail-closed).")
        path = self._path_for(entry.db_id)
        # the instant is part of the hashed payload (and travels in the
        # tombstone): stamp it before the hash, like the PG backend does
        entry.created_at = entry.created_at or datetime.utcnow()
        with self._locked(path) as fh:
            # the tail comes from the cached tail pointer (refreshed under
            # the lock), not from a full-file parse
            self._index_for(entry.db_id)
            tail = self._index_tail.get(entry.db_id)
            last_seq = tail[0] if tail else 0
            entry.seq = last_seq + 1
            entry.prev_hash = tail[1] if tail else GENESIS_HASH
            entry.hash = entry.compute_hash()
            fh.seek(0, os.SEEK_END)
            fh.write((json.dumps(self._line(entry), sort_keys=True, ensure_ascii=False) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())
            self._index_record_write(entry.db_id, entry.seq, entry.hash,
                                     entry.origin_hash, entry.scope)
        return entry

    @staticmethod
    def _line(entry: DeletionEntry) -> dict:
        return {
            "seq": entry.seq,
            "db_id": entry.db_id,
            "tenant_id": entry.tenant_id,
            "app_id": entry.app_id,
            "user_id": entry.user_id,
            "scope": entry.scope,
            "scope_key": entry.scope_key,
            "fingerprints": entry.fingerprints,
            "generation": entry.generation,
            "purpose": entry.purpose,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "prev_hash": entry.prev_hash,
            "hash": entry.hash,
            "global_key": entry.global_key,
            "origin": entry.origin,
            "origin_seq": entry.origin_seq,
            "origin_hash": entry.origin_hash,
        }

    def _iter_entries(self, path: Path) -> Iterator[DeletionEntry]:
        with open(path, "rb") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    # a torn/truncated write or manual tampering: fail
                    # closed with an actionable corruption error
                    raise DeletionLogCorrupted(
                        f"{path.name} line {lineno} is not valid JSON "
                        f"(the log was truncated or corrupted mid-write): "
                        f"{exc}.  Restore/adopt the log explicitly; the "
                        "chain cannot be trusted past this point."
                    ) from exc
                if not isinstance(record, dict) or record.get("header"):
                    continue
                yield DeletionEntry(
                    db_id=record["db_id"],
                    seq=record["seq"],
                    tenant_id=record["tenant_id"],
                    app_id=record["app_id"],
                    user_id=record["user_id"],
                    scope=record["scope"],
                    scope_key=record["scope_key"],
                    fingerprints=record["fingerprints"],
                    generation=record["generation"],
                    purpose=record["purpose"],
                    created_at=datetime.fromisoformat(record["created_at"]) if record["created_at"] else None,
                    prev_hash=record["prev_hash"],
                    hash=record["hash"],
                    global_key=record.get("global_key", ""),
                    origin=record.get("origin", ""),
                    origin_seq=int(record.get("origin_seq", 0) or 0),
                    origin_hash=record.get("origin_hash", ""),
                )

    def read_from(self, db_id: str, after_seq: int) -> Iterator[DeletionEntry]:
        # NOTE (batch-2 patch 4, task 3): deliberately a linear scan of the
        # file -- the callers (export, replay, verify) need the entries'
        # full payloads, which only a parse produces; the position index
        # serves the per-entry lookups, not this.  O(file) per call.
        path = self._path_for(db_id)
        if not path.exists():
            return
        for entry in self._iter_entries(path):
            if entry.seq > after_seq:
                yield entry

    def has_genesis(self, db_id: str) -> bool:
        path = self._path_for(db_id)
        if not path.exists():
            return False
        return any(e.scope == "genesis" for e in self._iter_entries(path))

    def genesis_entries(self) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        if not self.directory.exists():
            return out
        for path in sorted(self.directory.glob("*.deletions.jsonl")):
            for entry in self._iter_entries(path):
                if entry.scope == "genesis":
                    out.append((entry.db_id, entry.created_at))
                    break
        return out

    def head(self, db_id: str) -> tuple[int, str]:
        self._index_for(db_id)  # refresh if the file changed on disk
        tail = self._index_tail.get(db_id)
        return tail if tail is not None else (0, GENESIS_HASH)

    def checkpoint_seq(self, db_id: str) -> int:
        self._index_for(db_id)  # build/refresh if the file changed
        return self._index_checkpoint.get(db_id, 0)

    def chain_hash_at(self, db_id: str, seq: int) -> str:
        if seq <= 0:
            return GENESIS_HASH
        entry = self._index_for(db_id).get(seq)
        return entry[0] if entry is not None else GENESIS_HASH

    def has_tombstone(self, db_id: str, seq: int, entry_hash: str) -> bool:
        entry = self._index_for(db_id).get(seq)
        return entry is not None and entry_hash in (entry[0], entry[1])

    def prune(self, db_id: str, before_seq: int, **_kwargs) -> int:
        """Rewrite the file: the header, the genesis, a CHECKPOINT entry
        at *before_seq* carrying the chain hash of the last pruned entry,
        then every entry above it.  The chain continues from the
        checkpoint's recorded hash; startup verification stays whole.
        Returns the number of entries removed."""
        import os

        self._index_invalidate(db_id)  # the file is about to be rewritten

        path = self._path_for(db_id)
        if not path.exists():
            return 0
        with self._locked(path) as fh:
            entries = [e for e in self._iter_entries(path) if e.scope != "genesis"]
            genesis_line = None
            with open(path, "rb") as rf:
                for line in rf:
                    record = json.loads(line)
                    if record.get("scope") == "genesis":
                        genesis_line = line
                        break
            if genesis_line is None:
                return 0  # no lineage record: never prune a foreign file
            genesis_hash = json.loads(genesis_line)["hash"]
            if any(e.scope == "checkpoint" and e.seq >= before_seq for e in entries):
                return 0  # already compacted through this point (idempotent)
            keep = [e for e in entries if e.seq > before_seq]
            chain_at = next((e.hash for e in entries if e.seq == before_seq), None)
            if chain_at is None:
                return 0  # nothing at or below before_seq to prune through
            # the count reports DELETED entries: a subsumed checkpoint is
            # rewritten, not removed
            removed = sum(1 for e in entries
                          if e.seq <= before_seq and e.scope != "checkpoint")
            if removed <= 0 and not any(e.scope == "checkpoint" for e in entries):
                return 0
            checkpoint = DeletionEntry(
                db_id=db_id, seq=before_seq, tenant_id="", app_id="", user_id="",
                scope="checkpoint", scope_key="", fingerprints=[],
                generation=0, purpose=f"pruned through seq {before_seq}",
                # prev continues the kept chain (the genesis); the hash is
                # the chain hash AT the pruned-through point -- a recorded
                # fact, not recomputable (its payload is gone)
                prev_hash=genesis_hash, hash=chain_at,
                created_at=datetime.utcnow(),
            )
            fh.seek(0)
            fh.truncate()
            fh.seek(0, os.SEEK_END)
            for line in (genesis_line,
                         (json.dumps(self._line(checkpoint), sort_keys=True,
                                     ensure_ascii=False) + "\n").encode()):
                fh.write(line)
            for e in keep:
                fh.write((json.dumps(self._line(e), sort_keys=True,
                                     ensure_ascii=False) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())
        return removed

    def verify_chain(self, db_id: str) -> None:
        path = self._path_for(db_id)
        if not path.exists():
            return
        super().verify_chain(db_id)
        self._verified.add(db_id)


# ---------------------------------------------------------------------------
# Cloud backend: a SEPARATE PostgreSQL database (its own connection string,
# outside any pg_dump of the main DB).  The app role gets INSERT / SELECT
# only -- the log is append-only at the database level.
# ---------------------------------------------------------------------------


class PgDeletionLog(DeletionLog):
    """The log table lives in its own database (``MM_DELETION_LOG_URL``);
    the main DB's pg_dump never contains it, so restoring the main DB can
    never take the log back in time.  Appends take a transaction-level
    advisory lock (in the LOG database) so the hash chain cannot
    interleave."""

    def __init__(self, url: str | Engine, maintenance_url: str | None = None):
        self._engine: Engine = (
            url if isinstance(url, Engine) else create_engine(url)
        )
        # compaction (prune) DELETES rows: the app role holds INSERT/SELECT
        # only, so the prune runs on a separate maintenance connection
        self._maintenance_engine: Engine | None = (
            create_engine(maintenance_url) if maintenance_url else None
        )
        self._verified: set[str] = set()

    def _ensure_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS mm_deletion_log ("
                " db_id CHAR(32) NOT NULL,"
                " seq BIGINT GENERATED BY DEFAULT AS IDENTITY,"
                " tenant_id VARCHAR(64) NOT NULL,"
                " app_id VARCHAR(64) NOT NULL,"
                " user_id VARCHAR(64) NOT NULL,"
                " scope VARCHAR(16) NOT NULL,"
                " scope_key VARCHAR(128) NOT NULL DEFAULT '',"
                " fingerprints_json TEXT NOT NULL DEFAULT '[]',"
                " generation INTEGER NOT NULL DEFAULT 0,"
                " purpose VARCHAR(64) NOT NULL DEFAULT '',"
                " prev_hash CHAR(64) NOT NULL,"
                " hash CHAR(64) NOT NULL,"
                " created_at TIMESTAMP NOT NULL DEFAULT now(),"
                " global_key VARCHAR(128) NOT NULL DEFAULT '',"
                " origin CHAR(32) NOT NULL DEFAULT '',"
                " origin_seq BIGINT NOT NULL DEFAULT 0,"
                " PRIMARY KEY (db_id, seq))"
            ))
            # a log database created before batch 2: add the sync columns
            for ddl in (
                "ALTER TABLE mm_deletion_log ADD COLUMN IF NOT EXISTS "
                "global_key VARCHAR(128) NOT NULL DEFAULT ''",
                "ALTER TABLE mm_deletion_log ADD COLUMN IF NOT EXISTS "
                "origin CHAR(32) NOT NULL DEFAULT ''",
                "ALTER TABLE mm_deletion_log ADD COLUMN IF NOT EXISTS "
                "origin_seq BIGINT NOT NULL DEFAULT 0",
                "ALTER TABLE mm_deletion_log ADD COLUMN IF NOT EXISTS "
                "origin_hash VARCHAR(64) NOT NULL DEFAULT ''",
            ):
                conn.execute(text(ddl))

    def _insert(self, conn, entry: DeletionEntry) -> None:
        conn.execute(text(
            "INSERT INTO mm_deletion_log (db_id, tenant_id, app_id, user_id, "
            "scope, scope_key, fingerprints_json, generation, purpose, "
            "prev_hash, hash, created_at, global_key, origin, origin_seq, "
            "origin_hash) "
            "VALUES (:db, :tenant, :app, :user, "
            ":scope, :scope_key, :fp, :generation, :purpose, :prev, :hash, "
            ":created, :global_key, :origin, :origin_seq, :origin_hash)"
        ), {
            "db": entry.db_id, "tenant": entry.tenant_id, "app": entry.app_id,
            "user": entry.user_id, "scope": entry.scope,
            "scope_key": entry.scope_key,
            "fp": json.dumps(entry.fingerprints, ensure_ascii=False),
            "generation": entry.generation, "purpose": entry.purpose,
            "prev": entry.prev_hash, "hash": entry.hash,
            "created": entry.created_at,
            "global_key": entry.global_key, "origin": entry.origin,
            "origin_seq": entry.origin_seq, "origin_hash": entry.origin_hash,
        })

    def append(self, entry: DeletionEntry) -> DeletionEntry:
        self._ensure_schema()
        if not self.has_genesis(entry.db_id):
            raise DeletionLogUnavailable(
                f"cannot append: no genesis entry exists in the deletion "
                f"log table for database {entry.db_id[:8]}...  reconcile/"
                "adopt first; deletions stay refused (fail-closed).")
        entry.created_at = entry.created_at or datetime.utcnow()
        with self._engine.begin() as conn:
            # serialise appends per database life so prev_hash is exact
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
                         {"k": f"deletion-log:{entry.db_id}"})
            row = conn.execute(text(
                "SELECT seq, hash FROM mm_deletion_log WHERE db_id = :db "
                "ORDER BY seq DESC LIMIT 1"
            ), {"db": entry.db_id}).first()
            entry.seq = (row[0] if row else 0) + 1
            entry.prev_hash = row[1] if row else GENESIS_HASH
            entry.hash = entry.compute_hash()
            self._insert(conn, entry)
        return entry

    def read_from(self, db_id: str, after_seq: int) -> Iterator[DeletionEntry]:
        self._ensure_schema()
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT seq, tenant_id, app_id, user_id, scope, scope_key, "
                "fingerprints_json, generation, purpose, prev_hash, hash, created_at, "
                "global_key, origin, origin_seq, origin_hash "
                "FROM mm_deletion_log WHERE db_id = :db AND seq > :after ORDER BY seq"
            ), {"db": db_id, "after": after_seq}).all()
        for row in rows:
            yield DeletionEntry(
                db_id=db_id, seq=int(row[0]), tenant_id=row[1], app_id=row[2],
                user_id=row[3], scope=row[4], scope_key=row[5],
                fingerprints=json.loads(row[6]), generation=int(row[7]),
                purpose=row[8], prev_hash=row[9], hash=row[10],
                created_at=row[11],
                global_key=(row[12] or "").strip(),
                origin=(row[13] or "").strip(),
                origin_seq=int(row[14] or 0),
                origin_hash=(row[15] or "").strip(),
            )

    def head(self, db_id: str) -> tuple[int, str]:
        self._ensure_schema()
        with self._engine.connect() as conn:
            row = conn.execute(text(
                "SELECT seq, hash FROM mm_deletion_log WHERE db_id = :db "
                "ORDER BY seq DESC LIMIT 1"
            ), {"db": db_id}).first()
        if row is None:
            return 0, GENESIS_HASH
        return int(row[0]), row[1]

    def has_genesis(self, db_id: str) -> bool:
        self._ensure_schema()
        with self._engine.connect() as conn:
            row = conn.execute(text(
                "SELECT 1 FROM mm_deletion_log WHERE db_id = :db AND seq = 0"
            ), {"db": db_id}).first()
        return row is not None

    def genesis_entries(self) -> list[tuple[str, object]]:
        self._ensure_schema()
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT db_id, created_at FROM mm_deletion_log "
                "WHERE scope = 'genesis' ORDER BY created_at, db_id")).all()
        return [(str(r[0]), r[1]) for r in rows]

    def checkpoint_seq(self, db_id: str) -> int:
        self._ensure_schema()
        with self._engine.connect() as conn:
            row = conn.execute(text(
                "SELECT max(seq) FROM mm_deletion_log "
                "WHERE db_id = :db AND scope = 'checkpoint'"), {"db": db_id}).first()
        return int(row[0] or 0) if row and row[0] is not None else 0

    def chain_hash_at(self, db_id: str, seq: int) -> str:
        self._ensure_schema()
        if seq <= 0:
            return GENESIS_HASH
        with self._engine.connect() as conn:
            row = conn.execute(text(
                "SELECT hash FROM mm_deletion_log "
                "WHERE db_id = :db AND seq = :s"), {"db": db_id, "s": seq}).first()
        return str(row[0]) if row else GENESIS_HASH

    def has_tombstone(self, db_id: str, seq: int, entry_hash: str) -> bool:
        self._ensure_schema()
        with self._engine.connect() as conn:
            row = conn.execute(text(
                "SELECT 1 FROM mm_deletion_log WHERE db_id = :db "
                "AND seq = :s AND (hash = :h OR origin_hash = :h) LIMIT 1"),
                {"db": db_id, "s": seq, "h": entry_hash}).first()
        return row is not None

    def prune(self, db_id: str, before_seq: int, **_kwargs) -> int:
        """Delete the entries at or below *before_seq* (never the genesis)
        and write the checkpoint row that carries the chain hash at that
        point.  The DELETE runs on the MAINTENANCE connection (the app role
        has no DELETE -- the log is append-only for it); the checkpoint is
        inserted there too, in the same transaction.  Returns the number
        of entries removed."""
        if self._maintenance_engine is None:
            raise RuntimeError(
                "the deletion log cannot be compacted: no maintenance "
                "connection is configured (deletion_log.maintenance_url) -- "
                "the app role has no DELETE on the log table")
        if self.checkpoint_seq(db_id) >= before_seq:
            return 0  # already compacted through this point (idempotent)
        self._ensure_schema()
        engine = self._maintenance_engine
        with engine.begin() as conn:
            chain = conn.execute(text(
                "SELECT hash FROM mm_deletion_log "
                "WHERE db_id = :db AND seq = :s"), {"db": db_id, "s": before_seq}).first()
            if chain is None:
                return 0
            # the checkpoint's prev continues the KEPT chain (the genesis);
            # its own hash is the recorded chain hash at the pruned point
            genesis = conn.execute(text(
                "SELECT hash FROM mm_deletion_log "
                "WHERE db_id = :db AND seq = 0"), {"db": db_id}).first()
            prev = genesis[0] if genesis else chain[0]
            already = conn.execute(text(
                "SELECT 1 FROM mm_deletion_log "
                "WHERE db_id = :db AND seq = :s AND scope = 'checkpoint'"),
                {"db": db_id, "s": before_seq}).first()
            if already:
                return 0
            removed = conn.execute(text(
                "DELETE FROM mm_deletion_log WHERE db_id = :db "
                "AND seq BETWEEN 1 AND :s AND scope <> 'checkpoint'"),
                {"db": db_id, "s": before_seq}).rowcount
            conn.execute(text(
                "INSERT INTO mm_deletion_log (db_id, seq, tenant_id, app_id, "
                "user_id, scope, scope_key, fingerprints_json, generation, "
                "purpose, prev_hash, hash) VALUES (:db, :s, '', '', '', "
                "'checkpoint', '', '[]', 0, :purpose, :prev, :hash)"),
                {"db": db_id, "s": before_seq,
                 "purpose": f"pruned through seq {before_seq}",
                 "prev": prev, "hash": chain[0]})
        return int(removed or 0)

    def ensure_genesis(self, db_id: str, *, applied: int = 0) -> None:
        # M5 patch 4: the same first-creation proof as the file backend --
        # a missing genesis with applied > 0 means the table's history was
        # lost (rows deleted); never mint a fresh one over it.
        if not self.has_genesis(db_id) and applied > 0:
            raise DeletionLogCorrupted(
                f"the deletion-log table has no genesis entry for database "
                f"{db_id[:8]}... while the watermark records {applied} "
                "applied deletion(s): the table's history was lost or "
                "cleared.  Refusing to mint new history -- restore the "
                "log table and adopt, or retire this history deliberately.")
        if self.has_genesis(db_id):
            return
        genesis = _genesis_entry(db_id)
        genesis.seq = 0
        genesis.prev_hash = GENESIS_HASH
        genesis.hash = genesis.compute_hash()
        with self._engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
                         {"k": f"deletion-log:{db_id}"})
            if not self.has_genesis(db_id):
                # explicit seq 0 (GENERATED BY DEFAULT allows it); the
                # identity sequence continues from 1 for real entries
                conn.execute(text(
                    "INSERT INTO mm_deletion_log (db_id, seq, tenant_id, app_id, "
                    "user_id, scope, scope_key, fingerprints_json, generation, "
                    "purpose, prev_hash, hash, created_at) VALUES (:db, 0, :tenant, "
                    ":app, :user, 'genesis', '', '[]', 0, '', :prev, :hash, :created)"
                ), {"db": db_id, "tenant": genesis.tenant_id, "app": genesis.app_id,
                    "user": genesis.user_id, "prev": genesis.prev_hash,
                    "hash": genesis.hash, "created": genesis.created_at})


def new_db_id() -> str:
    return uuid.uuid4().hex


def default_directory(database_url: str) -> str:
    """The default log directory for *database_url* -- the ONE derivation
    every component (engine, worker, server) must use, so the log a
    deletion was written to is the log the replay reads:
    SQLite -> a ``deletions/`` directory next to the database file;
    everything else -> a temp directory keyed by the URL hash."""
    if database_url.startswith("sqlite:///"):
        db_path = database_url[len("sqlite:///"):]
        return str((Path(db_path).parent / "deletions").resolve())
    import hashlib
    import tempfile

    from sqlalchemy.engine import make_url

    try:
        # hash the SANITIZED form: str(engine.url) hides the password, and
        # the engine (raw string) and the worker (bind.url) must agree
        sanitized = make_url(database_url).render_as_string(hide_password=True)
    except Exception:
        sanitized = database_url
    digest = hashlib.sha1(sanitized.encode()).hexdigest()[:12]
    return str(Path(tempfile.gettempdir()) / f"mm-deletions-{digest}")


def deletion_log_from_config(settings, database_url: str) -> "DeletionLog | None":
    """The deletion log from configuration -- the ONE derivation and
    validation every component (engine, worker, migration, adopt) shares
    (M4 patch 2), so a migration reads the SAME log the engines write.

    *settings* is ``config.deletion_log``.  Raises ``ValueError`` on an
    invalid configuration (missing url / missing directory for a server
    database); returns ``None`` only for ``backend="none"``.
    """
    if settings.retention_days < settings.backup_retention_days:
        raise ValueError(
            f"deletion_log.retention_days ({settings.retention_days}) is "
            f"shorter than backup_retention_days ({settings.backup_retention_days}): "
            "a log pruned before the backups it protects can no longer replay "
            "them -- raise the retention or lower the backup retention")
    if settings.backend == "none":
        return None
    if settings.backend == "postgres":
        url = settings.url
        if not url:
            raise ValueError("deletion_log.backend=postgres requires deletion_log.url")
        return PgDeletionLog(url, maintenance_url=settings.maintenance_url or None)
    if settings.directory:
        return FileDeletionLog(settings.directory)
    if not database_url.startswith(
            ("postgresql", "mysql", "mssql", "oracle")):
        # embedded single-tenant SQLite (file or in-memory) -- the
        # URL-derived directory is unambiguous (a file database's log
        # lives next to it).  Anything that is not a known SERVER scheme
        # takes the derived default (in-memory test engines, injected-
        # engine placeholders); only real server deployments must say
        # where the log lives.
        return FileDeletionLog(default_directory(database_url))
    raise ValueError(
        "deletion_log.backend=file with a non-SQLite database requires "
        "deletion_log.directory (the URL-derived default is only for "
        "embedded single-tenant SQLite)")
