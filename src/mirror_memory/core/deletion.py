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


class LogChainBroken(RuntimeError):
    """The log's hash chain does not verify: entries were lost, altered
    or written out of order.  Startup is refused."""


class DeletionLogStale(RuntimeError):
    """The main DB applied more log entries than the log still holds:
    the log's recovery point is OLDER than the main database.  Restore
    the log (or a newer backup of it) before starting."""


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

    def payload(self) -> dict:
        """The canonical, hashable form (everything except seq/hash)."""
        return {
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

    def compute_hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


class DeletionLog:
    """The log interface: append / read_from / head, per database life."""

    def append(self, entry: DeletionEntry) -> DeletionEntry:  # pragma: no cover - interface
        raise NotImplementedError

    def read_from(self, db_id: str, after_seq: int) -> Iterator[DeletionEntry]:  # pragma: no cover
        raise NotImplementedError

    def head(self, db_id: str) -> tuple[int, str]:  # pragma: no cover
        """(last seq, last hash); (0, GENESIS) for an empty log."""
        raise NotImplementedError

    def verify_chain(self, db_id: str) -> None:
        """Walk the whole chain; raise LogChainBroken on any mismatch."""
        prev = GENESIS_HASH
        for entry in self.read_from(db_id, 0):
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

    def _ensure_file(self, db_id: str) -> Path:
        path = self._path_for(db_id)
        if not path.exists():
            with self._locked(path) as fh:
                if path.exists():
                    return path
                fh.seek(0)
                fh.write((json.dumps({"db_id": db_id, "header": True}) + "\n").encode())
                fh.flush()
                os.fsync(fh.fileno())
        return path

    def append(self, entry: DeletionEntry) -> DeletionEntry:
        path = self._ensure_file(entry.db_id)
        with self._locked(path) as fh:
            entries = list(self._iter_entries(path))
            entry.seq = (entries[-1].seq if entries else 0) + 1
            entry.prev_hash = entries[-1].hash if entries else GENESIS_HASH
            entry.hash = entry.compute_hash()
            fh.seek(0, os.SEEK_END)
            fh.write((json.dumps(self._line(entry), sort_keys=True, ensure_ascii=False) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())
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
        }

    def _iter_entries(self, path: Path) -> Iterator[DeletionEntry]:
        with open(path, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("header"):
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
                )

    def read_from(self, db_id: str, after_seq: int) -> Iterator[DeletionEntry]:
        path = self._path_for(db_id)
        if not path.exists():
            return
        for entry in self._iter_entries(path):
            if entry.seq > after_seq:
                yield entry

    def head(self, db_id: str) -> tuple[int, str]:
        path = self._path_for(db_id)
        if not path.exists():
            return 0, GENESIS_HASH
        last: DeletionEntry | None = None
        for entry in self._iter_entries(path):
            last = entry
        if last is None:
            return 0, GENESIS_HASH
        return last.seq, last.hash

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

    def __init__(self, url: str | Engine):
        self._engine: Engine = (
            url if isinstance(url, Engine) else create_engine(url)
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
                " PRIMARY KEY (db_id, seq))"
            ))

    def append(self, entry: DeletionEntry) -> DeletionEntry:
        self._ensure_schema()
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
            conn.execute(text(
                "INSERT INTO mm_deletion_log (db_id, tenant_id, app_id, user_id, "
                "scope, scope_key, fingerprints_json, generation, purpose, "
                "prev_hash, hash, created_at) VALUES (:db, :tenant, :app, :user, "
                ":scope, :scope_key, :fp, :generation, :purpose, :prev, :hash, :created)"
            ), {
                "db": entry.db_id, "tenant": entry.tenant_id, "app": entry.app_id,
                "user": entry.user_id, "scope": entry.scope,
                "scope_key": entry.scope_key,
                "fp": json.dumps(entry.fingerprints, ensure_ascii=False),
                "generation": entry.generation, "purpose": entry.purpose,
                "prev": entry.prev_hash, "hash": entry.hash,
                "created": entry.created_at,
            })
        return entry

    def read_from(self, db_id: str, after_seq: int) -> Iterator[DeletionEntry]:
        self._ensure_schema()
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT seq, tenant_id, app_id, user_id, scope, scope_key, "
                "fingerprints_json, generation, purpose, prev_hash, hash, created_at "
                "FROM mm_deletion_log WHERE db_id = :db AND seq > :after ORDER BY seq"
            ), {"db": db_id, "after": after_seq}).all()
        for row in rows:
            yield DeletionEntry(
                db_id=db_id, seq=int(row[0]), tenant_id=row[1], app_id=row[2],
                user_id=row[3], scope=row[4], scope_key=row[5],
                fingerprints=json.loads(row[6]), generation=int(row[7]),
                purpose=row[8], prev_hash=row[9], hash=row[10],
                created_at=row[11],
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


def new_db_id() -> str:
    return uuid.uuid4().hex
