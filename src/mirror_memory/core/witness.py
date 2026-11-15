"""The deletion-log WITNESS: an independent, durable record of where a
database life's deletion history stands (M5 patch 5).

Why this exists
---------------
The log file can be truncated, rolled back or replaced, and a restored main
database brings its OLD watermark back -- very often ``applied_seq == 0``.
Neither the watermark nor the mere existence of a log file proves that the
history is COMPLETE.  A sidecar that only records "some log existed" (patch
4's ``.bound``) cannot tell a healthy log from one whose trailing lines were
removed.  This witness records the log's **head** (tail seq + tail hash),
the genesis hash and the compaction checkpoint seq, so a startup can ask the
only question that matters: *is the log I am about to trust at least as
complete as the last state I durably observed?*

Design (deliberately conservative)
----------------------------------
* One small JSON file per database life, in a directory separate from the
  log directory, written ATOMICALLY (tmp + fsync + os.replace + directory
  fsync), so a torn write can never be mistaken for a valid witness.
* The file carries a keyed MAC over its payload.  The key lives in a
  ``.witness.key`` file (0600) created on first use.  A witness that cannot
  be authenticated is NOT trusted -- startup refuses rather than guessing.
* Ordering on every append: log bytes are fsynced FIRST, the witness AFTER.
  A crash between the two leaves ``witness.tail < log.head``, which is
  recoverable (the log is ahead, nothing was lost).  The reverse ordering
  would allow ``witness.tail > log.head`` after a crash, which looks exactly
  like a truncation -- so it is never used.

Honest threat model
-------------------
The witness defeats: a truncated/rolled-back log file, a restored main DB
with an old watermark, a deleted log file, and an accidental/malicious edit
of the witness that cannot reproduce the MAC.

It does NOT defeat: an attacker with write access to BOTH the log directory
and the witness directory (and the key file) deleting or rolling back both
together.  No local first-party mechanism can distinguish "history never
existed" from "history was perfectly erased" without an external, offline
anchor.  That limit is documented in ``docs/integration.md`` together with
the recovery preconditions; nothing here claims otherwise.
"""

from __future__ import annotations

import contextlib
import hashlib
from datetime import datetime
import hmac
import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path

#: Bumped when the on-disk shape changes; a witness of another version is
#: treated as untrusted (refused) rather than reinterpreted.
WITNESS_VERSION = 1

#: The MAC key file, created on first use with 0600.
KEY_FILENAME = ".witness.key"

#: Where the witness MAC key lives (overridable; 0600 file).  Outside the
#: log/witness trees by default so a log-directory restore cannot take it.
WITNESS_KEY_ENV = "MM_DELETION_WITNESS_KEY"
DEFAULT_KEY_PATH = "~/.mirror-memory/deletion-witness.key"


_log = logging.getLogger(__name__)


def _log_fsync_problem(directory: Path, exc: OSError) -> None:
    """A directory fsync that failed: the rename may not be crash-durable."""
    _log.warning(
        "deletion-log witness: directory fsync failed for %s (%s).  The "
        "witness update may not survive a power loss; the LOG is already "
        "durable, so a crash can only leave the witness BEHIND the log "
        "(recoverable), never ahead of it.", directory, exc)


def _relation_exists(engine, name: str) -> bool:
    """True when ``public.<name>`` already exists (cheap catalog lookup)."""
    from sqlalchemy import text as _text

    with engine.connect() as conn:
        return conn.execute(
            _text("SELECT to_regclass(:n)"),
            {"n": f"public.{name}"},
        ).scalar() is not None


def _create_if_absent(engine, name: str, ddl: str) -> None:
    """Run *ddl* only when the relation is ABSENT.

    M5 patch 10 §4: the runtime role deliberately holds no DDL, but
    PostgreSQL checks CREATE privilege for ``CREATE TABLE IF NOT EXISTS``
    even when the table already exists.  Skipping the statement when the
    catalog already has the relation is what lets a restricted runtime login
    run the real log/witness chain.  A genuinely missing table still fails
    closed with the original error (the owner must provision).
    """
    from sqlalchemy import text as _text

    if _relation_exists(engine, name):
        return
    from sqlalchemy.exc import ProgrammingError

    try:
        with engine.begin() as conn:
            conn.execute(_text(ddl))
    except ProgrammingError:
        # lost a race to another provisioner: fine if it exists now
        if not _relation_exists(engine, name):
            raise


class WitnessError(RuntimeError):
    """The witness cannot be trusted (missing key, bad MAC, bad shape)."""


class WitnessAhead(RuntimeError):
    """The witness records a state the log cannot reach.

    This is the truncation/rollback signal: the history on disk stopped
    short of a state that was already durably observed, so entries were lost.
    """


@dataclass(frozen=True)
class WitnessState:
    """One observed state of a database life's deletion history."""

    db_id: str
    genesis_hash: str
    tail_seq: int
    tail_hash: str
    checkpoint_seq: int
    updated_at: str


class DeletionLogWitness:
    """Durable, MAC-authenticated head-of-history witness for one log store."""

    def __init__(self, directory: str | Path, *, key_path: str | Path | None = None):
        self.directory = Path(directory)
        # The MAC key is deliberately NOT stored next to the witness: an
        # attacker (or a sloppy restore) that can write the witness directory
        # must not also be able to read the key and re-forge the MAC.  Default
        # is outside the log tree entirely (overridable for deployments that
        # mount a proper secret store).
        self.key_path = Path(
            key_path if key_path is not None
            else os.environ.get(WITNESS_KEY_ENV, DEFAULT_KEY_PATH)
        ).expanduser()
        # Serialises witness writes across THREADS in this process (processes
        # are serialised by the log's own file lock, which the caller holds).
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------

    def _path(self, db_id: str) -> Path:
        return self.directory / f"{db_id}.witness.json"

    def _key_path(self) -> Path:
        return self.key_path

    def exists(self, db_id: str) -> bool:
        return self._path(db_id).exists()

    # -- key management ------------------------------------------------

    def _key(self, *, create: bool) -> bytes | None:
        """The MAC key.  Created (once, atomically, 0600) when *create*."""
        path = self._key_path()
        if path.exists():
            try:
                data = path.read_bytes().strip()
            except OSError as exc:  # pragma: no cover - unreadable key file
                raise WitnessError(f"witness key unreadable: {exc}") from exc
            if not data:
                raise WitnessError("witness key file is empty")
            return data
        if not create:
            return None
        # the key file may live OUTSIDE the witness directory: create its own
        # parent and stage the temp file THERE (same filesystem for the link)
        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32).encode()
        tmp = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, key + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(tmp, path)          # atomic publish; loser re-reads
        except FileExistsError:
            key = path.read_bytes().strip()
        except OSError:                 # filesystems without hard links
            if not path.exists():
                os.replace(tmp, path)
                key = path.read_bytes().strip()
            else:
                key = path.read_bytes().strip()
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
        return key

    # -- MAC -----------------------------------------------------------

    @staticmethod
    def _mac(key: bytes, payload: dict) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(key, canonical.encode(), hashlib.sha256).hexdigest()

    # -- read ----------------------------------------------------------

    def read(self, db_id: str) -> WitnessState | None:
        """The authenticated witness for *db_id*, or ``None`` when absent.

        Raises :class:`WitnessError` when a witness file exists but cannot be
        trusted -- callers must refuse rather than fall back."""
        path = self._path(db_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WitnessError(
                f"witness for {db_id[:8]}... is not valid JSON: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise WitnessError(f"witness for {db_id[:8]}... is not an object")
        if record.get("version") != WITNESS_VERSION:
            raise WitnessError(
                f"witness for {db_id[:8]}... has version "
                f"{record.get('version')!r}, expected {WITNESS_VERSION}"
            )
        payload = record.get("payload")
        mac = record.get("mac")
        if not isinstance(payload, dict) or not isinstance(mac, str):
            raise WitnessError(f"witness for {db_id[:8]}... is malformed")
        key = self._key(create=False)
        if key is None:
            raise WitnessError(
                f"witness for {db_id[:8]}... exists but the witness key is "
                "missing: the witness cannot be authenticated"
            )
        if not hmac.compare_digest(self._mac(key, payload), mac):
            raise WitnessError(
                f"witness for {db_id[:8]}... failed its MAC check: the file "
                "was modified outside this engine"
            )
        if payload.get("db_id") != db_id:
            raise WitnessError(
                f"witness for {db_id[:8]}... records a different database "
                f"({str(payload.get('db_id'))[:8]}...)"
            )
        return WitnessState(
            db_id=payload["db_id"],
            genesis_hash=payload.get("genesis_hash", ""),
            tail_seq=int(payload.get("tail_seq", 0)),
            tail_hash=payload.get("tail_hash", ""),
            checkpoint_seq=int(payload.get("checkpoint_seq", 0)),
            updated_at=payload.get("updated_at", ""),
        )

    # -- write ---------------------------------------------------------

    def record(self, state: WitnessState, *, key: bytes | None = None) -> None:
        """Durably publish *state* (atomic replace + directory fsync)."""
        key = key or self._key(create=True)
        payload = {
            "db_id": state.db_id,
            "genesis_hash": state.genesis_hash,
            "tail_seq": state.tail_seq,
            "tail_hash": state.tail_hash,
            "checkpoint_seq": state.checkpoint_seq,
            "updated_at": state.updated_at,
        }
        record = {
            "version": WITNESS_VERSION,
            "payload": payload,
            "mac": self._mac(key, payload),
        }
        with self._lock:
            self._record_locked(record, state.db_id, key)

    def _record_locked(self, record: dict, db_id: str, key: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(db_id)
        tmp = self.directory / (
            f".{db_id}.witness.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
            try:
                os.write(fd, json.dumps(record, sort_keys=True).encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)       # atomic publish
            self._fsync_dir()
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)

    def retire(self, db_id: str) -> None:
        """Remove the witness for *db_id* (explicit operator action only)."""
        with contextlib.suppress(FileNotFoundError):
            self._path(db_id).unlink()
            self._fsync_dir()

    def _fsync_dir(self) -> None:
        """Make a rename/link durable: fsync the DIRECTORY entry.

        A failure here means the rename may not survive a power loss, i.e. the
        witness could silently be OLDER than the log (safe direction, but not
        what we want to *claim*).  It is therefore logged loudly rather than
        swallowed; the append already succeeded, so raising would be worse."""
        try:
            fd = os.open(self.directory, os.O_RDONLY)
        except OSError as exc:           # pragma: no cover - exotic FS
            _log_fsync_problem(self.directory, exc)
            return
        try:
            os.fsync(fd)
        except OSError as exc:           # pragma: no cover - not supported
            _log_fsync_problem(self.directory, exc)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# PostgreSQL backend witness: a table in the LOG database (never in a
# pg_dump of the MAIN database), MAC'd with a key from a FILE so that
# write access to the table alone cannot forge a witness.
# ---------------------------------------------------------------------------

def witness_key_path() -> Path:
    return Path(os.environ.get(WITNESS_KEY_ENV, DEFAULT_KEY_PATH)).expanduser()


class PgDeletionLogWitness:
    """The witness for a PostgreSQL deletion log (``mm_deletion_log_witness``).

    Rationale for the placement: the log lives in ITS OWN database, so a
    restore of the main database cannot roll it back.  The witness must not
    be weaker than the log, so it lives in that same separate database -- but
    in a different TABLE, and its rows are authenticated with a key from a
    FILE.  An attacker who can write only the table (the app role cannot even
    UPDATE it) cannot produce a valid MAC.
    """

    TABLE = "mm_deletion_log_witness"

    def __init__(self, engine_getter, *, _ensure=None):
        self._engine_getter = engine_getter
        self._ensure_schema = _ensure
        self._key_path = witness_key_path()

    # -- key (file, not the database) ----------------------------------

    def _key(self, *, create: bool) -> bytes | None:
        path = self._key_path
        if path.exists():
            data = path.read_bytes().strip()
            if not data:
                raise WitnessError(f"witness key file {path} is empty")
            return data
        if not create:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32).encode()
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return path.read_bytes().strip()
        try:
            os.write(fd, key + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        return key

    @staticmethod
    def _mac(key: bytes, payload: dict) -> str:
        return DeletionLogWitness._mac(key, payload)

    # -- schema --------------------------------------------------------

    def _ensure(self) -> None:
        engine = self._engine_getter()
        _create_if_absent(
            engine, self.TABLE,
            f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
            " db_id CHAR(32) PRIMARY KEY,"
            " genesis_hash CHAR(64) NOT NULL,"
            " tail_seq BIGINT NOT NULL,"
            " tail_hash CHAR(64) NOT NULL,"
            " checkpoint_seq BIGINT NOT NULL DEFAULT 0,"
            " updated_at VARCHAR(64) NOT NULL,"
            " mac CHAR(64) NOT NULL)")

    # -- read / write --------------------------------------------------

    def exists(self, db_id: str) -> bool:
        from sqlalchemy import text

        self._ensure()
        engine = self._engine_getter()
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT 1 FROM {self.TABLE} WHERE db_id = :db"),
                {"db": db_id}).first()
        return row is not None

    def read(self, db_id: str) -> WitnessState | None:
        from sqlalchemy import text

        self._ensure()
        engine = self._engine_getter()
        with engine.connect() as conn:
            row = conn.execute(text(
                f"SELECT genesis_hash, tail_seq, tail_hash, checkpoint_seq, "
                f"updated_at, mac FROM {self.TABLE} WHERE db_id = :db"),
                {"db": db_id}).first()
        if row is None:
            return None
        payload = {
            "db_id": db_id,
            "genesis_hash": row[0],
            "tail_seq": int(row[1]),
            "tail_hash": row[2],
            "checkpoint_seq": int(row[3]),
            "updated_at": str(row[4]),
        }
        key = self._key(create=False)
        if key is None:
            raise WitnessError(
                f"a witness row exists for {db_id[:8]}... but the witness key "
                f"file {self._key_path} is missing: cannot authenticate it")
        if not hmac.compare_digest(self._mac(key, payload), row[5]):
            raise WitnessError(
                f"witness row for {db_id[:8]}... failed its MAC check: it was "
                "modified outside this engine")
        return WitnessState(**payload)

    def record(self, state: WitnessState, *, key: bytes | None = None) -> None:
        from sqlalchemy import text

        key = key or self._key(create=True)
        self._ensure()
        payload = {
            "db_id": state.db_id,
            "genesis_hash": state.genesis_hash,
            "tail_seq": state.tail_seq,
            "tail_hash": state.tail_hash,
            "checkpoint_seq": state.checkpoint_seq,
            "updated_at": state.updated_at,
        }
        mac = self._mac(key, payload)
        engine = self._engine_getter()
        with engine.begin() as conn:
            # serialize witness writers per db_id (its own advisory lock, in
            # the SAME database as the witness row -- the log's lock guards
            # the log, not this table)
            conn.execute(text(
                "SELECT pg_advisory_xact_lock(hashtext(:k))"),
                {"k": f"deletion-witness:{state.db_id}"})
            # MONOTONIC compare-and-set: the row is only advanced (or left
            # equal).  A lagging writer whose head is behind the recorded one
            # is REFUSED rather than allowed to regress the witness.
            updated = conn.execute(text(
                f"UPDATE {self.TABLE} SET genesis_hash = :g, tail_seq = :ts, "
                f"tail_hash = :th, checkpoint_seq = :cs, updated_at = :ua, "
                f"mac = :mac WHERE db_id = :db AND "
                f"(tail_seq < :ts OR (tail_seq = :ts AND tail_hash = :th))"),
                {"g": state.genesis_hash, "ts": state.tail_seq,
                 "th": state.tail_hash, "cs": state.checkpoint_seq,
                 "ua": state.updated_at, "mac": mac, "db": state.db_id}).rowcount
            if not updated:
                existing = conn.execute(text(
                    f"SELECT tail_seq, tail_hash FROM {self.TABLE} "
                    f"WHERE db_id = :db"), {"db": state.db_id}).first()
                if existing is None:
                    conn.execute(text(
                        f"INSERT INTO {self.TABLE} (db_id, genesis_hash, "
                        f"tail_seq, tail_hash, checkpoint_seq, updated_at, mac) "
                        f"VALUES (:db, :g, :ts, :th, :cs, :ua, :mac)"),
                        {"db": state.db_id, "g": state.genesis_hash,
                         "ts": state.tail_seq, "th": state.tail_hash,
                         "cs": state.checkpoint_seq, "ua": state.updated_at,
                         "mac": mac})
                elif int(existing[0]) > state.tail_seq:
                    raise WitnessAhead(
                        f"cannot record the deletion-log witness for "
                        f"{state.db_id[:8]}...: the stored witness is at seq "
                        f"{existing[0]} but this writer only reached seq "
                        f"{state.tail_seq} -- recording it would REGRESS the "
                        "witness.  This instance is behind; it must catch up "
                        "before writing.")
                # equal seq + equal hash: idempotent, nothing to do

    def retire(self, db_id: str) -> None:
        from sqlalchemy import text

        self._ensure()
        engine = self._engine_getter()
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {self.TABLE} WHERE db_id = :db"),
                         {"db": db_id})


class PgCreationRegistry:
    """The PostgreSQL creation credential: one row per life this log store
    created, in the LOG database (``mm_creation_registry``), MAC'd with the
    same key file as the witness.

    Appended once at first creation; never rewritten or deleted by the
    engine.  It is what makes 'no witness' decidable on the postgres
    backend -- a forged replacement log has a DIFFERENT genesis hash than
    the credential records."""

    TABLE = "mm_creation_registry"

    def __init__(self, engine_getter, *, _ensure=None):
        self._engine_getter = engine_getter
        self._ensure_schema = _ensure
        self._witness = PgDeletionLogWitness(engine_getter, _ensure=_ensure)

    @property
    def key_path(self):
        return self._witness.key_path

    def _key(self, *, create: bool):
        return self._witness._key(create=create)

    @staticmethod
    def _mac(key, payload):
        return PgDeletionLogWitness._mac(key, payload)

    def _ensure(self):
        engine = self._engine_getter()
        _create_if_absent(
            engine, self.TABLE,
            f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
            " db_id CHAR(32) PRIMARY KEY,"
            " genesis_hash CHAR(64) NOT NULL,"
            " created_at VARCHAR(64) NOT NULL,"
            " mac CHAR(64) NOT NULL)")

    def record(self, db_id: str, genesis_hash: str) -> None:
        from sqlalchemy import text

        key = self._key(create=True)
        self._ensure()
        payload = {"db_id": db_id, "genesis_hash": genesis_hash,
                   "created_at": datetime.utcnow().isoformat()}
        mac = self._mac(key, payload)
        engine = self._engine_getter()
        with engine.begin() as conn:
            conn.execute(text(
                f"INSERT INTO {self.TABLE} (db_id, genesis_hash, created_at, "
                f"mac) VALUES (:db, :g, :ua, :mac) "
                f"ON CONFLICT (db_id) DO NOTHING"),
                {"db": db_id, "g": genesis_hash,
                 "ua": payload["created_at"], "mac": mac})

    def find(self, db_id: str):
        from sqlalchemy import text

        self._ensure()
        engine = self._engine_getter()
        with engine.connect() as conn:
            row = conn.execute(text(
                f"SELECT genesis_hash, created_at, mac FROM {self.TABLE} "
                f"WHERE db_id = :db"), {"db": db_id}).first()
        if row is None:
            return None
        payload = {"db_id": db_id, "genesis_hash": row[0],
                   "created_at": str(row[1])}
        key = self._key(create=False)
        if key is None:
            raise WitnessError(
                f"a creation credential exists for {db_id[:8]}... but the "
                f"witness key file {self.key_path} is missing: it cannot be "
                "authenticated")
        if not hmac.compare_digest(self._mac(key, payload), row[2]):
            raise WitnessError(
                f"creation credential for {db_id[:8]}... failed its MAC "
                "check: modified outside this engine")
        from mirror_memory.core.creation_registry import CreationRecord

        return CreationRecord(
            db_id=db_id, genesis_hash=row[0], created_at=str(row[1]))
