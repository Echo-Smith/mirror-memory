"""The deletion-log CREATION CREDENTIAL (M5 patch 6 §1).

Why this exists
---------------
The witness records the log's head, but a witness alone cannot answer the
question every refusal rule needs: *was this database life EVER initialised,
by anyone, at any time?*  An attacker (or an unfortunate restore) that removes
the witness AND replaces the log with a self-consistent one (fresh genesis +
valid chain) defeats a witness-only scheme: the "no witness + applied == 0 +
chain verifies" legacy branch would then bless the forged history as intact.

The credential closes that hole with a **persistent, independent record of
every database life this log store has EVER seen**:

* one append-only entry per initialised life: ``db_id``, the genesis hash,
  the creation instant, and a MAC over all of it;
* written ONCE, at first creation, in the SAME transaction-like window as the
  genesis itself (under the creation lock), and never rewritten afterwards;
* entries are never removed -- retiring a witness is an operator action on
  the witness, not on the registry.

With the registry in place the startup rules become provable:

* a log with NO credential entry was never created by this engine -> refuse;
* a credential entry whose genesis hash differs from the log's genesis ->
  the history was replaced -> refuse;
* a missing witness WITH a credential entry -> the witness store was lost ->
  refuse (an operator must reconcile explicitly);
* a missing witness with NO credential entry -> brand-new life -> safe to
  initialise.

Honest threat model
-------------------
The registry is stored NEXT TO the log (same trust domain) and MAC'd with the
witness key.  An actor who can erase log, witness, registry AND key together
still wins -- that is the documented limit of any purely local mechanism
(needs an offline anchor).  What the registry adds: erasing the witness alone
is no longer sufficient to make a forged history look new.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_log = logging.getLogger(__name__)

#: Bumped when the on-disk shape changes.
REGISTRY_VERSION = 1

REGISTRY_FILENAME = ".creation-registry.jsonl"


class RegistryError(RuntimeError):
    """The creation registry cannot be trusted (bad MAC, bad shape)."""


@dataclass(frozen=True)
class CreationRecord:
    """One initialised database life, as durably observed at creation."""

    db_id: str
    genesis_hash: str
    created_at: str


class CreationRegistry:
    """Append-only, MAC'd registry of every life this log store created."""

    def __init__(self, directory: str | Path, *, key_path: str | Path | None = None):
        self.directory = Path(directory)
        self.key_path = Path(
            key_path if key_path is not None
            else os.environ.get(
                "MM_DELETION_WITNESS_KEY",
                "~/.mirror-memory/deletion-witness.key",
            )
        ).expanduser()
        # one writer at a time within this process; cross-process writes are
        # serialised by the caller holding the log's flock
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------

    def _path(self) -> Path:
        return self.directory / REGISTRY_FILENAME

    # -- key ------------------------------------------------------------

    def _key(self, *, create: bool) -> bytes | None:
        """The MAC key, shared with the witness (one key, two artifacts)."""
        path = self.key_path
        if path.exists():
            data = path.read_bytes().strip()
            if not data:
                raise RegistryError(f"registry key file {path} is empty")
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
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(key, canonical.encode(), hashlib.sha256).hexdigest()

    # -- write ----------------------------------------------------------

    def record(self, db_id: str, genesis_hash: str) -> None:
        """Durably append one creation record (fsync file + directory).

        Cross-process safe: the duplicate check AND the append run under an
        flock on the registry file itself, so racing creators serialise --
        the loser observes the winner's line and appends nothing.  Idempotent
        for the same (db_id, genesis_hash) pair."""
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._path()
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    existing = self._find(db_id)
                    if existing is not None:
                        if existing.genesis_hash != genesis_hash:
                            raise RegistryError(
                                f"creation registry already records db_id "
                                f"{db_id[:8]}... with a DIFFERENT genesis "
                                "hash: the history was replaced.  Refusing.")
                        return              # same life, already recorded
                    key = self._key(create=True)
                    payload = {
                        "db_id": db_id,
                        "genesis_hash": genesis_hash,
                        "created_at": datetime.utcnow().isoformat(),
                    }
                    line = json.dumps({
                        "version": REGISTRY_VERSION,
                        "payload": payload,
                        "mac": self._mac(key, payload),
                    }, sort_keys=True) + "\n"
                    os.lseek(fd, 0, os.SEEK_END)
                    os.write(fd, line.encode())
                    os.fsync(fd)
                    self._fsync_dir()
                finally:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # -- read -----------------------------------------------------------

    def _iter_records(self) -> list[CreationRecord]:
        path = self._path()
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return []
        key = self._key(create=False)
        records: list[CreationRecord] = []
        for lineno, line in enumerate(raw.split(b"\n"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RegistryError(
                    f"creation registry line {lineno} is not valid JSON: "
                    f"{exc}.  The registry was tampered with or torn mid-"
                    "write; refusing to trust it.") from exc
            payload = record.get("payload")
            mac = record.get("mac")
            if not isinstance(payload, dict) or not isinstance(mac, str):
                raise RegistryError(
                    f"creation registry line {lineno} is malformed")
            if key is None:
                raise RegistryError(
                    "a creation registry exists but the witness key is "
                    "missing: its records cannot be authenticated")
            if not hmac.compare_digest(self._mac(key, payload), mac):
                raise RegistryError(
                    f"creation registry line {lineno} failed its MAC check: "
                    "the registry was modified outside this engine")
            records.append(CreationRecord(
                db_id=payload["db_id"],
                genesis_hash=payload.get("genesis_hash", ""),
                created_at=payload.get("created_at", ""),
            ))
        return records

    def _find(self, db_id: str) -> CreationRecord | None:
        for record in self._iter_records():
            if record.db_id == db_id:
                return record
        return None

    def find(self, db_id: str) -> CreationRecord | None:
        """The creation record for *db_id*, authenticated.  ``None`` when this
        store never created that life.  Raises RegistryError on tampering."""
        with self._lock:
            return self._find(db_id)

    def iter_records(self) -> list[CreationRecord]:
        """Every authenticated creation record (records, not id strings)."""
        with self._lock:
            return list(self._iter_records())

    def created_lives(self) -> list[str]:
        """Every db_id this store ever created (authenticated)."""
        with self._lock:
            return [r.db_id for r in self._iter_records()]

    def _fsync_dir(self) -> None:
        try:
            fd = os.open(self.directory, os.O_RDONLY)
        except OSError as exc:            # pragma: no cover - exotic FS
            _log.warning("creation registry: directory fsync failed for %s "
                         "(%s); the record may not survive a power loss",
                         self.directory, exc)
            return
        try:
            os.fsync(fd)
        except OSError as exc:            # pragma: no cover - not supported
            _log.warning("creation registry: directory fsync failed for %s "
                         "(%s); the record may not survive a power loss",
                         self.directory, exc)
        finally:
            os.close(fd)
