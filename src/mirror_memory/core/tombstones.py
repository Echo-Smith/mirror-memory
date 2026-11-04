"""Tombstone synchronization (M4 batch 2 tasks 1-2): the cross-device
format of a deletion, how it is exported from a local deletion log and
how a batch of them is validated before it is applied.

A tombstone names WHAT was deleted, never its content and never a local
fingerprint: ``origin`` (the db_id of the device the deletion happened
on), ``origin_seq`` (its seq in that device's log -- together the global
identity of the deletion), the tenant, the user, the scope
(belief / session / user) and the cross-device scope key -- a belief's
``global_id`` (never its local integer id: restores and other devices
reuse those), the session id, or "" for a whole-user deletion -- the
origin's deletion generation, the instant, and a hash over all of it.

Applying a tombstone (``MemoryEngine.apply_tombstones``) runs the normal
local deletion path for that scope -- local fingerprints are collected,
the local log is written, the local generation advances -- and records
``(origin, origin_seq)`` as applied.  Entries produced that way carry the
tombstone's origin, so an export never echoes them back to where they
came from.

Rejection is of the WHOLE batch, before any write: a malformed entry, a
hash that does not verify, a tenant that is not the receiver's, a user
the tenant does not know (cloud side), or an ``origin_seq`` gap -- any of
them raises :class:`TombstoneBatchRejected` with the reason, and nothing
is written: no rows, no log entry, no applied record.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime

from mirror_memory.core.deletion import DeletionEntry, DeletionLog

SCOPES = ("belief", "session", "user")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: Non-content scopes that live in the log but never travel (the birth
#: certificate and the compaction checkpoint).
_LOCAL_ONLY_SCOPES = frozenset({"genesis", "checkpoint"})


class TombstoneBatchRejected(RuntimeError):
    """The whole batch was refused; ``reason`` is one of ``format``,
    ``hash``, ``tenant``, ``user``, ``sequence``.  Nothing was written."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class TombstoneCursorPruned(RuntimeError):
    """The peer's cursor lies before the log's compaction checkpoint: the
    entries it needs are gone.  The peer must rebuild (M4 batch 2 task 4);
    an export that silently skipped them would lose deletions."""


@dataclass
class TombstoneEntry:
    origin: str
    origin_seq: int
    tenant_id: str
    app_id: str
    user_id: str
    scope: str
    scope_key: str
    generation: int
    created_at: str  # ISO 8601; a string so the hash is stable across transports
    # the hash of the ORIGIN's log entry this tombstone was exported from:
    # the stable identity that survives relays and sender rollbacks (a
    # rolled-back sender reuses origin_seq for a DIFFERENT deletion whose
    # entry hash differs)
    origin_hash: str = ""
    hash: str = ""

    def payload(self) -> dict:
        return {
            "origin": self.origin,
            "origin_seq": self.origin_seq,
            "tenant_id": self.tenant_id,
            "app_id": self.app_id,
            "user_id": self.user_id,
            "scope": self.scope,
            "scope_key": self.scope_key,
            "generation": self.generation,
            "created_at": self.created_at,
            "origin_hash": self.origin_hash,
        }

    def compute_hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def seal(self) -> "TombstoneEntry":
        self.hash = self.compute_hash()
        return self

    def to_dict(self) -> dict:
        data = self.payload()
        data["hash"] = self.hash
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TombstoneEntry":
        """Parse a transported entry; a missing or mistyped field is a
        ``format`` rejection (never a KeyError the transport swallows)."""
        if not isinstance(data, dict):
            raise TombstoneBatchRejected("format", "tombstone entry is not an object")
        required = ("origin", "origin_seq", "tenant_id", "app_id", "user_id",
                    "scope", "scope_key", "generation", "created_at", "hash",
                    "origin_hash")
        missing = [k for k in required if k not in data]
        if missing:
            raise TombstoneBatchRejected(
                "format", f"tombstone entry is missing field(s) {missing}")
        try:
            return cls(
                origin=str(data["origin"]), origin_seq=int(data["origin_seq"]),
                tenant_id=str(data["tenant_id"]), app_id=str(data["app_id"]),
                user_id=str(data["user_id"]), scope=str(data["scope"]),
                scope_key=str(data["scope_key"] or ""),
                generation=int(data["generation"]),
                created_at=str(data["created_at"]), hash=str(data["hash"]),
                origin_hash=str(data.get("origin_hash", "")),
            )
        except (TypeError, ValueError) as exc:
            raise TombstoneBatchRejected("format", f"tombstone entry has a mistyped field: {exc}")


@dataclass
class TombstoneBatch:
    """A transport batch of tombstones (M4 batch-2 patch 2): the sender's
    identity, the sender-local cursor range the entries came from, the
    CHAIN HASHES at both ends of that range, the entries, and a hash over
    all of it.

    The receiver stores (cursor, chain_hash) per sender.  A batch whose
    from_cursor continues the stored cursor but whose from_chain_hash
    differs means the SENDER ROLLED BACK (its log was restored/rewritten
    under the same seqs): the receiver rescans from 0, deduplicating via
    (origin, origin_hash) applied records, so deletions that survived on
    the receiver are kept and the sender's re-issued deletions re-apply
    onto its restored data.  An empty batch (everything excluded/skipped)
    still advances the cursor."""

    sender: str
    from_cursor: int
    to_cursor: int
    from_chain_hash: str
    to_chain_hash: str
    entries: list[TombstoneEntry]
    hash: str = ""

    def payload(self) -> dict:
        return {
            "sender": self.sender,
            "from_cursor": self.from_cursor,
            "to_cursor": self.to_cursor,
            "from_chain_hash": self.from_chain_hash,
            "to_chain_hash": self.to_chain_hash,
            "entry_hashes": [e.hash for e in self.entries],
        }

    def compute_hash(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def seal(self) -> "TombstoneBatch":
        self.hash = self.compute_hash()
        return self

    def to_dict(self) -> dict:
        data = self.payload()
        data["entries"] = [e.to_dict() for e in self.entries]
        data["hash"] = self.hash
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TombstoneBatch":
        if not isinstance(data, dict) or not all(
                k in data for k in ("sender", "from_cursor", "to_cursor",
                                    "from_chain_hash", "to_chain_hash",
                                    "entries", "hash")):
            raise TombstoneBatchRejected("format", "batch envelope is missing fields")
        entries = [TombstoneEntry.from_dict(e) for e in data["entries"]]
        batch = cls(sender=str(data["sender"]),
                    from_cursor=int(data["from_cursor"]),
                    to_cursor=int(data["to_cursor"]),
                    from_chain_hash=str(data["from_chain_hash"]),
                    to_chain_hash=str(data["to_chain_hash"]),
                    entries=entries, hash=str(data["hash"]))
        if batch.compute_hash() != batch.hash:
            raise TombstoneBatchRejected(
                "hash", "batch hash does not verify: the envelope or the entry "
                "list was altered in transit")
        return batch

    def to_dict(self) -> dict:
        data = self.payload()
        data["entries"] = [e.to_dict() for e in self.entries]
        data["hash"] = self.hash
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TombstoneBatch":
        if not isinstance(data, dict) or not all(
                k in data for k in ("sender", "from_cursor", "to_cursor",
                                    "entries", "hash")):
            raise TombstoneBatchRejected("format", "batch envelope is missing fields")
        entries = [TombstoneEntry.from_dict(e) for e in data["entries"]]
        batch = cls(sender=str(data["sender"]),
                    from_cursor=int(data["from_cursor"]),
                    to_cursor=int(data["to_cursor"]),
                    entries=entries, hash=str(data["hash"]))
        if batch.compute_hash() != batch.hash:
            raise TombstoneBatchRejected(
                "hash", "batch hash does not verify: the envelope or the entry "
                "list was altered in transit")
        return batch


def validate_entry_format(entry: TombstoneEntry) -> None:
    """Shape and value constraints of ONE entry (reason ``format``)."""
    if not _HEX32.match(entry.origin or ""):
        raise TombstoneBatchRejected("format", f"origin {entry.origin!r} is not a 32-hex db_id")
    if entry.origin_seq < 1:
        raise TombstoneBatchRejected("format", f"origin_seq {entry.origin_seq} must be >= 1")
    if not entry.tenant_id or not entry.app_id:
        raise TombstoneBatchRejected("format", "tenant_id and app_id must be non-empty")
    if not entry.user_id or not entry.user_id.strip():
        raise TombstoneBatchRejected("format", "user_id must be non-empty")
    if entry.scope not in SCOPES:
        raise TombstoneBatchRejected("format", f"scope {entry.scope!r} is not one of {SCOPES}")
    if entry.scope == "belief" and not _HEX32.match(entry.scope_key or ""):
        raise TombstoneBatchRejected(
            "format", "a belief tombstone needs the belief's 32-hex global_id as scope_key")
    if entry.scope == "session" and not entry.scope_key:
        raise TombstoneBatchRejected("format", "a session tombstone needs the session_id as scope_key")
    if entry.scope == "user" and entry.scope_key:
        raise TombstoneBatchRejected("format", "a user tombstone carries an empty scope_key")
    if entry.generation < 0:
        raise TombstoneBatchRejected("format", "generation must be >= 0")
    try:
        datetime.fromisoformat(entry.created_at)
    except (TypeError, ValueError):
        raise TombstoneBatchRejected("format", f"created_at {entry.created_at!r} is not ISO 8601")
    if not _HEX64.match(entry.hash or ""):
        raise TombstoneBatchRejected("format", "hash must be a 64-hex SHA-256")


def validate_batch(batch: TombstoneBatch, *, expected_tenant: tuple[str, str]) -> None:
    """Everything that can be checked WITHOUT the database: the envelope
    hash, per-entry format and hash, tenant, and in-batch duplicates.
    The per-sender cursor check needs the receiver's stored state and
    runs inside the apply transaction."""
    if not _HEX32.match(batch.sender or ""):
        raise TombstoneBatchRejected("format", f"sender {batch.sender!r} is not a 32-hex db_id")
    if batch.from_cursor < 0 or batch.to_cursor < batch.from_cursor:
        raise TombstoneBatchRejected(
            "sequence",
            f"batch cursors are inverted (from {batch.from_cursor} to {batch.to_cursor})")
    if not _HEX64.match(batch.from_chain_hash or "") or not _HEX64.match(batch.to_chain_hash or ""):
        raise TombstoneBatchRejected("format", "batch chain hashes must be 64-hex SHA-256")
    if batch.compute_hash() != batch.hash:
        raise TombstoneBatchRejected(
            "hash", "batch hash does not verify: the envelope or the entry "
            "list was altered in transit")
    seen: set[tuple[str, str]] = set()
    for entry in batch.entries:
        validate_entry_format(entry)
        if entry.compute_hash() != entry.hash:
            raise TombstoneBatchRejected(
                "hash", f"tombstone {entry.origin[:8]}.../{entry.origin_seq}: hash does not verify")
        if (entry.tenant_id, entry.app_id) != tuple(expected_tenant):
            raise TombstoneBatchRejected(
                "tenant",
                f"tombstone {entry.origin[:8]}.../{entry.origin_seq} belongs to tenant "
                f"({entry.tenant_id!r}, {entry.app_id!r}), not to this receiver's "
                f"({expected_tenant[0]!r}, {expected_tenant[1]!r}): the whole batch is refused")
        key = (entry.origin, entry.origin_hash)
        if key in seen:
            raise TombstoneBatchRejected(
                "sequence", f"tombstone {entry.origin[:8]}.../{entry.origin_seq} appears twice in the batch")
        seen.add(key)


def from_deletion_entry(entry: DeletionEntry) -> TombstoneEntry | None:
    """The cross-device form of a local log entry, or None when the entry
    cannot travel: a belief deletion recorded before the belief carried a
    global_id (pre-batch-2) has no cross-device key."""
    if entry.scope in _LOCAL_ONLY_SCOPES:
        return None
    if entry.scope == "belief":
        scope_key = entry.global_key
        if not _HEX32.match(scope_key or ""):
            return None
    elif entry.scope == "session":
        scope_key = entry.scope_key
    else:
        scope_key = ""
    created = entry.created_at.isoformat() if entry.created_at else ""
    return TombstoneEntry(
        origin=entry.origin or entry.db_id,
        origin_seq=entry.origin_seq if entry.origin else entry.seq,
        tenant_id=entry.tenant_id, app_id=entry.app_id, user_id=entry.user_id,
        scope=entry.scope, scope_key=scope_key, generation=entry.generation,
        created_at=created,
        origin_hash=entry.origin_hash or entry.hash,
    ).seal()


def export_from_log(
    log: DeletionLog, db_id: str, since_cursor: int, limit: int,
) -> tuple[TombstoneBatch, int, int]:
    """Read the local log after *since_cursor* and seal the next batch of
    tombstones.  Returns ``(batch, next_cursor, skipped)``:

    - the batch carries sender=db_id, from_cursor=the ACTUAL served start,
      to_cursor=the LOCAL seq of the last entry read, the chain hashes at
      both ends, the entries, and the envelope hash.  The receiver stores
      (to_cursor, to_chain_hash) for the next batch's continuity check;
    - a cursor ABOVE the log's head means the CALLER's cursor overshoots
      (the sender rolled back and its log rewound below what the receiver
      already consumed): the export REWINDS to 0 and serves everything --
      the receiver's (origin, origin_hash) applied records deduplicate
      the re-delivery, so a rollback is healed automatically instead of
      silently skipped;
    - ``skipped`` counts entries that cannot travel (pre-batch-2 belief
      deletions without a global_id).

    A cursor before the log's compaction checkpoint raises
    :class:`TombstoneCursorPruned` (task 4) -- the peer must rebuild.
    """
    checkpoint = getattr(log, "checkpoint_seq", None)
    if callable(checkpoint):
        cp = checkpoint(db_id)
        if cp and since_cursor < cp:
            raise TombstoneCursorPruned(
                f"cursor {since_cursor} lies before the deletion log's compaction "
                f"checkpoint {cp} for db {db_id[:8]}...: the entries between were "
                "pruned and cannot be exported.  The peer must rebuild from a "
                "full state instead of catching up.")
    head_seq, _head_hash = log.head(db_id)
    rescan = since_cursor > head_seq
    if rescan:
        since_cursor = 0  # the caller's cursor overshoots a rewound log
    entries: list[TombstoneEntry] = []
    cursor = since_cursor
    skipped = 0
    for raw in log.read_from(db_id, since_cursor):
        cursor = raw.seq
        if raw.scope in _LOCAL_ONLY_SCOPES:
            continue
        tombstone = from_deletion_entry(raw)
        if tombstone is None:
            skipped += 1
            continue
        entries.append(tombstone)
        if len(entries) >= limit:
            break
    from_chain = log.chain_hash_at(db_id, since_cursor)
    to_chain = log.chain_hash_at(db_id, cursor)
    batch = TombstoneBatch(sender=db_id, from_cursor=since_cursor,
                           to_cursor=cursor, from_chain_hash=from_chain,
                           to_chain_hash=to_chain, entries=entries).seal()
    return batch, cursor, skipped
