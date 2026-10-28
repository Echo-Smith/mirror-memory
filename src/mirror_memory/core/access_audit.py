"""Access audit (M3 batch-2 task 2): one append-only row per read of memory.

Every entry that returns memory to a caller -- recall, panel, explain,
get_beliefs, belief_history, the verification candidates, stats, export,
and the server routes in front of them -- writes one :class:`AccessAudit`
row in the SAME transaction as the read: who (user_id), under which purpose
and grant version, which entry, how many memories came back and which ids.
A refused access writes a row too (decision ``denied`` with the refusal
reason), so a probing caller leaves a trace.

Fail-closed: the audit INSERT and the commit are not optional.  If either
fails the entry raises and returns NOTHING -- memory never leaves the
engine unrecorded.

Only ids are stored, never content, so the rows survive
``delete_memories`` (the user's right to erasure is about content; the
record that content was read is not content).  Retention is a config
value; purging is an explicit, tenant-scoped maintenance call
(:func:`purge_access_audit`), never automatic.
"""

from __future__ import annotations

import contextlib
import json
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Iterator

from sqlalchemy.orm import Session

from mirror_memory.core.models import AccessAudit
from mirror_memory.core.repository import _sanctioned_writer

#: Decisions an audit row can carry.
ALLOWED = "allowed"
DENIED = "denied"

#: Upper bound on the id-list cap a config may set: a row stays a few KB.
MAX_IDS_HARD_CAP = 5000

#: The entry label the server sets around an engine call so the row names
#: the route, not the library method behind it (labelling only -- the
#: authorization decision never reads this).
_entry_label: ContextVar[str | None] = ContextVar("mm_access_entry_label", default=None)


@contextlib.contextmanager
def entry_label(name: str) -> Iterator[None]:
    """Label the audit rows written inside the block with *name* (the
    ENTRY_REGISTRY key of the outermost entry, e.g. a server route)."""
    token = _entry_label.set(name)
    try:
        yield
    finally:
        _entry_label.reset(token)


def effective_entry(default: str) -> str:
    """The label for a row: the outermost entry's label if one is set,
    else *default* (the library method's registry key)."""
    return _entry_label.get() or default


@_sanctioned_writer("audit")
def record_access(
    session: Session,
    *,
    user_id: str,
    purpose: str,
    operation: str,
    entry: str,
    decision: str,
    reason: str = "",
    grant_version: int | None = None,
    memory_ids: list[int] | tuple[int, ...] = (),
    max_ids: int,
) -> AccessAudit:
    """Append one audit row.  *memory_ids* beyond *max_ids* are dropped and
    the row is marked truncated; ``memory_count`` is always the full count."""
    if decision not in (ALLOWED, DENIED):
        raise ValueError(f"unknown audit decision {decision!r}")
    if max_ids < 1 or max_ids > MAX_IDS_HARD_CAP:
        raise ValueError(f"max_ids must be in 1..{MAX_IDS_HARD_CAP}, got {max_ids}")
    ids = [int(i) for i in memory_ids]
    row = AccessAudit(
        user_id=user_id,
        purpose=purpose or "",
        operation=operation,
        entry=effective_entry(entry),
        decision=decision,
        reason=reason or "",
        grant_version=grant_version,
        memory_count=len(ids),
        memory_ids_json=json.dumps(ids[:max_ids]),
        truncated=len(ids) > max_ids,
    )
    session.add(row)
    session.flush()
    return row


def retention_cutoff(retention_days: int, *, now: datetime | None = None) -> datetime:
    """The instant before which audit rows are out of retention."""
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    base = now or datetime.now(UTC)
    if base.tzinfo is not None:
        base = base.astimezone(UTC).replace(tzinfo=None)
    return base - timedelta(days=retention_days)


@_sanctioned_writer("audit")
def purge_access_audit(session: Session, *, older_than: datetime) -> int:
    """Delete this tenant's audit rows created before *older_than*; returns
    the number removed.  The ONE sanctioned deletion on the append-only
    table: an explicit maintenance call under the caller's tenant scope.
    A host with many tenants runs it per tenant -- the engine never
    enumerates tenants on a host's behalf."""
    if older_than.tzinfo is not None:
        older_than = older_than.astimezone(UTC).replace(tzinfo=None)
    return int(
        session.query(AccessAudit)
        .filter(AccessAudit.created_at < older_than)
        .delete(synchronize_session=False)
    )
