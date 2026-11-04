"""In-process operational counters (M3 batch 3 task 3): metrics that must
not cost a query.

A ``missing_purpose`` refusal is decided in Python before any SQL (the
batch-1 invariant), so counting it in the database would either break that
invariant or need a second write path for a refusal.  The counter lives in
the process instead: cheap, lock-protected, and reset on restart -- it is
an operational signal (an integration suddenly calling without purposes),
not an audit record.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_counters: dict[str, int] = {}
_gauges: dict[str, float] = {}

#: A request (engine entry or worker job) was refused for having no purpose.
MISSING_PURPOSE = "missing_purpose_refusals"

#: The worker skipped a whole tick because its deletion-log reconciliation
#: failed (broken chain, unavailable log, missing watermark, infrastructure
#: error) -- no lease recovery, no claims, no publishes until it recovers.
DELETION_RECONCILE_FAILURES = "deletion_reconcile_failures"

#: Cross-device tombstones applied by this process (M4 batch 2).
TOMBSTONES_APPLIED = "tombstones_applied"
#: Tombstones skipped as already applied (idempotent re-delivery).
TOMBSTONES_SKIPPED = "tombstones_skipped"
#: Whole batches refused, by reason: tombstones_rejected_<reason>
#: (format / hash / tenant / user / sequence / apply).
TOMBSTONES_REJECTED_PREFIX = "tombstones_rejected_"
#: Local log entries an export could not turn into tombstones.
TOMBSTONES_EXPORT_SKIPPED = "tombstones_export_skipped"

#: The tombstone channel was rescanned (sender rollback detected: the
#: batch's from_cursor/from_chain_hash did not continue the stored
#: channel) -- the whole sender log is re-fetched with entry-level dedup.
TOMBSTONES_RESCANS = "tombstones_rescans"


def inc(name: str, n: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


def value(name: str) -> int:
    with _lock:
        return _counters.get(name, 0)


#: Gauges (M4 batch 2 task 5): the current VALUE of a state -- the replay
#: progress, the sync mode, the age of the last catch-up -- as opposed to
#: the monotone counters above.
REPLAY_CURRENT_SEQ = "deletion_replay_current_seq"
REPLAY_TAIL_SEQ = "deletion_replay_tail_seq"
REPLAY_ACTIVE = "deletion_replay_active"  # 1 while a replay is in progress
SYNC_CATCHING_UP = "sync_catching_up"  # 1 while the edge is catching up
SYNC_SINCE_LAST_CATCHUP_SECONDS = "sync_since_last_catchup_seconds"


def set_gauge(name: str, value: float) -> None:
    with _lock:
        _gauges[name] = float(value)


def gauge(name: str) -> float | None:
    with _lock:
        return _gauges.get(name)


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counters)


def gauges() -> dict[str, float]:
    with _lock:
        return dict(_gauges)
