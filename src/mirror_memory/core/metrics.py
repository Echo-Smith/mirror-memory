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


def inc(name: str, n: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + n


def value(name: str) -> int:
    with _lock:
        return _counters.get(name, 0)


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counters)
