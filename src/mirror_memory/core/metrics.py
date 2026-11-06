"""In-process operational telemetry (M3 batch 3 task 3, extended for M5).

Everything here must be answerable WITHOUT a query on the request path --
that is the governing principle ("no extra queries").  The counters and
gauges are incremented/overwritten where the event already happens; the
histograms are fixed-bucket aggregates updated per call; storage
statistics -- the one thing that inherently needs SQL -- are computed by an
ON-DEMAND function (:func:`storage_stats`) or a low-frequency worker task,
never from an entry.

Three families:

Counters (monotone, ``inc``/``value``)
    ``missing_purpose_refusals``, tombstone counters, and the per-entry cost
    counters: LLM calls, input/output tokens (``unknown`` when the client
    surfaces no usage), and SQL statements issued.

Gauges (current value, ``set_gauge``/``gauge``)
    Replay progress, sync state, the age of the last catch-up.

Histograms (latency, bounded memory, ``observe_latency``/``latency_summaries``)
    Labelled by ENTRY only (``observe`` / ``recall`` / ``extraction_worker``).
    Fixed exponential buckets: constant memory, no per-sample retention,
    thread-safe.  Quantiles are a bucket-index interpolation -- accurate to
    the bucket width and never unbounded.

Privacy / label discipline (M5 task 2)
    Metrics NEVER carry memory content, user text, or any label other than
    the metric name and (for histograms) the ``entry`` value.  A user id
    appears ONLY inside :func:`storage_stats` (a library value) and the
    ``storage`` section of :func:`report` -- never in a Prometheus label.
    :func:`render_prometheus` is the ONLY serialiser that touches the wire,
    and it never sees a user id.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counters: dict[str, int] = {}
_gauges: dict[str, float] = {}

#: Process-wide on/off switch.  Recording is cheap but not free; an operator
#: (or a controlled-overhead benchmark) may turn it off.  When off, the
#: *recording* helpers become no-ops, but the entry-scope context still works
#: so call sites do not branch.  Default on.
_enabled = True

#: Whether the ONE global SQL cursor listener has been attached.  Tracked
#: here (module-level) rather than per engine so multiple ``MemoryEngine``
#: instances never each try to install their own class listener.
_sql_listener_installed = False


def sql_listener_installed() -> bool:
    with _lock:
        return _sql_listener_installed


def mark_sql_listener_installed() -> None:
    global _sql_listener_installed
    with _lock:
        _sql_listener_installed = True


_listener_lock = threading.Lock()


def ensure_sql_listener(install) -> bool:
    """Install the process-wide cursor listener AT MOST ONCE, atomically.

    ``install`` is called inside a dedicated lock with a double-check, so two
    engines constructed concurrently on different threads cannot both attach a
    listener (which would double-count every statement).  Returns True when
    THIS call performed the installation.
    """
    global _sql_listener_installed
    if _sql_listener_installed:
        return False
    with _listener_lock:
        if _sql_listener_installed:
            return False
        install()
        _sql_listener_installed = True
        return True


def set_enabled(value: bool) -> None:
    """Turn metric recording on/off process-wide (M5 task 2 overhead switch).

    This only gates *recording*; it never changes engine behaviour.  It exists
    so the overhead experiment can measure the same workload with metrics
    enabled vs disabled, and so an operator can drop the (tiny) in-process
    cost entirely."""
    global _enabled
    with _lock:
        _enabled = bool(value)


def enabled() -> bool:
    with _lock:
        return _enabled


# ---------------------------------------------------------------------------
# Per-thread "current entry" context
#
# The WHOLE attribution model lives here, in ONE thread-local, rather than in
# each MemoryEngine.  Two reasons (M5 fix):
#   1. SQL counting must not bind a global Engine-class listener to a single
#      engine instance -- multiple MemoryEngines (one per tenant) each read
#      the SAME thread-local, so every engine counts correctly.
#   2. An unrelated engine's SQL on the same thread must not be attributed to
#      whichever entry happens to be open -- the listener only counts while a
#      scope is actually active on THIS thread, and the scope names one entry.
# Contexts nest (observe -> nested pipeline helpers keep 'observe'); closing
# restores the previous value exactly (no leakage on exceptions).
# ---------------------------------------------------------------------------

_entry_tls = threading.local()

#: Entries whose cost may be attributed.  Closed set -- an arbitrary string is
#: never accepted as a label (it could smuggle content or an identifier onto
#: the Prometheus wire).  ``UNKNOWN`` is the only extra token, and it is used
#: for SQL with no entry scope rather than as a label from a call site.
ENTRY_OBSERVE = "observe"
ENTRY_RECALL = "recall"
ENTRY_EXTRACTION_WORKER = "extraction_worker"
#: The metabolism cycle is its OWN entry -- it must not be folded into
#: ``observe`` (that would pollute observe's latency/token accounting).
ENTRY_METABOLISM = "metabolism"

ALLOWED_ENTRIES = frozenset(
    {ENTRY_OBSERVE, ENTRY_RECALL, ENTRY_EXTRACTION_WORKER, ENTRY_METABOLISM}
)


def is_allowed_entry(entry: object) -> bool:
    return isinstance(entry, str) and entry in ALLOWED_ENTRIES


def current_entry() -> str | None:
    return getattr(_entry_tls, "entry", None)


def current_engine_key() -> int | None:
    """Identity of the SQLAlchemy engine that opened the active entry scope."""
    return getattr(_entry_tls, "engine_key", None)


@contextmanager
def entry_scope(entry: str, *, engine: object | None = None) -> Iterator[None]:
    """Mark *entry* (and optionally *engine*) as the active attribution target
    on this thread for the span of the ``with`` block.

    ``engine`` is the SQLAlchemy ``Engine`` (or a ``MemoryEngine`` exposing
    ``._engine``) whose SQL belongs to this entry.  The global cursor listener
    only counts statements issued on THAT engine, so an unrelated engine's SQL
    on the same thread during an active scope is not miscounted.  Nested
    scopes restore the previous values exactly (also on exception).

    An entry not in :data:`ALLOWED_ENTRIES` is rejected (``ValueError``) -- the
    label set is closed, never caller-controlled.
    """
    if not is_allowed_entry(entry):
        raise ValueError(
            f"unknown metrics entry {entry!r}; allowed: {sorted(ALLOWED_ENTRIES)}"
        )
    engine_obj = engine
    if engine_obj is not None and not hasattr(engine_obj, "dialect"):
        engine_obj = getattr(engine_obj, "_engine", None)
    previous = getattr(_entry_tls, "entry", None)
    previous_engine = getattr(_entry_tls, "engine_key", None)
    _entry_tls.entry = entry
    _entry_tls.engine_key = id(engine_obj) if engine_obj is not None else None
    # Per-call cost capture: baseline of this entry's counters at scope start;
    # deltas at scope end become ONE per-call record (bounded ring, §2
    # "actual per-call cost, not only cross-call averages").
    baseline = {p: value(f"{p}.{entry}") for p in _COST_CAPTURE_PREFIXES}
    started = time.monotonic()
    try:
        yield
    finally:
        _entry_tls.entry = previous
        _entry_tls.engine_key = previous_engine
        try:
            _record_recent_cost(entry, baseline, started)
        except Exception:  # never break the entry
            logger.debug("metrics: recent-cost capture failed", exc_info=True)


def record_sql_current(engine: object | None = None, n: int = 1) -> None:
    """Count *n* SQL statements against the CURRENT entry, if it applies.

    Called by the single process-wide cursor listener with the *connection's*
    engine.  The count happens only when BOTH hold:

    * an entry scope is open on this thread (no scope -> no attribution), AND
    * the scope is bound to an engine AND that is the engine which issued the
      statement.  A scope with NO engine binding (cold start, before
      ``MemoryEngine._engine`` exists) attributes NOTHING -- ``engine=None``
      never means "all engines"; unrelated SQL on the same thread is never
      counted.  ``metrics.bind_scope_engine`` binds the scope once the engine
      exists, so genuine engine-initialisation SQL IS attributed.
    """
    entry = getattr(_entry_tls, "entry", None)
    if entry is None:
        return
    scope_engine = getattr(_entry_tls, "engine_key", None)
    if scope_engine is None:
        return                      # unbound scope: never "all engines"
    if engine is None:
        return
    issued_by = getattr(engine, "engine", engine)
    if id(issued_by) != scope_engine:
        return                      # another engine's SQL on this thread
    record_sql(entry, n)


def bind_scope_engine(engine: object | None) -> bool:
    """Bind the CURRENT entry scope to *engine* (once it exists).

    Cold start: an entry is opened before ``MemoryEngine._engine`` exists, so
    the scope initially carries no engine and attributes no SQL.  As soon as
    the engine is constructed, the engine calls this to bind the scope -- from
    that point, its SQL (schema creation, RLS posture, the entry's own work)
    IS attributed to the entry.  Binding only upgrades an UNBOUND scope; it
    never re-binds a scope that another engine already owns, and it does
    nothing when no scope is active.  Returns True when binding happened."""
    if getattr(_entry_tls, "entry", None) is None:
        return False
    if getattr(_entry_tls, "engine_key", None) is not None:
        return False
    engine_obj = engine
    if engine_obj is not None and not hasattr(engine_obj, "dialect"):
        engine_obj = getattr(engine_obj, "_engine", None)
    if engine_obj is None:
        return False
    _entry_tls.engine_key = id(engine_obj)
    return True


class _LLMClientProxy:
    """Marker base so :func:`instrument_llm_client` is idempotent."""

    _inner: object


#: Per-call usage capture for the CURRENT thread.  ``reported`` records whether
#: the provider surfaced a usage object for the call in flight; ``inp``/``out``
#: carry whatever it did surface (either may be None -> partial usage).
_llm_tls = threading.local()


def record_llm_tokens(
    entry: str,
    *,
    input_tokens: int | str | None = None,
    output_tokens: int | str | None = None,
) -> None:
    """Add MEASURED token counts for *entry* WITHOUT touching the call count.

    The call itself is counted exactly once by the instrumented client proxy.
    A provider usage hook contributes the real tokens through here; numbers
    that were not surfaced are simply not added (the proxy books the call as
    partial/unknown usage instead of guessing)."""
    if not enabled() or not is_allowed_entry(entry):
        return

    def _add(prefix: str, tok: int | str | None) -> None:
        if isinstance(tok, bool) or not isinstance(tok, int):
            return
        inc(f"{prefix}.{entry}", int(tok))

    _add(COST_INPUT_TOKENS_PREFIX, input_tokens)
    _add(COST_OUTPUT_TOKENS_PREFIX, output_tokens)


#: Lock guarding hook installation on a shared client (concurrent
#: ``instrument_llm_client`` calls must not install two chains).
_llm_install_lock = threading.Lock()


def _installed_hook(inner) -> bool:
    return bool(getattr(inner, "_mm_metrics_hook", False))


def _chain_installed(inner) -> bool:
    """True when OUR chain is the client's current ``_on_call``.

    A host holding the RAW client can call ``set_on_call`` directly and
    overwrite the chain; the proxy detects that on the next call and
    re-asserts itself while adopting the host's callback."""
    return getattr(inner, "_on_call", None) is getattr(inner, "_mm_metrics_chain", None)


def _install_chained_hook(inner) -> None:
    """Install ONE hook on *inner* that records usage AND still calls the
    host's CURRENT callback (read dynamically at call time).

    Lifecycle rules (M5 patch 3):

    * the host callback is stored in ``_mm_metrics_host_on_call`` -- never
      captured in a closure -- so replacing or clearing it later (through the
      proxy) is picked up on the next call and nothing goes stale;
    * installation is idempotent and lock-guarded: two threads instrumenting
      the same shared client install exactly one chain;
    * if the host bypasses the proxy and calls ``set_on_call`` on the RAW
      client, our chain is overwritten -- the proxy detects that (by identity
      comparison) on the next ``generate()`` and re-asserts the chain while
      adopting whatever the host installed.  The host's callback keeps
      working; usage accounting resumes.
    """
    if not hasattr(inner, "set_on_call"):
        return
    with _llm_install_lock:
        if _installed_hook(inner) and _chain_installed(inner):
            return
        # A host callback installed between our install and now (raw-reference
        # replacement) is adopted as the host callback, not lost.
        host_current = getattr(inner, "_on_call", None)
        if host_current is not None and host_current is not getattr(
            inner, "_mm_metrics_chain", None
        ):
            inner._mm_metrics_host_on_call = host_current

        def _chain(info: dict) -> None:
            try:
                _llm_tls.reported = True
                _llm_tls.ok = info.get("ok", True)
                _llm_tls.input_tokens = info.get("input_tokens")
                _llm_tls.output_tokens = info.get("output_tokens")
                entry = getattr(_entry_tls, "entry", None)
                if entry is not None:
                    record_llm_tokens(entry, input_tokens=_llm_tls.input_tokens,
                                      output_tokens=_llm_tls.output_tokens)
            except Exception:  # metrics must never break a provider callback
                logger.debug("metrics: usage hook failed", exc_info=True)
            # Forward to the host's CURRENT callback, read dynamically -- a
            # replacement or clearing done through the proxy is honoured here.
            host_cb = getattr(inner, "_mm_metrics_host_on_call", None)
            if host_cb is not None:
                try:
                    host_cb(info)
                except Exception:
                    logger.debug("metrics: host LLM callback raised", exc_info=True)

        try:
            inner.set_on_call(_chain)
            inner._mm_metrics_hook = True
            inner._mm_metrics_chain = _chain
            if not hasattr(inner, "_mm_metrics_host_on_call"):
                inner._mm_metrics_host_on_call = None
        except Exception:  # pragma: no cover - exotic client
            logger.debug("metrics: could not install usage hook", exc_info=True)


def instrument_llm_client(client):
    """Wrap an LLM client so every ``generate()`` is counted exactly once,
    whatever the client (M5 patch 3).

    Per call, on the CALLING THREAD:

    * ``llm_calls_total.<entry>``   -- always, when an entry scope is open;
    * ``llm_calls_failed_total``    -- ``generate()`` raised, OR the provider
      reported ``ok=False`` (e.g. the built-in ``OpenAILLM`` swallowed an
      error and returned its ``fallback`` -- that is still a failed call);
    * ``llm_input_tokens`` / ``llm_output_tokens`` -- real numbers only;
    * ``llm_usage_partial_total``   -- usage surfaced but only one side;
    * ``llm_usage_unknown_total``   -- no usage surfaced at all.

    A call made with no entry scope open records nothing (no attribution, no
    cross-talk).  ``set_on_call`` through the proxy sets or clears the HOST
    callback without touching the metrics chain; a host that replaces the
    callback on the RAW client reference is detected and adopted on the next
    call.  The proxy is transparent for every other attribute, idempotent, and
    never raises from bookkeeping."""
    if client is None:
        return None
    if isinstance(client, _LLMClientProxy):
        return client

    _install_chained_hook(client)

    class _Proxy(_LLMClientProxy):
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_inner"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_inner"), name, value)

        def set_on_call(self, hook):
            """Set or clear the HOST callback without disturbing metrics.

            The metrics chain stays installed; the chain forwards to whatever
            host callback is current at call time (``None`` clears it)."""
            inner = object.__getattribute__(self, "_inner")
            inner._mm_metrics_host_on_call = hook
            if not _installed_hook(inner) or not _chain_installed(inner):
                # chain missing (never installed, or clobbered by a raw
                # set_on_call while a host callback exists) -> (re)install
                _install_chained_hook(inner)

        def generate(self, *args, **kwargs):
            inner = object.__getattribute__(self, "_inner")
            entry = getattr(_entry_tls, "entry", None)
            # Lifecycle: if the host replaced our chain through the RAW client
            # reference, adopt its callback and re-assert the chain.
            if _installed_hook(inner) and not _chain_installed(inner):
                inner._mm_metrics_host_on_call = getattr(inner, "_on_call", None)
                _install_chained_hook(inner)
            _llm_tls.reported = False
            _llm_tls.ok = True
            _llm_tls.input_tokens = None
            _llm_tls.output_tokens = None
            raised = False
            try:
                return inner.generate(*args, **kwargs)
            except BaseException:
                raised = True
                raise
            finally:
                if entry is not None and is_allowed_entry(entry) and enabled():
                    inc(f"{COST_LLM_CALLS_PREFIX}.{entry}")
                    failed = raised or not getattr(_llm_tls, "ok", True)
                    if failed:
                        # includes ok=False + fallback-return (the built-in
                        # OpenAILLM swallows provider errors into a fallback)
                        inc(f"{COST_LLM_CALLS_FAILED_PREFIX}.{entry}")
                    elif not getattr(_llm_tls, "reported", False):
                        inc(f"{COST_USAGE_UNKNOWN_PREFIX}.{entry}")
                    else:
                        inp = getattr(_llm_tls, "input_tokens", None)
                        out = getattr(_llm_tls, "output_tokens", None)
                        if inp is None and out is None:
                            # provider surfaced a usage object with NO token
                            # counts on either side -> unknown, never zero
                            inc(f"{COST_USAGE_UNKNOWN_PREFIX}.{entry}")
                        elif inp is None or out is None:
                            # exactly one side reported -> partial; the side
                            # that WAS reported is still added to its counter
                            inc(f"{COST_USAGE_PARTIAL_PREFIX}.{entry}")

    return _Proxy(client)


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


#: Closed set of tombstone-batch rejection reasons (the ``reason`` label).
#: Anything else is bucketed under ``other`` -- a raw string from an exception
#: never becomes a label value.
ALLOWED_REASONS = frozenset(
    {"apply", "format", "hash", "sender", "sequence", "tenant", "user"}
)


def record_tombstone_rejection(reason: str, n: int = 1) -> None:
    """Count one rejected tombstone batch by reason (closed label set).

    A reason outside :data:`ALLOWED_REASONS` is counted under ``other`` so a
    new or unexpected reason stays visible without ever putting an
    unconstrained string on the Prometheus wire."""
    label = reason if reason in ALLOWED_REASONS else "other"
    inc(TOMBSTONES_REJECTED_PREFIX + label, n)


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
    """The counter snapshot -- HISTORICAL ``dict[str, int]`` contract.

    Every counter name maps to an ``int``; there are no other keys, so
    ``snapshot().get(name, 0)`` and iteration over counter names keep working
    for every existing caller.

    Gauges, latency distributions, per-call costs, recent per-call records and
    the (optional, on-demand) storage section live on the SEPARATE full
    interface :func:`snapshot_full` (or :func:`report`).
    """
    with _lock:
        return dict(_counters)


def snapshot_full(*, storage: dict | None = None) -> dict:
    """The complete telemetry snapshot: everything the library knows.

    Sections (fixed keys, never colliding with counter names):

    ``counters``  {name: int} -- identical to :func:`snapshot`;
    ``gauges``    {name: float};
    ``latency``   {entry: {count, p50, p95, p99}} (fixed-bucket distributions);
    ``cost``      {entry: per-call cost averages} (:func:`entry_cost`);
    ``recent``    {entry: [per-call cost records]} (:func:`recent_costs`);
    ``storage``   {(tenant, app, user): {table: {rows, bytes}}} -- ONLY when
                  the caller passes the on-demand ``storage=`` value (it needs
                  SQL; never computed on the request path);
    ``enabled``   whether recording is on.
    """
    return {
        "counters": snapshot(),
        "gauges": gauges(),
        "latency": latency_summaries(),
        "cost": entry_costs(),
        "recent": {e: recent_costs(e) for e in sorted(ALLOWED_ENTRIES)},
        "storage": storage if storage is not None else {},
        "enabled": enabled(),
    }


def gauges() -> dict[str, float]:
    with _lock:
        return dict(_gauges)


# ---------------------------------------------------------------------------
# Per-entry cost counters (M5 task 2 requirement 1)
#
# Three counters, one entry label apiece.  The entry is a COMPILE-TIME
# constant at every call site, never user input -- so no label can smuggle
# content or a user id into the metric name.
# ---------------------------------------------------------------------------

# (`ENTRY_*` and `ALLOWED_ENTRIES` are defined once, above the entry scope.)
COST_LLM_CALLS_PREFIX = "llm_calls_total"        # + ".<entry>"
COST_INPUT_TOKENS_PREFIX = "llm_input_tokens"    # + ".<entry>"
COST_OUTPUT_TOKENS_PREFIX = "llm_output_tokens"  # + ".<entry>"
#: Calls whose provider surfaced NO usage object -> counted separately so the
#: token counters are never inflated by a guess.
COST_USAGE_UNKNOWN_PREFIX = "llm_usage_unknown_total"  # + ".<entry>"
#: Calls where the provider surfaced usage but only one side (in OR out).  The
#: side that was reported is still added to its token counter.
COST_USAGE_PARTIAL_PREFIX = "llm_usage_partial_total"  # + ".<entry>"
#: Calls where ``generate()`` raised (still counted as a call attempt).
COST_LLM_CALLS_FAILED_PREFIX = "llm_calls_failed_total"  # + ".<entry>"

#: SQL statements issued, by entry ("unknown" when no request context is
#: active -- e.g. a manual DDL script).
SQL_STATEMENTS_PREFIX = "sql_statements_total"  # + ".<entry>"

#: ``unknown`` is the sentinel recorded for a token count a client could not
#: provide (a stub, an error path, a provider without usage).
UNKNOWN = "unknown"


def record_llm(
    entry: str,
    *,
    input_tokens: int | str | None = None,
    output_tokens: int | str | None = None,
) -> None:
    """Record one LLM call for *entry*.

    The call count always increments (a call was made).  Token counts are
    added when numeric; when a value is ``None`` or not an int the call is
    counted under the ``unknown`` counter instead so the token totals mean
    "tokens we actually measured", never a fabricated number.
    """
    if not enabled() or not is_allowed_entry(entry):
        return
    inc(f"{COST_LLM_CALLS_PREFIX}.{entry}")

    def _add(prefix: str, tok: int | str | None) -> bool:
        if isinstance(tok, bool) or not isinstance(tok, int):
            return False
        inc(f"{prefix}.{entry}", int(tok))
        return True

    got_in = _add(COST_INPUT_TOKENS_PREFIX, input_tokens)
    got_out = _add(COST_OUTPUT_TOKENS_PREFIX, output_tokens)
    if not (got_in and got_out):
        inc(f"{COST_USAGE_UNKNOWN_PREFIX}.{entry}")


def record_sql(entry: str = UNKNOWN, n: int = 1) -> None:
    """Record *n* SQL statements issued for *entry* (count only).

    ``entry`` must be a closed-set entry name (or the ``unknown`` sentinel for
    statements with no entry scope); anything else is dropped, never encoded
    as a label."""
    if not enabled() or (entry != UNKNOWN and not is_allowed_entry(entry)):
        return
    inc(f"{SQL_STATEMENTS_PREFIX}.{entry}", n)


# ---------------------------------------------------------------------------
# Latency histograms (M5 task 2 requirement 2)
#
# Fixed exponential buckets in milliseconds: [0,1,2,4,8,...) plus a +Inf
# overflow.  Bounded memory (one int array per entry, independent of the
# number of samples), thread-safe under the shared lock, and quantiles are
# reported from the bucket edges -- never from retained samples.
# ---------------------------------------------------------------------------

HIST_ENTRIES = (ENTRY_OBSERVE, ENTRY_RECALL, ENTRY_EXTRACTION_WORKER, ENTRY_METABOLISM)
_HIST_BUCKET_COUNT = 27  # 2**0 .. 2**26 ms (~18.6 h) then +Inf
_histograms: dict[str, list[int]] = {}
_hist_sums: dict[str, float] = {}


def _bucket_index(ms: float) -> int:
    """Bucket for *ms*: index i holds values in [2**(i-1), 2**i) with a
    separate bucket for [0,1).  Index == _HIST_BUCKET_COUNT is the overflow
    (>= 2**26 ms)."""
    if ms < 1.0:
        return 0
    idx = int(math.ceil(math.log2(ms))) if ms > 0 else 0
    return min(idx, _HIST_BUCKET_COUNT)


def observe_latency(entry: str, ms: float) -> None:
    """Record one latency sample (milliseconds) for *entry*.

    Only a closed-set entry is accepted: the entry becomes a Prometheus label,
    so an arbitrary string is rejected rather than encoded onto the wire."""
    if not enabled() or not is_allowed_entry(entry):
        return
    if ms < 0:
        ms = 0.0
    with _lock:
        buckets = _histograms.get(entry)
        if buckets is None:
            buckets = [0] * (_HIST_BUCKET_COUNT + 1)
            _histograms[entry] = buckets
            _hist_sums[entry] = 0.0
        buckets[_bucket_index(ms)] += 1
        _hist_sums[entry] = _hist_sums.get(entry, 0.0) + ms


def _bucket_upper(i: int) -> float:
    if i >= _HIST_BUCKET_COUNT:
        return float(2 ** _HIST_BUCKET_COUNT)
    return float(2 ** i)


def _qname(q: float) -> str:
    return f"p{int(round(q * 100))}"


def _quantiles_from_buckets(buckets: list[int], quantiles: tuple[float, ...]) -> dict[str, float]:
    total = sum(buckets)
    out: dict[str, float] = {}
    if total == 0:
        for q in quantiles:
            out[_qname(q)] = 0.0
        return out
    # Reported quantile = the UPPER edge of the bucket the quantile rank
    # falls in.  This bounds error by the (multiplicative) bucket width --
    # the standard exponential-bucket tradeoff: bounded memory, a fixed
    # number of counters regardless of traffic.
    targets = {q: q * total for q in quantiles}
    running = 0
    edge = _bucket_upper(0)
    qs = sorted(quantiles)
    qi = 0
    for i, count in enumerate(buckets):
        running += count
        edge = _bucket_upper(i)
        while qi < len(qs) and running >= targets[qs[qi]]:
            out[_qname(qs[qi])] = round(edge, 3)
            qi += 1
        if qi >= len(qs):
            break
    while qi < len(qs):
        out[_qname(qs[qi])] = round(edge, 3)
        qi += 1
    return out


def histogram(entry: str) -> dict:
    """The bounded aggregate for *entry*: cumulative ``le`` buckets + the
    standard quantile summary.  Empty (zero count) entries are still
    present so a dashboard never loses a series."""
    with _lock:
        buckets = list(_histograms.get(entry, [0] * (_HIST_BUCKET_COUNT + 1)))
        total_sum = float(_hist_sums.get(entry, 0.0))
    total = sum(buckets)
    cumulative = []
    running = 0
    for i, count in enumerate(buckets):
        running += count
        upper = _bucket_upper(i)
        if i >= _HIST_BUCKET_COUNT:
            cumulative.append((float("inf"), running))
        else:
            cumulative.append((upper, running))
    return {
        "count": total,
        "sum": total_sum,
        "buckets": cumulative,  # [(le_ms_or_inf, cumulative_count)]
        "summary": _quantiles_from_buckets(buckets, (0.5, 0.95, 0.99)),
    }


def latency_summaries() -> dict[str, dict]:
    """``{entry: {"count", "p50", "p95", "p99"}}`` for every entry with
    samples (plus known entries that have none, at zero)."""
    out: dict[str, dict] = {}
    with _lock:
        seen = set(_histograms) | set(HIST_ENTRIES)
    for entry in sorted(seen):
        h = histogram(entry)
        out[entry] = {"count": h["count"], **h["summary"]}
    return out


def histogram_all() -> dict[str, dict]:
    """Every histogram entry (known entries included at zero) -- the sealed
    set of entry labels the exposition may emit."""
    with _lock:
        entries = set(_histograms) | set(HIST_ENTRIES)
    return {e: histogram(e) for e in sorted(entries)}


# ---------------------------------------------------------------------------
# Per-call cost (M5 §3) -- AVERAGE cost of ONE entry invocation, as opposed to
# the monotonically-growing aggregate counters.
#
# The denominator is the latency-histogram sample count for that entry, which
# is exactly "how many times this entry ran" (each entry records one latency
# sample per invocation, covering its whole span -- init and sync gate
# included; see :meth:`MemoryEngine._metrics_entry`).
# ---------------------------------------------------------------------------


def entry_cost(entry: str) -> dict:
    """Average cost of ONE invocation of *entry*.

    Returns ``calls`` (invocations observed) plus per-call averages of LLM
    calls, measured tokens, failed calls and SQL statements.  Averages are
    ``None`` when the entry has not run yet -- never a fabricated zero.
    Totals stay available through :func:`snapshot` (the counters).
    """
    if not is_allowed_entry(entry):
        raise ValueError(
            f"unknown metrics entry {entry!r}; allowed: {sorted(ALLOWED_ENTRIES)}"
        )
    calls = histogram(entry)["count"]

    def _per_call(prefix: str) -> tuple[int, float | None]:
        total = value(f"{prefix}.{entry}")
        return total, (round(total / calls, 3) if calls else None)

    llm_calls, llm_per = _per_call(COST_LLM_CALLS_PREFIX)
    llm_failed, failed_per = _per_call(COST_LLM_CALLS_FAILED_PREFIX)
    tok_in, in_per = _per_call(COST_INPUT_TOKENS_PREFIX)
    tok_out, out_per = _per_call(COST_OUTPUT_TOKENS_PREFIX)
    usage_unknown, unknown_per = _per_call(COST_USAGE_UNKNOWN_PREFIX)
    usage_partial, partial_per = _per_call(COST_USAGE_PARTIAL_PREFIX)
    sql, sql_per = _per_call(SQL_STATEMENTS_PREFIX)
    return {
        "entry": entry,
        "calls": calls,
        "llm_calls": llm_calls,
        "llm_calls_per_call": llm_per,
        "llm_calls_failed": llm_failed,
        "llm_calls_failed_per_call": failed_per,
        "input_tokens": tok_in,
        "input_tokens_per_call": in_per,
        "output_tokens": tok_out,
        "output_tokens_per_call": out_per,
        "usage_unknown_calls": usage_unknown,
        "usage_unknown_per_call": unknown_per,
        "usage_partial_calls": usage_partial,
        "usage_partial_per_call": partial_per,
        "sql_statements": sql,
        "sql_statements_per_call": sql_per,
    }


def entry_costs() -> dict[str, dict]:
    """Per-call cost for every closed-set entry (zeros/None when never run)."""
    return {e: entry_cost(e) for e in sorted(ALLOWED_ENTRIES)}


#: Counter prefixes captured into each per-call cost record.
_COST_CAPTURE_PREFIXES = (
    COST_LLM_CALLS_PREFIX,
    COST_LLM_CALLS_FAILED_PREFIX,
    COST_INPUT_TOKENS_PREFIX,
    COST_OUTPUT_TOKENS_PREFIX,
    COST_USAGE_UNKNOWN_PREFIX,
    COST_USAGE_PARTIAL_PREFIX,
    SQL_STATEMENTS_PREFIX,
)

#: Bounded ring of the most recent per-call cost records per entry.  Fixed
#: capacity -- memory is bounded no matter the traffic.  Records carry ONLY
#: durations and counter deltas: no content, no user/tenant ids, no timestamps
#: beyond monotonic-clock arithmetic (never exposed).
_RECENT_RING_SIZE = 256
_recent_costs: dict[str, deque] = {}


def _record_recent_cost(entry: str, baseline: dict[str, int], started: float) -> None:
    """Append ONE per-call cost record (duration + counter deltas).

    Skipped entirely when recording is disabled -- with metrics off, no ring
    entry is produced (see the instrumentation-residue note in
    ``docs/reports``: the proxy/scope/listener structure remains, only the
    recording stops)."""
    if not enabled():
        return
    duration_ms = (time.monotonic() - started) * 1000.0
    record = {"duration_ms": round(duration_ms, 3)}
    for prefix in _COST_CAPTURE_PREFIXES:
        name = prefix.split("_total")[0].replace("llm_", "").replace("sql_", "")
        total = value(f"{prefix}.{entry}")
        delta = total - baseline.get(prefix, 0)
        key = {
            "calls": "llm_calls", "failed": "llm_failed",
            "input": "input_tokens", "output": "output_tokens",
            "usage_unknown": "usage_unknown", "usage_partial": "usage_partial",
            "statements": "sql_statements",
        }.get(name, name)
        record[key] = delta
    ring = _recent_costs.get(entry)
    if ring is None:
        ring = deque(maxlen=_RECENT_RING_SIZE)
        _recent_costs[entry] = ring
    ring.append(record)


def recent_costs(entry: str) -> list[dict]:
    """The most recent PER-CALL cost records for *entry* (oldest first).

    Each record: ``duration_ms`` plus the counter deltas that occurred during
    that one invocation (``llm_calls``, ``llm_failed``, ``input_tokens``,
    ``output_tokens``, ``usage_unknown``, ``usage_partial``,
    ``sql_statements``).  Bounded to the last
    ``{_RECENT_RING_SIZE}`` invocations.  These are ACTUAL per-call figures --
    not the cross-call averages of :func:`entry_cost` -- and they carry no
    content and no user/tenant identifiers.  Empty for an entry that has not
    run.  Library value only; never emitted to Prometheus.
    """
    if not is_allowed_entry(entry):
        raise ValueError(
            f"unknown metrics entry {entry!r}; allowed: {sorted(ALLOWED_ENTRIES)}"
        )
    ring = _recent_costs.get(entry)
    return [dict(r) for r in ring] if ring else []          # oldest first


def totals() -> dict[str, int]:
    """The monotonically-growing aggregate counters (the raw totals).

    This is what :func:`snapshot` exposes at its top level; :func:`entry_cost`
    turns the same numbers into per-invocation averages."""
    with _lock:
        return dict(_counters)


# ---------------------------------------------------------------------------
# Storage growth (M5 task 2 requirement 3; §2 patch)
#
# Row counts and estimated bytes PER (tenant, app, user), PER TABLE, by an
# ON-DEMAND SQL pass -- never on the request path.
#
# AUTHORIZATION (the important part): the tenant identity is NOT a free
# caller parameter.  It is taken from the BOUND tenancy context
# (``core.tenancy.current_tenant()``), which only the engine entry / server
# middleware can open.  With no bound context the call FAILS CLOSED
# (``TenantContextMissing``); the caller cannot ask for another tenant's
# numbers by passing one.  The SQL additionally carries the explicit
# ``WHERE tenant_id = :t AND app_id = :a`` predicate, and on PostgreSQL+RLS
# the row-security policy applies on top -- defence in depth, not a bypass.
#
# There is deliberately NO cross-tenant storage API: a capacity report that
# crosses tenants would need its own verifiable admin authorization and a
# dedicated privileged connection, which the library does not have.  Do not
# reintroduce one via a logical scope (``system_scope``) -- that is not a
# database grant.
# ---------------------------------------------------------------------------

#: Columns whose on-disk size dominates a row; a cheap per-table estimate
#: sums their octet length.  Keys are the ORM table names.
_ESTIMATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "mm_beliefs": ("claim_text", "key", "dimension", "predicate", "object"),
    "mm_evidence": ("content",),
    "mm_evidence_digests": ("digest",),
    "mm_session_summaries": ("summary",),
    "mm_belief_events": ("reason",),
}

#: The composite grouping key: a user id alone is NOT identity (two apps in
#: the same tenant may both have user ``u1``) -- the key carries all three so
#: those rows can never overwrite each other in the result map.
StorageKey = tuple[str, str, str]


def _q(engine, name: str) -> str:
    """Quote one reflected identifier with the engine's own preparer."""
    return engine.dialect.identifier_preparer.quote(name)


def storage_stats(engine) -> dict[StorageKey, dict[str, dict[str, int]]]:
    """Per-(tenant, app, user), per-table row counts + estimated bytes.

    The tenant is resolved from the BOUND context (``current_tenant()``) and
    the query is explicitly filtered to it.  No tenant argument is accepted:
    a caller cannot widen or redirect the report.

    Returns ``{(tenant_id, app_id, user_id): {table: {"rows", "bytes"}}}``.
    User ids appear ONLY in this library value (and in :func:`report`'s
    ``storage`` section) -- never in a Prometheus label.

    Raises
    ------
    TenantContextMissing
        When no tenant context is bound (fail-closed; no default tenant).
    """
    from sqlalchemy import inspect as _inspect
    from sqlalchemy import text as _text

    from mirror_memory.core.tenancy import current_tenant

    tenant_id, app_id = current_tenant()  # raises when unbound -> fail-closed
    if not tenant_id or not app_id:  # defensive: scope guarantees non-empty
        raise ValueError("storage_stats requires a bound, non-empty tenant context")

    inspector = _inspect(engine)
    tables = sorted(inspector.get_table_names())
    groups = ("tenant_id", "app_id", "user_id")

    out: dict[StorageKey, dict[str, dict[str, int]]] = {}
    with engine.connect() as conn:
        for table in tables:
            if not table.startswith("mm_"):
                continue
            cols = {c["name"] for c in inspector.get_columns(table)}
            if not set(groups) <= cols:
                continue
            size_cols = [c for c in _ESTIMATE_COLUMNS.get(table, ()) if c in cols]
            if size_cols:
                if engine.dialect.name == "postgresql":
                    size_expr = " + ".join(
                        f"COALESCE(octet_length({_q(engine, c)}), 0)" for c in size_cols
                    )
                else:
                    size_expr = " + ".join(
                        f"COALESCE(length(CAST({_q(engine, c)} AS BLOB)), 0)"
                        for c in size_cols
                    )
                select_sql = (
                    f"{_q(engine, 'tenant_id')}, {_q(engine, 'app_id')}, "
                    f"{_q(engine, 'user_id')}, COUNT(*) AS n, "
                    f"COALESCE(SUM({size_expr}), 0) AS b"
                )
            else:
                select_sql = (
                    f"{_q(engine, 'tenant_id')}, {_q(engine, 'app_id')}, "
                    f"{_q(engine, 'user_id')}, COUNT(*) AS n, 0 AS b"
                )
            sql = (
                f"SELECT {select_sql} FROM {_q(engine, table)} "
                f"WHERE {_q(engine, 'tenant_id')} = :t "
                f"AND {_q(engine, 'app_id')} = :a "
                f"GROUP BY {_q(engine, 'tenant_id')}, {_q(engine, 'app_id')}, "
                f"{_q(engine, 'user_id')}"
            )
            for row_tenant, row_app, user_id, n, b in conn.execute(
                _text(sql), {"t": tenant_id, "a": app_id}
            ):
                # Defence in depth: the explicit predicate already pins these,
                # and on PG+RLS the policy does too.  If a row from another
                # tenant ever appears, refuse loudly instead of reporting it.
                if str(row_tenant) != tenant_id or str(row_app) != app_id:
                    raise RuntimeError(
                        "storage_stats read a row outside the bound tenant "
                        "(tenant/app mismatch) -- refusing to report it"
                    )
                key: StorageKey = (str(row_tenant), str(row_app), str(user_id))
                out.setdefault(key, {})[table] = {"rows": int(n), "bytes": int(b)}
    return out


# ---------------------------------------------------------------------------
# The rich report (backward-compatible ``snapshot()`` stays untouched)
# ---------------------------------------------------------------------------


def report(*, storage: dict | None = None) -> dict:
    """The full telemetry report (same data as :func:`snapshot_full`)."""
    return snapshot_full(storage=storage)


def reset_for_tests() -> None:
    """Clear every in-process aggregate.  Test-only; never called by the
    engine."""
    global _enabled
    with _lock:
        _counters.clear()
        _gauges.clear()
        _histograms.clear()
        _hist_sums.clear()
        _enabled = True
        _recent_costs.clear()
    _entry_tls.entry = None
    _entry_tls.engine_key = None
    _llm_tls.reported = False
    _llm_tls.input_tokens = None
    _llm_tls.output_tokens = None


# ---------------------------------------------------------------------------
# Prometheus text exposition (M5 task 2 requirement 5)
#
# Labels are a CLOSED set: metric family name + (for histograms) ``entry``.
# There is deliberately no ``user``/``tenant``/``purpose`` label anywhere,
# so a user id cannot reach the wire even by accident.
# ---------------------------------------------------------------------------

_DYNAMIC_PREFIXES = (
    COST_LLM_CALLS_PREFIX,
    COST_LLM_CALLS_FAILED_PREFIX,
    COST_INPUT_TOKENS_PREFIX,
    COST_OUTPUT_TOKENS_PREFIX,
    COST_USAGE_UNKNOWN_PREFIX,
    COST_USAGE_PARTIAL_PREFIX,
    SQL_STATEMENTS_PREFIX,
)

_HELP = {
    COST_LLM_CALLS_PREFIX: "LLM calls made, by engine entry.",
    COST_INPUT_TOKENS_PREFIX: "Measured LLM input (prompt) tokens, by entry.",
    COST_OUTPUT_TOKENS_PREFIX: "Measured LLM output (completion) tokens, by entry.",
    COST_USAGE_UNKNOWN_PREFIX: "LLM calls with no provider usage object, by entry.",
    COST_USAGE_PARTIAL_PREFIX: "LLM calls with partial provider usage (one side only), by entry.",
    COST_LLM_CALLS_FAILED_PREFIX: "LLM calls that raised, by entry.",
    SQL_STATEMENTS_PREFIX: "SQL statements issued, by entry.",
}


def escape_label_value(value: str) -> str:
    """Encode a Prometheus label VALUE (backslash, double-quote, newline).

    Label values may never smuggle a newline or a quote into the exposition.
    Combined with the closed-set entry check this is the whole defence: the
    only labels emitted are ``entry`` (closed set), ``reason`` (a fixed token
    set) and ``le`` (a number or ``+Inf``)."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _emit_counter_family(lines: list[str], family: str, samples: dict[str, int]) -> None:
    """Emit one family; samples whose label is not in the closed entry set are
    dropped (they can never have been recorded, but defence in depth)."""
    safe = {e: v for e, v in samples.items() if is_allowed_entry(e) or e == UNKNOWN}
    if not safe:
        return
    lines.append(f"# HELP {family} {_HELP.get(family, '')}".rstrip())
    lines.append(f"# TYPE {family} counter")
    for entry, val in sorted(safe.items()):
        lines.append(
            f'{family}{{entry="{escape_label_value(entry)}"}} {val}'
        )


def render_prometheus() -> str:
    """Serialise counters, gauges and latency histograms to Prometheus text
    exposition format.  NEVER includes storage stats, user ids, or content.
    """
    counters = snapshot()
    gauges_ = gauges()
    lines: list[str] = []

    # Fixed operational counters (no labels).
    fixed = {
        MISSING_PURPOSE: "mirror_memory_missing_purpose_refusals_total",
        DELETION_RECONCILE_FAILURES: "mirror_memory_deletion_reconcile_failures_total",
        TOMBSTONES_APPLIED: "mirror_memory_tombstones_applied_total",
        TOMBSTONES_SKIPPED: "mirror_memory_tombstones_skipped_total",
        TOMBSTONES_EXPORT_SKIPPED: "mirror_memory_tombstones_export_skipped_total",
        TOMBSTONES_RESCANS: "mirror_memory_tombstones_rescans_total",
    }
    for name, pname in fixed.items():
        lines.append(f"# HELP {pname} {name} counter.")
        lines.append(f"# TYPE {pname} counter")
        lines.append(f"{pname} {counters.get(name, 0)}")

    # Tombstone rejections by reason: label is the reason token only.
    rejected: dict[str, int] = {}
    for k, v in counters.items():
        if k.startswith(TOMBSTONES_REJECTED_PREFIX):
            reason = k[len(TOMBSTONES_REJECTED_PREFIX):]
            # Closed set: anything unrecognised is reported under "other" so
            # the count stays visible but no raw string reaches the wire.
            if reason not in ALLOWED_REASONS:
                reason = "other"
            rejected[reason] = rejected.get(reason, 0) + v
    if rejected:
        lines.append("# HELP mirror_memory_tombstones_rejected_total Rejected tombstone batches by reason.")
        lines.append("# TYPE mirror_memory_tombstones_rejected_total counter")
        for reason, v in sorted(rejected.items()):
            lines.append(
                f'mirror_memory_tombstones_rejected_total'
                f'{{reason="{escape_label_value(reason)}"}} {v}'
            )

    # Cost counters grouped by dynamic family.
    for prefix in _DYNAMIC_PREFIXES:
        family_samples: dict[str, int] = {}
        for k, v in counters.items():
            if k.startswith(prefix + "."):
                family_samples[k[len(prefix) + 1:]] = v
        _emit_counter_family(lines, prefix, family_samples)

    # Gauges (fixed names, no labels).
    for name, val in sorted(gauges_.items()):
        pname = f"mirror_memory_{name}"
        lines.append(f"# HELP {pname} {name} gauge.")
        lines.append(f"# TYPE {pname} gauge")
        lines.append(f"{pname} {val}")

    # Latency histograms: ONE family, family name + entry label ONLY.
    pname = "mirror_memory_latency_ms"
    lines.append(f"# HELP {pname} Entry latency in milliseconds, by entry.")
    lines.append(f"# TYPE {pname} histogram")
    for entry, h in histogram_all().items():
        for le, count in h["buckets"]:
            le_label = "+Inf" if le == float("inf") else repr(float(le))
            lines.append(f'{pname}_bucket{{entry="{entry}",le="{le_label}"}} {count}')
        lines.append(f'{pname}_sum{{entry="{entry}"}} {h.get("sum", 0.0)}')
        lines.append(f'{pname}_count{{entry="{entry}"}} {h["count"]}')

    return "\n".join(lines) + "\n"
