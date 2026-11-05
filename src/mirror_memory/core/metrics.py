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

import math
import threading
from contextlib import contextmanager
from typing import Iterator

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


def current_entry() -> str | None:
    return getattr(_entry_tls, "entry", None)


@contextmanager
def entry_scope(entry: str) -> Iterator[None]:
    """Mark *entry* as the active attribution target on this thread for the
    span of the ``with`` block.  Nested scopes restore the previous entry on
    exit (including on exception)."""
    previous = getattr(_entry_tls, "entry", None)
    _entry_tls.entry = entry
    try:
        yield
    finally:
        _entry_tls.entry = previous


def record_sql_current(n: int = 1) -> None:
    """Count *n* SQL statements against the CURRENT entry, if any is active.

    This is what the single global cursor listener calls.  When no entry
    scope is open on this thread it is a no-op -- so migrations, other
    engines' pools, and unrelated SQL are never attributed."""
    entry = getattr(_entry_tls, "entry", None)
    if entry is None:
        return
    record_sql(entry, n)


class _LLMClientProxy:
    """Marker base so :func:`instrument_llm_client` is idempotent."""

    _inner: object


def record_llm_tokens(
    entry: str,
    *,
    input_tokens: int | str | None = None,
    output_tokens: int | str | None = None,
) -> None:
    """Add MEASURED token counts for *entry* WITHOUT touching the call count.

    The call itself is counted exactly once by the instrumented client proxy
    (:func:`instrument_llm_client`) or by an explicit ``record_llm``.  A
    provider usage hook calls THIS to contribute the real tokens; when a
    provider surfaces no usage the proxy has already booked the call as
    usage-unknown, so nothing is invented here."""
    if not enabled():
        return

    def _add(prefix: str, tok: int | str | None) -> bool:
        if isinstance(tok, bool) or not isinstance(tok, int):
            return False
        inc(f"{prefix}.{entry}", int(tok))
        return True

    _add(COST_INPUT_TOKENS_PREFIX, input_tokens)
    _add(COST_OUTPUT_TOKENS_PREFIX, output_tokens)


def instrument_llm_client(client):
    """Wrap an LLM client so EVERY ``generate()`` call is counted exactly
    once, for ANY client (M5 fix for the "custom client counts 0" and
    "concurrent hook cross-talk" defects).

    * One ``llm_calls_total.<entry>`` increment per ``generate()`` -- entry
      read from the calling thread's context (no shared mutable state, so no
      cross-talk and no stale hook to restore).  If the provider surfaces no
      usage, the call is also booked as ``llm_usage_unknown_total``.
    * Real tokens, when available, are added by a usage hook the engine
      installs via :func:`record_llm_tokens` (token-only, never a second call
      count).
    * The proxy forwards all other attribute access to the wrapped client, is
      idempotent, and never raises from metric bookkeeping."""
    if client is None:
        return None
    if isinstance(client, _LLMClientProxy):
        return client

    class _Proxy(_LLMClientProxy):
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_inner"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_inner"), name, value)

        def generate(self, *args, **kwargs):
            inner = object.__getattribute__(self, "_inner")
            entry = getattr(_entry_tls, "entry", None)
            try:
                return inner.generate(*args, **kwargs)
            finally:
                if entry is not None and enabled():
                    inc(f"{COST_LLM_CALLS_PREFIX}.{entry}")
                    # If the inner client has NO usage hook, we cannot know
                    # the tokens -> book the call as usage-unknown so the
                    # token counters stay honest.
                    if not getattr(inner, "_mm_usage_hook", False):
                        inc(f"{COST_USAGE_UNKNOWN_PREFIX}.{entry}")

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
    """The counter snapshot -- ``snapshot()`` keeps its historical shape
    (``dict[str, int]``) so existing callers (``snapshot().get(name, 0)``)
    keep working.  The richer view lives in :func:`report`."""
    with _lock:
        return dict(_counters)


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

#: Entries whose cost we attribute.  Closed set.
ENTRY_OBSERVE = "observe"
ENTRY_RECALL = "recall"
ENTRY_EXTRACTION_WORKER = "extraction_worker"
#: The metabolism cycle is its OWN entry -- it must not be folded into
#: ``observe`` (that would pollute observe's latency/token accounting).
ENTRY_METABOLISM = "metabolism"

COST_LLM_CALLS_PREFIX = "llm_calls_total"        # + ".<entry>"
COST_INPUT_TOKENS_PREFIX = "llm_input_tokens"    # + ".<entry>"
COST_OUTPUT_TOKENS_PREFIX = "llm_output_tokens"  # + ".<entry>"
#: Calls whose provider surfaced no usage object -> counted separately so
#: the token counters are never inflated by a guess.
COST_USAGE_UNKNOWN_PREFIX = "llm_usage_unknown_total"  # + ".<entry>"

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
    if not enabled():
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
    """Record *n* SQL statements issued for *entry* (count only)."""
    if not enabled():
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
    """Record one latency sample (milliseconds) for *entry*."""
    if not enabled():
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
# Storage growth (M5 task 2 requirement 3)
#
# Row counts and estimated bytes PER USER, PER TABLE, by an ON-DEMAND SQL
# pass -- never on the request path.  Call it from an admin endpoint, a
# low-frequency worker tick, or a benchmark.  SQLite uses
# ``length(CAST(col AS BLOB))``; PostgreSQL uses ``octet_length(col)``.
# The aggregate read crosses tenants by design (a capacity report), so it
# runs inside ``system_scope`` with a recorded reason; it is never reached
# from an engine request path.
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


def _storage_rows(engine, *, where: tuple[str, tuple[dict, ...]] | None):
    """Shared body: one aggregate per tenant table, optionally filtered.

    ``where`` is ``(sql_fragment, params_list)`` where the fragment is a
    trailing predicate appended to the WHERE clause (e.g.
    ``"WHERE tenant_id = :t AND app_id = :a"``), or ``None`` for the
    explicitly-privileged all-tenant report.  Identifiers (table/column) come
    from the DB catalog, never from user input; only VALUES are bound.
    """
    from sqlalchemy import inspect as _inspect
    from sqlalchemy import text as _text

    out: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    inspector = _inspect(engine)
    tables = set(inspector.get_table_names())
    groups = ("tenant_id", "app_id", "user_id")
    with engine.connect() as conn:
        for table in sorted(tables):
            if not table.startswith("mm_"):
                continue
            cols = {c["name"] for c in inspector.get_columns(table)}
            if not set(groups) <= cols:
                continue
            size_cols = [c for c in _ESTIMATE_COLUMNS.get(table, ()) if c in cols]
            if size_cols:
                if engine.dialect.name == "postgresql":
                    size_expr = " + ".join(
                        f"COALESCE(octet_length({c}), 0)" for c in size_cols
                    )
                else:
                    size_expr = " + ".join(
                        f"COALESCE(length(CAST({c} AS BLOB)), 0)" for c in size_cols
                    )
                cols_sql = (
                    f"tenant_id, app_id, user_id, COUNT(*) AS n, "
                    f"COALESCE(SUM({size_expr}), 0) AS b"
                )
            else:
                cols_sql = "tenant_id, app_id, user_id, COUNT(*) AS n, 0 AS b"
            frag = where[0] if where else ""
            params = where[1] if where else ()
            group_sql = "GROUP BY tenant_id, app_id, user_id"
            sql = f"SELECT {cols_sql} FROM {table} {frag} {group_sql}"
            for tenant_id, app_id, user_id, n, b in conn.execute(_text(sql), params):
                key = (str(tenant_id), str(user_id))
                out.setdefault(key, {})[table] = {"rows": int(n), "bytes": int(b)}
    return out


def storage_stats(engine, *, tenant: tuple[str, str]) -> dict[tuple[str, str], dict[str, dict[str, int]]]:
    """Per-(tenant, user), per-table row counts + estimated bytes for ONE
    tenant, with an EXPLICIT ``WHERE tenant_id = :t AND app_id = :a`` filter.

    This is the safe default: it runs on the ordinary engine, and the filter
    is applied in SQL, so it returns only the caller's tenant even when RLS
    is switched off (SQLite).  Under PostgreSQL + RLS both the explicit
    predicate AND the row-security policy apply -- defence in depth.

    Returns ``{(tenant_id, user_id): {table: {"rows": int, "bytes": int}}}``.
    User ids live ONLY in this library return value and in
    :func:`report`'s ``storage`` section -- never in a Prometheus label.
    """
    tenant_id, app_id = tenant
    if not tenant_id or not app_id:
        raise ValueError("storage_stats requires a non-empty (tenant_id, app_id)")
    where = (
        "WHERE tenant_id = :t AND app_id = :a",
        {"t": tenant_id, "a": app_id},
    )
    return _storage_rows(engine, where=where)


def storage_stats_all_tenants(engine) -> dict[tuple[str, str], dict[str, dict[str, int]]]:
    """Per-(tenant, user) storage for an EXPLICITLY PRIVILEGED report.

    This crosses tenants BY DESIGN (a capacity report).  Two independent
    requirements make it *verifiable* rather than a silent escalation:

    1. It runs inside ``system_scope(reason=...)`` -- the ONE sanctioned
       cross-tenant bypass.  That scope is (a) recorded by the SQL auditor
       with its reason, (b) subject to the ``tools/check_system_scope.py``
       allowlist, so this call site is registered and reviewable.  It is a
       LOGICAL scope, not a DB GRANT.
    2. ``system_scope`` does NOT elevate database privileges.  On PostgreSQL
       + RLS this function therefore returns only what the CALLER's engine
       can see: to get a true all-tenant reading the caller MUST pass the
       maintenance/owner engine (``core.rls.maintenance_engine_for``) which
       carries the real elevated DB role.  Passing the ordinary app engine
       yields the app role's (RLS-filtered) view -- no silent privilege gain.
       On SQLite there is no RLS, so the ordinary engine is fine.

    Returns the same shape as :func:`storage_stats` (keys carry tenant+user,
    so a shared ``user_id`` across tenants is never conflated).
    """
    from mirror_memory.core.tenancy import system_scope

    with system_scope(reason="metrics.storage_stats_all_tenants: privileged capacity report"):
        return _storage_rows(engine, where=None)


# ---------------------------------------------------------------------------
# The rich report (backward-compatible ``snapshot()`` stays untouched)
# ---------------------------------------------------------------------------


def report(*, storage: dict | None = None) -> dict:
    """Everything the library knows: counters, gauges, latency summaries,
    and (only when explicitly computed) storage stats.

    ``storage`` is passed in by the caller that just ran :func:`storage_stats`
    -- the request path never computes it.
    """
    return {
        "counters": snapshot(),
        "gauges": gauges(),
        "latency": latency_summaries(),
        "storage": storage if storage is not None else {},
    }


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
    _entry_tls.entry = None


# ---------------------------------------------------------------------------
# Prometheus text exposition (M5 task 2 requirement 5)
#
# Labels are a CLOSED set: metric family name + (for histograms) ``entry``.
# There is deliberately no ``user``/``tenant``/``purpose`` label anywhere,
# so a user id cannot reach the wire even by accident.
# ---------------------------------------------------------------------------

_DYNAMIC_PREFIXES = (
    COST_LLM_CALLS_PREFIX,
    COST_INPUT_TOKENS_PREFIX,
    COST_OUTPUT_TOKENS_PREFIX,
    COST_USAGE_UNKNOWN_PREFIX,
    SQL_STATEMENTS_PREFIX,
)

_HELP = {
    COST_LLM_CALLS_PREFIX: "LLM calls made, by engine entry.",
    COST_INPUT_TOKENS_PREFIX: "Measured LLM input (prompt) tokens, by entry.",
    COST_OUTPUT_TOKENS_PREFIX: "Measured LLM output (completion) tokens, by entry.",
    COST_USAGE_UNKNOWN_PREFIX: "LLM calls with no provider usage object, by entry.",
    SQL_STATEMENTS_PREFIX: "SQL statements issued, by entry.",
}


def _emit_counter_family(lines: list[str], family: str, samples: dict[str, int]) -> None:
    if not samples:
        return
    lines.append(f"# HELP {family} {_HELP.get(family, '')}".rstrip())
    lines.append(f"# TYPE {family} counter")
    for entry, val in sorted(samples.items()):
        lines.append(f'{family}{{entry="{entry}"}} {val}')


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
    rejected = {
        k[len(TOMBSTONES_REJECTED_PREFIX):]: v
        for k, v in counters.items() if k.startswith(TOMBSTONES_REJECTED_PREFIX)
    }
    if rejected:
        lines.append("# HELP mirror_memory_tombstones_rejected_total Rejected tombstone batches by reason.")
        lines.append("# TYPE mirror_memory_tombstones_rejected_total counter")
        for reason, v in sorted(rejected.items()):
            lines.append(f'mirror_memory_tombstones_rejected_total{{reason="{reason}"}} {v}')

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
