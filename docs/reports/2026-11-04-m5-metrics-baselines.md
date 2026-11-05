# M5 — Metrics & Recall-Latency Baselines (2026-11-05 update)

Deliverable for M5 §1 (observability) + §5b (overhead ≤1%, recall baselines).
This file records the **current** measurements; the earlier 2026-11-04 numbers
were taken while several benches contended the same host and are superseded by
the runs below.

Environment: Python 3.12.13, SQLAlchemy 2.1.5, FastAPI 0.153.0,
pydantic 2.14.1, PostgreSQL 16 (local Docker, `postgres@127.0.0.1:5437/mm_test`).
Host is a shared workstation — absolute latencies are indicative, not portable.

Harnesses:
- recall baselines: [`tools/m5_metrics_bench.py`](../../tools/m5_metrics_bench.py)
- metrics overhead: [`tools/m5_metrics_overhead.py`](../../tools/m5_metrics_overhead.py)

## Recall latency baseline (p50 / p95, milliseconds)

Beliefs seeded directly as belief rows (no LLM, no extraction), so the number
is the **recall path**, not ingestion. 40 recall samples per size after one
warm-up call. A fresh database per size.

| backend    | beliefs | p50 (ms) | p95 (ms) |
|------------|--------:|---------:|---------:|
| SQLite     |     100 |    7.4   |   14.4   |
| SQLite     |   1,000 |   14.0   |   14.7   |
| SQLite     |  10,000 |   93.8   |  226.7   |
| PostgreSQL |     100 |   25.0   |   49.5   |
| PostgreSQL |   1,000 |   37.6   |   75.5   |
| PostgreSQL |  10,000 |   84.4   |  104.6   |

- PostgreSQL leg uses `MM_PG_RLS=0` (superuser direct connection) for a
  raw-latency baseline; the RLS posture check refuses a superuser, and a
  superuser bypasses RLS, so this is the documented measurement-only shape.
- p50 grows with the number of active beliefs (scanner + renderer work
  outranks a fixed per-call cost); p95 tracks query-aware scoring + the
  audit-commit tail.

## Metrics overhead — controlled experiment

Two experiments, reported honestly:

1. **Full-longrun ON/OFF pairs** (50–70 s per run, 3 pairs on SQLite): wall and
   CPU ratios swung between **+0.6 % and +34.7 %** (median +10.7 %). That is
   machine-contention noise, not metric cost — the sign flipped between pairs,
   which a real overhead cannot do. **Inconclusive on this host.**
2. **Paired-block experiment (decisive)** — one process, same database, 8
   alternating blocks of 60 real `MemoryEngine.observe()` calls (960 observes),
   measuring **CPU time** (`process_time`), which is robust to scheduler noise:

| block | ON CPU (ms) | OFF CPU (ms) | CPU Δ |
|------:|------------:|-------------:|------:|
| 1 | 43.7 | 42.2 | +3.49 % |
| 2 | 42.1 | 42.0 | +0.33 % |
| 3 | 42.1 | 41.9 | +0.52 % |
| 4 | 42.1 | 41.9 | +0.41 % |
| 5 | 42.2 | 41.9 | +0.61 % |
| 6 | 42.2 | 42.0 | +0.53 % |
| 7 | 42.1 | 41.9 | +0.42 % |
| 8 | 42.1 | 41.8 | +0.65 % |

**CPU median delta +0.526 % (wall +0.527 %) → ≤ 1 % gate PASS.** Block 1 is the
warm-up effect; blocks 2–8 cluster at +0.33 %…+0.65 %.

Deterministic cross-check (exact metric work a recall adds, min-of-N):

| backend | stmts/recall | added µs/recall | % of recall p50 |
|---|---:|---:|---:|
| SQLite | 10 | +7.59 | 0.054 % |
| SQLite | 15 | +11.02 | 0.079 % |
| SQLite | 25 | +14.87 | 0.106 % |
| PostgreSQL | 10 | +6.45 | 0.017 % |
| PostgreSQL | 15 | +8.98 | 0.024 % |
| PostgreSQL | 25 | +14.94 | 0.040 % |

Overhead comes from the per-observe entry scope, fixed-bucket histogram updates
and one counter per SQL statement — all lock-protected in-process work. The SQL
count rides the engine's existing `after_cursor_execute` event, so **no extra
query is ever issued**.

## What is measured

- Counters per entry (`observe` / `recall` / `extraction_worker` / `metabolism`):
  LLM calls, input/output tokens (real values only; `unknown` when the provider
  surfaces no usage), SQL statements.
- Latency histograms (fixed exponential buckets, constant memory): p50/p95/p99
  per entry.
- Storage: per-`(tenant, user)` row counts + estimated bytes via the on-demand
  `metrics.storage_stats(engine, tenant=(t, a))` — explicit tenant filter, never
  on the request path. The privileged all-tenant report is
  `metrics.storage_stats_all_tenants`, separately allowlisted.
- `/metrics`: Prometheus text, admin-authenticated (`MM_ADMIN_API_KEY`,
  fail-closed 404 when unset). Labels limited to `entry` / `reason` / `le` —
  no user ids, no content; locked by `tests/test_metrics_observability.py` and
  `tests/test_metrics_integration.py`.