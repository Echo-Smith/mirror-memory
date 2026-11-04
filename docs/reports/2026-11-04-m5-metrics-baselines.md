# M5 — Metrics & Recall-Latency Baselines (2026-11-04)

Task 2 (Observability) deliverable for the M5 readiness batch. Records the
recall-latency baseline (feeds Task 5's latency budget) and the measured cost
of the metrics layer itself.

Measurement harness: [`tools/m5_metrics_bench.py`](../../tools/m5_metrics_bench.py).

## Recall latency baseline (p50 / p95, milliseconds)

Beliefs seeded directly as belief rows (no LLM, no extraction) so the number
is the **recall path**, not ingestion. 40 recall samples per size after one
warm-up call. Host: local dev machine, contended (other benches running).

| backend    | beliefs | p50 (ms) | p95 (ms) |
|------------|--------:|---------:|---------:|
| SQLite     |     100 |   16.62  |   31.69  |
| SQLite     |   1,000 |   41.28  |   76.26  |
| SQLite     |  10,000 |  102.89  |  117.50  |
| PostgreSQL |     100 |   21.80  |   42.85  |
| PostgreSQL |   1,000 |   27.28  |   38.59  |
| PostgreSQL |  10,000 |  100.66  |  176.10  |

Notes:
- PostgreSQL leg runs on `postgresql://postgres:mm@127.0.0.1:5437/mm_test`
  with `MM_PG_RLS=0` (the supplied `postgres` login is a superuser; the RLS
  posture check refuses a superuser and a superuser bypasses RLS anyway, so
  this is the raw-latency measurement baseline, perf mode C). Each size seeds
  a **fresh database** so the schema is current.
- p50 grows roughly with the number of active beliefs (scanner + renderer
  work outranks a fixed per-call cost); p95 shows the tail from query-aware
  scoring and audit-commit variance.

## Metrics overhead (deterministic, per recall)

Wall-clock A/B of the full recall loop on a contended machine cannot resolve
a microsecond-scale cost inside a 12–200 ms recall — repeated runs swung by
tens of percent in *both* directions from CPU/scheduler noise, not telemetry.
So the overhead is measured **directly**: the exact metric work a recall adds
(`_metrics_entry` = two `time.monotonic()` + TLS set/reset + one
`observe_latency`; plus `record_sql` once per SQL statement), timed as a
min-of-N tight loop. A typical recall issues 10–25 statements.

| backend    | stmts/recall | added µs/recall | % of recall p50 |
|------------|:------------:|----------------:|----------------:|
| SQLite     |      10      |     +4.39 µs    |    0.0106%      |
| SQLite     |      15      |     +6.81 µs    |    0.0165%      |
| SQLite     |      25      |     +9.80 µs    |    0.0238%      |
| PostgreSQL |      10      |     +5.70 µs    |    0.0209%      |
| PostgreSQL |      15      |     +7.31 µs    |    0.0268%      |
| PostgreSQL |      25      |    +11.07 µs    |    0.0406%      |

**Conclusion: metrics overhead is ≤ 0.041 % of a recall**, far under the 1 %
bar — a same-order-of-magnitude conclusion as the "no extra queries"
principle: the counters/histograms are in-process lock-protected dict/array
updates, and the SQL count rides the engine's existing
`after_cursor_execute` event (no extra statement is ever issued to measure).

## What is measured (summary)

- Counters: LLM calls, input/output tokens (`unknown` when the client returns
  no usage), SQL statements issued — per entry (`observe` / `recall` /
  `extraction_worker`).
- Latency histograms: fixed exponential buckets (bounded memory, thread-safe)
  for `observe`, `recall`, `extraction_worker`; p50/p95/p99.
- Storage: per-user, per-table row counts + estimated bytes via the on-demand
  `metrics.storage_stats(engine)` (never on the request path).
- `/metrics`: Prometheus text, admin-authenticated (`MM_ADMIN_API_KEY`); labels
  limited to `entry`/`reason` — no user ids, no content (locked by
  `tests/test_metrics_observability.py`).
