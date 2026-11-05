# M5 §5a — PostgreSQL longrun diagnosis (M2 → M3 → M4), balanced order

**Scope:** DIAGNOSIS ONLY. No optimisation code lands in `src/`.
**Branch / revisions:** `feat/m5-readiness`; M2=`403f38d`, M3=`aed0639`,
M4=`d61e7b7`.
**Mode:** A = production today (RLS on + application-layer read filter;
`MM_TENANT_READ_FILTER=app`, SQL auditor off in every leg).
**Database:** local Docker PostgreSQL 16, `postgres@127.0.0.1:5437/mm_test`,
tenant work through the **app role** under RLS.
**Workload unit:** one full `run_longrun` (510 `observe` entries) via
`tests/test_longrun_bench.py::TestLongRun::test_year_survives_and_reports`.
**Tools / artifacts:** [`tools/m5_perf_diag.py`](../../tools/m5_perf_diag.py);
raw JSON under `/tmp/m5_perf_bal/` (`measure.json`, `statements_M*.json`,
`profile_M*.json`, `explain_top5.json`).

Each revision is materialised in its **own detached git worktree**
(`/tmp/mm-perf-403|aed|d61`) and imported via `PYTHONPATH=<worktree>/src`, so the
shared checkout's `src/`/`tests/` are never swapped during measurement.

---

## (a) Three-point timing — balanced / rotated order, 3 rounds

Interleaving rotates the order every round so each revision occupies a
different position (round 1: M2→M3→M4; round 2: M3→M4→M2; round 3: M4→M2→M3).
That removes the position/last-run bias a fixed order injects on a contended
host. Every round was **VALID** (each leg: 1 passed, 0 failed, 0 errors).

| rev | round 1 (s) | round 2 (s) | round 3 (s) | **min** | **median** | min / M2 | median / M2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2 (403f38d) | 7.3 | 7.1 | 6.7 | **6.7** | **7.1** | 1.000 | 1.000 |
| M3 (aed0639) | 7.5 | 7.9 | 7.9 | **7.5** | **7.9** | **1.119** | **1.113** |
| M4 (d61e7b7) | 7.9 | 8.0 | 8.0 | **7.9** | **8.0** | **1.179** | **1.127** |

**Reading:** the increment is **M2→M3 +11.3 % and M3→M4 +1.3 % (median),
cumulative M2→M4 +12.7 %** (min column: +11.9 % / +5.4 % / +17.9 %). The
ordering M4 ≥ M3 ≥ M2 held in all three rounds regardless of position, so —
unlike the earlier fixed-order sweep — this is **not** a last-run artifact.

**Honest caveats (read before quoting any single number):**
1. **Absolute seconds are not portable.** The host is a shared workstation and
   unrelated GUI workloads pushed load average to 8–12 during measurement. Only
   *within-sweep ratios* and the *statement counts* below are meaningful.
2. These absolute values (6.7–8.0 s) are for ONE `run_longrun`; the earlier
   report's 158–254 s figures were the full 18-test file (≈18× the same
   scenario) under heavier contention — **superseded, do not use for ranking**.
3. Latency percentiles are not attempted here: the unit is whole-run wall time,
   not a latency distribution.

---

## (b) Increment source split (load-invariant)

Two load-independent measurements: a `before_cursor_execute` listener bucketing
each statement by the issuing `mirror_memory`/`tests` frame, and cProfile's
`_execute_context` round-trip count.

| source | M2 | M3 | M4 |
|---|---:|---:|---:|
| purpose authorisation gate | 0 | 1062 | 1062 |
| cross-purpose visibility filter | 0 | 3689 | 3699 |
| access-audit writes | 0 | 41 | 41 |
| deletion / snapshot / generation checks | 16 | 16 | 52 |
| sync catch-up gate | 2 | 2 | 2 |
| bench grant (one-off) | 0 | 5 | 5 |
| other (baseline ORM) | 17276 | 13095 | 13242 |
| **total** | **17294** | **17910** | **18103** |
| **per observe entry** | **33.9** | **35.1** | **35.5** |

Deltas: **M2→M3 +616 statements (+3.6 %, ≈ +1.2 per entry)**;
**M3→M4 +193 (+1.1 %, ≈ +0.4 per entry)**.

The "other" bucket shrinks M2→M3 because statements that were plain unfiltered
reads in M2 now carry the cross-purpose predicate and are attributed to the
visibility bucket — the **net** new load is the total delta, not the
reclassification inside "other".

**cProfile (same workload; round-trips are the load-invariant signal)**

| rev | `_execute_internal` calls | `_persist_claims` cumulative | run wall (profiled) |
|---|---:|---:|---:|
| M2 | 11,842 | 11.56 s | 17.4 s |
| M3 | 12,390 | 9.75 s | 15.4 s |
| M4 | 12,442 | 11.31 s | 17.2 s |

Round-trip growth **+548 / +52** matches the statement deltas. The dominant cost
in every revision is `_persist_claims` — the baseline ingest write path that
exists **identically in M2** — not the increment.

**EXPLAIN (ANALYZE, BUFFERS) — five hottest statements (M4, app role, RLS on)**

| # | calls | statement (head) | execution | RLS predicate |
|---|---:|---|---:|---|
| 1 | 32,993 | `SELECT mm_memory_preferences …` | 0.011 ms | ✅ One-Time Filter |
| 2 | 18,238 | `SELECT mm_memory_preferences.state_revision …` | 0.006 ms | ✅ |
| 3 | 15,535 | `SELECT mm_evidence …` | 0.028 ms | ✅ |
| 4 | 10,813 | `SELECT mm_evidence …` (other param shape) | 0.022 ms | ✅ |
| 5 | 7,682 | `SELECT mm_proposal_log …` | 0.051 ms | (Limit/Result) |

Every tenant statement carries RLS as a cheap `One-Time Filter`
(`current_setting('mm.app_id')/('mm.tenant_id')`) on top of the row's own
tenant column predicate. Plans are `Result`/`Limit` over indexed probes on a
small schema — **no plan regression**.

**Diagnosis in one line:** the M2→M3→M4 increment is a **bounded per-entry
statement-count increase (+1.2 then +0.4 SQL round-trips per observe entry)** —
not super-linear, not an RLS plan regression. Host-level wall spread is
contention, not the diff.

---

## (c) Optimisation proposals (diagnosis only; none shrinks a guard)

1. **Cache the per-transaction decision row** (grants + consent + preferences;
   statements #1/#2/#5). ≈ **−4,400 of ~18,100 statements (−24 %)**, 35.5 →
   ~27/entry. Risk: a mid-transaction revoke must invalidate (the existing
   `version` / `state_revision` CAS columns are the hook); intra-transaction
   only. **Guard scope unchanged** — the same authorisation SQL still runs; only
   repeated reads of the *identical row within one transaction* are skipped, and
   RLS still applies to the cached load.
2. **Batch / merge the visibility-filter probes** (3,689 statements, ≈7/entry)
   into the parent query. ≈ **−1,800 (−10 %)**. Risk: larger SQL; the per-row
   predicate must stay (never become a post-fetch Python filter); read path
   only. **Guard scope unchanged** — every row is still filtered; no RLS/tenant
   predicate is dropped.
3. **One `set_config` per transaction** (603 calls). ≈ **−600 (−3 %)**, zero
   semantic change. Risk: the GUC must be set before any tenant statement in
   every transaction and must keep `LOCAL` scope so a pooled connection cannot
   leak it. **Guard scope unchanged** — a fresh transaction still sets the GUC;
   RLS still enforces from `current_setting`.

Combined realistic target: **~35 → ~22–24 statements per observe entry
(−30…35 % round-trips)** with every authorisation, RLS, visibility and
deletion/sync check intact.

---

## (d) Reproduce

```bash
git worktree add --detach /tmp/mm-perf-403 403f38d
git worktree add --detach /tmp/mm-perf-aed aed0639
git worktree add --detach /tmp/mm-perf-d61 d61e7b7

MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test \
  .venv/bin/python tools/m5_perf_diag.py measure    --rounds 3 --work /tmp/m5_perf_bal
MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test \
  .venv/bin/python tools/m5_perf_diag.py statements --work /tmp/m5_perf_bal
MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test \
  .venv/bin/python tools/m5_perf_diag.py profile    --work /tmp/m5_perf_bal

# EXPLAIN top-5 (reuses the repo's own PG/RLS fixtures):
MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test MM_PG_RLS=1 \
MM_EXPLAIN_CAPTURE=1 MM_EXPLAIN_OUT=/tmp/m5_perf_bal/explain_top5.json \
  .venv/bin/python -m pytest tests/test_m5_explain_capture.py -q -s -p no:cacheprovider

.venv/bin/python tools/m5_perf_diag.py cleanup
```

`MEASURE_TEST=all` switches the measure command back to the whole 18-test file
(CI-shaped, ≈18× the wall time); the default measures the single representative
scenario run.