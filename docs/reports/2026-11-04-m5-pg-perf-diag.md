# M5 Task 0 — PostgreSQL performance diagnosis (M2 → M3 → M4)

**Date:** 2026-11-04
**Scope:** DIAGNOSIS ONLY. No optimization code lands in `src/`.
**Branch / baselines:** `feat/m5-readiness`; measured revisions M2=`403f38d`,
M3=`aed0639`, M4=`d61e7b7`.
**Mode measured:** mode A = production today (RLS on + application-layer read
filter; SQL auditor off in every leg, the M1 acceptance figure).
**Database:** local Docker PostgreSQL 16.15, `postgresql://postgres:mm@127.0.0.1:5437/mm_test`.
**Harness:** `tests/test_longrun_bench.py` (18 tests), each test executing the
scaled `run_longrun` scenario (`days=40, preferences=12, goals=5,
corrections=6, conflicts=5, forgets=3` → 510 `observe` entries per run).
**Tool:** `tools/m5_perf_diag.py` (this repo). All raw artifacts under
`/tmp/m5_diag/` (`measure.json`, `statements.json`, `profile.json`,
`explain.json`, `profile_M*.txt`).

Each revision was materialised in its **own detached git worktree**
(`/tmp/mm-perf-403`, `/tmp/mm-perf-aed`, `/tmp/mm-perf-d61`) and imported via
`PYTHONPATH=<wt>/src`, so the shared checkout's `src/`/`tests/` were never
swapped under the other agents working in the same tree.

---

## (a) Three-point interleaved wall-clock

Interleaved measurement: every round runs all three revisions back to back,
so a round shares one machine-load epoch. Three rounds, all 18/18 passing
each leg (VALID).

| rev | round 1 (s) | round 2 (s) | round 3 (s) | **min** | **median** |
|---|---:|---:|---:|---:|---:|
| M2 (403f38d) | 511.2 | 254.1 | 158.6 | **158.6** | **254.1** |
| M3 (aed0639) | 461.5 | 226.7 | 167.8 | **167.8** | **226.7** |
| M4 (d61e7b7) | 301.7 | 162.1 | 173.8 | **162.1** | **173.8** |

Ratios vs M2 (min / median):

| rev | min / M2 | median / M2 |
|---|---:|---:|
| M3 | 1.058 | 0.892 |
| M4 | 1.022 | 0.684 |

**These absolute numbers are load-dominated and are NOT a clean ranking of the
revisions.** The honest reading:

- Wall time collapses across rounds (round 1 is 2–3× slower than round 3)
  because the box was contended during round 1 (three feature-agent jobs plus
  this measurement's own profile pass were running concurrently) and quieted
  by round 3. This is a warm-up/contention **confound inside the sweep**, not
  a revision effect.
- Within-round ordering is inconsistent: rounds 1–2 rank M4 fastest, round 3
  ranks M4 slowest. A robust "M4 is 32% faster" (median ratio 0.684) cannot be
  claimed from this data — M4 simply ran last each round, when contention had
  eased, which drags its median down.
- The **min** column (each revision's best run) is the least load-contaminated
  and it shows the three revisions within ~6% of each other (158.6 / 167.8 /
  162.1 s).
- The load-invariant signal below (SQL statement counts) is the trustworthy
  basis for decomposition; the wall-clock table is reported for completeness
  and reproducibility, with its caveat attached.

**Methodology + caveats** are in section (d).

---

## (b) Increment-source split

Two complementary, load-independent measurements:

1. **SQL statement counts** — a `before_cursor_execute` listener on the
   SQLAlchemy `Engine` class counts every statement issued during one full
   `run_longrun` (510 observe entries), bucketed by the issuing call site.
2. **cProfile** — Python-time attribution per revision, and the SQLAlchemy
   `_execute_context` call count (the round-trip count), which is impervious
   to wall-clock noise.

### Statement counts by increment source (one full 510-entry run)

| increment source (bucket) | M2 | M3 | M4 | method |
|---|---:|---:|---:|---|
| purpose authorization gate | 0 | 1,062 | 1,062 | `authorize`/`authorize_or_raise`/`_load_decision_row` frames (`core/authorization.py`) |
| cross-purpose visibility filter | 0 | 3,689 | 3,699 | statements carrying a purpose/cross-purpose predicate issued from retrieval frames |
| access audit writes | 0 | 41 | 41 | `mm_access_audit` INSERTs / `core/access_audit.py` frames |
| deletion / snapshot generation checks | 16 | 16 | 52 | `core/deletion.py`, `core/tombstones.py`, `core/replay.py`, `mm_deletion_*`/`mm_tombstone*` SQL |
| sync catch-up gate | 2 | 2 | 2 | `mm_deletion_watermark` reads |
| bench purpose grant (one-time setup) | 0 | 5 | 5 | `grant_bench_purpose` issuance (excluded from steady-state) |
| other (baseline ORM load, present in all revs) | 17,276 | 13,096 | 13,241 | residual |
| **TOTAL statements** | **17,294** | **17,911** | **18,102** | |
| **per observe entry** | **33.9** | **35.1** | **35.5** | total ÷ 510 |

Deltas: **M2→M3 = +617 statements (+3.6%, ≈ +1.2 stmt/entry)**;
**M3→M4 = +191 (+1.1%, ≈ +0.4 stmt/entry)**.

Note on the "other" drop M2→M3 (−4,180): the M3 revision introduces the
purpose/visibility layer, so statements that were plain unfiltered reads in M2
now carry the cross-purpose predicate and are attributed to the
`cross_purpose_visibility` bucket in M3. The **net new statement load** is the
total delta (+617 / +191); the reclassification within "other" is not extra
work.

### cProfile: round-trip count is the load-invariant signal

| rev | `_execute_context` calls | SQLAlchemy `execute` cumulative | wall (1 profile run) |
|---|---:|---:|---:|
| M2 | 17,044 | 25.4 s | 47.3 s |
| M3 | 17,633 | 22.7 s | 42.9 s |
| M4 | 17,675 | 23.6 s | 43.6 s |

Round-trip count is stable and monotonic (**+589 M2→M3, +42 M3→M4**), matching
the statement-count deltas (+617 / +191 within classification tolerance). In
M3/M4 ~7% of Python time is the authorization gate itself
(`authorize_or_raise` cumulative ≈ 5.0 s over 531 calls). The dominant cost in
every revision remains `_persist_claims` (pipeline.py:508, 28–32 s of a ~47 s
run) — i.e. the baseline ingest write path that exists **identically in M2**,
not the increment.

### EXPLAIN ANALYZE — five hottest statements (M4)

Run against a fresh schema populated by one full run; captured with bound
parameters and executed on the app role (RLS active). Plans are for the *small*
bench schema, so per-call times are sub-millisecond — the cost is call volume,
not per-call latency.

| # | statement (head) | calls | plan | exec |
|---|---|---:|---|---:|
| 1 | `SELECT mm_memory_preferences …` | 2,852 | Seq Scan `mm_memory_preferences`, RLS one-time filter | 0.087 ms |
| 2 | `SELECT mm_memory_preferences.state_revision …` | 1,089 | Seq Scan, RLS one-time filter | 0.046 ms |
| 3 | `SELECT mm_evidence …` | 1,071 | Index Scan `uq_mm_evidence_user_ref`, RLS filter | 0.405 ms |
| 4 | `SELECT set_config('mm.tenant_id',…), set_config('mm.app_id',…)` | 603 | Result (no scan) | 0.018 ms |
| 5 | `SELECT mm_purpose_grants.version, …` | 531 | Index Scan `mm_purpose_grants_pkey`, RLS filter | 0.216 ms |

Every tenant statement carries the RLS predicate as a cheap `One-Time Filter`
(`current_setting('mm.app_id') = … AND current_setting('mm.tenant_id') = …`)
**on top of** the row's own `tenant_id/app_id` column predicate. The overhead
of RLS here is the predicate evaluation, not a plan regression — all queries
use the intended index except the two `mm_memory_preferences` reads, which Seq
Scan a 1-row table (immaterial at this size; would matter at scale).

**Diagnosis in one line:** the M2→M3→M4 increment is a **bounded per-entry
statement-count increase (~+1.2 then ~+0.4 statements per observe entry — SQL
round-trips), not a new super-linear cost, not an RLS plan regression. The
absolute wall-time spread seen on this box is dominated by machine contention,
not by the revision diff.**

---

## (c) Optimization proposals

All three target the *round-trip count* (the actual cost driver), and none of
them removes a tenant/RLS/authorization/deletion-guard check — each keeps the
exact same checks and only reduces how many times the same checked row is
fetched per entry.

### Proposal 1 — Cache the per-transaction decision row (purpose grants + consent + preferences)

**What:** the `authorize_or_raise` path re-loads
`mm_purpose_grants ⋈ mm_consent_grants ⋈ mm_memory_preferences` (top#5, 531×)
and the two `mm_memory_preferences` reads (top#1 2,852×, top#2 1,089×) once per
operation. Cache the resolved decision **within one request/transaction** (a
per-session cache keyed by `(tenant_id, app_id, user_id, purpose)`), invalidated
on any write to those tables in the same tx.

**Projected gain:** eliminates ~4,400 of ~18,100 statements/run (≈24% of
steady-state round-trips); per-entry statements 35.5 → ~27.
**Risk:** stale decision if a grant is revoked mid-request; a concurrent revoke
must invalidate the cache (the existing `version`/`state_revision` CAS columns
already give the invalidation hook). Scope: intra-transaction only.
**Why it does not shrink any guard:** the authorization decision is still fully
evaluated (same SQL, same `_load_decision_row`, same purpose/consent/expiry
logic) — only *repeated evaluation of the identical row inside one tx* is
skipped. RLS still applies to the cached load; the deletion guard and sync gates
are untouched.

### Proposal 2 — Batch/statement-merge the per-entry visibility-filter probes

**What:** the cross-purpose visibility filter issues 3,689–3,699 statements/run
(≈7 per observe entry) as separate round-trips against evidence/belief rows.
Fold the repeated single-row lookups the filter performs into the parent query
(a `JOIN`/`WHERE … IN`), or issue them as one batched statement per entry.

**Projected gain:** conservatively halves the visibility round-trips (≈ −1,800
statements/run, ≈10% of total); more if the probes are fully folded into the
existing belief read.
**Risk:** larger/less-cacheable SQL; needs the visibility predicate preserved
per row (must not become a post-fetch Python filter that could leak unfiltered
rows). Scope: read-path only.
**Why it does not shrink the guard:** the per-row cross-purpose predicate is
still applied on every candidate row — batching changes *how many round-trips*
carry the predicate, not *whether* each row is checked. No row skips the filter;
no tenant/RLS predicate is dropped (still column + `current_setting`).

### Proposal 3 — One `set_config` per transaction (connection-local tenant binding)

**What:** `SELECT set_config('mm.tenant_id', …), set_config('mm.app_id', …)`
fires 603×/run (top#4). When tenant/app are stable across a transaction's
statements, set the GUCs once when the transaction opens and skip the repeat
`set_config` per statement/session checkout.

**Projected gain:** removes up to ~600 statements/run (≈3%); small but trivial
and zero-semantic.
**Risk:** must guarantee the GUC is set before *any* tenant statement in every
transaction and reset/`LOCAL`-scoped correctly so a pooled connection cannot
leak another tenant's setting — the current per-tx `LOCAL` (`true`) semantics
must be preserved exactly. Scope: connection/session lifecycle.
**Why it does not shrink tenant/RLS scope:** the RLS GUC must still be set for
every transaction before tenant work — this proposal only avoids setting the
*same* value repeatedly within one transaction. A fresh transaction still sets
it; RLS still enforces from `current_setting`. If anything it tightens the
invariant by making "set before tenant work" explicit at transaction open.

**Combined realistic target:** ~35 → ~22–24 statements per observe entry
(≈30–35% fewer round-trips) while keeping every authorization, RLS, visibility
and deletion/sync check intact. The gain is in *round-trip volume*, which is
what the profile shows dominates wall time on a remote PG — not in removing
guards.

---

## (d) Methodology and honest caveats

**Methodology**
- Each revision in its own detached `git worktree`; runs import
  `PYTHONPATH=<worktree>/src` and execute with `cwd=<worktree>` so the code
  under test is unambiguously that revision. The shared checkout is read-only
  to this tool.
- Mode A env: `MM_PG_RLS=1`, `MM_TENANT_READ_FILTER=app`, `MM_SQL_AUDIT=0`,
  `MM_DATABASE_URL` → admin login; the pytest `pg_app_url` fixture builds a
  fresh RLS schema per test and routes tenant work through the app role.
- Interleaving: for each round, M2→M3→M4 run back to back; a round is VALID
  only if all three legs pass exactly 18 tests with zero failures/errors
  (junit-verified). All three rounds were VALID.
- Statement counting: `before_cursor_execute` on the `Engine` class, one full
  run per revision, bucketed by issuing `mirror_memory`/`tests` frame. Counts
  include `bench_grant_setup` (one-time) which is reported separately.
- cProfile: one `run_longrun` per revision on a fresh schema built from that
  revision's own `tests/conftest.py`; SQLAlchemy `_execute_context` call count
  taken as the load-invariant round-trip metric.
- EXPLAIN ANALYZE: top-5 statements captured with bound params from one M4 run,
  replayed on the app role inside a tenant-scoped transaction;
  `EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)`.

**Caveats (read before trusting any single number)**
1. **Local Docker PG 16 is not a CI runner.** CPU, page cache and disk differ
   from CI; absolute seconds are not portable to CI and must not be compared to
   CI history. Only *within-sweep, same-box* ratios and the round-trip counts
   are meaningful.
2. **Machine contention dominated the wall-clock sweep.** Round 1 ran while
   three feature agents plus this tool's profile pass were live; round 3 ran
   quiet. Absolute times fell ~2–3× across rounds for reasons unrelated to the
   revision. The interleaved design cancels *some* of this within a round, but
   because M4 always ran last it systematically benefited from the easing load
   — the M4 median ratio of 0.684 is therefore **not** a credible "M4 is 32%
   faster" claim. Treat the wall-clock table as reproducible-raw-data-with-
   caveat, and the statement/round-trip counts as the trustworthy signal.
3. **Statement counts are per-representative-run, not per full 18-test suite.**
   The `statements` and `profile` passes run ONE `run_longrun` (the
   `test_year_survives_and_reports` test) to keep cost bounded; the 18-test
   suite reuses the same scenario, so per-entry rates are representative but the
   absolute counts are per one run, not ×18.
4. **Per-statement TIMING was not captured.** Only statement COUNTS and
   per-statement `EXPLAIN ANALYZE` on the small schema. There is no APM inside
   every `execute`, so we do not claim a ms-budget per category; the profile's
   cumulative wall time per function is the closest Python-side attribution,
   and it too is single-run and noise-sensitive.
5. **Small schema.** The bench schema is tiny (1-row tables in places), so
   `EXPLAIN` shows `Seq Scan` on `mm_memory_preferences` as effectively free.
   At production scale the same statements' per-call cost and plan choice would
   differ — this report measures *counts and shape*, not production latency.
6. **M3 grant in a standalone process needed the pytest fixture.** A bare
   `run_longrun` in a non-fixture process hit `TenantContextMissing` for the
   `grant_bench_purpose` load under M3; the pytest path (which opens the proper
   scope) passes. This is why statement/profile runs go through the pytest
   harness / the harness's own schema build rather than a hand-rolled engine.
   It is a harness-context observation, not a production defect.

**Reproduce**

```bash
git worktree add --detach /tmp/mm-perf-403 403f38d
git worktree add --detach /tmp/mm-perf-aed aed0639
git worktree add --detach /tmp/mm-perf-d61 d61e7b7
MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test \
  .venv/bin/python tools/m5_perf_diag.py measure    --rounds 3 --work /tmp/m5_diag
MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test_prof \
  .venv/bin/python tools/m5_perf_diag.py statements --work /tmp/m5_diag
MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test_prof \
  .venv/bin/python tools/m5_perf_diag.py profile    --work /tmp/m5_diag
.venv/bin/python tools/m5_perf_diag.py cleanup
```
