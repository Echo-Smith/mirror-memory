# Invariants → tests map (I1–I6)

The six runtime invariants are defined authoritatively in
[v2-inherited/invariants.md](v2-inherited/invariants.md). This document maps
each one to the v1 code/test surface that enforces and pins it, and records
which database legs run each test.

> **Re-audited against the real test code** (docs/invariants.md §3 pass).
> Every row below was checked by opening the referenced file/class/function
> and the `conftest.py` fixtures, not copied from an earlier table. Exact
> pytest ids, parametrizations and DB legs are as the code stands at
> `feat/m5-readiness @ 00672c7`. Corrections made in this pass are called out
> in **Corrections** at the bottom.

**How to read the DB columns:**

- **SQLite?** — runs on the default in-memory (or `tmp_path`-file) SQLite leg.
- **PG?** — runs on the PostgreSQL leg (a run with `MM_DATABASE_URL` pointing
  at PostgreSQL).
- The shared `db_session` fixture is **dual-leg**: it builds in-memory SQLite
  when `MM_DATABASE_URL` is unset, and rebuilds a per-test schema on
  PostgreSQL (app role under RLS) when it is set. A test that takes
  `db_session` therefore runs on **both** legs. `pg_app_url` / `pg_admin_engine`
  / `pg_maintenance_url` are **PG only** (they `pytest.skip` without
  `MM_DATABASE_URL`). A module-level `pytest.mark.skipif("postgresql" not in
  MM_DATABASE_URL)` makes the whole module **PG only**. `tmp_path` + a hand-
  built `create_engine("sqlite:///…")` is **SQLite only**.

## The table

| Invariant | Meaning (v2) | v1 meaning / enforcement | Test (file::function) | SQLite? | PG? | Notes |
|---|---|---|---|---|---|---|
| **I1** | A job is held by at most one worker at any instant. | Claim is a conditional UPDATE (CAS on `state`+`lease_token`); exactly one claimant wins. | `tests/test_concurrency_ported.py::test_two_threads_race_same_job` | No | **Yes** | PG-only (module-level `skipif` needs `MM_DATABASE_URL`; real threads — SQLite has no real concurrency). 2 threads × 50 rounds. |
| **I1** | (same) | (same) — single-winner claim, second claim returns no token. | `tests/test_lease_tokens.py::test_concurrent_claim_single_winner` | **Yes** | No | Builds its own `sqlite:///{tmp_path}/l2.db`; 4 threads, exactly one non-None token, `lease_token == 1`. Complements the PG race test on the SQLite leg. |
| **I2** | A deleted (revoked/deleted) scope cannot produce a new atom. | A deletion advances the user's deletion generation; a stale publish dies on the lease/generation fence and writes nothing (0 rows). | `tests/test_concurrency_ported.py::test_delete_between_claim_and_publish_creates_nothing` | No | **Yes** | PG-only (module `skipif`). Parametrized `renew_first` with ids `["before-compute", "mid-compute"]` — the `mid-compute` case is the interleaved PI variant. Asserts `belief_count == 0`, job gone, exactly one `DeletionTombstone`. |
| **I2** | (same) — deletion survives a local backup restore (deletion log replays). | Deletions are logged before commit; a restored backup is reconciled by replay before any entry serves. | `tests/test_deletion_replay.py::test_replay_restores_the_deletion_after_a_backup_restore` | **Yes** | **No** | **SQLite only.** Its `env` fixture hard-codes `create_engine(f"sqlite:///{tmp_path}/main.db")` and drives the real **sqlite3** `.backup()` API (`src.connection.backup(...)`); it does **not** take `db_session`/`pg_app_url`, so it runs identically whether or not `MM_DATABASE_URL` is set. The **PG** backup-restore/replay coverage is a *different* test: `tests/test_deletion_race_backup.py::test_backup_restore_replay_end_to_end_pg` (PG-only). Both are indexed in `tests/test_m4_acceptance.py::I2_TARGETS`. |
| **I3** | complete/fail may only be committed by the current lease holder. | Completion is fenced on `lease_token`; the loser's whole transaction (including ghost writes) rolls back. | `tests/test_concurrency_ported.py::test_two_workers_complete_only_holder_wins` | No | **Yes** | PG-only (module `skipif`), 50 rounds. Ghost belief from the stale holder must not survive. Related token fences (all SQLite, own `tmp_path`): `tests/test_lease_tokens.py::test_expired_lease_old_holder_commit_rolls_back`, `::test_failed_renewal_then_commit_lease_lost`, `::test_lease_token_monotonic`, `::test_token_and_tenant_both_enforced`. |
| **I4** | The same `source_event_id` is globally unique within one scope (idempotency). v2 enforces it at **CLAIM** time (a `source_event_id` column + claim-time `UNIQUE`); `Scope` = the trusted `(tenant, app, subject)` triple; `source_event_id` is the caller's event id, written on claim. | **Partial equivalence — NOT equivalent.** v1 has no `source_event_id` column and no claim phase for evidence. Closest analogue: `mm_evidence.ref` (the pipeline's `evidence_message_ids` = the source event ids) + `uq_mm_evidence_user_ref` = UNIQUE(`tenant_id`,`app_id`,`user_id`,`ref`) + `record_evidence` idempotency. Serial re-record → same row (**equivalent**); concurrent re-record → constraint still yields exactly one row but `record_evidence` is a read-then-write, so a racer loses with `IntegrityError` (result preserved, graceful dedup **not** equivalent); scope boundary is per `(tenant,app,user)` (**equivalent** isolation). `mm_proposal_log.proposal_id` is a random uuid per proposal — **not** source-event dedup. `mm_evolution_jobs` `idempotency_key` = `user:watermark:prompt_version` is **job/batch**-level dedup, not per source event. | `tests/test_invariant_mapping.py`: `::test_i4_v1_serial_idempotency_ref_uniqueness` (serial, dual-leg), `::test_i4_concurrent_same_ref_yields_one_row` (real threads, sqlite+pg), `::test_i4_same_ref_different_user_is_distinct` (dual-leg), `::test_i4_same_ref_other_tenant_does_not_collide_or_leak` (sqlite+pg). Literal v2 shape: `tests/test_invariants_xfail.py::test_i4_source_event_id_global_uniqueness` | **Yes** | **Yes** | Serial / different-user via `db_session` (both legs); concurrency + cross-tenant parametrized `["sqlite","postgres"]` (PG leg runs both). All PASS on both legs (verified, incl. real PG threads). **Gap:** v2's claim-time `source_event_id` uniqueness has **no v1 counterpart** → guarded by a **conditional** `xfail(strict=True)` (XFAIL today, XPASS only if the column gains in-scope uniqueness; no unconditional `pytest.fail`). |
| **I5** | The authorized purpose must match the requested purpose. | `authorize_or_raise` refuses on missing purpose, no grant, revoked, expired, or operation-not-allowed; no implicit default purpose. | `tests/test_purpose_authorization.py::test_no_grant_is_refused` (+ `test_revoked_is_refused`, `test_expired_is_refused`, `test_missing_purpose_is_refused_not_fail_open`, `test_operation_not_allowed_is_refused`) | **Yes** | **Yes** | All take `db_session` → both legs (verified by real runs on both). |
| **I5** | (same) — mismatch refused on **every** read/write path. | Every entry (`recall`, `panel`, `get_beliefs`, `export`, …) refuses a never-granted purpose; X stays usable. | `tests/test_m3_acceptance.py::test_i5_wrong_purpose_is_refused_on_every_path` | **Yes** | **Yes** | Parametrized `db_url` over `["sqlite", "postgres"]`. `[sqlite]` passes on the SQLite leg; `[postgres]` skips there (`needs MM_DATABASE_URL`) and runs on the PG leg — so on a PG run **both** params execute. |
| **I6** | An atom's `source_evidence_id` must belong to the same scope. | The Publisher's authority gate refuses evidence that does not exist or belongs to another user (`evidence_authority_mismatch` / `evidence_missing`), writing nothing. | `tests/test_publisher.py::TestPublisherGates::test_cross_user_evidence_refused` | **Yes** | **Yes** | Takes `db_session` → both legs (verified). Asserts `reason == "evidence_authority_mismatch"` and DB dump unchanged. PG race variant: `tests/test_concurrency_ported.py::test_evidence_deleted_between_claim_and_publish` (PG-only). |

## Summary

- **Mapped to a real v1 test that PASSES:** I1, I2, I3, I5, I6 — 5 of 6.
- **I4 — partial equivalence only.** v1 *does* provide a runnable guarantee
  (serial `ref` idempotency; concurrent one-row outcome; per-`(tenant,app,user)`
  scope isolation), pinned by passing tests in `tests/test_invariant_mapping.py`.
  But v2's specific **claim-time `source_event_id` uniqueness has no v1
  counterpart** (no such column, no evidence-claim phase), and the concurrency
  strength differs (v1 relies on a racing caller losing with `IntegrityError`,
  not a graceful shared-row claim). Verdict: **partial equivalence, not
  equivalent.** The literal v2 shape is absent and guarded by a conditional
  `xfail(strict=True)` (`tests/test_invariants_xfail.py::test_i4_source_event_id_global_uniqueness`).
- The I2 surface additionally has a machine-checked index in
  `tests/test_m4_acceptance.py` (`I2_TARGETS`, verified by
  `test_i2_targets_exist_and_are_not_skipped` — existence/collection only; it
  does **not** assert which DB leg a target runs on).

### I4 note — partial equivalence: what v1 covers and what it does not

v2 defined I4 as a **claim-time** uniqueness constraint on `source_event_id`
within one `Scope`. In v2, `Scope` is the trusted `(tenant_id, app_id,
subject_id)` triple (R01; the Claim SQL filters `scope_control` on exactly
those three), `source_event_id` is the caller's event identifier, written when
the event is claimed, and uniqueness is enforced **at claim time** (see
`docs/v2-inherited/invariants.md`, and R03: same event re-sent ⇒ no duplicate
result; same key + different content ⇒ conflict).

v1 has no `source_event_id` column and no evidence-claim phase. Per mechanism:

| v1 mechanism | Relation to v2 I4 |
|---|---|
| `mm_evidence.ref` + `uq_mm_evidence_user_ref` UNIQUE(tenant,app,user,ref) + `record_evidence` **serial** idempotency | **Equivalent** for serial re-ingestion (repeat `ref` → same row). |
| same, **concurrent** re-ingestion | **Partial.** The DB constraint still yields exactly ONE persisted row, but `record_evidence` is a read-then-write, not an atomic upsert: a racing caller loses with `IntegrityError` (swallowed by the pipeline) instead of receiving the shared row. Result preserved; graceful-dedup semantics **not** equivalent. |
| same, **scope** boundary | **Equivalent** isolation: same `ref` under a different `user`/`tenant` is a distinct source, no collision, no cross-tenant leak (ORM filter on SQLite; RLS as second line on PG). |
| `mm_proposal_log.proposal_id` UNIQUE | **Not equivalent.** `proposal_id` is a random uuid4 per proposal object — dedups replays of the *same* proposal, not distinct proposals from the *same* source event. |
| `mm_evolution_jobs` UNIQUE(tenant,app,`idempotency_key`) | **Partial / not equivalent to I4.** key = `user:watermark:prompt_version` is **job/batch**-level dedup, not per source event. |

**The gap (stated explicitly).** v2's claim-time `source_event_id` uniqueness
has **no v1 counterpart** — there is no column, no claim stage, and the
concurrency strength differs (v1's "one row" survives only because the loser's
INSERT faults against the UNIQUE constraint, which the ingestion pipeline then
swallows). This cannot be expressed as a v1 test because the target site does
not exist; the honest characterisation is **partial equivalence**.

The runnable evidence for the parts v1 *does* guarantee, all **passing on both
SQLite and PostgreSQL** (real runs, incl. real PG threads):
`tests/test_invariant_mapping.py::test_i4_v1_serial_idempotency_ref_uniqueness`
(serial), `::test_i4_concurrent_same_ref_yields_one_row` (concurrent, 8 real
threads), `::test_i4_same_ref_different_user_is_distinct` (cross-user),
`::test_i4_same_ref_other_tenant_does_not_collide_or_leak` (cross-tenant).

`test_invariants_xfail.py::test_i4_source_event_id_global_uniqueness` keeps the
**literal v2 gap** visible as a `xfail(strict=True)` whose body does a **real
assertion** (`_source_event_id_uniqueness_enforced()`): it returns False today
(→ XFAIL) and would return True only if `Evidence.source_event_id` appeared
**with an in-scope unique constraint** (→ the body passes → strict mode reports
XPASS, a failure that forces the mapping to be updated). A non-`source_event_id`
column or an unrelated/superset unique constraint is **not** accepted as proof;
there is no unconditional `pytest.fail`.

## Corrections made in this re-audit

1. **I2 deletion-replay `PG?` was wrong ("Yes").** `test_replay_restores_the_deletion_after_a_backup_restore`
   is **SQLite only**: its `env` fixture hard-codes a `sqlite:///` file and
   uses the sqlite3 `.backup()` API. Running it with
   `MM_DATABASE_URL=postgresql://… MM_PG_RLS=1 … pytest <that test>` *passes*,
   but `--setup-plan` shows it uses only `tmp_path`/`env` (no `db_session`,
   no `pg_app_url`), so the PG connection is not exercised by the test DB
   itself. The PG backup-restore surface is
   `tests/test_deletion_race_backup.py::test_backup_restore_replay_end_to_end_pg`.
2. **I6 test id omitted the class.** It is
   `tests/test_publisher.py::TestPublisherGates::test_cross_user_evidence_refused`
   (a method of `TestPublisherGates`), not a module-level function.
3. **I4 was claimed "equivalent" — downgraded to PARTIAL equivalence.** Corrected
   in this §4 pass: v1's `mm_evidence.ref` (`uq_mm_evidence_user_ref`) gives
   **serial** idempotency and scope isolation equivalent to I4, but (a) concurrent
   ingestion relies on a racing caller faulting on the UNIQUE constraint (not a
   graceful claim-time dedup), so the *concurrency* semantics only partially
   match, and (b) `proposal_id` (random uuid) / `idempotency_key`
   (`user:watermark:prompt_version`) are **not** source-event-level dedup. v2's
   claim-time `source_event_id` uniqueness has **no v1 counterpart**. Added real
   concurrent + cross-tenant tests; the remaining xfail stays **conditional**
   (no unconditional `pytest.fail`).
4. **I2 deletion-replay "shared fixture" mislabel.** `env` is a per-module
   fixture, not the shared `db_session`; the table now says so.
