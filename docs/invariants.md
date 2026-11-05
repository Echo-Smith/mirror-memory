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
| **I4** | The same `source_event_id` is globally unique within one scope (idempotency). | **Equivalent guarantee, different site.** No literal `source_event_id` column, but v1 enforces source-identifier uniqueness at the **evidence** layer: `mm_evidence` has `uq_mm_evidence_user_ref` = UNIQUE(`tenant_id`,`app_id`,`user_id`,`ref`) and `record_evidence` is idempotent on `(user_id, ref)`; publish-time idempotency via `mm_proposal_log.proposal_id` (UNIQUE) and `mm_evolution_jobs` UNIQUE(`tenant_id`,`app_id`,`idempotency_key`). | Behaviour: `tests/test_invariant_mapping.py::test_i4_v1_equivalent_source_ref_uniqueness` (dual-leg, PASSES). Literal v2 shape: `tests/test_invariants_xfail.py::test_i4_source_event_id_global_uniqueness` | **Yes** | **Yes** | Behavioural equivalence runs on **both** legs via `db_session` (verified). The literal `source_event_id` column is absent → its guard is a **conditional** `xfail(strict=True)` (XFAIL today, XPASS only if the column later gains in-scope uniqueness; no unconditional `pytest.fail`). |
| **I5** | The authorized purpose must match the requested purpose. | `authorize_or_raise` refuses on missing purpose, no grant, revoked, expired, or operation-not-allowed; no implicit default purpose. | `tests/test_purpose_authorization.py::test_no_grant_is_refused` (+ `test_revoked_is_refused`, `test_expired_is_refused`, `test_missing_purpose_is_refused_not_fail_open`, `test_operation_not_allowed_is_refused`) | **Yes** | **Yes** | All take `db_session` → both legs (verified by real runs on both). |
| **I5** | (same) — mismatch refused on **every** read/write path. | Every entry (`recall`, `panel`, `get_beliefs`, `export`, …) refuses a never-granted purpose; X stays usable. | `tests/test_m3_acceptance.py::test_i5_wrong_purpose_is_refused_on_every_path` | **Yes** | **Yes** | Parametrized `db_url` over `["sqlite", "postgres"]`. `[sqlite]` passes on the SQLite leg; `[postgres]` skips there (`needs MM_DATABASE_URL`) and runs on the PG leg — so on a PG run **both** params execute. |
| **I6** | An atom's `source_evidence_id` must belong to the same scope. | The Publisher's authority gate refuses evidence that does not exist or belongs to another user (`evidence_authority_mismatch` / `evidence_missing`), writing nothing. | `tests/test_publisher.py::TestPublisherGates::test_cross_user_evidence_refused` | **Yes** | **Yes** | Takes `db_session` → both legs (verified). Asserts `reason == "evidence_authority_mismatch"` and DB dump unchanged. PG race variant: `tests/test_concurrency_ported.py::test_evidence_deleted_between_claim_and_publish` (PG-only). |

## Summary

- **Mapped to a real v1 test that PASSES:** I1, I2, I3, I5, I6 — 5 of 6.
- **I4:** the v1 **behavioural equivalent** passes on both legs
  (`tests/test_invariant_mapping.py::test_i4_v1_equivalent_source_ref_uniqueness`);
  the **literal v2 shape** (`source_event_id` column) is absent and guarded by
  a conditional `xfail(strict=True)`
  (`tests/test_invariants_xfail.py::test_i4_source_event_id_global_uniqueness`).
- The I2 surface additionally has a machine-checked index in
  `tests/test_m4_acceptance.py` (`I2_TARGETS`, verified by
  `test_i2_targets_exist_and_are_not_skipped` — existence/collection only; it
  does **not** assert which DB leg a target runs on).

### I4 note — equivalent guarantee, and why a *conditional* xfail remains

v2 defined I4 as a claim-time uniqueness constraint on `source_event_id`.
v1 has no such column. But v1 is **not** without the guarantee: it moved it
to the ingestion/evidence layer. `mm_evidence.ref` is the caller's source
identifier and is protected by a real UNIQUE constraint
(`uq_mm_evidence_user_ref`) keyed on the scope (`tenant_id`, `app_id`,
`user_id`); `record_evidence` returns the existing row on a repeat `ref`
instead of duplicating it. Publish-time idempotency is separately secured by
`proposal_id` (UNIQUE) and the job `idempotency_key` UNIQUE. That is the same
"source identifier is globally unique within one scope" property I4 names —
just expressed on `ref` rather than `source_event_id`. A real, dual-leg test
(`test_i4_v1_equivalent_source_ref_uniqueness`) pins it and **passes on both
SQLite and PostgreSQL**.

The earlier "I4 satisfied" conclusion is therefore correct at the
**behavioural** level; the earlier doc's "**No v1 equivalent**" was too
strong and is corrected here. What genuinely remains missing is only the
**literal v2 column/site**. `test_invariants_xfail.py` keeps that one gap
visible as a `xfail(strict=True)` whose body does a **real assertion**
(`_source_event_id_uniqueness_enforced()`): it returns False today (→ XFAIL)
and would return True only if `Evidence.source_event_id` appeared **with an
in-scope unique constraint** (→ the body passes → strict mode reports XPASS,
a failure that forces the mapping to be updated). There is no unconditional
`pytest.fail`.

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
3. **I4 "No v1 equivalent" overstated.** v1 has an equivalent guarantee on
   `mm_evidence.ref` (`uq_mm_evidence_user_ref`) plus `proposal_id` /
   `idempotency_key`; only the literal `source_event_id` column is missing.
   Added a real dual-leg passing test and made the remaining xfail
   **conditional** (removed the prior unconditional `pytest.fail` tail).
4. **I2 deletion-replay "shared fixture" mislabel.** `env` is a per-module
   fixture, not the shared `db_session`; the table now says so.
