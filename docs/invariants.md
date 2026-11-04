# Invariants → tests map (I1–I6)

The six runtime invariants are defined authoritatively in
[v2-inherited/invariants.md](v2-inherited/invariants.md). This document maps
each one to the v1 code/test surface that enforces and pins it, and records
which database legs run each test.

**How to read the DB columns:**

- **SQLite?** — runs on the default in-memory SQLite leg.
- **PG?** — runs on the PostgreSQL leg (a run with `MM_DATABASE_URL` pointing
  at PostgreSQL). Tests that use the shared `db_session` / `tmp_path` /
  parametrized fixtures run on **both** legs; `@pytest.mark.skipif(...)` on
  `MM_DATABASE_URL` (or PG-only `pg_app_url` fixtures) makes a test **PG
  only**; `pytest.mark.sqlite_only` is SQLite only.

## The table

| Invariant | Meaning (v2) | v1 meaning / enforcement | Test (file::function) | SQLite? | PG? | Notes |
|---|---|---|---|---|---|---|
| **I1** | A job is held by at most one worker at any instant. | Claim is a conditional UPDATE (CAS on `state`+`lease_token`); exactly one claimant wins. | `tests/test_concurrency_ported.py::test_two_threads_race_same_job` | No | **Yes** | PG-only (real threads; SQLite has no real concurrency). Ported from v2, 50 rounds. |
| **I1** | (same) | (same) — single-winner claim, second claim returns no token. | `tests/test_lease_tokens.py::test_concurrent_claim_single_winner` | **Yes** | No | Runs on its own `tmp_path` SQLite file; complements the PG race test. |
| **I2** | A deleted (revoked/deleted) scope cannot produce a new atom. | A deletion advances the user's deletion generation; a stale publish dies on the lease/generation fence and writes nothing (0 rows). | `tests/test_concurrency_ported.py::test_delete_between_claim_and_publish_creates_nothing` | No | **Yes** | PG-only. The job-scope variant (`renew_first` param covers the interleaved PI case). |
| **I2** | (same) — deletion survives a local backup restore (deletion log replays). | Deletions are logged before commit; a restored backup is reconciled by replay before any entry serves. | `tests/test_deletion_replay.py::test_replay_restores_the_deletion_after_a_backup_restore` | **Yes** | **Yes** | Shared fixture; M4 acceptance indexes every I2 surface (`tests/test_m4_acceptance.py::I2_TARGETS`). Also: `test_deletion_race_backup.py`, `test_tombstone_sync.py`, `test_sync_mesh.py`, `test_edge_sync.py` cover the restore + cross-device surfaces. |
| **I3** | complete/fail may only be committed by the current lease holder. | Completion is fenced on `lease_token`; the loser's whole transaction (including ghost writes) rolls back. | `tests/test_concurrency_ported.py::test_two_workers_complete_only_holder_wins` | No | **Yes** | PG-only, 50 rounds. Related token fences: `tests/test_lease_tokens.py::test_expired_lease_old_holder_commit_rolls_back`, `::test_failed_renewal_then_commit_lease_lost`, `::test_lease_token_monotonic` (SQLite, `tmp_path`). |
| **I4** | The same `source_event_id` is globally unique within one scope (idempotency). | **No v1 equivalent.** v1 has no `source_event_id` column or claim-time idempotency constraint; ingestion idempotency is achieved at a different layer (publish `proposal_id` + `idempotency_key`, content-addressed evidence `ref`). | `tests/test_invariants_xfail.py::test_i4_source_event_id_global_uniqueness` | — | — | **xfail(strict=True) stub** — v1 does not implement I4 as specified; kept visible so it cannot be silently forgotten (a future implementation must make it XPASS and flip it). |
| **I5** | The authorized purpose must match the requested purpose. | `authorize_or_raise` refuses on missing purpose, no grant, revoked, expired, or operation-not-allowed; there is no implicit default purpose. | `tests/test_purpose_authorization.py::test_no_grant_is_refused` (+ `test_revoked_is_refused`, `test_expired_is_refused`, `test_missing_purpose_is_refused_not_fail_open`, `test_operation_not_allowed_is_refused`) | **Yes** | **Yes** | Uses the shared `db_session` fixture → both legs. |
| **I5** | (same) — mismatch refused on **every** read/write path. | Every entry (`observe`, `recall`, `panel`, `get_beliefs`, `export`, `explain`, `belief_history`, `correct`, …) refuses a never-granted purpose; X stays usable. | `tests/test_m3_acceptance.py::test_i5_wrong_purpose_is_refused_on_every_path` | **Yes** | **Yes** | Parametrized `db_url` over `["sqlite", "postgres"]`. |
| **I6** | An atom's `source_evidence_id` must belong to the same scope. | The Publisher's authority gate refuses evidence that does not exist or belongs to another user (`evidence_authority_mismatch` / `evidence_missing`), writing nothing. | `tests/test_publisher.py::test_cross_user_evidence_refused` | **Yes** | **Yes** | Shared `db_session` → both legs. `tests/test_concurrency_ported.py::test_evidence_deleted_between_claim_and_publish` is the PG race variant. |

## Summary

- **Mapped to a real test:** I1, I2, I3, I5, I6 — 5 of 6.
- **xfail(strict=True) stub (no v1 test):** **I4** — 1 of 6
  (`tests/test_invariants_xfail.py`, marked in the table above).
- Every mapped test either uses a both-legs fixture (`db_session`,
  `db_url`, `tmp_path`) or is explicitly PG-only where real concurrency is
  required. The I2 surface additionally has a machine-checked index in
  `tests/test_m4_acceptance.py` (`I2_TARGETS`, verified by
  `test_i2_targets_exist_and_are_not_skipped`).

### I4 note (why an xfail stub, not a skip)

v2 defined I4 as a claim-time uniqueness constraint on `source_event_id`.
v1 rewrote ingestion around content-addressed evidence `ref` uniqueness plus
publish-time `proposal_id`/`idempotency_key` idempotency; the v2 column and
constraint were **not** carried over. The stub therefore asserts the v2
property against v1 and is expected to fail (`xfail(strict=True)`). If I4 is
ever ported to v1, the test must start passing — strict xfail turns that into
a **failure**, forcing the mapping (and product copy) to be updated rather
than silently drifting.
