| file | total | skipped |
|---|---:|---:|
| tests/test_access_audit.py | 16 | 0 |
| tests/test_activity.py | 10 | 0 |
| tests/test_api.py | 28 | 0 |
| tests/test_behavioral.py | 16 | 0 |
| tests/test_belief_ledger.py | 15 | 0 |
| tests/test_benchmark_funnel.py | 34 | 0 |
| tests/test_budget.py | 10 | 0 |
| tests/test_canonicalize.py | 12 | 0 |
| tests/test_concurrency_ported.py | 15 | 15 |
| tests/test_confidence.py | 14 | 0 |
| tests/test_config.py | 25 | 0 |
| tests/test_delete_preserves_authorization.py | 4 | 0 |
| tests/test_deletion_privacy.py | 6 | 0 |
| tests/test_display.py | 8 | 0 |
| tests/test_evidence_graph.py | 28 | 0 |
| tests/test_export.py | 9 | 0 |
| tests/test_extraction.py | 45 | 0 |
| tests/test_extraction_storage_loss.py | 16 | 0 |
| tests/test_harness_lazy_grant.py | 6 | 0 |
| tests/test_identity.py | 25 | 0 |
| tests/test_integration.py | 13 | 0 |
| tests/test_lease_tokens.py | 5 | 0 |
| tests/test_longmemeval_ingest.py | 4 | 1 |
| tests/test_longmemeval_regressions.py | 33 | 0 |
| tests/test_longrun_bench.py | 18 | 0 |
| tests/test_m3_acceptance.py | 2 | 0 |
| tests/test_metabolism.py | 61 | 0 |
| tests/test_metabolism_archive.py | 23 | 0 |
| tests/test_metabolism_bench.py | 11 | 0 |
| tests/test_metabolism_compaction.py | 26 | 0 |
| tests/test_metabolism_runtime.py | 29 | 0 |
| tests/test_no_autogrant.py | 4 | 0 |
| tests/test_pg_backfill_composite.py | 4 | 4 |
| tests/test_pg_migration_locks.py | 3 | 3 |
| tests/test_pg_pool.py | 15 | 4 |
| tests/test_pk_addressing.py | 3 | 0 |
| tests/test_polarity_lifecycle.py | 54 | 0 |
| tests/test_predicate_variants.py | 30 | 0 |
| tests/test_publish_protocol.py | 18 | 0 |
| tests/test_publisher.py | 35 | 0 |
| tests/test_publisher_write_guard.py | 12 | 0 |
| tests/test_purpose_authorization.py | 23 | 4 |
| tests/test_purpose_entries.py | 9 | 4 |
| tests/test_purpose_revocation.py | 4 | 4 |
| tests/test_purpose_revocation_sqlite.py | 1 | 0 |
| tests/test_query_aware_retrieval.py | 39 | 0 |
| tests/test_relation_lifecycle.py | 47 | 0 |
| tests/test_renderer.py | 17 | 0 |
| tests/test_repository.py | 50 | 0 |
| tests/test_retrieval.py | 12 | 0 |
| tests/test_rls.py | 49 | 20 |
| tests/test_runtime_invariants.py | 23 | 0 |
| tests/test_schema_migration.py | 21 | 0 |
| tests/test_scope_allowlist.py | 4 | 0 |
| tests/test_server_tenant.py | 7 | 0 |
| tests/test_single_tenant.py | 6 | 0 |
| tests/test_source_purpose.py | 8 | 0 |
| tests/test_sql_auditor.py | 56 | 0 |
| tests/test_statebench_dataset.py | 15 | 0 |
| tests/test_statebench_v1_1.py | 21 | 0 |
| tests/test_temporal.py | 50 | 0 |
| tests/test_tenancy_context.py | 7 | 0 |
| tests/test_tenant_cache_security.py | 12 | 0 |
| tests/test_tenant_isolation.py | 11 | 0 |
| tests/test_tenant_pk.py | 9 | 1 |
| tests/test_utils.py | 23 | 0 |
| tests/test_verification.py | 40 | 0 |
| tests/test_worker.py | 32 | 0 |
| tests/test_worker_pg_transaction.py | 4 | 4 |
| tests/test_worker_sqlite_busy.py | 1 | 0 |
| tests/test_write_narrowing.py | 4 | 0 |
| tests/test_writer_callers.py | 9 | 0 |
| **TOTAL** | **1359** | **64** |

| file | skipped | declared reason |
|---|---:|---|
| tests/test_concurrency_ported.py | 15 | `real-concurrency tests require PostgreSQL` |
| tests/test_longmemeval_ingest.py | 1 | `LongMemEval oracle not available` |
| tests/test_pg_backfill_composite.py | 4 | `needs MM_DATABASE_URL pointing at PostgreSQL` |
| tests/test_pg_migration_locks.py | 3 | `needs MM_DATABASE_URL pointing at PostgreSQL` |
| tests/test_pg_pool.py | 4 | `needs MM_DATABASE_URL pointing at PostgreSQL` |
| tests/test_purpose_authorization.py | 4 | `needs MM_DATABASE_URL pointing at PostgreSQL` |
| tests/test_purpose_entries.py | 4 | `needs MM_DATABASE_URL pointing at PostgreSQL` |
| tests/test_purpose_revocation.py | 4 | `real-concurrency revocation tests require PostgreSQL` |
| tests/test_rls.py | 20 | `needs MM_DATABASE_URL pointing at PostgreSQL` |
| tests/test_tenant_pk.py | 1 | `physical composite PK only exists on PostgreSQL` |
| tests/test_worker_pg_transaction.py | 4 | `idle_in_transaction_session_timeout is a PostgreSQL setting` |
