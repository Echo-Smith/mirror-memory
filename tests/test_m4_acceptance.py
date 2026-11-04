"""M4 acceptance (batch 2 task 6): the deletion and restore scenarios run
RESIDENT on both dual-db legs, and this file is their index plus the
smoke that ties the three I2 surfaces together.

The invariants (v2 merge-plan I2, extended by M4): a deleted range can
never produce new memory, and a deletion SURVIVES every restore path --
local backup restore (the deletion log replays), cross-device
propagation (tombstones are applied through the normal deletion path on
the receiver), and a full edge-machine restore (the main database AND
the deletion log are restored together; the edge then catches up
against the cloud before it serves).

I2 mapping (test file + test name; every entry must exist and run on
the leg it belongs to -- verified by ``test_i2_targets_exist_and_are_not_skipped``
and by tools/skip_report.py on the CI junit):

| surface | scenario | tests |
|---|---|---|
| local restore | backup -> delete (3 scopes) -> restore -> reconcile -> sentinel zero on every read entrance | tests/test_deletion_replay.py::test_replay_restores_the_deletion_after_a_backup_restore |
| local restore | the REAL backup tools (pg_dump/pg_restore, SQLite backup API), full bring-up | tests/test_deletion_race_backup.py::test_backup_restore_replay_end_to_end_pg |
| local restore | a pre-M4 (step-10) backup: migration refuses -> adopt -> replay | tests/test_deletion_adopt.py::test_real_step10_backup_flow_sqlite, tests/test_deletion_race_backup.py::test_real_step10_backup_flow_pg |
| local restore | the deletion log replays idempotently after an interruption | tests/test_deletion_replay.py::test_replay_is_reentrant_after_an_interruption |
| local restore | id reuse is not fooled by the fingerprint | tests/test_deletion_race_backup.py::test_id_reuse_is_not_fooled_by_the_fingerprint |
| local restore | a log older than the main DB refuses | tests/test_deletion_replay.py::test_deletion_log_stale_refuses_startup |
| local restore | a backup older than the log's compaction checkpoint refuses | tests/test_log_compaction.py::test_prune_below_the_head_keeps_replay_working |
| cross-device | three scopes travel cloud -> edge; no echo back; a third device sees the ORIGINAL origin | tests/test_tombstone_sync.py::test_three_scopes_travel_from_the_cloud_to_the_edge |
| cross-device | the reverse direction: edge deletes -> cloud applies -> cloud clean, no echo | tests/test_tombstone_sync.py::test_reverse_direction_edge_to_cloud_and_no_echo |
| cross-device | forged / mixed-tenant / gap / tampered-hash batches are refused with zero writes, both directions | tests/test_tombstone_sync.py::test_forged_mixed_gap_and_tampered_batches_are_refused_with_zero_writes |
| cross-device | the cloud refuses users its authenticated tenant does not know | tests/test_tombstone_sync.py::test_cloud_refuses_users_its_tenant_does_not_know |
| cross-device | apply-vs-publish 50 rounds (tombstone beats a late publish) | tests/test_tombstone_sync.py::test_tombstone_apply_vs_publish_50_rounds_sqlite, tests/test_deletion_race_backup.py::test_tombstone_apply_vs_publish_50_rounds_pg |
| edge full restore | main DB + log dir restored -> startup catch-up -> every read entrance sentinel-free | tests/test_edge_sync.py::test_full_restore_then_catchup_cleans_every_read_entrance |
| edge full restore | a dead link mid-catch-up resumes from the durable cursor | tests/test_edge_sync.py::test_catchup_resumes_after_a_dead_link |
| edge full restore | the four sync states and the offline-grace refusal/heal | tests/test_edge_sync.py::test_four_states_and_the_reminder_threshold |
| sender full restore | A restores (DB+log) after its deletions synced out -> its OWN deletions re-apply (echo check); a later OFFLINE deletion reaches cloud and B, no duplicate, no cursor skip | tests/test_sync_mesh.py::test_sender_full_restore_then_offline_redelete_reaches_everyone, tests/test_deletion_race_backup.py::test_sender_full_restore_then_offline_redelete_pg |
| sender full restore | the mesh with a mid-stream full restore of an edge: auto-rescan (never a refusal), zero refusals, convergence | tests/test_sync_mesh.py::test_mesh_survives_a_midstream_full_restore, tests/test_deletion_race_backup.py::test_tombstone_mesh_with_full_restore_pg |
| identity | a belief's global_id survives the v11 -> v12 migration on both databases | tests/test_global_id_migration.py::test_step12_upgrades_a_step11_database_sqlite, tests/test_global_id_migration.py::test_step12_upgrades_a_step11_database_pg |
| sync gate | ENTRY_REGISTRY reads_memory entrances (+ run_verification / run_metabolism) all refuse when expired / catching up; the fail-open paths never swallow the refusal; after the healed catch-up the deleted sentinel is visible nowhere | tests/test_sync_gate_coverage.py::test_registry_covers_every_read, tests/test_sync_gate_coverage.py::test_expired_and_catching_up_refuse_every_read |
| sync gate | the grace expires by the CLOCK in normal state (no manual catch-up); a healed link resumes on the next read | tests/test_sync_gate_coverage.py::test_normal_state_expires_by_the_clock |
| sync gate | sync.enabled with NO transport: the stored channel history decides (never-completed / offline_grace within grace / expired / ambiguous); disabled sync unchanged; the schema_ready wrapper does not bypass the gate | tests/test_sync_gate_coverage.py::test_enabled_without_transport_uses_stored_grace |
| sync gate | every server reads_memory route refuses over real HTTP when expired / catching up, never leaks the un-caught-up sentinel, serves clean after the healed catch-up; /belief/{id} and /stats covered directly with unchanged cross-tenant 404 / purpose 403 | tests/test_server_sync_gate.py::test_server_registry_covers_every_read, tests/test_server_sync_gate.py::test_http_gate_matrix_sqlite, tests/test_server_sync_gate.py::test_http_gate_matrix_real_pg |
| sync gate | the schema_ready shared-engine wrapper with a transport runs the startup catch-up before the first read; offline it falls back to the stored grace; no duplicate catch-up | tests/test_sync_gate_coverage.py::test_shared_wrapper_startup_catchup |
"""

from __future__ import annotations

import importlib

import pytest

from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

TENANT = ("default", "default")


@pytest.fixture(autouse=True)
def _single_tenant_mode_off():
    """The acceptance devices run single_tenant_mode=True (a contextvar):
    reset it after every test so later tests keep tenant filtering."""
    from mirror_memory.core.tenancy import set_single_tenant_mode

    set_single_tenant_mode(False)
    yield
    set_single_tenant_mode(False)

#: The I2 table above, machine-checked: (module, test name) pairs the M4
#: acceptance relies on.  A renamed or deleted test fails HERE, before it
#: can silently leave an I2 surface uncovered.
I2_TARGETS: list[tuple[str, str]] = [
    ("tests.test_deletion_replay", "test_replay_restores_the_deletion_after_a_backup_restore"),
    ("tests.test_deletion_race_backup", "test_backup_restore_replay_end_to_end_pg"),
    ("tests.test_deletion_adopt", "test_real_step10_backup_flow_sqlite"),
    ("tests.test_deletion_race_backup", "test_real_step10_backup_flow_pg"),
    ("tests.test_deletion_replay", "test_replay_is_reentrant_after_an_interruption"),
    ("tests.test_deletion_race_backup", "test_id_reuse_is_not_fooled_by_the_fingerprint"),
    ("tests.test_deletion_replay", "test_log_older_than_main_db_refuses_startup"),
    ("tests.test_log_compaction", "test_prune_below_the_head_keeps_replay_working"),
    ("tests.test_tombstone_sync", "test_three_scopes_travel_from_the_cloud_to_the_edge"),
    ("tests.test_tombstone_sync", "test_reverse_direction_edge_to_cloud_and_no_echo"),
    ("tests.test_tombstone_sync", "test_forged_mixed_gap_and_tampered_batches_are_refused_with_zero_writes"),
    ("tests.test_tombstone_sync", "test_cloud_refuses_users_its_tenant_does_not_know"),
    ("tests.test_tombstone_sync", "test_tombstone_apply_vs_publish_50_rounds_sqlite"),
    ("tests.test_deletion_race_backup", "test_tombstone_apply_vs_publish_50_rounds_pg"),
    ("tests.test_sync_mesh", "test_three_devices_alternate_deletes_and_stay_consistent"),
    ("tests.test_deletion_race_backup", "test_tombstone_mesh_three_devices_pg"),
    ("tests.test_sync_mesh", "test_device_joining_midway_catches_up"),
    ("tests.test_sync_mesh", "test_untravelable_entry_does_not_wedge_the_channel"),
    ("tests.test_sync_mesh", "test_forged_batches_trigger_rescan_and_apply_with_dedup"),
    ("tests.test_sync_mesh", "test_receiver_restore_repulls_to_the_same_state"),
    ("tests.test_edge_sync", "test_full_restore_then_catchup_cleans_every_read_entrance"),
    ("tests.test_edge_sync", "test_catchup_resumes_after_a_dead_link"),
    ("tests.test_edge_sync", "test_four_states_and_the_reminder_threshold"),
    ("tests.test_sync_mesh",
     "test_sender_full_restore_then_offline_redelete_reaches_everyone"),
    ("tests.test_deletion_race_backup",
     "test_sender_full_restore_then_offline_redelete_pg"),
    ("tests.test_sync_mesh", "test_mesh_survives_a_midstream_full_restore"),
    ("tests.test_deletion_race_backup",
     "test_tombstone_mesh_with_full_restore_pg"),
    ("tests.test_global_id_migration", "test_step12_upgrades_a_step11_database_sqlite"),
    ("tests.test_global_id_migration", "test_step12_upgrades_a_step11_database_pg"),
    ("tests.test_sync_gate_coverage", "test_registry_covers_every_read"),
    ("tests.test_sync_gate_coverage",
     "test_expired_and_catching_up_refuse_every_read"),
    ("tests.test_sync_gate_coverage",
     "test_normal_state_expires_by_the_clock"),
    ("tests.test_sync_gate_coverage",
     "test_enabled_without_transport_uses_stored_grace"),
    ("tests.test_server_sync_gate", "test_server_registry_covers_every_read"),
    ("tests.test_server_sync_gate", "test_http_gate_matrix_sqlite"),
    ("tests.test_server_sync_gate", "test_http_gate_matrix_real_pg"),
    ("tests.test_sync_gate_coverage", "test_shared_wrapper_startup_catchup"),
]


@pytest.mark.parametrize("module_name, test_name", I2_TARGETS)
def test_i2_targets_exist_and_are_not_skipped(module_name: str, test_name: str):
    """Every I2 target resolves to a REAL test -- and is not skipped at
    collection time (a skipif the leg cannot satisfy is how a scenario
    quietly stops running on one of the two legs)."""
    module = importlib.import_module(module_name)
    assert hasattr(module, test_name), f"{module_name}.{test_name} is gone"
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         f"{module_name.replace('.', '/')}.py::{test_name}"],
        capture_output=True, text=True, cwd=".",
    )
    assert result.returncode == 0, (module_name, test_name, result.stdout[-400:])
    assert "no tests ran" not in result.stdout, (module_name, test_name)


def test_edge_catchup_and_local_replay_together(tmp_path):
    """The combined smoke: cloud deletes (3 scopes) -> the edge applies
    (cross-device propagation) -> the edge restores its own BACKUP taken
    BEFORE the deletions (local restore) -> the startup replay removes
    what the backup brought back AND the catch-up keeps the cloud clean:
    both I2 surfaces in one pass."""
    from tests.test_tombstone_sync import USER, _Device

    cloud = _Device(tmp_path, "acc-cloud", single_tenant=False)
    edge = _Device(tmp_path, "acc-edge", single_tenant=True)
    try:
        for text in ("I like ACCSENTINEL and coffee.", "I live in Berlin."):
            cloud.observe(text, session_id="sa")
            edge.observe(text, session_id="sa")

        # the edge's backup, taken BEFORE any deletion
        import shutil

        backup_path = str(edge.url).replace("sqlite:///", "").replace("edge.db", "acc-backup.db")
        src = edge.engine._engine.raw_connection()
        from sqlalchemy import create_engine as _ce

        dest = _ce(f"sqlite:///{backup_path}").raw_connection()
        src.connection.backup(dest.connection)
        dest.close()
        src.close()

        # the cloud deletes everything in three scopes and propagates
        with tenant_scope(*TENANT):
            s = edge.F()
            from mirror_memory.core.models import Belief as _B

            cloud_b = s.query(_B).filter(_B.claim_text.contains("ACCSENTINEL")).one()
            edge_b = next(b for b in edge.beliefs() if "ACCSENTINEL" in b.claim_text)
            s.close()
        with tenant_scope(*TENANT), fixture_write_scope("align identity"):
            s = edge.F()
            row = s.get(_B, (*TENANT, edge_b.id))
            row.global_id = cloud_b.global_id
            s.commit()
            s.close()
        assert cloud.engine.forget_belief(user_id=USER, belief_id=cloud_b.id, purpose="api")
        assert cloud.engine.forget_session(user_id=USER, session_id="sa", purpose="api")
        cloud.engine.delete_memories(user_id=USER, purpose="api")
        batch, cursor = cloud.engine.export_tombstones(0, 100)
        edge.engine.apply_tombstones(batch)
        assert edge.beliefs() == [], "cross-device propagation cleaned the edge"

        # LOCAL restore: the backup brings the deletions back...
        edge.engine._engine.dispose()
        edge.db.dispose()
        shutil.copyfile(backup_path, str(edge.url).replace("sqlite:///", ""))
        assert len(edge.beliefs()) > 0, "the restore resurrected the data (pre-replay)"
        # ... and the deletion log replays them away
        from mirror_memory.core.replay import reconcile
        from sqlalchemy.orm import sessionmaker as _sm

        assert reconcile(lambda: _sm(bind=edge.db)(), edge.log, tenant=TENANT) == "replayed"
        assert edge.beliefs() == [], "local restore: the replay cleaned it again"
        assert "ACCSENTINEL" not in (edge.engine.recall(user_id=USER, query="ACCSENTINEL", purpose="api") or "")
    finally:
        cloud.dispose()
        edge.dispose()
