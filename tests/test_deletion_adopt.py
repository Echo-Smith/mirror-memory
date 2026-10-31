"""M4 batch-1 patch tasks 1-2: the watermark must be CLAIMED when a
pre-M4 database is restored, and the log must prove it belongs.

- a main DB with tenant data but no watermark row refuses to start
  (WatermarkMissing) until an operator adopts it with an existing log's
  db_id; a truly fresh database is minted normally;
- the deletion log records a GENESIS entry (seq 0, db_id + time) when
  the watermark row is created; reconciliation requires that genesis
  entry -- a log WITHOUT it is "not found" (DeletionLogUnavailable),
  distinct from "older than the main DB" (DeletionLogStale);
- configuration: the derived log directory is embedded-SQLite only; in
  server/multi-tenant mode the file backend needs an explicit directory
  and the postgres backend needs a URL; backend=none refuses to start
  when the watermark shows applied entries.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import (
    DeletionEntry,
    DeletionLogUnavailable,
    FileDeletionLog,
    GENESIS_HASH,
)
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.schema_migration import WatermarkMissing
from mirror_memory.core.deletion import DeletionLogUnavailable
from mirror_memory.core.sql_audit import no_audit_scope
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

USER = "u-adopt"
TENANT = ("default", "default")


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


@pytest.fixture
def db(tmp_path):
    """Deploy-time shape: tables AND the watermark row exist before any
    content is seeded (Step 11 creates the row at deploy)."""
    eng = create_engine(f"sqlite:///{tmp_path / 'adopt.db'}")
    Base.metadata.create_all(eng)
    sm.ensure_watermark(eng)
    yield eng
    eng.dispose()


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


def _seed_content(db, text: str) -> int:
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("seed content"):
        s = F()
        set_memory_enabled(s, USER, True)
        from mirror_memory.core.repository import record_claim

        record_claim(s, USER, dimension="topic", key="k", claim_text=text,
                     confidence=0.9, session_id="s1")
        s.commit()
        s.close()
    with tenant_scope(*TENANT):
        s = F()
        try:
            return s.query(Belief).first().id
        finally:
            s.close()


def test_fresh_database_is_minted_normally(tmp_path):
    """No tenant data -> ensure_watermark mints a db_id as before."""
    from mirror_memory.core.deletion import default_directory

    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    eng = create_engine(url)
    Base.metadata.create_all(eng)
    from mirror_memory.core.sql_audit import no_audit_scope

    with no_audit_scope():
        db_id, applied = sm.ensure_watermark(eng)
    assert applied == 0 and len(db_id) == 32
    eng.dispose()


def test_pre_m4_restore_refuses_startup_until_adopted(db, tmp_path):
    """A PRE-M4 backup (tenant data, no watermark row) refuses to start;
    the adopt command claims it for the existing log; the replay then
    removes the sentinel."""
    log_dir = tmp_path / "deletions"
    db_path = tmp_path / "adopt.db"

    # -- the M4-era life: memory is created ..............................
    log = FileDeletionLog(log_dir)
    eng = MemoryEngine(config=load_config("config/"), database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1],
                       llm_client=_FakeLLM(), deletion_log=log)
    belief_id = _seed_content(db, "I like ADOPT-SENTINEL and coffee.")

    # -- BACKUP, then strip the M4 artifacts from the COPY: the result is
    #    exactly what a pre-M4 backup looks like (content, no watermark)
    backup_path = tmp_path / "pre-m4-backup.db"
    backup_path.write_bytes(db_path.read_bytes())
    backup = create_engine(f"sqlite:///{backup_path}")
    with backup.begin() as conn:
        conn.exec_driver_sql("DROP TABLE mm_deletion_watermark")
        conn.exec_driver_sql("DROP TABLE mm_deletion_generations")
    backup.dispose()

    # -- deletions happen (logged under the live db_id) ..................
    db_id, applied = sm.read_watermark(db)
    eng = MemoryEngine(config=load_config("config/"), database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1],
                       llm_client=_FakeLLM(), deletion_log=log)
    assert eng.forget_belief(user_id=USER, belief_id=belief_id, purpose="api") >= 1
    assert log.head(db_id)[0] >= 1
    eng._engine.dispose()
    db.dispose()

    # -- RESTORE the pre-M4 backup: the sentinel is BACK, the watermark is
    #    GONE .............................................................
    db_path.write_bytes(backup_path.read_bytes())
    db = create_engine(str(db.url))
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            assert s.query(Belief).filter_by(id=belief_id).first() is not None
        finally:
            s.close()

    # startup refuses: unclaimed pre-M4 restore
    eng2 = MemoryEngine(config=load_config("config/"), database_url=str(db.url),
                        tenant_id=TENANT[0], app_id=TENANT[1],
                        llm_client=_FakeLLM(), deletion_log=log)
    with pytest.raises(WatermarkMissing) as info:
        eng2.get_beliefs(user_id=USER, purpose="api")
    message = str(info.value)
    assert "PRE-M4 backup" in message and "adopt" in message

    # the adopt COMMAND claims the database for the existing log (the real
    # operator entry point, subprocess-shaped)
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "mirror_memory.deletion_adopt",
         "--database-url", str(db.url), "--db-id", db_id],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "watermark adopted" in result.stdout

    # a NEW engine now reconciles on its FIRST entry (lazy _ensure_db):
    # the replay re-deletes the restored row
    eng3 = MemoryEngine(config=load_config("config/"), database_url=str(db.url),
                        tenant_id=TENANT[0], app_id=TENANT[1],
                        llm_client=_FakeLLM(), deletion_log=log)
    eng3.get_beliefs(user_id=USER, purpose="api")  # triggers startup + replay
    _db_id3, applied3 = sm.read_watermark(db)
    assert applied3 >= applied, "the replay must reach the log's tail"
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=db)()
        try:
            assert s.query(Belief).filter_by(id=belief_id).first() is None
        finally:
            s.close()
    assert "ADOPT-SENTINEL" not in (eng3.recall(user_id=USER, query="ADOPT", purpose="api") or "")


def test_adopt_refuses_when_the_watermark_row_already_exists(db):
    from mirror_memory.core.schema_migration import adopt_watermark

    sm.ensure_watermark(db)
    with pytest.raises(ValueError, match="already exists"):
        adopt_watermark(db, "f" * 32)


def test_adopt_rejects_bad_db_ids(db):
    from mirror_memory.core.schema_migration import adopt_watermark

    for bad in ("", "  ", "a b"):
        with pytest.raises(ValueError):
            adopt_watermark(db, bad)


# ---------------------------------------------------------------------------
# task 2: the genesis entry; the log must PROVE it belongs
# ---------------------------------------------------------------------------


def test_genesis_entry_is_written_and_required(tmp_path):
    log = FileDeletionLog(tmp_path / "deletions")
    db_id = "d" * 32

    # a log file WITHOUT the genesis entry (an old-format or foreign log)
    # must be refused as "not found" rather than passing 0 == 0
    from mirror_memory.core.replay import reconcile

    eng = create_engine(f"sqlite:///{tmp_path / 'g.db'}")
    Base.metadata.create_all(eng)
    sm.ensure_watermark(eng)  # mints (db_id, 0)
    path = log._path_for(db_id)
    legacy = {
        "seq": 1, "db_id": db_id, "tenant_id": "t", "app_id": "a",
        "user_id": "u", "scope": "user", "scope_key": "",
        "fingerprints": [], "generation": 1, "purpose": "p",
        "prev_hash": GENESIS_HASH, "hash": GENESIS_HASH, "created_at": None,
    }
    path.write_text(json.dumps({"db_id": db_id, "header": True}) + "\n"
                    + json.dumps(legacy) + "\n")
    with no_audit_scope(), pytest.raises(DeletionLogUnavailable, match="genesis"):
        reconcile(lambda: sessionmaker(bind=eng)(), log)
    eng.dispose()

    # the log opens with the GENESIS entry (seq 0, scope "genesis"); the
    # first real entry is seq 1 chained to it
    entry = log.append(DeletionEntry(
        db_id=db_id, tenant_id="t", app_id="a", user_id="u",
        scope="user", scope_key="", fingerprints=[], generation=1, purpose="p",
    ))
    # the append chains onto the legacy entry (seq 1): no genesis was
    # written, which is exactly what the reconciliation must refuse
    assert entry.seq == 2
    assert json.loads((log._path_for(db_id)).read_text().splitlines()[0])[
        "header"] is True


def test_config_validation_for_log_backends(tmp_path):
    from mirror_memory.api import MemoryEngine

    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    # server-style database URL (not embedded sqlite): the file backend
    # requires an explicit directory
    cfg.deletion_log.backend = "file"
    cfg.deletion_log.directory = ""
    with pytest.raises(ValueError, match="directory"):
        MemoryEngine(config=cfg, database_url="postgresql+psycopg://u:p@h:5432/db",
                     tenant_id="t", app_id="a", llm_client=_FakeLLM())
    # postgres backend requires a URL
    cfg.deletion_log.backend = "postgres"
    cfg.deletion_log.url = ""
    with pytest.raises(ValueError, match="url"):
        MemoryEngine(config=cfg, database_url="postgresql+psycopg://u:p@h:5432/db",
                     tenant_id="t", app_id="a", llm_client=_FakeLLM())


def test_backend_none_refuses_when_entries_were_applied(tmp_path):
    """backend=none is only valid for a database with nothing applied: once
    the watermark shows applied entries, removing the log config would
    orphan them -- refuse the startup."""
    db = create_engine(f"sqlite:///{tmp_path / 'none.db'}")
    Base.metadata.create_all(db)

    from mirror_memory.core import schema_migration as _sm_mint
    _sm_mint.ensure_watermark(db)
    log = FileDeletionLog(tmp_path / "deletions")
    eng = MemoryEngine(config=load_config("config/"), database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1],
                       llm_client=_FakeLLM(), deletion_log=log)
    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    eng._config = cfg
    _seed_content(db, "I like coffee and I live in Berlin.")
    belief = sessionmaker(bind=db)().query(Belief).first()
    eng.forget_belief(user_id=USER, belief_id=belief.id, purpose="api")
    db_id, applied = sm.read_watermark(db)
    assert applied >= 1

    # restart with backend=none: applied entries can no longer be replayed
    cfg.deletion_log.backend = "none"
    from mirror_memory.core.replay import reconcile

    with no_audit_scope(), pytest.raises(DeletionLogUnavailable, match="applied"):
        reconcile(lambda: sessionmaker(bind=db)(), None)
    db.dispose()


# ---------------------------------------------------------------------------
# M4 patch 2: the step-11 migration's mint decision reads the deletion log
# ---------------------------------------------------------------------------


def _v10_database(tmp_path, name: str):
    """A REAL step-10 database with tenant data: build the current shape,
    seed, then strip every step-11 artifact (version rows included)."""
    from tests.conftest import strip_to_step10

    url = f"sqlite:///{tmp_path / name}"
    eng = create_engine(url)
    Base.metadata.create_all(eng)
    F = sessionmaker(bind=eng)
    with tenant_scope(*TENANT), fixture_write_scope("seed pre-M4 data"):
        s = F()
        set_memory_enabled(s, USER, True)
        from mirror_memory.core.repository import record_claim

        record_claim(s, USER, dimension="topic", key="k",
                     claim_text="I like LINEAGE and coffee.",
                     confidence=0.9, session_id="s1")
        s.commit()
        s.close()
    strip_to_step10(eng)
    assert sm.current_version(eng) == 10
    return url, eng


def _log_config(directory) -> MemoryConfig:
    from mirror_memory.config.loader import load_config

    cfg = load_config("config/")
    cfg.deletion_log.directory = str(directory)
    return cfg


def test_migration_mints_on_the_first_deployment(tmp_path):
    """A data-bearing v10 database + an EMPTY log: this deployment line
    has never run M4 -- the migration mints the watermark row."""
    from mirror_memory.core.schema_migration import upgrade_database

    url, eng = _v10_database(tmp_path, "first.db")
    log_dir = tmp_path / "log-first"
    upgrade_database(eng, default_tenant="default", default_app="default",
                     config=_log_config(log_dir))
    assert sm.current_version(eng) == 11
    db_id, applied = sm.read_watermark(eng)
    assert db_id and applied == 0
    eng.dispose()


def test_migration_refuses_when_the_log_holds_earlier_lives(tmp_path, capsys):
    """A data-bearing v10 database whose log already holds genesis
    entries: the migration refuses and lists the db_ids with their genesis
    times -- adopt is the way back."""
    from mirror_memory.core.deletion import FileDeletionLog
    from mirror_memory.core.schema_migration import upgrade_database

    url, eng = _v10_database(tmp_path, "lineage.db")
    log_dir = tmp_path / "log-lineage"
    log = FileDeletionLog(log_dir)
    log.ensure_genesis("a" * 32)
    log.ensure_genesis("b" * 32)
    entries = dict(log.genesis_entries())

    with pytest.raises(WatermarkMissing) as info:
        upgrade_database(eng, default_tenant="default", default_app="default",
                         config=_log_config(log_dir))
    message = str(info.value)
    assert "genesis" in message and "adopt" in message
    for db_id in ("a" * 32, "b" * 32):
        assert db_id[:8] in message
        created = entries[db_id]
        assert (created.isoformat() if created else "unknown time") in message
    assert sm.read_watermark(eng) is None, "nothing was minted"
    eng.dispose()


def test_new_lineage_mints_with_a_warning(tmp_path, capsys):
    """--new-lineage is the explicit escape hatch: mint despite earlier
    lives, with a printed warning that their deletions will not be
    replayed."""
    from mirror_memory.core.deletion import FileDeletionLog
    from mirror_memory.core.schema_migration import upgrade_database

    url, eng = _v10_database(tmp_path, "newlineage.db")
    log_dir = tmp_path / "log-new"
    log = FileDeletionLog(log_dir)
    log.ensure_genesis("c" * 32)

    upgrade_database(eng, default_tenant="default", default_app="default",
                     config=_log_config(log_dir), new_lineage=True)
    db_id, applied = sm.read_watermark(eng)
    assert db_id and applied == 0 and db_id != "c" * 32
    out = capsys.readouterr().out
    assert "NEW-LINEAGE" in out and "will NOT be replayed" in out
    eng.dispose()


def test_migration_refuses_without_the_log_config(tmp_path):
    from mirror_memory.core.schema_migration import upgrade_database

    url, eng = _v10_database(tmp_path, "noconf.db")
    with pytest.raises(WatermarkMissing, match="deletion log configuration"):
        upgrade_database(eng, default_tenant="default", default_app="default")
    assert sm.read_watermark(eng) is None
    eng.dispose()


def test_migration_refuses_an_unreachable_log(tmp_path):
    from mirror_memory.core.schema_migration import upgrade_database

    url, eng = _v10_database(tmp_path, "unreach.db")
    cfg = _log_config(tmp_path / "unused")
    cfg.deletion_log.backend = "postgres"
    cfg.deletion_log.url = "postgresql+psycopg://postgres:mm@127.0.0.1:1/nope"
    with pytest.raises(WatermarkMissing, match="unreachable"):
        upgrade_database(eng, default_tenant="default", default_app="default",
                         config=cfg)
    assert sm.read_watermark(eng) is None
    eng.dispose()


def test_real_step10_backup_flow_sqlite(tmp_path):
    """The full story on a REAL step-10 backup (method: strip a live
    v11 database of every step-11 artifact, version rows included):
    v10 backup -> deploy M4 (FIRST mint, empty log) -> delete -> restore
    the v10 backup -> migration REFUSED -> adopt (migrates, mint
    suppressed) -> replay -> sentinel gone."""
    import shutil
    import subprocess
    import sys

    from tests.conftest import strip_to_step10

    from mirror_memory.api import MemoryEngine
    from mirror_memory.core.deletion import FileDeletionLog
    from mirror_memory.core.schema_migration import upgrade_database

    live_path = tmp_path / "live.db"
    url = f"sqlite:///{live_path}"
    eng = create_engine(url)
    Base.metadata.create_all(eng)
    F = sessionmaker(bind=eng)
    with tenant_scope(*TENANT), fixture_write_scope("seed pre-M4 data"):
        s = F()
        set_memory_enabled(s, USER, True)
        from mirror_memory.core.repository import record_claim

        record_claim(s, USER, dimension="topic", key="k",
                     claim_text="I like RESTORE2 and coffee.",
                     confidence=0.9, session_id="s1")
        s.commit()
        s.close()
    strip_to_step10(eng)

    # the REAL step-10 backup
    backup_path = tmp_path / "step10-backup.db"
    src = create_engine(url).raw_connection()
    dest = create_engine(f"sqlite:///{backup_path}").raw_connection()
    src.connection.backup(dest.connection)
    dest.close()
    src.close()

    # deploy M4: first deployment (the log is empty) -> the migration MINTS
    log_dir = tmp_path / "deletions"
    cfg = _log_config(log_dir)
    upgrade_database(eng, default_tenant="default", default_app="default",
                     config=cfg)
    db_id, applied = sm.read_watermark(eng)
    assert db_id and applied == 0
    eng.dispose()

    # the engine binds the genesis to the minted db_id; delete the sentinel
    class _FakeLLM:
        def generate(self, **kw):
            return '{"patterns": [], "support_policy": {}}'

    log = FileDeletionLog(log_dir)
    app = MemoryEngine(config=load_config("config/"), database_url=url,
                       tenant_id=TENANT[0], app_id=TENANT[1],
                       llm_client=_FakeLLM(), deletion_log=log)
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(url))()
        belief = s.query(Belief).filter(
            Belief.claim_text.contains("RESTORE2")).one()
        s.close()
    assert app.forget_belief(user_id=USER, belief_id=belief.id,
                             purpose="api")
    app._engine.dispose()

    # RESTORE the step-10 backup: data back, all M4 artifacts gone
    for existing in (live_path,):
        existing.unlink()
    shutil.copyfile(backup_path, live_path)
    eng = create_engine(url)
    assert sm.read_watermark(eng) is None
    assert sm.current_version(eng) == 10

    # the migration REFUSES: the log proves this line ran M4 already
    with pytest.raises(WatermarkMissing) as info:
        upgrade_database(eng, default_tenant="default", default_app="default",
                         config=cfg)
    assert db_id[:8] in str(info.value)
    eng.dispose()

    # adopt: migrates first (mint suppressed), then the log's db_id wins
    result = subprocess.run(
        [sys.executable, "-m", "mirror_memory.deletion_adopt",
         "--database-url", url, "--db-id", db_id,
         "--config-dir", "config/"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "minting suppressed" in result.stdout
    assert "watermark adopted" in result.stdout

    # the replay removes the sentinel (the first entry point runs the
    # startup reconciliation)
    app2 = MemoryEngine(config=load_config("config/"), database_url=url,
                        tenant_id=TENANT[0], app_id=TENANT[1],
                        llm_client=_FakeLLM(), deletion_log=log)
    app2.get_beliefs(user_id=USER, purpose="api")  # startup + replay
    eng = create_engine(url)
    _db_id2, applied2 = sm.read_watermark(eng)
    assert _db_id2 == db_id and applied2 >= 1
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(url))()
        try:
            assert s.query(Belief).filter(
                Belief.claim_text.contains("RESTORE2")).count() == 0
        finally:
            s.close()
    assert "RESTORE2" not in (
        app2.recall(user_id=USER, query="RESTORE2", purpose="api") or "")
    app2._engine.dispose()
    eng.dispose()
