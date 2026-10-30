"""M4 task 4: deletion racing the publish, 50 real rounds -- and task 5's
backup/restore e2e with the REAL backup tools (pg_dump / pg_restore on
PostgreSQL; SQLite's backup API embedded).

Each race round: memory + a claimed job; a deletion (user scope)
commits while the worker publishes.  Every round ends ONE of two ways:
- the publish completed BEFORE the deletion committed -> the deletion
  removes what it wrote (nothing lands after the deletion);
- the job was cancelled/deleted (deletion_generation_changed,
  LeaseLost, or the job row gone) -> zero writes.
A publish that lands AFTER the deletion committed is the failure.

The e2e flows drive the REAL restore path: pg_dump the main DB, delete,
pg_restore it back (the watermark row returns with the backup), startup
reconciliation replays the log, and no read surface (inventory-driven)
carries the sentinel.
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
import os
import subprocess

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.schema_migration import Step11DeletionReplay
from mirror_memory.core.deletion import FileDeletionLog
from mirror_memory.core.models import (
    Base, Belief, DeletionGeneration, EvolutionJob, Snapshot,
)
from mirror_memory.core.replay import reconcile
from mirror_memory.core.sql_audit import no_audit_scope
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.worker.evolution import _claim_job, _process_single_job, enqueue_job

USER = "u-race"
TENANT = ("default", "default")
SENTINEL = "RACE-SENTINEL-9"


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


def _config():
    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    cfg.worker.job_lease_seconds = 60
    return cfg


def _seed(db) -> None:
    Base.metadata.create_all(db)
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()


def _race_round(db, round_no) -> None:
    """One deletion-vs-publish round (SQLite: writer serialisation; the
    worker's publish and the deletion interleave through the file lock)."""
    cfg = _config()
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("seed the round"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()

    outcomes: dict = {}

    def run_publish():
        try:
            with tenant_scope(*TENANT):
                s = F()
                try:
                    job = enqueue_job(s, USER, cfg, purpose="api")
                    s.commit()
                    token = _claim_job(s, job.id, lease_seconds=60)
                    row = s.get(EvolutionJob, job.id)
                    _process_single_job(s, row, cfg, token)
                    s.commit()
                    s.expire_all()
                    final = s.get(EvolutionJob, job.id)
                    outcomes["publish"] = (final.status if final else "deleted",
                                           final.error_code if final else "")
                except Exception as exc:
                    # a job ROW deleted under the publisher (user-scope
                    # delete) surfaces here -- that IS the "job deleted"
                    # outcome, not an error
                    s.rollback()
                    outcomes["publish"] = ("deleted-or-cancelled", repr(exc))
                finally:
                    s.close()
        except Exception as exc:  # noqa: BLE001 -- recorded
            outcomes["publish"] = ("error", repr(exc))

    def run_delete():
        try:
            with tenant_scope(*TENANT):
                s = F()
                try:
                    me = MemoryEngine(config=cfg, database_url=str(db.url),
                                      tenant_id=TENANT[0], app_id=TENANT[1],
                                      llm_client=_FakeLLM())
                    me.delete_memories(user_id=USER, purpose="api")
                    outcomes["delete"] = "committed"
                finally:
                    s.close()
        except Exception as exc:  # noqa: BLE001
            outcomes["delete"] = repr(exc)

    # the memory the publish would write through (claimed work): observe
    # first, then enqueue so the publish has something to synthesise
    with tenant_scope(*TENANT), fixture_write_scope("seed content"):
        s = F()
        from mirror_memory.core.repository import record_claim

        record_claim(s, USER, dimension="topic", key=f"k{round_no}",
                     claim_text=f"round {round_no} content", confidence=0.8,
                     session_id=f"sr{round_no}")
        s.commit()
        s.close()

    tp = threading.Thread(target=run_publish)
    td = threading.Thread(target=run_delete)
    td.start()
    tp.start()
    td.join(timeout=60)
    tp.join(timeout=60)

    # BOTH threads finished (no poisoning for the next round)
    assert "delete" in outcomes and "publish" in outcomes, outcomes
    assert "error', repr" not in str(outcomes.get("publish", "")), outcomes

    # the invariant: whatever the publish wrote is GONE once the deletion
    # has committed
    with tenant_scope(*TENANT):
        s = F()
        try:
            left = s.query(Belief).filter(Belief.user_id == USER).count()
            gen = s.execute(
                select(DeletionGeneration.generation)
                .where(DeletionGeneration.user_id == USER)
                .order_by(DeletionGeneration.generation.desc())
            ).scalars().first()
        finally:
            s.close()
    assert left == 0, (outcomes, left)
    assert gen is not None and gen >= 1


def test_deletion_vs_publish_50_rounds_sqlite(tmp_path):
    db = create_engine(f"sqlite:///{tmp_path / 'race.db'}", connect_args={"timeout": 30.0})
    _seed(db)
    for round_no in range(50):
        _race_round(db, round_no)
    db.dispose()


# ---------------------------------------------------------------------------
# task 5: the real backup tools
# ---------------------------------------------------------------------------


def _docker() -> str | None:
    """The docker CLI when the PG server runs in the container mm-pg (the
    local dev shape); CI installs the client tools natively."""
    import shutil

    if shutil.which("docker") and subprocess.run(
        ["docker", "inspect", "mm-pg"], capture_output=True
    ).returncode == 0:
        return "docker"
    return None


def _pg_dump(main_db: str, out: str, main_url: str | None = None) -> None:
    """A custom-format dump of *main_db*, taken inside the container when
    the server runs there (the file lands on the host either way)."""
    env = dict(os.environ, PGPASSWORD="mm")
    docker = _docker()
    if docker:
        result = subprocess.run(
            ["docker", "exec", "-e", "PGPASSWORD=mm", "mm-pg",
             "pg_dump", "-U", "postgres", "--format=custom", main_db],
            env=env, capture_output=True, check=True)
        Path(out).write_bytes(result.stdout)
    else:
        # the CI runner has the client tools; connect explicitly (a bare
        # db name would use the OS user's local socket).  Strip ONLY the
        # driver segment: "+psycopg" leaves a bare scheme otherwise.
        pg_url = (main_url or "").replace("postgresql+psycopg://", "postgresql://", 1)
        subprocess.run(["pg_dump", "--format=custom", "--file", out, pg_url],
                       env=env, check=True)


def _pg_restore(main_db: str, dump: str) -> None:
    env = dict(os.environ, PGPASSWORD="mm")
    docker = _docker()
    if docker:
        psql = (["docker", "exec", "-e", "PGPASSWORD=mm", "mm-pg", "psql",
                 "-U", "postgres"]) + ["-c"]
    else:
        # the CI runner's psql connects via TCP with the password (there is
        # no local socket for a service-container server)
        pg_admin = os.environ["MM_DATABASE_URL"].rsplit("/", 1)[0] + "/postgres"
        pg_admin = pg_admin.replace("postgresql+psycopg://", "postgresql://", 1)
        psql = ["psql", pg_admin, "-c"]
    import time

    result = None
    for attempt in range(5):
        # WITH (FORCE) terminates attached backends (PG 13+) -- the e2e's
        # own leaked pools would otherwise hold the drop forever
        result = subprocess.run(
            psql + [f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)'],
            env=env, capture_output=True, text=True)
        if result.returncode == 0:
            break
        time.sleep(0.5 * (attempt + 1))  # terminated backends take a moment
    if result.returncode != 0:
        raise RuntimeError(f"drop failed: {result.stderr or result.stdout}")
    subprocess.run(psql + [f'CREATE DATABASE "{main_db}"'], env=env, check=True)
    restore = (["docker", "exec", "-i", "-e", "PGPASSWORD=mm", "mm-pg",
                "pg_restore", "-U", "postgres",
                "--dbname", main_db] if docker else
               ["pg_restore", "--dbname", main_db])
    subprocess.run(restore, input=Path(dump).read_bytes(), env=env, check=True)


@pytest.mark.skipif(
    not os.environ.get("MM_DATABASE_URL", "").startswith("postgresql"),
    reason="needs MM_DATABASE_URL pointing at PostgreSQL",
)
def test_backup_restore_replay_end_to_end_pg(tmp_path):
    """The FULL restore story with the real tools: pg_dump -> deletions ->
    pg_restore -> startup reconciliation -> every read surface clean.
    Runs against its OWN database (the dump/restore drops and recreates
    it), owner-side first: the app role never runs DDL."""
    main_db = "mm_test_m4_e2e"
    admin_url = os.environ["MM_DATABASE_URL"].rsplit("/", 1)[0] + "/postgres"
    main_url = os.environ["MM_DATABASE_URL"].rsplit("/", 1)[0] + "/" + main_db
    dump_path = str(tmp_path / "main.dump")
    log = FileDeletionLog(tmp_path / "deletions")

    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with no_audit_scope(), admin_engine.connect() as conn:
            # start clean: a leftover from a failed previous run would carry
            # a watermark ahead of this run's fresh log; WITH (FORCE)
            # terminates attached backends (the e2e's own leaked pools)
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
            conn.exec_driver_sql(f'CREATE DATABASE "{main_db}"')
        admin_engine.dispose()

        # the owner builds the schema + RLS + watermark (the app role never
        # runs DDL); the app role then serves
        from mirror_memory.core.models import Base as _Base

        owner_engine = create_engine(main_url)
        _Base.metadata.create_all(owner_engine)
        from mirror_memory.core.schema_migration import (
            Step11DeletionReplay, install_row_level_security,
        )

        # the same call the harness makes: ownership, grants, ENABLE+FORCE,
        # policies -- everything the app role needs to serve
        install_row_level_security(owner_engine, verify=False)
        Step11DeletionReplay().apply(owner_engine)
        Step11DeletionReplay().verify(owner_engine)
        owner_engine.dispose()

        # the runtime serves through the APP role (mode A) like production
        # -- re-point THIS database's URL at the app login
        from sqlalchemy.engine import make_url

        from tests.conftest import PG_TEST_ROLE_PASSWORDS

        runtime_url = (
            make_url(main_url)
            .set(username="mm_app", password=PG_TEST_ROLE_PASSWORDS["app"])
            .render_as_string(hide_password=False)
        )
        e2e_cfg = _config()
        e2e_cfg.session_summary_enabled = True  # the session-scope delete needs rows
        eng = MemoryEngine(config=e2e_cfg, database_url=runtime_url,
                           tenant_id=TENANT[0], app_id=TENANT[1],
                           llm_client=_FakeLLM(), deletion_log=log)
        eng._ensure_db()
        sqlalchemy_engine = eng._engine
        eng.observe(user_id=USER, session_id="s1",
                    text="I like PGSENTINEL and coffee.", purpose="api")

        # BACKUP (the real tool)
        _pg_dump(main_db, dump_path, main_url=main_url)

        # three scopes of deletion
        F = sessionmaker(bind=create_engine(runtime_url))
        with tenant_scope(*TENANT):
            s = F()
            try:
                belief = s.query(Belief).filter(
                    Belief.claim_text.contains("PGSENTINEL")).first()
            finally:
                s.close()
        assert eng.forget_belief(user_id=USER, belief_id=belief.id, purpose="api") is True
        eng.observe(user_id=USER, session_id="s2", text="I also nap at noon.", purpose="api")
        eng.forget_session(user_id=USER, session_id="s2", purpose="api")
        eng.delete_memories(user_id=USER, purpose="api")
        db_id, applied = sm.read_watermark(create_engine(runtime_url))
        tail = log.head(db_id)[0]
        assert applied == tail >= 3

        # RESTORE with the real tool: the watermark row comes back with it
        sqlalchemy_engine.dispose()
        _pg_restore(main_db, dump_path)

        # STARTUP: the replay happens at reconcile -- the app role cannot
        # run the watermark DDL, so the check runs owner-side first (a
        # restored production database is brought up by the owner too),
        # then the app engine serves
        owner_engine = create_engine(main_url)
        reconcile(lambda: sessionmaker(bind=owner_engine)(), log)
        owner_engine.dispose()
        _db_id2, applied2 = sm.read_watermark(create_engine(runtime_url))
        assert applied2 == tail, (applied2, tail)

        # every read surface is clean (the SAME log instance: a default-
        # constructed engine would look at a different log and refuse)
        eng2 = MemoryEngine(config=e2e_cfg, database_url=runtime_url,
                            tenant_id=TENANT[0], app_id=TENANT[1],
                            llm_client=_FakeLLM(), deletion_log=log)
        assert "PGSENTINEL" not in (eng2.recall(user_id=USER, query="PGSENTINEL", purpose="api") or "")
        assert "PGSENTINEL" not in json.dumps(
            eng2.export_page(user_id=USER, purpose="api").to_dict(), default=str)
        assert eng2.get_beliefs(user_id=USER, purpose="api") == []
        eng2._engine.dispose()

        # the e2e's own database is dropped again (it must not leak)
        with no_audit_scope(), admin_engine.connect() as conn:
            conn.exec_driver_sql(
                f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
        admin_engine.dispose()
    finally:
        with contextlib.suppress(Exception):
            cleanup = create_engine(admin_url, isolation_level="AUTOCOMMIT")
            with no_audit_scope(), cleanup.connect() as conn:
                conn.exec_driver_sql(
                    f'DROP DATABASE IF EXISTS "{main_db}" WITH (FORCE)')
            cleanup.dispose()


def test_id_reuse_is_not_fooled_by_the_fingerprint(tmp_path):
    """BACKUP -> create A -> delete A -> RESTORE -> create B (same id) ->
    replay: B survives (different created_at), A does not resurrect."""
    db = create_engine(f"sqlite:///{tmp_path / 'reuse.db'}")
    _seed(db)
    log = FileDeletionLog(tmp_path / "deletions")
    main_engine = create_engine(str(db.url))
    eng = MemoryEngine(config=_config(), database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1],
                       llm_client=_FakeLLM(), deletion_log=log)

    backup_url = str(db.url).replace("reuse.db", "reuse-backup.db")
    src = main_engine.raw_connection()
    dest = create_engine(backup_url).raw_connection()
    src.connection.backup(dest.connection)
    dest.close()
    src.close()

    # create A (with a KNOWN created_at), then delete it
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("create A"):
        s = F()
        from mirror_memory.core.repository import record_claim

        record_claim(s, USER, dimension="topic", key="a", claim_text="belief A",
                     confidence=0.9, session_id="s1")
        row = s.query(Belief).filter_by(key="a").one()
        row.created_at = row.created_at.replace(year=2020)
        s.commit()
        a_id = row.id
        s.close()
    eng.forget_belief(user_id=USER, belief_id=a_id, purpose="api")

    # RESTORE (A is back), then create B with the SAME id and a DIFFERENT
    # created_at (the autoincrement continues after a restore into the
    # same file -- force the reuse explicitly to pin the mechanism)
    main_engine.dispose()
    db.dispose()
    restored = create_engine(backup_url)
    raw = restored.raw_connection()
    dest2 = create_engine(str(eng._database_url)).raw_connection()
    raw.connection.backup(dest2.connection)
    dest2.close()
    raw.close()
    restored.dispose()

    F = sessionmaker(bind=create_engine(eng._database_url))
    with tenant_scope(*TENANT), fixture_write_scope("create B with A's id"):
        s = F()
        from mirror_memory.core.repository import record_claim

        record_claim(s, USER, dimension="topic", key="b", claim_text="belief B",
                     confidence=0.9, session_id="s2")
        row = s.query(Belief).filter_by(key="b").one()
        row.id = a_id  # force the id reuse
        row.created_at = row.created_at.replace(year=2026)
        s.commit()
        b_id = row.id
        s.close()
    assert b_id == a_id, "the test must actually reuse the id"

    reconcile(lambda: F(), log, tenant=TENANT)

    with tenant_scope(*TENANT):
        s = F()
        try:
            rows = s.query(Belief).all()
        finally:
            s.close()
    assert [(r.id, r.claim_text) for r in rows] == [(a_id, "belief B")], (
        "B (recreated) must survive; A must not resurrect")
