"""M4 batch-2 patch 5, tasks 1-3: the sync gate is complete.

- task 1: the offline grace expires by the CLOCK in normal state too -- a
  long-lived engine whose last COMPLETE catch-up ages past the grace has
  its reads refused on the next entry (no manual catch-up), one catch-up
  attempt is made on the spot, and a healed link resumes service without
  a restart;
- task 2: every reads_memory entrance in ENTRY_REGISTRY (plus the two
  read-shaped entrances without the audit flag, run_verification and
  run_metabolism) refuses while expired or catching up -- the refusal is
  raised, never swallowed into an empty result by an entrance's
  fail-open -- and after a complete catch-up the deleted sentinel is
  visible NOWHERE;
- task 3: sync.enabled=True with NO injected transport is not the
  disabled behavior -- reads serve only under the standard grace
  measured from the STORED completion (never invented, never refreshed),
  refuse when there is none / it is out / the completed peer is
  ambiguous; disabled sync keeps the original behavior; the
  schema_ready shared-engine wrapper does not bypass the gate.

The drivers below are derived from ENTRY_REGISTRY: a new reads_memory
entrance fails this test until a driver is written for it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import (
    MemoryEngine,
    SyncCatchupRequired,
    SyncCatchingUp,
)
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.authorization import ENTRY_REGISTRY
from mirror_memory.core.deletion import FileDeletionLog
from mirror_memory.core.models import Belief, SyncPeer
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from tests.test_edge_sync import _FakeLLM, _Transport
from tests.test_tombstone_sync import USER, _cfg

TENANT = ("default", "default")
SENTINEL = "SENGATE"


@pytest.fixture(autouse=True)
def _single_tenant_mode_off():
    from mirror_memory.core.tenancy import set_single_tenant_mode

    set_single_tenant_mode(False)
    yield
    set_single_tenant_mode(False)


def _engine(root, name, *, transport=None, enabled=True, shared=None,
            schema_ready=False):
    log_dir = root / f"{name}-deletions"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(log_dir)
    cfg.sync.enabled = enabled
    engine = MemoryEngine(
        config=cfg, database_url=f"sqlite:///{root / name}.db",
        tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
        deletion_log=FileDeletionLog(log_dir), single_tenant_mode=True,
        sync_transport=transport, engine=shared, schema_ready=schema_ready)
    engine._ensure_db()
    return engine


def _observe(engine, text, session_id="sg"):
    engine.observe(user_id=USER, session_id=session_id, text=text,
                   purpose="api")


def _beliefs(engine):
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(str(engine._database_url)))()
        try:
            return list(s.query(Belief).filter(Belief.user_id == USER).all())
        finally:
            s.close()


def _align(cloud, edge):
    """The edge's beliefs adopt the cloud's global_ids (a shared history)."""
    from mirror_memory.core.tenancy import fixture_write_scope as _fws

    ids = {b.key: b.global_id for b in _beliefs(cloud)}
    with tenant_scope(*TENANT), _fws("align identity"):
        s = sessionmaker(bind=create_engine(str(edge._database_url)))()
        for b in s.query(Belief).filter(Belief.user_id == USER).all():
            if b.key in ids:
                b.global_id = ids[b.key]
        s.commit()
        s.close()


def _backup(engine, backup_db: str, backup_log_dir) -> None:
    src = engine._engine.raw_connection()
    from sqlalchemy import create_engine as _ce

    dest = _ce(f"sqlite:///{backup_db}").raw_connection()
    src.connection.backup(dest.connection)
    dest.close()
    src.close()
    import shutil

    shutil.copytree(engine._deletion_log.directory, backup_log_dir,
                    dirs_exist_ok=True)


def _restore(engine, backup_db: str, backup_log_dir) -> None:
    import shutil

    engine._engine.dispose()
    shutil.copyfile(backup_db, str(engine._database_url).replace("sqlite:///", ""))
    shutil.rmtree(engine._deletion_log.directory, ignore_errors=True)
    shutil.copytree(backup_log_dir, engine._deletion_log.directory)


def _age(engine, days: float) -> None:
    with tenant_scope(*TENANT), fixture_write_scope("age the anchor"):
        s = sessionmaker(bind=create_engine(str(engine._database_url)))()
        s.query(SyncPeer).update(
            {"last_catchup_at": datetime.utcnow() - timedelta(days=days)},
            synchronize_session=False)
        s.commit()
        s.close()


# --------------------------------------------------------------------------
# The driver map: one driver per reads_memory entrance (plus the two
# read-shaped entrances without the audit flag).  A NEW reads_memory
# entrance in ENTRY_REGISTRY fails test_registry_covers_every_read until
# a driver exists here -- that is the auto-inclusion.
# --------------------------------------------------------------------------

def _drv_recall(e):
    return e.recall(user_id=USER, query=SENTINEL, purpose="api") or ""


def _drv_panel(e):
    return str(e.panel(user_id=USER, purpose="api").__dict__)


def _drv_get_beliefs(e):
    return str(e.get_beliefs(user_id=USER, purpose="api"))


def _drv_candidates(e):
    return str(e.get_verification_candidates(user_id=USER, limit=5,
                                             purpose="api"))


def _drv_belief_history(e):
    bid = next((b.id for b in _beliefs(e)
                if SENTINEL in b.claim_text), 12345)
    return str(e.belief_history(user_id=USER, belief_id=bid, purpose="api"))


def _drv_explain(e):
    bid = next((b.id for b in _beliefs(e)
                if SENTINEL in b.claim_text), 12345)
    return str(e.explain(user_id=USER, belief_id=bid, purpose="api"))


def _drv_export_page(e):
    return str(e.export_page(user_id=USER, purpose="api").to_dict())


def _drv_export(e):
    # the generator is FULLY iterated -- every page gates and audits
    chunks = []
    for page in e.export(user_id=USER, purpose="api"):
        chunks.append(str(page.to_dict()))
    return "".join(chunks)


def _drv_run_verification(e):
    return str(e.run_verification(user_id=USER, session_id="sg",
                                  user_text="i like it", purpose="api"))


def _drv_run_metabolism(e):
    return str(e.run_metabolism(user_id=USER, dry_run=True, purpose="api"))


DRIVERS = {
    "recall": _drv_recall,
    "panel": _drv_panel,
    "get_beliefs": _drv_get_beliefs,
    "get_verification_candidates": _drv_candidates,
    "belief_history": _drv_belief_history,
    "explain": _drv_explain,
    "export_page": _drv_export_page,
    "export": _drv_export,
    "run_verification": _drv_run_verification,
    "run_metabolism": _drv_run_metabolism,
}

# registry entries the drivers must cover: every reads_memory entrance of
# the engine, plus the two read-shaped entrances that carry no audit flag
REQUIRED = sorted({
    key.rsplit(".", 1)[-1]
    for key, meta in ENTRY_REGISTRY.items()
    if key.startswith("mirror_memory.api.MemoryEngine.")
    and (meta.get("reads_memory") or key.endswith(
        (".run_verification", ".run_metabolism")))
})


def test_registry_covers_every_read():
    """Every reads_memory entrance of ENTRY_REGISTRY (plus
    run_verification / run_metabolism) has a driver here -- a new read
    entrance without a driver fails this test."""
    missing = [name for name in REQUIRED if name not in DRIVERS]
    assert missing == [], f"gate drivers missing for: {missing}"


# --------------------------------------------------------------------------
# The scenario: cloud deletes the sentinel, the edge restores a backup
# taken before the deletions, and the channel has NOT caught up.
# --------------------------------------------------------------------------

def _seed_and_delete(cloud, edge):
    # the sentinel message lives ALONE in its own session: its evidence
    # then supports only the sentinel belief, so the belief tombstone
    # makes it an orphan and the deletion removes the raw text too (a
    # message shared with a SURVIVING belief legitimately survives)
    _observe(cloud, f"I like {SENTINEL} and coffee.", session_id="sg1")
    _observe(cloud, "I live in GateCity.", session_id="sg2")
    _observe(edge, f"I like {SENTINEL} and coffee.", session_id="sg1")
    _observe(edge, "I live in GateCity.", session_id="sg2")
    _align(cloud, edge)
    # a second version so belief_history has a chain holding the sentinel
    belief = next(b for b in _beliefs(edge) if SENTINEL in b.claim_text)
    edge.correct(user_id=USER, belief_id=belief.id,
                 new_claim_text=f"I like {SENTINEL} and tea.",
                 purpose="api")
    edge._engine.dispose()
    # the CLOUD deletes by ITS OWN local id (the tombstone travels by the
    # shared global_id the edge aligned to)
    cloud_belief = next(b for b in _beliefs(cloud)
                        if SENTINEL in b.claim_text)
    cloud.forget_belief(user_id=USER, belief_id=cloud_belief.id,
                        purpose="api")
    cloud.forget_session(user_id=USER, session_id="sg1", purpose="api")
    cloud.forget_session(user_id=USER, session_id="sg2", purpose="api")


def test_expired_and_catching_up_refuse_every_read(tmp_path):
    """Task 2: after the restore (data back, channel NOT caught up) and
    with the anchor aged past the grace, EVERY reads_memory entrance
    raises SyncCatchupRequired -- never an empty result -- and while
    catching up every entrance raises SyncCatchingUp.  After the healed
    catch-up every entrance runs and none of them shows the sentinel."""
    root = tmp_path
    cloud = _engine(root / "cl", "gc-cloud")
    edge_root = root / "edge"
    edge_root.mkdir()
    edge = _engine(edge_root, "gc-edge", transport=_Transport(cloud))

    _seed_and_delete(cloud, edge)

    # the edge's backup, taken BEFORE the deletions reached it
    backup_db = str(edge._database_url).replace("sqlite:///", "") + "-bak"
    backup_logs = edge_root / "log-bak"
    _backup(edge, backup_db, backup_logs)

    # the edge applied nothing yet: restore brings the sentinel back and
    # rewinds the channel cursor
    _restore(edge, backup_db, backup_logs)
    edge = _engine(edge_root, "gc-edge", transport=_Transport(
        cloud, fail_after=0))
    assert any(SENTINEL in b.claim_text for b in _beliefs(edge)), \
        "the restore resurrected the data (pre-catch-up)"

    # the anchor ages past the grace: the CLOCK expires the channel
    _age(edge, days=8.0)
    st = edge.sync_status()
    assert st["state"] == "expired", st

    # EVERY entrance refuses with the explicit error -- the fail-open
    # paths (candidates, run_verification) must NOT swallow it
    for name, driver in sorted(DRIVERS.items()):
        with pytest.raises(SyncCatchupRequired, match="") as info:
            driver(edge)
        assert "expired" in str(info.value) or "sync" in str(info.value).lower()

    # catching up refuses every entrance too
    edge._sync_state = "catching_up"
    for name, driver in sorted(DRIVERS.items()):
        with pytest.raises(SyncCatchingUp):
            driver(edge)
    edge._sync_state = "normal"

    # the link heals: the FIRST read re-attempts and completes the
    # catch-up -- service resumes with no restart and the sentinel is
    # visible nowhere
    edge._sync_transport = _Transport(cloud)
    for name, driver in sorted(DRIVERS.items()):
        result = driver(edge)  # must not raise
        assert SENTINEL not in result, name
        # NOTE: "GateCity" legitimately survives somewhere (a session-scope
        # deletion removes the summary layer, not the belief; a message
        # supporting a surviving belief survives as its evidence) -- the
        # DELETED belief's content is what must be gone everywhere
    edge._engine.dispose()
    cloud._engine.dispose()


def test_normal_state_expires_by_the_clock(tmp_path):
    """Task 1: the same engine instance, no manual _catch_up -- the aged
    anchor expires the channel on the next read (one automatic attempt is
    made), a healed link resumes service on that same read, within the
    grace the established offline behavior holds, and the reminder
    threshold still fires."""
    root = tmp_path
    cloud = _engine(root / "cl", "gc-cloud")
    edge_root = root / "edge"
    edge_root.mkdir()
    edge = _engine(edge_root, "gc-edge", transport=_Transport(cloud))
    _observe(edge, "I like clockedge and coffee.")
    edge._engine.dispose()
    cloud.forget_session(user_id=USER, session_id="sg", purpose="api")

    # complete catch-up at T0 (healthy transport): normal
    assert edge.sync_status()["state"] == "normal"
    t0 = edge.sync_status()["last_catchup_at"]

    # 6 days aged, link silently dead, NO manual catch-up: within the
    # grace the reads still serve (the established offline behavior)
    _age(edge, days=6.0)
    edge._sync_transport = _Transport(cloud, fail_after=0)
    assert isinstance(_drv_recall(edge), str)
    st = edge.sync_status()
    assert st["reminder"] is True and st["remaining_grace_days"] == \
        pytest.approx(1.0, abs=0.05)

    # 8 days aged, still no manual call: the FIRST read refuses (the
    # gate attempts once, the link is down)
    _age(edge, days=8.0)
    with pytest.raises(SyncCatchupRequired):
        _drv_recall(edge)
    assert edge.sync_status()["state"] == "expired"

    # the link heals: the next read re-attempts and serves -- no restart
    edge._sync_transport = _Transport(cloud)
    assert _drv_recall(edge) is not None
    assert edge.sync_status()["state"] == "normal"
    assert edge.sync_status()["last_catchup_at"] > t0
    edge._engine.dispose()
    cloud._engine.dispose()


def test_enabled_without_transport_uses_stored_grace(tmp_path):
    """Task 3: sync enabled, NO transport -- not the disabled behavior.
    No stored completion: reads refuse.  Within the stored grace: reads
    serve (the established offline window).  Past it: refuse.  Disabled
    sync: the original behavior.  Ambiguous history (two completed
    peers): refuse rather than guess.  The schema_ready wrapper does not
    bypass the gate."""
    root = tmp_path
    cloud = _engine(root / "cl", "gc-cloud")

    # (a) never completed: reads refuse, writes stay available
    root_a = tmp_path / "a"
    root_a.mkdir()
    e = _engine(root_a, "gc-a", transport=None)
    st = e.sync_status()
    assert st["state"] == "expired" and st["transport_injected"] is False
    with pytest.raises(SyncCatchupRequired):
        _drv_recall(e)
    _observe(e, "I like offline-write and coffee.")  # a write still works

    # (b) a completed catch-up in the stored history, within grace:
    # the honest label without a transport is offline_grace (patch 6
    # task 3) -- the engine IS offline, the stored grace still serves
    root_b = tmp_path / "b"
    root_b.mkdir()
    e2 = _engine(root_b, "gc-b", transport=_Transport(cloud))
    _observe(e2, "I like storedgrace and coffee.")
    assert e2.sync_status()["state"] == "normal"
    e2._engine.dispose()
    # reboot WITHOUT the transport: the stored completion serves the grace
    e2 = _engine(root_b, "gc-b", transport=None)
    st = e2.sync_status()
    assert st["state"] == "offline_grace", st
    assert st["transport_injected"] is False
    assert st["last_catchup_at"] is not None
    assert st["remaining_grace_days"] == pytest.approx(7.0, abs=0.1)
    assert st["reminder"] is False
    assert _drv_recall(e2) is not None
    # aged past the grace: refuse
    _age(e2, days=8.0)
    with pytest.raises(SyncCatchupRequired):
        _drv_recall(e2)
    e2._engine.dispose()

    # (c) disabled sync: the original behavior -- no gate at all
    root_c = tmp_path / "c"
    root_c.mkdir()
    e3 = _engine(root_c, "gc-c", transport=None, enabled=False)
    assert e3.sync_status()["state"] == "disabled"
    _drv_recall(e3)  # serves
    e3._engine.dispose()

    # (d) two completed peers: the trusted peer is ambiguous -- refuse
    root_d = tmp_path / "d"
    root_d.mkdir()
    e4 = _engine(root_d, "gc-d", transport=None)
    with tenant_scope(*TENANT), fixture_write_scope("two peers"):
        s = sessionmaker(bind=create_engine(str(e4._database_url)))()
        now = datetime.utcnow()
        s.add(SyncPeer(peer_db_id="a" * 32, cursor=5, last_catchup_at=now))
        s.add(SyncPeer(peer_db_id="b" * 32, cursor=7, last_catchup_at=now))
        s.commit()
        s.close()
    with pytest.raises(SyncCatchupRequired):
        _drv_recall(e4)
    e4._engine.dispose()

    # (e) the schema_ready shared-engine wrapper does not bypass the gate:
    # it skips the startup catch-up by design, the entry gate still fires
    root_e = tmp_path / "e"
    root_e.mkdir()
    base = _engine(root_e, "gc-e-base", transport=None)
    with tenant_scope(*TENANT), fixture_write_scope("two peers wrapper"):
        s = sessionmaker(bind=create_engine(str(base._database_url)))()
        s.add(SyncPeer(peer_db_id="c" * 32, cursor=3,
                       last_catchup_at=datetime.utcnow() - timedelta(days=30)))
        s.commit()
        s.close()
    wrapper = MemoryEngine(
        config=base._config, database_url=base._database_url,
        tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
        deletion_log=base._deletion_log, single_tenant_mode=True,
        engine=base._engine, schema_ready=True)
    wrapper._ensure_db()  # the schema_ready path: binds only
    with pytest.raises(SyncCatchupRequired):
        _drv_recall(wrapper)
    base._engine.dispose()
    cloud._engine.dispose()


# --- M4 batch-2 patch 6, task 2: the schema_ready shared-engine wrapper
# --- runs the startup catch-up too

def test_shared_wrapper_startup_catchup(tmp_path):
    """A schema_ready wrapper WITH a transport must not silently skip the
    startup catch-up: the cloud's new deletions are applied before the
    first read, a second _ensure_db does not re-run it, and offline the
    wrapper falls back to the stored grace (expired refuses, catching up
    refuses)."""
    cloud = _engine(tmp_path / "cl", "gc-cl")
    base = _engine(tmp_path / "b", "gc-b", transport=_Transport(cloud))
    _observe(base, "I like sharedwrap and coffee.")
    _align(base, cloud)
    assert base.sync_status()["state"] == "normal"
    base._engine.dispose()

    # the cloud holds its own copy of the sentinel memory
    _observe(cloud, "I like sharedwrap and coffee.")
    # the cloud deletes the sentinel AFTER the base engine's catch-up:
    # the cloud's OWN copy of the belief adopts the base's global_id
    # (matched by content -- the two extractors mint different keys), so
    # the tombstone travels to the wrapper by it
    from mirror_memory.core.tenancy import fixture_write_scope as _fws

    gid = next(b.global_id for b in _beliefs(base)
               if "sharedwrap" in b.claim_text)
    with tenant_scope(*TENANT), _fws("align wrapper cloud"):
        s = sessionmaker(bind=cloud._engine)()
        for b in s.query(Belief).filter(Belief.user_id == USER).all():
            if "sharedwrap" in b.claim_text:
                b.global_id = gid
        s.commit()
        s.close()
    cb = next(b for b in _beliefs(cloud) if "sharedwrap" in b.claim_text)
    assert cloud.forget_belief(user_id=USER, belief_id=cb.id, purpose="api")

    # the wrapper binds the SAME (already-prepared) database with
    # schema_ready=True and a live transport: its startup catch-up must
    # apply the deletion before anything serves
    transport = _Transport(cloud)
    wrapper = MemoryEngine(
        config=base._config, database_url=base._database_url,
        tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
        deletion_log=base._deletion_log, single_tenant_mode=True,
        engine=base._engine, schema_ready=True, sync_transport=transport)
    wrapper._ensure_db()
    calls_after_startup = transport.calls
    assert calls_after_startup >= 1, "the startup catch-up must run"
    assert not any("sharedwrap" in b.claim_text
                   for b in _beliefs(wrapper)), \
        "the cloud's deletion must be applied before the first read"

    # a second _ensure_db: idempotent, no duplicate catch-up
    wrapper._ensure_db()
    assert transport.calls == calls_after_startup
    assert wrapper.sync_status()["state"] == "normal"

    # offline: the link dies -- reads fall back to the stored grace; the
    # anchor is the wrapper's OWN fresh completion, so reads serve within
    # it and refuse once it is out; catching up refuses outright
    wrapper._sync_transport = _Transport(cloud, fail_after=0)
    wrapper._catch_up()  # a failing attempt puts the state machine offline
    assert wrapper.sync_status()["state"] in ("offline_grace", "expired")
    wrapper._sync_state = "catching_up"
    with pytest.raises(SyncCatchingUp):
        _drv_recall(wrapper)
    wrapper._sync_state = "normal"
    wrapper._engine.dispose()
    cloud._engine.dispose()


def _session_of(engine):
    return sessionmaker(bind=engine._engine)()
