"""M4 batch 2 task 3: the edge catches up against the cloud BEFORE it
serves, and the offline grace period decides what happens when the peer
is unreachable.

- full machine restore: main database AND log directory restored
  together -> the first startup pulls and applies the cloud's tombstones
  -> every read entrance is sentinel-free;
- a catch-up interrupted by a dead link resumes from the durable
  per-batch cursor on the next startup;
- the offline grace (sync.offline_grace_days, from the last successful
  catch-up stored in the main database): within it the edge serves
  (offline_grace, with a reminder once the remaining grace is at or
  below sync.reminder_days); past it read entries refuse with
  SyncCatchupRequired -- and coming back online heals WITHOUT a restart;
- the four states (normal / catching_up / offline_grace / expired) and
  the user-facing sync_status().
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine, SyncCatchupRequired, SyncCatchingUp
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import FileDeletionLog
from mirror_memory.core.models import Base, Belief, SessionSummary, SyncPeer
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from tests.test_tombstone_sync import USER, _cfg

TENANT = ("default", "default")


@pytest.fixture(autouse=True)
def _single_tenant_mode_off():
    """The edge engines run single_tenant_mode=True, which flips the
    process-wide contextvar in _ensure_db; reset it so the leak does not
    disable tenant filtering in the tests that follow (any order)."""
    from mirror_memory.core.tenancy import set_single_tenant_mode

    set_single_tenant_mode(False)
    yield
    set_single_tenant_mode(False)


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


class _Transport:
    """The in-process link to the cloud engine (no network in this batch):
    the peer's export, optionally failing from call N on."""

    def __init__(self, peer, *, fail_after=None):
        self.peer = peer
        self.calls = 0
        self.fail_after = fail_after

    @property
    def db_id(self):
        return self.peer._db_id

    def export_tombstones(self, cursor, limit, peer_db_id=None):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ConnectionError("the link is down")
        return self.peer.export_tombstones(cursor, limit, peer_db_id=peer_db_id)


def _edge(root: Path, transport=None, *, grace_days=7, reminder_days=2):
    cfg = _cfg(root / "edge-deletions")
    cfg.llm_client = _FakeLLM()
    cfg.sync.enabled = True
    cfg.sync.offline_grace_days = grace_days
    cfg.sync.reminder_days = reminder_days
    engine = MemoryEngine(
        config=cfg, database_url=f"sqlite:///{root / 'edge.db'}",
        tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
        deletion_log=FileDeletionLog(root / "edge-deletions"),
        single_tenant_mode=True, sync_transport=transport)
    engine._ensure_db()
    return engine


def test_full_restore_then_catchup_cleans_every_read_entrance(tmp_path):
    """Edge backup (main DB + log dir) -> the user deletes on the cloud
    (three scopes) -> the OLD edge applies -> FULL edge restore -> the
    next startup catches up -> all read entrances are sentinel-free."""
    root = tmp_path
    cloud_root = root / "cloud"
    cloud_root.mkdir()
    cloud_cfg = _cfg(cloud_root / "deletions")
    cloud_cfg.llm_client = _FakeLLM()
    cloud = MemoryEngine(config=cloud_cfg, database_url=f"sqlite:///{cloud_root / 'cloud.db'}",
                         tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
                         deletion_log=FileDeletionLog(cloud_root / "deletions"),
                         single_tenant_mode=False)
    cloud._ensure_db()

    def seed(engine, tag):
        engine.observe(user_id=USER, session_id="sa",
                       text=f"I like {tag} and coffee.", purpose="api")
        engine.observe(user_id=USER, session_id="sb",
                       text="I live in Berlin.", purpose="api")

    seed(cloud, "CLOUDSENTINEL")

    # the edge: seed, then BACKUP (main DB + log dir together)
    edge_root = root / "edge"
    edge_root.mkdir()
    edge = _edge(edge_root, transport=None)
    seed(edge, "CLOUDSENTINEL")
    edge._engine.dispose()
    backup_db = root / "edge-backup.db"
    src = create_engine(f"sqlite:///{root / 'edge.db'}").raw_connection()
    dest = create_engine(f"sqlite:///{backup_db}").raw_connection()
    src.connection.backup(dest.connection)
    dest.close()
    src.close()
    backup_logs = root / "edge-deletions-backup"
    (root / "edge-deletions").mkdir(exist_ok=True)
    shutil.copytree(root / "edge-deletions", backup_logs)

    # the user deletes on the CLOUD: belief + session + user scopes
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(f"sqlite:///{cloud_root / 'cloud.db'}"))()
        belief = s.query(Belief).filter(Belief.claim_text.contains("CLOUDSENTINEL")).one()
        s.close()
    cloud.forget_belief(user_id=USER, belief_id=belief.id, purpose="api")
    cloud.forget_session(user_id=USER, session_id="sb", purpose="api")
    cloud.delete_memories(user_id=USER, purpose="api")

    # the old edge applies the tombstones (it is still online)
    edge = _edge(edge_root, transport=None)
    batch, _ = cloud.export_tombstones(0, 100, peer_db_id=edge._db_id)
    edge.apply_tombstones(batch)
    edge._engine.dispose()

    # FULL MACHINE RESTORE: the main DB and the log directory go back
    shutil.copyfile(backup_db, root / "edge.db")
    shutil.rmtree(root / "edge-deletions", ignore_errors=True)
    shutil.copytree(backup_logs, root / "edge-deletions")

    # the next startup CATCHES UP against the cloud before serving
    transport = _Transport(cloud)
    edge2 = _edge(edge_root, transport=transport)
    assert "CLOUDSENTINEL" not in (edge2.recall(user_id=USER, query="CLOUDSENTINEL", purpose="api") or "")
    assert "CLOUDSENTINEL" not in (edge2.get_beliefs(user_id=USER, purpose="api") and
                                   str([b.__dict__ for b in edge2.get_beliefs(user_id=USER, purpose="api")]))
    assert "CLOUDSENTINEL" not in (edge2.export_page(user_id=USER, purpose="api").to_dict().__str__())
    assert edge2.sync_status()["state"] == "normal"
    edge2._engine.dispose()
    cloud._engine.dispose()


def test_catchup_resumes_after_a_dead_link(tmp_path):
    """The link dies mid-catch-up: the durable per-batch cursor means the
    NEXT startup finishes the job without redoing finished batches."""
    root = tmp_path
    cloud_root = root / "cloud"
    cloud_root.mkdir()
    cloud_cfg = _cfg(cloud_root / "deletions")
    cloud_cfg.llm_client = _FakeLLM()
    cloud = MemoryEngine(config=cloud_cfg, database_url=f"sqlite:///{cloud_root / 'cloud.db'}",
                         tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
                         deletion_log=FileDeletionLog(cloud_root / "deletions"),
                         single_tenant_mode=False)
    cloud._ensure_db()
    for i in range(6):
        cloud.observe(user_id=USER, session_id=f"s{i}", text=f"I like thing{i}.", purpose="api")
        with tenant_scope(*TENANT):
            s = sessionmaker(bind=create_engine(f"sqlite:///{cloud_root / 'cloud.db'}"))()
            b = s.query(Belief).filter(Belief.claim_text.contains(f"thing{i}")).first()
            s.close()
        if b is not None:
            cloud.forget_belief(user_id=USER, belief_id=b.id, purpose="api")

    edge_root = root / "edge"
    edge_root.mkdir()
    broken = _Transport(cloud, fail_after=1)  # the 2nd export call: link down
    e1 = _edge(edge_root, transport=broken)
    assert e1.sync_status()["state"] in ("offline_grace", "expired")
    e1._engine.dispose()

    healed = _Transport(cloud)
    e2 = _edge(edge_root, transport=healed)
    assert e2.sync_status()["state"] == "normal"
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(f"sqlite:///{edge_root / 'edge.db'}"))()
        left = s.query(Belief).filter(Belief.user_id == USER).count()
        s.close()
    assert left == 0
    assert healed.calls >= 1
    e2._engine.dispose()
    cloud._engine.dispose()


def _age_last_catchup(engine, days: float) -> None:
    """Rewind the stored last catch-up: the grace arithmetic then sees an
    edge that has not synced for *days*."""
    F = sessionmaker(bind=create_engine(str(engine._database_url)))
    with tenant_scope(*TENANT), fixture_write_scope("test: age the peer state"):
        s = F()
        s.query(SyncPeer).update(
            {"last_catchup_at": datetime.now(UTC) - timedelta(days=days)},
            synchronize_session=False)
        s.commit()
        s.close()


def test_four_states_and_the_reminder_threshold(tmp_path):
    root = tmp_path / "edge"
    root.mkdir()
    cloud_root = tmp_path / "cloud"
    cloud_root.mkdir()
    cloud_cfg = _cfg(cloud_root / "deletions")
    cloud_cfg.llm_client = _FakeLLM()
    cloud = MemoryEngine(config=cloud_cfg, database_url=f"sqlite:///{cloud_root / 'cloud.db'}",
                         tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
                         deletion_log=FileDeletionLog(cloud_root / "deletions"),
                         single_tenant_mode=False)
    cloud._ensure_db()
    # grace 10 days, reminder at 2: remaining > 2 -> served, no reminder
    edge = _edge(root, transport=_Transport(cloud), grace_days=10, reminder_days=2)
    assert edge.sync_status()["state"] == "normal"
    assert edge.sync_status()["reminder"] is False

    # 8 days since the last catch-up, link down: within the grace (2 days
    # left) -- offline_grace, the reminder fires, reads still serve
    _age_last_catchup(edge, days=8.0)
    edge._sync_transport = _Transport(cloud, fail_after=0)  # every call fails
    edge._catch_up()
    st = edge.sync_status()
    assert st["state"] == "offline_grace" and st["reminder"] is True, st
    assert st["remaining_grace_days"] == pytest.approx(2.0, abs=0.05)
    edge.recall(user_id=USER, query="x", purpose="api")  # serves

    # 11 days, still down: expired -- reads refuse with the user-facing
    # error type (its message is for the product, no internals)
    _age_last_catchup(edge, days=11.0)
    with pytest.raises(SyncCatchupRequired) as info:
        edge.recall(user_id=USER, query="x", purpose="api")
    assert info.value.user_message and "联网" in info.value.user_message
    assert info.value.last_catchup_at is not None
    assert edge.sync_status()["state"] == "expired"
    # a write still goes through (deleting offline is fine; it syncs later)
    edge.forget_session(user_id=USER, session_id="never", purpose="api")

    # back online: the entry itself re-attempts and service resumes
    # without a restart
    edge._sync_transport = _Transport(cloud)
    edge.recall(user_id=USER, query="anything", purpose="api")
    assert edge.sync_status()["state"] == "normal"
    assert edge.sync_status()["reminder"] is False

    # catching_up refuses entries (like replay mode)
    edge._sync_state = "catching_up"
    with pytest.raises(SyncCatchingUp):
        edge.recall(user_id=USER, query="x", purpose="api")
    edge._sync_state = "normal"
    edge._engine.dispose()
    cloud._engine.dispose()
