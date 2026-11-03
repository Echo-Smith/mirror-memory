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

from mirror_memory.api import (
    MemoryEngine,
    SyncCatchupRequired,
    SyncCatchingUp,
    SyncTransportIdentityMissing,
)
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
    the peer's export, optionally failing from call N on and optionally
    capping the batch size (to drive multi-batch catch-ups)."""

    def __init__(self, peer, *, fail_after=None, batch_limit=None):
        self.peer = peer
        self.calls = 0
        self.fail_after = fail_after
        self.batch_limit = batch_limit

    @property
    def db_id(self):
        return self.peer._db_id

    def export_tombstones(self, cursor, limit, peer_db_id=None):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ConnectionError("the link is down")
        return self.peer.export_tombstones(
            cursor, min(limit, self.batch_limit) if self.batch_limit
            else limit)


class _LyingTransport(_Transport):
    """A transport that declares a DIFFERENT peer than the one it dials
    (the envelope's sender will disagree with the declared db_id)."""

    def __init__(self, peer, *, declared_db_id):
        super().__init__(peer)
        self._declared = declared_db_id

    @property
    def db_id(self):
        return self._declared


class _NoIdentityTransport(_Transport):
    """A transport that cannot name its peer at all (no db_id)."""

    @property
    def db_id(self):
        return None


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
    batch, _ = cloud.export_tombstones(0, 100)
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


def _align_belief(cloud, edge, tag: str) -> None:
    """The edge's copy of the belief carrying *tag* adopts the cloud's
    global_id (what a shared history implies once sync is on)."""
    from mirror_memory.core.models import Belief as _B
    from mirror_memory.core.tenancy import fixture_write_scope

    cb = next(b for b in _beliefs(cloud) if tag in b.claim_text)
    with tenant_scope(*TENANT), fixture_write_scope("align identity"):
        s = sessionmaker(bind=create_engine(str(edge._database_url)))()
        row = s.get(_B, (*TENANT,
                       next(b.id for b in _beliefs(edge)
                            if tag in b.claim_text)))
        row.global_id = cb.global_id
        s.commit()
        s.close()


def _beliefs(engine) -> list:
    from mirror_memory.core.tenancy import tenant_scope

    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(str(engine._database_url)))()
        try:
            return list(s.query(Belief).filter(
                Belief.user_id == USER).all())
        finally:
            s.close()


def _peer_state(engine, peer_db_id):
    """(cursor, chain_hash, last_catchup_at) of the engine's channel row
    for *peer_db_id*."""
    from mirror_memory.core.repository import read_sync_peer
    from mirror_memory.core.tenancy import tenant_scope

    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(
            str(engine._database_url)))()
        try:
            row = read_sync_peer(s, peer_db_id)
            return (row.cursor if row else None,
                    row.chain_hash if row else None,
                    row.last_catchup_at if row else None)
        finally:
            s.close()


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


# --- M4 batch-2 patch 4, task 1: only a COMPLETE catch-up refreshes the
# --- offline grace; task 2: one trusted peer identity, fail-closed.

def _seed_pair(cloud, edge, tags):
    """Both engines observe the tagged texts; the edge's copies adopt the
    cloud's global_ids; the cloud then deletes every tagged scope (one
    log entry each)."""
    for tag in tags:
        for eng in (cloud, edge):
            eng.observe(user_id=USER, session_id=f"s-{tag}",
                        text=f"I like {tag} and coffee.", purpose="api")
            _align_belief(cloud, eng, tag)
    for tag in tags:
        b = next(b for b in _beliefs(cloud) if tag in b.claim_text)
        assert cloud.forget_belief(user_id=USER, belief_id=b.id,
                                   purpose="api")


def test_partial_catchup_never_earns_grace(tmp_path):
    """Task 1, case A: the edge has NEVER completed a catch-up.  Batch 1
    commits, the link dies before batch 2 -- the state is expired (a
    partial catch-up is not a complete one), every read entrance refuses,
    and the not-yet-synced sentinels surface nowhere.  A process restart
    keeps the committed cursor and the expired state; a healed link
    finishes the job and service resumes with a fresh completion
    instant."""
    root = tmp_path
    cloud_root = root / "cloud"
    cloud_root.mkdir()
    cloud = _edge(cloud_root, transport=None)  # a plain engine as the cloud
    tags = [f"g{i}" for i in range(5)]
    edge_root = root / "edge"
    edge_root.mkdir()
    # seed both sides FIRST (the edge without a transport), then the
    # cloud deletes all five scopes
    e0 = _edge(edge_root, transport=None)
    _seed_pair(cloud, e0, tags)
    e0._engine.dispose()

    # batch cap 2: 5 tombstones need 3 data batches + the empty tail
    broken = _Transport(cloud, fail_after=1, batch_limit=2)
    e1 = _edge(edge_root, transport=broken)
    assert e1.sync_status()["state"] == "expired"
    assert e1.sync_status()["last_catchup_at"] is None, \
        "no complete catch-up has ever happened"
    cursor, _chain, _last = _peer_state(e1, cloud._db_id)
    assert cursor == 2, "batch 1's progress is durable"
    for call in (lambda: e1.recall(user_id=USER, query="g3", purpose="api"),
                 lambda: e1.get_beliefs(user_id=USER, purpose="api"),
                 lambda: e1.export_page(user_id=USER, purpose="api")):
        with pytest.raises(SyncCatchupRequired):
            call()

    # the process restarts: the same link, the cursor resumes (batch 2),
    # still never completed -> still expired
    e1._engine.dispose()
    e2 = _edge(edge_root, transport=_Transport(cloud, fail_after=1,
                                               batch_limit=2))
    assert e2.sync_status()["state"] == "expired"
    cursor, _chain, _last = _peer_state(e2, cloud._db_id)
    assert cursor == 4, "the restart resumed from the committed cursor"

    # the link heals: the catch-up runs to the tail, service resumes and
    # the completion instant is stamped
    e2._engine.dispose()
    e3 = _edge(edge_root, transport=_Transport(cloud, batch_limit=2))
    assert e3.sync_status()["state"] == "normal"
    cursor, _chain, last = _peer_state(e3, cloud._db_id)
    assert last is not None
    assert cursor == 5, "the tail is reached"
    for tag in tags:
        assert all(tag not in b.claim_text for b in _beliefs(e3)), tag
    assert "g4" not in (e3.recall(user_id=USER, query="g4", purpose="api") or "")
    e3._engine.dispose()
    cloud._engine.dispose()


def test_partial_progress_does_not_extend_the_grace(tmp_path):
    """Task 1, case B: a COMPLETE catch-up earns the grace at T0; later
    partial progress (a batch applied, then the link dies) must leave the
    anchor at T0 -- the remaining grace keeps shrinking with real time
    and never restarts.  Healing completes the catch-up, resumes service
    and stamps a fresh instant."""
    from datetime import UTC, datetime

    root = tmp_path
    cloud_root = root / "cloud"
    cloud_root.mkdir()
    cloud = _edge(cloud_root, transport=None)
    edge_root = root / "edge"
    edge_root.mkdir()
    e = _edge(edge_root, transport=_Transport(cloud), grace_days=7,
              reminder_days=2)
    assert e.sync_status()["state"] == "normal"
    t0 = _peer_state(e, cloud._db_id)[2]
    assert t0 is not None

    # the cloud gains tombstones; the edge ages to 6.5 days of its grace
    _seed_pair(cloud, e, ["h1", "h2", "h3", "h4"])
    _age_last_catchup(e, days=6.5)
    # one batch applies, then the link dies
    e._sync_transport = _Transport(cloud, fail_after=1, batch_limit=1)
    e._catch_up()
    assert e.sync_status()["state"] == "offline_grace"
    st = e.sync_status()
    assert st["remaining_grace_days"] == pytest.approx(0.5, abs=0.1), \
        "the partial catch-up must NOT have restarted the 7-day grace"
    assert _peer_state(e, cloud._db_id)[0] == 1, "batch 1 still committed"
    last_after_partial = _peer_state(e, cloud._db_id)[2]
    assert last_after_partial is not None
    assert last_after_partial < datetime.utcnow() - timedelta(days=6), \
        "the partial catch-up did not refresh the completion instant"

    # aged past the grace, the same partial attempt leaves it expired
    _age_last_catchup(e, days=8.0)
    e._sync_transport = _Transport(cloud, fail_after=1, batch_limit=1)
    e._catch_up()
    assert e.sync_status()["state"] == "expired"
    with pytest.raises(SyncCatchupRequired):
        e.recall(user_id=USER, query="h4", purpose="api")

    # healed: the catch-up finishes, service resumes, the instant is fresh
    e._sync_transport = _Transport(cloud, batch_limit=1)
    e._catch_up()
    assert e.sync_status()["state"] == "normal"
    st = e.sync_status()
    assert st["remaining_grace_days"] == pytest.approx(7.0, abs=0.1)
    for tag in ("h1", "h4"):
        assert all(tag not in b.claim_text for b in _beliefs(e)), tag
    e._engine.dispose()
    cloud._engine.dispose()


def test_rescan_then_offline_keeps_the_old_grace(tmp_path):
    """Task 1, rescan coverage: the edge's stored cursor overshoots the
    cloud's rewound log -- the rescan re-anchors to 0 and commits, then
    the link dies BEFORE the tail.  The state is not normal, the
    completion instant is untouched, the committed re-anchor is not lost,
    and the next catch-up finishes the job."""
    root = tmp_path
    cloud_root = root / "cloud"
    cloud_root.mkdir()
    cloud = _edge(cloud_root, transport=None)
    edge_root = root / "edge"
    edge_root.mkdir()
    e = _edge(edge_root, transport=_Transport(cloud))
    assert e.sync_status()["state"] == "normal"
    t0 = _peer_state(e, cloud._db_id)[2]

    _seed_pair(cloud, e, ["r1", "r2", "r3", "r4", "r5"])
    _age_last_catchup(e, days=1.0)
    # simulate a restored cloud: the edge's stored cursor overshoots
    from mirror_memory.core.models import SyncPeer as _SP
    from mirror_memory.core.tenancy import fixture_write_scope

    with tenant_scope(*TENANT), fixture_write_scope("simulate restored peer"):
        s = sessionmaker(bind=create_engine(str(e._database_url)))()
        s.query(_SP).update({"cursor": 7}, synchronize_session=False)
        s.commit()
        s.close()

    # the rescan batch applies (all 5 re-delivered, deduped/applied) and
    # re-anchors to 0; the link then dies before the tail
    e._sync_transport = _Transport(cloud, fail_after=1, batch_limit=2)
    e._catch_up()
    cursor, _chain, last = _peer_state(e, cloud._db_id)
    assert cursor == 0, \
        "the rescan re-anchor is committed; the link died on the next export"
    assert e.sync_status()["state"] in ("offline_grace", "expired")
    assert last is not None \
        and last < datetime.utcnow() - timedelta(hours=12), \
        "the rescan must not count as a complete catch-up"

    # healed: the pull resumes from the committed cursor and completes
    e._sync_transport = _Transport(cloud, batch_limit=2)
    e._catch_up()
    assert e.sync_status()["state"] == "normal"
    cursor, _chain, last = _peer_state(e, cloud._db_id)
    assert cursor == 5
    for tag in ("r1", "r5"):
        assert all(tag not in b.claim_text for b in _beliefs(e)), tag
    e._engine.dispose()
    cloud._engine.dispose()


def test_transport_without_identity_refuses(tmp_path):
    """Task 2: a transport that cannot name its peer is fail-closed --
    _catch_up raises SyncTransportIdentityMissing, no channel row is
    written, and a proper transport afterwards works normally."""
    root = tmp_path
    cloud_root = root / "cloud"
    cloud_root.mkdir()
    cloud = _edge(cloud_root, transport=None)
    edge_root = root / "edge"
    edge_root.mkdir()
    e = _edge(edge_root, transport=None)  # starts without any transport
    e._sync_transport = _NoIdentityTransport(cloud)
    with pytest.raises(SyncTransportIdentityMissing) as info:
        e._catch_up()
    assert "db_id" in str(info.value)
    from mirror_memory.core.models import SyncPeer as _SP2

    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(str(e._database_url)))()
        assert s.query(_SP2).count() == 0, "no channel state was written"
        s.close()

    # a transport WITH an identity works and completes
    e._sync_transport = _Transport(cloud)
    e._catch_up()
    assert e.sync_status()["state"] == "normal"
    e._engine.dispose()
    cloud._engine.dispose()


def test_sender_mismatch_refuses(tmp_path):
    """Task 2: the transport declares one peer but the envelope claims
    another -- the batch is refused before any write and no channel row
    appears under either identity."""
    from mirror_memory.core import metrics
    from mirror_memory.core.models import SyncPeer as _SP3
    from mirror_memory.core.tombstones import TombstoneBatchRejected

    root = tmp_path
    cloud_root = root / "cloud"
    cloud_root.mkdir()
    cloud = _edge(cloud_root, transport=None)
    edge_root = root / "edge"
    edge_root.mkdir()
    e = _edge(edge_root, transport=None)
    before = metrics.value(metrics.TOMBSTONES_REJECTED_PREFIX + "sender")
    e._sync_transport = _LyingTransport(cloud, declared_db_id="f" * 32)
    with pytest.raises(TombstoneBatchRejected) as info:
        e._catch_up()
    assert info.value.reason == "sender"
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=create_engine(str(e._database_url)))()
        assert s.query(_SP3).count() == 0, \
            "no channel state under a surprise identity"
        s.close()
    assert metrics.value(metrics.TOMBSTONES_REJECTED_PREFIX + "sender") \
        > before

    # the truthful transport completes normally
    e._sync_transport = _Transport(cloud)
    e._catch_up()
    assert e.sync_status()["state"] == "normal"
    e._engine.dispose()
    cloud._engine.dispose()


def test_consecutive_catchups_and_repointed_transport(tmp_path):
    """Task 2: with one identity, a second catch-up neither rescans nor
    applies anything and still refreshes the completion instant;
    re-pointing the transport at a DIFFERENT peer opens its own channel
    row and leaves the first peer's row untouched."""
    from mirror_memory.core import metrics

    root = tmp_path
    c1_root = root / "cloud1"
    c1_root.mkdir()
    c1 = _edge(c1_root, transport=None)
    c2_root = root / "cloud2"
    c2_root.mkdir()
    c2 = _edge(c2_root, transport=None)
    edge_root = root / "edge"
    edge_root.mkdir()
    e = _edge(edge_root, transport=_Transport(c1))

    assert e.sync_status()["state"] == "normal"
    first_last = _peer_state(e, c1._db_id)[2]
    rescans = metrics.value(metrics.TOMBSTONES_RESCANS)

    # a second catch-up against the same peer: nothing to apply, no rescan
    e._catch_up()
    assert metrics.value(metrics.TOMBSTONES_RESCANS) == rescans
    cursor, _chain, last = _peer_state(e, c1._db_id)
    assert cursor == 0 and last is not None
    assert last >= first_last, "the completion instant refreshes"

    # re-point at a different peer: its OWN channel row, from cursor 0
    _seed_pair(c2, e, ["p1"])  # c2 gains a tombstone
    e._sync_transport = _Transport(c2)
    e._catch_up()
    assert e.sync_status()["state"] == "normal"
    c2_cursor, _chain2, _last2 = _peer_state(e, c2._db_id)
    assert c2_cursor == 1
    assert _peer_state(e, c1._db_id)[0] == 0, \
        "the first peer's channel row is untouched"
    assert all("p1" not in b.claim_text for b in _beliefs(e))
    e._engine.dispose()
    c1._engine.dispose()
    c2._engine.dispose()
