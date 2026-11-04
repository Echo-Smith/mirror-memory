"""M4 batch-2 patch 6, task 1: the sync gate through the REAL HTTP layer.

Every server reads_memory entrance in ENTRY_REGISTRY (recall / panel /
beliefs / belief_by_id / stats / export) gets an HTTP driver; a new
server read entrance without a driver fails the registry test.  The
scenario drives a restored server whose channel has NOT caught up:

- expired (anchor aged past the grace): every read route answers 503
  with the product-facing, distinguishable detail -- never a generic
  500, never content;
- catching up: every read route answers 503 with the catching-up detail;
- healed (a fresh app over the same database with a live transport):
  every route serves -- and the deleted sentinel appears nowhere;
  /belief/{id} of the deleted belief is a 404.

/belief/{id} and /stats are covered directly, including the unchanged
cross-tenant 404 and purpose 403 semantics.

Databases and restore scope (patch 7, task 3 -- stated exactly): the
SQLite matrix restores the MAIN DATABASE FILE only (the deletion-log
directory is NOT rewound -- a main-DB restore with the log retained).
The PostgreSQL run performs NO restore: it is the un-caught-up state on
a real PG server database (anchor aged past the grace), NOT a restore
test -- the full-machine restore scenarios live in
tests/test_sync_mesh.py and tests/test_edge_sync.py and are untouched.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.authorization import ENTRY_REGISTRY
from mirror_memory.core.deletion import FileDeletionLog
from mirror_memory.core.models import Base, Belief, SyncPeer
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.server import (
    _tenant_key_registry,
    _tenant_engines,
    create_app,
    register_tenant_key,
)

TENANT = ("tA", "app")
USER = "u-http"
SENTINEL = "SENHTTP"


@pytest.fixture(autouse=True)
def _clean_registry():
    _tenant_key_registry.clear()
    _tenant_engines.clear()
    yield
    _tenant_key_registry.clear()
    _tenant_engines.clear()


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


class _SwappingTransport:
    """The transport a server engine holds: `.dead` flips the link."""

    def __init__(self, peer_engine, dead: bool = False):
        self.peer = peer_engine
        self.dead = dead
        self.calls = 0

    @property
    def db_id(self):
        return self.peer._db_id

    def export_tombstones(self, cursor, limit, peer_db_id=None):
        self.calls += 1
        if self.dead:
            raise ConnectionError("the link is down")
        return self.peer.export_tombstones(cursor, limit)


def _server_config(log_dir):
    cfg = load_config("config/")
    cfg.sync.enabled = True
    cfg.sync.offline_grace_days = 7
    cfg.sync.reminder_days = 2
    cfg.deletion_log.directory = str(log_dir)
    return cfg


def _build_cloud(root):
    log_dir = root / "cloud-deletions"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config("config/")
    cfg.deletion_log.directory = str(log_dir)
    cloud = MemoryEngine(
        config=cfg, database_url=f"sqlite:///{root / 'cloud.db'}",
        tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
        deletion_log=FileDeletionLog(log_dir), single_tenant_mode=False)
    cloud._ensure_db()
    # the cloud holds the same two memories (its copies get the shared
    # global_ids in _align_cloud)
    with tenant_scope(*TENANT), fixture_write_scope("cloud seed memory"):
        s = sessionmaker(bind=cloud._engine)()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    cloud.observe(user_id=USER, session_id="sg1",
                  text=f"I like {SENTINEL} and coffee.", purpose="api")
    cloud.observe(user_id=USER, session_id="sg2",
                  text="I live in HTTPCity.", purpose="api")
    return cloud


def _seed_server_db(db_url, log_dir):
    """The server database's rows, seeded directly: two beliefs with
    chosen global_ids, memory enabled, watermark + genesis."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    sm.ensure_watermark(engine)
    FileDeletionLog(log_dir).ensure_genesis(sm.read_watermark(engine)[0])
    with tenant_scope(*TENANT), fixture_write_scope("http test seed"):
        s = sessionmaker(bind=engine)()
        set_memory_enabled(s, USER, True)
        s.add(Belief(user_id=USER, dimension="topic", key="k-sentinel",
                     global_id="1" * 32,
                     claim_text=f"I like {SENTINEL} and coffee.",
                     confidence=0.9))
        s.add(Belief(user_id=USER, dimension="topic", key="k-city",
                     global_id="2" * 32, claim_text="I live in HTTPCity.",
                     confidence=0.9))
        s.commit()
        s.close()
    engine.dispose()


def _align_cloud(cloud):
    """The cloud's copies of the two beliefs adopt the same global_ids --
    matched by CONTENT (the extractor mints its own keys)."""
    with tenant_scope(*TENANT), fixture_write_scope("align http cloud"):
        s = sessionmaker(bind=cloud._engine)()
        for b in s.query(Belief).filter(Belief.user_id == USER).all():
            if SENTINEL in b.claim_text:
                b.global_id = "1" * 32
            elif "HTTPCity" in b.claim_text:
                b.global_id = "2" * 32
        s.commit()
        s.close()


def _cloud_beliefs(cloud):
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=cloud._engine)()
        try:
            return list(s.query(Belief).filter(Belief.user_id == USER).all())
        finally:
            s.close()


def _http_headers(key="key-A"):
    return {"Authorization": f"Bearer {key}", "X-MM-Purpose": "test"}


def _http_drivers(client):
    """One HTTP driver per server reads_memory registry entry."""
    h = _http_headers()
    return {
        "recall": lambda: client.post(
            "/recall", json={"user_id": USER, "query": SENTINEL}, headers=h),
        "panel": lambda: client.get(
            "/panel", params={"user_id": USER}, headers=h),
        "beliefs": lambda: client.get(
            "/beliefs", params={"user_id": USER}, headers=h),
        # belief 2 SURVIVES the deletion (the sentinel belief 1 does not):
        # the driver checks the route's behavior for a live resource; the
        # deleted one's 404 is asserted separately in the healed phase
        "belief_by_id": lambda: client.get(
            "/belief/2", params={"user_id": USER}, headers=h),
        "stats": lambda: client.get(
            "/stats", params={"user_id": USER}, headers=h),
        "export": lambda: client.get(
            "/export", params={"user_id": USER}, headers=h),
    }


def _registry_server_reads():
    return sorted({
        key.rsplit(".", 1)[-1]
        for key, meta in ENTRY_REGISTRY.items()
        if key.startswith("mirror_memory.server.") and meta.get("reads_memory")
    })


def test_server_registry_covers_every_read():
    """Every server reads_memory registry entry has an HTTP driver here --
    a new server read entrance without one fails this test."""
    missing = [n for n in _registry_server_reads()
               if n not in _http_drivers(None)]
    assert missing == [], f"HTTP drivers missing for: {missing}"


def _age_server_anchor(database_url, days: float, tenant=None):
    with tenant_scope(*(tenant or TENANT)), \
            fixture_write_scope("age the server anchor"):
        s = sessionmaker(bind=create_engine(database_url))()
        s.query(SyncPeer).update(
            {"last_catchup_at": datetime.utcnow() - timedelta(days=days)},
            synchronize_session=False)
        s.commit()
        s.close()


def _run_http_scenario(tmp_path, database_url, *, pg_cleanup=None):
    log_dir = tmp_path / "server-deletions"
    log_dir.mkdir(parents=True, exist_ok=True)
    _seed_server_db(database_url, log_dir)
    cloud = _build_cloud(tmp_path)
    _align_cloud(cloud)
    register_tenant_key("key-A", TENANT[0], TENANT[1], purposes={"test"})

    cfg = _server_config(log_dir)
    captured: list = []

    def factory(eng):
        # the HOST decides which tenants have a channel: only tA syncs --
        # a tB wrapper gets no transport (its catch-up must not pull
        # another tenant's tombstones)
        if eng._tenant != TENANT:
            return None
        captured.append(eng)
        return _SwappingTransport(cloud)

    # --- healthy phase: every route serves, the sentinel is visible
    app = create_app(config=cfg, database_url=database_url,
                     llm_client=_FakeLLM(), sync_transport_factory=factory)
    with TestClient(app) as client:
        drivers = _http_drivers(client)
        for name, drv in sorted(drivers.items()):
            r = drv()
            assert r.status_code == 200, (name, r.status_code, r.text[:200])
        assert SENTINEL in drivers["beliefs"]().text
    if pg_cleanup:
        pass  # the shared pool is disposed by the lifespan

    _tenant_engines.clear()

    # the pre-deletion backup (SQLite: DB file + log dir together)
    backup_db = None
    if database_url.startswith("sqlite"):
        db_file = database_url.replace("sqlite:///", "")
        backup_db = db_file + "-bak"
        src = create_engine(database_url).raw_connection()
        from sqlalchemy import create_engine as _ce

        dest = _ce(f"sqlite:///{backup_db}").raw_connection()
        src.connection.backup(dest.connection)
        dest.close()
        src.close()

    # the cloud deletes the sentinel (its own local id; the tombstone
    # travels by the shared global_id)
    cloud_belief = next(b for b in _cloud_beliefs(cloud)
                        if SENTINEL in b.claim_text)
    cloud.forget_belief(user_id=USER, belief_id=cloud_belief.id,
                        purpose="api")

    if backup_db is not None:
        shutil.copyfile(backup_db, db_file)

    # --- expired phase: the anchor aged past the grace, the link dead.
    # The detail must be the EXACT expired sentence (patch 7, task 3 --
    # the two refusal texts share no common substring test anymore).
    EXPIRED_DETAIL = "此设备已超过允许的离线时间，需要联网同步后才能继续查看记忆。"
    CATCHING_UP_DETAIL = "记忆正在同步，请稍后重试。"

    def dead_factory(eng):
        if eng._tenant != TENANT:
            return None
        captured.append(eng)
        t = _SwappingTransport(cloud)
        t.dead = True
        return t

    _tenant_engines.clear()
    _age_server_anchor(database_url, days=8.0)
    app2 = create_app(config=cfg, database_url=database_url,
                      llm_client=_FakeLLM(), sync_transport_factory=dead_factory)
    with TestClient(app2) as client:
        drivers = _http_drivers(client)
        for name, drv in sorted(drivers.items()):
            r = drv()
            assert r.status_code == 503, (name, r.status_code, r.text[:200])
            assert r.json()["detail"] == EXPIRED_DETAIL, (name, r.text[:200])
            assert SENTINEL not in r.text, name

    # --- catching-up phase: the state machine of the engine THAT SERVES
    # this app's requests (a warmup request materializes the wrapper,
    # then the state flips; the previous version mutated a stale engine
    # and verified the expired text by a loose shared substring)
    _tenant_engines.clear()
    app3 = create_app(config=cfg, database_url=database_url,
                      llm_client=_FakeLLM(), sync_transport_factory=dead_factory)
    with TestClient(app3) as client:
        warm = client.get("/beliefs", params={"user_id": USER},
                          headers=_http_headers())
        assert warm.status_code == 503  # still expired
        eng = captured[-1]
        assert eng._tenant == TENANT
        eng._sync_state = "catching_up"
        drivers = _http_drivers(client)
        for name, drv in sorted(drivers.items()):
            r = drv()
            assert r.status_code == 503, (name, r.status_code, r.text[:200])
            assert r.json()["detail"] == CATCHING_UP_DETAIL, \
                (name, r.text[:200])
            assert SENTINEL not in r.text, name
        eng._sync_state = "normal"

    # --- healed phase: a fresh app over the same database, live link;
    # the startup catch-up completes, the sentinel is gone everywhere
    _tenant_engines.clear()
    app4 = create_app(config=cfg, database_url=database_url,
                      llm_client=_FakeLLM(), sync_transport_factory=factory)
    with TestClient(app4) as client:
        drivers = _http_drivers(client)
        for name, drv in sorted(drivers.items()):
            r = drv()
            assert r.status_code == 200, (name, r.status_code, r.text[:200])
            assert SENTINEL not in r.text, name
        # the deleted belief's own route is a 404 now
        r = client.get("/belief/1", params={"user_id": USER},
                       headers=_http_headers())
        assert r.status_code == 404, r.text[:200]

    # --- cross-tenant and purpose semantics unchanged (healthy server)
    register_tenant_key("key-B", "tB", "app", purposes={"test"})
    # tenant B has its OWN completed channel history (a different peer):
    # per the no-transport rule a tenant with no determinable history
    # refuses reads (expired) -- so B gets a history to serve under
    with tenant_scope(*("tB", "app")), fixture_write_scope("tB channel history"):
        s = sessionmaker(bind=create_engine(database_url))()
        s.add(SyncPeer(peer_db_id="b" * 32, cursor=3,
                       last_catchup_at=datetime.utcnow()))
        s.commit()
        s.close()
    with TestClient(app4) as client:
        h_b = _http_headers("key-B")
        # A's belief id against B's key -> 404, no existence leak (the
        # SURVIVING belief 2: the deleted one is a 404 for everyone)
        own = client.get("/belief/2", params={"user_id": USER},
                         headers=_http_headers())
        assert own.status_code == 200
        assert client.get("/belief/2", params={"user_id": USER},
                          headers=h_b).status_code == 404
        # a purpose outside the key's registered set is still 403 pre-SQL
        assert client.get(
            "/belief/2", params={"user_id": USER},
            headers={"Authorization": "Bearer key-A",
                     "X-MM-Purpose": "other"}).status_code == 403
        # /stats behaves the same way
        assert client.get("/stats", params={"user_id": USER},
                          headers=_http_headers()).status_code == 200
        assert client.get("/stats", params={"user_id": USER},
                          headers={"Authorization": "Bearer key-A",
                                   "X-MM-Purpose": "other"}).status_code == 403
    cloud._engine.dispose()
    if pg_cleanup:
        pg_cleanup()


def test_http_gate_matrix_sqlite(tmp_path):
    _run_http_scenario(tmp_path, f"sqlite:///{tmp_path / 'server.db'}")


def _pg_server_database(tmp_path):
    """A REAL PostgreSQL server database, provisioned owner-side; the app
    gets the app-role runtime URL."""
    import os

    if not os.environ.get("MM_DATABASE_URL", "").startswith("postgresql"):
        pytest.skip("needs MM_DATABASE_URL pointing at PostgreSQL")
    from tests.conftest import PG_TEST_ROLE_PASSWORDS
    from tests.test_deletion_race_backup import _pg_own_database

    runtime_url, admin_url = _pg_own_database("mm_test_http_gate")
    return runtime_url, admin_url


def test_http_gate_matrix_real_pg(tmp_path):
    """The same scenario over a real PostgreSQL server database.  On this
    shape the per-tenant wrappers are the schema_ready shared-engine
    form, so the healed phase's startup catch-up runs through them."""
    runtime_url, admin_url = _pg_server_database(tmp_path)
    try:
        _run_http_scenario(tmp_path, runtime_url)
    finally:
        from tests.test_deletion_race_backup import _pg_drop_database

        _pg_drop_database("mm_test_http_gate", admin_url)


# --- M4 batch-2 patch 7, task 1: the default and legacy MM_API_KEY
# --- service modes get the factory transport on the lifespan engine

def _seed_default_db(db_url, log_dir):
    """The server rows for the DEFAULT tenant (the default / legacy
    single-key modes serve this tenant through the lifespan engine)."""
    from mirror_memory.core.models import Base as _Base

    dflt = ("default", "default")
    engine = create_engine(db_url)
    _Base.metadata.create_all(engine)
    sm.ensure_watermark(engine)
    FileDeletionLog(log_dir).ensure_genesis(sm.read_watermark(engine)[0])
    with tenant_scope(*dflt), fixture_write_scope("default-mode seed"):
        s = sessionmaker(bind=engine)()
        set_memory_enabled(s, USER, True)
        s.add(Belief(user_id=USER, dimension="topic", key="k-sentinel",
                     global_id="1" * 32,
                     claim_text=f"I like {SENTINEL} and coffee.",
                     confidence=0.9))
        s.commit()
        s.close()
    engine.dispose()


def _build_default_cloud(root):
    log_dir = root / "cloud-deletions"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config("config/")
    cfg.deletion_log.directory = str(log_dir)
    cloud = MemoryEngine(
        config=cfg, database_url=f"sqlite:///{root / 'cloud.db'}",
        tenant_id="default", app_id="default", llm_client=_FakeLLM(),
        deletion_log=FileDeletionLog(log_dir), single_tenant_mode=False)
    cloud._ensure_db()
    with tenant_scope(*("default", "default")), \
            fixture_write_scope("default-mode cloud seed"):
        s = sessionmaker(bind=cloud._engine)()
        set_memory_enabled(s, USER, True)
        s.add(Belief(user_id=USER, dimension="topic", key="k-sentinel",
                     global_id="1" * 32,
                     claim_text=f"I like {SENTINEL} and coffee.",
                     confidence=0.9))
        s.commit()
        s.close()
    return cloud


def _default_mode_scenario(tmp_path, *, legacy: bool, monkeypatch):
    """No tenant registry (or legacy MM_API_KEY): _get_engine(None)
    serves through the LIFESPAN engine, which carries the factory's
    transport -- the startup catch-up applies the cloud's deletion
    before the first read."""
    root = tmp_path
    log_dir = root / "server-deletions"
    log_dir.mkdir(parents=True, exist_ok=True)
    db_url = f"sqlite:///{root / 'server.db'}"
    _seed_default_db(db_url, log_dir)
    cloud = _build_default_cloud(root)
    register_tenant_key("key-A", "tA", "app", purposes={"test"}) \
        if False else None
    if legacy:
        monkeypatch.setenv("MM_API_KEY", "legacy-key")
    headers = ({"Authorization": "Bearer legacy-key", "X-MM-Purpose": "api"}
               if legacy else {})

    cfg = _server_config(log_dir)
    captured: list = []

    def factory(eng):
        captured.append(eng)
        return _SwappingTransport(cloud)

    # the cloud deletes the sentinel BEFORE the server starts
    with tenant_scope(*("default", "default")):
        s = sessionmaker(bind=cloud._engine)()
        cbs = list(s.query(Belief).filter(Belief.user_id == USER).all())
        s.close()
    cb = next(b for b in cbs if SENTINEL in b.claim_text)
    assert cloud.forget_belief(user_id=USER, belief_id=cb.id, purpose="api")

    app = create_app(config=cfg, database_url=db_url, llm_client=_FakeLLM(),
                     sync_transport_factory=factory)
    with TestClient(app) as client:
        # the first read: the startup catch-up has ALREADY applied the
        # deletion -- the sentinel is gone, and the route serves
        r = client.get("/beliefs", params={"user_id": USER}, headers=headers)
        assert r.status_code == 200, r.text[:200]
        assert SENTINEL not in r.text
        assert captured and captured[0]._tenant == ("default", "default"), \
            "the lifespan engine carries the factory transport"
        eng = captured[0]
        from mirror_memory.core.repository import read_sync_peer

        with tenant_scope(*("default", "default")):
            s = sessionmaker(bind=create_engine(db_url))()
            row = read_sync_peer(s, cloud._db_id)
            s.close()
        assert row is not None and row.cursor >= 1, \
            "the startup catch-up consumed the peer's tombstone"

    # offline: a dead link keeps the stored grace; aged past it, refuse
    _age_server_anchor(db_url, days=8.0, tenant=("default", "default"))
    app2 = create_app(
        config=cfg, database_url=db_url, llm_client=_FakeLLM(),
        sync_transport_factory=lambda eng: _SwappingTransport(cloud, dead=True))
    with TestClient(app2) as client:
        r = client.get("/beliefs", params={"user_id": USER}, headers=headers)
        assert r.status_code == 503
        assert r.json()["detail"] == \
            "此设备已超过允许的离线时间，需要联网同步后才能继续查看记忆。"
    cloud._engine.dispose()


def test_http_default_mode_catches_up_before_first_read(tmp_path, monkeypatch):
    _default_mode_scenario(tmp_path, legacy=False, monkeypatch=monkeypatch)


def test_http_legacy_api_key_mode_catches_up(tmp_path, monkeypatch):
    _default_mode_scenario(tmp_path, legacy=True, monkeypatch=monkeypatch)


def test_http_sync_off_unchanged(tmp_path):
    """sync.enabled=False: no gate, no transport -- the original behavior
    (reads serve whatever the database holds)."""
    root = tmp_path
    log_dir = root / "server-deletions"
    log_dir.mkdir(parents=True, exist_ok=True)
    db_url = f"sqlite:///{root / 'server.db'}"
    _seed_default_db(db_url, log_dir)
    cfg = _server_config(log_dir)
    cfg.sync.enabled = False
    app = create_app(config=cfg, database_url=db_url, llm_client=_FakeLLM())
    with TestClient(app) as client:
        r = client.get("/beliefs", params={"user_id": USER})
        assert r.status_code == 200
        assert SENTINEL in r.text  # sync is off: nothing gates the read
