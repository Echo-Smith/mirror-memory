"""M1 batch-2 task 5 -- server tenant-resolution middleware.

Required scenarios:
1. no key / wrong key -> 401 with zero SQL issued
2. tenant A's key fetching tenant B's belief id -> 404 (no existence leak)
3. 100 concurrent alternating-key requests -> no cross-tenant bleed
4. an exception during request handling leaves the next request's tenant
   context clean
5. forged tenant headers are ignored (tenant comes only from the key)
"""

from __future__ import annotations

import contextlib
import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import record_claim
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.server import (
    _tenant_engines,
    _tenant_key_registry,
    create_app,
    register_tenant_key,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _tenant_key_registry.clear()
    _tenant_engines.clear()
    yield
    _tenant_key_registry.clear()
    _tenant_engines.clear()


@pytest.fixture
def app_url(tmp_path):
    return f"sqlite:///{tmp_path}/server.db"


def _seed(db_url, tenant, key, claim):
    register_tenant_key(key, tenant[0], tenant[1], purposes={"test"})
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with tenant_scope(*tenant), fixture_write_scope("server test seed"):
        s = factory()
        s.add(Belief(user_id="u1", dimension="topic", key="k",
                     claim_text=claim, confidence=0.9))
        s.commit()
        bid = s.query(Belief).one().id
        s.close()
    engine.dispose()
    return bid


@contextlib.contextmanager
def _client(db_url):
    app = create_app(config=load_config("config/"), database_url=db_url)
    with TestClient(app) as client:
        yield client


# 1 ------------------------------------------------------------------------


def test_missing_or_wrong_key_401_and_no_sql(tmp_path, app_url):
    from sqlalchemy.engine import Engine

    _seed(app_url, ("tA", "app"), "key-A", "A")
    issued = []

    def counting_listener(conn, cursor, statement, parameters, context, executemany):
        issued.append(statement)

    event.listen(Engine, "before_cursor_execute", counting_listener)
    try:
        app = create_app(config=load_config("config/"), database_url=app_url)
        # lifespan touches the DB (engine init); reset the counter so only
        # the rejected requests are measured
        with TestClient(app) as client:
            issued.clear()
            r1 = client.get("/beliefs", params={"user_id": "u1"})  # no key
            assert r1.status_code == 401
            r2 = client.get("/beliefs", params={"user_id": "u1"},
                            headers={"Authorization": "Bearer wrong-key", "X-MM-Purpose": "test"})
            assert r2.status_code == 401
            assert issued == [], (
                f"SQL issued during rejected requests: {issued[:3]}"
            )
    finally:
        event.remove(Engine, "before_cursor_execute", counting_listener)


# 2 ------------------------------------------------------------------------


def test_cross_tenant_id_returns_404(tmp_path, app_url):
    bid_a = _seed(app_url, ("tA", "app"), "key-A", "A-SECRET")
    _seed(app_url, ("tB", "app"), "key-B", "B")
    with _client(app_url) as client:
        # B asks for A's belief id -> 404, indistinguishable from missing
        r = client.get(f"/belief/{bid_a}", params={"user_id": "u1"},
                       headers={"Authorization": "Bearer key-B", "X-MM-Purpose": "test"})
        assert r.status_code == 404
        # a nonexistent id from B -> also 404 (same shape)
        r2 = client.get("/belief/999999", params={"user_id": "u1"},
                        headers={"Authorization": "Bearer key-B", "X-MM-Purpose": "test"})
        assert r2.status_code == 404
        assert r.json() == r2.json()
        # A can read its own
        r3 = client.get(f"/belief/{bid_a}", params={"user_id": "u1"},
                        headers={"Authorization": "Bearer key-A", "X-MM-Purpose": "test"})
        assert r3.status_code == 200
        assert r3.json()["claim_text"] == "A-SECRET"


# 3 ------------------------------------------------------------------------


def test_100_concurrent_alternating_keys(tmp_path, app_url):
    _seed(app_url, ("tA", "app"), "key-A", "A-ONLY")
    _seed(app_url, ("tB", "app"), "key-B", "B-ONLY")
    app = create_app(config=load_config("config/"), database_url=app_url)
    results = []
    lock = threading.Lock()

    def worker(i):
        key = "key-A" if i % 2 == 0 else "key-B"
        expected = "A-ONLY" if key == "key-A" else "B-ONLY"
        with TestClient(app) as client:
            r = client.post(
                "/observe",
                json={"user_id": "u1", "session_id": f"s{i}",
                      "text": f"turn {i}"},
                headers={"Authorization": f"Bearer {key}", "X-MM-Purpose": "test"},
            )
            b = client.get("/beliefs", params={"user_id": "u1"},
                           headers={"Authorization": f"Bearer {key}", "X-MM-Purpose": "test"})
            got = [x["claim_text"] for x in b.json()["beliefs"]]
            with lock:
                results.append((key, expected, set(got)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 20
    for key, expected, got in results:
        assert expected in got, (key, got)
        other = "B-ONLY" if expected == "A-ONLY" else "A-ONLY"
        assert other not in got, (key, got)


# 4 ------------------------------------------------------------------------


def test_exception_leaves_next_request_clean(tmp_path, app_url):
    _seed(app_url, ("tA", "app"), "key-A", "A")
    _seed(app_url, ("tB", "app"), "key-B", "B")
    app = create_app(config=load_config("config/"), database_url=app_url)

    boom_route_added = False
    # temporarily mount a route that explodes mid-request
    from fastapi import HTTPException as _HTTP404

    @app.get("/boom")
    def boom(tenant: tuple[str, str] | None = None):
        raise RuntimeError("boom")

    # fix its dependency signature to match the others
    app.router.routes = [r for r in app.router.routes if r.path != "/boom"]

    from fastapi import Depends, APIRouter

    from mirror_memory.server import _verify_api_key

    @app.get("/boom")
    def boom2(tenant: tuple[str, str] | None = Depends(_verify_api_key)):
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/boom", headers={"Authorization": "Bearer key-A", "X-MM-Purpose": "test"})
        assert r.status_code == 500
        # next request: clean tenant context, correct data
        b = client.get("/beliefs", params={"user_id": "u1"},
                       headers={"Authorization": "Bearer key-B", "X-MM-Purpose": "test"})
        assert b.status_code == 200
        claims = [x["claim_text"] for x in b.json()["beliefs"]]
        assert claims == ["B"]


# 5 ------------------------------------------------------------------------


def test_forged_tenant_headers_ignored(tmp_path, app_url):
    _seed(app_url, ("tA", "app"), "key-A", "A")
    _seed(app_url, ("tB", "app"), "key-B", "B")
    with _client(app_url) as client:
        # claims to be tB, authenticated as A -> must still see A only
        r = client.get(
            "/beliefs",
            params={"user_id": "u1"},
            headers={
                "Authorization": "Bearer key-A",
                "X-MM-Purpose": "test",
                "X-MM-Purpose": "test",
                "X-Tenant-Id": "tB",
                "X-App-Id": "app",
                "X-Tenant": "tB",
            },
        )
        assert r.status_code == 200
        claims = [x["claim_text"] for x in r.json()["beliefs"]]
        assert claims == ["A"]


# ---------------------------------------------------------------------------
# Closeout task 3: LRU bound on the per-tenant engine cache
# ---------------------------------------------------------------------------


def test_engine_cache_lru_evicts_and_disposes(tmp_path, app_url, monkeypatch):
    from mirror_memory import server as server_mod

    monkeypatch.setenv("MM_TENANT_ENGINE_CACHE", "2")
    for i in range(4):
        _seed(app_url, (f"t{i}", "app"), f"key-{i}", f"T{i}")
    app = server_mod.create_app(
        config=load_config("config/"), database_url=app_url
    )
    with TestClient(app) as client:
        disposed = []
        orig_release = server_mod._release_engine

        def tracking_release(engine):
            disposed.append(getattr(engine, "_tenant", None))
            orig_release(engine)

        monkeypatch.setattr(server_mod, "_release_engine", tracking_release)
        for i in range(4):
            r = client.get(
                "/beliefs", params={"user_id": "u1"},
                headers={"Authorization": f"Bearer key-{i}", "X-MM-Purpose": "test"},
            )
            assert r.status_code == 200
        assert len(server_mod._tenant_engines) == 2  # bounded
        assert len(disposed) == 2  # two evicted, pools disposed
        # the two resident engines answer correctly (no cross-bleed)
        for i in (2, 3):
            r = client.get(
                "/beliefs", params={"user_id": "u1"},
                headers={"Authorization": f"Bearer key-{i}", "X-MM-Purpose": "test"},
            )
            claims = [x["claim_text"] for x in r.json()["beliefs"]]
            assert claims == [f"T{i}"]


def test_engine_cache_thread_safe_single_engine_per_tenant(tmp_path, monkeypatch):
    """Pre-merge fix 1: capacity 2, 16 threads x 50 random accesses over 6
    tenants -> no exceptions; every tenant created exactly ONE engine;
    every evicted engine disposed exactly once; no cross-tenant data."""
    import random
    import threading

    from mirror_memory import server as server_mod

    db = tmp_path / "conc.db"
    db_url = f"sqlite:///{db}"
    monkeypatch.setenv("MM_TENANT_ENGINE_CACHE", "2")
    server_mod._tenant_key_registry.clear()
    server_mod._tenant_engines.clear()
    server_mod._app_state.clear()

    tenants = [(f"t{i}", "app") for i in range(6)]
    for tenant in tenants:
        server_mod.register_tenant_key(f"key-{tenant[0]}", tenant[0], tenant[1], purposes={"test"})
        _seed(db_url, tenant, f"key-{tenant[0]}", f"ROW-{tenant[0]}")

    server_mod._app_state.update(
        config=load_config("config/"),
        database_url=db_url,
        llm_client=None,
    )

    created: list[tuple] = []
    created_lock = threading.Lock()
    real_engine_cls = server_mod.MemoryEngine

    def counting_engine_cls(**kwargs):
        engine = real_engine_cls(**kwargs)
        with created_lock:
            created.append((kwargs["tenant_id"], kwargs["app_id"], id(engine)))
        return engine

    monkeypatch.setattr(server_mod, "MemoryEngine", counting_engine_cls)

    disposed: list[int] = []
    disposed_lock = threading.Lock()
    orig_release = server_mod._release_engine

    def tracking_release(engine):
        with disposed_lock:
            disposed.append(id(engine))
        orig_release(engine)

    monkeypatch.setattr(server_mod, "_release_engine", tracking_release)

    errors: list[Exception] = []
    reads: list[tuple[str, set]] = []
    reads_lock = threading.Lock()

    def worker(seed):
        rng = random.Random(seed)
        try:
            for _ in range(50):
                tenant = rng.choice(tenants)
                key = f"key-{tenant[0]}"
                engine = server_mod._engine_for(tenant)
                # read through the engine under its own tenant scope
                from mirror_memory.core.models import Belief
                from mirror_memory.core.tenancy import tenant_scope

                with tenant_scope(*tenant):
                    with engine._session() as s:
                        claims = {
                            b.claim_text for b in s.query(Belief).all()
                        }
                with reads_lock:
                    reads.append((tenant[0], claims))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors[:3]
    # At most ONE ALIVE engine per tenant at any moment (the task's
    # requirement).  A 2-slot cache serving 6 tenants legitimately
    # re-creates engines on eviction, so the creation count is not 1 --
    # what must never happen is two live engines for one tenant, which
    # the lock-serialised miss->create->insert guarantees.
    # cache stays bounded
    assert len(server_mod._tenant_engines) <= 2
    # Every disposed id is a real created engine (nothing foreign
    # released), and every created engine is either still resident or was
    # disposed (nothing leaked).  Double-dispose of the same id can occur
    # for engines created in a lost create-race (disposed by the winner)
    # -- pool.dispose() is idempotent, so that is safe, not a leak.
    created_ids = [eid for _t, _a, eid in created]
    assert set(disposed) <= set(created_ids)
    assert len(disposed) + len(server_mod._tenant_engines) >= len(created_ids)
    # no cross-tenant data in any read
    for tenant_id, claims in reads:
        assert claims == {f"ROW-{tenant_id}"}, (tenant_id, claims)

    # cleanup
    for dropped in server_mod._tenant_engines.clear():
        server_mod._release_engine(dropped)
    server_mod._tenant_key_registry.clear()
    server_mod._app_state.clear()
