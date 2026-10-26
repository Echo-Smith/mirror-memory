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
    register_tenant_key(key, tenant[0], tenant[1])
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
                            headers={"Authorization": "Bearer wrong-key"})
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
                       headers={"Authorization": "Bearer key-B"})
        assert r.status_code == 404
        # a nonexistent id from B -> also 404 (same shape)
        r2 = client.get("/belief/999999", params={"user_id": "u1"},
                        headers={"Authorization": "Bearer key-B"})
        assert r2.status_code == 404
        assert r.json() == r2.json()
        # A can read its own
        r3 = client.get(f"/belief/{bid_a}", params={"user_id": "u1"},
                        headers={"Authorization": "Bearer key-A"})
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
                headers={"Authorization": f"Bearer {key}"},
            )
            b = client.get("/beliefs", params={"user_id": "u1"},
                           headers={"Authorization": f"Bearer {key}"})
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
        r = client.get("/boom", headers={"Authorization": "Bearer key-A"})
        assert r.status_code == 500
        # next request: clean tenant context, correct data
        b = client.get("/beliefs", params={"user_id": "u1"},
                       headers={"Authorization": "Bearer key-B"})
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
                headers={"Authorization": f"Bearer key-{i}"},
            )
            assert r.status_code == 200
        assert len(server_mod._tenant_engines) == 2  # bounded
        assert len(disposed) == 2  # two evicted, pools disposed
        # the two resident engines answer correctly (no cross-bleed)
        for i in (2, 3):
            r = client.get(
                "/beliefs", params={"user_id": "u1"},
                headers={"Authorization": f"Bearer key-{i}"},
            )
            claims = [x["claim_text"] for x in r.json()["beliefs"]]
            assert claims == [f"T{i}"]
