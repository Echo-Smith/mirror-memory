"""M2 batch-2 task 2: connection pool, timeouts, the shared PostgreSQL
engine and lock-free tenant-engine creation.

PostgreSQL-only tests take ``pg_app_url`` (skipped on SQLite); the cache
and wrapper tests are dialect-independent.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.pg_engine import (
    POOL_DEFAULTS,
    TIMEOUT_DEFAULTS_MS,
    EngineConfigError,
    build_engine,
    connect_options,
)
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope


# ---------------------------------------------------------------------------
# Settings surface
# ---------------------------------------------------------------------------


def test_connect_options_render_the_three_timeouts_as_integers():
    assert connect_options(TIMEOUT_DEFAULTS_MS) == (
        "-c statement_timeout=30000 -c lock_timeout=5000 "
        "-c idle_in_transaction_session_timeout=60000"
    )


def test_sqlite_engines_are_untouched(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path}/plain.db")
    assert engine.dialect.name == "sqlite"
    assert engine.pool.__class__.__name__ in ("QueuePool", "NullPool", "SingletonThreadPool")
    engine.dispose()


def test_pg_engine_applies_pool_and_timeout_settings(pg_app_url, monkeypatch):
    monkeypatch.setenv("MM_PG_POOL_SIZE", "3")
    monkeypatch.setenv("MM_PG_MAX_OVERFLOW", "2")
    monkeypatch.setenv("MM_PG_POOL_PRE_PING", "0")
    monkeypatch.setenv("MM_PG_POOL_RECYCLE", "600")
    monkeypatch.setenv("MM_PG_STATEMENT_TIMEOUT_MS", "1234")
    monkeypatch.setenv("MM_PG_LOCK_TIMEOUT_MS", "567")
    monkeypatch.setenv("MM_PG_IDLE_IN_TRANSACTION_TIMEOUT_MS", "8900")
    engine = build_engine(pg_app_url)
    try:
        assert engine.pool.size() == 3
        assert engine.pool._max_overflow == 2
        assert engine.pool._pre_ping is False
        assert engine.pool._recycle == 600
        with engine.connect() as conn:
            settings = dict(
                conn.execute(text(
                    "SELECT name, setting FROM pg_catalog.pg_settings WHERE name IN "
                    "('statement_timeout', 'lock_timeout', "
                    "'idle_in_transaction_session_timeout', 'application_name')"
                )).all()
            )
        assert settings == {
            "statement_timeout": "1234",
            "lock_timeout": "567",
            "idle_in_transaction_session_timeout": "8900",
            "application_name": "mirror-memory",
        }
    finally:
        engine.dispose()


def test_pg_engine_defaults(pg_app_url):
    engine = build_engine(pg_app_url)
    try:
        assert engine.pool.size() == POOL_DEFAULTS["pool_size"]
        assert engine.pool._max_overflow == POOL_DEFAULTS["max_overflow"]
        assert engine.pool._pre_ping is True
        assert engine.pool._recycle == POOL_DEFAULTS["pool_recycle"]
        with engine.connect() as conn:
            settings = dict(
                conn.execute(text(
                    "SELECT name, setting FROM pg_catalog.pg_settings WHERE name IN "
                    "('statement_timeout', 'lock_timeout', "
                    "'idle_in_transaction_session_timeout')"
                )).all()
            )
        assert settings == {k: str(v) for k, v in TIMEOUT_DEFAULTS_MS.items()}
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "name, value",
    [
        ("MM_PG_POOL_SIZE", "zero"),
        ("MM_PG_POOL_SIZE", "0"),
        ("MM_PG_MAX_OVERFLOW", "-1"),
        ("MM_PG_STATEMENT_TIMEOUT_MS", "30s"),
        ("MM_PG_LOCK_TIMEOUT_MS", "-5"),
        ("MM_PG_POOL_PRE_PING", "maybe"),
    ],
)
def test_malformed_engine_settings_are_refused(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(EngineConfigError, match=name):
        build_engine("postgresql+psycopg://x:y@localhost/db")


# ---------------------------------------------------------------------------
# statement_timeout: cancelled AND rolled back cleanly
# ---------------------------------------------------------------------------


def test_statement_timeout_cancels_the_statement_and_rolls_back(pg_app_url, monkeypatch):
    monkeypatch.setenv("MM_PG_STATEMENT_TIMEOUT_MS", "200")
    engine = build_engine(pg_app_url)
    try:
        with tenant_scope("tA", "app"):
            session = Session(engine)
            with fixture_write_scope("pool test: row that must roll back"):
                session.add(Belief(user_id="u", dimension="topic", key="k",
                                   claim_text="pending", confidence=0.5))
            session.flush()
            with pytest.raises(OperationalError, match="statement timeout"):
                session.execute(text("select pg_sleep(1)"))
            session.rollback()
            # the flushed row went with the transaction ...
            assert session.query(Belief).count() == 0
            # ... and the connection is healthy and still configured
            assert session.execute(text("select 1")).scalar() == 1
            assert session.execute(text(
                "SELECT setting FROM pg_catalog.pg_settings WHERE name = 'statement_timeout'"
            )).scalar() == "200"
            session.close()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Shared engine: 50 tenants, connections bounded by pool_size + max_overflow
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_server_state():
    from mirror_memory import server as server_mod

    server_mod._tenant_key_registry.clear()
    for dropped in server_mod._tenant_engines.clear():
        server_mod._release_engine(dropped)
    server_mod._app_state.clear()
    yield
    server_mod._tenant_key_registry.clear()
    for dropped in server_mod._tenant_engines.clear():
        server_mod._release_engine(dropped)
    server_mod._app_state.clear()


def test_fifty_tenants_share_one_pool_within_its_bound(
    pg_admin_engine, pg_app_url, monkeypatch, _clean_server_state
):
    from mirror_memory import server as server_mod

    pool_size, max_overflow = 3, 2
    monkeypatch.setenv("MM_PG_POOL_SIZE", str(pool_size))
    monkeypatch.setenv("MM_PG_MAX_OVERFLOW", str(max_overflow))
    monkeypatch.setenv("MM_TENANT_ENGINE_CACHE", "16")  # 50 tenants -> evictions

    tenants = [(f"t{i:02d}", "app") for i in range(50)]
    for tenant in tenants:
        server_mod.register_tenant_key(f"key-{tenant[0]}", *tenant, purposes={"test"})
    app = server_mod.create_app(config=load_config("config/"), database_url=pg_app_url)

    samples: list[int] = []
    stop = threading.Event()

    def monitor():
        with pg_admin_engine.connect() as conn:
            while not stop.is_set():
                samples.append(conn.execute(text(
                    "SELECT count(*) FROM pg_catalog.pg_stat_activity "
                    "WHERE application_name = 'mirror-memory' "
                    "AND datname = current_database()"
                )).scalar())
                conn.rollback()
                time.sleep(0.01)

    statuses: list[int] = []
    lock = threading.Lock()

    def worker(offset: int):
        client = TestClient(app)  # no lifespan: the primary client ran it once
        for i in range(25):
            tenant = tenants[(offset + i * 7) % len(tenants)]
            r = client.get("/beliefs", params={"user_id": "u1"},
                           headers={"Authorization": f"Bearer key-{tenant[0]}",
                                    "X-MM-Purpose": "test"})
            with lock:
                statuses.append(r.status_code)

    with TestClient(app):
        shared = server_mod._app_state["sa_engine"]
        assert shared is not None and shared.dialect.name == "postgresql"
        assert shared.pool.size() == pool_size
        pool_before = shared.pool
        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        mon.join(timeout=5)

        assert statuses and set(statuses) == {200}
        assert samples, "monitor took no samples"
        print(
            f"\npg_stat_activity peak for application_name=mirror-memory: "
            f"{max(samples)} (bound pool_size+max_overflow = {pool_size + max_overflow}); "
            f"{len(statuses)} requests over {len(tenants)} tenants, {len(samples)} samples"
        )
        assert max(samples) <= pool_size + max_overflow, max(samples)
        # Every resident wrapper rides the one shared engine; eviction of
        # 34 wrappers disposed nothing the survivors depend on.
        resident = server_mod._tenant_engines.items()
        assert 0 < len(resident) <= 16
        assert all(e._engine is shared and not e._owns_engine for _, e in resident)
        assert shared.pool is pool_before
        probe = TestClient(app).get("/beliefs", params={"user_id": "u1"},
                                    headers={"Authorization": "Bearer key-t00",
                                             "X-MM-Purpose": "test"})
        assert probe.status_code == 200


# ---------------------------------------------------------------------------
# Lock-free creation: a cold start never blocks other tenants
# ---------------------------------------------------------------------------


def test_cold_start_of_one_tenant_does_not_block_others():
    from mirror_memory.server import _TenantEngineCache

    cache = _TenantEngineCache()
    gate = threading.Event()
    results: dict[str, object] = {}
    create_calls: list[str] = []

    def slow_x():
        create_calls.append("x")
        gate.wait(5)
        return "X-engine"

    def never():
        create_calls.append("never")
        return "never"

    t_x = threading.Thread(target=lambda: results.__setitem__("x", cache.get_or_create(("x", "a"), slow_x)[0]))
    t_x.start()
    time.sleep(0.05)  # x's placeholder is parked, its creation is blocked on the gate

    t0 = time.monotonic()
    y, evicted = cache.get_or_create(("y", "a"), lambda: "Y-engine")
    assert y == "Y-engine" and evicted == []
    assert time.monotonic() - t0 < 0.5  # not behind x's cold start

    t_x2 = threading.Thread(target=lambda: results.__setitem__("x2", cache.get_or_create(("x", "a"), never)[0]))
    t_x2.start()
    time.sleep(0.05)
    assert t_x2.is_alive()  # waiting for the same result, not creating a second engine
    assert cache.get(("x", "a")) is None  # placeholder is invisible to get()
    assert len(cache) == 1  # only y counts

    gate.set()
    t_x.join(5)
    t_x2.join(5)
    assert results["x"] == results["x2"] == "X-engine"
    assert create_calls == ["x"]
    assert len(cache) == 2


def test_failed_creation_releases_the_placeholder_and_propagates():
    from mirror_memory.server import _TenantEngineCache

    cache = _TenantEngineCache()

    def boom():
        raise RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        cache.get_or_create(("x", "a"), boom)
    assert len(cache) == 0 and cache.get(("x", "a")) is None

    gate = threading.Event()
    errors: list[BaseException] = []

    def slow_boom():
        gate.wait(5)
        raise RuntimeError("database down")

    def capture(create):
        try:
            cache.get_or_create(("x", "a"), create)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=capture, args=(slow_boom,))
    t1.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=capture, args=(lambda: "late",))
    t2.start()
    time.sleep(0.05)
    gate.set()
    t1.join(5)
    t2.join(5)
    assert len(errors) == 2 and all("database down" in str(e) for e in errors)
    # the next attempt starts fresh
    assert cache.get_or_create(("x", "a"), lambda: "fresh")[0] == "fresh"


def test_wrapper_never_disposes_a_shared_pool(tmp_path):
    from mirror_memory import server as server_mod
    from mirror_memory.api import MemoryEngine

    shared = create_engine(f"sqlite:///{tmp_path}/shared.db")
    Base.metadata.create_all(shared)
    disposed = []
    original = shared.pool.dispose
    shared.pool.dispose = lambda: (disposed.append(1), original())  # type: ignore[method-assign]

    wrapper = MemoryEngine(config=load_config("config/"), database_url="unused",
                           tenant_id="tA", app_id="app", engine=shared, schema_ready=True)
    wrapper._ensure_db()
    assert wrapper._engine is shared and not wrapper._owns_engine
    server_mod._release_engine(wrapper)
    assert disposed == []

    owner = MemoryEngine(config=load_config("config/"),
                         database_url=f"sqlite:///{tmp_path}/own.db", tenant_id="tA", app_id="app")
    owner._ensure_db()
    assert owner._owns_engine
    server_mod._release_engine(owner)  # owned pools are still disposed on eviction
    shared.dispose()
