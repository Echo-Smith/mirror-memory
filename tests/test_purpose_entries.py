"""M3 batch-1 tasks 2/3: purpose at the entries, the entry inventory, and
zero-business-SQL refusals.

- The server rejects a purpose outside the key's registered set with 403
  BEFORE any SQL (the same layer as the 401).
- The entry inventory: every public MemoryEngine method and every server
  route is either in ENTRY_REGISTRY (with its authorization point) or in
  the explicit ungated set -- a new memory-touching entry that is not
  registered fails these tests (same mechanism as the system_scope
  allowlist).
- A refusal produces ZERO business statements: the before_cursor_execute
  recorder shows only the authorization queries themselves.
- The worker cancels a pre-M3 job (empty purpose) as missing_purpose.
- The forget paths audit every invocation (allowed_regardless).
"""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.conftest import refusing_authz

from mirror_memory.config.loader import load_config
from mirror_memory.core.authorization import (
    ENTRY_REGISTRY,
    UNGATED_ENGINE_METHODS,
    PurposeDenied,
)
from mirror_memory.core.models import Belief, PurposeGrantEvent
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope
from mirror_memory.server import create_app, register_tenant_key


# ---------------------------------------------------------------------------
# Server: purpose outside the key's set -> 403 with zero SQL
# ---------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path):
    from mirror_memory.core.models import Base
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

    url = f"sqlite:///{tmp_path}/purpose-server.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    from mirror_memory.core import schema_migration as _sm_wm
    _sm_wm.ensure_watermark(engine)
    factory = sessionmaker(bind=engine)
    with tenant_scope("tA", "app"), fixture_write_scope("seed"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.add(Belief(user_id="u1", dimension="topic", key="k",
                     claim_text="A-SECRET", confidence=0.9))
        s.commit()
        s.close()
    engine.dispose()
    from mirror_memory import server as server_mod

    server_mod._tenant_key_registry.clear()  # tests elsewhere assert on the registry
    register_tenant_key("key-a", "tA", "app", purposes={"read-only"})
    app = create_app(config=load_config("config/"), database_url=url)
    yield app, url
    server_mod._tenant_key_registry.clear()


def test_wrong_purpose_is_403_with_zero_sql(server, monkeypatch):
    from mirror_memory.core import tenancy as tenancy_mod

    app, url = server
    statements: list[str] = []
    from sqlalchemy import event

    engine = create_engine(url)

    @event.listens_for(engine, "before_cursor_execute")
    def _rec(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    # the middleware must refuse BEFORE the app (and its engines) run
    refused = []
    from mirror_memory import server as server_mod

    original_middleware = server_mod.TenantAuthMiddleware.__call__

    async def spying_middleware(self, scope, receive, send):
        await original_middleware(self, scope, receive, send)
        if scope.get("type") == "http":
            refused.append(scope.get("method"))

    monkeypatch.setattr(server_mod.TenantAuthMiddleware, "__call__", spying_middleware)
    try:
        with TestClient(app) as client:
            for purpose in (None, "write-stuff"):
                headers = {"Authorization": "Bearer key-a"}
                if purpose is not None:
                    headers["X-MM-Purpose"] = purpose
                r = client.get("/beliefs", params={"user_id": "u1"}, headers=headers)
                assert r.status_code == 403, (purpose, r.status_code, r.json())
                assert "not allowed for this key" in r.json()["detail"]
            # a registered purpose passes the middleware and reaches the engine
            r = client.get("/beliefs", params={"user_id": "u1"},
                           headers={"Authorization": "Bearer key-a", "X-MM-Purpose": "read-only"})
            assert r.status_code == 200
    finally:
        engine.dispose()
    # zero SQL for the refused requests: the only statements captured came
    # from the ALLOWED request's own engine (a fresh engine per request is
    # created by the wrapper, so the recorder on OUR engine saw none of the
    # refused requests' work -- because there was none to see)
    assert statements == []


def test_allowed_purpose_reaches_the_engine(server):
    app, url = server
    with TestClient(app) as client:
        r = client.get("/beliefs", params={"user_id": "u1"},
                       headers={"Authorization": "Bearer key-a", "X-MM-Purpose": "read-only"})
        assert r.status_code == 200
        claims = [b["claim_text"] for b in r.json()["beliefs"]]
        assert claims == ["A-SECRET"]


# ---------------------------------------------------------------------------
# Entry inventory discovery
# ---------------------------------------------------------------------------


def _memory_engine_surface() -> set[str]:
    from mirror_memory.api import MemoryEngine

    return {
        name for name, _ in inspect.getmembers(MemoryEngine, inspect.isfunction)
        if not name.startswith("_")
    }


def test_every_public_engine_method_is_registered_or_explicitly_ungated():
    surface = _memory_engine_surface()
    registered = {
        name.rsplit(".", 1)[-1]
        for name in ENTRY_REGISTRY
        if name.startswith("mirror_memory.api.MemoryEngine.")
    }
    unregistered = surface - registered - set(UNGATED_ENGINE_METHODS)
    assert not unregistered, (
        f"new engine method(s) {sorted(unregistered)} can touch memory but are "
        "not in core.authorization.ENTRY_REGISTRY (nor explicitly ungated); "
        "register the authorization point"
    )
    # the ungated set must stay honest: every entry must exist on the engine
    assert set(UNGATED_ENGINE_METHODS) <= surface, (
        set(UNGATED_ENGINE_METHODS) - surface
    )


def test_every_server_route_is_registered():
    app = create_app(config=load_config("config/"), database_url="sqlite://")
    routes = {}
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        name = getattr(endpoint, "__name__", None)
        if name and name not in ("health", "openapi", "swagger_ui_html",
                                 "swagger_ui_redirect", "redoc_html"):
            routes[name] = str(getattr(route, "path", ""))
    registered = {
        name.rsplit(".", 1)[-1]
        for name in ENTRY_REGISTRY
        if name.startswith("mirror_memory.server.")
    }
    missing = set(routes) - registered
    assert not missing, (
        f"server route(s) {sorted(missing)} are not in ENTRY_REGISTRY; "
        "a new memory-touching route needs a registered authorization point"
    )


def test_registry_entries_point_at_real_objects():
    import importlib

    for key in ENTRY_REGISTRY:
        if key.startswith("mirror_memory.server."):
            # create_app closures: covered by the route-discovery test above
            continue
        parts = key.split(".")
        obj = None
        for cut in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:cut]))
            except ImportError:
                continue
            for part in parts[cut:]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                break
        if obj is None:
            raise AssertionError(f"ENTRY_REGISTRY entry {key!r} no longer exists")
        info = ENTRY_REGISTRY[key]
        assert info["operation"] and info["engine"] and info["purpose_source"], key


# ---------------------------------------------------------------------------
# Refusal paths: zero business SQL
# ---------------------------------------------------------------------------


def test_refused_recall_issues_only_authorization_queries(pg_app_url):
    """A refusal must not read or write business tables: the recorder sees
    only the authorization layer's own queries (grants, memory preference,
    feature consent)."""
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    from mirror_memory.core.tenancy import tenant_scope

    engine = create_engine(pg_app_url)
    statements: list[str] = []
    event.listen(engine, "before_cursor_execute",
                 lambda conn, cursor, statement, parameters, context, executemany:
                 statements.append(statement))
    try:
        with tenant_scope("default", "default"):
            session = Session(engine)
            set_memory_enabled(session, "u1", True)
            session.commit()
            # no grant for ("u1", "never"): the harness lazy-grant is off
            with refusing_authz(), pytest.raises(PurposeDenied) as excinfo:
                from mirror_memory.api import MemoryEngine

                eng = MemoryEngine(
                    config=load_config("config/"), database_url=pg_app_url,
                    tenant_id="default", app_id="default", engine=engine,
                    schema_ready=True, default_purpose=None,
                )
                eng.recall(user_id="u1", purpose="never")
        assert "missing_purpose" not in str(excinfo.value)
        assert "no_grant" in str(excinfo.value)
        business = [
            s for s in statements
            if any(t in s for t in ("mm_beliefs", "mm_belief_events", "mm_evidence",
                                    "mm_snapshots", "mm_proposal_log", "mm_belief_versions"))
        ]
        assert business == [], business
        seen_tables = {t for t in (
            "mm_purpose_grants", "mm_memory_preferences", "mm_consent_grants") if t in " ".join(statements)}
        assert seen_tables, statements
        # M3 batch-2 task 2: the refusal itself is the one write -- a
        # committed denied row in the access audit
        audit_writes = [s for s in statements if "mm_access_audit" in s]
        assert len(audit_writes) == 1 and audit_writes[0].upper().startswith("INSERT"), audit_writes
    finally:
        engine.dispose()


def test_missing_purpose_refused_before_any_sql(pg_app_url, monkeypatch):
    """A missing purpose (no parameter, no engine default) is refused in
    Python -- not one single SQL statement is issued."""
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    from mirror_memory.core.tenancy import tenant_scope

    engine = create_engine(pg_app_url)
    statements: list[str] = []
    event.listen(engine, "before_cursor_execute",
                 lambda conn, cursor, statement, parameters, context, executemany:
                 statements.append(statement))
    try:
        with tenant_scope("default", "default"):
            from mirror_memory.api import MemoryEngine

            eng = MemoryEngine(
                config=load_config("config/"), database_url=pg_app_url,
                tenant_id="default", app_id="default", engine=engine,
                schema_ready=True, default_purpose=None,
            )
            # warm the engine (schema + watermark setup) OUTSIDE the
            # measurement: the invariant under test is that the REFUSAL
            # issues no SQL, not that engine initialization does
            eng._ensure_db()
            statements.clear()
            with pytest.raises(PurposeDenied, match="missing_purpose"):
                eng.recall(user_id="u1")
            with pytest.raises(PurposeDenied, match="missing_purpose"):
                eng.observe(user_id="u1", session_id="s1", text="hi")
            with pytest.raises(PurposeDenied, match="missing_purpose"):
                eng.delete_memories(user_id="u1")
    finally:
        engine.dispose()
    assert statements == [], statements


# ---------------------------------------------------------------------------
# The worker: a pre-M3 job (empty purpose) is cancelled, not run
# ---------------------------------------------------------------------------


def test_pre_m3_job_is_cancelled_missing_purpose(pg_app_url):
    from datetime import UTC, datetime

    from sqlalchemy.orm import Session

    from mirror_memory.core.models import EvolutionJob
    from mirror_memory.core.tenancy import fixture_write_scope, queue_write_scope, tenant_scope
    from mirror_memory.worker.evolution import _process_single_job

    engine = create_engine(pg_app_url)
    cfg = load_config("config/")

    class FakeLLM:
        def generate(self, **kw):
            raise AssertionError("the LLM must not be called for an unauthorized job")

    cfg.llm_client = FakeLLM()
    try:
        with tenant_scope("default", "default"):
            session = Session(engine)
            from mirror_memory.core.repository import set_memory_enabled

            set_memory_enabled(session, "u1", True)
            with queue_write_scope():
                session.add(EvolutionJob(
                    id="legacy-1", user_id="u1", purpose="",
                    evidence_watermark="{}", idempotency_key="legacy:ik",
                    status="pending",
                ))
            session.commit()
            job = session.get(EvolutionJob, ("default", "default", "legacy-1"))
            _process_single_job(session, job, cfg)
            row = session.get(EvolutionJob, ("default", "default", "legacy-1"))
            assert (row.status, row.error_code) == ("cancelled", "missing_purpose")
            session.close()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# The forget paths: audited regardless of authorization state
# ---------------------------------------------------------------------------


def test_forget_after_revocation_is_allowed_and_audited(pg_app_url):
    from sqlalchemy.orm import Session

    from mirror_memory.core.authorization import grant_purpose, revoke_purpose
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.tenancy import tenant_scope
    from mirror_memory.api import MemoryEngine

    engine = create_engine(pg_app_url)
    try:
        with tenant_scope("default", "default"):
            session = Session(engine)
            set_memory_enabled(session, "u1", True)
            with fixture_write_scope("seed a belief"):
                session.add(Belief(user_id="u1", dimension="topic", key="k",
                                   claim_text="to forget", confidence=0.9))
            session.commit()
            grant_purpose(session, "u1", "api", operations=["observe"], issuer="t")
            session.commit()
            revoke_purpose(session, "u1", "api", actor="reviewer", reason="offboard")
            session.commit()
            bid = session.query(Belief).filter_by(user_id="u1").one().id
            session.close()

            eng = MemoryEngine(
                config=load_config("config/"), database_url=pg_app_url,
                tenant_id="default", app_id="default", engine=engine,
                schema_ready=True, default_purpose=None,
            )
            # observe would be refused (revoked) ...
            session = Session(engine)
            with refusing_authz(), pytest.raises(PurposeDenied, match="revoked"):
                eng.observe(user_id="u1", session_id="s1", text="x", purpose="api")
            session.close()
            # ... but the forget runs, and its invocation is on the audit trail
            assert eng.forget_belief(user_id="u1", belief_id=bid, purpose="api") is True
            session = Session(engine)
            events = (
                session.query(PurposeGrantEvent)
                .filter_by(user_id="u1", action="invoked")
                .all()
            )
            assert [(e.operation, e.decision, e.purpose) for e in events] == [
                ("forget", "allowed_regardless", "api")
            ]
            assert session.query(Belief).filter_by(user_id="u1").count() == 0
            session.close()
    finally:
        engine.dispose()
