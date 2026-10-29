"""FastAPI server for mirror-memory.

Provides HTTP endpoints for memory ingestion, retrieval, and management.

Usage::

    # As a module:
    uvicorn mirror_memory.server:app --host 0.0.0.0 --port 8000

    # Or programmatically:
    from mirror_memory.server import create_app
    app = create_app(config_path="config/", database_url="sqlite:///mm.db")
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import text

from mirror_memory import __version__
from mirror_memory.core.authorization import PurposeDenied
from mirror_memory.core import access_audit
from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.config.schema import MemoryConfig
from mirror_memory.exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

_security = HTTPBearer(auto_error=False)


class TenantAuthMiddleware:
    """Authenticates the Bearer key to a tenant and wraps the WHOLE
    request in tenant_scope (M1 batch-2 task 5).

    Why an ASGI middleware and not a yield-dependency: dependency
    teardown runs in a different contextvars Context than setup on this
    stack, so a scope token reset there raises.  The middleware opens and
    closes the scope inside one task -- set/reset share a Context.

    Rejects unauthenticated requests BEFORE routing, so no handler (and
    no SQL) ever runs for them.  Client-supplied tenant headers are never
    read: the tenant comes only from the key.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _tenant_key_registry:
            await self.app(scope, receive, send)
            return

        from starlette.responses import JSONResponse

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        presented = None
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            presented = auth[7:].strip()
        purpose = (headers.get("x-mm-purpose") or "").strip() or None
        try:
            tenant = _resolve_tenant(presented, purpose)
        except HTTPException as exc:
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            await response(scope, receive, send)
            return

        from mirror_memory.core.tenancy import tenant_scope

        scope.setdefault("state", {})["mm_tenant"] = tenant
        scope["state"]["mm_purpose"] = purpose
        with tenant_scope(*tenant):
            await self.app(scope, receive, send)


def _verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(_security),
):
    """Route guard + tenant source for handlers.

    Tenant mode: the middleware has already authenticated the request
    and opened tenant_scope; this dependency merely surfaces the resolved
    tenant to the handler (from request.state).  MM_API_KEY legacy mode:
    unchanged single-key check.
    """
    if _tenant_key_registry:
        return getattr(request.state, "mm_tenant", None)
    expected = os.environ.get("MM_API_KEY")
    if expected is None:
        return None  # no auth required
    if credentials is None or not hmac.compare_digest(
        credentials.credentials, expected
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return None


def _request_purpose(request: Request) -> str | None:
    """The request's purpose (middleware-validated against the key's set in
    tenant mode; taken from the header as-is in legacy MM_API_KEY mode,
    where the engine's authorization layer enforces it)."""
    value = request.headers.get("x-mm-purpose", "").strip()
    return value or None


def _get_engine(tenant: tuple[str, str] | None = None) -> MemoryEngine:
    if tenant is not None:
        return _engine_for(tenant)
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized. Call create_app() first.")
    return _engine


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class ObserveRequest(BaseModel):
    """Ingest a single conversation turn."""

    user_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=256)
    text: str = Field(..., min_length=1, max_length=50000)
    turn_count: int = Field(default=1, ge=0)


class ObserveResponse(BaseModel):
    status: str = "ok"
    claims_extracted: int = 0


class RecallRequest(BaseModel):
    """Retrieve memory context for a query."""

    user_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(default="", max_length=5000)
    language: str = Field(default="en", pattern="^(zh|en)$")


class RecallResponse(BaseModel):
    user_id: str
    context: str | None = None
    char_count: int = 0


class BeliefItem(BaseModel):
    dimension: str
    key: str
    claim_text: str
    confidence: float
    layer: str
    source: str
    last_evidence_at: str | None = None


class BeliefsResponse(BaseModel):
    user_id: str
    beliefs: list[BeliefItem]
    total: int


class PanelResponse(BaseModel):
    user_id: str
    panel: dict


class DeleteRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)


class DeleteResponse(BaseModel):
    status: str = "deleted"
    counts: dict[str, int]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    dimensions: int
    anchors: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_engine: MemoryEngine | None = None

# ---------------------------------------------------------------------------
# Tenant-resolving authentication (M1 batch-2 task 5)
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402
import hmac  # noqa: E402
import threading  # noqa: E402

#: sha256(key) -> (tenant_id, app_id, frozenset(purposes)).  Raw keys never
#: live here.  The purpose set is what the key may declare in the
#: X-MM-Purpose header (M3 batch-1 task 2); anything else is 403 before any
#: SQL, the same layer as the 401.
_tenant_key_registry: dict[str, tuple[str, str, frozenset]] = {}
#: Bounded LRU cache of per-tenant engines (closeout task 3): an unbounded
#: dict would leak one engine (and its connection pool) per tenant forever.
#: Bound is MM_TENANT_ENGINE_CACHE (default 32); evicted engines get their
#: pool disposed.
_TENANT_ENGINE_CACHE_DEFAULT = 32


def _engine_cache_limit() -> int:
    try:
        return max(1, int(os.environ.get("MM_TENANT_ENGINE_CACHE", _TENANT_ENGINE_CACHE_DEFAULT)))
    except ValueError:
        return _TENANT_ENGINE_CACHE_DEFAULT


def _release_engine(engine: "MemoryEngine") -> None:
    """Dispose an evicted engine's connection pool -- unless the pool is the
    server's shared PostgreSQL engine, which the wrapper does not own."""
    if not getattr(engine, "_owns_engine", True):
        return
    pool = getattr(getattr(engine, "_engine", None), "pool", None)
    if pool is not None:
        try:
            pool.dispose()
        except Exception:  # noqa: BLE001 -- eviction must never break a request
            logger.warning("tenant engine eviction: pool dispose failed", exc_info=True)


class _Pending:
    """Placeholder a creating thread parks in the cache while it builds the
    engine OUTSIDE the lock; same-tenant requests wait on its future."""

    __slots__ = ("future",)

    def __init__(self) -> None:
        import concurrent.futures

        self.future: "concurrent.futures.Future[MemoryEngine]" = concurrent.futures.Future()


class _TenantEngineCache:
    """Thread-safe bounded LRU cache (insertion order = recency order).

    M2 batch-2 task 2: engine CREATION happens outside the lock.  A miss
    parks a :class:`_Pending` placeholder under the lock and releases it;
    the creating thread builds the engine, then re-takes the lock to swap
    the placeholder for the engine and evict.  Other requests for the same
    tenant find the placeholder and wait on its future, so a tenant still
    gets exactly one engine -- but a cold start for one tenant no longer
    holds the lock other tenants' lookups need.  Placeholders are never
    counted or evicted.  Evicted engines are returned to the CALLER, whose
    dispose happens outside the lock as before.
    """

    def __init__(self):
        self._data: dict[tuple[str, str], object] = {}
        self._lock = threading.Lock()

    def _real_count(self) -> int:
        return sum(1 for v in self._data.values() if not isinstance(v, _Pending))

    def get(self, key):
        """Return the cached engine (marking it recently used) or None."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None or isinstance(entry, _Pending):
                return None
            del self._data[key]
            self._data[key] = entry
            return entry

    def get_or_create(self, key, create) -> tuple[MemoryEngine, list]:
        """Return the engine for *key*, creating it via *create()* when
        absent.  Returns (engine, evicted_engines) -- dispose the evicted
        list OUTSIDE the lock."""
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and not isinstance(entry, _Pending):
                del self._data[key]
                self._data[key] = entry
                return entry, []
            creator = entry is None
            if creator:
                entry = _Pending()
                self._data[key] = entry
            future = entry.future
        if not creator:
            # Same result -- or the same exception -- as the creating thread.
            return future.result(), []
        try:
            engine = create()
        except BaseException as exc:
            with self._lock:
                if self._data.get(key) is entry:
                    del self._data[key]
            future.set_exception(exc)
            raise
        with self._lock:
            if self._data.get(key) is entry:
                del self._data[key]
            self._data[key] = engine
            evicted = []
            limit = _engine_cache_limit()
            while self._real_count() > limit:
                oldest = next(k for k, v in self._data.items() if not isinstance(v, _Pending))
                evicted.append(self._data.pop(oldest))
        future.set_result(engine)
        return engine, evicted

    def __len__(self) -> int:
        with self._lock:
            return self._real_count()

    def items(self):
        with self._lock:
            return [(k, v) for k, v in self._data.items() if not isinstance(v, _Pending)]

    def clear(self) -> list:
        """Drop every engine; returns them for outside-lock disposal.
        In-flight placeholders stay so their creators can still land."""
        with self._lock:
            dropped = [v for v in self._data.values() if not isinstance(v, _Pending)]
            for k in [k for k, v in self._data.items() if not isinstance(v, _Pending)]:
                del self._data[k]
            return dropped


_tenant_engines: _TenantEngineCache = _TenantEngineCache()
_app_state: dict[str, Any] = {}


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def register_tenant_key(api_key: str, tenant_id: str, app_id: str, *, purposes) -> None:
    """Register an API key mapping (call before/after create_app).

    *purposes* is the set of purposes this key may declare per request
    (the X-MM-Purpose header value must be one of them).  Required and
    fail-closed: a key without purposes cannot use purpose-gated endpoints
    at all.  Only the SHA-256 digest is kept; comparison at request time
    is constant-time over digests.
    """
    purpose_set = frozenset(purposes or ())
    if not purpose_set:
        raise ValueError(
            "register_tenant_key requires a non-empty purposes set: a key "
            "without allowed purposes cannot call purpose-gated endpoints"
        )
    _tenant_key_registry[_hash_key(api_key)] = (tenant_id, app_id, purpose_set)


def load_tenant_keys_file(path: str | Path) -> None:
    """Load {"keys": {"<sha256-hex>": [tenant_id, app_id]}} from JSON.

    The file must contain DIGESTS, never raw keys.
    """
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for digest, entry in data["keys"].items():
        if len(entry) < 3 or not entry[2]:
            raise ValueError(
                f"tenant keys file: digest {digest[:8]}... has no purpose set; "
                "entries are [tenant_id, app_id, [purposes]] -- fail-closed, "
                "a key without purposes is refused"
            )
        _tenant_key_registry[digest] = (str(entry[0]), str(entry[1]), frozenset(entry[2]))


def _resolve_tenant(presented: str | None, purpose: str | None) -> tuple[str, str]:
    """Constant-time key check -> tenant, with the purpose checked against
    the key's registered set (403).  Both raise before any database work:
    an unauthenticated or unauthorized request must not cause a single SQL
    statement.
    """
    if presented is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    digest = _hash_key(presented)
    for stored, entry in _tenant_key_registry.items():
        if hmac.compare_digest(stored, digest):
            tenant_id, app_id, purposes = entry
            if purpose is None or purpose not in purposes:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Purpose {purpose!r} is not allowed for this key "
                        f"(allowed: {sorted(purposes)}); set the X-MM-Purpose header"
                    ),
                )
            return (tenant_id, app_id)
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _engine_for(tenant: tuple[str, str]) -> MemoryEngine:
    """Per-tenant engine cache (LRU, bounded by MM_TENANT_ENGINE_CACHE):
    each engine binds its own sessions inside its tenant scope (engines
    are single-tenant by construction).

    PostgreSQL (M2 batch-2 task 2): every tenant wrapper shares the ONE
    SQLAlchemy engine / pool the lifespan engine opened (tenants are told
    apart by the per-transaction setting and the loader criteria, not by
    pools), so the cache bounds wrappers, not connections, and eviction
    never disposes the shared pool.  SQLite keeps one engine per tenant.

    Concurrency: creation runs outside the cache lock behind a
    placeholder; concurrent first requests for a tenant share ONE engine
    and other tenants are not blocked while it is built."""

    def _create() -> MemoryEngine:
        shared = _app_state.get("sa_engine")
        engine = MemoryEngine(
            config=_app_state["config"],
            database_url=_app_state["database_url"],
            llm_client=_app_state["llm_client"],
            tenant_id=tenant[0],
            app_id=tenant[1],
            engine=shared,
            schema_ready=shared is not None,
        )
        engine._ensure_db()
        return engine

    engine, evicted = _tenant_engines.get_or_create(tenant, _create)
    for dropped in evicted:
        _release_engine(dropped)
    return engine


def create_app(
    *,
    config_path: str | Path | None = None,
    config: MemoryConfig | None = None,
    database_url: str | None = None,
    llm_client: Any | None = None,
    llm_api_key: str | None = None,
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str | None = None,
) -> FastAPI:
    """Create a FastAPI app with mirror-memory endpoints.

    Parameters
    ----------
    config_path:
        Path to the configuration directory. Ignored if *config* is provided.
    config:
        Pre-loaded ``MemoryConfig``. Takes precedence over *config_path*.
    database_url:
        SQLAlchemy database URL. Defaults to env var ``MM_DATABASE_URL``
        or ``sqlite:///mirror_memory.db``.
    llm_client:
        Pre-built LLM client. Takes precedence over llm_api_key/llm_model.
    llm_api_key:
        API key for the built-in OpenAI-compatible LLM adapter.
    llm_model:
        Model identifier for the built-in adapter.
    llm_base_url:
        API base URL for non-OpenAI providers.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Configure structured logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        global _engine
        from mirror_memory.core.tenancy import READ_FILTER_NONE, read_filter_mode

        if read_filter_mode() == READ_FILTER_NONE:
            # The unfiltered baseline exists for measurement only; a server
            # must never answer requests without a tenant read filter.
            raise RuntimeError(
                "MM_TENANT_READ_FILTER=none is a measurement-only baseline; "
                "the HTTP server refuses to start without a tenant read filter"
            )
        _engine = MemoryEngine(
            config_path=config_path,
            config=config,
            database_url=database_url or os.environ.get("MM_DATABASE_URL", "sqlite:///mirror_memory.db"),
            llm_client=llm_client,
            llm_api_key=llm_api_key or os.environ.get("MM_LLM_API_KEY"),
            llm_model=os.environ.get("MM_LLM_MODEL", llm_model),
            llm_base_url=llm_base_url or os.environ.get("MM_LLM_BASE_URL"),
        )
        _engine._ensure_db()
        # Task 5: per-tenant engine factory state + optional key file.
        # PostgreSQL: the lifespan engine's SQLAlchemy engine is THE shared
        # pool every tenant wrapper binds to (M2 batch-2 task 2).
        from mirror_memory.core.rls import is_postgres

        _app_state.clear()
        _app_state.update(
            config=_engine._config,
            database_url=_engine._database_url,
            llm_client=_engine._llm_client,
            sa_engine=_engine._engine if is_postgres(_engine._engine) else None,
        )
        if getattr(_engine, "_single_tenant", False) and _tenant_key_registry:
            raise RuntimeError(
                "single-tenant mode is mutually exclusive with the server "
                "tenant middleware"
            )
        keys_file = os.environ.get("MM_TENANT_KEYS_FILE")
        if keys_file:
            load_tenant_keys_file(keys_file)
            logger.info("tenant auth enabled: %d key(s) registered", len(_tenant_key_registry))
        logger.info("mirror-memory engine initialized")
        yield
        primary = _engine
        _engine = None
        for dropped in _tenant_engines.clear():
            _release_engine(dropped)
        # The shared pool belongs to the lifespan engine; wrappers never
        # disposed it, so it goes last.
        if primary is not None and getattr(primary, "_engine", None) is not None:
            primary._engine.dispose()

    app = FastAPI(
        title="mirror-memory",
        description="Structured memory engine for AI agents",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # outermost: authenticate + tenant_scope before anything else runs
    app.add_middleware(TenantAuthMiddleware)

    # -- Endpoints -----------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health():
        engine = _get_engine()
        # Verify database connectivity
        try:
            with engine._session() as session:
                session.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(status_code=503, detail="Database unavailable")
        return HealthResponse(
            status="ok",
            version=__version__,
            dimensions=len(engine._config.dimensions),
            anchors=len(engine._config.anchors),
        )

    @app.post("/observe", response_model=ObserveResponse)
    def observe(
        req: ObserveRequest,
        tenant: tuple[str, str] | None = Depends(_verify_api_key),
        purpose: str | None = Depends(_request_purpose),
    ):
        """Ingest a single conversation turn into memory."""
        engine = _get_engine(tenant)
        try:
            count = engine.observe(
                user_id=req.user_id,
                session_id=req.session_id,
                text=req.text,
                turn_count=req.turn_count,
                purpose=purpose,
            )
            return ObserveResponse(status="ok", claims_extracted=count)
        except PurposeDenied as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except LLMError as e:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except Exception as e:
            logger.warning("observe failed for user %s: %s", req.user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.post("/recall", response_model=RecallResponse)
    def recall(
        req: RecallRequest,
        tenant: tuple[str, str] | None = Depends(_verify_api_key),
        purpose: str | None = Depends(_request_purpose),
    ):
        """Retrieve memory context for a query."""
        engine = _get_engine(tenant)
        try:
            with access_audit.entry_label("mirror_memory.server.recall"):
                ctx = engine.recall(
                    user_id=req.user_id,
                    query=req.query,
                    language=req.language,
                    purpose=purpose,
                )
            return RecallResponse(
                user_id=req.user_id,
                context=ctx,
                char_count=len(ctx or ""),
            )
        except PurposeDenied as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except LLMError as e:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except Exception as e:
            logger.warning("recall failed for user %s: %s", req.user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.get("/panel", response_model=PanelResponse)
    def panel(
        user_id: str = Query(..., min_length=1),
        language: str = Query("en", pattern="^(zh|en)$"),
        tenant: tuple[str, str] | None = Depends(_verify_api_key),
        purpose: str | None = Depends(_request_purpose),
    ):
        """Get structured panel data for frontend display."""
        engine = _get_engine(tenant)
        try:
            with access_audit.entry_label("mirror_memory.server.panel"):
                data = engine.panel(user_id=user_id, language=language, purpose=purpose)
            return PanelResponse(
                user_id=user_id,
                panel={"memory": data.memory, "sections": data.sections, "avatar": data.avatar},
            )
        except PurposeDenied as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except LLMError as e:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except Exception as e:
            logger.warning("panel failed for user %s: %s", user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.get("/beliefs", response_model=BeliefsResponse)
    def beliefs(
        user_id: str = Query(..., min_length=1),
        limit: int = Query(50, ge=1, le=200),
        tenant: tuple[str, str] | None = Depends(_verify_api_key),
        purpose: str | None = Depends(_request_purpose),
    ):
        """List active beliefs for a user."""
        engine = _get_engine(tenant)
        try:
            with access_audit.entry_label("mirror_memory.server.beliefs"):
                raw = engine.get_beliefs(user_id=user_id, limit=limit, purpose=purpose)
            items = [
                BeliefItem(
                    dimension=b.dimension, key=b.key, claim_text=b.claim_text,
                    confidence=b.confidence, layer=b.layer, source=b.source,
                    last_evidence_at=b.last_evidence_at.isoformat() if b.last_evidence_at else None,
                )
                for b in raw
            ]
            return BeliefsResponse(user_id=user_id, beliefs=items, total=len(items))
        except PurposeDenied as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except LLMError as e:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except Exception as e:
            logger.warning("beliefs failed for user %s: %s", user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.post("/delete", response_model=DeleteResponse)
    def delete_memories(
        req: DeleteRequest,
        tenant: tuple[str, str] | None = Depends(_verify_api_key),
        purpose: str | None = Depends(_request_purpose),
    ):
        """Delete all memory data for a user."""
        engine = _get_engine(tenant)
        try:
            result = engine.delete_memories(user_id=req.user_id, purpose=purpose)
            return DeleteResponse(status="deleted", counts=result.as_dict())
        except PurposeDenied as e:
            raise HTTPException(status_code=403, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except LLMError as e:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except Exception as e:
            logger.warning("delete failed for user %s: %s", req.user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.get("/belief/{belief_id}")
    def belief_by_id(
        belief_id: int,
        user_id: str = Query(..., min_length=1),
        tenant: tuple[str, str] | None = Depends(_verify_api_key),
        purpose: str | None = Depends(_request_purpose),
    ):
        """Fetch one belief by id.

        Cross-tenant ids (or nonexistent ones) both return 404: the
        response must not reveal whether the resource exists in another
        tenant.
        """
        engine = _get_engine(tenant)
        try:
            with engine._session() as session:
                authorized = engine._authorize(
                    session, user_id, purpose, "explain",
                    audit_entry="mirror_memory.server.belief_by_id",
                )
                from mirror_memory.core import provenance
                from mirror_memory.core.models import Belief

                row = session.get(Belief, belief_id)
                found = row is not None and row.user_id == user_id
                if found:
                    # cross-purpose visibility: an invisible belief is a 404
                    # here, exactly like a cross-tenant one
                    rule = provenance.visibility(session, user_id, authorized.purpose)
                    found = provenance.belief_is_visible(session, user_id, rule, belief_id)
                # the audit row rides the same transaction as the read and
                # is committed before anything is returned -- a miss is an
                # access attempt too
                engine._audit(session, authorized, user_id=user_id, operation="explain",
                              entry="mirror_memory.server.belief_by_id",
                              memory_ids=[row.id] if found else [])
                session.commit()
                if not found:
                    raise HTTPException(status_code=404, detail="Not found")
                return {
                    "belief_id": row.id,
                    "dimension": row.dimension,
                    "key": row.key,
                    "claim_text": row.claim_text,
                    "confidence": row.confidence,
                    "status": row.status,
                }
        except HTTPException:
            raise
        except PurposeDenied as e:
            raise HTTPException(status_code=403, detail=str(e))
        except Exception as e:
            logger.warning("belief fetch failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.get("/export")
    def export(
        user_id: str = Query(..., min_length=1),
        cursor: str | None = Query(None),
        page_size: int = Query(500, ge=1, le=5000),
        tenant: tuple[str, str] | None = Depends(_verify_api_key),
        purpose: str | None = Depends(_request_purpose),
    ):
        """One page of the user's export (mm-export/1).  Pass the returned
        ``next_cursor`` to fetch the next page; every page is authorized
        again, so a revoked grant stops the export at the next request."""
        from mirror_memory.core.export import InvalidCursor

        engine = _get_engine(tenant)
        try:
            with access_audit.entry_label("mirror_memory.server.export"):
                page = engine.export_page(
                    user_id=user_id, purpose=purpose, cursor=cursor, page_size=page_size,
                )
            return page.to_dict()
        except PurposeDenied as e:
            raise HTTPException(status_code=403, detail=str(e))
        except (InvalidCursor, ValidationError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.warning("export failed for user %s: %s", user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.get("/stats")
    def stats(
        user_id: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=100),
        tenant: tuple[str, str] | None = Depends(_verify_api_key),
        purpose: str | None = Depends(_request_purpose),
    ):
        """Get extraction statistics for a user."""
        engine = _get_engine(tenant)
        try:
            with engine._session() as session:
                authorized = engine._authorize(
                    session, user_id, purpose, "explain", audit_entry="mirror_memory.server.stats",
                )
                from mirror_memory.core.models import ExtractionStats

                rows = (
                    session.query(ExtractionStats)
                    .filter(ExtractionStats.user_id == user_id)
                    .order_by(ExtractionStats.created_at.desc())
                    .limit(limit)
                    .all()
                )
                # extraction stats are telemetry, not memory items: the
                # access is recorded with no memory ids
                engine._audit(session, authorized, user_id=user_id, operation="explain",
                              entry="mirror_memory.server.stats", memory_ids=[])
                session.commit()
                return {
                    "user_id": user_id,
                    "stats": [
                        {
                            "trigger": r.trigger,
                            "model": r.model,
                            "claims_out": r.claims_out,
                            "latency_ms": r.latency_ms,
                            "status": r.status,
                            "error": r.error,
                            "created_at": r.created_at.isoformat() if r.created_at else None,
                        }
                        for r in rows
                    ],
                }
        except PurposeDenied as e:
            raise HTTPException(status_code=403, detail=str(e))
        except Exception as e:
            logger.warning("stats failed for user %s: %s", user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    return app


# Default app instance (for `uvicorn mirror_memory.server:app`)
app = create_app(
    config_path=os.environ.get("MM_CONFIG_PATH", "config/"),
    database_url=os.environ.get("MM_DATABASE_URL", "sqlite:///mirror_memory.db"),
)
