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

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import text

from mirror_memory import __version__
from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.config.schema import MemoryConfig
from mirror_memory.exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

_security = HTTPBearer(auto_error=False)


def _verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_security),
) -> None:
    """Verify API key if MM_API_KEY is set."""
    expected = os.environ.get("MM_API_KEY")
    if expected is None:
        return  # no auth required
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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


def _get_engine() -> MemoryEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized. Call create_app() first.")
    return _engine


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
        logger.info("mirror-memory engine initialized")
        yield
        _engine = None

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
    def observe(req: ObserveRequest, _: None = Depends(_verify_api_key)):
        """Ingest a single conversation turn into memory."""
        engine = _get_engine()
        try:
            count = engine.observe(
                user_id=req.user_id,
                session_id=req.session_id,
                text=req.text,
                turn_count=req.turn_count,
            )
            return ObserveResponse(status="ok", claims_extracted=count)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except LLMError as e:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except Exception as e:
            logger.warning("observe failed for user %s: %s", req.user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.post("/recall", response_model=RecallResponse)
    def recall(req: RecallRequest, _: None = Depends(_verify_api_key)):
        """Retrieve memory context for a query."""
        engine = _get_engine()
        try:
            ctx = engine.recall(
                user_id=req.user_id,
                query=req.query,
                language=req.language,
            )
            return RecallResponse(
                user_id=req.user_id,
                context=ctx,
                char_count=len(ctx or ""),
            )
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
        _: None = Depends(_verify_api_key),
    ):
        """Get structured panel data for frontend display."""
        engine = _get_engine()
        try:
            data = engine.panel(user_id=user_id, language=language)
            return PanelResponse(
                user_id=user_id,
                panel={"memory": data.memory, "sections": data.sections, "avatar": data.avatar},
            )
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
        _: None = Depends(_verify_api_key),
    ):
        """List active beliefs for a user."""
        engine = _get_engine()
        try:
            raw = engine.get_beliefs(user_id=user_id, limit=limit)
            items = [
                BeliefItem(
                    dimension=b.dimension, key=b.key, claim_text=b.claim_text,
                    confidence=b.confidence, layer=b.layer, source=b.source,
                    last_evidence_at=b.last_evidence_at.isoformat() if b.last_evidence_at else None,
                )
                for b in raw
            ]
            return BeliefsResponse(user_id=user_id, beliefs=items, total=len(items))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except LLMError as e:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except Exception as e:
            logger.warning("beliefs failed for user %s: %s", user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.post("/delete", response_model=DeleteResponse)
    def delete_memories(req: DeleteRequest, _: None = Depends(_verify_api_key)):
        """Delete all memory data for a user."""
        engine = _get_engine()
        try:
            result = engine.delete_memories(user_id=req.user_id)
            return DeleteResponse(status="deleted", counts=result.as_dict())
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except LLMError as e:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except Exception as e:
            logger.warning("delete failed for user %s: %s", req.user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    @app.get("/stats")
    def stats(
        user_id: str = Query(..., min_length=1),
        limit: int = Query(20, ge=1, le=100),
        _: None = Depends(_verify_api_key),
    ):
        """Get extraction statistics for a user."""
        engine = _get_engine()
        try:
            with engine._session() as session:
                from mirror_memory.core.models import ExtractionStats

                rows = (
                    session.query(ExtractionStats)
                    .filter(ExtractionStats.user_id == user_id)
                    .order_by(ExtractionStats.created_at.desc())
                    .limit(limit)
                    .all()
                )
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
        except Exception as e:
            logger.warning("stats failed for user %s: %s", user_id[:8], e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    return app


# Default app instance (for `uvicorn mirror_memory.server:app`)
app = create_app(
    config_path=os.environ.get("MM_CONFIG_PATH", "config/"),
    database_url=os.environ.get("MM_DATABASE_URL", "sqlite:///mirror_memory.db"),
)
