"""MemoryEngine — the public API for mirror-memory.

Usage::

    from mirror_memory import MemoryEngine

    # One-liner with built-in LLM adapter:
    engine = MemoryEngine(
        config_path="config/",
        llm_api_key="sk-xxx",
        llm_model="deepseek-chat",
        llm_base_url="https://api.deepseek.com",
    )

    engine.observe(user_id="u1", session_id="s1", text="I love painting")
    context = engine.recall(user_id="u1", query="What are the user's hobbies?")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from mirror_memory.config.loader import load_config
from mirror_memory.config.schema import MemoryConfig
from mirror_memory.exceptions import ValidationError
from mirror_memory.models import BeliefInfo, DeleteResult, PanelData

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Structured memory engine for AI agents.

    Wraps extraction, storage, retrieval, and rendering into a single API.
    All domain-specific content is loaded from configuration files.

    Parameters
    ----------
    config_path:
        Path to the configuration directory.
    config:
        Pre-loaded ``MemoryConfig``.  Takes precedence over *config_path*.
    database_url:
        SQLAlchemy database URL.
    llm_client:
        Pre-built LLM client (must expose ``generate()``).
        Takes precedence over ``llm_api_key``/``llm_model``/``llm_base_url``.
    llm_api_key:
        API key for the built-in OpenAI-compatible LLM adapter.
    llm_model:
        Model identifier (e.g. ``"gpt-4o-mini"``, ``"deepseek-chat"``).
    llm_base_url:
        API base URL for non-OpenAI providers.
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        config: MemoryConfig | None = None,
        database_url: str = "sqlite:///mirror_memory.db",
        llm_client: Any | None = None,
        llm_api_key: str | None = None,
        llm_model: str = "gpt-4o-mini",
        llm_base_url: str | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        elif config_path is not None:
            self._config = load_config(config_path)
        else:
            raise ValueError("Provide either config_path or config")

        self._database_url = database_url
        self._llm_client = llm_client or self._build_llm(llm_api_key, llm_model, llm_base_url)
        self._initialized = False
        self._pipeline = None  # lazy-initialized

    @staticmethod
    def _build_llm(api_key: str | None, model: str, base_url: str | None) -> Any:
        """Build the built-in OpenAI-compatible LLM adapter."""
        if api_key is None:
            return None
        from mirror_memory.llm import OpenAILLM

        return OpenAILLM(api_key=api_key, model=model, base_url=base_url)

    def _ensure_db(self) -> None:
        """Create tables on first use."""
        if self._initialized:
            return
        from sqlalchemy import create_engine

        from mirror_memory.core.models import Base

        self._engine = create_engine(self._database_url)
        Base.metadata.create_all(self._engine)
        self._initialized = True

    def _session(self):
        """Create a new database session."""
        from sqlalchemy.orm import Session

        return Session(self._engine)

    def _get_pipeline(self):
        """Lazy-initialize the extraction pipeline (singleton per engine)."""
        if self._pipeline is None:
            from mirror_memory.extraction.pipeline import ExtractionPipeline

            self._pipeline = ExtractionPipeline(config=self._config, llm_client=self._llm_client)
        return self._pipeline

    def observe(
        self,
        *,
        user_id: str,
        session_id: str,
        text: str,
        turn_count: int = 1,
        **kwargs: Any,
    ) -> int:
        """Process a single conversation turn — extract and store beliefs.

        Fail-open: extraction errors never block the caller.

        Returns
        -------
        int
            The number of claims extracted this turn.

        Raises
        ------
        ValidationError
            If user_id, session_id, or text is empty, or text exceeds
            the maximum length.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if not session_id or not session_id.strip():
            raise ValidationError("session_id must be a non-empty string")
        if not text or not text.strip():
            raise ValidationError("text must be a non-empty string")
        if len(text) > 50000:
            raise ValidationError("text exceeds maximum length of 50000 characters")

        self._ensure_db()
        pipeline = self._get_pipeline()
        started = time.monotonic()
        with self._session() as session:
            claims = pipeline.observe(
                session,
                user_id=user_id,
                session_id=session_id,
                text=text,
                turn_count=turn_count,
                **kwargs,
            )
            session.commit()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "observe: user=%s session=%s claims=%d latency=%dms",
            user_id[:8], session_id[:16], len(claims), elapsed_ms,
        )
        return len(claims)

    def recall(
        self,
        *,
        user_id: str,
        query: str = "",
        language: str = "en",
        tail_load: int = 0,
    ) -> str | None:
        """Retrieve a rendered memory block for prompt injection.

        Returns
        -------
        str or None
            The rendered memory block, or ``None`` if no memory is
            available.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.render.renderer import render_memory_block

        started = time.monotonic()
        with self._session() as session:
            result = render_memory_block(
                session,
                user_id,
                config=self._config,
                language=language,
                user_message=query,
                tail_load=tail_load,
            )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "recall: user=%s chars=%d latency=%dms",
            user_id[:8], len(result or ""), elapsed_ms,
        )
        return result

    def panel(
        self,
        *,
        user_id: str,
        language: str = "en",
    ) -> PanelData:
        """Build a structured panel of the user's memory for frontend display.

        Returns
        -------
        PanelData
            Structured panel with memory status, sections, and avatar info.

        Raises
        ------
        ValidationError
            If user_id is empty or language is invalid.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if language not in ("zh", "en"):
            raise ValidationError(f"language must be 'zh' or 'en', got '{language}'")

        self._ensure_db()
        from mirror_memory.render.panel import build_panel

        with self._session() as session:
            raw = build_panel(session, user_id, config=self._config, language=language)
            return PanelData(
                memory=raw.get("memory", {}),
                sections=raw.get("sections", []),
                avatar=raw.get("avatar", {}),
            )

    def get_beliefs(
        self,
        *,
        user_id: str,
        dimensions: list[str] | None = None,
        limit: int = 50,
    ) -> list[BeliefInfo]:
        """List active beliefs for a user.

        Returns
        -------
        list[BeliefInfo]
            Active beliefs sorted by last evidence time.

        Raises
        ------
        ValidationError
            If user_id is empty or limit is out of range.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if limit < 1 or limit > 200:
            raise ValidationError("limit must be between 1 and 200")

        self._ensure_db()
        from mirror_memory.core.repository import list_active_beliefs

        with self._session() as session:
            beliefs = list_active_beliefs(session, user_id, dimensions=dimensions, limit=limit)
            return [
                BeliefInfo(
                    dimension=b.dimension,
                    key=b.key,
                    claim_text=b.claim_text,
                    confidence=b.confidence,
                    layer=b.layer,
                    source=b.source,
                    last_evidence_at=b.last_evidence_at,
                )
                for b in beliefs
            ]

    def get_verification_candidates(self, *, user_id: str, limit: int = 1, session_id: str = "") -> list[dict]:
        """Get beliefs eligible for verification questioning.

        Selects L4 beliefs (confidence >= question threshold, cross-session
        evidence) sorted by ``question_value_tiers * confirm_rate_weight``,
        and records a ``question_injected`` intervention event for the top
        candidate so that :meth:`run_verification` can judge the user's reply
        on a later turn.  A new question is never injected while a previous
        one is still unanswered.

        Parameters
        ----------
        user_id:
            The user identifier.
        limit:
            Maximum number of candidates to return.
        session_id:
            Optional session the question is surfaced in (recorded on the
            injection event for the pending-verification window).

        Returns
        -------
        list[dict]
            Candidate dicts with ``belief_id``, ``dimension``, ``key``,
            ``claim_text``, ``confidence``, ``question`` and ``injected``.
            Empty when the loop is disabled (no LLM client or empty
            ``verification`` prompt) or on any error -- fail-open.

        Raises
        ------
        ValidationError
            If user_id is empty or limit is out of range.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if limit < 1 or limit > 200:
            raise ValidationError("limit must be between 1 and 200")
        if self._llm_client is None or not (self._config.prompts.verification or "").strip():
            logger.info("verification: disabled (no LLM client or empty verification prompt)")
            return []

        self._ensure_db()
        from mirror_memory.extraction.verification import (
            build_verification_question,
            get_question_candidates,
            record_question_injection,
        )

        with self._session() as session:
            try:
                beliefs = get_question_candidates(session, user_id, self._config, limit=limit)
                results: list[dict] = []
                injected = False
                for belief in beliefs:
                    recorded = False
                    if not injected:
                        event = record_question_injection(
                            session, user_id, session_id, belief, self._config
                        )
                        if event is not None:
                            recorded = True
                            injected = True
                    results.append(
                        {
                            "belief_id": int(belief.id),
                            "dimension": belief.dimension,
                            "key": belief.key,
                            "claim_text": belief.claim_text,
                            "confidence": float(belief.confidence),
                            "question": build_verification_question(belief, self._config),
                            "injected": recorded,
                        }
                    )
                session.commit()
                return results
            except Exception:
                logger.warning("get_verification_candidates failed; returning []", exc_info=True)
                session.rollback()
                return []

    def run_verification(self, *, user_id: str, session_id: str, user_text: str) -> bool:
        """Run verification judgment for this turn. Returns True if turn consumed.

        Judges the user's reply against the pending verification question
        (if any) and drives the belief lifecycle: confirm -> L4 -> L2,
        deny -> rejected, unclear -> event only.  Fail-open: errors never
        propagate; the caller should proceed with normal extraction when
        ``False`` is returned.

        Raises
        ------
        ValidationError
            If user_id or session_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if not session_id or not session_id.strip():
            raise ValidationError("session_id must be a non-empty string")
        if not user_text or not user_text.strip():
            logger.debug("run_verification: empty user_text -- nothing to judge")
            return False

        self._ensure_db()
        from mirror_memory.extraction.verification import run_verification_judgment

        with self._session() as session:
            try:
                consumed = run_verification_judgment(
                    session,
                    user_id,
                    session_id,
                    user_text,
                    self._config,
                    self._llm_client,
                )
                session.commit()
                return consumed
            except Exception:
                logger.warning("run_verification failed; fail-open", exc_info=True)
                session.rollback()
                return False

    def set_enabled(self, *, user_id: str, enabled: bool) -> None:
        """Enable or disable memory for a user.

        Returns
        -------
        None

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import set_memory_enabled

        with self._session() as session:
            set_memory_enabled(session, user_id, enabled)
            session.commit()

    def set_consent(self, *, user_id: str, feature: str, granted: bool) -> None:
        """Set feature-level consent for a user.

        Parameters
        ----------
        user_id:
            The user identifier.
        feature:
            Feature name (e.g. ``"extraction"``, ``"recall"``, ``"verification"``).
        granted:
            Whether the feature is allowed.

        Raises
        ------
        ValidationError
            If user_id or feature is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")
        if not feature or not feature.strip():
            raise ValidationError("feature must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import set_feature_consent

        with self._session() as session:
            set_feature_consent(session, user_id, feature.strip(), granted)
            session.commit()

    def delete_memories(self, *, user_id: str) -> DeleteResult:
        """Delete all memory data for a user.

        Returns
        -------
        DeleteResult
            Per-table deletion counts.

        Raises
        ------
        ValidationError
            If user_id is empty.
        """
        if not user_id or not user_id.strip():
            raise ValidationError("user_id must be a non-empty string")

        self._ensure_db()
        from mirror_memory.core.repository import delete_user_memories

        with self._session() as session:
            counts = delete_user_memories(session, user_id)
            session.commit()
            return DeleteResult(**counts)
