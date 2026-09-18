"""Extraction pipeline -- main entry point.

Orchestrates K1 (deterministic) and K2 (LLM semantic) extraction.
All domain content comes from ``config``.  Fail-open: any exception
is logged but never blocks the caller.
"""

from __future__ import annotations

import logging

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import (
    SESSION_SUMMARY_MAX_LENGTH,
    SNIPPET_MAX_LENGTH,
    SNIPPET_MIN_LENGTH,
)
from mirror_memory.extraction.deterministic import extract_claims
from mirror_memory.extraction.semantic import SemanticExtractor
from mirror_memory.extraction.throttle import should_extract

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Main extraction pipeline.

    Call ``observe`` once per user turn.  Internally runs:
    1. K1 deterministic keyword/regex extraction.
    2. K2 LLM semantic extraction (throttled).
    3. Statistics recording.

    All claims are persisted via the repository layer.  Any failure
    is caught and logged -- the pipeline never raises.

    Parameters
    ----------
    config:
        The ``MemoryConfig`` providing all extraction parameters,
        prompts, anchors, and the LLM client.
    llm_client:
        Optional override for the LLM client.  If provided, takes
        precedence over ``config.llm_client``.
    """

    def __init__(self, config: MemoryConfig, llm_client: object | None = None) -> None:
        self._config = config
        self._llm_client = llm_client or config.llm_client
        # Build a SemanticExtractor; it reads allowed keys from config.
        self._semantic = SemanticExtractor(config)

    def observe(
        self,
        session: object,
        user_id: str,
        session_id: str,
        text: str,
        turn_count: int,
        **kwargs: object,
    ) -> list[dict]:
        """Run extraction for one conversation turn.

        Parameters
        ----------
        session:
            SQLAlchemy database session.
        user_id:
            The user identifier.
        session_id:
            The conversation session identifier.
        text:
            The user's message text.
        turn_count:
            The current turn number (0-indexed).
        **kwargs:
            Additional context (unused by default; available for
            domain-specific extensions).

        Returns
        -------
        list[dict]
            All claims extracted this turn (K1 + K2 combined).
        """
        all_claims: list[dict] = []

        # -- K1: Deterministic extraction ------------------------------------
        try:
            k1_claims = extract_claims(text, self._config)
            all_claims.extend(k1_claims)
            logger.info("pipeline: K1 extracted %d claims", len(k1_claims))
        except Exception:
            logger.warning("pipeline: K1 extraction failed; continuing", exc_info=True)

        # -- K2: LLM semantic extraction (throttled) -------------------------
        try:
            if self._should_run_k2(text, turn_count):
                # Temporarily override llm_client if provided at init
                effective_config = self._config
                if self._llm_client is not self._config.llm_client:
                    effective_config = self._config.model_copy(
                        update={"llm_client": self._llm_client}
                    )
                k2_claims = self._semantic.extract(text, session, user_id, effective_config)
                all_claims.extend(k2_claims)
                logger.info("pipeline: K2 extracted %d claims", len(k2_claims))
        except Exception:
            logger.warning("pipeline: K2 extraction failed; continuing", exc_info=True)

        # -- Persist claims --------------------------------------------------
        try:
            self._persist_claims(session, user_id, session_id, all_claims, **kwargs)
        except Exception:
            logger.warning("pipeline: claim persistence failed", exc_info=True)

        # -- Store conversation snippet for fallback retrieval ---------------
        if self._config.session_summary_enabled:
            try:
                self._store_snippet(session, user_id, session_id, text, turn_count)
            except Exception:
                logger.warning("pipeline: snippet storage failed", exc_info=True)

        return all_claims

    def _should_run_k2(self, text: str, turn_count: int) -> bool:
        """Check throttle for K2 extraction."""
        return should_extract(text, turn_count, self._config)

    def _store_snippet(
        self,
        session: object,
        user_id: str,
        session_id: str,
        text: str,
        turn_count: int,
    ) -> None:
        """Store conversation snippet via repository layer."""
        if not text or len(text.strip()) < SNIPPET_MIN_LENGTH:
            return
        from mirror_memory.core.repository import store_session_summary

        store_session_summary(
            session, user_id, session_id, text, turn_count,
            max_length=SESSION_SUMMARY_MAX_LENGTH,
            snippet_max=SNIPPET_MAX_LENGTH,
        )

    def _persist_claims(
        self,
        session: object,
        user_id: str,
        session_id: str,
        claims: list[dict],
        **kwargs: object,
    ) -> None:
        """Persist extracted claims via the repository layer.

        Uses lazy import to avoid circular dependency with
        ``core.repository``.
        """
        if not claims:
            return

        from mirror_memory.core.repository import record_claim

        for claim in claims:
            try:
                record_claim(
                    session,
                    user_id,
                    dimension=claim.get("dimension", ""),
                    key=claim.get("key", ""),
                    claim_text=claim.get("claim_text", ""),
                    value=claim.get("value"),
                    relation=claim.get("relation", "supports"),
                    confidence=claim.get("confidence", 0.0),
                    layer=claim.get("layer", "L4"),
                    source=claim.get("source", "extracted"),
                    session_id=session_id,
                    evidence_message_ids=claim.get("evidence_message_ids"),
                    blocked_key_prefixes=self._config.blocked_key_prefixes,
                )
            except Exception:
                logger.warning(
                    "pipeline: failed to persist claim: dim=%s key=%s",
                    claim.get("dimension"),
                    claim.get("key"),
                    exc_info=True,
                )
