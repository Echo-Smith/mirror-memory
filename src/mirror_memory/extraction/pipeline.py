"""Extraction pipeline -- main entry point.

Orchestrates the verification judgment half-loop, K1 (deterministic) and
K2 (LLM semantic) extraction.  All domain content comes from ``config``.
Fail-open: any exception is logged but never blocks the caller.
"""

from __future__ import annotations

import logging

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import (
    SESSION_SUMMARY_MAX_LENGTH,
    SNIPPET_MAX_LENGTH,
    SNIPPET_MIN_LENGTH,
)
from mirror_memory.core.utils import safe_json
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
        *,
        context: str = "",
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

        # -- Verification half-loop (question -> answer) ----------------------
        # If a previous turn injected a verification question and this turn
        # answers it, the judgment consumes the turn: its signal belongs to
        # the hypothesis, not to new topics, so K1/K2 extraction is skipped.
        try:
            if self._run_verification(session, user_id, session_id, text):
                logger.info("pipeline: turn consumed by verification judgment")
                return all_claims
        except Exception:
            logger.warning("pipeline: verification failed; continuing", exc_info=True)

        # -- Context suppression check -----------------------------------------
        # If a suppression rule matches the current context, skip extraction
        # for the suppressed dimensions (or entirely if no dimensions listed).
        if context and self._is_suppressed(context):
            logger.info("pipeline: extraction suppressed in context=%s", context)
            return all_claims

        # -- K1: Deterministic extraction ------------------------------------
        try:
            k1_claims = extract_claims(text, self._config)
            all_claims.extend(k1_claims)
            logger.info("pipeline: K1 extracted %d claims", len(k1_claims))
        except Exception:
            logger.warning("pipeline: K1 extraction failed; continuing", exc_info=True)

        # -- K2: LLM semantic extraction (throttled) -------------------------
        k2_context_tags: list[str] = []
        try:
            if self._should_run_k2(text, turn_count, session, user_id):
                # Temporarily override llm_client if provided at init
                effective_config = self._config
                if self._llm_client is not self._config.llm_client:
                    effective_config = self._config.model_copy(
                        update={"llm_client": self._llm_client}
                    )
                k2_result = self._semantic.extract(text, session, user_id, effective_config)
                if isinstance(k2_result, dict):
                    # New structured format: claims + subject + context_tags.
                    subject = k2_result.get("subject", "user")
                    k2_context_tags = k2_result.get("context_tags", [])
                    if subject == "third_party":
                        # Claims about someone else are not stored as user beliefs.
                        logger.info("pipeline: K2 claims skipped (subject=third_party)")
                    else:
                        all_claims.extend(k2_result["claims"])
                        logger.info("pipeline: K2 extracted %d claims", len(k2_result["claims"]))
                elif isinstance(k2_result, list):
                    # Legacy list format from _parse_extraction_json.
                    all_claims.extend(k2_result)
                    logger.info("pipeline: K2 extracted %d claims", len(k2_result))
        except Exception:
            logger.warning("pipeline: K2 extraction failed; continuing", exc_info=True)

        # -- Persist claims --------------------------------------------------
        try:
            # Assemble: scarce dimensions first, cap fill.
            from mirror_memory.extraction.assembler import assemble_claims

            dim_counts = self._get_dimension_counts(session, user_id)
            all_claims = assemble_claims(all_claims, self._config, active_dimension_counts=dim_counts)
            self._persist_claims(
                session, user_id, session_id, all_claims,
                context_tags=k2_context_tags or None,
                **kwargs,
            )
        except Exception:
            logger.warning("pipeline: claim persistence failed", exc_info=True)

        # -- Store conversation snippet for fallback retrieval ---------------
        if self._config.session_summary_enabled:
            try:
                self._store_snippet(session, user_id, session_id, text, turn_count)
            except Exception:
                logger.warning("pipeline: snippet storage failed", exc_info=True)

        return all_claims

    def _is_suppressed(self, context: str) -> bool:
        """Check if extraction is suppressed in the given context."""
        for rule in self._config.extraction.suppression_rules:
            if rule.context == context:
                # Empty suppress_dimensions = suppress all extraction.
                return len(rule.suppress_dimensions) == 0
        return False

    def _get_dimension_counts(self, session: object, user_id: str) -> dict[str, int]:
        """Get count of active beliefs per dimension (for scarcity scoring)."""
        from collections import Counter

        from mirror_memory.core.models import Belief

        try:
            rows = (
                session.query(Belief.dimension)
                .filter(Belief.user_id == user_id, Belief.status == "active")
                .all()
            )
            return dict(Counter(r[0] for r in rows))
        except Exception:
            return {}

    def _should_run_k2(self, text: str, turn_count: int, session: object, user_id: str) -> bool:
        """Check throttle for K2 extraction (base rules + information-gain gates)."""
        return should_extract(text, turn_count, self._config, session=session, user_id=user_id)

    def _run_verification(self, session: object, user_id: str, session_id: str, text: str) -> bool:
        """Run the verification judgment half-loop (no-op when disabled)."""
        from mirror_memory.extraction.verification import run_verification_judgment

        return run_verification_judgment(
            session,
            user_id,
            session_id,
            text,
            self._config,
            self._llm_client,
        )

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
        *,
        context_tags: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        """Persist extracted claims via the repository layer.

        When a claim has predicate/object in its value, the IdentityResolver
        decides the action (CREATE/SUPPORT/UPDATE) before persistence.
        """
        if not claims:
            return

        from mirror_memory.core.repository import list_active_beliefs, record_claim
        from mirror_memory.memory.atom import ACTION_CONTRADICT, ACTION_SUPPORT, ACTION_UPDATE, CandidateAtom
        from mirror_memory.memory.canonicalize import canonicalize_atom
        from mirror_memory.memory.identity import resolve_identity

        allowed = (
            {d.dimension_id for d in self._config.dimensions}
            if self._config.strict_dimensions
            else None
        )
        policy = self._config.identity_policy
        synonyms = self._config.predicate_synonyms

        for claim in claims:
            try:
                value = claim.get("value") or {}
                pred = value.get("predicate", "")
                obj = value.get("object", "")
                canon_pred = pred
                canon_obj = obj

                # If claim has cognitive triple fields, run identity resolution.
                if pred and obj and policy:
                    # Canonicalize
                    _, canon_pred, canon_obj = canonicalize_atom(
                        claim.get("subject", "user"), pred, obj, synonyms
                    )

                    # Build candidate
                    candidate = CandidateAtom(
                        subject=claim.get("subject", "user"),
                        predicate=canon_pred,
                        object=canon_obj,
                        dimension=claim.get("dimension", ""),
                        claim_text=claim.get("claim_text", ""),
                        confidence=claim.get("confidence", 0.0),
                        context_tags=context_tags or [],
                        temporal=value.get("temporal", ""),
                        relation=claim.get("relation", "supports"),
                    )

                    # Get existing beliefs for this user
                    existing = list_active_beliefs(session, user_id, limit=200)
                    existing_dicts = [
                        {
                            "id": b.id,
                            "predicate": getattr(b, "predicate", ""),
                            "object": getattr(b, "object", ""),
                            "status": b.status,
                            "confidence": b.confidence,
                            "temporal": safe_json(getattr(b, "value_json", "{}")).get("temporal", ""),
                        }
                        for b in existing
                    ]

                    resolution = resolve_identity(candidate, existing_dicts, policy)
                    logger.info(
                        "pipeline: identity resolution: action=%s pred=%s obj=%s reason=%s",
                        resolution.action, canon_pred, canon_obj, resolution.reason,
                    )

                    # Dispatch based on resolver action.
                    if resolution.action == ACTION_SUPPORT and resolution.target_belief_id:
                        from mirror_memory.core.repository import support_belief_by_id

                        support_belief_by_id(
                            session,
                            resolution.target_belief_id,
                            claim_text=claim.get("claim_text", ""),
                            session_id=session_id,
                            evidence_message_ids=claim.get("evidence_message_ids"),
                        )
                        continue  # SUPPORT handled, skip record_claim

                    if resolution.action == ACTION_UPDATE and resolution.target_belief_id:
                        from mirror_memory.core.repository import update_belief_by_id

                        # Preserve metadata through the update.
                        value["predicate"] = canon_pred
                        value["object"] = canon_obj
                        claim["value"] = value

                        old, new = update_belief_by_id(
                            session,
                            resolution.target_belief_id,
                            new_subject=claim.get("subject", "user"),
                            new_predicate=canon_pred,
                            new_object=canon_obj,
                            new_dimension=claim.get("dimension", ""),
                            new_key=claim.get("key", ""),
                            new_claim_text=claim.get("claim_text", ""),
                            new_confidence=claim.get("confidence", 0.0),
                            new_cardinality=policy.get(canon_pred, "multi"),
                            session_id=session_id,
                            evidence_message_ids=claim.get("evidence_message_ids"),
                            new_value=value,
                        )
                        if new:
                            logger.info("pipeline: UPDATE %s -> %s", old.id if old else "?", new.id)
                        continue  # UPDATE handled, skip record_claim

                    if resolution.action == ACTION_CONTRADICT and resolution.target_belief_id:
                        # CONTRADICT: use target belief's key so record_claim
                        # finds the right belief to attenuate.
                        from mirror_memory.core.models import Belief as _Belief

                        target = session.get(_Belief, resolution.target_belief_id)
                        if target:
                            claim["key"] = target.key
                        claim["relation"] = "contradicts"

                    # Update claim value with canonical triple.
                    value["predicate"] = canon_pred
                    value["object"] = canon_obj
                    if value.get("temporal"):
                        value["temporal"] = value["temporal"]
                    claim["value"] = value

                # CREATE / CONTRADICT: persist via record_claim.
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
                    allowed_dimensions=allowed,
                    context_tags=context_tags,
                    # Triple fields.
                    subject=claim.get("subject", "user"),
                    predicate=canon_pred,
                    object=canon_obj,
                    cardinality=policy.get(canon_pred, "multi"),
                )
            except Exception:
                logger.warning(
                    "pipeline: failed to persist claim: dim=%s key=%s",
                    claim.get("dimension"),
                    claim.get("key"),
                    exc_info=True,
                )
