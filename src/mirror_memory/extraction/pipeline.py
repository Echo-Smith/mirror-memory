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
from mirror_memory.core.proposal import (
    TRANSITION_CONTRADICT,
    TRANSITION_CREATE,
    TRANSITION_SUPPORT,
    TRANSITION_UPDATE,
)
from mirror_memory.core.utils import safe_json
from mirror_memory.extraction.deterministic import extract_claims
from mirror_memory.extraction.semantic import SemanticExtractor
from mirror_memory.extraction.throttle import should_extract

logger = logging.getLogger(__name__)


_RETURN_PHRASINGS = (
    "went back to",
    "going back to",
    "go back to",
    "returned to",
    "returning to",
    "return to",
    "moved back to",
    "moving back to",
    "switched back to",
    "switching back to",
    "transferred back to",
    "back at ",
    "am back at",
    "started back at",
    "enrolled again at",
    "rejoined",
)


def _is_return_phrasing(claim_text: str) -> bool:
    """Does this claim say the user is back at a value they held before?

    Matched on the claim text because the extractor folds the marker into
    the predicate (``went_to``), where it is indistinguishable from a plain
    visit.  The identity resolver's revival path then decides whether the
    object actually belongs to a single-valued slot.
    """
    lowered = (claim_text or "").lower()
    return any(marker in lowered for marker in _RETURN_PHRASINGS)


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

    def __init__(
        self,
        config: MemoryConfig,
        llm_client: object | None = None,
        trace_hook: object | None = None,
    ) -> None:
        self._config = config
        self._llm_client = llm_client or config.llm_client
        # Build a SemanticExtractor; it reads allowed keys from config.
        self._semantic = SemanticExtractor(config)
        # Optional observability hook: called as ``trace_hook(stage, **fields)``
        # at each funnel stage.  Never affects behaviour -- purely additive.
        self._trace_hook = trace_hook
        # K2 health, read off the extractor that actually makes the calls: a
        # failed K2 call is swallowed so the turn still yields K1 claims, which
        # means a run whose every K2 call fails looks exactly like a
        # deterministic run -- the caller must be able to tell the difference,
        # or a rate-limited "forced" run silently reports K1 numbers.
        self._k2_stage_failures = 0

    @property
    def k2_attempts(self) -> int:
        return getattr(self._semantic, "llm_calls", 0)

    @property
    def k2_failures(self) -> int:
        return getattr(self._semantic, "llm_failures", 0) + self._k2_stage_failures

    @property
    def k2_health(self) -> dict[str, int]:
        """Full K2 health: attempts, failures, and why claims came back empty."""
        semantic = self._semantic
        return {
            "k2_attempts": self.k2_attempts,
            "k2_failures": self.k2_failures,
            "k2_empty_responses": getattr(semantic, "empty_responses", 0),
            "k2_parse_failures": getattr(semantic, "parse_failures", 0),
            "k2_schema_rejections": getattr(semantic, "schema_rejections", 0),
        }

    def _trace(self, stage: str, **fields: object) -> None:
        """Emit one funnel-stage event to the observability hook, if any."""
        if self._trace_hook is None:
            return
        try:
            self._trace_hook(stage, **fields)
        except Exception:
            logger.debug("pipeline: trace hook failed at stage=%s", stage, exc_info=True)

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
            self._trace(
                "extraction",
                source="k1",
                count=len(k1_claims),
                predicates=[(c.get("value") or {}).get("predicate", "") for c in k1_claims],
                objects=[(c.get("value") or {}).get("object", "") for c in k1_claims],
                claim_texts=[c.get("claim_text", "") for c in k1_claims],
            )
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
                        self._trace(
                            "extraction",
                            source="k2",
                            count=len(k2_result["claims"]),
                            predicates=[
                                (c.get("value") or {}).get("predicate", "")
                                for c in k2_result["claims"]
                            ],
                            objects=[
                                (c.get("value") or {}).get("object", "")
                                for c in k2_result["claims"]
                            ],
                            claim_texts=[
                                c.get("claim_text", "") for c in k2_result["claims"]
                            ],
                        )
                elif isinstance(k2_result, list):
                    # Legacy list format from _parse_extraction_json.
                    all_claims.extend(k2_result)
                    logger.info("pipeline: K2 extracted %d claims", len(k2_result))
        except Exception:
            self._k2_stage_failures += 1
            logger.warning("pipeline: K2 extraction failed; continuing", exc_info=True)

        # -- Persist claims --------------------------------------------------
        try:
            # Assemble: scarce dimensions first, cap fill.
            from mirror_memory.extraction.assembler import arbitrate_claims, assemble_claims

            dim_counts = self._get_dimension_counts(session, user_id)
            all_claims = assemble_claims(all_claims, self._config, active_dimension_counts=dim_counts)
            # Arbitrate same-turn K1/K2 disagreements before anything is
            # written: once both readings are beliefs, the cross-predicate
            # revival path can close the correct one.
            all_claims = arbitrate_claims(all_claims, self._config.identity_policy)
            self._persist_claims(
                session, user_id, session_id, all_claims,
                context_tags=k2_context_tags or None,
                source_text=text,
                turn_ref=f"{session_id}:{turn_count}",
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

    def _ensure_evidence_rows(
        self,
        session: object,
        user_id: str,
        session_id: str,
        claim: dict,
        *,
        source_text: str = "",
        turn_ref: str = "",
    ) -> list[int]:
        """Record a claim's source messages as Evidence rows; return their ids.

        Split out of :meth:`_attach_evidence_rows` so the ids exist *before*
        a proposal is published — the Publisher's authority gate needs the
        evidence to already belong to the user and the session the proposal
        claims.
        """
        from mirror_memory.core.repository import record_evidence

        refs = [str(mid) for mid in (claim.get("evidence_message_ids") or [])]
        if not refs and turn_ref:
            refs = [turn_ref]
        if not refs:
            return []

        method = claim.get("source") or "extracted"
        ids: list[int] = []
        for ref in refs:
            try:
                evidence = record_evidence(
                    session,
                    user_id,
                    ref=ref,
                    content=source_text,
                    session_id=session_id,
                    source_type="message",
                    extraction_method=method,
                    authority="user",
                )
                ids.append(evidence.id)
            except Exception:
                logger.debug(
                    "pipeline: evidence row failed for ref=%s", ref, exc_info=True,
                )
        return ids

    @staticmethod
    def _link_evidence_rows(
        session: object,
        belief_id: int | None,
        evidence_ids: list[int],
        *,
        relation: str,
    ) -> None:
        """Link already-created Evidence rows to a belief with a typed relation."""
        if belief_id is None or not evidence_ids:
            return
        from mirror_memory.core.repository import link_evidence

        for evidence_id in evidence_ids:
            try:
                link_evidence(session, belief_id, evidence_id, relation=relation)
            except Exception:
                logger.debug(
                    "pipeline: evidence link failed for belief=%s evidence=%s",
                    belief_id, evidence_id, exc_info=True,
                )

    def _attach_evidence_rows(
        self,
        session: object,
        user_id: str,
        session_id: str,
        claim: dict,
        belief_id: int | None,
        *,
        relation: str,
        source_text: str = "",
        turn_ref: str = "",
    ) -> None:
        """Record a claim's source messages as Evidence and link them.

        The claim's ``evidence_message_ids`` name the source messages; each
        becomes one :class:`Evidence` row keyed on that id, so re-observing
        the same message reuses the row instead of duplicating it.  When the
        claim names no messages, *turn_ref* stands in for the turn it came
        from, which is what keeps K1-only extraction attributable.

        Links are typed by *relation*, which is what lets one message support
        one belief and contradict another.  A claim that produced no belief
        (no cognitive triple) still leaves its observation behind: provenance
        is a record of what was seen, independent of what was concluded.
        """
        ids = self._ensure_evidence_rows(
            session, user_id, session_id, claim,
            source_text=source_text, turn_ref=turn_ref,
        )
        self._link_evidence_rows(session, belief_id, ids, relation=relation)

    def _publish(
        self,
        session: object,
        user_id: str,
        session_id: str | None,
        transition: str,
        *,
        target_belief_id: int | None = None,
        payload: dict,
        evidence_ids: list[int],
        trace_action: str,
        predicate: str = "",
        object_: str = "",
        claim_text: str = "",
    ):
        """Publish one claim decision through the Publisher.

        The revision is read fresh for every proposal: several claims in one
        turn each advance it, and the Publisher's compare-and-swap refuses a
        proposal computed against a revision that has since moved.
        """
        from mirror_memory.core.proposal import StateTransitionProposal
        from mirror_memory.core.publisher import Publisher
        from mirror_memory.core.repository import get_state_revision

        proposal = StateTransitionProposal(
            transition=transition,
            user_id=user_id,
            session_id=session_id,
            target_belief_id=target_belief_id,
            payload=payload,
            evidence_ids=evidence_ids,
            expected_revision=get_state_revision(session, user_id),
            actor_type="extraction",
            actor_id=session_id or "",
        )
        decision = Publisher(session, self._config).publish(proposal)
        persisted = bool(decision.committed)
        belief_id = (decision.detail or {}).get("belief_id") if persisted else None
        self._trace(
            "publish",
            action=trace_action,
            committed=persisted,
            reason=decision.reason,
            proposal_id=proposal.proposal_id,
            belief_id=belief_id,
        )
        if not persisted:
            logger.info(
                "pipeline: publish refused action=%s reason=%s",
                trace_action, decision.reason,
            )
        return decision, belief_id

    @staticmethod
    def _round_trip_close_time(candidate) -> object:
        """The instant a round-trip's middle value stops being true.

        The candidate's own start when it has one; otherwise now, because the
        re-stating turn is itself the observation that ended it.
        """
        from datetime import UTC, datetime

        from mirror_memory.core.utils import coerce_datetime

        vf = coerce_datetime(candidate.valid_from)
        if vf is not None:
            return vf
        return datetime.now(UTC)

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
        source_text: str = "",
        turn_ref: str = "",
        **kwargs: object,
    ) -> None:
        """Persist extracted claims via the repository layer.

        When a claim has predicate/object in its value, the IdentityResolver
        decides the action (CREATE/SUPPORT/UPDATE) before persistence.

        Every persisted claim also records its source as a first-class
        :class:`Evidence` row and links it to the belief with a typed
        relation, so provenance survives independently of the belief row.
        """
        if not claims:
            return

        from mirror_memory.core.repository import (
            CONFLICT_SELF_CORRECTION,
            CONFLICT_SOURCE_CONFLICT,
            beliefs_with_predicates,
        )
        from mirror_memory.memory.atom import ACTION_CONTRADICT, ACTION_SUPPORT, ACTION_UPDATE, CandidateAtom
        from mirror_memory.memory.canonicalize import canonicalize_atom
        from mirror_memory.memory.identity import resolve_identity
        from mirror_memory.memory.polarity import (
            infer_lifecycle,
            infer_polarity,
        )

        stored: list[dict] = []

        def _record_store(**fields: object) -> None:
            stored.append(dict(fields))
            self._trace("store", **fields)

        def _attach_evidence(claim: dict, belief_id: int | None, relation: str) -> None:
            """Record the claim's source as Evidence and link it to the belief."""
            if belief_id is None:
                return
            self._attach_evidence_rows(
                session, user_id, session_id, claim, belief_id,
                relation=relation, source_text=source_text, turn_ref=turn_ref,
            )

        allowed = (
            {d.dimension_id for d in self._config.dimensions}
            if self._config.strict_dimensions
            else None
        )
        policy = self._config.identity_policy
        temporal_policy = self._config.temporal_policy
        synonyms = self._config.predicate_synonyms

        for claim in claims:
            try:
                value = claim.get("value") or {}
                pred = value.get("predicate", "")
                obj = value.get("object", "")
                canon_pred = pred
                canon_obj = obj
                # Structured polarity and goal lifecycle, inferred from the
                # canonicalised triple rather than matched against a word list
                # at read time.  Declared before the triple branch so the
                # CREATE/CONTRADICT path below can always see them.
                claim_polarity = ""
                claim_lifecycle = ""

                # Claims with no cognitive triple (e.g. K1 keyword hits) are
                # turn-classification signals, not slot assertions.  Persisting
                # them as beliefs puts an undated, never-superseded row holding
                # the raw snippet into the current-state surface, so the value
                # the user moved away from keeps answering "where do they live
                # now" after the replacement arrives.  The snippet itself
                # survives as session-summary source context, so recall of the
                # text is not lost -- only its false claim to be current state.
                if not (pred and obj):
                    self._trace(
                        "identity",
                        action="SKIPPED_NO_TRIPLE",
                        predicate=pred,
                        object=obj,
                        reason="claim carries no cognitive triple",
                    )
                    # The observation still enters the ledger even though no
                    # belief was concluded from it.
                    self._attach_evidence_rows(
                        session, user_id, session_id, claim, None,
                        relation="support", source_text=source_text,
                        turn_ref=turn_ref,
                    )
                    continue

                # If claim has cognitive triple fields, run identity resolution.
                if policy:
                    # Canonicalize
                    _, canon_pred, canon_obj = canonicalize_atom(
                        claim.get("subject", "user"), pred, obj, synonyms
                    )
                    # A return-to-value phrasing ("I went back to Falcon
                    # Works", "I returned to Shanghai") says the user is back
                    # at a value they held before, so the slot is whichever
                    # one holds that value -- not whichever verb the
                    # extractor guessed (``went_to`` for a workplace, or
                    # ``works_at`` for a school).  Rewriting it to the
                    # deliberately-unmapped ``returned_to`` hands the
                    # decision to the resolver's revival path, which matches
                    # on the object and closes the values the user left.
                    if (
                        _is_return_phrasing(claim.get("claim_text", ""))
                        and canon_pred != "returned_to"
                    ):
                        canon_pred = "returned_to"
                    claim_polarity = infer_polarity(canon_pred)
                    claim_lifecycle = infer_lifecycle(
                        canon_pred, claim.get("claim_text", "")
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
                        observed_at=value.get("observed_at"),
                        valid_from=value.get("valid_from"),
                        valid_to=value.get("valid_to"),
                        temporal_scope=temporal_policy.get(canon_pred, ""),
                        relation=claim.get("relation", "supports"),
                        # A return phrasing matches its slot by value, so the
                        # object comparison widens to shared identity tokens.
                        returning=_is_return_phrasing(claim.get("claim_text", "")),
                    )

                    # Get the beliefs this candidate could match.  The set is
                    # determined by the predicate, not by recency: asking for
                    # "the 200 most recent" silently hid older predicates past
                    # that bound, and the resolver then turned what should have
                    # been a SUPPORT or UPDATE into a CREATE.
                    from mirror_memory.memory.polarity import (
                        is_resumption,
                        is_self_correction,
                        is_termination,
                        object_tokens,
                    )

                    retracting = is_self_correction(
                        claim.get("claim_text", "")
                    ) or is_termination(claim.get("claim_text", ""))
                    resuming = is_resumption(claim.get("claim_text", ""))
                    # A value revival ("I went back to Falcon Works") has to
                    # see the *superseded* row it returns to, so it takes the
                    # object-aware candidate set (which includes history)
                    # rather than the active-only resumption surface.
                    returning = _is_return_phrasing(claim.get("claim_text", ""))
                    # A retraction or termination asserts the negation of the
                    # value it closes ("I gave the camera away" is not owning
                    # it).  The extractor's predicate for these ("gave_away",
                    # "not_speak") carries no polarity, so without this the
                    # closing row reads as a second positive fact about the
                    # same object and a current-state question keeps
                    # returning the withdrawn value.
                    if retracting and claim_polarity != "negative":
                        claim_polarity = "negative"
                    if retracting:
                        # A retraction refers to its target by meaning, so the
                        # candidate set is widened to every active belief that
                        # mentions one of the candidate's object tokens; the
                        # predicate/exact-object lookup would miss the very row
                        # the retraction has to close.
                        from mirror_memory.core.repository import (
                            active_goal_beliefs,
                            beliefs_referenced_by_tokens,
                        )

                        existing = beliefs_referenced_by_tokens(
                            session, user_id, object_tokens(canon_obj),
                        )
                        if not existing:
                            # "I gave up on the goal" shares no token with the
                            # goal it ends, so the only candidate set left is
                            # the user's live goals.
                            existing = active_goal_beliefs(session, user_id)
                    elif resuming and not returning:
                        # A resumption retires the ended stage of a goal, which
                        # is tagged rather than named by the new claim, so the
                        # candidate set is the recent active surface.
                        from mirror_memory.core.repository import list_active_beliefs

                        existing = list_active_beliefs(session, user_id, limit=25)
                    else:
                        existing = beliefs_with_predicates(
                            session, user_id, {canon_pred},
                            candidate_objects={canon_obj} if canon_obj else set(),
                        )
                        # The predicate/object lookup misses the rows that
                        # state the *opposite* of this claim about the same
                        # attribute: "dislikes museums" against "enjoys
                        # museum trips" has a different predicate and a
                        # differently-spelled object, so none of the query's
                        # clauses reach it -- and the withdrawal then stays
                        # active beside its own reversal.  Token overlap
                        # finds those rows; the resolver's polarity rule
                        # decides whether they are the same attribute.
                        from mirror_memory.core.repository import (
                            beliefs_referenced_by_tokens,
                        )

                        referenced = beliefs_referenced_by_tokens(
                            session, user_id, object_tokens(canon_obj)
                        )
                        if referenced:
                            seen_ids = {b.id for b in existing}
                            existing = existing + [
                                b for b in referenced if b.id not in seen_ids
                            ]
                    existing_dicts = [
                        {
                            "id": b.id,
                            "predicate": getattr(b, "predicate", ""),
                            "object": getattr(b, "object", ""),
                            "status": b.status,
                            "confidence": b.confidence,
                            "temporal": safe_json(getattr(b, "value_json", "{}")).get("temporal", ""),
                            "transition": safe_json(getattr(b, "value_json", "{}")).get("transition", ""),
                            "ended_object": safe_json(getattr(b, "value_json", "{}")).get("ended_object", ""),
                            "valid_from": getattr(b, "valid_from", None),
                            "valid_to": getattr(b, "valid_to", None),
                        }
                        for b in existing
                    ]

                    resolution = resolve_identity(
                        candidate, existing_dicts, policy, temporal_policy
                    )
                    logger.info(
                        "pipeline: identity resolution: action=%s pred=%s obj=%s reason=%s",
                        resolution.action, canon_pred, canon_obj, resolution.reason,
                    )
                    self._trace(
                        "identity",
                        action=resolution.action,
                        lifecycle=resolution.lifecycle,
                        temporal_relation=resolution.temporal_relation,
                        predicate=canon_pred,
                        object=canon_obj,
                        target_belief_id=resolution.target_belief_id,
                        reason=resolution.reason,
                    )

                    # Dispatch based on resolver action.
                    if resolution.action == ACTION_SUPPORT and resolution.target_belief_id:
                        evidence_ids = self._ensure_evidence_rows(
                            session, user_id, session_id, claim,
                            source_text=source_text, turn_ref=turn_ref,
                        )
                        decision, _belief_id = self._publish(
                            session, user_id, session_id, TRANSITION_SUPPORT,
                            target_belief_id=resolution.target_belief_id,
                            payload={
                                "claim_text": claim.get("claim_text", ""),
                                "evidence_message_ids": claim.get("evidence_message_ids"),
                            },
                            evidence_ids=evidence_ids,
                            trace_action="SUPPORT",
                            predicate=canon_pred,
                            object_=canon_obj,
                            claim_text=claim.get("claim_text", ""),
                        )
                        _record_store(
                            action="SUPPORT",
                            belief_id=resolution.target_belief_id,
                            persisted=decision.committed,
                            predicate=canon_pred,
                            object=canon_obj,
                            claim_text=claim.get("claim_text", ""),
                        )
                        self._link_evidence_rows(
                            session, resolution.target_belief_id, evidence_ids,
                            relation="support",
                        )
                        continue  # SUPPORT handled, skip record_claim

                    if resolution.action == ACTION_UPDATE and resolution.target_belief_id:
                        # Preserve metadata through the update.
                        value["predicate"] = canon_pred
                        value["object"] = canon_obj
                        claim["value"] = value
                        # A row created by ending a current value is itself an
                        # ended stage: a later resumption has to be able to
                        # find and retire it, and the row's own text ("I gave
                        # up the goal") says nothing about which goal it ended.
                        # The ended belief's identity travels with the row so
                        # the resumption matches by it, not by the row's own
                        # (generic) object.
                        if resolution.lifecycle == "END_CURRENT":
                            value["transition"] = "END_CURRENT"
                            value["ended_object"] = (
                                resolution.detail.get("ended_object") or ""
                            )
                            value["ended_belief_id"] = (
                                resolution.detail.get("ended_belief_id")
                            )
                            claim["value"] = value

                        evidence_ids = self._ensure_evidence_rows(
                            session, user_id, session_id, claim,
                            source_text=source_text, turn_ref=turn_ref,
                        )
                        # Round-trip revival: the resolver may name other
                        # same-attribute beliefs to close (the values the user
                        # left and has now returned past).  They travel in the
                        # payload so the Publisher closes them inside the same
                        # transaction as the revival -- a partial round-trip
                        # (new value live, middle value still active) is
                        # exactly the corruption this prevents.
                        close_ids = [
                            int(i)
                            for i in (resolution.detail.get("close_belief_ids") or [])
                            if i is not None and int(i) != resolution.target_belief_id
                        ]
                        # A revival returns to a previous value, so the new
                        # row belongs to the *revived* slot: it inherits the
                        # target's predicate and key rather than the verb the
                        # return phrasing was rewritten to.  Without this the
                        # revived value lands under ``returned_to`` and leaves
                        # the slot it belongs to empty.
                        revival_predicate, revival_key = canon_pred, claim.get("key", "")
                        if resolution.detail.get("revival"):
                            from mirror_memory.core.models import Belief as _Belief

                            target_row = session.get(_Belief, resolution.target_belief_id)
                            if target_row is not None:
                                revival_predicate = target_row.predicate or canon_pred
                                revival_key = target_row.key or revival_key
                        decision, new_id = self._publish(
                            session, user_id, session_id, TRANSITION_UPDATE,
                            target_belief_id=resolution.target_belief_id,
                            payload={
                                "subject": claim.get("subject", "user"),
                                "predicate": revival_predicate,
                                "object": canon_obj,
                                "dimension": claim.get("dimension", ""),
                                "key": revival_key,
                                "claim_text": claim.get("claim_text", ""),
                                "confidence": claim.get("confidence", 0.0),
                                # The cardinality follows the predicate the
                                # row will actually carry: a revival inherits
                                # the target's (``lives_in`` is single), so
                                # keying it off the rewritten ``returned_to``
                                # would file the returning value as a
                                # multi-valued preference and let it cool.
                                "cardinality": policy.get(revival_predicate, "multi"),
                                "evidence_message_ids": claim.get("evidence_message_ids"),
                                "value": value,
                                "observed_at": candidate.observed_at,
                                "valid_from": candidate.valid_from,
                                "valid_to": candidate.valid_to,
                                "temporal_scope": temporal_policy.get(canon_pred, ""),
                                "polarity": claim_polarity,
                                "lifecycle_state": claim_lifecycle,
                                "raw_predicate": pred,
                                "close_belief_ids": close_ids,
                                # A revival targets a superseded row on
                                # purpose; the Publisher's closed-row
                                # invariant lets this one through.
                                "revival": bool(resolution.detail.get("revival")),
                                "close_at": self._round_trip_close_time(candidate),
                            },
                            evidence_ids=evidence_ids,
                            trace_action="UPDATE",
                            predicate=canon_pred,
                            object_=canon_obj,
                            claim_text=claim.get("claim_text", ""),
                        )
                        if new_id is not None:
                            logger.info("pipeline: UPDATE -> %s", new_id)
                        _record_store(
                            action="UPDATE",
                            belief_id=new_id,
                            persisted=decision.committed,
                            predicate=canon_pred,
                            object=canon_obj,
                            claim_text=claim.get("claim_text", ""),
                        )
                        self._link_evidence_rows(
                            session, new_id, evidence_ids, relation="support"
                        )
                        if decision.committed and resolution.target_belief_id:
                            # The superseded value stays in the graph, marked as
                            # what it now is: replaced.
                            self._link_evidence_rows(
                                session, resolution.target_belief_id, evidence_ids,
                                relation="correct",
                            )
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
                evidence_ids = self._ensure_evidence_rows(
                    session, user_id, session_id, claim,
                    source_text=source_text, turn_ref=turn_ref,
                )
                relation = claim.get("relation", "supports")
                decision, belief_id = self._publish(
                    session, user_id, session_id,
                    TRANSITION_CONTRADICT if relation == "contradicts" else TRANSITION_CREATE,
                    payload={
                        "dimension": claim.get("dimension", ""),
                        "key": claim.get("key", ""),
                        "claim_text": claim.get("claim_text", ""),
                        "value": claim.get("value"),
                        "confidence": claim.get("confidence", 0.0),
                        "layer": claim.get("layer", "L4"),
                        "source": claim.get("source", "extracted"),
                        "evidence_message_ids": claim.get("evidence_message_ids"),
                        "blocked_key_prefixes": list(self._config.blocked_key_prefixes),
                        # None (non-strict mode) means "any dimension".
                        "allowed_dimensions": allowed,
                        "context_tags": context_tags,
                        # Triple fields.
                        "subject": claim.get("subject", "user"),
                        "predicate": canon_pred,
                        "object": canon_obj,
                        "cardinality": policy.get(canon_pred, "multi"),
                        "polarity": claim_polarity,
                        "lifecycle_state": claim_lifecycle,
                        # The extractor's own spelling, kept for audit: the
                        # canonical predicate folds many surface forms onto one
                        # slot and the raw form is the only record of which was
                        # actually produced.
                        "raw_predicate": pred,
                        # Every contradiction the extractor reports is the user
                        # retracting their own earlier statement -- the two claims
                        # come from the same subject in the same conversation.  A
                        # cross-source disagreement never reaches this path.
                        "conflict_kind": (
                            CONFLICT_SELF_CORRECTION
                            if relation == "contradicts"
                            else CONFLICT_SOURCE_CONFLICT
                        ),
                    },
                    evidence_ids=evidence_ids,
                    trace_action="CREATE" if relation != "contradicts" else "CONTRADICT",
                    predicate=canon_pred,
                    object_=canon_obj,
                    claim_text=claim.get("claim_text", ""),
                )
                _record_store(
                    action="CONTRADICT" if relation == "contradicts" else "CREATE",
                    belief_id=belief_id,
                    persisted=decision.committed,
                    predicate=canon_pred,
                    object=canon_obj,
                    claim_text=claim.get("claim_text", ""),
                )
                self._link_evidence_rows(
                    session, belief_id, evidence_ids,
                    relation="contradict" if relation == "contradicts" else "support",
                )
            except Exception:
                logger.warning(
                    "pipeline: failed to persist claim: dim=%s key=%s",
                    claim.get("dimension"),
                    claim.get("key"),
                    exc_info=True,
                )
