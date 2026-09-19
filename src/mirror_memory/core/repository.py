"""Belief repository -- merge strategy, confidence update, resurrection guard.

Merge strategy (generic):
- ``(user_id, key)`` has exactly one current row; ``relation=supports``
  pushes confidence and appends evidence, ``relation=contradicts`` attenuates
  confidence.
- After rejection (``reject_belief``) the same key is never resurrected:
  new claims return ``(None, "resurrection_guard")`` with no event logged.
- ``source="extracted"`` rows are forced to ``layer="L4"``; ``L2`` can only
  come from user confirmation (``confirm_belief``) or program hard evidence.

All queries use SQLAlchemy 2.0 parameterised construction (bound parameters,
no string concatenation).

Domain overrides:
- ``blocked_key_prefixes``: list of key prefixes that are structurally
  forbidden from persisting (e.g. ``["distortion."]`` in a psychology
  domain).  Defaults to empty.
- ``question_value_tiers``: mapping of dimension -> int tier for question
  candidate prioritisation.  Dimensions not in the map get
  ``DEFAULT_QUESTION_TIER``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mirror_memory.core.confidence import compute_confidence_weight
from mirror_memory.core.utils import safe_json
from mirror_memory.core.constants import (
    CONFIDENCE_CEILING,
    DEFAULT_QUESTION_TIER,
    L4_QUESTION_THRESHOLD,
    MAX_EVIDENCE_REFS,
    QUESTION_TIER_WEIGHT_MAX,
    QUESTION_TIER_WEIGHT_MIN,
    SUPPORT_GAIN,
)
from mirror_memory.core.models import (
    Belief,
    BeliefEvent,
    EvolutionJob,
    ExtractionStats,
    InterventionEvent,
    MemoryPreference,
    Snapshot,
    User,
    utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory preference
# ---------------------------------------------------------------------------


def is_memory_enabled(session: Session, user_id: str) -> bool:
    """Return the user's memory preference; missing rows = enabled (legacy-on)."""
    pref = session.get(MemoryPreference, user_id)
    return pref.enabled if pref is not None else True


def set_memory_enabled(session: Session, user_id: str, enabled: bool) -> MemoryPreference:
    pref = session.get(MemoryPreference, user_id)
    if pref is None:
        pref = MemoryPreference(user_id=user_id)
        session.add(pref)
    pref.enabled = enabled
    pref.disabled_at = None if enabled else utcnow()
    pref.updated_at = utcnow()
    session.flush()
    return pref


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_event(
    session: Session,
    belief: Belief,
    event_type: str,
    *,
    evidence: list[int],
    detail: dict | None = None,
    stats_id: int | None = None,
) -> None:
    session.add(
        BeliefEvent(
            belief_id=belief.id,
            user_id=belief.user_id,
            event_type=event_type,
            evidence_json=json.dumps(evidence),
            detail_json=json.dumps(detail or {}, ensure_ascii=False),
            origin_stats_id=stats_id,
        )
    )


# ---------------------------------------------------------------------------
# Belief CRUD
# ---------------------------------------------------------------------------


def get_belief(session: Session, user_id: str, key: str) -> Belief | None:
    """Return the current row for ``(user_id, key)`` (including rejected)."""
    return session.scalar(select(Belief).where(Belief.user_id == user_id, Belief.key == key))


def get_active_belief(session: Session, user_id: str, key: str) -> Belief | None:
    belief = get_belief(session, user_id, key)
    if belief is not None and belief.status == "active":
        return belief
    return None


def list_active_beliefs(
    session: Session,
    user_id: str,
    *,
    dimensions: tuple[str, ...] | None = None,
    limit: int = 50,
) -> list[Belief]:
    """Active beliefs, ordered by most-recent evidence.  Returns [] if memory
    is disabled for the user.
    """
    if not is_memory_enabled(session, user_id):
        return []
    conditions = [Belief.user_id == user_id, Belief.status == "active"]
    if dimensions:
        conditions.append(Belief.dimension.in_(dimensions))
    stmt = select(Belief).where(*conditions).order_by(desc(Belief.last_evidence_at)).limit(limit)
    return list(session.scalars(stmt))


def list_rejected_beliefs(session: Session, user_id: str, *, limit: int = 5) -> list[Belief]:
    """Rejected beliefs, most-recently updated first."""
    if not is_memory_enabled(session, user_id):
        return []
    stmt = (
        select(Belief)
        .where(Belief.user_id == user_id, Belief.status == "rejected")
        .order_by(desc(Belief.updated_at))
        .limit(limit)
    )
    return list(session.scalars(stmt))


def get_belief_events(session: Session, user_id: str, belief_id: int, *, limit: int = 50) -> list[BeliefEvent]:
    stmt = (
        select(BeliefEvent)
        .where(BeliefEvent.user_id == user_id, BeliefEvent.belief_id == belief_id)
        .order_by(desc(BeliefEvent.id))
        .limit(limit)
    )
    return list(session.scalars(stmt))


# ---------------------------------------------------------------------------
# record_claim -- the main write path
# ---------------------------------------------------------------------------


def record_claim(
    session: Session,
    user_id: str,
    *,
    dimension: str,
    key: str,
    claim_text: str,
    value: dict | None = None,
    relation: str = "supports",
    confidence: float = 0.0,
    layer: str = "L4",
    source: str = "extracted",
    session_id: str | None = None,
    evidence_message_ids: list[int] | None = None,
    stats_id: int | None = None,
    origin_slice_id: str | None = None,
    context_tags: list[str] | None = None,
    blocked_key_prefixes: tuple[str, ...] = (),
) -> tuple[Belief | None, str]:
    """Write a claim, applying the merge strategy.

    Returns ``(belief, event_type)`` where *event_type* is one of:
    ``"created"``, ``"supported"``, ``"contradicted"``, ``"resurrection_guard"``,
    ``"profile_memory_disabled"``.

    Parameters
    ----------
    blocked_key_prefixes:
        Key prefixes that are structurally forbidden from persisting
        (e.g. ``("distortion.",)``).  The caller supplies this list so
        the repository layer stays domain-agnostic.
    """
    if not is_memory_enabled(session, user_id):
        return None, "profile_memory_disabled"

    if relation not in {"supports", "contradicts"}:
        raise ValueError(f"invalid claim relation: {relation!r}")

    for prefix in blocked_key_prefixes:
        if key.startswith(prefix):
            raise ValueError(f"blocked key prefix {prefix!r} cannot be persisted: {key!r}")

    evidence = [int(mid) for mid in (evidence_message_ids or [])]

    # Extracted claims are always L4.
    if source == "extracted" and layer != "L4":
        logger.warning("Clamping non-L4 layer from extracted claim (layer=%s)", layer)
        layer = "L4"

    existing = get_belief(session, user_id, key)
    if existing is not None and existing.status == "rejected":
        return None, "resurrection_guard"

    if existing is None:
        # New belief.
        merged_value = dict(value or {})
        if context_tags:
            existing_tags = merged_value.get("context_tags", [])
            merged_value["context_tags"] = sorted(set(existing_tags) | set(context_tags))
        belief = Belief(
            user_id=user_id,
            dimension=dimension,
            key=key,
            claim_text=claim_text,
            value_json=json.dumps(merged_value, ensure_ascii=False),
            layer=layer,
            status="active",
            confidence=min(CONFIDENCE_CEILING, max(0.0, confidence)),
            source=source,
            evidence_json=json.dumps(evidence[-MAX_EVIDENCE_REFS:]),
            origin_stats_id=stats_id,
            origin_slice_id=origin_slice_id,
            origin_session_id=session_id,
            last_evidence_session_id=session_id,
        )
        session.add(belief)
        session.flush()
        _append_event(session, belief, "created", evidence=evidence, stats_id=stats_id)
        return belief, "created"

    if relation == "supports":
        # Merge supporting evidence.
        merged_evidence = json.loads(existing.evidence_json or "[]")
        for mid in evidence:
            if mid not in merged_evidence:
                merged_evidence.append(mid)
        existing.evidence_json = json.dumps(merged_evidence[-MAX_EVIDENCE_REFS:])
        existing.last_evidence_session_id = session_id
        existing.last_evidence_at = datetime.now(UTC)

        # Update claim_text with the latest version (preserves specific details
        # like dates, names, numbers that may be more precise in later mentions).
        if claim_text and len(claim_text) > len(existing.claim_text or ""):
            existing.claim_text = claim_text

        if context_tags:
            val = safe_json(existing.value_json)
            old_tags = val.get("context_tags", [])
            val["context_tags"] = sorted(set(old_tags) | set(context_tags))
            existing.value_json = json.dumps(val, ensure_ascii=False)

        # Multi-factor confidence: base gain * 7-factor weight.
        now = datetime.now(UTC)
        last_ev = existing.last_evidence_at
        if last_ev.tzinfo is None:
            last_ev = last_ev.replace(tzinfo=UTC)
        days_since = max(0.0, (now - last_ev).total_seconds() / 86400)
        val = safe_json(existing.value_json)
        ctx_div = len(val.get("context_tags", []))
        events = get_belief_events(session, existing.user_id, existing.id)
        contradict_count = sum(1 for e in events if e.event_type in {"contradicted", "downgraded"})
        support_count = len(merged_evidence)

        weight = compute_confidence_weight(
            source=existing.source,
            layer=existing.layer,
            support_count=support_count,
            contradict_count=contradict_count,
            context_diversity=ctx_div,
            days_since_last_evidence=days_since,
            value_json=existing.value_json,
        )
        existing.confidence = min(CONFIDENCE_CEILING, existing.confidence + SUPPORT_GAIN * weight)
        _append_event(
            session,
            existing,
            "supported",
            evidence=evidence,
            detail={"confidence": existing.confidence},
            stats_id=stats_id,
        )
        return existing, "supported"

    # Contradicts: attenuate confidence, mark needs_clarification.
    from mirror_memory.core.constants import CONTRADICT_FACTOR

    existing.confidence = round(existing.confidence * CONTRADICT_FACTOR, 4)
    val = safe_json(existing.value_json)
    val["clarification_status"] = "needs_clarification"
    existing.value_json = json.dumps(val, ensure_ascii=False)
    detail: dict = {"confidence": existing.confidence, "clarification_status": "needs_clarification"}
    _append_event(session, existing, "contradicted", evidence=evidence, detail=detail, stats_id=stats_id)
    return existing, "contradicted"


# ---------------------------------------------------------------------------
# confirm / reject
# ---------------------------------------------------------------------------


def confirm_belief(session: Session, user_id: str, belief_id: int) -> Belief | None:
    """User confirmation: L4 -> L2 (questioning loop upgrade primitive)."""
    belief = session.get(Belief, belief_id)
    if belief is None or belief.user_id != user_id or belief.status != "active":
        return None
    belief.layer = "L2"
    belief.source = "user_confirmed"
    belief.confidence = max(belief.confidence, 0.9)
    _append_event(session, belief, "confirmed", evidence=[], detail={"layer": "L2"})
    return belief


def reject_belief(session: Session, user_id: str, belief_id: int) -> Belief | None:
    """User rejection: status -> rejected (same key never resurrected)."""
    belief = session.get(Belief, belief_id)
    if belief is None or belief.user_id != user_id or belief.status != "active":
        return None
    belief.status = "rejected"
    belief.confidence = 0.0
    _append_event(session, belief, "rejected", evidence=[], detail={"reason": "user_denied"})
    return belief


def update_belief_value(
    session: Session,
    user_id: str,
    key: str,
    value: dict,
    *,
    stats_id: int | None = None,
    evidence: list[int] | None = None,
) -> Belief | None:
    """Update a belief's structured value (value_json), preserving the
    previous value in the event log.
    """
    belief = get_belief(session, user_id, key)
    if belief is None or belief.status != "active":
        return None
    previous = belief.value_json
    belief.value_json = json.dumps(value, ensure_ascii=False)
    _append_event(
        session,
        belief,
        "value_updated",
        evidence=evidence or [],
        detail={"previous_value": previous, "value": value},
        stats_id=stats_id,
    )
    return belief


# ---------------------------------------------------------------------------
# Question candidates
# ---------------------------------------------------------------------------


def _question_confirm_rate_weight(rate: float) -> float:
    """Confirm-rate [0,1] -> weight [MIN, MAX] linear map; rate=0.5 -> 1.0."""
    clamped = min(max(rate, 0.0), 1.0)
    weight = QUESTION_TIER_WEIGHT_MIN + (QUESTION_TIER_WEIGHT_MAX - QUESTION_TIER_WEIGHT_MIN) * clamped
    return round(weight, 4)


def _needs_clarification_boost(belief: Belief) -> int:
    """Needs-clarification beliefs get priority in the question queue."""
    val = safe_json(belief.value_json)
    return 1 if val.get("clarification_status") == "needs_clarification" else 0


def list_question_candidates(
    session: Session,
    user_id: str,
    *,
    limit: int = 1,
    question_value_tiers: dict[str, int] | None = None,
) -> list[Belief]:
    """Question-candidate beliefs: L4, confidence >= threshold, cross-session
    evidence.

    Sorting = (clarification boost, tier * learn_weight, confidence, recency).

    Parameters
    ----------
    question_value_tiers:
        Mapping of dimension -> int priority tier.  Dimensions absent from
        the map fall back to ``DEFAULT_QUESTION_TIER``.
    """
    if not is_memory_enabled(session, user_id):
        return []

    tiers = question_value_tiers or {}

    conditions = (
        Belief.user_id == user_id,
        Belief.status == "active",
        Belief.layer == "L4",
    )
    rows = list(session.scalars(select(Belief).where(*conditions)))

    # Filter: confidence >= threshold AND cross-session evidence.
    pending = [
        b
        for b in rows
        if b.confidence >= L4_QUESTION_THRESHOLD
        and (
            not b.origin_session_id
            or not b.last_evidence_session_id
            or b.origin_session_id != b.last_evidence_session_id
        )
    ]

    pending.sort(
        key=lambda b: (
            _needs_clarification_boost(b),
            tiers.get(b.dimension, DEFAULT_QUESTION_TIER) * 1.0,  # learn_weight=1.0 default
            b.confidence,
            b.last_evidence_at,
        ),
        reverse=True,
    )
    return pending[:limit]


# ---------------------------------------------------------------------------
# Intervention events (verification loop)
# ---------------------------------------------------------------------------

QUESTION_INJECTED_KIND = "question_injected"
QUESTION_ANSWERED_KIND = "question_answered"


def record_intervention_event(
    session: Session,
    user_id: str,
    session_id: str,
    *,
    kind: str,
    detail: dict | None = None,
) -> InterventionEvent | None:
    """Record a question injection or answer event.

    Only action metadata is stored (belief label/key, verdict) -- never the
    conversation content.  Returns ``None`` when memory is disabled for the
    user.
    """
    if not is_memory_enabled(session, user_id):
        return None
    row = InterventionEvent(
        user_id=user_id,
        session_id=session_id,
        kind=kind,
        detail_json=json.dumps(detail or {}, ensure_ascii=False),
    )
    session.add(row)
    session.flush()
    return row


def get_pending_verification(
    session: Session,
    user_id: str,
    session_id: str,
    *,
    max_window_turns: int = 3,
) -> InterventionEvent | None:
    """Get the pending question injection within the N-turn window.

    Returns the InterventionEvent with kind='question_injected' that:
    - is the most recent injection for the user,
    - has NOT been answered yet (no later ``question_answered`` event),
    - is within the last N sessions (approximate turn window): the injection
      was made in the current session or in one of the last ``N-1`` sessions
      with recorded activity.  Intervention events are the generic
      session-activity proxy (this library has no message table).

    Returns None if no pending verification exists.
    """
    injection = session.scalar(
        select(InterventionEvent)
        .where(InterventionEvent.user_id == user_id, InterventionEvent.kind == QUESTION_INJECTED_KIND)
        .order_by(desc(InterventionEvent.id))
        .limit(1)
    )
    if injection is None:
        return None

    answered = session.scalar(
        select(InterventionEvent)
        .where(InterventionEvent.user_id == user_id, InterventionEvent.kind == QUESTION_ANSWERED_KIND)
        .order_by(desc(InterventionEvent.id))
        .limit(1)
    )
    if answered is not None and answered.id > injection.id:
        return None

    if injection.session_id == session_id:
        return injection

    if max_window_turns <= 0:
        return None

    # Collect the most recent session ids from the event stream; the current
    # session counts as the first entry of the window.
    window: list[str] = [session_id]
    seen = {session_id}
    for sid in session.scalars(
        select(InterventionEvent.session_id)
        .where(InterventionEvent.user_id == user_id)
        .order_by(desc(InterventionEvent.id))
    ):
        if len(window) >= max_window_turns:
            break
        if sid not in seen:
            seen.add(sid)
            window.append(sid)
    return injection if injection.session_id in seen else None


def has_unanswered_injection(session: Session, user_id: str) -> bool:
    """Check if there's an unanswered question injection (prevents double-inject).

    Only one hypothesis may be pending at a time: when the latest
    ``question_injected`` has no later ``question_answered``, injecting a new
    question would shadow the previous one and the verification loop could
    never close.
    """
    last_injected = session.scalar(
        select(InterventionEvent)
        .where(InterventionEvent.user_id == user_id, InterventionEvent.kind == QUESTION_INJECTED_KIND)
        .order_by(desc(InterventionEvent.id))
        .limit(1)
    )
    if last_injected is None:
        return False
    last_answered = session.scalar(
        select(InterventionEvent)
        .where(InterventionEvent.user_id == user_id, InterventionEvent.kind == QUESTION_ANSWERED_KIND)
        .order_by(desc(InterventionEvent.id))
        .limit(1)
    )
    return last_answered is None or last_answered.id < last_injected.id


# ---------------------------------------------------------------------------
# Extraction stats
# ---------------------------------------------------------------------------


def record_extraction_stats(
    session: Session,
    user_id: str,
    *,
    session_id: str | None = None,
    trigger: str = "",
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    claims_out: int = 0,
    status: str = "ok",
    error: str = "",
) -> ExtractionStats:
    """Record one extraction-call statistics row."""
    row = ExtractionStats(
        user_id=user_id,
        session_id=session_id,
        trigger=trigger,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        claims_out=claims_out,
        status=status,
        error=error,
    )
    session.add(row)
    session.flush()
    return row


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def delete_user_memories(session: Session, user_id: str) -> dict[str, int]:
    """Cascade-delete all inferred beliefs, events, stats, snapshots,
    intervention events, and evolution jobs for a user.  Returns counts of
    deleted rows per table.
    """
    events_deleted = (
        session.query(BeliefEvent)
        .filter(BeliefEvent.user_id == user_id)
        .delete(synchronize_session=False)
    )
    beliefs_deleted = (
        session.query(Belief).filter(Belief.user_id == user_id).delete(synchronize_session=False)
    )
    stats_deleted = (
        session.query(ExtractionStats)
        .filter(ExtractionStats.user_id == user_id)
        .delete(synchronize_session=False)
    )
    jobs_deleted = (
        session.query(EvolutionJob)
        .filter(EvolutionJob.user_id == user_id)
        .delete(synchronize_session=False)
    )
    snapshots_deleted = (
        session.query(Snapshot).filter(Snapshot.user_id == user_id).delete(synchronize_session=False)
    )
    interventions_deleted = (
        session.query(InterventionEvent)
        .filter(InterventionEvent.user_id == user_id)
        .delete(synchronize_session=False)
    )
    return {
        "beliefs": int(beliefs_deleted),
        "belief_events": int(events_deleted),
        "extraction_stats": int(stats_deleted),
        "evolution_jobs": int(jobs_deleted),
        "snapshots": int(snapshots_deleted),
        "intervention_events": int(interventions_deleted),
    }


# ---------------------------------------------------------------------------
# Session summary (unstructured fallback storage)
# ---------------------------------------------------------------------------


def store_session_summary(
    session: Session,
    user_id: str,
    session_id: str,
    text: str,
    turn_count: int,
    *,
    max_length: int = 3000,
    snippet_max: int = 500,
) -> None:
    """Append a conversation snippet to the session summary.

    Preserves specific details (dates, names, numbers) that structured
    beliefs may lose.  Creates a new row on first call, appends on
    subsequent calls.
    """
    from mirror_memory.core.models import SessionSummary

    snippet = text.strip()[:snippet_max]

    existing = (
        session.query(SessionSummary)
        .filter(
            SessionSummary.user_id == user_id,
            SessionSummary.session_id == session_id,
        )
        .first()
    )
    if existing:
        combined = existing.summary_text + " | " + snippet
        if len(combined) > max_length:
            combined = combined[-max_length:]
        existing.summary_text = combined
        existing.turn_count = turn_count
    else:
        session.add(SessionSummary(
            user_id=user_id,
            session_id=session_id,
            summary_text=snippet,
            turn_count=turn_count,
        ))
    session.flush()
