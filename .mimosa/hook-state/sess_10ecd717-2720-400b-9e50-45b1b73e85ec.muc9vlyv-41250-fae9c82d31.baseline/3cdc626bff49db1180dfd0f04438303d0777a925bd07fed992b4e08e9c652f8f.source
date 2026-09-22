"""Verification loop -- candidate selection, judgment, and belief lifecycle.

Closes the questioning loop in four stages:

1. Candidate   : :func:`get_question_candidates` picks L4 hypotheses that are
                 confident enough and supported by cross-session evidence.
2. Injection   : :func:`record_question_injection` records a
                 ``question_injected`` intervention event (guarded by
                 ``has_unanswered_injection`` so only one hypothesis is
                 pending at a time).
3. Judgment    : :func:`run_verification_judgment` matches the user's reply
                 against the pending injection and asks the LLM (using the
                 ``verification`` prompt template from
                 ``config/prompts/verification.txt``) for confirm/deny/unclear.
4. Lifecycle   : confirm -> ``confirm_belief`` (L4 -> L2);
                 deny -> ``reject_belief`` (into rejected, never resurrected);
                 unclear -> event recorded only, belief untouched.

Fail-open: every entry point catches its own errors and never blocks the
caller.  When no LLM client or verification prompt is configured the loop is
inert (judgment returns False immediately).
"""

from __future__ import annotations

import logging
import re
import time

from sqlalchemy import select

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import (
    DEFAULT_QUESTION_TIER,
    L4_QUESTION_THRESHOLD,
    QUESTION_TIER_WEIGHT_MAX,
    QUESTION_TIER_WEIGHT_MIN,
)
from mirror_memory.core.models import Belief
from mirror_memory.core.repository import (
    QUESTION_ANSWERED_KIND,
    QUESTION_INJECTED_KIND,
    confirm_belief,
    get_belief,
    get_pending_verification,
    has_unanswered_injection,
    is_memory_enabled,
    record_extraction_stats,
    record_intervention_event,
    reject_belief,
)
from mirror_memory.core.utils import parse_llm_json, safe_json

logger = logging.getLogger(__name__)

# Generic question phrasing for the injected verification question.  Hosts
# may rephrase freely -- only the event metadata drives the loop.
QUESTION_TEMPLATE = 'Quick check — "{claim}" — does that sound right?'

# Verdicts the verification prompt may return.
VALID_VERDICTS = ("confirm", "deny", "unclear")

_VERDICT_PATTERN = re.compile(r"\b(confirm|deny|unclear)\b")


# ---------------------------------------------------------------------------
# Stage 1: candidate selection
# ---------------------------------------------------------------------------


def _confirm_rate(belief: Belief) -> float:
    """Confirm-rate prior from the belief's value_json (default 0.5)."""
    rate = safe_json(belief.value_json).get("confirm_rate", 0.5)
    try:
        return min(1.0, max(0.0, float(rate)))
    except (TypeError, ValueError):
        return 0.5


def _confirm_rate_weight(rate: float) -> float:
    """Map confirm-rate [0, 1] to a learning weight [MIN, MAX]; 0.5 -> 1.0."""
    clamped = min(max(rate, 0.0), 1.0)
    return round(QUESTION_TIER_WEIGHT_MIN + (QUESTION_TIER_WEIGHT_MAX - QUESTION_TIER_WEIGHT_MIN) * clamped, 4)


def _needs_clarification_boost(belief: Belief) -> int:
    """Needs-clarification beliefs get priority in the question queue."""
    return 1 if safe_json(belief.value_json).get("clarification_status") == "needs_clarification" else 0


def get_question_candidates(
    session: object,
    user_id: str,
    config: MemoryConfig,
    *,
    limit: int = 1,
) -> list[Belief]:
    """Get L4 beliefs eligible for verification questioning.

    Criteria:
    - layer == "L4" (still a hypothesis, not user-confirmed),
    - confidence >= L4_QUESTION_THRESHOLD (0.7),
    - cross-session evidence (``origin_session_id != last_evidence_session_id``).

    Sorted by ``question_value_tiers[dimension] * confirm_rate_weight``
    (descending), with needs-clarification beliefs first; confidence and
    recency break ties.  Dimensions absent from ``config.question_value_tiers``
    fall back to ``DEFAULT_QUESTION_TIER``.

    Returns [] when memory is disabled for the user.
    """
    if not is_memory_enabled(session, user_id):
        return []

    tiers = dict(getattr(config, "question_value_tiers", None) or {})

    rows = list(
        session.scalars(
            select(Belief).where(
                Belief.user_id == user_id,
                Belief.status == "active",
                Belief.layer == "L4",
            )
        )
    )

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
            tiers.get(b.dimension, DEFAULT_QUESTION_TIER) * _confirm_rate_weight(_confirm_rate(b)),
            b.confidence,
            b.last_evidence_at,
        ),
        reverse=True,
    )
    return pending[: max(1, limit)]


# ---------------------------------------------------------------------------
# Stage 2: injection
# ---------------------------------------------------------------------------


def _display_label(belief: Belief, config: MemoryConfig) -> str:
    """Friendly label for a belief from the display config; falls back to
    the claim text, then the raw key."""
    for lbl in config.display_labels:
        if lbl.key == belief.key and lbl.dimension == belief.dimension:
            if lbl.zh or lbl.en:
                return lbl.zh or lbl.en
    return belief.claim_text or belief.key


def build_verification_question(belief: Belief, config: MemoryConfig) -> str:
    """Build the question text shown to the user for a candidate belief."""
    return QUESTION_TEMPLATE.format(claim=_display_label(belief, config))


def record_question_injection(
    session: object,
    user_id: str,
    session_id: str,
    belief: Belief,
    config: MemoryConfig,
) -> object | None:
    """Record a ``question_injected`` event for *belief* (stage 2).

    No-op when a previous injection is still unanswered (prevents
    double-inject: a second pending hypothesis would shadow the first and the
    judgment half-loop could never close).
    """
    if has_unanswered_injection(session, user_id):
        logger.info("verification: injection skipped -- previous question still pending")
        return None
    label = _display_label(belief, config)
    event = record_intervention_event(
        session,
        user_id,
        session_id,
        kind=QUESTION_INJECTED_KIND,
        detail={
            "belief_label": label,
            "belief_key": belief.key,
            "belief_id": int(belief.id),
            "question": build_verification_question(belief, config),
        },
    )
    logger.info("verification: question injected for belief=%s", belief.key)
    return event


# ---------------------------------------------------------------------------
# Stage 3: judgment
# ---------------------------------------------------------------------------


def parse_verdict(raw: str) -> str:
    """Parse the LLM verification output into confirm/deny/unclear.

    Accepts the bare word mandated by the prompt as well as JSON envelopes
    (``{"verdict": "confirm"}``).  Anything unparseable is ``unclear`` --
    never force a verdict.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("json"):
            text = text[4:]
    lowered = text.lower()

    data = parse_llm_json(lowered)
    if isinstance(data, dict):
        verdict = str(data.get("verdict") or "").lower()
        if verdict in VALID_VERDICTS:
            return verdict

    match = _VERDICT_PATTERN.search(lowered)
    if match:
        return match.group(1)
    return "unclear"


def _update_confirm_rate(session: object, belief: object, verdict: str) -> None:
    """Update belief's confirm_rate using Beta(1,1) shrinkage.

    Called after every verdict (including unclear, which is a no-op for
    counts but keeps the rate computation consistent).
    """
    import json as _json

    from mirror_memory.core.constants import BETA_PRIOR_ALPHA, BETA_PRIOR_BETA

    try:
        val = _json.loads(belief.value_json or "{}")
    except (TypeError, ValueError):
        val = {}

    alpha = val.get("confirm_alpha", BETA_PRIOR_ALPHA)
    beta = val.get("confirm_beta", BETA_PRIOR_BETA)

    if verdict == "confirm":
        alpha += 1
    elif verdict == "deny":
        beta += 1
    # unclear: no count update

    total = alpha + beta
    val["confirm_alpha"] = alpha
    val["confirm_beta"] = beta
    val["confirm_rate"] = round(alpha / total, 4) if total > 0 else 0.5
    belief.value_json = _json.dumps(val, ensure_ascii=False)
    session.flush()


def run_verification_judgment(
    session: object,
    user_id: str,
    session_id: str,
    user_text: str,
    config: MemoryConfig,
    llm_client: object | None,
) -> bool:
    """Run the verification half-loop for this turn.

    1. Check for pending verification (:func:`get_pending_verification`).
    2. If found, call the LLM with the ``config.prompts.verification`` prompt
       to judge confirm/deny/unclear against the user's reply.
    3. confirm -> ``confirm_belief`` (L4 -> L2).
    4. deny -> ``reject_belief`` (status=rejected; the key is never
       resurrected by the repository's resurrection guard).
    5. unclear -> record event only, belief state unchanged.
    6. Record a ``question_answered`` intervention event (closes the loop).

    Returns True if this turn was consumed by verification (caller should
    skip K2).  Fail-open: any error rolls back and returns False.
    """
    try:
        if llm_client is None:
            return False
        system_prompt = (config.prompts.verification or "").strip()
        if not system_prompt:
            logger.debug("verification: prompt template empty -- loop disabled")
            return False
        if not is_memory_enabled(session, user_id):
            return False

        injection = get_pending_verification(session, user_id, session_id)
        if injection is None:
            return False

        detail = safe_json(injection.detail_json)
        belief_key = str(detail.get("belief_key") or "")
        belief = get_belief(session, user_id, belief_key) if belief_key else None
        if belief is None or belief.status != "active":
            # Stale injection (belief already resolved or deleted elsewhere):
            # consume it so the pending state cannot block future injections.
            record_intervention_event(
                session,
                user_id,
                session_id,
                kind=QUESTION_ANSWERED_KIND,
                detail={"verdict": "stale", "belief_key": belief_key},
            )
            logger.info("verification: stale injection consumed (belief_key=%s)", belief_key)
            return False

        stats = record_extraction_stats(
            session,
            user_id,
            session_id=session_id,
            trigger="verification",
            model=str(getattr(llm_client, "model", "") or ""),
        )

        payload = (
            f"[Hypothesis]\n{(belief.claim_text or belief.key).strip()}\n\n"
            f"[User reply]\n{(user_text or '').strip()}"
        )
        started = time.monotonic()
        try:
            raw = llm_client.generate(
                system_prompt=system_prompt,
                payload_text=payload,
                fallback=lambda: "unclear",
            )
        except Exception:
            logger.warning("verification: LLM call failed; falling back to unclear", exc_info=True)
            raw = "unclear"
            stats.status = "error"
            stats.error = "llm_failed_fallback"
        stats.latency_ms = int((time.monotonic() - started) * 1000)

        verdict = parse_verdict(raw)
        if verdict == "confirm":
            confirm_belief(session, user_id, belief.id)
            _update_confirm_rate(session, belief, "confirm")
        elif verdict == "deny":
            reject_belief(session, user_id, belief.id)
            _update_confirm_rate(session, belief, "deny")
        else:
            _update_confirm_rate(session, belief, "unclear")
        # unclear: record event only -- belief state unchanged.

        record_intervention_event(
            session,
            user_id,
            session_id,
            kind=QUESTION_ANSWERED_KIND,
            detail={
                "verdict": verdict,
                "belief_key": belief.key,
                "belief_label": str(detail.get("belief_label") or ""),
            },
        )
        stats.claims_out = 1 if verdict in {"confirm", "deny"} else 0
        logger.info("verification: verdict=%s belief=%s", verdict, belief.key)
        return True
    except Exception:  # noqa: BLE001 -- verification is optional, fail-open
        logger.warning("verification: judgment failed; skipping", exc_info=True)
        try:
            session.rollback()
        except Exception:  # noqa: BLE001 -- the rollback itself must be safe
            pass
        return False
