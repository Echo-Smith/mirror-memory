"""Belief retrieval scoring -- generic weighted scoring, zero domain coupling.

Each belief is scored by multiple signals; the total determines its rank in
the retrieval set.  A score of 0 means "not relevant" and the belief is
filtered out.

Dimension-specific boosts (e.g. risk-aware boost for a safety dimension)
are loaded from configuration rather than hard-coded.  The caller passes a
``dimension_boosts`` mapping of ``dimension -> float`` that is added when
the belief's dimension matches.

Scoring signals:
- Layer confirmed (L1/L2): +3.0
- Activity (confidence * decay * layer_factor): +2.0 * activity
- Multi-evidence bonus (evidence_count >= 3): +1.0
- Freshness (new evidence within 2h): +1.0
- Topic hit (key in current topics): +5.0  [caller provides topics]
- Dimension boost (configurable): +N    [caller provides boost map]
"""

from __future__ import annotations

from datetime import UTC, datetime

from mirror_memory.core.activity import belief_activity
from mirror_memory.core.constants import FRESHNESS_THRESHOLD_HOURS, MULTI_EVIDENCE_THRESHOLD

# -- Scoring weights ---------------------------------------------------------

_SCORE_LAYER_CONFIRMED = 3.0   # L1/L2 user-confirmed beliefs
_SCORE_ACTIVITY_WEIGHT = 2.0   # activity score weight
_SCORE_EVIDENCE_BONUS = 1.0   # multi-evidence support (count >= 3)
_SCORE_FRESHNESS = 1.0        # new evidence in the last 2 hours
_SCORE_TOPIC_HIT = 5.0        # key matches a current-topic token


def score_belief(
    belief,
    *,
    topics: set[str] | None = None,
    dimension_boosts: dict[str, float] | None = None,
    now: datetime | None = None,
) -> float:
    """Multi-signal weighted score for a single belief.

    Parameters
    ----------
    belief:
        Any object with ``dimension`` (str), ``key`` (str), ``layer`` (str),
        ``confidence`` (float), ``value_json`` (str), ``last_evidence_at``
        (datetime | None) attributes.
    topics:
        Set of topic tokens extracted from the current user message.
        If ``belief.key`` is in this set, the topic-hit bonus applies.
    dimension_boosts:
        Mapping of ``dimension -> float`` added when the belief's dimension
        matches.  Use this for domain-specific boosts (e.g. risk-aware
        weighting) without hard-coding them.
    now:
        Reference time; defaults to ``datetime.now(UTC)``.

    Returns
    -------
    float
        Score >= 0.  Zero means "not relevant".
    """
    score = 0.0

    # Topic hit
    if topics and belief.key in topics:
        score += _SCORE_TOPIC_HIT

    # Dimension boost (configurable, caller-provided)
    boosts = dimension_boosts or {}
    dim_boost = boosts.get(belief.dimension)
    if dim_boost is not None:
        score += dim_boost

    # Layer confirmation
    if belief.layer in ("L1", "L2"):
        score += _SCORE_LAYER_CONFIRMED

    # Activity score (confidence * time decay * layer factor)
    activity = belief_activity(belief, now=now)
    score += activity * _SCORE_ACTIVITY_WEIGHT

    # Multi-evidence bonus
    from mirror_memory.core.utils import safe_json

    val = safe_json(belief.value_json)
    evidence_count = len(val.get("evidence_ids") or [])
    if evidence_count >= MULTI_EVIDENCE_THRESHOLD:
        score += _SCORE_EVIDENCE_BONUS

    # Freshness: evidence within the last 2 hours
    ref_now = now or datetime.now(UTC)
    if belief.last_evidence_at:
        last = belief.last_evidence_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        hours_since = (ref_now - last).total_seconds() / 3600
        if hours_since < FRESHNESS_THRESHOLD_HOURS:
            score += _SCORE_FRESHNESS

    return round(score, 4)
