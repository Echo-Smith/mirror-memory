"""Belief retrieval scoring -- generic weighted scoring, zero domain coupling.

Each belief is scored by multiple signals; the total determines its rank in
the retrieval set.  A score of 0 means "not relevant" and the belief is
filtered out.

Scoring signals:
- Topic hit (key in current topics): +5.0
- Predicate hit (belief.predicate matches query predicates): +4.0
- Object hit (belief.object matches query objects): +3.0
- Layer confirmed (L1/L2): +3.0
- Activity (confidence * decay * layer_factor): +2.0 * activity
- Dimension boost (configurable): +N
- Multi-evidence bonus (evidence_count >= 3): +1.0
- Freshness (new evidence within 2h): +1.0
"""

from __future__ import annotations

from datetime import UTC, datetime

from mirror_memory.core.activity import belief_activity
from mirror_memory.core.constants import FRESHNESS_THRESHOLD_HOURS, MULTI_EVIDENCE_THRESHOLD

# -- Scoring weights ---------------------------------------------------------

_SCORE_TOPIC_HIT = 5.0          # key matches a current-topic token
_SCORE_PREDICATE_HIT = 4.0      # belief.predicate matches a query predicate
_SCORE_OBJECT_HIT = 3.0         # belief.object matches a query object
_SCORE_LAYER_CONFIRMED = 3.0    # L1/L2 user-confirmed beliefs
_SCORE_ACTIVITY_WEIGHT = 2.0    # activity score weight
_SCORE_EVIDENCE_BONUS = 1.0     # multi-evidence support (count >= 3)
_SCORE_FRESHNESS = 1.0          # new evidence in the last 2 hours


def score_belief(
    belief,
    *,
    topics: set[str] | None = None,
    query_predicates: set[str] | None = None,
    query_objects: set[str] | None = None,
    dimension_boosts: dict[str, float] | None = None,
    now: datetime | None = None,
) -> float:
    """Multi-signal weighted score for a single belief.

    Parameters
    ----------
    belief:
        Any object with ``dimension`` (str), ``key`` (str), ``layer`` (str),
        ``confidence`` (float), ``value_json`` (str), ``last_evidence_at``
        (datetime | None), ``predicate`` (str), ``object`` (str) attributes.
    topics:
        Set of topic tokens extracted from the current user message.
    query_predicates:
        Set of predicates extracted from the query (e.g. ``{"likes"}``).
    query_objects:
        Set of objects extracted from the query (e.g. ``{"coffee"}``).
    dimension_boosts:
        Mapping of ``dimension -> float`` added when the belief's dimension
        matches.
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

    # Predicate hit (cognitive triple matching)
    pred = getattr(belief, "predicate", "") or ""
    if query_predicates and pred in query_predicates:
        score += _SCORE_PREDICATE_HIT

    # Object hit (cognitive triple matching)
    obj = getattr(belief, "object", "") or ""
    if query_objects and obj in query_objects:
        score += _SCORE_OBJECT_HIT

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
