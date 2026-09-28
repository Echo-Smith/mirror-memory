"""Heat score — how alive a belief is, as one number in [0, 1].

Rules only, no model.  The score decides which storage tier a belief
*deserves* (the planner in PR2 turns that into proposals)::

    heat = 0.30 * freshness          exp decay since last use
         + 0.20 * access             log-scaled recall count
         + 0.20 * evidence_strength  support links, saturating at 5
         + 0.15 * authority          user-confirmed beats extracted
         + 0.15 * importance         explicit per-belief score

    > 0.70  hot
    0.40-0.70  warm
    0.15-0.40  dormant
    < 0.15     archive candidate

A protected belief (see :mod:`mirror_memory.metabolism.protection`) is
floored at the hot threshold — protection wins over any age signal, which is
the whole point of protection: a user correction always outranks "this
hasn't been used in months".

Pure functions over belief-like objects; deterministic for the same inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from mirror_memory.core.utils import safe_json

# -- Defaults (mirrored by config/metabolism.yaml; config wins at runtime) ----

DEFAULT_HALF_LIFE_DAYS = 60.0
DEFAULT_EVIDENCE_SATURATION = 5.0   # support_count at which the signal maxes
DEFAULT_ACCESS_SATURATION = 5.0     # log1p(access_count) at which it maxes


@dataclass(frozen=True)
class HeatWeights:
    freshness: float = 0.30
    access: float = 0.20
    evidence_strength: float = 0.20
    authority: float = 0.15
    importance: float = 0.15


@dataclass(frozen=True)
class HeatThresholds:
    hot: float = 0.70
    warm: float = 0.40
    dormant: float = 0.15


DEFAULT_HEAT_WEIGHTS = HeatWeights()
DEFAULT_HEAT_THRESHOLDS = HeatThresholds()

# Authority inputs: a user-confirmed belief carries more weight than an
# extracted hypothesis at the same age.
_CONFIRMED_AUTHORITY = 1.0
_EXTRACTED_AUTHORITY = 0.5
_CONFIRMED_LAYERS = frozenset({"L1", "L2"})
_CONFIRMED_SOURCES = frozenset({"user_confirmed", "user_corrected"})


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def heat_score(
    belief,
    *,
    now: datetime | None = None,
    evidence_count: int | None = None,
    weights: HeatWeights | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    protected: bool = False,
    protected_floor: float = DEFAULT_HEAT_THRESHOLDS.hot,
) -> float:
    """Compute the heat score for *belief*.

    ``evidence_count`` counts typed support links; pass it when the caller
    already has the link counts (the renderer does).  When omitted, the
    score falls back to the belief's legacy ``evidence_json`` id list —
    coarse, but never zero for a genuinely supported belief.

    Freshness anchors on ``last_accessed_at`` when present (a recall is the
    most recent proof the belief matters) and on ``last_evidence_at``
    otherwise, so heat works from day one before any access has been
    recorded.
    """
    ref = now or datetime.now(UTC)
    w = weights or DEFAULT_HEAT_WEIGHTS

    # Freshness: exponential decay since the last time the belief proved
    # itself useful — either by being recalled or by new evidence.
    last_use = _as_utc(getattr(belief, "last_accessed_at", None)) or _as_utc(
        getattr(belief, "last_evidence_at", None)
    )
    if last_use is None:
        freshness = 0.0
    else:
        age_days = max((ref - last_use).total_seconds() / 86400.0, 0.0)
        freshness = math.exp(-age_days / max(half_life_days, 1e-6))

    # Access: log-scaled — the first recall matters most, the fortieth adds
    # almost nothing.
    access_count = int(getattr(belief, "access_count", 0) or 0)
    access = min(math.log1p(access_count) / DEFAULT_ACCESS_SATURATION, 1.0)

    # Evidence strength: saturating support count.
    if evidence_count is None:
        evidence_count = len(
            [mid for mid in safe_json(getattr(belief, "evidence_json", "[]") or "[]")
             if isinstance(mid, int)]
        )
    evidence_strength = min(float(evidence_count) / DEFAULT_EVIDENCE_SATURATION, 1.0)

    # Authority: user-confirmed / user-corrected beats extracted.
    layer = getattr(belief, "layer", "") or ""
    source = getattr(belief, "source", "") or ""
    if layer in _CONFIRMED_LAYERS or source in _CONFIRMED_SOURCES:
        authority = _CONFIRMED_AUTHORITY
    else:
        authority = _EXTRACTED_AUTHORITY

    # Importance is an explicit per-belief knob, so 0.0 is a real value —
    # only a missing/None score falls back to the neutral default.
    raw_importance = getattr(belief, "importance_score", None)
    importance = 0.5 if raw_importance is None else float(raw_importance)

    score = (
        w.freshness * freshness
        + w.access * access
        + w.evidence_strength * evidence_strength
        + w.authority * authority
        + w.importance * importance
    )
    score = min(max(score, 0.0), 1.0)

    if protected:
        score = max(score, protected_floor)
    return round(score, 6)


def heat_tier(score: float, thresholds: HeatThresholds | None = None) -> str:
    """The tier a heat score maps onto.

    Returns ``"archive_candidate"`` below the dormant threshold — archiving
    is a proposal the planner must still make (and protection may veto), not
    something the score does on its own.
    """
    t = thresholds or DEFAULT_HEAT_THRESHOLDS
    if score >= t.hot:
        return "hot"
    if score >= t.warm:
        return "warm"
    if score >= t.dormant:
        return "dormant"
    return "archive_candidate"
