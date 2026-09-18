"""Belief activity score -- pure function, zero domain coupling.

Activity = confidence * time_decay * layer_factor.

- Time decay: ``0.5 ** (age_days / half_life)`` -- beliefs with fresh
  evidence stay near the top; old beliefs sink without being deleted.
- Layer factor: L1/L2 (user-confirmed) = 1.0, L4 (hypothesis) = 0.7.
  Same confidence, confirmed beliefs sort before unconfirmed.
- Output is in [0, 1] (confidence <= 0.95).

Deterministic: same input -> same output.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mirror_memory.core.constants import ACTIVITY_HALF_LIFE_DAYS

# Layer activity factors.
_CONFIRMED_LAYER_ACTIVITY = 1.0   # L1 / L2 -- user has claimed this belief
_HYPOTHESIS_LAYER_ACTIVITY = 0.7  # L4 -- pending verification


def belief_activity(
    belief,
    *,
    now: datetime | None = None,
    half_life_days: float = ACTIVITY_HALF_LIFE_DAYS,
) -> float:
    """Return the current activity score for *belief* (pure function).

    Parameters
    ----------
    belief:
        Any object with ``confidence`` (float), ``layer`` (str), and
        ``last_evidence_at`` (datetime | None) attributes.  Works with both
        the ORM model and lightweight stand-ins.
    now:
        Reference time; defaults to ``datetime.now(UTC)``.
    half_life_days:
        Number of days after which activity halves.
    """
    current = now or datetime.now(UTC)
    last = belief.last_evidence_at
    if last is None:
        return 0.0
    # SQLite via SQLAlchemy may return naive datetimes -- interpret as UTC.
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    age_days = max((current - last).total_seconds() / 86400.0, 0.0)
    decay = 0.5 ** (age_days / max(half_life_days, 1e-6))
    layer_factor = _CONFIRMED_LAYER_ACTIVITY if belief.layer in {"L1", "L2"} else _HYPOTHESIS_LAYER_ACTIVITY
    return round(float(belief.confidence) * decay * layer_factor, 6)
