"""Valence / outcome feedback primitive.

Detects positive/negative feedback signals in user text and drives
belief value updates (e.g. neutral → worked). Domain-agnostic: word
lists are loaded from ``config.valence``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def detect_valence(text: str, *, positive: list[str], negative: list[str]) -> str | None:
    """Detect the valence (outcome) signal in *text*.

    Negative words are checked first ("not helpful" should not be
    overridden by "helpful" inside it).

    Returns
    -------
    str or None
        ``"aversive"`` if a negative word is found,
        ``"worked"`` if a positive word is found,
        ``None`` if no valence signal is detected.
    """
    if not text:
        return None

    lowered = text.lower()

    # Negative first — order matters.
    for word in negative:
        if word.lower() in lowered:
            return "aversive"

    for word in positive:
        if word.lower() in lowered:
            return "worked"

    return None


def apply_valence_feedback(
    session: object,
    user_id: str,
    key: str,
    effect: str,
) -> bool:
    """Apply a valence feedback to an existing belief's ``value_json``.

    Updates ``value_json.effect`` from ``neutral`` (or absent) to
    ``worked`` / ``aversive``.  The update is append-only: an existing
    ``worked`` is never overwritten by ``neutral``, but ``aversive``
    can override ``worked`` (and vice-versa) because the user's latest
    feedback takes priority.

    Returns ``True`` if the belief was updated, ``False`` if no
    matching belief was found.
    """
    import json

    from mirror_memory.core.models import Belief

    belief = (
        session.query(Belief)
        .filter(Belief.user_id == user_id, Belief.key == key, Belief.status == "active")
        .first()
    )
    if belief is None:
        return False

    try:
        val = json.loads(belief.value_json or "{}")
    except (TypeError, ValueError):
        val = {}

    current = val.get("effect")
    if current == effect:
        return False  # no change

    # Neutral is the initial state — always overwritten.
    # worked/aversive override each other (latest feedback wins).
    if current is None or current == "neutral" or effect in ("worked", "aversive"):
        val["effect"] = effect
        belief.value_json = json.dumps(val, ensure_ascii=False)
        session.flush()
        logger.info("valence: belief %s effect %s -> %s", key, current, effect)
        return True

    return False
