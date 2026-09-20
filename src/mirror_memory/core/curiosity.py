"""Curiosity signal module — three generic signal types for exploration.

Detects situations where the agent should ask the user for more
information rather than assuming. All signals are domain-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any

from mirror_memory.core.constants import (
    LOW_CONFIDENCE_PATTERN_CEIL,
    LOW_CONFIDENCE_PATTERN_FLOOR,
    PROFILE_SIGNAL_THRESHOLD,
)

logger = logging.getLogger(__name__)


def detect_curiosity_signals(
    session: object,
    user_id: str,
    *,
    current_topics: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Detect curiosity signals for the current turn.

    Returns up to 3 signals, one per type (first match wins per type).

    Parameters
    ----------
    session:
        SQLAlchemy database session.
    user_id:
        The user identifier.
    current_topics:
        Topics detected in the current user message.  If provided,
        the "new_topic" signal checks for topics not yet covered by
        existing beliefs.
    """
    from mirror_memory.core.models import Belief
    from mirror_memory.core.utils import safe_json

    signals: list[dict[str, Any]] = []

    # 1. needs_clarification — beliefs the agent is unsure about.
    clarified = (
        session.query(Belief)
        .filter(
            Belief.user_id == user_id,
            Belief.status == "active",
            Belief.confidence >= PROFILE_SIGNAL_THRESHOLD,
        )
        .all()
    )
    for b in clarified:
        val = safe_json(b.value_json)
        if val.get("clarification_status") == "needs_clarification":
            signals.append({
                "type": "needs_clarification",
                "belief_key": b.key,
                "belief_label": b.claim_text[:80],
                "detail": f"confidence={b.confidence:.2f}",
            })
            break  # one per type

    # 2. Low-confidence pattern — beliefs in the curiosity range.
    if not any(s["type"] == "low_confidence" for s in signals):
        low_conf = (
            session.query(Belief)
            .filter(
                Belief.user_id == user_id,
                Belief.status == "active",
                Belief.confidence >= LOW_CONFIDENCE_PATTERN_FLOOR,
                Belief.confidence < LOW_CONFIDENCE_PATTERN_CEIL,
            )
            .all()
        )
        for b in low_conf:
            signals.append({
                "type": "low_confidence",
                "belief_key": b.key,
                "belief_label": b.claim_text[:80],
                "detail": f"confidence={b.confidence:.2f} (in curiosity range)",
            })
            break

    # 3. New topic — current message mentions topics not yet covered.
    if current_topics and not any(s["type"] == "new_topic" for s in signals):
        known_keys = {
            b.key
            for b in session.query(Belief.key)
            .filter(Belief.user_id == user_id, Belief.status == "active", Belief.dimension == "topic")
            .all()
        }
        novel = current_topics - known_keys
        if novel:
            signals.append({
                "type": "new_topic",
                "belief_key": "",
                "belief_label": ", ".join(sorted(novel)[:3]),
                "detail": f"{len(novel)} new topic(s) not yet explored",
            })

    return signals
