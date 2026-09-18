"""Confidence weight computation -- pure function, zero domain coupling.

The 7-factor weight modulates how much each supporting evidence item pushes
a belief's confidence upward.  Design: base = 1.0 (full gain), negative
signals subtract; floor = 0.5 (new beliefs never stagnate entirely).
"""

from __future__ import annotations

import json


def compute_confidence_weight(
    *,
    source: str = "extracted",
    layer: str = "L4",
    support_count: int = 0,
    contradict_count: int = 0,
    context_diversity: int = 0,
    days_since_last_evidence: float = 0.0,
    value_json: str = "{}",
) -> float:
    """7-factor confidence weight in [0.5, 1.0].

    Parameters
    ----------
    source:
        Provenance of the belief (e.g. ``"user_confirmed"``,
        ``"user_stated"``, ``"program"``, ``"extracted"``).
    layer:
        Confirmation layer (``"L1"`` through ``"L4"``).
    support_count:
        Total number of distinct supporting evidence items.
    contradict_count:
        Number of contradict/downgrade events.
    context_diversity:
        Number of distinct context tags attached to the belief.
    days_since_last_evidence:
        Days elapsed since the most recent evidence item.
    value_json:
        JSON string of the belief's structured value; checked for a
        ``"clarification_status"`` key (generic hook, not domain-specific).
    """
    weight = 1.0

    # 1. Source credibility
    weight -= {
        "user_confirmed": 0.0,
        "user_stated": 0.05,
        "program": 0.05,
        "extracted": 0.1,
    }.get(source, 0.1)

    # 2. Confirmation layer
    weight -= {"L1": 0.05, "L2": 0.0, "L3": 0.1, "L4": 0.1}.get(layer, 0.1)

    # 3. Evidence accumulation (0 items -> -0.15; 3+ -> no penalty)
    if support_count < 3:
        weight -= 0.15 * (1 - support_count / 3)

    # 4. Context diversity (penalise only when evidence is sufficient)
    if support_count >= 2 and context_diversity < 3:
        weight -= 0.1 * (1 - context_diversity / 3)

    # 5. Recency (30+ days -> -0.15; 7 days -> no penalty)
    if days_since_last_evidence > 7:
        weight -= min(0.15, 0.05 * (days_since_last_evidence - 7) / 23)

    # 6. Contradiction penalty (each event -> -0.1)
    weight -= 0.1 * contradict_count

    # 7. Clarification-needed penalty (generic hook)
    from mirror_memory.core.utils import safe_json

    val = safe_json(value_json)
    if val.get("clarification_status") == "needs_clarification":
        weight -= 0.1

    return round(min(1.0, max(0.5, weight)), 4)
