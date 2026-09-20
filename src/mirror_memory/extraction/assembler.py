"""Claim priority assembler -- scarce dimensions first, cap fill.

Reorders and truncates the combined K1+K2 claim set before persistence,
activating ``config.extraction.max_claims_per_turn`` and
``DimensionConfig.render_priority`` (both previously dead config).
"""

from __future__ import annotations

import logging

from mirror_memory.config.schema import MemoryConfig

logger = logging.getLogger(__name__)


def assemble_claims(
    claims: list[dict],
    config: MemoryConfig,
    *,
    active_dimension_counts: dict[str, int] | None = None,
) -> list[dict]:
    """Assemble the final claim set for persistence.

    Priority order:
    1. Scarce dimensions first (fewer active beliefs = more novel information)
    2. Higher render_priority dimensions first within the same scarcity band
    3. Higher confidence first as tiebreaker

    Truncates to ``config.extraction.max_claims_per_turn``.

    Parameters
    ----------
    claims:
        Combined K1+K2 claims for this turn.
    config:
        The ``MemoryConfig``.
    active_dimension_counts:
        Mapping of dimension -> count of active beliefs (from DB).
        If None, all dimensions treated as equally scarce.

    Returns
    -------
    list[dict]
        Ordered, truncated claim list.
    """
    if not claims:
        return []

    counts = active_dimension_counts or {}
    max_claims = config.extraction.max_claims_per_turn

    # Build render_priority lookup.
    priorities = {d.dimension_id: d.render_priority for d in config.dimensions}

    def _sort_key(claim: dict) -> tuple:
        dim = claim.get("dimension", "")
        scarcity = counts.get(dim, 0)
        priority = priorities.get(dim, 0)
        confidence = claim.get("confidence", 0.0)
        # Bonus for cognitive triple structure (predicate+object present).
        value = claim.get("value") or {}
        has_triple = 1 if (value.get("predicate") and value.get("object")) else 0
        # Ascending sort: lower scarcity (fewer existing) = comes first.
        # has_triple negated: claims with triples come first.
        # priority/confidence negated: higher values come first.
        return (scarcity, -has_triple, -priority, -confidence)

    ordered = sorted(claims, key=_sort_key)
    result = ordered[:max_claims]
    if len(ordered) > max_claims:
        logger.info("assembler: %d claims -> %d (max_claims_per_turn)", len(ordered), max_claims)
    return result
