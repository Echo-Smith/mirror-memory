"""Panel data assembly -- domain-agnostic.

Builds a structured dict for UI panel display.  All dimension
definitions and display labels come from ``config``.
"""

from __future__ import annotations

import json
import logging

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.render.display import DisplayDict

logger = logging.getLogger(__name__)


def build_panel(
    session: object,
    user_id: str,
    config: MemoryConfig,
    *,
    language: str = "zh",
) -> dict:
    """Build panel data for the user's memory profile.

    Parameters
    ----------
    session:
        SQLAlchemy database session.
    user_id:
        The user identifier.
    config:
        The ``MemoryConfig`` providing dimensions, display labels,
        and render parameters.
    language:
        ``"zh"`` or ``"en"``.

    Returns
    -------
    dict
        A dict with keys:
        - ``memory``: ``{enabled: bool, has_data: bool}``
        - ``sections``: list of section dicts, each with
          ``dimension``, ``header``, ``items``.
    """
    from mirror_memory.core.repository import (
        is_memory_enabled,
        list_active_beliefs,
        list_rejected_beliefs,
    )

    display = DisplayDict(config)
    enabled = is_memory_enabled(session, user_id)

    if not enabled:
        return {
            "memory": {"enabled": False, "has_data": _has_data(session, user_id)},
            "sections": [],
        }

    sections: list[dict] = []
    seen_dimensions: set[str] = set()

    # Iterate configured dimensions (all dimensions are shown in panel by default).
    for dim in config.dimensions:
        dim_id = dim.dimension_id
        header = display.dimension_header(dim_id, language)
        items: list[dict] = []

        beliefs = list_active_beliefs(session, user_id, dimensions=(dim_id,), limit=10)
        for belief in beliefs:
            item = _item_for(belief, display, language)
            if item:
                items.append(item)

        if header and items:
            seen_dimensions.add(dim_id)
            sections.append({
                "dimension": dim_id,
                "header": header,
                "items": items,
            })

    # Rejected beliefs (boundary items)
    rejected = list_rejected_beliefs(session, user_id, limit=5)
    if rejected:
        boundary_labels = [
            label
            for label in (display.friendly_label(b.key, language) for b in rejected)
            if label
        ]
        if boundary_labels:
            is_en = language.strip().lower() == "en"
            sep = ", " if is_en else "\u3001"
            sections.append({
                "dimension": "rejected",
                "header": display.dimension_header("rejected", language),
                "items": [{
                    "key": "boundary.avoided_topics",
                    "label": sep.join(boundary_labels),
                    "detail": "",
                }],
            })

    return {
        "memory": {
            "enabled": True,
            "has_data": _has_data(session, user_id),
        },
        "sections": sections,
    }


def _item_for(belief: object, display: DisplayDict, language: str) -> dict | None:
    """Convert a single belief to a panel item dict, or ``None``
    if it should not appear in the panel.

    Rules:
    - Must have a display label (no raw key leakage).
    - L4 beliefs below the render threshold are excluded.
    """
    label = display.friendly_label(belief.key, language)
    if not label:
        return None

    # Skip low-confidence L4 beliefs
    if belief.layer == "L4" and belief.confidence < 0.55:
        return None

    return {"key": belief.key, "label": label, "detail": ""}


def _has_data(session: object, user_id: str) -> bool:
    """Check if the user has any memory data at all."""
    try:
        from mirror_memory.core.models import Belief

        count = (
            session.query(Belief)
            .filter(Belief.user_id == user_id)
            .limit(1)
            .count()
        )
        return count > 0
    except Exception:
        return False
