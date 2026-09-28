"""Storage tiers and the tier state machine.

A belief carries two orthogonal state dimensions:

- **truth state** — ``status`` / ``valid_from`` / ``valid_to``: whether the
  claim is current, superseded, or rejected.  Owned by the identity /
  temporal lifecycle; metabolism never touches it.
- **storage state** — ``memory_tier`` / ``metabolism_state``: how hot the
  belief's storage is.  "superseded + warm" means "no longer the current
  value, but historically true and plausibly useful soon" — history that
  stays reachable instead of being deleted or frozen into the present.

Tier transitions are decided by the metabolism planner and committed by the
Publisher like any other write.  This module only names the states and the
legal edges::

    HOT ──► WARM ──► DORMANT ──► ARCHIVED ──► (purge: physical delete)
      ▲       │           │
      └───────┴───────────┘   recall / new support / user confirmation reheat

ARCHIVED is not a dead end: an explicit restore proposal (user asking about
the past, or the archive index hitting) moves a belief back to WARM.
"""

from __future__ import annotations

TIER_HOT = "hot"
TIER_WARM = "warm"
TIER_DORMANT = "dormant"
TIER_ARCHIVED = "archived"

MEMORY_TIERS = (TIER_HOT, TIER_WARM, TIER_DORMANT, TIER_ARCHIVED)

# Every new belief starts hot; only maintenance proposals move it down.
DEFAULT_TIER = TIER_HOT

# metabolism_state — the runtime's own bookkeeping, orthogonal to the tier.
# ``protected`` is derived (see metabolism.protection), not assigned here.
METABOLISM_ACTIVE = "active"
METABOLISM_COOLING = "cooling"
METABOLISM_COMPACTED = "compacted"
METABOLISM_PROTECTED = "protected"

METABOLISM_STATES = (
    METABOLISM_ACTIVE,
    METABOLISM_COOLING,
    METABOLISM_COMPACTED,
    METABOLISM_PROTECTED,
)

# Rank used for "which direction is this transition going?" — cooling moves
# down, reheat moves up.  Purge is not a tier transition (it is a delete).
_TIER_ORDER = {TIER_HOT: 0, TIER_WARM: 1, TIER_DORMANT: 2, TIER_ARCHIVED: 3}

# Legal edges of the tier state machine.  Anything not listed is refused by
# the planner (and, later, by the Publisher's invariant check): a belief must
# cool through warm rather than jumping straight to dormant, and archived
# data comes back through an explicit restore, never silently.
LEGAL_TIER_TRANSITIONS = frozenset(
    {
        (TIER_HOT, TIER_WARM),        # cool
        (TIER_WARM, TIER_HOT),        # reheat (new evidence / user action)
        (TIER_WARM, TIER_DORMANT),    # cool
        (TIER_DORMANT, TIER_HOT),     # reheat (recalled / supported again)
        (TIER_DORMANT, TIER_ARCHIVED),  # archive
        (TIER_ARCHIVED, TIER_WARM),   # explicit restore
    }
)


def tier_rank(tier: str) -> int:
    """Ordinal of *tier*; lower is hotter.  Unknown tiers sort as archived."""
    return _TIER_ORDER.get(tier, _TIER_ORDER[TIER_ARCHIVED])


def is_legal_tier_transition(from_tier: str, to_tier: str) -> bool:
    """Is ``from_tier -> to_tier`` an edge of the tier state machine?"""
    return (from_tier, to_tier) in LEGAL_TIER_TRANSITIONS
