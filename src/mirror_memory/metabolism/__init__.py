"""Memory metabolism — the long-run lifecycle layer.

Mirror's cognition (evidence → belief → proposal → publisher) answers "what
is true".  Metabolism answers "what stays hot": which beliefs the default
retrieval scan should see, which evidence can be folded into a digest, and
what must never be touched.  It computes; it never writes — every storage
change is a proposal through the Publisher, like every other write.

PR1 shipped the rule-based core (tiers / policy / heat / protection /
eligibility).  PR2 adds the runtime: :mod:`planner` turns heat scores into
tier-transition proposals and hands them to the Publisher; the
``mm_memory_transitions`` audit log records every committed move with its
reason and score.  Evidence compaction (PR3) and archive/forget cascade
(PR4) land on top.
"""

from mirror_memory.metabolism.eligibility import (
    DEFAULT_SCAN_TIERS,
    CURRENT_SCAN_TIERS,
    MODE_ALL,
    MODE_ALL_OCCURRENCES,
    MODE_CURRENT,
    MODE_EVIDENCE,
    MODE_EXPERIENCE,
    MODE_HISTORICAL,
    QUERY_MODES,
    detect_query_mode,
    is_enumeration_query,
    tier_eligible,
    tiers_for_mode,
)
from mirror_memory.metabolism.heat import (
    DEFAULT_HEAT_THRESHOLDS,
    DEFAULT_HEAT_WEIGHTS,
    HeatThresholds,
    HeatWeights,
    heat_score,
    heat_tier,
)
from mirror_memory.metabolism.policy import (
    DEFAULT_RETENTION_POLICIES,
    RETENTION_BEHAVIORAL,
    RETENTION_CANONICAL,
    RETENTION_CLASSES,
    RETENTION_EPISODIC,
    RETENTION_PREFERENCE,
    RETENTION_TRANSIENT,
    RetentionPolicy,
    classify_retention,
    retention_policy_for,
)
from mirror_memory.metabolism.protection import (
    PROTECTED_HEAT_FLOOR,
    is_protected,
    protection_reason,
)
from mirror_memory.metabolism.tiers import (
    DEFAULT_TIER,
    LEGAL_TIER_TRANSITIONS,
    MEMORY_TIERS,
    TIER_ARCHIVED,
    TIER_DORMANT,
    TIER_HOT,
    TIER_WARM,
    is_legal_tier_transition,
    tier_rank,
)

__all__ = [
    "CURRENT_SCAN_TIERS",
    "DEFAULT_HEAT_THRESHOLDS",
    "DEFAULT_HEAT_WEIGHTS",
    "DEFAULT_RETENTION_POLICIES",
    "DEFAULT_SCAN_TIERS",
    "DEFAULT_TIER",
    "HeatThresholds",
    "HeatWeights",
    "LEGAL_TIER_TRANSITIONS",
    "MEMORY_TIERS",
    "MetabolismReport",
    "MODE_ALL",
    "MODE_ALL_OCCURRENCES",
    "MODE_CURRENT",
    "MODE_EVIDENCE",
    "MODE_EXPERIENCE",
    "MODE_HISTORICAL",
    "PROTECTED_HEAT_FLOOR",
    "QUERY_MODES",
    "RETENTION_BEHAVIORAL",
    "RETENTION_CANONICAL",
    "RETENTION_CLASSES",
    "RETENTION_EPISODIC",
    "RETENTION_PREFERENCE",
    "RETENTION_TRANSIENT",
    "RetentionPolicy",
    "TierDecision",
    "TIER_ARCHIVED",
    "TIER_DORMANT",
    "TIER_HOT",
    "TIER_WARM",
    "classify_retention",
    "decide_tier",
    "detect_query_mode",
    "heat_score",
    "heat_tier",
    "is_enumeration_query",
    "is_legal_tier_transition",
    "is_protected",
    "protection_reason",
    "retention_policy_for",
    "tier_eligible",
    "tier_rank",
    "tiers_for_mode",
]


def __getattr__(name: str):
    """Lazily export the planner.

    The planner imports the Publisher (and the repository), which import
    this package's leaf modules back; importing it eagerly from here would
    close a circular-import loop.  Deferring it keeps ``import mirror_memory
    .metabolism`` cheap and cycle-free either way.
    """
    if name in {"MetabolismReport", "TierDecision", "decide_tier", "run_metabolism"}:
        from mirror_memory.metabolism import planner

        return getattr(planner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
