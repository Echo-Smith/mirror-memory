"""Retention classes — the rule-based half of memory metabolism.

Every belief belongs to one *retention class* that fixes how aggressively
the metabolism runtime may cool, compact, and archive it::

    canonical   name, job, employer, long-lived facts   never auto-archived
    preference  likes, formats, tastes                  cool slowly
    behavioral  goals, habits, patterns                 strong decay
    episodic    a meeting, a trip, an occurrence        compact then archive
    transient   short-lived task state                  TTL

The class is derived from what the engine already knows about a claim —
cardinality, temporal scope, goal lifecycle — so no model call and no new
predicate table is required for the default mapping.  Explicit per-predicate
overrides come from ``metabolism.yaml`` (``retention_predicates``).

Archive-after-days is *not* deletion: it means "default retrieval no longer
scans it".  Physical deletion is a separate, user-driven cascade.
"""

from __future__ import annotations

from dataclasses import dataclass

RETENTION_CANONICAL = "canonical"
RETENTION_PREFERENCE = "preference"
RETENTION_BEHAVIORAL = "behavioral"
RETENTION_EPISODIC = "episodic"
RETENTION_TRANSIENT = "transient"

RETENTION_CLASSES = (
    RETENTION_CANONICAL,
    RETENTION_PREFERENCE,
    RETENTION_BEHAVIORAL,
    RETENTION_EPISODIC,
    RETENTION_TRANSIENT,
)


@dataclass(frozen=True)
class RetentionPolicy:
    """How long a belief of one class may stay hot before cooling.

    ``None`` means "no automatic cooling/archiving at this stage" — the
    canonical class is never moved by the background runtime on age alone.
    """

    cool_after_days: int | None
    archive_after_days: int | None


# Days are deliberately generous: cooling too early loses answers, cooling
# too late only costs storage.  Every value here is overridable from
# ``config/metabolism.yaml``.
DEFAULT_RETENTION_POLICIES: dict[str, RetentionPolicy] = {
    RETENTION_CANONICAL: RetentionPolicy(cool_after_days=None, archive_after_days=None),
    RETENTION_PREFERENCE: RetentionPolicy(cool_after_days=180, archive_after_days=720),
    RETENTION_BEHAVIORAL: RetentionPolicy(cool_after_days=60, archive_after_days=180),
    RETENTION_EPISODIC: RetentionPolicy(cool_after_days=30, archive_after_days=90),
    RETENTION_TRANSIENT: RetentionPolicy(cool_after_days=7, archive_after_days=30),
}

# Goal / intention predicates: they describe what the user is moving toward,
# so their value decays with progress rather than with evidence age.
DEFAULT_BEHAVIORAL_PREDICATES = frozenset({"wants_to", "plans_to", "hopes_to"})

# Fallback for claims that carry no usable signal.  Deliberately mild: a
# wrong "transient" (7-day cooling) is a lost answer, a wrong "preference"
# (180-day) is only a slower cooldown.
DEFAULT_RETENTION_CLASS = RETENTION_PREFERENCE


def classify_retention(
    *,
    predicate: str = "",
    cardinality: str = "multi",
    lifecycle_state: str = "",
    overrides: dict[str, str] | None = None,
) -> str:
    """Map a claim onto a retention class (pure, rule-based, no I/O).

    Parameters
    ----------
    predicate / cardinality / lifecycle_state:
        The identity fields the claim already carries.
    overrides:
        Optional ``predicate -> class`` map from ``metabolism.yaml``; an
        explicit override beats every derived rule.

    The default mapping reuses the identity axes instead of a second
    predicate table:

    - explicit override → that class
    - a goal lifecycle state (active/paused/cancelled/resumed) → behavioral
    - a goal predicate (``wants_to`` & co.) → behavioral
    - event cardinality → episodic (each occurrence is its own fact)
    - single cardinality → canonical (one current value: lives_in, works_at)
    - anything else → preference
    """
    if overrides:
        override = overrides.get(predicate)
        if override in RETENTION_CLASSES:
            return override
    if lifecycle_state:
        return RETENTION_BEHAVIORAL
    if predicate in DEFAULT_BEHAVIORAL_PREDICATES:
        return RETENTION_BEHAVIORAL
    if cardinality == "event":
        return RETENTION_EPISODIC
    if cardinality == "single":
        return RETENTION_CANONICAL
    return DEFAULT_RETENTION_CLASS


def retention_policy_for(
    retention_class: str,
    policies: dict[str, RetentionPolicy] | None = None,
) -> RetentionPolicy:
    """Policy for *retention_class*; unknown classes get the preference policy."""
    table = policies or DEFAULT_RETENTION_POLICIES
    return table.get(retention_class, table[RETENTION_PREFERENCE])
