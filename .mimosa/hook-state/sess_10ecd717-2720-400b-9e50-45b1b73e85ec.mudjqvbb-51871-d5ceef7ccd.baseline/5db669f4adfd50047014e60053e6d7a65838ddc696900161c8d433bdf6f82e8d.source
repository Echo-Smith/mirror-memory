"""Temporal reasoning -- how two facts about the same subject relate in time.

The old rule was ``SINGLE cardinality → UPDATE``: a new value for
``lives_in`` blindly replaced the old one.  That cannot answer "where did
they live before?", because the old value was destroyed rather than closed
off.  Temporal V2 replaces the blind rule with a two-step decision::

    identity relation (same subject + predicate?)
      + temporal relation (does the new fact start after the old one ends?)
      → lifecycle transition

So ``lives_in Shanghai`` with ``valid_to = 2026-05`` followed by
``lives_in Beijing`` with ``valid_from = 2026-05`` produces a
``TEMPORAL_UPDATE``: Shanghai is closed rather than deleted, Beijing becomes
current, and both remain answerable.

This module is pure: it compares intervals and names the transition.  It
never touches the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from mirror_memory.core.utils import coerce_datetime

__all__ = [
    "SCOPE_CURRENT_STATE",
    "SCOPE_EPISODIC",
    "SCOPE_PERSISTENT",
    "LifecycleAction",
    "TemporalRelation",
    "TemporalWindow",
    "coerce_datetime",
    "compare_windows",
    "lifecycle_transition",
]

# -- Temporal scopes (the predicate policy's third axis) ---------------------

# Only one value is true at a time; a later observation closes the earlier
# one.  ``lives_in``, ``works_at``, ``studies_at``.
SCOPE_CURRENT_STATE = "current_state"

# Once true it stays true until something explicitly ends it.  ``has_pet``,
# ``is_married``.  A second, different value is a conflict, not an update.
SCOPE_PERSISTENT = "persistent"

# Each occurrence is its own fact and never supersedes another.  ``went_to``,
# ``attended``, ``bought``.
SCOPE_EPISODIC = "episodic"

TEMPORAL_SCOPES = (SCOPE_CURRENT_STATE, SCOPE_PERSISTENT, SCOPE_EPISODIC)


# -- Interval relations ------------------------------------------------------


class TemporalRelation(str, Enum):
    """How the validity interval of a new fact relates to an existing one."""

    # The new fact starts at or after the old one ends: a genuine handover.
    FOLLOWS = "follows"
    # The new fact ends at or before the old one starts.
    PRECEDES = "precedes"
    # The intervals overlap without one containing the other.
    OVERLAPS = "overlaps"
    # One interval contains the other.
    CONTAINS = "contains"
    # Same interval.
    SAME = "same"
    # At least one interval is open-ended, so ordering cannot be decided.
    UNKNOWN = "unknown"


# -- Lifecycle transitions ---------------------------------------------------


class LifecycleAction(str, Enum):
    """What the belief lifecycle should do with a candidate fact."""

    CREATE = "CREATE"
    SUPPORT = "SUPPORT"
    # Supersede, but close the old interval instead of discarding it.
    TEMPORAL_UPDATE = "TEMPORAL_UPDATE"
    # Same as TEMPORAL_UPDATE but the old value stays current: the new fact is
    # about a different period, not a replacement.
    COEXIST = "COEXIST"
    CONTRADICT = "CONTRADICT"
    NOOP = "NOOP"


@dataclass(frozen=True)
class TemporalWindow:
    """The validity interval of one fact.

    ``None`` bounds mean open-ended: ``valid_from=None`` is "since always"
    and ``valid_to=None`` is "still true".  Open bounds are why the relation
    between two windows is often UNKNOWN rather than ordered.
    """

    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def is_open_ended(self) -> bool:
        return self.valid_from is None or self.valid_to is None

    def is_current(self, at: datetime | None = None) -> bool:
        """Is this window true at *at* (default: now)?

        Bounds are normalised to aware UTC before comparison: beliefs may be
        written by callers using either naive or aware datetimes, and mixing
        the two raises rather than answering.
        """
        reference = _as_utc(at) if at is not None else datetime.now(UTC)
        valid_from = _as_utc(self.valid_from)
        valid_to = _as_utc(self.valid_to)
        if valid_from is not None and reference < valid_from:
            return False
        if valid_to is not None and reference >= valid_to:
            return False
        return True


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalise a datetime to aware UTC; pass ``None`` through."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce(value) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed
    return None


def window_from_belief(belief) -> TemporalWindow:
    """Read a belief's validity interval."""
    return TemporalWindow(
        valid_from=_coerce(getattr(belief, "valid_from", None)),
        valid_to=_coerce(getattr(belief, "valid_to", None)),
    )


def compare_windows(new: TemporalWindow, old: TemporalWindow) -> TemporalRelation:
    """Relate *new* to *old*.

    Only the bounds a given decision actually needs are required.  "The new
    fact started in May 2026 and is still true" against "the old fact ended in
    May 2026" is a decidable handover even though the new window has no end,
    so demanding both ends would throw away the answer.

    Returns :attr:`TemporalRelation.UNKNOWN` when the needed bounds are
    missing -- an assumption here would silently pick the wrong transition.
    """
    new_start, new_end = _as_utc(new.valid_from), _as_utc(new.valid_to)
    old_start, old_end = _as_utc(old.valid_from), _as_utc(old.valid_to)

    if new_start is None and new_end is None:
        return TemporalRelation.UNKNOWN
    if old_start is None and old_end is None:
        return TemporalRelation.UNKNOWN

    # Both windows fully bounded: the relation is decidable outright.
    if new_start is not None and new_end is not None and old_start is not None and old_end is not None:
        if new_start == old_start and new_end == old_end:
            return TemporalRelation.SAME
        if new_start >= old_end:
            return TemporalRelation.FOLLOWS
        if new_end <= old_start:
            return TemporalRelation.PRECEDES
        if new_start <= old_start and new_end >= old_end:
            return TemporalRelation.CONTAINS
        return TemporalRelation.OVERLAPS

    # Partially bounded: a handover only needs the new start and the old end.
    if new_start is not None and old_end is not None:
        if new_start >= old_end:
            return TemporalRelation.FOLLOWS
        # The new fact began while the old one was still true.
        return TemporalRelation.OVERLAPS
    # The reverse only needs the new end and the old start.
    if new_end is not None and old_start is not None and new_end <= old_start:
        return TemporalRelation.PRECEDES

    return TemporalRelation.UNKNOWN


def lifecycle_transition(
    *,
    same_subject: bool,
    same_predicate: bool,
    same_object: bool,
    scope: str,
    relation: TemporalRelation,
    claims_contradiction: bool = False,
) -> LifecycleAction:
    """Decide the lifecycle action from identity + temporal relation.

    This is the replacement for ``SINGLE → UPDATE``.  The identity question
    ("is this the same subject and predicate?") and the temporal question
    ("does the new fact start after the old one ends?") are answered
    separately, and only their combination picks the transition.
    """
    if not same_subject or not same_predicate:
        return LifecycleAction.CREATE

    if same_object:
        # Re-stating the same fact: support it, unless the claimant says the
        # opposite, in which case the conflict wins.
        return LifecycleAction.CONTRADICT if claims_contradiction else LifecycleAction.SUPPORT

    if scope == SCOPE_CURRENT_STATE:
        # A contradiction aimed at a *different* value of a current-state
        # predicate is an assertion that the old value no longer holds -- a
        # handover, not a standing conflict.  "I don't live in Shanghai, I
        # live in Beijing" replaces; it does not dispute.
        if relation is TemporalRelation.FOLLOWS:
            # A genuine handover: close the old interval, open the new one.
            return LifecycleAction.TEMPORAL_UPDATE
        if relation in (TemporalRelation.PRECEDES, TemporalRelation.OVERLAPS, TemporalRelation.CONTAINS):
            # Different period or an unresolved overlap: keep both alive.
            return LifecycleAction.COEXIST
        # No interval evidence on either side.  For a current_state predicate
        # that is a re-statement of the present with a new value, so supersede
        # -- the old behaviour -- but the caller only closes the old interval
        # when the new fact actually carries a start time.
        return LifecycleAction.TEMPORAL_UPDATE

    if claims_contradiction:
        return LifecycleAction.CONTRADICT

    if scope == SCOPE_EPISODIC:
        # Every occurrence is its own fact.
        return LifecycleAction.CREATE

    if scope == SCOPE_PERSISTENT:
        # A persistent predicate with a different object is a conflict about
        # the present, not a replacement of a past state.
        return LifecycleAction.CONTRADICT

    return LifecycleAction.CREATE
