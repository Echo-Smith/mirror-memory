"""Polarity and goal lifecycle — structured state, not word lists.

The read-side withdrawal filter was a word list: it caught "avoided" and
"gave up" and missed "coffee is off the table".  This module replaces it with
a structured model:

- ``infer_polarity`` maps a canonical predicate onto positive / negative /
  neutral.  ``likes`` and ``dislikes`` about the same object are opposite
  polarities, and only one can be current.
- ``infer_lifecycle`` maps a claim onto a goal state transition.  A goal is
  ``active``, ``paused``, ``cancelled`` or ``resumed``; "I gave up" then "I
  started again" is a resumption of the same goal, not a new one.

Both are pure functions over the canonicalised triple, so the write side can
close the superseded state at the moment it happens instead of guessing later.
"""

from __future__ import annotations

# Predicates whose two halves are opposite polarities of one attribute.
_POSITIVE_PREDICATES = frozenset({
    "likes", "loves", "enjoys", "prefers", "wants", "wants_to", "likes_to",
    "values", "appreciates", "admires",
})
_NEGATIVE_PREDICATES = frozenset({
    "dislikes", "hates", "avoids", "rejects", "dislikes_to", "disprefers",
})
_NEUTRAL_PREDICATES = frozenset({
    "has", "owns", "lives_in", "works_at", "studies_at", "went_to", "bought",
    "is", "says", "knows", "attended", "experienced", "profession", "age",
})

POLARITY_POSITIVE = "positive"
POLARITY_NEGATIVE = "negative"
POLARITY_NEUTRAL = "neutral"

# Goal lifecycle states.
GOAL_ACTIVE = "active"
GOAL_PAUSED = "paused"
GOAL_CANCELLED = "cancelled"
GOAL_RESUMED = "resumed"

# Predicates that denote a goal or intention rather than a settled fact.
_GOAL_PREDICATES = frozenset({
    "wants_to", "wants", "plans_to", "hopes_to", "intends_to", "aims_to",
    "would_like_to", "wishes_to",
})

# Phrases that pause or cancel a goal, and those that resume one.
_CANCEL_MARKERS = (
    "gave up", "give up", "abandoned", "dropped the idea", "cancel",
    "no longer want", "don't want", "do not want", "quit", "scrapped",
    "not going to", "decided against",
)
_PAUSE_MARKERS = (
    "put on hold", "paused", "shelved", "postponed", "on hold", "deferred",
    "not right now", "later",
)
_RESUME_MARKERS = (
    "started again", "restarted", "resumed", "picked it up again", "back to",
    "renewed", "revisiting", "trying again", "started training again",
    "signed up again", "bought a", "enrolled again", "booked",
)


def infer_polarity(predicate: str) -> str:
    """Map a canonical predicate onto positive / negative / neutral."""
    p = (predicate or "").strip().lower()
    if p in _POSITIVE_PREDICATES:
        return POLARITY_POSITIVE
    if p in _NEGATIVE_PREDICATES:
        return POLARITY_NEGATIVE
    return POLARITY_NEUTRAL


def is_goal_predicate(predicate: str) -> bool:
    p = (predicate or "").strip().lower()
    return p in _GOAL_PREDICATES


def infer_lifecycle(predicate: str, claim_text: str, *, prior_state: str = "") -> str:
    """Infer a goal's lifecycle state from the claim and any prior state.

    An empty return means "no goal transition" -- the claim is either not a
    goal or is an ordinary restatement of an active goal.
    """
    if not is_goal_predicate(predicate):
        return ""
    text = (claim_text or "").lower()
    if any(marker in text for marker in _RESUME_MARKERS):
        return GOAL_RESUMED
    if any(marker in text for marker in _CANCEL_MARKERS):
        return GOAL_CANCELLED
    if any(marker in text for marker in _PAUSE_MARKERS):
        return GOAL_PAUSED
    # A goal stated with no withdrawal or resumption marker is (re)affirmed.
    if prior_state in (GOAL_CANCELLED, GOAL_PAUSED):
        return GOAL_RESUMED
    return GOAL_ACTIVE


def opposite_polarity(polarity: str) -> str:
    """The polarity that conflicts with *polarity*, or ``""`` when none."""
    if polarity == POLARITY_POSITIVE:
        return POLARITY_NEGATIVE
    if polarity == POLARITY_NEGATIVE:
        return POLARITY_POSITIVE
    return ""
