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

import re

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


def goal_predicates() -> frozenset[str]:
    """The predicate set that denotes a goal or intention."""
    return frozenset(_GOAL_PREDICATES)


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


# Phrases that retract an earlier statement.  These are meta-linguistic --
# they talk about the conversation, not about the fact -- so matching them is
# not the "negation word carries fact-model responsibility" anti-pattern the
# polarity model replaced.  A correction is the user withdrawing their own
# earlier claim, which is a different transition from a source conflict.
_CORRECTION_MARKERS = (
    "correction",
    "i was wrong",
    "i mistook",
    "i misspoke",
    "that was mistaken",
    "that was incorrect",
    "i need to correct",
    "let me correct",
    "i take that back",
    "i take it back",
    "scratch that",
    "on second thought",
    "to correct myself",
    "i correct myself",
    # Substantive negations of a prior positive claim.  These are not
    # meta-linguistic, but they retract exactly what a correction does:
    # "I have never learned Rust" withdraws "I can code in Rust", "I am
    # not certified to dive" withdraws "I am a certified diver".  The
    # retraction path only fires when an active belief shares the
    # object's tokens, so a first-ever negative claim still creates.
    "never learned",
    "never learnt",
    "have never",
    "has never",
    "not certified",
    "isn't certified",
    "not qualified",
    "do not speak",
    "does not speak",
    "don't speak",
    "cannot speak",
    "can't speak",
    "not fluent",
    "only know a few",
    "only knows a few",
    "just a few",
    "barely any",
    "was wrong about",
    "was mistaken about",
)

# Generic object tokens that carry no identity: two beliefs sharing only one
# of these are not about the same thing.
_GENERIC_OBJECT_TOKENS = frozenset({
    "not", "never", "few", "some", "any", "all", "the", "and", "with",
    "for", "from", "into", "about", "very", "really", "just", "also",
    "than", "then", "when", "what", "have", "has", "had", "was", "were",
    "are", "did", "does", "done", "can", "could", "would", "should",
})


def is_self_correction(claim_text: str) -> bool:
    """True when the claim explicitly retracts an earlier statement."""
    text = (claim_text or "").casefold()
    return any(marker in text for marker in _CORRECTION_MARKERS)


# Phrases that end a current state rather than assert a new one.  Like the
# correction markers these are meta-linguistic, but they terminate a goal, a
# possession or an attribute instead of retracting a statement: "I gave up
# the marathon goal", "I sold the red bicycle", "I no longer live in Shanghai".
_TERMINATION_MARKERS = (
    "cancelled", "canceled", "called off", "backed out",
    "gave up", "give up", "given up", "dropped the", "abandoned",
    "scrapped", "quit", "sold my", "sold the", "got rid of",
    "no longer", "not anymore", "decided not to", "decided against",
    "ended the", "broke up with", "parted ways",
    # Disposing of a possession ends it as surely as selling it does.
    "gave away", "given away", "threw away", "threw out",
    "recycled", "donated", "handed down", "lost my", "lost the",
    # Ways of taking something off the table.
    "rejected", "postponed", "put off", "removed", "took off",
    "shelved", "paused", "dropped out of",
    # Withdrawing from something you used to do.  Guarded by the token
    # overlap the caller requires, so "stopped by the shop" closes nothing.
    "stopped enjoying", "stopped liking", "stopped doing", "stopped going",
    "stopped practicing", "stopped playing", "no longer enjoy",
)


# Disposal verbs whose object sits between the verb and the particle:
# "gave the film camera away", "threw the old laptop out".  A contiguous
# substring never matches these, so they are checked as a verb/particle
# pair instead.
_DISPOSAL_VERBS = ("gave", "given", "threw", "throw", "tossed", "handed")
_DISPOSAL_PARTICLES = ("away", "out")


def _is_disposal_phrasing(text: str) -> bool:
    words = text.split()
    return any(
        verb in words and any(particle in words for particle in _DISPOSAL_PARTICLES)
        for verb in _DISPOSAL_VERBS
    )


def is_termination(claim_text: str) -> bool:
    """True when the claim ends a current state instead of asserting one."""
    text = (claim_text or "").casefold()
    return (
        any(marker in text for marker in _TERMINATION_MARKERS)
        or _is_disposal_phrasing(text)
    )


# Phrases that bring a previously ended goal back.  A resumption closes the
# ended stage: "I gave up the marathon goal" then "I have started marathon
# training again" leaves the cancellation current unless the resumption
# retires it.
_RESUMPTION_MARKERS = (
    "again", "restarted", "resumed", "back to", "renewed", "revisiting",
    "trying again", "signed up", "booked", "enrolled", "after all",
    "picked it up", "started training", "applying for",
)

# Words that refer to a goal without naming it.  "I gave up on the goal"
# carries no object tokens that overlap the goal belief it ends, so the
# referent has to be recognised as the generic one.
_GENERIC_GOAL_WORDS = frozenset({
    "goal", "plan", "project", "idea", "intention", "resolution", "ambition",
})


def is_resumption(claim_text: str) -> bool:
    """True when the claim brings a previously ended goal back."""
    text = (claim_text or "").casefold()
    return any(marker in text for marker in _RESUMPTION_MARKERS)


def refers_to_a_goal_generically(text_or_object: str) -> bool:
    """True when the text points at "the goal" without naming it."""
    tokens = re.split(r"[^a-z0-9]+", (text_or_object or "").casefold())
    return bool({t for t in tokens if t} & _GENERIC_GOAL_WORDS)


def object_tokens(obj: str) -> set[str]:
    """The identity-bearing tokens of a canonical object.

    ``fluent_korean`` and ``few_korean_phrases`` are two surfaces of one
    thing; the shared token is what lets a correction find the belief it
    retracts when the extractor spells the object differently each time.
    """
    tokens = re.split(r"[^a-z0-9]+", (obj or "").casefold())
    return {t for t in tokens if len(t) >= 3 and t not in _GENERIC_OBJECT_TOKENS}
