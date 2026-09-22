"""IdentityResolver — orchestrates identity, relation, and lifecycle policy.

The resolver is the entry point, but it no longer *decides* anything itself.
It delegates to three separate concerns:

1. :class:`~mirror_memory.memory.relation.RelationReasoner` — what is the
   relationship between the candidate and each existing belief?
2. :class:`~mirror_memory.memory.lifecycle.LifecyclePolicy` — given that
   relationship, what should happen?
3. Itself — picking *which* existing belief to compare against, and turning
   the answer into a :class:`Resolution`.

This is a **pure function** — no database writes, no side effects.

The split exists because the old resolver answered "is this the same thing?",
"what is their relationship?", and "how should the lifecycle respond?" in one
breath.  Collapsing those is what produced the ``SINGLE = UPDATE`` rule, which
destroyed the previous value and made "where did they live before?"
unanswerable.  See ``memory.temporal`` for the interval semantics.
"""

from __future__ import annotations

import logging

from mirror_memory.memory.atom import (
    ACTION_CONTRADICT,
    ACTION_CREATE,
    ACTION_NOOP,
    CandidateAtom,
    Resolution,
)
from mirror_memory.memory.lifecycle import LifecyclePolicy, to_action
from mirror_memory.memory.relation import judge_relation
from mirror_memory.memory.temporal import (
    TemporalWindow,
    coerce_datetime,
)

logger = logging.getLogger(__name__)


def resolve_identity(
    candidate: CandidateAtom,
    existing_beliefs: list[dict],
    policy: dict[str, str],
    temporal_policy: dict[str, str] | None = None,
) -> Resolution:
    """Decide what happens to *candidate* given existing beliefs.

    Parameters
    ----------
    candidate:
        The new atom from extraction.
    existing_beliefs:
        Active beliefs for this user.  Each dict must have at least
        ``id``, ``predicate``, ``object``, ``status``, ``confidence``, and
        optionally ``valid_from`` / ``valid_to``.
    policy:
        Mapping of predicate → cardinality (``single`` / ``multi`` / ``event``).
        Predicates not in the map default to ``multi``.
    temporal_policy:
        Optional mapping of predicate → temporal scope (``current_state`` /
        ``persistent`` / ``episodic``).  When a predicate appears here the
        lifecycle decision runs through the temporal branch; the cardinality
        alone keeps the legacy path.

    Returns
    -------
    Resolution
        The action to take and, for SUPPORT/UPDATE/CONTRADICT, the target
        belief to act on.
    """
    if not candidate.predicate:
        return Resolution(action=ACTION_NOOP, reason="empty predicate")

    lifecycle_policy = LifecyclePolicy(cardinality=policy, scopes=temporal_policy)

    same_predicate = [
        b for b in existing_beliefs
        if b.get("predicate") == candidate.predicate
        and b.get("status") == "active"
    ]

    # ── No existing belief with this predicate → CREATE ──────────────────
    # (even contradictions create — there's nothing to contradict yet)
    if not same_predicate:
        return Resolution(action=ACTION_CREATE, reason="first claim for this predicate")

    candidate_window = _candidate_window(candidate)
    target = _pick_target(same_predicate, candidate_window, candidate)

    judgement = judge_relation(
        candidate,
        target,
        candidate_window=candidate_window,
        belief_window=_belief_window(target),
    )

    # ── CONTRADICT: same identity + relation=contradicts ──────────────────
    # Checked before the policy so contradictory evidence does not
    # accidentally strengthen an existing belief.
    if judgement.claims_contradiction and judgement.same_object:
        return Resolution(
            action=ACTION_CONTRADICT,
            target_belief_id=target.get("id"),
            reason="contradiction on existing belief",
            lifecycle="CONTRADICT",
            temporal_relation=judgement.temporal_relation.value,
        )

    lifecycle = lifecycle_policy.decide(judgement, candidate.predicate)
    reason = _explain(candidate, target, judgement, lifecycle)

    return Resolution(
        action=to_action(lifecycle),
        target_belief_id=target.get("id") if lifecycle is not None else None,
        reason=reason,
        lifecycle=lifecycle.value,
        temporal_relation=judgement.temporal_relation.value,
    )


def _pick_target(
    same_predicate: list[dict],
    candidate_window: TemporalWindow,
    candidate: CandidateAtom,
) -> dict:
    """Choose which existing belief to compare against.

    When several beliefs share the predicate, the currently-true one is the
    right comparison: a closed interval is history and superseding it would
    answer the wrong question.  Ties fall back to the first.
    """
    if len(same_predicate) == 1:
        return same_predicate[0]
    for belief in same_predicate:
        if _belief_window(belief).is_current():
            return belief
    return same_predicate[0]


def _explain(candidate, target, judgement, lifecycle) -> str:
    """A human-readable reason, so the decision is auditable."""
    relation = judgement.temporal_relation.value
    if lifecycle is None:
        return "no target belief"
    if not judgement.same_identity:
        return "different subject or predicate"
    if judgement.same_object:
        return "same predicate + same object → support"
    return (
        f"{lifecycle.value}: {target.get('object')} → {candidate.object} ({relation})"
    )


def _candidate_window(candidate: CandidateAtom) -> TemporalWindow:
    """Build the candidate's validity window from its temporal fields.

    A bare ``temporal`` string that parses as a date becomes ``valid_from``:
    "in May 2026" is when the fact started being true.
    """
    valid_from = coerce_datetime(candidate.valid_from)
    valid_to = coerce_datetime(candidate.valid_to)
    if valid_from is None and candidate.temporal:
        valid_from = coerce_datetime(candidate.temporal)
    return TemporalWindow(valid_from=valid_from, valid_to=valid_to)


def _belief_window(belief: dict) -> TemporalWindow:
    return TemporalWindow(
        valid_from=coerce_datetime(belief.get("valid_from")),
        valid_to=coerce_datetime(belief.get("valid_to")),
    )
