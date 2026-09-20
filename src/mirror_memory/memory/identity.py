"""IdentityResolver — decides what happens when a new CandidateAtom arrives.

The resolver compares a CandidateAtom against existing persisted beliefs
and returns a Resolution (CREATE / SUPPORT / UPDATE / CONTRADICT / NOOP).

This is a **pure function** — no database writes, no side effects.
"""

from __future__ import annotations

import logging
from typing import Any

from mirror_memory.memory.atom import (
    ACTION_CONTRADICT,
    ACTION_CREATE,
    ACTION_NOOP,
    ACTION_SUPPORT,
    ACTION_UPDATE,
    CARDINALITY_EVENT,
    CARDINALITY_MULTI,
    CARDINALITY_SINGLE,
    CandidateAtom,
    Resolution,
)

logger = logging.getLogger(__name__)


def resolve_identity(
    candidate: CandidateAtom,
    existing_beliefs: list[dict],
    policy: dict[str, str],
) -> Resolution:
    """Decide what happens to *candidate* given existing beliefs.

    Parameters
    ----------
    candidate:
        The new atom from extraction.
    existing_beliefs:
        Active beliefs for this user.  Each dict must have at least
        ``id``, ``predicate``, ``object``, ``status``, ``confidence``.
    policy:
        Mapping of predicate → cardinality (``single`` / ``multi`` / ``event``).
        Predicates not in the map default to ``multi``.

    Returns
    -------
    Resolution
        The action to take and, for SUPPORT/UPDATE/CONTRADICT, the target
        belief to act on.
    """
    if not candidate.predicate:
        return Resolution(action=ACTION_NOOP, reason="empty predicate")

    cardinality = policy.get(candidate.predicate, CARDINALITY_MULTI)
    is_contradiction = candidate.relation == "contradicts"

    # Find active beliefs with the same predicate.
    same_predicate = [
        b for b in existing_beliefs
        if b.get("predicate") == candidate.predicate
        and b.get("status") == "active"
    ]

    # ── No existing belief with this predicate → CREATE ──────────────────
    # (even contradictions create — there's nothing to contradict yet)
    if not same_predicate:
        return Resolution(action=ACTION_CREATE, reason="first claim for this predicate")

    # ── CONTRADICT: same identity + relation=contradicts ──────────────────
    # This must be checked BEFORE SUPPORT so contradictory evidence does
    # not accidentally strengthen an existing belief.
    if is_contradiction:
        same_object = [b for b in same_predicate if _is_same_object(b, candidate)]
        if same_object:
            return Resolution(
                action=ACTION_CONTRADICT,
                target_belief_id=same_object[0].get("id"),
                reason="contradiction on existing belief",
            )

    # ── SINGLE cardinality ───────────────────────────────────────────────
    if cardinality == CARDINALITY_SINGLE:
        current = same_predicate[0]
        if _is_same_object(current, candidate):
            return Resolution(
                action=ACTION_SUPPORT,
                target_belief_id=current.get("id"),
                reason="same predicate + same object → support",
            )
        else:
            return Resolution(
                action=ACTION_UPDATE,
                target_belief_id=current.get("id"),
                reason=f"SINGLE: {current.get('object')} → {candidate.object}",
            )

    # ── MULTI cardinality ────────────────────────────────────────────────
    if cardinality == CARDINALITY_MULTI:
        same_object = [b for b in same_predicate if _is_same_object(b, candidate)]
        if same_object:
            return Resolution(
                action=ACTION_SUPPORT,
                target_belief_id=same_object[0].get("id"),
                reason="same predicate + same object → support",
            )
        return Resolution(action=ACTION_CREATE, reason="MULTI: new object for this predicate")

    # ── EVENT cardinality ────────────────────────────────────────────────
    if cardinality == CARDINALITY_EVENT:
        same_object = [b for b in same_predicate if _is_same_object(b, candidate)]
        if same_object and not is_contradiction:
            candidate_temporal = candidate.temporal or ""
            for b in same_object:
                b_temporal = b.get("temporal") or ""
                if candidate_temporal and b_temporal and candidate_temporal != b_temporal:
                    continue
                return Resolution(
                    action=ACTION_SUPPORT,
                    target_belief_id=b.get("id"),
                    reason="exact duplicate event (pred+obj+temporal) → support",
                )
        if same_object and is_contradiction:
            return Resolution(
                action=ACTION_CONTRADICT,
                target_belief_id=same_object[0].get("id"),
                reason="contradiction on event",
            )
        return Resolution(action=ACTION_CREATE, reason="EVENT: new occurrence")

    return Resolution(action=ACTION_CREATE, reason="unknown cardinality, default CREATE")


def _is_same_object(belief: dict, candidate: CandidateAtom) -> bool:
    """Check if a belief and a candidate refer to the same object.

    Simple string comparison for now.  Future: fuzzy matching / embedding.
    """
    return (belief.get("object") or "").strip().lower() == candidate.object.strip().lower()
