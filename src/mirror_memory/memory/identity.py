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
    ACTION_UPDATE,
    CandidateAtom,
    Resolution,
)
from mirror_memory.memory.lifecycle import LifecyclePolicy, to_action
from mirror_memory.memory.polarity import (
    infer_lifecycle,
    infer_polarity,
    is_goal_predicate,
    is_resumption,
    is_self_correction,
    is_termination,
    object_tokens,
    opposite_polarity,
    refers_to_a_goal_generically,
)
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

    cand_obj = (candidate.object or "").strip().lower()

    # ── Retraction: correction or termination closes what it refers to ─────
    # "Correction: I have never learned Rust" withdraws an earlier claim;
    # "I gave up the marathon goal" ends a current one.  Either way the
    # belief it refers to must stop being current, or a current-state
    # question keeps returning the retracted or ended value.  The referent is
    # found by object tokens rather than by predicate or exact object: the
    # extractor spells the same thing differently each turn ("rust_coding"
    # then "rust", "run_a_marathon" then "marathon_goal") and often under a
    # different predicate ("wants_to" then "gave_up").  Without this the
    # retraction lands as a second active belief beside the one it ends.
    claim_text = getattr(candidate, "claim_text", "") or ""
    correcting = is_self_correction(claim_text)
    terminating = is_termination(claim_text)
    if correcting or terminating:
        cand_tokens = object_tokens(candidate.object)
        sharing = [
            b for b in existing_beliefs
            if b.get("status") == "active"
            and object_tokens(b.get("object") or "") & cand_tokens
        ]
        if not sharing and terminating:
            # "I gave up on the goal" names no object that overlaps the goal
            # belief it ends, so the referent is the generic one: the user's
            # live goal.  Only when there is exactly one -- with several live
            # goals "the goal" is ambiguous and closing all of them would be
            # a guess, so the claim falls through to an ordinary CREATE.
            goals = [
                b for b in existing_beliefs
                if b.get("status") == "active"
                and is_goal_predicate(b.get("predicate") or "")
                and refers_to_a_goal_generically(
                    f"{candidate.object} {claim_text}"
                )
            ]
            if len(goals) == 1:
                sharing = goals
        if sharing:
            primary = max(sharing, key=lambda x: x.get("id") or 0)
            return Resolution(
                action=ACTION_UPDATE,
                target_belief_id=primary.get("id"),
                reason=(
                    f"{'self-correction' if correcting else 'termination'} "
                    f"closes {primary.get('object')!r} via {candidate.predicate!r}"
                ),
                lifecycle="SELF_CORRECTION" if correcting else "END_CURRENT",
                temporal_relation="follows",
                detail={
                    "close_belief_ids": [
                        b["id"] for b in sharing
                        if b["id"] != primary.get("id")
                    ],
                    "ended_object": primary.get("object") or "",
                    "ended_belief_id": primary.get("id"),
                },
            )

    # ── Resumption: bringing an ended goal back retires the ended stage ────
    # "I gave up the marathon goal" then "I have started marathon training
    # again": without this the cancellation stays the current row of the goal
    # and the goal reads as abandoned.  The ended row records which belief it
    # ended, so the resumption is matched by that identity -- an unrelated
    # "again" ("I like coffee again" after selling a car) closes nothing.
    if is_resumption(claim_text):
        cand_tokens = object_tokens(candidate.object)
        ended = [
            b for b in existing_beliefs
            if b.get("status") == "active"
            and (b.get("transition") or "") == "END_CURRENT"
            and (
                not cand_tokens
                or object_tokens(b.get("ended_object") or b.get("object") or "")
                & cand_tokens
            )
        ]
        if ended:
            primary = max(ended, key=lambda x: x.get("id") or 0)
            return Resolution(
                action=ACTION_UPDATE,
                target_belief_id=primary.get("id"),
                reason=(
                    f"resumption retires the ended stage "
                    f"{primary.get('object')!r}"
                ),
                lifecycle="RESUME",
                temporal_relation="follows",
                detail={"close_belief_ids": [
                    b["id"] for b in ended if b["id"] != primary.get("id")
                ]},
            )

    same_predicate = [
        b for b in existing_beliefs
        if b.get("predicate") == candidate.predicate
        and b.get("status") == "active"
    ]
    # Same value through a different verb: "I returned to Shanghai" against a
    # lives_in belief.  A verb is not identity -- the attribute (the value)
    # is.  Superseded rows are included because returning to a previous value
    # is exactly a revival of one of them; the lifecycle decides below.
    #
    # The gate is deliberately narrow: the candidate's predicate must be one
    # the policy does not know (an unmapped verb like ``returned_to``) and the
    # matched rows must all belong to one predicate.  "Same object + same
    # polarity" was too loose -- ``lives_in`` and ``works_at`` are both
    # neutral, so a K1 pattern reading an employer change as a move folded
    # the correct works_at row into a wrong lives_in one.  A candidate that
    # names a configured predicate has its own slot and must not be folded
    # into another attribute's row.
    attribute_matches = [
        b for b in existing_beliefs
        if b.get("status") in ("active", "superseded")
        and (b.get("object") or "").strip().lower() == cand_obj
        and (b.get("predicate") or "").strip().lower() != candidate.predicate.strip().lower()
    ]
    same_attribute_any = (
        attribute_matches
        if candidate.predicate not in policy
        and len({(b.get("predicate") or "").strip().lower() for b in attribute_matches}) <= 1
        else []
    )

    # ── No existing belief with this predicate → CREATE ──────────────────
    # (even contradictions create — there's nothing to contradict yet),
    # unless a different-verb belief about the same value exists, in which
    # case the round-trip revival path below handles it.
    if not same_predicate and not same_attribute_any:
        # One current polarity per object: a claim of the opposite polarity
        # about an object that already holds a current value replaces that
        # value, it does not coexist with it.  "I do not like coffee" closes
        # "I like coffee"; without this the positive row stayed active and a
        # current-state question kept returning a preference the user had
        # withdrawn.  The opposing row becomes the update target so the
        # normal path closes its interval and opens the new polarity.
        cand_polarity_early = infer_polarity(candidate.predicate)
        opposing_early = opposite_polarity(cand_polarity_early)
        opposing_rows = [
            b for b in existing_beliefs
            if b.get("status") == "active"
            and (b.get("object") or "").strip().lower() == cand_obj
            and opposing_early
            and infer_polarity(b.get("predicate") or "") == opposing_early
        ]
        if opposing_rows:
            target_row = max(opposing_rows, key=lambda x: x.get("id") or 0)
            return Resolution(
                action=ACTION_UPDATE,
                target_belief_id=target_row.get("id"),
                reason=(
                    f"polarity replacement: {candidate.predicate!r} closes the "
                    f"opposite current value {target_row.get('object')!r}"
                ),
                lifecycle="TEMPORAL_UPDATE",
                temporal_relation="follows",
                detail={
                    "close_belief_ids": [
                        b["id"] for b in opposing_rows
                        if b["id"] != target_row.get("id")
                    ]
                },
            )
        return Resolution(action=ACTION_CREATE, reason="first claim for this predicate")

    # ── Round-trip revival ────────────────────────────────────────────────
    # "I live in Shanghai" → "moved to Berlin" → "returned to Shanghai".
    # The returned value's newest row is revived; every *other* active belief
    # of the revived row's attribute (berlin, above) gets closed -- they are
    # the values the user left and has now returned past.  Keyed on the
    # candidate's own predicate so the revived row keeps the vocabulary of
    # the turn that re-stated it.
    if not same_predicate and same_attribute_any:
        revived = max(same_attribute_any, key=lambda x: x.get("id") or 0)
        revived_attr_pred = (revived.get("predicate") or "").strip().lower()
        closed = [
            b["id"]
            for b in existing_beliefs
            if b.get("status") == "active"
            and b.get("id") != revived.get("id")
            and (b.get("predicate") or "").strip().lower() == revived_attr_pred
        ]
        return Resolution(
            action=ACTION_UPDATE,
            target_belief_id=revived.get("id"),
            reason=(
                f"round-trip revival: same value ({cand_obj}) via "
                f"{candidate.predicate!r}; closing {closed}"
            ),
            lifecycle="TEMPORAL_UPDATE",
            temporal_relation="follows",
            detail={"close_belief_ids": closed},
        )

    # One current polarity per object, decided structurally: the candidate's
    # canonical predicate maps onto a polarity, and only rows of the *opposite*
    # polarity about the same object are conflicts.  Comparing predicates
    # directly would treat any two different verbs as a conflict, which is
    # both over- and under-inclusive.
    cand_polarity = infer_polarity(candidate.predicate)
    opposing = opposite_polarity(cand_polarity)
    cand_lifecycle = infer_lifecycle(
        candidate.predicate, getattr(candidate, "claim_text", "") or ""
    )
    polarity_conflicts = [
        b["id"] for b in existing_beliefs
        if b.get("status") == "active"
        and (b.get("object") or "").strip().lower() == cand_obj
        and (
            # Opposite polarity about the same object: "likes X" vs "avoids X".
            (opposing and infer_polarity(b.get("predicate") or "") == opposing)
            # Goal lifecycle transition: the same goal moving between
            # active / paused / cancelled / resumed is a state change, and
            # the previous stage must stop being current.
            or (
                cand_lifecycle
                and b.get("lifecycle_state")
                and b.get("lifecycle_state") != cand_lifecycle
                and (b.get("predicate") or "").strip().lower()
                == candidate.predicate.strip().lower()
            )
        )
    ]

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

    close_ids = [i for i in polarity_conflicts if i != target.get("id")]
    return Resolution(
        action=to_action(lifecycle),
        target_belief_id=target.get("id") if lifecycle is not None else None,
        reason=reason,
        lifecycle=lifecycle.value,
        temporal_relation=judgement.temporal_relation.value,
        detail={"close_belief_ids": close_ids} if close_ids else {},
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
