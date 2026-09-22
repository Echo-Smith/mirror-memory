"""RelationReasoner — what is the relationship between two facts?

This is the half of the old ``IdentityResolver`` that answered "are these the
same thing?" *and* "how should the lifecycle respond?" in one breath.  The
first question is about identity; the second is about policy.  Mixing them is
what made the old ``SINGLE = UPDATE`` rule impossible to reason about.

The RelationReasoner answers only the first two questions and returns a
:class:`RelationJudgement`::

    same subject?
    same predicate?
    same object?
    + temporal relation between their validity intervals
    + did the claimant explicitly contradict?

It never chooses a lifecycle action.  That is ``LifecyclePolicy``'s job, and
keeping them apart is what lets the policy change without re-deriving what
the facts *are*.
"""

from __future__ import annotations

from dataclasses import dataclass

from mirror_memory.memory.atom import CandidateAtom
from mirror_memory.memory.temporal import (
    TemporalRelation,
    TemporalWindow,
    coerce_datetime,
    compare_windows,
)


@dataclass(frozen=True)
class RelationJudgement:
    """What the RelationReasoner concluded about a candidate vs a belief.

    ``temporal_relation`` is only meaningful when ``same_subject`` and
    ``same_predicate`` both hold -- two unrelated facts have no ordering to
    speak of, and reporting one would be noise.
    """

    same_subject: bool
    same_predicate: bool
    same_object: bool
    temporal_relation: TemporalRelation
    claims_contradiction: bool
    # Do both sides carry the same temporal *qualifier* ("monday" vs
    # "friday")?  Only meaningful for episodic predicates, where two
    # occurrences of the same event are distinct facts.
    same_temporal_qualifier: bool = True
    subject_matches_expected: bool = True

    @property
    def same_identity(self) -> bool:
        """Is this the same subject talking about the same predicate?"""
        return self.same_subject and self.same_predicate


def judge_relation(
    candidate: CandidateAtom,
    belief: dict,
    *,
    candidate_window: TemporalWindow | None = None,
    belief_window: TemporalWindow | None = None,
) -> RelationJudgement:
    """Compare *candidate* against one existing belief.

    Parameters
    ----------
    candidate:
        The new atom from extraction.
    belief:
        One existing belief, as a dict with at least ``predicate``,
        ``object``, ``status`` and optionally ``valid_from`` / ``valid_to``.
    candidate_window / belief_window:
        Pre-computed validity intervals.  Derived from the inputs when
        omitted.

    Returns
    -------
    RelationJudgement
    """
    if candidate_window is None:
        candidate_window = _window_from_candidate(candidate)
    if belief_window is None:
        belief_window = _window_from_belief_dict(belief)

    same_predicate = (belief.get("predicate") or "").strip().lower() == (
        candidate.predicate or ""
    ).strip().lower()
    same_object = (belief.get("object") or "").strip().lower() == (
        candidate.object or ""
    ).strip().lower()

    # Identity is about the subject and the predicate; the object is what the
    # predicate is applied to, not part of "is this the same thing".
    same_subject = True

    temporal_relation = TemporalRelation.UNKNOWN
    if same_subject and same_predicate:
        temporal_relation = compare_windows(candidate_window, belief_window)

    # A qualifier only counts as "the same" when both sides state one and they
    # agree.  A blank on either side is no evidence of difference, so it does
    # not split two occurrences of the same event.
    candidate_qualifier = (candidate.temporal or "").strip().lower()
    belief_qualifier = (belief.get("temporal") or "").strip().lower()
    same_temporal_qualifier = not (
        candidate_qualifier and belief_qualifier and candidate_qualifier != belief_qualifier
    )

    return RelationJudgement(
        same_subject=same_subject,
        same_predicate=same_predicate,
        same_object=same_object,
        temporal_relation=temporal_relation,
        claims_contradiction=candidate.relation == "contradicts",
        same_temporal_qualifier=same_temporal_qualifier,
    )


def _window_from_candidate(candidate: CandidateAtom) -> TemporalWindow:
    """Read a candidate's validity window, treating a parseable ``temporal``
    string as the start of the interval."""
    valid_from = coerce_datetime(candidate.valid_from)
    valid_to = coerce_datetime(candidate.valid_to)
    if valid_from is None and candidate.temporal:
        valid_from = coerce_datetime(candidate.temporal)
    return TemporalWindow(valid_from=valid_from, valid_to=valid_to)


def _window_from_belief_dict(belief: dict) -> TemporalWindow:
    return TemporalWindow(
        valid_from=coerce_datetime(belief.get("valid_from")),
        valid_to=coerce_datetime(belief.get("valid_to")),
    )
