"""LifecyclePolicy — what should happen to a belief, given a relation.

The old code embedded this decision inside ``IdentityResolver`` as a chain of
``if cardinality == SINGLE: ... UPDATE``.  Policy deserves its own module: it
is the part that changes when the product's semantics change, and it should be
readable without also reading the identity comparison.

The policy is a pure function of a :class:`RelationJudgement` plus the
predicate's declared scope.  It never reads the database and never inspects
the candidate's text.
"""

from __future__ import annotations

from mirror_memory.memory.atom import (
    ACTION_CONTRADICT,
    ACTION_CREATE,
    ACTION_SUPPORT,
    ACTION_UPDATE,
)
from mirror_memory.memory.relation import RelationJudgement
from mirror_memory.memory.temporal import (
    SCOPE_CURRENT_STATE,
    SCOPE_EPISODIC,
    SCOPE_PERSISTENT,
    LifecycleAction,
    TemporalRelation,
    lifecycle_transition,
)


class LifecyclePolicy:
    """Maps a relation judgement onto a lifecycle action.

    Parameters
    ----------
    cardinality:
        Mapping of predicate → cardinality (``single`` / ``multi`` / ``event``).
    scopes:
        Mapping of predicate → temporal scope (``current_state`` /
        ``persistent`` / ``episodic``).
    """

    def __init__(
        self,
        cardinality: dict[str, str] | None = None,
        scopes: dict[str, str] | None = None,
    ) -> None:
        self._cardinality = cardinality or {}
        self._scopes = scopes or {}

    def cardinality_for(self, predicate: str) -> str:
        return self._cardinality.get(predicate, "multi")

    def scope_for(self, predicate: str) -> str:
        """The temporal scope for *predicate*.

        An explicit scope wins.  Otherwise a ``single`` cardinality is treated
        as ``current_state`` so interval semantics still apply; anything else
        has no temporal meaning and returns ``""``.
        """
        declared = self._scopes.get(predicate)
        if declared:
            return declared
        if self._cardinality.get(predicate) == "single":
            return SCOPE_CURRENT_STATE
        return ""

    def uses_temporal_lifecycle(self, predicate: str) -> bool:
        """Does *predicate* mean "one value at a time"?

        Only then does the interval machinery apply.  A multi-cardinality
        predicate keeps its coexistence semantics regardless of scope:
        "likes coffee" then "likes tea" is two facts, not a handover.
        """
        return (
            self.cardinality_for(predicate) == "single"
            or self.scope_for(predicate) == SCOPE_CURRENT_STATE
        )

    def decide(self, judgement: RelationJudgement, predicate: str) -> LifecycleAction:
        """Pick the lifecycle action for *judgement*.

        Multi and event cardinalities short-circuit to their own rules; only
        single-value predicates reach the temporal lifecycle.
        """
        if not judgement.same_identity:
            return LifecycleAction.CREATE

        cardinality = self.cardinality_for(predicate)
        if cardinality == "event":
            return self._decide_event(judgement)
        if not self.uses_temporal_lifecycle(predicate):
            return self._decide_multi(judgement)

        scope = self.scope_for(predicate) or SCOPE_CURRENT_STATE
        return lifecycle_transition(
            same_subject=judgement.same_subject,
            same_predicate=judgement.same_predicate,
            same_object=judgement.same_object,
            scope=scope,
            relation=judgement.temporal_relation,
            claims_contradiction=judgement.claims_contradiction,
        )

    def _decide_multi(self, judgement: RelationJudgement) -> LifecycleAction:
        if judgement.same_object:
            if judgement.claims_contradiction:
                return LifecycleAction.CONTRADICT
            return LifecycleAction.SUPPORT
        return LifecycleAction.CREATE

    def _decide_event(self, judgement: RelationJudgement) -> LifecycleAction:
        """Episodic: each occurrence is its own fact.

        Two mentions of the same event with *different* temporal qualifiers
        ("went to the museum Monday" vs "...Friday") are two occurrences, not
        a duplicate -- so they create rather than support.
        """
        if not judgement.same_object:
            return LifecycleAction.CREATE
        if judgement.claims_contradiction:
            return LifecycleAction.CONTRADICT
        if not judgement.same_temporal_qualifier:
            return LifecycleAction.CREATE
        return LifecycleAction.SUPPORT


# Lifecycle actions map back onto the coarse actions the persistence layer
# understands.  TEMPORAL_UPDATE needs the interval-closing write path, which
# is what UPDATE does once it records valid_from/valid_to; COEXIST creates a
# second row so both periods stay answerable.
LIFECYCLE_TO_ACTION = {
    LifecycleAction.CREATE: ACTION_CREATE,
    LifecycleAction.SUPPORT: ACTION_SUPPORT,
    LifecycleAction.TEMPORAL_UPDATE: ACTION_UPDATE,
    LifecycleAction.COEXIST: ACTION_CREATE,
    LifecycleAction.CONTRADICT: ACTION_CONTRADICT,
    LifecycleAction.NOOP: "NOOP",
}


def to_action(lifecycle: LifecycleAction) -> str:
    """The coarse persistence action for a lifecycle decision."""
    return LIFECYCLE_TO_ACTION.get(lifecycle, ACTION_CREATE)


__all__ = [
    "LIFECYCLE_TO_ACTION",
    "LifecycleAction",
    "LifecyclePolicy",
    "RelationJudgement",
    "SCOPE_CURRENT_STATE",
    "SCOPE_EPISODIC",
    "SCOPE_PERSISTENT",
    "TemporalRelation",
    "to_action",
]
