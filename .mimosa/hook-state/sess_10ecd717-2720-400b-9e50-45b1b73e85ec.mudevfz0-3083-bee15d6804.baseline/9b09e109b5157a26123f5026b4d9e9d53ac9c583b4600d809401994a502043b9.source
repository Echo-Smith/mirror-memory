"""StateTransitionProposal — a write that has been decided but not committed.

Every state change (CREATE / SUPPORT / UPDATE / CONTRADICT / VERIFY / CORRECT
/ FORGET / SYNTHESIZE) is expressed as a proposal.  A proposal is *computed*:
it describes an intent and carries the evidence for it, but it writes nothing.

The :class:`~mirror_memory.core.publisher.Publisher` is the only component
allowed to commit a proposal, and it does so only after checking revision,
consent, scope, authority, and invariants.  Nothing else in the engine may
write to the database.

This is what makes the runtime trustworthy: a proposal can be inspected,
logged, rejected, or replayed without any of it having happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# The state-changing operations the engine can propose.
TRANSITION_CREATE = "CREATE"
TRANSITION_SUPPORT = "SUPPORT"
TRANSITION_UPDATE = "UPDATE"
TRANSITION_CONTRADICT = "CONTRADICT"
TRANSITION_VERIFY = "VERIFY"
TRANSITION_CORRECT = "CORRECT"
TRANSITION_FORGET = "FORGET"
TRANSITION_SYNTHESIZE = "SYNTHESIZE"

STATE_TRANSITIONS = (
    TRANSITION_CREATE,
    TRANSITION_SUPPORT,
    TRANSITION_UPDATE,
    TRANSITION_CONTRADICT,
    TRANSITION_VERIFY,
    TRANSITION_CORRECT,
    TRANSITION_FORGET,
    TRANSITION_SYNTHESIZE,
)

# Transitions that mutate belief rows.  SYNTHESIZE writes derived state
# (snapshots) rather than beliefs, which is why it is listed separately.
BELIEF_MUTATING_TRANSITIONS = (
    TRANSITION_CREATE,
    TRANSITION_SUPPORT,
    TRANSITION_UPDATE,
    TRANSITION_CONTRADICT,
    TRANSITION_CORRECT,
    TRANSITION_FORGET,
)


@dataclass
class StateTransitionProposal:
    """One decided-but-uncommitted state change.

    Fields
    ------
    transition:
        One of :data:`STATE_TRANSITIONS`.
    user_id / session_id:
        Who the change belongs to and where it came from.
    target_belief_id:
        The belief being acted on, when the transition has a target.
    payload:
        The change itself -- the claim fields for CREATE/UPDATE, the relation
        for CONTRADICT, the correction for CORRECT.
    evidence_ids:
        :class:`~mirror_memory.core.models.Evidence` rows backing the change.
    claimed_revision:
        The state revision the decision was computed against.  The Publisher
        refuses to commit if the live revision has moved.
    lifecycle / temporal_relation:
        The policy's reasoning, carried through so the commit is auditable.
    """

    transition: str
    user_id: str
    session_id: str | None = None
    target_belief_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[int] = field(default_factory=list)
    claimed_revision: int = 0
    lifecycle: str = ""
    temporal_relation: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.transition not in STATE_TRANSITIONS:
            raise ValueError(f"unknown state transition: {self.transition!r}")

    @property
    def mutates_beliefs(self) -> bool:
        return self.transition in BELIEF_MUTATING_TRANSITIONS

    def describe(self) -> str:
        """A one-line, log-safe description."""
        target = self.target_belief_id if self.target_belief_id is not None else "-"
        return f"{self.transition}(user={self.user_id}, belief={target})"


@dataclass
class PublishDecision:
    """The Publisher's verdict on one proposal.

    ``committed`` is the only field a caller should branch on.  ``reason``
    explains a refusal and is always populated when ``committed`` is false.
    """

    committed: bool
    reason: str = ""
    proposal: StateTransitionProposal | None = None
    revision_after: int | None = None
    detail: dict = field(default_factory=dict)
