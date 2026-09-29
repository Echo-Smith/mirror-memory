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

import uuid
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
# Memory metabolism: a storage-tier move (hot -> warm -> dormant, or a
# reheat).  Changes how the belief is stored and scanned, never what it
# claims -- truth state is untouched, which is also why committing one does
# not bump the state revision.
TRANSITION_TIER_TRANSITION = "TIER_TRANSITION"
# Memory metabolism: fold a belief's redundant support evidence into an
# EvidenceDigest, keeping a representative sample live.  The conclusion the
# evidence supports is untouched; only the provenance bulk is aggregated.
TRANSITION_EVIDENCE_COMPACT = "EVIDENCE_COMPACT"
# The user denied a verification question: the belief is closed as rejected
# (its key is never resurrected).  Spelled out rather than smuggled through
# another transition, because a denial is a user decision with its own
# invariant -- not an inference.
TRANSITION_REJECT = "REJECT"

STATE_TRANSITIONS = (
    TRANSITION_CREATE,
    TRANSITION_SUPPORT,
    TRANSITION_UPDATE,
    TRANSITION_CONTRADICT,
    TRANSITION_VERIFY,
    TRANSITION_CORRECT,
    TRANSITION_FORGET,
    TRANSITION_SYNTHESIZE,
    TRANSITION_TIER_TRANSITION,
    TRANSITION_EVIDENCE_COMPACT,
    TRANSITION_REJECT,
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
    TRANSITION_TIER_TRANSITION,
    TRANSITION_EVIDENCE_COMPACT,
    TRANSITION_REJECT,
)

# The transitions that assert something about the *truth* state, and
# therefore must carry an ``expected_revision``: a decision about what is
# true is only valid against the state it was computed from.  Storage-only
# transitions (tier moves, compaction, synthesis) do not bump the revision
# and may omit it.
TRUTH_TRANSITIONS = (
    TRANSITION_CREATE,
    TRANSITION_SUPPORT,
    TRANSITION_UPDATE,
    TRANSITION_CONTRADICT,
    TRANSITION_VERIFY,
    TRANSITION_CORRECT,
    TRANSITION_FORGET,
    TRANSITION_REJECT,
)

# The subset that changes the slot's *value*, and therefore appends a new
# version to the ledger.  SUPPORT / VERIFY deepen the same value;
# CONTRADICT contests it; REJECT closes it.
VALUE_TRANSITIONS = (
    TRANSITION_CREATE,
    TRANSITION_UPDATE,
    TRANSITION_CORRECT,
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
    expected_revision:
        The state revision the decision was computed against.  The Publisher
        refuses to commit if the live revision has moved.  **Required for
        every truth mutation** (CREATE / SUPPORT / UPDATE / CONTRADICT /
        VERIFY / CORRECT / FORGET): a proposal without it is refused rather
        than silently skipping the stale check.  Storage-only transitions
        (TIER_TRANSITION / EVIDENCE_COMPACT / SYNTHESIZE) may omit it.
    proposal_id / idempotency_key:
        Identity of the decision.  Replaying a ``proposal_id`` that already
        committed is answered ``already_committed`` without re-executing;
        ``idempotency_key`` lets a caller collapse *different* proposal
        objects that represent the same intended change (a retry after a
        timeout, say) onto one commit.
    actor_type / actor_id:
        Who is asking: ``user`` / ``worker`` / ``metabolism`` / ``system``.
        Recorded in the proposal log so a committed change can always be
        traced back to its requester.
    lifecycle / temporal_relation:
        The policy's reasoning, carried through so the commit is auditable.
    payload (TIER_TRANSITION):
        ``from_tier`` / ``to_tier`` must name a legal edge of the tier state
        machine and ``from_tier`` must still match the live row; ``reason``
        and ``score`` carry the decision's provenance into the audit log.
    payload (EVIDENCE_COMPACT):
        ``keep_ids`` / ``fold_ids`` must match the selection the Publisher
        recomputes from the live evidence graph under the payload's
        ``policy`` (clamped to sane bounds) and ``min_support_links``.
    """

    transition: str
    user_id: str
    session_id: str | None = None
    target_belief_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[int] = field(default_factory=list)
    expected_revision: int = 0
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    idempotency_key: str = ""
    actor_type: str = "system"
    actor_id: str = ""
    lifecycle: str = ""
    temporal_relation: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.transition not in STATE_TRANSITIONS:
            raise ValueError(f"unknown state transition: {self.transition!r}")
        if not self.proposal_id:
            self.proposal_id = uuid.uuid4().hex
        if not self.idempotency_key:
            # Default the key to the proposal id: one proposal object is
            # one intended change.  Callers that build a *new* object for a
            # retry pass a stable key of their own.
            self.idempotency_key = self.proposal_id

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
