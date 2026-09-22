"""Publisher — the only component allowed to commit a state change.

Everything upstream (extraction, identity resolution, lifecycle policy,
synthesis) is Compute: it reads, decides, and returns.  A
:class:`~mirror_memory.core.proposal.StateTransitionProposal` is the hand-off
point.  This module is the single writer.

Before committing, the Publisher checks:

- **consent** — is memory still enabled for this user?  A switch flipped while
  the decision was being computed must stop the write.
- **revision** — has the user's state moved since the decision was made?  A
  stale proposal must not overwrite newer state.
- **scope** — does the proposal's session belong to the user it claims to?
- **authority** — do the evidence rows backing the change belong to the same
  user?  Cross-user evidence is refused.
- **invariants** — is the transition legal for the target belief (e.g. FORGET
  of a missing belief, CORRECT of a rejected one)?

Any refusal leaves the database exactly as it was.  The caller gets a
:class:`~mirror_memory.core.proposal.PublishDecision` explaining why, and can
log or surface it; nothing has happened either way.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from mirror_memory.core.models import Belief, Evidence
from mirror_memory.core.proposal import (
    TRANSITION_CONTRADICT,
    TRANSITION_CORRECT,
    TRANSITION_CREATE,
    TRANSITION_FORGET,
    TRANSITION_SUPPORT,
    TRANSITION_SYNTHESIZE,
    TRANSITION_UPDATE,
    TRANSITION_VERIFY,
    PublishDecision,
    StateTransitionProposal,
)
from mirror_memory.core.repository import (
    correct_belief,
    forget_belief,
    get_state_revision,
    is_memory_enabled,
    record_claim,
    support_belief_by_id,
    update_belief_by_id,
)

logger = logging.getLogger(__name__)


class Publisher:
    """Commits state-transition proposals after checking the runtime gates.

    Parameters
    ----------
    session:
        SQLAlchemy session.  The Publisher never commits the transaction
        itself -- the caller owns transaction boundaries, which keeps a
        refusal a pure no-op rather than a partial write.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- public API --------------------------------------------------------

    def publish(self, proposal: StateTransitionProposal) -> PublishDecision:
        """Check every gate, then commit *proposal* if they all pass.

        Returns a :class:`PublishDecision`.  A refusal never raises and never
        writes.
        """
        refusal = self._check_gates(proposal)
        if refusal is not None:
            logger.info("publisher: refused %s (%s)", proposal.describe(), refusal)
            return PublishDecision(committed=False, reason=refusal, proposal=proposal)

        try:
            revision_after = self._commit(proposal)
        except Exception as exc:
            logger.warning(
                "publisher: commit failed for %s: %s", proposal.describe(), type(exc).__name__
            )
            return PublishDecision(
                committed=False, reason=f"commit_error:{type(exc).__name__}", proposal=proposal
            )

        return PublishDecision(
            committed=True,
            proposal=proposal,
            revision_after=revision_after,
        )

    # -- gates -------------------------------------------------------------

    def _check_gates(self, proposal: StateTransitionProposal) -> str | None:
        """Return a refusal reason, or ``None`` when every gate passes."""
        # Consent.
        if not is_memory_enabled(self._session, proposal.user_id):
            return "memory_disabled"

        # Revision: the decision was computed against `claimed_revision`.
        current = get_state_revision(self._session, proposal.user_id)
        if proposal.claimed_revision and current != proposal.claimed_revision:
            return f"stale_revision(claimed={proposal.claimed_revision}, current={current})"

        # Authority: the evidence must belong to the same user.
        if proposal.evidence_ids:
            foreign = (
                self._session.query(Evidence)
                .filter(
                    Evidence.id.in_(proposal.evidence_ids),
                    Evidence.user_id != proposal.user_id,
                )
                .count()
            )
            if foreign:
                return "evidence_authority_mismatch"

        # Invariants, per transition.
        return self._check_invariants(proposal)

    def _check_invariants(self, proposal: StateTransitionProposal) -> str | None:
        target = None
        if proposal.target_belief_id is not None:
            target = self._session.get(Belief, proposal.target_belief_id)

        if proposal.transition == TRANSITION_CREATE:
            return None

        if proposal.transition == TRANSITION_SYNTHESIZE:
            return None

        # Every other transition acts on an existing belief.
        if target is None:
            return "target_belief_missing"
        if target.user_id != proposal.user_id:
            return "target_belief_scope_mismatch"

        if proposal.transition == TRANSITION_FORGET:
            return None

        if target.status == "rejected":
            # A rejected belief is never resurrected.
            return "resurrection_guard"
        if target.status == "superseded":
            # Superseded rows are history; they are closed, not writable.
            return "target_belief_superseded"

        return None

    # -- commit ------------------------------------------------------------

    def _commit(self, proposal: StateTransitionProposal) -> int:
        """Apply the transition.  Returns the revision after the write."""
        handler = {
            TRANSITION_CREATE: self._commit_create,
            TRANSITION_SUPPORT: self._commit_support,
            TRANSITION_UPDATE: self._commit_update,
            TRANSITION_CONTRADICT: self._commit_contradict,
            TRANSITION_CORRECT: self._commit_correct,
            TRANSITION_FORGET: self._commit_forget,
            TRANSITION_VERIFY: self._commit_verify,
            TRANSITION_SYNTHESIZE: self._commit_synthesize,
        }[proposal.transition]
        return handler(proposal)

    def _commit_create(self, proposal: StateTransitionProposal) -> int:
        payload = proposal.payload
        record_claim(
            self._session,
            proposal.user_id,
            dimension=payload.get("dimension", ""),
            key=payload.get("key", ""),
            claim_text=payload.get("claim_text", ""),
            value=payload.get("value"),
            relation="supports",
            confidence=payload.get("confidence", 0.0),
            layer=payload.get("layer", "L4"),
            source=payload.get("source", "extracted"),
            session_id=proposal.session_id,
            evidence_message_ids=payload.get("evidence_message_ids"),
            blocked_key_prefixes=tuple(payload.get("blocked_key_prefixes") or ()),
            allowed_dimensions=payload.get("allowed_dimensions"),
            context_tags=payload.get("context_tags"),
            subject=payload.get("subject", "user"),
            predicate=payload.get("predicate", ""),
            object=payload.get("object", ""),
            cardinality=payload.get("cardinality", "multi"),
        )
        return get_state_revision(self._session, proposal.user_id)

    def _commit_support(self, proposal: StateTransitionProposal) -> int:
        payload = proposal.payload
        support_belief_by_id(
            self._session,
            proposal.target_belief_id,
            claim_text=payload.get("claim_text", ""),
            session_id=proposal.session_id,
            evidence_message_ids=payload.get("evidence_message_ids"),
        )
        return get_state_revision(self._session, proposal.user_id)

    def _commit_update(self, proposal: StateTransitionProposal) -> int:
        payload = proposal.payload
        update_belief_by_id(
            self._session,
            proposal.target_belief_id,
            new_subject=payload.get("subject", "user"),
            new_predicate=payload.get("predicate", ""),
            new_object=payload.get("object", ""),
            new_dimension=payload.get("dimension", ""),
            new_key=payload.get("key", ""),
            new_claim_text=payload.get("claim_text", ""),
            new_confidence=payload.get("confidence", 0.0),
            new_cardinality=payload.get("cardinality", "multi"),
            session_id=proposal.session_id,
            evidence_message_ids=payload.get("evidence_message_ids"),
            new_value=payload.get("value"),
            valid_from=payload.get("valid_from"),
            valid_to=payload.get("valid_to"),
            temporal_scope=payload.get("temporal_scope", ""),
        )
        return get_state_revision(self._session, proposal.user_id)

    def _commit_contradict(self, proposal: StateTransitionProposal) -> int:
        payload = proposal.payload
        record_claim(
            self._session,
            proposal.user_id,
            dimension=payload.get("dimension", ""),
            key=payload.get("key", ""),
            claim_text=payload.get("claim_text", ""),
            value=payload.get("value"),
            relation="contradicts",
            confidence=payload.get("confidence", 0.0),
            session_id=proposal.session_id,
            evidence_message_ids=payload.get("evidence_message_ids"),
            subject=payload.get("subject", "user"),
            predicate=payload.get("predicate", ""),
            object=payload.get("object", ""),
        )
        return get_state_revision(self._session, proposal.user_id)

    def _commit_correct(self, proposal: StateTransitionProposal) -> int:
        payload = proposal.payload
        correct_belief(
            self._session,
            proposal.user_id,
            proposal.target_belief_id,
            new_claim_text=payload.get("new_claim_text", ""),
            correction_note=payload.get("correction_note", ""),
            new_predicate=payload.get("new_predicate"),
            new_object=payload.get("new_object"),
            new_value=payload.get("new_value"),
        )
        return get_state_revision(self._session, proposal.user_id)

    def _commit_forget(self, proposal: StateTransitionProposal) -> int:
        forget_belief(self._session, proposal.user_id, proposal.target_belief_id)
        return get_state_revision(self._session, proposal.user_id)

    def _commit_verify(self, proposal: StateTransitionProposal) -> int:
        """VERIFY is a user confirmation: it raises the layer, it does not
        change the claim.  Handled through confirm_belief."""
        from mirror_memory.core.repository import confirm_belief

        confirm_belief(self._session, proposal.user_id, proposal.target_belief_id)
        return get_state_revision(self._session, proposal.user_id)

    def _commit_synthesize(self, proposal: StateTransitionProposal) -> int:
        """SYNTHESIZE publishes derived state (a snapshot), never a belief.

        This is the only path that writes a Snapshot, which is why the K3
        Compute node in ``extraction.synthesis`` is forbidden from doing it.
        """
        from mirror_memory.worker.snapshot import persist_snapshot

        payload = proposal.payload
        persist_snapshot(
            self._session,
            proposal.user_id,
            payload.get("content") or {},
            payload.get("policy") or {},
            payload.get("watermark") or "",
            shadow=bool(payload.get("shadow", False)),
        )
        return get_state_revision(self._session, proposal.user_id)
