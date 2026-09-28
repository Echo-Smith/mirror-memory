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
    TRANSITION_EVIDENCE_COMPACT,
    TRANSITION_FORGET,
    TRANSITION_SUPPORT,
    TRANSITION_SYNTHESIZE,
    TRANSITION_TIER_TRANSITION,
    TRANSITION_UPDATE,
    TRANSITION_VERIFY,
    PublishDecision,
    StateTransitionProposal,
)
from mirror_memory.core.repository import (
    apply_evidence_compaction,
    correct_belief,
    evidence_graph_for_beliefs,
    evidence_relation_counts_for_beliefs,
    forget_belief,
    get_state_revision,
    is_memory_enabled,
    pending_verification_keys,
    record_claim,
    set_belief_tier,
    support_belief_by_id,
    update_belief_by_id,
)
from mirror_memory.metabolism.compact import (
    build_digest_fields,
    clamp_policy,
    plan_compaction,
)
from mirror_memory.metabolism.protection import (
    compaction_protection_reason,
    protection_reason,
)
from mirror_memory.metabolism.tiers import is_legal_tier_transition

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
        # Consent gates *collection*: with memory off, nothing new may be
        # written.  Deletion is the one transition that must stay available
        # when memory is disabled — a user who turned memory off still has
        # the right to have what was collected erased.
        if proposal.transition != TRANSITION_FORGET:
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

        if proposal.transition == TRANSITION_TIER_TRANSITION:
            return self._check_tier_invariants(proposal, target)

        if proposal.transition == TRANSITION_EVIDENCE_COMPACT:
            return self._check_compact_invariants(proposal, target)

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

    def _check_tier_invariants(
        self, proposal: StateTransitionProposal, target: Belief | None
    ) -> str | None:
        """Gates for a storage-tier move (cool, reheat, archive, restore).

        A superseded belief *is* coolable — "superseded but historically
        true and plausibly useful" is exactly the combination the tiers
        exist to express — so the generic superseded refusal does not apply
        here.  Protection, however, is re-derived from live state rather
        than trusted from the planner's payload: the invariant must hold
        even against a buggy or stale planner.
        """
        if target is None:
            return "target_belief_missing"
        if target.user_id != proposal.user_id:
            return "target_belief_scope_mismatch"
        if target.status == "rejected":
            return "resurrection_guard"

        from_tier = str(proposal.payload.get("from_tier", ""))
        to_tier = str(proposal.payload.get("to_tier", ""))
        live_tier = target.memory_tier or "hot"

        # The decision was computed against from_tier; if the live row has
        # moved since, the decision describes a belief that no longer exists.
        if from_tier != live_tier:
            return f"tier_mismatch(claimed={from_tier}, live={live_tier})"
        if not is_legal_tier_transition(from_tier, to_tier):
            return f"illegal_tier_transition({from_tier}->{to_tier})"

        counts = evidence_relation_counts_for_beliefs(self._session, [target.id]).get(
            target.id, {}
        )
        reason = protection_reason(
            target,
            support_count=counts.get("support", 0),
            correct_count=counts.get("correct", 0),
            pending_verification=target.key in pending_verification_keys(
                self._session, proposal.user_id
            ),
        )
        if reason is not None:
            return f"protected:{reason}"
        return None

    def _check_compact_invariants(
        self, proposal: StateTransitionProposal, target: Belief | None
    ) -> str | None:
        """Gates for folding a belief's evidence bulk into a digest.

        Protection is re-derived from live state (same as tier moves), and
        the selection itself is **recomputed** from the live evidence graph
        under the payload's (bounds-clamped) policy — the planner's keep/fold
        lists must match exactly, so a stale or forged plan cannot fold rows
        the rules would keep.
        """
        if target is None:
            return "target_belief_missing"
        if target.user_id != proposal.user_id:
            return "target_belief_scope_mismatch"
        if target.status == "rejected":
            return "resurrection_guard"

        graph = evidence_graph_for_beliefs(self._session, [target.id]).get(target.id, [])
        counts: dict[str, int] = {}
        for row in graph:
            counts[row.relation] = counts.get(row.relation, 0) + 1

        # Compaction-specific protection: a live dispute or a pending
        # question blocks folding; a user correction does not — the
        # corrected rows are always kept by the keep rules.
        reason = compaction_protection_reason(
            target,
            pending_verification=target.key in pending_verification_keys(
                self._session, proposal.user_id
            ),
        )
        if reason is not None:
            return f"protected:{reason}"

        policy = clamp_policy(proposal.payload.get("policy"))
        min_support = min(
            max(int(proposal.payload.get("min_support_links", 12) or 12), 1), 1000
        )
        plan = plan_compaction(
            graph, target.id, policy=policy, min_support=min_support
        )
        if plan is None:
            return f"below_compaction_threshold(support={counts.get('support', 0)})"
        if not plan.folds_anything:
            return "nothing_to_fold"

        claimed_keep = {int(i) for i in proposal.payload.get("keep_ids", [])}
        claimed_fold = {int(i) for i in proposal.payload.get("fold_ids", [])}
        if claimed_keep != set(plan.keep_ids) or claimed_fold != set(plan.fold_ids):
            return "stale_selection"
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
            TRANSITION_TIER_TRANSITION: self._commit_tier_transition,
            TRANSITION_EVIDENCE_COMPACT: self._commit_evidence_compact,
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
            retention_class=payload.get("retention_class"),
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

    def _commit_tier_transition(self, proposal: StateTransitionProposal) -> int:
        """TIER_TRANSITION moves a belief between storage tiers.

        The audit row (who moved what where, and on which heat score) is
        written in the same transaction as the move.  The state revision is
        deliberately left alone — see :func:`set_belief_tier`.
        """
        payload = proposal.payload
        set_belief_tier(
            self._session,
            proposal.target_belief_id,
            from_tier=str(payload.get("from_tier", "")),
            to_tier=str(payload.get("to_tier", "")),
            reason=str(payload.get("reason", "")),
            score=float(payload.get("score", 0.0)),
            proposal_desc=proposal.describe(),
        )
        return get_state_revision(self._session, proposal.user_id)

    def _commit_evidence_compact(self, proposal: StateTransitionProposal) -> int:
        """EVIDENCE_COMPACT folds redundant support evidence into a digest.

        The gates already recomputed the plan, so the commit rebuilds the
        same plan (same pure functions, same live graph) and applies it.
        As with tier moves, the state revision is untouched: provenance
        bulk is storage, not truth.
        """
        target = self._session.get(Belief, proposal.target_belief_id)
        graph = evidence_graph_for_beliefs(self._session, [target.id]).get(target.id, [])
        policy = clamp_policy(proposal.payload.get("policy"))
        min_support = min(
            max(int(proposal.payload.get("min_support_links", 12) or 12), 1), 1000
        )
        plan = plan_compaction(graph, target.id, policy=policy, min_support=min_support)
        digest_fields = build_digest_fields(graph, plan.keep_ids)
        apply_evidence_compaction(
            self._session,
            target,
            keep_ids=list(plan.keep_ids),
            fold_ids=list(plan.fold_ids),
            digest_fields=digest_fields,
        )
        return get_state_revision(self._session, proposal.user_id)
