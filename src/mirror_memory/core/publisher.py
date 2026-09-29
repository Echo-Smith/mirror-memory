"""Publisher — the only component allowed to commit a state change.

Everything upstream (extraction, identity resolution, lifecycle policy,
synthesis, metabolism) is Compute: it reads, decides, and returns.  A
:class:`~mirror_memory.core.proposal.StateTransitionProposal` is the hand-off
point.  This module is the single writer.

Before committing, the Publisher checks:

- **consent** — is memory still enabled for this user?  A switch flipped while
  the decision was being computed must stop the write.  Deletion is exempt:
  consent governs collection, not erasure.
- **revision** — has the user's state moved since the decision was made?  A
  stale proposal must not overwrite newer state.  Truth mutations must name
  the revision they were computed against, and the bump itself is a
  compare-and-swap (``UPDATE ... WHERE revision = expected``), so two
  concurrent publishers cannot both win.
- **scope** — does the proposal's session match the evidence it cites?
- **authority** — do the evidence rows backing the change exist, and belong
  to the same user?  A missing id is refused, not silently ignored.
- **invariants** — is the transition legal for the target belief (e.g. FORGET
  of a missing belief, CORRECT of a rejected one)?

Idempotency: every committed proposal is recorded in
``mm_proposal_log`` under its unique ``proposal_id``.  Replaying one is
answered ``already_committed`` without re-executing the change.

Atomicity: the commit runs inside a SAVEPOINT.  A handler that fails
halfway (belief written, event failed) rolls the whole change back — a
refusal leaves the database exactly as it was.

Any refusal leaves the database exactly as it was.  The caller gets a
:class:`~mirror_memory.core.proposal.PublishDecision` explaining why, and can
log or surface it; nothing has happened either way.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from mirror_memory.core.models import Belief, Evidence, ProposalLog, utcnow
from mirror_memory.core.proposal import (
    TRANSITION_CONTRADICT,
    TRANSITION_CORRECT,
    TRANSITION_CREATE,
    TRANSITION_EVIDENCE_COMPACT,
    TRANSITION_FORGET,
    TRANSITION_REJECT,
    TRANSITION_SUPPORT,
    TRANSITION_SYNTHESIZE,
    TRANSITION_TIER_TRANSITION,
    TRANSITION_UPDATE,
    TRANSITION_VERIFY,
    TRUTH_TRANSITIONS,
    VALUE_TRANSITIONS,
    PublishDecision,
    StateTransitionProposal,
)
from mirror_memory.core.repository import (
    CONFLICT_SOURCE_CONFLICT,
    apply_evidence_compaction,
    cas_bump_state_revision,
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
from mirror_memory.exceptions import StaleRevisionError
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
        SQLAlchemy session.  The Publisher never commits the outer
        transaction itself -- the caller owns transaction boundaries, which
        keeps a refusal a pure no-op rather than a partial write.  The
        Publisher *does* establish its own savepoint around each commit, so
        a handler failure cannot leave a half-applied change behind.
    config:
        Optional :class:`~mirror_memory.config.schema.MemoryConfig`; its
        ``metabolism.ledger`` bounds the versioned ledger.  Without it the
        ledger is unbounded.
    """

    def __init__(self, session: Session, config: Any | None = None) -> None:
        self._session = session
        self._config = config

    # -- public API --------------------------------------------------------

    def publish(self, proposal: StateTransitionProposal) -> PublishDecision:
        """Check every gate, then commit *proposal* if they all pass.

        Returns a :class:`PublishDecision`.  A refusal never raises and never
        writes; an already-committed ``proposal_id`` is reported as such
        rather than applied a second time.
        """
        # Idempotency: a proposal that already committed is not re-executed.
        # Either an identical proposal_id (a literal replay) or a shared
        # idempotency_key (a retry that rebuilt the proposal object) counts.
        row = self._session.scalar(
            select(ProposalLog)
            .where(
                or_(
                    ProposalLog.proposal_id == proposal.proposal_id,
                    ProposalLog.idempotency_key == proposal.idempotency_key,
                )
            )
            .limit(1)
        )
        if row is not None:
            logger.info("publisher: already committed %s", proposal.describe())
            return PublishDecision(
                committed=False,
                reason=f"already_committed(proposal_id={row.proposal_id[:12]})",
                proposal=proposal,
                revision_after=row.revision_after,
            )

        refusal = self._check_gates(proposal)
        if refusal is not None:
            logger.info("publisher: refused %s (%s)", proposal.describe(), refusal)
            return PublishDecision(committed=False, reason=refusal, proposal=proposal)

        try:
            # The savepoint is the Publisher's own atomicity boundary: the
            # revision CAS, the handler's writes, and the proposal-log row
            # either all land or none do.
            with self._session.begin_nested():
                revision_after, belief_id = self._commit(proposal)
                self._record_commit(proposal, revision_after, belief_id)
        except StaleRevisionError as exc:
            # The CAS lost the race between the gate check and the commit.
            # The savepoint is already rolled back; report it as the stale
            # proposal it is.
            logger.info("publisher: stale %s (%s)", proposal.describe(), exc)
            return PublishDecision(
                committed=False,
                reason=f"stale_revision(expected={proposal.expected_revision}, cas_lost=True)",
                proposal=proposal,
            )
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
            detail={"belief_id": belief_id},
        )

    def _record_commit(self, proposal: StateTransitionProposal, revision_after: int, belief_id: int | None) -> None:
        """Append the idempotency + audit row for a committed proposal."""
        self._session.add(
            ProposalLog(
                proposal_id=proposal.proposal_id,
                idempotency_key=proposal.idempotency_key,
                user_id=proposal.user_id,
                transition=proposal.transition,
                target_belief_id=proposal.target_belief_id,
                created_belief_id=belief_id,
                session_id=proposal.session_id,
                actor_type=proposal.actor_type,
                actor_id=proposal.actor_id,
                expected_revision=proposal.expected_revision,
                revision_after=revision_after,
            )
        )
        self._session.flush()

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

        # Revision.  Truth mutations must name the revision they were
        # computed against — a proposal without one is refused rather than
        # silently skipping the stale check.  The compare-and-swap itself
        # happens in the commit; this is the cheap pre-check.
        current = get_state_revision(self._session, proposal.user_id)
        if proposal.transition in TRUTH_TRANSITIONS:
            if not proposal.expected_revision:
                return "expected_revision_required"
            if current != proposal.expected_revision:
                return f"stale_revision(expected={proposal.expected_revision}, current={current})"
        elif proposal.expected_revision and current != proposal.expected_revision:
            return f"stale_revision(expected={proposal.expected_revision}, current={current})"

        # Authority: the evidence must exist and belong to the same user.
        if proposal.evidence_ids:
            rows = (
                self._session.query(Evidence)
                .filter(Evidence.id.in_(proposal.evidence_ids))
                .all()
            )
            if len(rows) != len(set(proposal.evidence_ids)):
                return "evidence_missing"
            if any(row.user_id != proposal.user_id for row in rows):
                return "evidence_authority_mismatch"
            # Scope: evidence recorded in a different session than the one
            # the proposal claims to come from is a cross-session write.
            if proposal.session_id:
                foreign_session = [
                    row.id for row in rows if row.session_id and row.session_id != proposal.session_id
                ]
                if foreign_session:
                    return "evidence_session_mismatch"

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
            # Superseded rows are history; they are closed, not writable --
            # except by a revival, whose whole purpose is to reopen a
            # previous value as the current one (A -> B -> A).  The
            # resolver marks those, and the ledger records the return as a
            # new version rather than an edit of the closed one.
            if proposal.transition == TRANSITION_UPDATE and proposal.payload.get("revival"):
                return None
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

    def _commit(self, proposal: StateTransitionProposal) -> tuple[int, int | None]:
        """Apply the transition.  Returns ``(revision_after, belief_id)``.

        For a truth mutation the revision is advanced by a compare-and-swap
        *here*, before the handler runs, and the handler is told not to bump
        it again: the Publisher is the single component that moves the
        revision, so a committed truth change advances it exactly once.  A
        CAS that lands on nothing means a concurrent publisher won the race
        between the gate check and here — the exception rolls the savepoint
        back and the caller gets a refusal.
        """
        if proposal.transition in TRUTH_TRANSITIONS:
            new_revision = cas_bump_state_revision(
                self._session, proposal.user_id, proposal.expected_revision
            )
            if new_revision is None:
                raise StaleRevisionError(
                    f"cas failed: revision moved from {proposal.expected_revision}"
                )
            revision_after = new_revision
        else:
            revision_after = None

        handler = {
            TRANSITION_CREATE: self._commit_create,
            TRANSITION_SUPPORT: self._commit_support,
            TRANSITION_UPDATE: self._commit_update,
            TRANSITION_CONTRADICT: self._commit_contradict,
            TRANSITION_CORRECT: self._commit_correct,
            TRANSITION_FORGET: self._commit_forget,
            TRANSITION_VERIFY: self._commit_verify,
            TRANSITION_REJECT: self._commit_reject,
            TRANSITION_SYNTHESIZE: self._commit_synthesize,
            TRANSITION_TIER_TRANSITION: self._commit_tier_transition,
            TRANSITION_EVIDENCE_COMPACT: self._commit_evidence_compact,
        }[proposal.transition]
        belief_id = handler(proposal)
        # The ledger records a new version only when the slot's *value*
        # changes; SUPPORT / VERIFY / CONTRADICT deepen or contest the same
        # value, and REJECT closes it.  The append happens inside the same
        # savepoint as the change that produced it, so an interval can never
        # exist without the commit that opened it (and vice versa).
        if proposal.transition in VALUE_TRANSITIONS and belief_id is not None:
            self._append_version(proposal, belief_id)
        elif proposal.transition == TRANSITION_REJECT and belief_id is not None:
            self._close_version(belief_id)
        if revision_after is None:
            revision_after = get_state_revision(self._session, proposal.user_id)
        return revision_after, belief_id

    def _append_version(self, proposal: StateTransitionProposal, belief_id: int) -> None:
        """Project the changed belief onto the versioned ledger."""
        from mirror_memory.core.repository import append_belief_version

        belief = self._session.get(Belief, belief_id)
        if belief is None:
            return
        ledger = self._ledger_limits()
        append_belief_version(
            self._session,
            belief,
            proposal_id=proposal.proposal_id,
            max_versions=ledger[0],
            version_ttl_days=ledger[1],
        )

    def _ledger_limits(self) -> tuple[int | None, int | None]:
        """(max versions, TTL days) from the config, or unbounded."""
        config = getattr(self, "_config", None)
        ledger = getattr(config, "metabolism", None)
        ledger = getattr(ledger, "ledger", None) if ledger is not None else None
        if ledger is None:
            return None, None
        return ledger.max_versions_per_identity, ledger.version_ttl_days

    def _close_version(self, belief_id: int) -> None:
        """REJECT closes the slot's open version rather than appending one."""
        from mirror_memory.core.repository import close_open_versions_for_belief

        close_open_versions_for_belief(self._session, belief_id)

    def _commit_create(self, proposal: StateTransitionProposal) -> int | None:
        payload = proposal.payload
        belief, _event = record_claim(
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
            bump_revision=False,
        )
        return belief.id if belief is not None else None

    def _commit_support(self, proposal: StateTransitionProposal) -> int | None:
        payload = proposal.payload
        support_belief_by_id(
            self._session,
            proposal.target_belief_id,
            claim_text=payload.get("claim_text", ""),
            session_id=proposal.session_id,
            evidence_message_ids=payload.get("evidence_message_ids"),
            bump_revision=False,
        )
        return proposal.target_belief_id

    def _commit_update(self, proposal: StateTransitionProposal) -> int | None:
        payload = proposal.payload
        old, new = update_belief_by_id(
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
            observed_at=payload.get("observed_at"),
            valid_from=payload.get("valid_from"),
            valid_to=payload.get("valid_to"),
            temporal_scope=payload.get("temporal_scope", ""),
            polarity=payload.get("polarity", ""),
            lifecycle_state=payload.get("lifecycle_state", ""),
            raw_predicate=payload.get("raw_predicate", ""),
            bump_revision=False,
        )
        # Round-trip revival: the resolver may name other same-attribute
        # beliefs to close (the values the user left and has now returned
        # past).  They close here, inside the Publisher's savepoint, so the
        # revival and its closes are one atomic transition — not a direct
        # ORM mutation performed by the caller afterwards.
        close_ids = [int(i) for i in payload.get("close_belief_ids") or []]
        if new is not None and close_ids:
            from mirror_memory.core.utils import coerce_datetime

            close_time = coerce_datetime(payload.get("close_at")) or utcnow()
            for close_id in close_ids:
                if close_id in (None, proposal.target_belief_id, new.id):
                    continue
                middle = self._session.get(Belief, close_id)
                if middle is None or middle.status != "active":
                    continue
                if middle.valid_to is None:
                    middle.valid_to = close_time
                middle.status = "superseded"
                middle.superseded_by = new.id
        return new.id if new is not None else None

    def _commit_contradict(self, proposal: StateTransitionProposal) -> int | None:
        payload = proposal.payload
        belief, _event = record_claim(
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
            cardinality=payload.get("cardinality", "multi"),
            conflict_kind=payload.get(
                "conflict_kind", CONFLICT_SOURCE_CONFLICT
            ),
            bump_revision=False,
        )
        return belief.id if belief is not None else None

    def _commit_correct(self, proposal: StateTransitionProposal) -> int | None:
        payload = proposal.payload
        corrected = correct_belief(
            self._session,
            proposal.user_id,
            proposal.target_belief_id,
            new_claim_text=payload.get("new_claim_text", ""),
            correction_note=payload.get("correction_note", ""),
            new_predicate=payload.get("new_predicate"),
            new_object=payload.get("new_object"),
            new_value=payload.get("new_value"),
            bump_revision=False,
        )
        return corrected.id if corrected is not None else None

    def _commit_forget(self, proposal: StateTransitionProposal) -> int | None:
        forget_belief(
            self._session,
            proposal.user_id,
            proposal.target_belief_id,
            bump_revision=False,
        )
        return None

    def _commit_verify(self, proposal: StateTransitionProposal) -> int | None:
        """VERIFY is a user confirmation: it raises the layer, it does not
        change the claim.  Handled through confirm_belief."""
        from mirror_memory.core.repository import confirm_belief

        confirm_belief(
            self._session,
            proposal.user_id,
            proposal.target_belief_id,
            bump_revision=False,
        )
        return proposal.target_belief_id

    def _commit_reject(self, proposal: StateTransitionProposal) -> int | None:
        """REJECT is the user denying a verification question: the belief is
        closed as rejected, and its key is never resurrected."""
        from mirror_memory.core.repository import reject_belief

        reject_belief(
            self._session,
            proposal.user_id,
            proposal.target_belief_id,
            bump_revision=False,
        )
        return proposal.target_belief_id

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
