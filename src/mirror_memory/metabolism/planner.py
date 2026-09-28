"""Metabolism planner — the runtime that decides tier moves, as proposals.

The cycle (designed to run daily per user, e.g. at 02:00)::

    scan ──► score ──► protect ──► propose ──► publish ──► report

Every decision is expressed as a
:class:`~mirror_memory.core.proposal.StateTransitionProposal` with the
``TIER_TRANSITION`` transition and handed to the Publisher.  The planner
**never commits**: a refusal leaves the database byte-identical, exactly
like every other Compute node in the engine.  The Publisher re-derives
protection from live state, so even a buggy planner cannot cool something
protected.

Decision rules (all rule-based; no model in the loop):

- **demotion** (heat target below current tier): skipped when protected,
  when the retention class never cools (canonical), or when the belief has
  not been idle for ``cool_after_days``.  Proposes exactly one legal edge
  down (hot→warm, warm→dormant) — no skipping tiers, so a very cold belief
  cools over consecutive runs rather than in one jump.
- **reheat** (heat target above current tier): recalls (``last_accessed_at``)
  and new support are the signals; a dormant belief returns straight to hot
  per the state machine, and cools again naturally if unused.
- **archive** is deliberately not proposed yet (PR4); ``archive_candidate``
  heat maps to dormant here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.models import Belief
from mirror_memory.core.proposal import (
    TRANSITION_EVIDENCE_COMPACT,
    TRANSITION_TIER_TRANSITION,
    StateTransitionProposal,
)
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    evidence_graph_for_beliefs,
    get_state_revision,
    pending_verification_keys,
)
from mirror_memory.metabolism.compact import (
    DEFAULT_KEEP_POLICY,
    KeepPolicy,
    plan_compaction,
)
from mirror_memory.metabolism.heat import (
    HeatThresholds,
    HeatWeights,
    heat_score,
    heat_tier,
)
from mirror_memory.metabolism.policy import (
    RetentionPolicy,
    retention_policy_for,
)
from mirror_memory.metabolism.protection import (
    compaction_protection_reason,
    protection_reason,
)
from mirror_memory.metabolism.tiers import (
    TIER_ARCHIVED,
    TIER_DORMANT,
    TIER_HOT,
    TIER_WARM,
    is_legal_tier_transition,
    tier_rank,
)

# One legal step down from each tier.  A colder target still proposes only
# the next edge; the following run continues the descent if the belief stays
# cold.  The dormant -> archived edge is not here: archiving is driven by
# the retention class's archive_after_days clock (see decide_tier), not by
# the heat score.
_STEP_DOWN = {TIER_HOT: TIER_WARM, TIER_WARM: TIER_DORMANT}

REASON_SETTLED = "settled"
REASON_PROTECTED = "protected"
REASON_POLICY_NEVER_COOLS = "policy_never_cools"
REASON_NOT_IDLE_ENOUGH = "not_idle_enough"
REASON_CAPPED = "capped"
REASON_COOLING = "cooling"
REASON_REHEAT = "reheat"
# Archiving is driven by the retention class's archive_after_days clock,
# not by heat: a dormant belief with a long-ago last use is archived even
# if its evidence still scores warm.
REASON_ARCHIVE = "archive"
REASON_ARCHIVE_TOO_SOON = "archive_too_soon"
REASON_POLICY_NEVER_ARCHIVES = "policy_never_archives"

@dataclass
class TierDecision:
    """The planner's verdict for one belief."""

    belief_id: int
    key: str
    from_tier: str
    to_tier: str | None      # None = no proposal this run
    heat: float
    reason: str              # one of the REASON_* symbols, or "protected:<r>"
    # Why this belief is under protection ("" when it is not).  Kept
    # separately from *reason* because the protected floor (heat >= 0.70)
    # usually prevents a demotion from ever being attempted — the behaviour
    # reads as "settled"/"reheat", and the protection itself deserves its
    # own column in the report.
    protected_reason: str = ""


@dataclass
class MetabolismReport:
    """What one metabolism cycle did (or, on dry-run, would do)."""

    user_id: str
    scanned: int = 0
    protected: int = 0
    proposals: int = 0
    committed: int = 0
    refused: int = 0
    skipped_by_cap: int = 0
    dry_run: bool = False
    decisions: list[TierDecision] = field(default_factory=list)
    refusals: list[tuple[str, str]] = field(default_factory=list)  # (key, reason)
    tier_counts_before: dict[str, int] = field(default_factory=dict)
    tier_counts_after: dict[str, int] = field(default_factory=dict)
    # Evidence compaction (PR3).
    compaction_proposals: int = 0
    compacted_beliefs: int = 0
    folded_rows: int = 0

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "scanned": self.scanned,
            "protected": self.protected,
            "proposals": self.proposals,
            "committed": self.committed,
            "refused": self.refused,
            "skipped_by_cap": self.skipped_by_cap,
            "dry_run": self.dry_run,
            "compaction_proposals": self.compaction_proposals,
            "compacted_beliefs": self.compacted_beliefs,
            "folded_rows": self.folded_rows,
            "refusals": [{"key": k, "reason": r} for k, r in self.refusals],
            "tier_counts_before": dict(self.tier_counts_before),
            "tier_counts_after": dict(self.tier_counts_after),
            "transitions": [
                {
                    "belief_id": d.belief_id,
                    "key": d.key,
                    "from_tier": d.from_tier,
                    "to_tier": d.to_tier,
                    "heat": d.heat,
                    "reason": d.reason,
                    "protected_reason": d.protected_reason,
                }
                for d in self.decisions
                if d.to_tier is not None
            ],
        }


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _idle_days(belief: Belief, now: datetime) -> float:
    """Days since the belief last proved itself useful (recall or evidence)."""
    anchors = [
        _as_utc(getattr(belief, "last_accessed_at", None)),
        _as_utc(getattr(belief, "last_evidence_at", None)),
    ]
    stamps = [a for a in anchors if a is not None]
    if not stamps:
        return float("inf")
    return max((now - max(stamps)).total_seconds() / 86400.0, 0.0)


def decide_tier(
    belief: Belief,
    *,
    support_count: int,
    correct_count: int,
    pending_verification: bool,
    policies: dict[str, RetentionPolicy] | None = None,
    weights: HeatWeights | None = None,
    thresholds: HeatThresholds | None = None,
    half_life_days: float = 60.0,
    now: datetime | None = None,
) -> TierDecision:
    """Pure decision for one belief: which tier should it move to, if any?

    Separated from the scan/publish loop so the rules are testable without a
    database.  Protection blocks demotion (cooling *and* archiving) but
    never causes promotion: direction comes from the raw heat score, so a
    protected belief that nobody has used in a year stays where it is.
    """
    from_tier = belief.memory_tier or TIER_HOT
    protection = protection_reason(
        belief,
        support_count=support_count,
        correct_count=correct_count,
        pending_verification=pending_verification,
    )
    protection_name = protection or ""
    # Direction is decided on the *raw* heat: the protected floor exists to
    # block demotion, never to manufacture promotion — a dormant, thin,
    # long-unused belief must stay dormant even though it is protected.
    heat = heat_score(
        belief,
        now=now,
        evidence_count=support_count,
        weights=weights,
        half_life_days=half_life_days,
    )
    target = heat_tier(heat, thresholds)
    reference = now or datetime.now(UTC)
    rank_now, rank_target = tier_rank(from_tier), tier_rank(target)

    # Reheat first: a recalled / re-supported / confirmed belief climbs
    # back, and nothing (not even an overdue archive clock) may block
    # warming something the user is asking about.  From dormant the only
    # legal upward edge is straight to hot; the belief cools again on its
    # own if the renewed interest was a one-off.
    if rank_target < rank_now:
        if is_legal_tier_transition(from_tier, TIER_HOT):
            return TierDecision(
                belief.id, belief.key, from_tier, TIER_HOT, heat,
                REASON_REHEAT, protection_name,
            )
        return TierDecision(
            belief.id, belief.key, from_tier, None, heat, REASON_SETTLED, protection_name
        )

    # Archive: the retention class's clock, not the heat score, decides.
    # A dormant belief whose last use predates archive_after_days leaves
    # the default scan (archived rows are reachable only through an
    # explicit restore).  Protection vetoes; a class without an archive
    # schedule (canonical) is never archived by the runtime.
    if from_tier == TIER_DORMANT:
        if protection is not None:
            return TierDecision(
                belief.id, belief.key, from_tier, None, heat, REASON_SETTLED, protection_name
            )
        policy = retention_policy_for(belief.retention_class or "preference", policies)
        if policy.archive_after_days is None:
            return TierDecision(
                belief.id, belief.key, from_tier, None, heat,
                REASON_POLICY_NEVER_ARCHIVES, protection_name,
            )
        if _idle_days(belief, reference) >= float(policy.archive_after_days):
            return TierDecision(
                belief.id, belief.key, from_tier, TIER_ARCHIVED, heat,
                REASON_ARCHIVE, protection_name,
            )
        return TierDecision(
            belief.id, belief.key, from_tier, None, heat,
            REASON_ARCHIVE_TOO_SOON, protection_name,
        )

    if rank_target == rank_now:
        return TierDecision(
            belief.id, belief.key, from_tier, None, heat, REASON_SETTLED, protection_name
        )

    # Demotion by heat: protection and retention policy both get a veto.
    # (Unreachable while the protected floor holds — defense in depth
    # if the floor is ever configured away.)
    if protection is not None:
        return TierDecision(
            belief.id, belief.key, from_tier, None, heat,
            REASON_PROTECTED + ":" + protection, protection_name,
        )
    policy = retention_policy_for(belief.retention_class or "preference", policies)
    if policy.cool_after_days is None:
        return TierDecision(
            belief.id, belief.key, from_tier, None, heat,
            REASON_POLICY_NEVER_COOLS, protection_name,
        )
    if _idle_days(belief, reference) < float(policy.cool_after_days):
        return TierDecision(
            belief.id, belief.key, from_tier, None, heat,
            REASON_NOT_IDLE_ENOUGH, protection_name,
        )
    to_tier = _STEP_DOWN.get(from_tier)
    if to_tier is None:
        # Warm but already colder than warm maps to: keep stepping down
        # only along legal edges; anything else waits.
        return TierDecision(
            belief.id, belief.key, from_tier, None, heat, REASON_SETTLED, protection_name
        )
    return TierDecision(
        belief.id, belief.key, from_tier, to_tier, heat, REASON_COOLING, protection_name
    )


def _scan_beliefs(session: Session, user_id: str) -> list[Belief]:
    """Beliefs eligible for tier decisions this run.

    Rejected rows are never touched (the user's "avoid" boundary is not a
    cooling candidate); archived rows have left the default scan and come
    back only through an explicit restore, so there is nothing to decide
    about them here.
    """
    stmt = (
        select(Belief)
        .where(Belief.user_id == user_id)
        .where(Belief.status.in_(("active", "superseded")))
        .where(Belief.memory_tier.in_(("hot", "warm", "dormant")))
        .order_by(Belief.id)
    )
    return list(session.scalars(stmt))


def _tier_counts(session: Session, user_id: str) -> dict[str, int]:
    """Tier distribution over the user's beliefs (observability)."""
    stmt = select(Belief.memory_tier).where(Belief.user_id == user_id)
    counts = {"hot": 0, "warm": 0, "dormant": 0, "archived": 0}
    for tier in session.scalars(stmt):
        name = str(tier or "hot")
        counts[name] = counts.get(name, 0) + 1
    return counts


def run_metabolism(
    session: Session,
    user_id: str,
    config: MemoryConfig,
    *,
    now: datetime | None = None,
    publisher: Publisher | None = None,
    dry_run: bool = False,
) -> MetabolismReport:
    """Run one metabolism cycle for *user_id*.

    ``dry_run=True`` computes every decision (including what would be
    proposed) but publishes nothing.  The caller owns the transaction.
    """
    report = MetabolismReport(user_id=user_id, dry_run=dry_run)
    reference = now or datetime.now(UTC)

    beliefs = _scan_beliefs(session, user_id)
    report.scanned = len(beliefs)
    report.tier_counts_before = _tier_counts(session, user_id)

    graphs = evidence_graph_for_beliefs(session, [b.id for b in beliefs])
    pending_keys = pending_verification_keys(session, user_id)
    # One revision for the whole cycle: tier commits do not bump it, so all
    # proposals stay valid unless a *truth* write lands mid-run — in which
    # case the Publisher refuses the remainder, which is exactly the point.
    revision = get_state_revision(session, user_id)

    metab = config.metabolism
    weights = HeatWeights(**metab.heat.weights.model_dump())
    thresholds = HeatThresholds(**metab.heat.thresholds.model_dump())
    policies = {
        name: RetentionPolicy(cool_after_days=p.cool_after_days, archive_after_days=p.archive_after_days)
        for name, p in metab.retention_classes.items()
    }
    keep_policy = KeepPolicy(
        oldest=metab.compaction.keep.oldest,
        recent=metab.compaction.keep.recent,
        highest_authority=metab.compaction.keep.highest_authority,
    )
    min_support = metab.compaction.min_support_links
    max_transitions = metab.planner.max_transitions_per_run

    proposals: list[tuple[TierDecision, StateTransitionProposal]] = []
    compaction: list[tuple[str, StateTransitionProposal, int]] = []  # (key, proposal, fold count)
    for belief in beliefs:
        graph = graphs.get(belief.id, [])
        support_count = sum(1 for r in graph if r.relation == "support")
        correct_count = sum(1 for r in graph if r.relation == "correct")
        decision = decide_tier(
            belief,
            support_count=support_count,
            correct_count=correct_count,
            pending_verification=belief.key in pending_keys,
            policies=policies,
            weights=weights,
            thresholds=thresholds,
            half_life_days=metab.heat.half_life_days,
            now=reference,
        )
        report.decisions.append(decision)
        if decision.protected_reason:
            report.protected += 1

        # Evidence compaction: same cycle, same Publisher, but its own
        # protection rule — a corrected belief's redundant bulk is still
        # foldable (correct rows are always kept).  The belief's conclusion
        # is untouched; only the provenance bulk is aggregated.
        if not compaction_protection_reason(
            belief, pending_verification=belief.key in pending_keys
        ):
            plan = plan_compaction(
                graph, belief.id, policy=keep_policy, min_support=min_support
            )
            if plan is not None and plan.folds_anything:
                if len(proposals) + len(compaction) >= max_transitions:
                    report.skipped_by_cap += 1
                else:
                    compaction.append(
                        (
                            belief.key,
                            StateTransitionProposal(
                                transition=TRANSITION_EVIDENCE_COMPACT,
                                user_id=user_id,
                                target_belief_id=belief.id,
                                claimed_revision=revision,
                                payload={
                                    "keep_ids": list(plan.keep_ids),
                                    "fold_ids": list(plan.fold_ids),
                                    "policy": keep_policy.as_dict(),
                                    "min_support_links": min_support,
                                },
                            ),
                            len(plan.fold_ids),
                        )
                    )

        if decision.to_tier is None:
            continue
        if len(proposals) + len(compaction) >= max_transitions:
            decision.to_tier = None
            decision.reason = REASON_CAPPED
            report.skipped_by_cap += 1
            continue
        proposals.append(
            (
                decision,
                StateTransitionProposal(
                    transition=TRANSITION_TIER_TRANSITION,
                    user_id=user_id,
                    target_belief_id=decision.belief_id,
                    claimed_revision=revision,
                    payload={
                        "from_tier": decision.from_tier,
                        "to_tier": decision.to_tier,
                        "reason": decision.reason,
                        "score": decision.heat,
                    },
                ),
            )
        )

    report.proposals = len(proposals) + len(compaction)
    report.compaction_proposals = len(compaction)
    if not dry_run:
        pub = publisher or Publisher(session)
        for decision, proposal in proposals:
            result = pub.publish(proposal)
            if result.committed:
                report.committed += 1
            else:
                report.refused += 1
                report.refusals.append((decision.key, result.reason))
        for key, proposal, fold_count in compaction:
            result = pub.publish(proposal)
            if result.committed:
                report.committed += 1
                report.compacted_beliefs += 1
                report.folded_rows += fold_count
            else:
                report.refused += 1
                report.refusals.append((key, result.reason))

    report.tier_counts_after = _tier_counts(session, user_id)
    return report
