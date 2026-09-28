"""Metabolism runtime (PR2) — planner, TIER_TRANSITION proposals, Publisher
invariants, and the audit log.

Covers the full cycle (scan → score → protect → propose → publish) plus every
Publisher-side gate, on real database sessions.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import MemoryTransition
from mirror_memory.core.proposal import (
    TRANSITION_TIER_TRANSITION,
    StateTransitionProposal,
)
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    get_state_revision,
    link_evidence,
    record_claim,
    record_evidence,
    record_intervention_event,
)
from mirror_memory.metabolism.planner import (
    REASON_COOLING,
    REASON_NOT_IDLE_ENOUGH,
    REASON_POLICY_NEVER_COOLS,
    REASON_REHEAT,
    REASON_SETTLED,
    decide_tier,
    run_metabolism,
)
from mirror_memory.metabolism.protection import REASON_THIN_EVIDENCE


NOW = datetime(2026, 9, 27, 2, 0, 0, tzinfo=UTC)


def make_belief(
    session,
    key,
    *,
    support_links=2,
    days_old=400,
    tier="hot",
    status="active",
    importance=0.5,
    **claim_kw,
):
    """A belief with typed support links and a controllable last-use clock."""
    params = dict(
        dimension="preference",
        key=key,
        claim_text=f"likes {key}",
        predicate="likes",
        object=key,
        cardinality="multi",
        confidence=0.6,
    )
    params.update(claim_kw)
    belief, _ = record_claim(session, "u1", **params)
    for i in range(support_links):
        evidence = record_evidence(session, "u1", ref=f"{key}:m{i}", content=f"msg {i}")
        link_evidence(session, belief.id, evidence.id, relation="support")
    stale = NOW - timedelta(days=days_old)
    belief.last_evidence_at = stale
    belief.last_supported_at = stale
    belief.memory_tier = tier
    belief.importance_score = importance
    if status != "active":
        belief.status = status
    session.flush()
    return belief


# ---------------------------------------------------------------------------
# decide_tier — the pure rules
# ---------------------------------------------------------------------------


class TestDecideTier:
    def test_hot_belief_at_target_settles(self, db_session):
        belief = make_belief(db_session, "fresh", days_old=0)
        decision = decide_tier(
            belief,
            support_count=2,
            correct_count=0,
            pending_verification=False,
            now=NOW,
        )
        # Fresh (0 days) heat ~0.5 → warm == hot? rank differs; fresh heat:
        # 0.3*1 + 0.08 + 0.15 = 0.53 → target warm → would demote, but idle
        # 0 < 180 → blocked.  Use enough support to stay hot instead.
        hot_belief = make_belief(db_session, "hotone", days_old=0, support_links=5)
        hot_belief.access_count = 20
        decision = decide_tier(
            hot_belief, support_count=5, correct_count=0, pending_verification=False, now=NOW
        )
        assert decision.to_tier is None
        assert decision.reason in (REASON_SETTLED, REASON_NOT_IDLE_ENOUGH)

    def test_cold_unprotected_cools_one_step(self, db_session):
        belief = make_belief(db_session, "cold", days_old=400)
        decision = decide_tier(
            belief, support_count=2, correct_count=0, pending_verification=False, now=NOW
        )
        assert decision.from_tier == "hot"
        assert decision.to_tier == "warm"      # one legal edge, never skipping
        assert decision.reason == REASON_COOLING

    def test_recent_belief_is_not_idle_enough(self, db_session):
        belief = make_belief(db_session, "recent", days_old=30)
        decision = decide_tier(
            belief, support_count=2, correct_count=0, pending_verification=False, now=NOW
        )
        assert decision.to_tier is None
        assert decision.reason == REASON_NOT_IDLE_ENOUGH

    def test_canonical_never_cools(self, db_session):
        belief = make_belief(
            db_session, "lives_in:shanghai",
            predicate="lives_in", cardinality="single", confidence=0.6,
        )
        decision = decide_tier(
            belief, support_count=2, correct_count=0, pending_verification=False, now=NOW
        )
        assert decision.to_tier is None
        assert decision.reason == REASON_POLICY_NEVER_COOLS

    def test_thin_evidence_is_protected_from_cooling(self, db_session):
        belief = make_belief(db_session, "thin", support_links=0)
        decision = decide_tier(
            belief, support_count=0, correct_count=0, pending_verification=False, now=NOW
        )
        # The protected floor (0.70) keeps the target at hot, so a hot
        # belief reads "settled" behaviourally — the protection itself is
        # reported on its own field.
        assert decision.to_tier is None
        assert decision.protected_reason == REASON_THIN_EVIDENCE
        # And the demotion branch stays guarded even without the floor.
        assert decision.reason != REASON_COOLING

    def test_dormant_belief_recalled_reheats_to_hot(self, db_session):
        belief = make_belief(db_session, "back", days_old=1, tier="dormant")
        belief.access_count = 10
        belief.last_accessed_at = NOW
        decision = decide_tier(
            belief, support_count=2, correct_count=0, pending_verification=False, now=NOW
        )
        assert decision.from_tier == "dormant"
        assert decision.to_tier == "hot"
        assert decision.reason == REASON_REHEAT

    def test_dormant_branch_is_policy_clock_driven(self, db_session):
        # Archiving arrived in PR4 and is driven by the retention class's
        # archive_after_days clock (preference: 720 days), not by heat
        # alone; tests/test_metabolism_archive.py covers the archive rules.
        belief = make_belief(db_session, "deepcold", days_old=900, tier="dormant", importance=0.0)
        decision = decide_tier(
            belief, support_count=2, correct_count=0, pending_verification=False, now=NOW
        )
        assert decision.from_tier == "dormant"
        assert decision.reason == "archive"  # 900 idle days > 720-day clock
        assert decision.to_tier == "archived"


# ---------------------------------------------------------------------------
# run_metabolism — the cycle
# ---------------------------------------------------------------------------


class TestRunMetabolism:
    def test_cycle_cools_old_belief_and_writes_audit(self, db_session, config):
        belief = make_belief(db_session, "coolme", days_old=400)
        report = run_metabolism(db_session, "u1", config, now=NOW)
        db_session.flush()

        assert report.scanned >= 1
        assert report.proposals == 1
        assert report.committed == 1
        assert belief.memory_tier == "warm"
        assert belief.metabolism_state == "cooling"

        audit = (
            db_session.query(MemoryTransition)
            .filter(MemoryTransition.entity_id == belief.id)
            .one()
        )
        assert audit.from_tier == "hot"
        assert audit.to_tier == "warm"
        assert audit.reason == REASON_COOLING
        assert report.decisions
        assert audit.score == pytest.approx(report.decisions[0].heat)

    def test_descent_is_gradual_across_runs(self, db_session, config):
        belief = make_belief(db_session, "slow", days_old=400)
        run_metabolism(db_session, "u1", config, now=NOW)
        assert belief.memory_tier == "warm"
        run_metabolism(db_session, "u1", config, now=NOW)
        assert belief.memory_tier == "dormant"
        # Third run: nothing left to propose (archive arrives in PR4).
        third = run_metabolism(db_session, "u1", config, now=NOW)
        assert third.committed == 0

    def test_recall_reheats_dormant_on_next_cycle(self, db_session, config):
        belief = make_belief(db_session, "revive", days_old=400, tier="dormant")
        # The user asks about it and recall surfaces it.
        belief.access_count = 5
        belief.last_accessed_at = NOW - timedelta(days=1)
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert belief.memory_tier == "hot"
        assert any(d.reason == REASON_REHEAT for d in report.decisions)

    def test_protected_beliefs_are_skipped_and_counted(self, db_session, config):
        make_belief(db_session, "thin", support_links=0)
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert report.protected == 1
        assert report.proposals == 0

    def test_superseded_history_is_coolable(self, db_session, config):
        belief = make_belief(db_session, "lives_in:shanghai", days_old=400, status="superseded")
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert belief.memory_tier == "warm"
        assert report.committed == 1

    def test_rejected_rows_are_never_scanned(self, db_session, config):
        make_belief(db_session, "avoided", days_old=400, status="rejected")
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert report.scanned == 0

    def test_dry_run_changes_nothing(self, db_session, config):
        belief = make_belief(db_session, "planned", days_old=400)
        report = run_metabolism(db_session, "u1", config, now=NOW, dry_run=True)
        assert report.proposals == 1
        assert report.committed == 0
        assert belief.memory_tier == "hot"
        assert db_session.query(MemoryTransition).count() == 0

    def test_transition_cap_defers_excess(self, db_session, config):
        config.metabolism.planner.max_transitions_per_run = 1
        make_belief(db_session, "a", days_old=400)
        make_belief(db_session, "b", days_old=400)
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert report.committed == 1
        assert report.skipped_by_cap == 1

    def test_tier_commit_does_not_bump_state_revision(self, db_session, config):
        belief = make_belief(db_session, "revcheck", days_old=400)
        before = get_state_revision(db_session, "u1")
        run_metabolism(db_session, "u1", config, now=NOW)
        db_session.flush()
        assert get_state_revision(db_session, "u1") == before
        assert belief.memory_tier == "warm"

    def test_stale_truth_write_refuses_remaining_cooling(self, db_session, config):
        from mirror_memory.core.repository import bump_state_revision

        belief = make_belief(db_session, "racy", days_old=400)
        # Simulate the truth state moving between scan and publish: the
        # planner captured the old revision, the Publisher must refuse.
        revision = get_state_revision(db_session, "u1")
        bump_state_revision(db_session, "u1")
        proposal = StateTransitionProposal(
            transition=TRANSITION_TIER_TRANSITION,
            user_id="u1",
            target_belief_id=belief.id,
            claimed_revision=revision,
            payload={"from_tier": "hot", "to_tier": "warm", "reason": "cooling", "score": 0.2},
        )
        decision = Publisher(db_session).publish(proposal)
        assert not decision.committed
        assert "stale_revision" in decision.reason
        assert belief.memory_tier == "hot"


# ---------------------------------------------------------------------------
# Publisher invariants for TIER_TRANSITION
# ---------------------------------------------------------------------------


def _tier_proposal(belief, from_tier, to_tier, **extra):
    payload = {"from_tier": from_tier, "to_tier": to_tier, "reason": "cooling", "score": 0.2}
    payload.update(extra)
    return StateTransitionProposal(
        transition=TRANSITION_TIER_TRANSITION,
        user_id="u1",
        target_belief_id=belief.id,
        payload=payload,
    )


class TestPublisherTierInvariants:
    def test_protection_is_rederived_not_trusted(self, db_session):
        belief = make_belief(db_session, "fixed", support_links=2)
        evidence = record_evidence(db_session, "u1", ref="fixed:correct", content="no, actually")
        link_evidence(db_session, belief.id, evidence.id, relation="correct")
        # A buggy planner proposes a demotion anyway.
        decision = Publisher(db_session).publish(_tier_proposal(belief, "hot", "warm"))
        assert not decision.committed
        assert decision.reason == "protected:user_correction"
        assert belief.memory_tier == "hot"

    def test_pending_verification_blocks_demotion(self, db_session):
        belief = make_belief(db_session, "asked", support_links=2)
        record_intervention_event(
            db_session, "u1", "s1",
            kind="question_injected",
            detail={"belief_key": belief.key, "belief_label": "asked"},
        )
        decision = Publisher(db_session).publish(_tier_proposal(belief, "hot", "warm"))
        assert not decision.committed
        assert decision.reason == "protected:pending_verification"

    def test_answered_verification_no_longer_blocks(self, db_session):
        belief = make_belief(db_session, "answered", support_links=2)
        belief.last_evidence_at = NOW - timedelta(days=400)
        db_session.flush()
        record_intervention_event(
            db_session, "u1", "s1", kind="question_injected",
            detail={"belief_key": belief.key},
        )
        record_intervention_event(
            db_session, "u1", "s1", kind="question_answered",
            detail={"verdict": "confirm"},
        )
        decision = Publisher(db_session).publish(_tier_proposal(belief, "hot", "warm"))
        assert decision.committed

    def test_stale_from_tier_refused(self, db_session):
        belief = make_belief(db_session, "moved", support_links=2)
        decision = Publisher(db_session).publish(_tier_proposal(belief, "warm", "dormant"))
        assert not decision.committed
        assert decision.reason.startswith("tier_mismatch")

    def test_illegal_edge_refused(self, db_session):
        belief = make_belief(db_session, "jumpy", support_links=2)
        decision = Publisher(db_session).publish(_tier_proposal(belief, "hot", "dormant"))
        assert not decision.committed
        assert decision.reason.startswith("illegal_tier_transition")

    def test_rejected_target_refused(self, db_session):
        belief = make_belief(db_session, "gone", support_links=2, status="rejected")
        decision = Publisher(db_session).publish(_tier_proposal(belief, "hot", "warm"))
        assert not decision.committed
        assert decision.reason == "resurrection_guard"

    def test_scope_mismatch_refused(self, db_session):
        belief = make_belief(db_session, "mine", support_links=2)
        proposal = _tier_proposal(belief, "hot", "warm")
        proposal.user_id = "u2"
        decision = Publisher(db_session).publish(proposal)
        assert not decision.committed
        assert decision.reason == "target_belief_scope_mismatch"

    def test_expired_memory_user_refused(self, db_session, config):
        from mirror_memory.core.repository import set_memory_enabled

        belief = make_belief(db_session, "off", support_links=2)
        set_memory_enabled(db_session, "u1", False)
        decision = Publisher(db_session).publish(_tier_proposal(belief, "hot", "warm"))
        assert not decision.committed
        assert decision.reason == "memory_disabled"


# ---------------------------------------------------------------------------
# Engine API smoke
# ---------------------------------------------------------------------------


class TestEngineRunMetabolism:
    def test_engine_cycle_on_empty_store(self, config):
        from mirror_memory.api import MemoryEngine

        engine = MemoryEngine(config=config, database_url="sqlite://")
        report = engine.run_metabolism(user_id="nobody")
        assert report["scanned"] == 0
        assert report["committed"] == 0
        assert report["tier_counts_before"] == {"hot": 0, "warm": 0, "dormant": 0, "archived": 0}

    def test_engine_cycle_cools_via_real_session(self, config):
        from mirror_memory.api import MemoryEngine

        engine = MemoryEngine(config=config, database_url="sqlite://")
        engine._ensure_db()
        with engine._session() as session:
            make_belief(session, "engine_cooled", days_old=400)
            session.commit()
        report = engine.run_metabolism(user_id="u1")
        assert report["committed"] == 1
        assert report["tier_counts_after"]["warm"] == 1


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


class TestPlannerConfig:
    def test_defaults_without_file(self, tmp_path):
        cfg = load_config(tmp_path)
        assert cfg.metabolism.planner.max_transitions_per_run == 200

    def test_loaded_from_yaml(self):
        cfg = load_config("config/")
        assert cfg.metabolism.planner.max_transitions_per_run == 200
