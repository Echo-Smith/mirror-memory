"""Evidence compaction (PR3) — selection rules, EVIDENCE_COMPACT proposals,
Publisher gates, digest upsert, and read-side behaviour.

Covers the user's KEEP rules (earliest 1 / latest 3 / top-authority 2,
correct+contradict+verify never folded), threshold gating, stale-selection
refusal, idempotence, evidence-level reheat on re-observation, and the
evidence-intent rendering.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mirror_memory.core.models import BeliefEvent, Evidence, EvidenceDigest
from mirror_memory.core.proposal import (
    TRANSITION_EVIDENCE_COMPACT,
    StateTransitionProposal,
)
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    evidence_for_belief,
    evidence_graph_for_beliefs,
    evidence_summary,
    get_state_revision,
    link_evidence,
    record_claim,
    record_evidence,
)
from mirror_memory.metabolism.compact import (
    DEFAULT_KEEP_POLICY,
    build_digest_fields,
    clamp_policy,
    plan_compaction,
)
from mirror_memory.metabolism.planner import run_metabolism

NOW = datetime(2026, 9, 27, 2, 0, 0, tzinfo=UTC)


def seed_belief(session, key, *, supports=13, authority="user", step_days=10,
                extra_relations=(), confidence=0.6):
    """A belief with `supports` typed support links spread over time."""
    belief, _ = record_claim(
        session, "u1", dimension="preference", key=key,
        claim_text=f"likes {key.split(':')[-1]}", predicate="likes",
        object=key.split(":")[-1], cardinality="multi", confidence=confidence,
    )
    for i in range(supports):
        evidence = record_evidence(
            session, "u1", ref=f"{key}:m{i}", content=f"message about {key} number {i}",
            observed_at=NOW - timedelta(days=400 - i * step_days),
            authority=authority,
        )
        link_evidence(session, belief.id, evidence.id, relation="support")
    for relation, n in extra_relations:
        for j in range(n):
            evidence = record_evidence(
                session, "u1", ref=f"{key}:{relation}{j}",
                content=f"{relation} note {j}", observed_at=NOW - timedelta(days=5),
            )
            link_evidence(session, belief.id, evidence.id, relation=relation)
    session.flush()
    return belief


def _graph(session, belief_id):
    return evidence_graph_for_beliefs(session, [belief_id])[belief_id]


# ---------------------------------------------------------------------------
# Pure selection rules
# ---------------------------------------------------------------------------


class TestPlanCompaction:
    def test_below_threshold_returns_none(self, db_session):
        belief = seed_belief(db_session, "small", supports=12)
        assert plan_compaction(_graph(db_session, belief.id), belief.id) is None

    def test_above_threshold_folds_the_bulk(self, db_session):
        belief = seed_belief(db_session, "big", supports=13)
        plan = plan_compaction(_graph(db_session, belief.id), belief.id)
        assert plan is not None and plan.folds_anything
        # 13 supports, keep <= oldest 1 + recent 3 + top-authority 2 (union),
        # so at least 13 - 6 = 7 rows fold.
        assert len(plan.fold_ids) >= 7
        assert len(plan.keep_ids) <= 6
        assert set(plan.keep_ids) & set(plan.fold_ids) == set()

    def test_keep_set_honours_the_three_rules(self, db_session):
        belief = seed_belief(db_session, "rules", supports=13)
        rows = _graph(db_session, belief.id)
        plan = plan_compaction(rows, belief.id)
        by_time = sorted(
            (r for r in rows if r.relation == "support"), key=lambda r: r.observed_at
        )
        oldest = by_time[0].evidence_id
        recent = {r.evidence_id for r in by_time[-3:]}
        assert oldest in plan.keep_ids
        assert recent <= set(plan.keep_ids)

    def test_protected_relations_are_never_folded(self, db_session):
        belief = seed_belief(
            db_session, "guarded", supports=13,
            extra_relations=(("correct", 1), ("contradict", 2), ("verify", 1)),
        )
        rows = _graph(db_session, belief.id)
        plan = plan_compaction(rows, belief.id)
        protected_ids = {
            r.evidence_id for r in rows if r.relation in ("correct", "contradict", "verify")
        }
        assert protected_ids
        assert protected_ids <= set(plan.keep_ids)
        assert not (protected_ids & set(plan.fold_ids))

    def test_folded_rows_stay_folded(self, db_session):
        belief = seed_belief(db_session, "twice", supports=13)
        rows = _graph(db_session, belief.id)
        first = plan_compaction(rows, belief.id)
        # Simulate the first compaction's marks on the pure rows.
        folded = set(first.fold_ids)
        rerun_rows = [
            r._replace(retention_state="compacted" if r.evidence_id in folded else r.retention_state)
            for r in rows
        ]
        second = plan_compaction(rerun_rows, belief.id)
        assert second is not None
        # Nothing left live beyond the keep sample -> no second fold.
        assert not second.folds_anything
        assert set(second.keep_ids) <= set(first.keep_ids) | {
            r.evidence_id for r in rows if r.relation in ("correct", "contradict", "verify")
        }

    def test_clamp_policy_enforces_minimums(self):
        clamped = clamp_policy({"oldest": 0, "recent": -5, "highest_authority": 99})
        assert clamped.oldest >= 1
        assert clamped.recent >= 1
        assert clamped.highest_authority <= 10
        assert clamp_policy(None) == DEFAULT_KEEP_POLICY

    def test_digest_fields_aggregate_full_history(self, db_session):
        belief = seed_belief(
            db_session, "stats", supports=13, extra_relations=(("correct", 1),)
        )
        rows = _graph(db_session, belief.id)
        plan = plan_compaction(rows, belief.id)
        fields = build_digest_fields(rows, plan.keep_ids)
        assert fields["support_count"] == 13
        assert fields["correct_count"] == 1
        assert fields["first_seen_at"] is not None
        assert fields["last_seen_at"] is not None
        assert fields["first_seen_at"] < fields["last_seen_at"]
        assert "13 observations" in fields["summary"]
        assert fields["authority_distribution"] == {"user": 13}
        assert fields["representative_ids"] == sorted(plan.keep_ids)


# ---------------------------------------------------------------------------
# Publisher gates + commit
# ---------------------------------------------------------------------------


def _compact_proposal(belief, keep_ids, fold_ids, *, policy=None, min_support=12):
    return StateTransitionProposal(
        transition=TRANSITION_EVIDENCE_COMPACT,
        user_id="u1",
        target_belief_id=belief.id,
        payload={
            "keep_ids": list(keep_ids),
            "fold_ids": list(fold_ids),
            "policy": policy or DEFAULT_KEEP_POLICY.as_dict(),
            "min_support_links": min_support,
        },
    )


def _publish_plan(session, belief, **overrides):
    plan = plan_compaction(_graph(session, belief.id), belief.id)
    assert plan is not None and plan.folds_anything
    payload_policy = overrides.pop("policy", DEFAULT_KEEP_POLICY.as_dict())
    proposal = _compact_proposal(belief, plan.keep_ids, plan.fold_ids, policy=payload_policy)
    for key, value in overrides.items():
        proposal.payload[key] = value
    return Publisher(session).publish(proposal), plan


class TestPublisherCompaction:
    def test_commit_folds_and_writes_digest(self, db_session):
        belief = seed_belief(db_session, "commitme", supports=13)
        decision, plan = _publish_plan(db_session, belief)
        assert decision.committed

        digest = db_session.query(EvidenceDigest).filter_by(belief_id=belief.id).one()
        assert digest.support_count == 13
        assert belief.compacted_into == digest.id
        assert sorted(safe_ids(digest.representative_ids)) == sorted(plan.keep_ids)

        folded = (
            db_session.query(Evidence)
            .filter(Evidence.id.in_(plan.fold_ids))
            .all()
        )
        assert all(row.retention_state == "compacted" for row in folded)
        assert all(row.content == "" for row in folded)
        assert all(row.compaction_group_id == str(digest.id) for row in folded)
        kept = db_session.query(Evidence).filter(Evidence.id.in_(plan.keep_ids)).all()
        assert all(row.retention_state == "representative" for row in kept)
        assert any(row.content for row in kept)

        events = (
            db_session.query(BeliefEvent)
            .filter_by(belief_id=belief.id, event_type="evidence_compacted")
            .all()
        )
        assert len(events) == 1

    def test_compaction_does_not_bump_state_revision(self, db_session):
        belief = seed_belief(db_session, "rev", supports=13)
        before = get_state_revision(db_session, "u1")
        decision, _plan = _publish_plan(db_session, belief)
        assert decision.committed
        assert get_state_revision(db_session, "u1") == before

    def test_tampered_selection_refused(self, db_session):
        belief = seed_belief(db_session, "tamper", supports=13)
        plan = plan_compaction(_graph(db_session, belief.id), belief.id)
        # Move one fold row into keep: the Publisher recomputes and refuses.
        keep = list(plan.keep_ids) + [plan.fold_ids[0]]
        fold = [i for i in plan.fold_ids if i != plan.fold_ids[0]]
        decision = Publisher(db_session).publish(
            _compact_proposal(belief, keep, fold)
        )
        assert not decision.committed
        assert decision.reason == "stale_selection"

    def test_forged_policy_clamped_into_mismatch(self, db_session):
        belief = seed_belief(db_session, "forge", supports=13)
        plan = plan_compaction(_graph(db_session, belief.id), belief.id)
        # A zero-keep policy is clamped to >=1 per slot, so the forged
        # selection (everything folded) no longer matches the recomputed one.
        decision = Publisher(db_session).publish(
            _compact_proposal(belief, plan.keep_ids, plan.fold_ids,
                              policy={"oldest": 0, "recent": 0, "highest_authority": 0})
        )
        assert not decision.committed
        assert decision.reason == "stale_selection"

    def test_user_correction_does_not_block_compaction(self, db_session):
        belief = seed_belief(
            db_session, "corrected", supports=13, extra_relations=(("correct", 1),)
        )
        plan = plan_compaction(_graph(db_session, belief.id), belief.id)
        assert plan is not None
        decision = Publisher(db_session).publish(
            _compact_proposal(belief, plan.keep_ids, plan.fold_ids)
        )
        assert decision.committed
        # The corrected row is never folded.
        kept = set(plan.keep_ids)
        corrected_ids = {
            r.evidence_id for r in _graph(db_session, belief.id) if r.relation == "correct"
        }
        assert corrected_ids <= kept

    def test_unresolved_conflict_blocks_compaction(self, db_session):
        import json

        belief = seed_belief(db_session, "disputed", supports=13)
        belief.value_json = json.dumps({"clarification_status": "needs_clarification"})
        db_session.flush()
        plan = plan_compaction(_graph(db_session, belief.id), belief.id)
        assert plan is not None
        decision = Publisher(db_session).publish(
            _compact_proposal(belief, plan.keep_ids, plan.fold_ids)
        )
        assert not decision.committed
        assert decision.reason == "protected:unresolved_conflict"

    def test_pending_verification_blocks_compaction(self, db_session):
        from mirror_memory.core.repository import record_intervention_event

        belief = seed_belief(db_session, "asked", supports=13)
        record_intervention_event(
            db_session, "u1", "s1", kind="question_injected",
            detail={"belief_key": belief.key},
        )
        plan = plan_compaction(_graph(db_session, belief.id), belief.id)
        assert plan is not None
        decision = Publisher(db_session).publish(
            _compact_proposal(belief, plan.keep_ids, plan.fold_ids)
        )
        assert not decision.committed
        assert decision.reason == "protected:pending_verification"

    def test_below_threshold_refused_even_with_forged_payload(self, db_session):
        belief = seed_belief(db_session, "tiny", supports=12)
        decision = Publisher(db_session).publish(
            _compact_proposal(belief, keep_ids=[], fold_ids=[1, 2], min_support=1)
        )
        # The Publisher recomputes against the payload's own threshold but
        # the live graph has nothing foldable beyond the keep sample.
        assert not decision.committed
        assert decision.reason in ("below_compaction_threshold", "nothing_to_fold", "stale_selection")


def safe_ids(raw):
    import json

    return [int(i) for i in json.loads(raw or "[]")]


# ---------------------------------------------------------------------------
# Planner cycle integration
# ---------------------------------------------------------------------------


class TestCycleCompaction:
    def test_cycle_proposes_and_commits_compaction(self, db_session, config):
        belief = seed_belief(db_session, "cycle", supports=13)
        belief.last_evidence_at = NOW  # fresh: no tier move, pure compaction
        db_session.flush()
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert report.compaction_proposals == 1
        assert report.compacted_beliefs == 1
        assert report.folded_rows >= 7
        assert belief.compacted_into is not None

    def test_second_run_is_idempotent(self, db_session, config):
        belief = seed_belief(db_session, "once", supports=13)
        run_metabolism(db_session, "u1", config, now=NOW)
        second = run_metabolism(db_session, "u1", config, now=NOW)
        assert second.compaction_proposals == 0
        assert db_session.query(EvidenceDigest).filter_by(belief_id=belief.id).count() == 1

    def test_protected_belief_never_proposed(self, db_session, config):
        import json

        belief = seed_belief(db_session, "shielded", supports=13)
        belief.value_json = json.dumps({"clarification_status": "needs_clarification"})
        db_session.flush()
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert report.compaction_proposals == 0
        assert report.protected == 1

    def test_corrected_belief_compacts_with_correct_kept(self, db_session, config):
        belief = seed_belief(
            db_session, "keptcorrect", supports=13, extra_relations=(("correct", 1),)
        )
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert report.compacted_beliefs == 1
        digest = db_session.query(EvidenceDigest).filter_by(belief_id=belief.id).one()
        assert digest.correct_count == 1
        live = evidence_for_belief(db_session, belief.id, relations=("correct",))
        assert len(live) == 1

    def test_dry_run_compacts_nothing(self, db_session, config):
        belief = seed_belief(db_session, "planned", supports=13)
        report = run_metabolism(db_session, "u1", config, now=NOW, dry_run=True)
        assert report.compaction_proposals == 1
        assert report.compacted_beliefs == 0
        assert belief.compacted_into is None
        assert db_session.query(EvidenceDigest).count() == 0

    def test_new_evidence_after_compaction_folds_again(self, db_session, config):
        belief = seed_belief(db_session, "regrow", supports=13)
        run_metabolism(db_session, "u1", config, now=NOW)
        assert belief.compacted_into is not None
        # Six new observations arrive.
        for i in range(6):
            evidence = record_evidence(
                db_session, "u1", ref=f"regrow:new{i}", content=f"new message {i}",
                observed_at=NOW - timedelta(days=1),
            )
            link_evidence(db_session, belief.id, evidence.id, relation="support")
        db_session.flush()
        second = run_metabolism(db_session, "u1", config, now=NOW)
        assert second.compacted_beliefs == 1
        digest = db_session.query(EvidenceDigest).filter_by(belief_id=belief.id).one()
        assert digest.support_count == 19  # aggregate covers the whole history


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


class TestReadSide:
    def test_evidence_reads_skip_compacted_by_default(self, db_session, config):
        belief = seed_belief(db_session, "read", supports=13)
        run_metabolism(db_session, "u1", config, now=NOW)
        live = evidence_for_belief(db_session, belief.id)
        assert 0 < len(live) <= 6
        assert all(pair[0].retention_state != "compacted" for pair in live)
        everything = evidence_for_belief(db_session, belief.id, include_compacted=True)
        assert len(everything) == 13

    def test_summary_carries_the_digest(self, db_session, config):
        belief = seed_belief(db_session, "explain", supports=13)
        run_metabolism(db_session, "u1", config, now=NOW)
        summary = evidence_summary(db_session, "u1", belief.id)
        assert "digest" in summary
        assert summary["digest"]["support_count"] == 13
        assert "13 observations" in summary["digest"]["summary"]

    def test_reobserving_a_folded_message_unfolds_it(self, db_session, config):
        seed_belief(db_session, "reheat", supports=13)
        run_metabolism(db_session, "u1", config, now=NOW)
        folded = (
            db_session.query(Evidence)
            .filter(Evidence.retention_state == "compacted")
            .first()
        )
        assert folded is not None and folded.content == ""
        record_evidence(
            db_session, "u1", ref=folded.ref,
            content="message about reheat number 0 restored",
            observed_at=NOW,
        )
        db_session.flush()
        assert folded.retention_state == "hot"
        assert folded.content != ""
        assert folded.compaction_group_id is None


# ---------------------------------------------------------------------------
# Evidence-intent rendering
# ---------------------------------------------------------------------------


class TestEvidenceIntentRendering:
    def test_why_query_carries_digest_hint(self, db_session, config):
        from mirror_memory.render.renderer import render_memory_block

        belief = seed_belief(db_session, "why:coffee", supports=13)
        belief.claim_text = "likes coffee"
        belief.key = "why:coffee"
        run_metabolism(db_session, "u1", config, now=NOW)

        block = render_memory_block(
            db_session, "u1", config=config, language="en",
            user_message="why do they like coffee",
        )
        assert block is not None
        assert "[evidence]" in block
        assert "13 observations" in block

    def test_plain_query_gets_no_evidence_line(self, db_session, config):
        from mirror_memory.render.renderer import render_memory_block

        belief = seed_belief(db_session, "plain:coffee", supports=13)
        belief.claim_text = "likes coffee"
        run_metabolism(db_session, "u1", config, now=NOW)

        block = render_memory_block(
            db_session, "u1", config=config, language="en",
            user_message="what coffee do they like",
        )
        assert block is not None
        assert "[evidence]" not in block
