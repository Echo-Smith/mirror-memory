"""Tests for the unified Lifecycle + Publish runtime.

The rule under test: **nothing writes but the Publisher.**

Every state change is a :class:`StateTransitionProposal`, and the Publisher
commits it only after checking consent, revision, scope, authority, and the
per-transition invariants.  A refusal is a no-op -- the database is byte-for-
byte what it was, which is what makes the runtime trustworthy rather than
merely conventional.
"""

import pytest

from mirror_memory.core.models import Belief, BeliefEvent, Evidence, Snapshot
from mirror_memory.core.proposal import (
    BELIEF_MUTATING_TRANSITIONS,
    STATE_TRANSITIONS,
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
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    get_state_revision,
    record_claim,
    record_evidence,
    reject_belief,
    set_memory_enabled,
)


def _dump(session):
    """A fully comparable snapshot of every memory-state table."""
    out = {}
    for model in (Belief, BeliefEvent, Evidence, Snapshot):
        rows = [
            tuple(
                "" if (v := getattr(r, c.key)) is None else str(v)
                for c in model.__table__.columns
            )
            for r in session.query(model).all()
        ]
        out[model.__table__.name] = sorted(rows)
    return out


def _belief(session, key="sleep", **kwargs):
    belief, _ = record_claim(
        session, "u1", dimension="topic", key=key,
        claim_text="trouble sleeping", confidence=0.8,
        session_id="s1", evidence_message_ids=[1], **kwargs,
    )
    return belief


# ---------------------------------------------------------------------------
# Proposal shape
# ---------------------------------------------------------------------------


class TestProposal:
    def test_all_transitions_accepted(self):
        for transition in STATE_TRANSITIONS:
            proposal = StateTransitionProposal(transition=transition, user_id="u1")
            assert proposal.transition == transition

    def test_unknown_transition_rejected(self):
        with pytest.raises(ValueError, match="unknown state transition"):
            StateTransitionProposal(transition="DESTROY", user_id="u1")

    def test_synthesize_does_not_mutate_beliefs(self):
        proposal = StateTransitionProposal(transition=TRANSITION_SYNTHESIZE, user_id="u1")
        assert not proposal.mutates_beliefs

    @pytest.mark.parametrize("transition", BELIEF_MUTATING_TRANSITIONS)
    def test_belief_mutating_transitions(self, transition):
        proposal = StateTransitionProposal(transition=transition, user_id="u1")
        assert proposal.mutates_beliefs

    def test_describe_is_log_safe(self):
        proposal = StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=7
        )
        assert "SUPPORT" in proposal.describe()
        assert "u1" in proposal.describe()
        assert "7" in proposal.describe()

    def test_describe_without_target(self):
        proposal = StateTransitionProposal(transition=TRANSITION_CREATE, user_id="u1")
        assert "-" in proposal.describe()


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


class TestPublisherGates:
    def test_create_commits(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        revision = get_state_revision(db_session, "u1")
        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_CREATE, user_id="u1", session_id="s1",
            claimed_revision=revision,
            payload={"dimension": "topic", "key": "sleep", "claim_text": "x",
                     "confidence": 0.7},
        ))
        assert decision.committed
        assert db_session.query(Belief).count() == 1

    def test_memory_disabled_refuses(self, db_session):
        set_memory_enabled(db_session, "u1", False)
        before = _dump(db_session)
        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_CREATE, user_id="u1",
            payload={"dimension": "topic", "key": "sleep", "claim_text": "x"},
        ))
        assert not decision.committed
        assert decision.reason == "memory_disabled"
        assert _dump(db_session) == before

    def test_stale_revision_refuses(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        before = _dump(db_session)

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=belief.id,
            claimed_revision=1,  # the live revision has moved on
            payload={"claim_text": "newer"},
        ))
        assert not decision.committed
        assert decision.reason.startswith("stale_revision")
        assert _dump(db_session) == before

    def test_cross_user_evidence_refused(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        foreign = record_evidence(db_session, "u2", ref="m9", content="not yours")
        before = _dump(db_session)

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=belief.id,
            evidence_ids=[foreign.id],
            claimed_revision=get_state_revision(db_session, "u1"),
        ))
        assert not decision.committed
        assert decision.reason == "evidence_authority_mismatch"
        assert _dump(db_session) == before

    def test_own_evidence_accepted(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        evidence = record_evidence(db_session, "u1", ref="m1", content="mine")

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=belief.id,
            evidence_ids=[evidence.id],
            claimed_revision=get_state_revision(db_session, "u1"),
        ))
        assert decision.committed

    def test_missing_target_refused(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=9999,
        ))
        assert not decision.committed
        assert decision.reason == "target_belief_missing"

    def test_target_from_another_user_refused(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        set_memory_enabled(db_session, "u2", True)
        other, _ = record_claim(
            db_session, "u2", dimension="topic", key="sleep",
            claim_text="x", confidence=0.8, session_id="s2",
        )
        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=other.id,
        ))
        assert not decision.committed
        assert decision.reason == "target_belief_scope_mismatch"

    def test_rejected_belief_not_resurrected(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        reject_belief(db_session, "u1", belief.id)
        before = _dump(db_session)

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=belief.id,
        ))
        assert not decision.committed
        assert decision.reason == "resurrection_guard"
        assert _dump(db_session) == before

    def test_superseded_belief_is_history(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        from mirror_memory.core.repository import update_belief_by_id

        update_belief_by_id(db_session, belief.id, new_object="beijing")
        db_session.refresh(belief)
        assert belief.status == "superseded"

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=belief.id,
        ))
        assert not decision.committed
        assert decision.reason == "target_belief_superseded"


# ---------------------------------------------------------------------------
# Every transition commits
# ---------------------------------------------------------------------------


class TestPublisherCommits:
    def test_support_strengthens(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        before = belief.confidence

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SUPPORT, user_id="u1", target_belief_id=belief.id,
            claimed_revision=get_state_revision(db_session, "u1"),
            payload={"claim_text": "sleeps badly"},
        ))
        assert decision.committed
        assert decision.revision_after > 0
        assert belief.confidence > before

    def test_update_supersedes(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_UPDATE, user_id="u1", target_belief_id=belief.id,
            claimed_revision=get_state_revision(db_session, "u1"),
            payload={"object": "beijing", "claim_text": "lives in Berlin",
                     "confidence": 0.8},
        ))
        assert decision.committed
        db_session.refresh(belief)
        assert belief.status == "superseded"

    def test_contradict_attenuates(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        before = belief.confidence

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_CONTRADICT, user_id="u1", target_belief_id=belief.id,
            claimed_revision=get_state_revision(db_session, "u1"),
            payload={"key": belief.key, "claim_text": "actually not"},
        ))
        assert decision.committed
        assert belief.confidence < before

    def test_verify_confirms(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        assert belief.layer == "L4"

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_VERIFY, user_id="u1", target_belief_id=belief.id,
            claimed_revision=get_state_revision(db_session, "u1"),
        ))
        assert decision.committed
        assert belief.layer == "L2"
        assert belief.source == "user_confirmed"

    def test_correct_rewrites(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_CORRECT, user_id="u1", target_belief_id=belief.id,
            claimed_revision=get_state_revision(db_session, "u1"),
            payload={"new_claim_text": "sleeps fine actually"},
        ))
        assert decision.committed
        assert belief.claim_text == "sleeps fine actually"
        assert belief.source == "user_corrected"

    def test_forget_deletes(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        record_evidence(db_session, "u1", ref="1", content="mine")

        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_FORGET, user_id="u1", target_belief_id=belief.id,
            claimed_revision=get_state_revision(db_session, "u1"),
        ))
        assert decision.committed
        assert db_session.query(Belief).count() == 0

    def test_synthesize_writes_a_snapshot(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        decision = Publisher(db_session).publish(StateTransitionProposal(
            transition=TRANSITION_SYNTHESIZE, user_id="u1",
            claimed_revision=get_state_revision(db_session, "u1"),
            payload={"content": {"summary": "understood"}, "policy": {},
                     "watermark": "wm1"},
        ))
        assert decision.committed
        assert db_session.query(Snapshot).count() == 1
        # Derived state only -- no belief was touched.
        assert db_session.query(Belief).count() == 0


# ---------------------------------------------------------------------------
# Refusals are no-ops
# ---------------------------------------------------------------------------


class TestRefusalIsANoop:
    def test_every_refused_transition_leaves_the_db_untouched(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        record_evidence(db_session, "u1", ref="1", content="mine")

        refusals = [
            StateTransitionProposal(
                transition=TRANSITION_SUPPORT, user_id="u1",
                target_belief_id=belief.id, claimed_revision=1,
            ),
            StateTransitionProposal(
                transition=TRANSITION_UPDATE, user_id="u1",
                target_belief_id=belief.id, claimed_revision=1,
                payload={"object": "beijing"},
            ),
            StateTransitionProposal(
                transition=TRANSITION_FORGET, user_id="u1",
                target_belief_id=belief.id, claimed_revision=1,
            ),
            StateTransitionProposal(
                transition=TRANSITION_SYNTHESIZE, user_id="u1", claimed_revision=1,
                payload={"content": {"x": 1}},
            ),
        ]

        before = _dump(db_session)
        for proposal in refusals:
            decision = Publisher(db_session).publish(proposal)
            assert not decision.committed, proposal.transition
            assert _dump(db_session) == before, f"{proposal.transition} wrote something"

    def test_decision_carries_the_proposal(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        proposal = StateTransitionProposal(
            transition=TRANSITION_CREATE, user_id="u1",
            payload={"dimension": "topic", "key": "k", "claim_text": "c"},
        )
        decision = Publisher(db_session).publish(proposal)
        assert decision.proposal is proposal
        assert isinstance(decision, PublishDecision)


# ---------------------------------------------------------------------------
# The worker publishes only through the Publisher
# ---------------------------------------------------------------------------


class TestWorkerPublishesThroughPublisher:
    def test_worker_job_completes_via_publisher(self, db_session):
        import json

        from mirror_memory.config.loader import load_config
        from mirror_memory.worker.evolution import _process_single_job, enqueue_job

        set_memory_enabled(db_session, "u1", True)
        record_claim(
            db_session, "u1", dimension="topic", key="sleep",
            claim_text="User has trouble sleeping", confidence=0.8, session_id="s1",
        )
        record_claim(
            db_session, "u1", dimension="topic", key="work",
            claim_text="User works on a research team", confidence=0.8, session_id="s2",
        )

        config = load_config("config/")

        class FakeLLM:
            def generate(self, **kwargs):
                return json.dumps({"patterns": [], "support_policy": {}})

        config.llm_client = FakeLLM()

        job = enqueue_job(db_session, "u1", config)
        _process_single_job(db_session, job, config)

        assert job.status == "completed"
        assert db_session.query(Snapshot).count() == 1

    def test_stale_worker_job_writes_nothing(self, db_session, monkeypatch):
        from mirror_memory.config.loader import load_config
        from mirror_memory.core.repository import touch_memory_state
        from mirror_memory.worker.evolution import _process_single_job, enqueue_job

        set_memory_enabled(db_session, "u1", True)
        record_claim(
            db_session, "u1", dimension="topic", key="sleep",
            claim_text="User has trouble sleeping", confidence=0.8, session_id="s1",
        )
        config = load_config("config/")

        class FakeLLM:
            def generate(self, **kwargs):
                return '{"patterns": [], "support_policy": {}}'

        config.llm_client = FakeLLM()
        job = enqueue_job(db_session, "u1", config)

        def _formulate_with_concurrent_write(session, user_id, evidence, cfg):
            touch_memory_state(session, user_id)
            return {"patterns": [], "support_policy": {}}

        monkeypatch.setattr(
            "mirror_memory.worker.evolution._formulate", _formulate_with_concurrent_write
        )

        before = _dump(db_session)
        _process_single_job(db_session, job, config)

        assert _dump(db_session) == before
        assert job.status == "cancelled"
        assert job.error_code.startswith("stale_revision")
