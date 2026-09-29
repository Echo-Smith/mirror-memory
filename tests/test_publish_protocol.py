"""Publish protocol (Runtime Closure PR1) — the Publisher as a contract.

Covers the four capabilities the protocol adds on top of the original
gate set, plus the scenarios that motivated each:

- **compare-and-swap revision** — the stale check is a single conditional
  UPDATE, so two concurrent publishers cannot both win; a truth mutation
  must name the revision it was computed against.
- **idempotency** — replaying a committed ``proposal_id`` is answered
  ``already_committed`` instead of applying the change twice.
- **savepoint atomicity** — a handler that fails halfway (belief written,
  event failed) leaves the database byte-identical.
- **authority + scope** — evidence ids must exist, belong to the user, and
  come from the session the proposal claims.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief, ProposalLog
from mirror_memory.core.proposal import (
    TRANSITION_CREATE,
    TRANSITION_FORGET,
    TRANSITION_SUPPORT,
    StateTransitionProposal,
)
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    get_state_revision,
    record_claim,
    record_evidence,
    set_memory_enabled,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(engine)


def _dump(session) -> list[tuple]:
    """Every belief row, as comparable tuples (detects any write at all)."""
    return [
        (b.id, b.user_id, b.key, b.claim_text, b.status, b.confidence, b.memory_tier)
        for b in session.query(Belief).order_by(Belief.id)
    ]


def _belief(session, user_id: str = "u1", key: str = "sleep"):
    belief, _ = record_claim(
        session, user_id, dimension="topic", key=key,
        claim_text="trouble sleeping", confidence=0.8,
        session_id="s1", evidence_message_ids=[1],
    )
    return belief


def _create_proposal(**overrides):
    """A CREATE proposal; *overrides* may carry proposal fields and payload
    entries (``payload`` is merged over the defaults)."""
    payload = {
        "dimension": "topic",
        "key": "sleep",
        "claim_text": "trouble sleeping",
        "confidence": 0.7,
    }
    payload.update(overrides.pop("payload", {}))
    for field in ("key", "claim_text", "confidence"):
        if field in overrides:
            payload[field] = overrides.pop(field)
    proposal = StateTransitionProposal(
        transition=TRANSITION_CREATE,
        user_id="u1",
        session_id="s1",
        payload=payload,
        **overrides,
    )
    return proposal


# ---------------------------------------------------------------------------
# expected_revision is mandatory for truth mutations
# ---------------------------------------------------------------------------


class TestExpectedRevisionRequired:
    def test_create_without_revision_refused(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        before = _dump(db_session)
        decision = Publisher(db_session).publish(_create_proposal())
        assert not decision.committed
        assert decision.reason == "expected_revision_required"
        assert _dump(db_session) == before

    @pytest.mark.parametrize(
        "transition",
        [TRANSITION_SUPPORT, TRANSITION_FORGET],
    )
    def test_truth_mutations_require_revision(self, db_session, transition):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        before = _dump(db_session)
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=transition, user_id="u1", target_belief_id=belief.id
            )
        )
        assert not decision.committed
        assert decision.reason == "expected_revision_required"
        assert _dump(db_session) == before

    def test_storage_transitions_may_omit_revision(self, db_session):
        """Tier moves and synthesis do not assert truth, so no revision."""
        set_memory_enabled(db_session, "u1", True)
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition="SYNTHESIZE", user_id="u1",
                payload={"content": {"x": 1}, "policy": {}, "watermark": "w"},
            )
        )
        assert decision.committed


# ---------------------------------------------------------------------------
# Compare-and-swap
# ---------------------------------------------------------------------------


class TestCompareAndSwap:
    def test_stale_revision_refused_and_db_untouched(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        before = _dump(db_session)
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_SUPPORT, user_id="u1",
                target_belief_id=belief.id, expected_revision=1,
                payload={"claim_text": "newer"},
            )
        )
        assert not decision.committed
        assert decision.reason.startswith("stale_revision")
        assert _dump(db_session) == before

    def test_cas_bumps_exactly_once(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        revision = get_state_revision(db_session, "u1")
        decision = Publisher(db_session).publish(
            _create_proposal(expected_revision=revision)
        )
        assert decision.committed
        # The Publisher owns the bump; the handler does not bump again.
        assert get_state_revision(db_session, "u1") == revision + 1
        assert decision.revision_after == revision + 1

    def test_concurrent_publishers_only_one_wins(self, db_session):
        """Two sessions publish against the same expected revision."""
        engine = db_session.get_bind()
        other_factory = sessionmaker(bind=engine)
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        revision = get_state_revision(db_session, "u1")
        db_session.commit()

        # Session B reads the same revision, then A commits first.
        session_b = other_factory()
        try:
            first = Publisher(db_session).publish(
                StateTransitionProposal(
                    transition=TRANSITION_SUPPORT, user_id="u1",
                    target_belief_id=belief.id, expected_revision=revision,
                    payload={"claim_text": "writer A"},
                )
            )
            assert first.committed
            second = Publisher(session_b).publish(
                StateTransitionProposal(
                    transition=TRANSITION_SUPPORT, user_id="u1",
                    target_belief_id=belief.id, expected_revision=revision,
                    payload={"claim_text": "writer B"},
                )
            )
            assert not second.committed
            assert "stale_revision" in second.reason
        finally:
            session_b.rollback()
            session_b.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_replay_is_already_committed(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        revision = get_state_revision(db_session, "u1")
        proposal = _create_proposal(expected_revision=revision)

        first = Publisher(db_session).publish(proposal)
        assert first.committed
        assert db_session.query(Belief).count() == 1
        revision_after_first = get_state_revision(db_session, "u1")

        # Same proposal object replayed: no second write, no second bump.
        second = Publisher(db_session).publish(proposal)
        assert not second.committed
        assert second.reason.startswith("already_committed")
        assert db_session.query(Belief).count() == 1
        assert get_state_revision(db_session, "u1") == revision_after_first
        assert db_session.query(ProposalLog).count() == 1

    def test_shared_idempotency_key_collapses_retries(self, db_session):
        """A *new* proposal object for the same intended change."""
        set_memory_enabled(db_session, "u1", True)
        revision = get_state_revision(db_session, "u1")
        first = Publisher(db_session).publish(
            _create_proposal(expected_revision=revision, idempotency_key="turn-42")
        )
        assert first.committed

        # The caller timed out and rebuilt the proposal; the key says it is
        # the same change.
        retry = Publisher(db_session).publish(
            _create_proposal(expected_revision=revision, idempotency_key="turn-42")
        )
        assert not retry.committed
        assert retry.reason.startswith("already_committed")
        assert db_session.query(Belief).count() == 1

    def test_proposal_log_records_actor_and_revision(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        revision = get_state_revision(db_session, "u1")
        Publisher(db_session).publish(
            _create_proposal(
                expected_revision=revision,
                actor_type="worker", actor_id="job-7",
            )
        )
        row = db_session.query(ProposalLog).one()
        assert row.actor_type == "worker"
        assert row.actor_id == "job-7"
        assert row.expected_revision == revision
        assert row.revision_after == revision + 1
        assert row.transition == TRANSITION_CREATE


# ---------------------------------------------------------------------------
# Savepoint atomicity
# ---------------------------------------------------------------------------


class _ExplodingAfterBelief:
    """A stand-in that writes a belief, then raises like a failed event write.

    Explodes on the first call only, so a test can still publish a
    follow-up proposal on the same session afterwards.
    """

    def __init__(self, session):
        self._session = session
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls > 1:
            # args already carries the session (the Publisher passes it
            # positionally), so forward verbatim.
            return record_claim(*args, **kwargs)
        belief, _ = record_claim(
            self._session, "u1", dimension="topic", key="halfway",
            claim_text="written before the failure", confidence=0.6,
        )
        self._session.flush()
        raise RuntimeError("event write failed")


class TestSavepointAtomicity:
    def test_handler_failure_leaves_db_untouched(self, db_session, monkeypatch):
        set_memory_enabled(db_session, "u1", True)
        revision = get_state_revision(db_session, "u1")
        before = _dump(db_session)

        # Patch the CREATE handler with one that writes a belief and then
        # fails, exactly where an event write would.
        from mirror_memory.core import publisher as publisher_module

        monkeypatch.setattr(
            publisher_module, "record_claim", _ExplodingAfterBelief(db_session)
        )
        decision = Publisher(db_session).publish(
            _create_proposal(expected_revision=revision)
        )
        assert not decision.committed
        assert decision.reason == "commit_error:RuntimeError"
        # The savepoint rolled the belief write back with the failure.
        assert _dump(db_session) == before
        assert get_state_revision(db_session, "u1") == revision
        assert db_session.query(ProposalLog).count() == 0

    def test_session_usable_after_refused_commit(self, db_session, monkeypatch):
        """A rolled-back savepoint must not poison the caller's transaction."""
        set_memory_enabled(db_session, "u1", True)
        revision = get_state_revision(db_session, "u1")

        from mirror_memory.core import publisher as publisher_module

        monkeypatch.setattr(
            publisher_module, "record_claim", _ExplodingAfterBelief(db_session)
        )
        Publisher(db_session).publish(_create_proposal(expected_revision=revision))

        # The next proposal still commits on the same session.
        follow_up = Publisher(db_session).publish(
            _create_proposal(
                key="after", expected_revision=get_state_revision(db_session, "u1")
            )
        )
        assert follow_up.committed
        assert db_session.query(Belief).filter_by(key="after").count() == 1


# ---------------------------------------------------------------------------
# Authority + scope
# ---------------------------------------------------------------------------


class TestAuthorityAndScope:
    def test_missing_evidence_refused(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        before = _dump(db_session)
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_SUPPORT, user_id="u1",
                target_belief_id=belief.id,
                evidence_ids=[999999],  # does not exist
                expected_revision=get_state_revision(db_session, "u1"),
            )
        )
        assert not decision.committed
        assert decision.reason == "evidence_missing"
        assert _dump(db_session) == before

    def test_foreign_evidence_refused(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        set_memory_enabled(db_session, "u2", True)
        belief = _belief(db_session)
        foreign = record_evidence(db_session, "u2", ref="m9", content="not yours")
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_SUPPORT, user_id="u1",
                target_belief_id=belief.id,
                evidence_ids=[foreign.id],
                expected_revision=get_state_revision(db_session, "u1"),
            )
        )
        assert not decision.committed
        assert decision.reason == "evidence_authority_mismatch"

    def test_cross_session_evidence_refused(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        other_session = record_evidence(
            db_session, "u1", ref="m-other", content="from another session",
            session_id="s-other",
        )
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_SUPPORT, user_id="u1",
                target_belief_id=belief.id,
                evidence_ids=[other_session.id],
                session_id="s1",
                expected_revision=get_state_revision(db_session, "u1"),
            )
        )
        assert not decision.committed
        assert decision.reason == "evidence_session_mismatch"

    def test_same_session_evidence_accepted(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _belief(db_session)
        own = record_evidence(
            db_session, "u1", ref="m-s1", content="from this session", session_id="s1"
        )
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_SUPPORT, user_id="u1",
                target_belief_id=belief.id,
                evidence_ids=[own.id],
                expected_revision=get_state_revision(db_session, "u1"),
            )
        )
        assert decision.committed

    def test_target_from_another_user_refused(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        set_memory_enabled(db_session, "u2", True)
        other, _ = record_claim(
            db_session, "u2", dimension="topic", key="sleep",
            claim_text="their sleep", confidence=0.8,
        )
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_SUPPORT, user_id="u1",
                target_belief_id=other.id,
                expected_revision=get_state_revision(db_session, "u1"),
            )
        )
        assert not decision.committed
        assert decision.reason == "target_belief_scope_mismatch"


# ---------------------------------------------------------------------------
# Protocol end to end
# ---------------------------------------------------------------------------


class TestProtocolEndToEnd:
    def test_full_lifecycle_advances_revision_once_each(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        revision = get_state_revision(db_session, "u1")

        created = Publisher(db_session).publish(_create_proposal(expected_revision=revision))
        assert created.committed
        revision += 1

        belief = db_session.query(Belief).one()
        supported = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_SUPPORT, user_id="u1",
                target_belief_id=belief.id, expected_revision=revision,
                payload={"claim_text": "still true"},
            )
        )
        assert supported.committed
        revision += 1

        forgotten = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_FORGET, user_id="u1",
                target_belief_id=belief.id, expected_revision=revision,
            )
        )
        assert forgotten.committed
        assert db_session.query(Belief).count() == 0
        assert get_state_revision(db_session, "u1") == revision + 1
        # The CREATE and SUPPORT rows named the forgotten belief, so the
        # forget purged them; what remains is the FORGET's own audit row.
        assert db_session.query(ProposalLog).count() == 1
        assert db_session.query(ProposalLog).one().transition == "FORGET"
