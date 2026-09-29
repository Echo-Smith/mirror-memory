"""Deletion privacy — forgetting must remove the versioned ledger too.

``BeliefVersion.object`` is plaintext, so a forget that leaves the ledger
behind has not deleted the content: it has only hidden it from recall.
These tests pin both deletion paths (targeted forget and full-user delete)
against that, including the tombstone that must survive.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import (
    Base,
    BeliefIdentity,
    BeliefVersion,
    DeletionTombstone,
    ProposalLog,
)
from mirror_memory.core.proposal import (
    TRANSITION_CREATE,
    TRANSITION_FORGET,
    TRANSITION_UPDATE,
    StateTransitionProposal,
)
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    delete_user_memories,
    forget_belief,
    get_state_revision,
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


def _publish(session, proposal):
    return Publisher(session).publish(proposal)


def _create(session, key, predicate, object_, text):
    decision = _publish(
        session,
        StateTransitionProposal(
            transition=TRANSITION_CREATE,
            user_id="u1",
            session_id="s1",
            expected_revision=get_state_revision(session, "u1"),
            payload={
                "dimension": "fact",
                "key": key,
                "claim_text": text,
                "predicate": predicate,
                "object": object_,
                "cardinality": "single",
                "confidence": 0.8,
            },
        ),
    )
    assert decision.committed, decision.reason
    return decision.detail["belief_id"]


def _ledger_counts(session, user_id="u1"):
    return (
        session.query(BeliefVersion).filter_by(user_id=user_id).count(),
        session.query(BeliefIdentity).filter_by(user_id=user_id).count(),
        session.query(ProposalLog).filter_by(user_id=user_id).count(),
    )


def _plaintext_objects(session, user_id="u1"):
    return [
        row.object
        for row in session.query(BeliefVersion).filter_by(user_id=user_id)
    ]


class TestTargetedForgetPurgesLedger:
    def test_forget_removes_the_version_chain(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief_id = _create(db_session, "lives_in:shanghai", "lives_in", "shanghai",
                            "lives in Shanghai")
        # A second value opens a second version of the same slot.
        _publish(
            db_session,
            StateTransitionProposal(
                transition=TRANSITION_UPDATE,
                user_id="u1",
                target_belief_id=belief_id,
                expected_revision=get_state_revision(db_session, "u1"),
                payload={
                    "predicate": "lives_in", "object": "berlin",
                    "key": "lives_in:berlin", "claim_text": "lives in Berlin",
                    "confidence": 0.8, "cardinality": "single",
                },
            ),
        )
        versions, identities, _proposals = _ledger_counts(db_session)
        assert versions == 2
        assert identities == 1

        assert forget_belief(db_session, "u1", belief_id) is True

        versions, identities, _proposals = _ledger_counts(db_session)
        assert versions == 0, "the version chain still holds the deleted content"
        assert identities == 0, "the identity outlived its versions"
        assert _plaintext_objects(db_session) == []

    def test_forget_removes_the_publish_log(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief_id = _create(db_session, "my_name:lena", "name", "lena", "my name is Lena")
        _versions, _identities, proposals = _ledger_counts(db_session)
        assert proposals == 1

        assert forget_belief(db_session, "u1", belief_id) is True

        # The row that produced the deleted belief is gone; a direct
        # repository forget writes no log row of its own.
        assert db_session.query(ProposalLog).filter_by(user_id="u1").count() == 0

    def test_published_forget_keeps_only_its_own_audit_row(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief_id = _create(db_session, "my_name:lena", "name", "lena", "my name is Lena")

        decision = _publish(
            db_session,
            StateTransitionProposal(
                transition=TRANSITION_FORGET,
                user_id="u1",
                target_belief_id=belief_id,
                expected_revision=get_state_revision(db_session, "u1"),
            ),
        )
        assert decision.committed

        # The rows that produced or targeted the deleted belief are gone.
        # What remains is the FORGET's own entry, written after the purge —
        # the audit record that the deletion happened, carrying no content.
        remaining = db_session.query(ProposalLog).filter_by(user_id="u1").all()
        assert len(remaining) == 1
        assert remaining[0].transition == "FORGET"
        assert remaining[0].target_belief_id == belief_id
        assert _plaintext_objects(db_session) == []

    def test_tombstone_survives_the_forget(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief_id = _create(db_session, "age:34", "age", "34", "I am 34 years old")
        forget_belief(db_session, "u1", belief_id)

        tombstone = db_session.query(DeletionTombstone).one()
        assert tombstone.user_id == "u1"
        assert tombstone.scope == "belief"
        # The tombstone records that a deletion happened, never what it was.
        assert "34" not in tombstone.scope_hash
        assert "age" not in tombstone.scope_hash


class TestFullUserDeletePurgesLedger:
    def test_full_delete_removes_every_ledger_row(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        set_memory_enabled(db_session, "u2", True)
        _create(db_session, "lives_in:shanghai", "lives_in", "shanghai", "lives in Shanghai")
        other = _create(db_session, "lives_in:oslo", "lives_in", "oslo", "lives in Oslo")
        assert other  # u2's rows must survive u1's deletion
        # The second create above ran as u1 by default; make u2's explicit.
        db_session.query(BeliefVersion).filter_by(user_id="u2").delete()
        db_session.query(BeliefIdentity).filter_by(user_id="u2").delete()

        counts = delete_user_memories(db_session, "u1")

        assert counts["belief_versions"] >= 1
        assert counts["belief_identities"] >= 1
        assert counts["proposals"] >= 1
        assert _plaintext_objects(db_session) == []
        assert _ledger_counts(db_session)[0] == 0

    def test_tombstone_survives_full_delete(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        _create(db_session, "age:34", "age", "34", "I am 34 years old")
        delete_user_memories(db_session, "u1")
        tombstone = db_session.query(DeletionTombstone).one()
        assert tombstone.scope == "user"
