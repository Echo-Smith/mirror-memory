"""Versioned belief ledger (Runtime Closure PR3).

The ledger is the authoritative history: one :class:`BeliefIdentity` per
slot, one :class:`BeliefVersion` per value that slot has held, with the
truth clock and the belief clock recorded separately and the opening
proposal attached.  A→B→A is three versions of one identity — the first
stay in Shanghai is history that stays closed, not a row to reopen.

The ledger is written by the Publisher's truth commits (the only path
that writes), so every interval is traceable to the commit that opened it.
The existing ``Belief`` rows remain the compatibility read model.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief, BeliefIdentity, BeliefVersion
from mirror_memory.core.proposal import (
    TRANSITION_CORRECT,
    TRANSITION_CREATE,
    TRANSITION_SUPPORT,
    TRANSITION_UPDATE,
    StateTransitionProposal,
)
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    append_belief_version,
    backfill_belief_ledger,
    belief_history,
    belief_versions_for_identity,
    get_state_revision,
    record_claim,
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


def _publish(session, transition, *, target_belief_id=None, payload=None, user_id="u1"):
    return Publisher(session).publish(
        StateTransitionProposal(
            transition=transition,
            user_id=user_id,
            session_id="s1",
            target_belief_id=target_belief_id,
            payload=payload or {},
            expected_revision=get_state_revision(session, user_id),
        )
    )


def _create(session, key, *, predicate, object_, claim_text, cardinality="single", **extra):
    payload = {
        "dimension": "fact",
        "key": key,
        "claim_text": claim_text,
        "predicate": predicate,
        "object": object_,
        "cardinality": cardinality,
        "confidence": 0.8,
    }
    payload.update(extra)
    return _publish(session, TRANSITION_CREATE, payload=payload)


class TestIdentityAndVersions:
    def test_create_appends_first_version(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        decision = _create(
            db_session, "lives_in:shanghai",
            predicate="lives_in", object_="shanghai",
            claim_text="lives in Shanghai",
        )
        assert decision.committed
        belief_id = decision.detail["belief_id"]

        versions = belief_versions_for_identity(
            db_session, "u1", slot_key="lives_in"
        )
        assert len(versions) == 1
        version = versions[0]
        assert version.object == "shanghai"
        assert version.epistemic_to is None
        assert version.created_by_proposal == decision.proposal.proposal_id
        assert version.belief_id == belief_id
        assert db_session.query(BeliefIdentity).count() == 1

    def test_support_strengthens_without_new_version(self, db_session):
        """SUPPORT is not a new value of the slot: it deepens the same one."""
        set_memory_enabled(db_session, "u1", True)
        created = _create(
            db_session, "likes:coffee",
            predicate="likes", object_="coffee",
            claim_text="likes coffee", cardinality="multi",
        )
        _publish(
            db_session,
            TRANSITION_SUPPORT,
            target_belief_id=created.detail["belief_id"],
            payload={"claim_text": "still likes coffee"},
        )
        versions = belief_versions_for_identity(
            db_session, "u1", slot_key="likes:coffee"
        )
        assert len(versions) == 1

    def test_update_closes_the_previous_version(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        created = _create(
            db_session, "lives_in:shanghai",
            predicate="lives_in", object_="shanghai",
            claim_text="lives in Shanghai",
        )
        updated = _publish(
            db_session,
            TRANSITION_UPDATE,
            target_belief_id=created.detail["belief_id"],
            payload={
                "predicate": "lives_in",
                "object": "berlin",
                "key": "lives_in:berlin",
                "claim_text": "lives in Berlin",
                "confidence": 0.8,
            },
        )
        assert updated.committed

        versions = belief_versions_for_identity(
            db_session, "u1", slot_key="lives_in"
        )
        # The new value stamps its own logical slot key, and the chain is
        # one identity's history.
        assert [v.object for v in versions] == ["shanghai", "berlin"]
        assert versions[0].epistemic_to is not None
        assert versions[0].superseded_by_version == versions[1].id
        assert versions[1].epistemic_to is None
        # Both versions share one identity.
        assert versions[0].identity_id == versions[1].identity_id

    def test_round_trip_is_three_versions(self, db_session):
        """A→B→A: the first stay stays closed; the return is a third version."""
        set_memory_enabled(db_session, "u1", True)
        first = _create(
            db_session, "lives_in:shanghai",
            predicate="lives_in", object_="shanghai",
            claim_text="lives in Shanghai",
        )
        second = _publish(
            db_session,
            TRANSITION_UPDATE,
            target_belief_id=first.detail["belief_id"],
            payload={
                "predicate": "lives_in", "object": "berlin",
                "key": "lives_in:berlin", "claim_text": "lives in Berlin",
                "confidence": 0.8,
            },
        )
        third = _publish(
            db_session,
            TRANSITION_UPDATE,
            target_belief_id=second.detail["belief_id"],
            payload={
                "predicate": "lives_in", "object": "shanghai",
                "key": "lives_in:shanghai", "claim_text": "back in Shanghai",
                "confidence": 0.8,
            },
        )
        assert third.committed

        chain = belief_history(db_session, "u1", third.detail["belief_id"])
        assert [v["object"] for v in chain] == ["shanghai", "berlin", "shanghai"]
        assert chain[0]["epistemic_to"] is not None
        assert chain[1]["epistemic_to"] is not None
        assert chain[2]["epistemic_to"] is None
        # Each interval is traceable to the proposal that opened it.
        assert chain[0]["created_by_proposal"] == first.proposal.proposal_id
        assert chain[1]["created_by_proposal"] == second.proposal.proposal_id
        assert chain[2]["created_by_proposal"] == third.proposal.proposal_id
        # One identity, three versions.
        assert len({v["version_id"] for v in chain}) == 3
        identities = db_session.query(BeliefIdentity).count()
        assert identities == 1

    def test_correction_is_a_new_version_of_the_same_slot(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        created = _create(
            db_session, "works_at:acme",
            predicate="works_at", object_="acme",
            claim_text="works at Acme",
        )
        corrected = _publish(
            db_session,
            TRANSITION_CORRECT,
            target_belief_id=created.detail["belief_id"],
            payload={
                "new_claim_text": "works at Globex",
                "new_object": "Globex",
                "new_predicate": "works_at",
            },
        )
        assert corrected.committed
        chain = belief_history(db_session, "u1", corrected.detail["belief_id"])
        assert [v["object"] for v in chain] == ["acme", "Globex"]
        assert chain[0]["epistemic_to"] is not None

    def test_truth_and_belief_clocks_are_separate(self, db_session):
        """The fact's interval and the engine's belief interval are distinct."""
        from datetime import UTC, datetime, timedelta

        set_memory_enabled(db_session, "u1", True)
        started = datetime(2024, 1, 1, tzinfo=UTC)
        ended = datetime(2026, 5, 1, tzinfo=UTC)
        learned = started + timedelta(days=30)
        created = _create(
            db_session, "lives_in:shanghai",
            predicate="lives_in", object_="shanghai",
            claim_text="lives in Shanghai",
        )
        # The truth interval travels with an UPDATE (the path that sets
        # intervals), so the version inherits both clocks at append time.
        _publish(
            db_session,
            TRANSITION_UPDATE,
            target_belief_id=created.detail["belief_id"],
            payload={
                "predicate": "lives_in", "object": "shanghai",
                "key": "lives_in:shanghai",
                "claim_text": "lived in Shanghai 2024-2026",
                "confidence": 0.8,
                "valid_from": started, "valid_to": ended,
                "observed_at": learned,
            },
        )
        belief = db_session.query(Belief).filter_by(status="active").one()
        assert belief.observed_at is not None
        db_session.flush()

        versions = belief_versions_for_identity(
            db_session, "u1", slot_key="lives_in"
        )
        version = versions[-1]
        # The fact was true Jan 2024 - May 2026... (stored naive-UTC, the
        # engine's storage convention)...
        assert version.valid_from == started.replace(tzinfo=None)
        assert version.valid_to == ended.replace(tzinfo=None)
        # ...and the engine learned it a month later.
        assert version.epistemic_from == learned.replace(tzinfo=None)


class TestReadModelCompatibility:
    def test_belief_rows_still_answer_the_read_paths(self, db_session):
        """The read model is unchanged: retrieval/rendering keep working."""
        from mirror_memory.core.repository import list_active_beliefs

        set_memory_enabled(db_session, "u1", True)
        _create(
            db_session, "lives_in:shanghai",
            predicate="lives_in", object_="shanghai",
            claim_text="lives in Shanghai",
        )
        active = list_active_beliefs(db_session, "u1")
        assert len(active) == 1
        assert active[0].object == "shanghai"

    def test_history_of_unknown_belief_is_empty(self, db_session):
        assert belief_history(db_session, "u1", 4242) == []

    def test_history_of_other_users_belief_is_empty(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        set_memory_enabled(db_session, "u2", True)
        created = _create(
            db_session, "lives_in:shanghai",
            predicate="lives_in", object_="shanghai",
            claim_text="lives in Shanghai",
            user_id="u2",
        ) if False else _publish(
            db_session,
            TRANSITION_CREATE,
            payload={
                "dimension": "fact", "key": "lives_in:shanghai",
                "claim_text": "lives in Shanghai", "predicate": "lives_in",
                "object": "shanghai", "cardinality": "single", "confidence": 0.8,
            },
            user_id="u2",
        )
        assert created.committed
        assert belief_history(db_session, "u1", created.detail["belief_id"]) == []


# ---------------------------------------------------------------------------
# Backfill and capacity (ledger completion)
# ---------------------------------------------------------------------------


def _update_to(session, belief, *, object_, key):
    return Publisher(session).publish(
        StateTransitionProposal(
            transition=TRANSITION_UPDATE,
            user_id="u1",
            target_belief_id=belief.id,
            expected_revision=get_state_revision(session, "u1"),
            payload={
                "predicate": "lives_in", "object": object_,
                "key": key, "claim_text": f"lives in {object_}",
                "confidence": 0.8, "cardinality": "single",
            },
        )
    )


def _pre_ledger_belief(session, key="lives_in:a", object_="a"):
    belief, _ = record_claim(
        session, "u1", dimension="fact", key=key,
        claim_text=f"lives in {object_}", confidence=0.8,
        predicate="lives_in", object=object_, cardinality="single",
    )
    return belief


class TestBackfill:
    def test_pre_ledger_beliefs_get_an_opening_version(self, db_session):
        """A belief written without the ledger must still receive its first
        version, or the history chain starts mid-way."""
        set_memory_enabled(db_session, "u1", True)
        _pre_ledger_belief(db_session)
        assert db_session.query(BeliefVersion).count() == 0

        result = backfill_belief_ledger(db_session, "u1")

        assert result["versions_added"] == 1
        versions = belief_versions_for_identity(db_session, "u1", slot_key="lives_in")
        assert [v.object for v in versions] == ["a"]
        assert versions[0].created_by_proposal == "backfill"

    def test_backfill_is_idempotent(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        _pre_ledger_belief(db_session)
        backfill_belief_ledger(db_session, "u1")
        second = backfill_belief_ledger(db_session, "u1")
        assert second["versions_added"] == 0
        assert db_session.query(BeliefVersion).count() == 1

    def test_backfilled_history_is_addressable(self, db_session):
        """After a backfill, the next update opens a second version rather
        than orphaning the first."""
        set_memory_enabled(db_session, "u1", True)
        belief = _pre_ledger_belief(db_session)
        backfill_belief_ledger(db_session, "u1")
        decision = _update_to(db_session, belief, object_="berlin", key="lives_in:berlin")
        assert decision.committed
        chain = belief_history(db_session, "u1", (decision.detail or {}).get("belief_id"))
        assert [v["object"] for v in chain] == ["a", "berlin"]


def _ledger_config(config, **overrides):
    """The shipped config with the ledger bounds overridden."""
    ledger = config.metabolism.ledger.model_copy(update=overrides)
    return config.model_copy(
        update={"metabolism": config.metabolism.model_copy(update={"ledger": ledger})}
    )


class TestLedgerCapacity:
    def test_version_cap_prunes_the_oldest(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        belief = _pre_ledger_belief(db_session)
        current = belief
        for city in ("a", "b", "c", "d", "e"):
            decision = Publisher(db_session, _ledger_config(config, max_versions_per_identity=3)).publish(
                StateTransitionProposal(
                    transition=TRANSITION_CREATE if current is belief else TRANSITION_UPDATE,
                    user_id="u1",
                    target_belief_id=None if current is belief else current.id,
                    expected_revision=get_state_revision(db_session, "u1"),
                    payload={
                        "dimension": "fact",
                        "predicate": "lives_in", "object": city,
                        "key": f"lives_in:{city}", "claim_text": f"lives in {city}",
                        "confidence": 0.8, "cardinality": "single",
                    },
                )
            )
            assert decision.committed
            current = db_session.get(Belief, decision.detail["belief_id"])
        versions = belief_versions_for_identity(db_session, "u1", slot_key="lives_in")
        # Five values, cap of three: the two oldest are gone.
        assert [v.object for v in versions] == ["c", "d", "e"]

    def test_ttl_prunes_long_closed_versions(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        belief = _pre_ledger_belief(db_session)
        # A version the engine stopped believing in 2020.
        append_belief_version(db_session, belief, proposal_id="seed")
        old = belief_versions_for_identity(db_session, "u1", slot_key="lives_in")[0]
        old.epistemic_to = datetime(2020, 6, 1)
        db_session.flush()

        decision = Publisher(db_session, _ledger_config(config, version_ttl_days=30)).publish(
            StateTransitionProposal(
                transition=TRANSITION_UPDATE,
                user_id="u1",
                target_belief_id=belief.id,
                expected_revision=get_state_revision(db_session, "u1"),
                payload={
                    "predicate": "lives_in", "object": "b",
                    "key": "lives_in:b", "claim_text": "lives in b",
                    "confidence": 0.8, "cardinality": "single",
                },
            )
        )
        assert decision.committed

        versions = belief_versions_for_identity(db_session, "u1", slot_key="lives_in")
        # The 2020 version closed long ago (TTL 30 days) and was pruned;
        # the new one stays.
        assert [v.object for v in versions] == ["b"]

    def test_newest_version_is_never_pruned(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief = _pre_ledger_belief(db_session)
        append_belief_version(db_session, belief, proposal_id="seed", version_ttl_days=1)
        versions = belief_versions_for_identity(db_session, "u1", slot_key="lives_in")
        assert len(versions) == 1
