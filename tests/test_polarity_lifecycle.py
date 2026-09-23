"""Tests for structured polarity and goal lifecycle (P0-2 / P0-3).

The read-side withdrawal filter was a word list: it caught "avoided" and
"gave up" and missed "coffee is off the table".  These tests pin the
structured replacement -- polarity on the belief row, lifecycle on goals,
and close-and-replace on user correction.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import correct_belief, record_claim, set_memory_enabled
from mirror_memory.memory.polarity import (
    GOAL_ACTIVE,
    GOAL_CANCELLED,
    GOAL_PAUSED,
    GOAL_RESUMED,
    POLARITY_NEGATIVE,
    POLARITY_POSITIVE,
    infer_lifecycle,
    infer_polarity,
    is_goal_predicate,
    opposite_polarity,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory()
    Base.metadata.drop_all(engine)
    engine.dispose()


# ---------------------------------------------------------------------------
# Polarity inference
# ---------------------------------------------------------------------------


class TestPolarityInference:
    @pytest.mark.parametrize("predicate,expected", [
        ("likes", POLARITY_POSITIVE),
        ("loves", POLARITY_POSITIVE),
        ("enjoys", POLARITY_POSITIVE),
        ("wants_to", POLARITY_POSITIVE),
        ("dislikes", POLARITY_NEGATIVE),
        ("hates", POLARITY_NEGATIVE),
        ("avoids", POLARITY_NEGATIVE),
        ("rejects", POLARITY_NEGATIVE),
        # Non-attitudinal predicates carry no polarity.
        ("lives_in", "neutral"),
        ("works_at", "neutral"),
        ("went_to", "neutral"),
        ("", "neutral"),
    ])
    def test_inference(self, predicate, expected):
        assert infer_polarity(predicate) == expected

    def test_opposite(self):
        assert opposite_polarity(POLARITY_POSITIVE) == POLARITY_NEGATIVE
        assert opposite_polarity(POLARITY_NEGATIVE) == POLARITY_POSITIVE
        assert opposite_polarity("neutral") == ""

    def test_case_and_whitespace_insensitive(self):
        assert infer_polarity("  Likes ") == POLARITY_POSITIVE


# ---------------------------------------------------------------------------
# Goal lifecycle
# ---------------------------------------------------------------------------


class TestGoalLifecycle:
    def test_plain_statement_is_active(self):
        assert infer_lifecycle("wants_to", "I want to learn Japanese") == GOAL_ACTIVE

    @pytest.mark.parametrize("text,expected", [
        ("I gave up on that idea", GOAL_CANCELLED),
        ("I abandoned the plan", GOAL_CANCELLED),
        ("I decided against it", GOAL_CANCELLED),
        ("I put it on hold", GOAL_PAUSED),
        ("I shelved the idea", GOAL_PAUSED),
        ("I started training again", GOAL_RESUMED),
        ("I picked it up again", GOAL_RESUMED),
    ])
    def test_transitions(self, text, expected):
        assert infer_lifecycle("wants_to", text) == expected

    def test_resume_beats_cancel_when_both_present(self):
        # "I gave up, but now I started again" -- the resumption is the state.
        assert infer_lifecycle(
            "wants_to", "I gave up on it, then I started training again"
        ) == GOAL_RESUMED

    def test_prior_cancelled_state_means_reaffirmation_is_resume(self):
        assert infer_lifecycle(
            "wants_to", "I want to run a marathon", prior_state=GOAL_CANCELLED
        ) == GOAL_RESUMED

    def test_non_goal_predicate_has_no_lifecycle(self):
        assert infer_lifecycle("likes", "I gave up on coffee") == ""
        assert not is_goal_predicate("likes")
        assert is_goal_predicate("wants_to")
        assert is_goal_predicate("plans_to")


# ---------------------------------------------------------------------------
# correct_belief closes and replaces
# ---------------------------------------------------------------------------


class TestCorrectClosesAndReplaces:
    def test_old_row_survives_as_history(self, session):
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="age",
            claim_text="User is 28 years old", confidence=0.9,
            predicate="age", object="28", cardinality="single",
        )
        corrected = correct_belief(
            session, "u1", belief.id,
            new_claim_text="User is actually 30 years old", new_object="30",
        )

        # A new row; the old one is closed, not overwritten.
        assert corrected.id != belief.id
        assert corrected.claim_text == "User is actually 30 years old"
        assert corrected.source == "user_corrected"
        assert corrected.status == "active"

        assert belief.status == "superseded"
        assert belief.claim_text == "User is 28 years old"
        assert belief.object == "28"
        assert belief.superseded_by == corrected.id
        assert belief.valid_to is not None

    def test_history_still_queryable(self, session):
        """The point of close-and-replace: the old value is a belief row, not
        just an event-log entry."""
        from mirror_memory.core.repository import list_active_beliefs

        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="age",
            claim_text="User is 28 years old", confidence=0.9,
            predicate="age", object="28", cardinality="single",
        )
        correct_belief(
            session, "u1", belief.id,
            new_claim_text="User is 30", new_object="30",
        )
        # Current query sees only the correction.
        current = list_active_beliefs(session, "u1", temporal_mode="current")
        assert [b.object for b in current] == ["30"]
        # Historical query still reaches the original.
        history = list_active_beliefs(session, "u1", temporal_mode="historical")
        assert "28" in {b.object for b in history}

    def test_no_unique_collision(self, session):
        """(user_id, key) is unique regardless of status."""
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="age",
            claim_text="User is 28", confidence=0.9,
            predicate="age", object="28", cardinality="single",
        )
        corrected = correct_belief(
            session, "u1", belief.id, new_claim_text="User is 30", new_object="30",
        )
        assert corrected.key != belief.key
        assert session.query(Belief).filter_by(user_id="u1").count() == 2

    def test_correction_preserves_polarity(self, session):
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="preference", key="likes_coffee",
            claim_text="User likes coffee", confidence=0.8,
            predicate="likes", object="coffee", polarity="positive",
        )
        corrected = correct_belief(
            session, "u1", belief.id,
            new_claim_text="User likes strong coffee", new_object="coffee",
        )
        assert corrected.polarity == "positive"


# ---------------------------------------------------------------------------
# Polarity stored on the belief row
# ---------------------------------------------------------------------------


class TestPolarityPersisted:
    def test_record_claim_stores_polarity(self, session):
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="preference", key="likes_coffee",
            claim_text="User likes coffee", confidence=0.8,
            predicate="likes", object="coffee", polarity="positive",
        )
        assert belief.polarity == "positive"

    def test_default_polarity_is_neutral(self, session):
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="k",
            claim_text="something", confidence=0.5,
        )
        assert belief.polarity == "neutral"

    def test_lifecycle_state_stored(self, session):
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="goal", key="wants_japanese",
            claim_text="User wants to learn Japanese", confidence=0.7,
            predicate="wants_to", object="japanese",
            lifecycle_state="active",
        )
        assert belief.lifecycle_state == "active"


# ---------------------------------------------------------------------------
# Polarity propagates through the update path
# ---------------------------------------------------------------------------


class TestPolarityPropagation:
    def test_update_preserves_polarity(self, session):
        """The replacement row must carry the new claim's polarity, not the
        schema default -- otherwise a negative update reads as neutral and
        never closes the positive row it contradicts."""
        from mirror_memory.core.repository import update_belief_by_id

        set_memory_enabled(session, "u1", True)
        old, _ = record_claim(
            session, "u1", dimension="preference", key="likes_coffee",
            claim_text="User likes coffee", confidence=0.8,
            predicate="likes", object="coffee", polarity="positive",
        )
        _old, new = update_belief_by_id(
            session, old.id,
            new_predicate="dislikes", new_object="coffee",
            new_claim_text="User dislikes coffee", new_confidence=0.7,
            polarity="negative",
        )
        assert new.polarity == "negative"
        assert new.status == "active"
        assert old.status == "superseded"

    def test_update_inherits_polarity_when_unspecified(self, session):
        from mirror_memory.core.repository import update_belief_by_id

        set_memory_enabled(session, "u1", True)
        old, _ = record_claim(
            session, "u1", dimension="fact", key="lives_in:shanghai",
            claim_text="lives in Shanghai", confidence=0.8,
            predicate="lives_in", object="shanghai", polarity="neutral",
        )
        _old, new = update_belief_by_id(
            session, old.id, new_object="beijing",
            new_claim_text="lives in Berlin", new_confidence=0.8,
        )
        assert new.polarity == "neutral"

    def test_revival_preserves_polarity(self, session):
        """A revived historical row keeps the polarity it was stored with.

        A -> B -> A: the third turn targets the superseded A row's key and
        revives it, so the polarity stored on that row must survive.
        """
        from mirror_memory.core.repository import update_belief_by_id

        set_memory_enabled(session, "u1", True)
        a, _ = record_claim(
            session, "u1", dimension="preference", key="likes_coffee",
            claim_text="User likes coffee", confidence=0.8,
            predicate="likes", object="coffee", polarity="positive",
        )
        a_key = a.key
        # A -> B
        _old, b = update_belief_by_id(
            session, a.id, new_object="tea",
            new_claim_text="User likes tea", new_confidence=0.8,
        )
        # B -> A: name A's key so the revival branch finds and reopens it.
        update_belief_by_id(
            session, b.id, new_key=a_key, new_object="coffee",
            new_claim_text="User likes coffee again", new_confidence=0.8,
        )
        revived = session.get(Belief, a.id)
        assert revived.status == "active"
        assert revived.polarity == "positive"
        assert session.get(Belief, b.id).status == "superseded"
