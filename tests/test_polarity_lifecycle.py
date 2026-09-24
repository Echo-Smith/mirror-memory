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


# ---------------------------------------------------------------------------
# P0-3-4: behavioural evidence does not upgrade into a preference
# ---------------------------------------------------------------------------


class TestBehaviouralEvidenceDoesNotUpgrade:
    """A behaviour is not an attitude.

    "I went hiking every weekend" is an event, not a preference.  Two guards
    keep them apart: the extractor's preference keywords are attitude verbs
    only, and behavioural predicates infer as neutral polarity so they can
    never close or merge with a preference row.
    """

    def test_behavioural_predicates_are_polarity_neutral(self):
        for predicate in ("went_to", "bought", "attended", "experienced",
                          "visited", "drank", "says", "has", "owns"):
            assert infer_polarity(predicate) == "neutral", predicate

    def test_attitude_words_in_a_behavioural_claim_do_not_flip_polarity(self):
        """The predicate decides polarity, not the claim text.

        "I went to a coffee shop I love" is still an event: its predicate is
        ``went_to``.  Reading polarity off the text would make every
        behavioural mention of a liked thing a preference statement.
        """
        assert infer_polarity("went_to") == "neutral"
        assert infer_lifecycle("went_to", "I went to a place I love") == ""

    def test_behavioural_and_attitude_claims_do_not_merge(self, session):
        """Different predicates never resolve to the same belief."""
        from mirror_memory.memory.atom import CandidateAtom
        from mirror_memory.memory.identity import resolve_identity

        set_memory_enabled(session, "u1", True)
        existing = [{
            "id": 1, "predicate": "went_to", "object": "hiking",
            "status": "active", "confidence": 0.7,
            "valid_from": None, "valid_to": None,
        }]
        result = resolve_identity(
            CandidateAtom(subject="user", predicate="likes", object="hiking"),
            existing,
            {"went_to": "event", "likes": "multi"},
            {"went_to": "episodic", "likes": "persistent"},
        )
        # A new belief, not a SUPPORT of the behavioural one.
        assert result.action == "CREATE"

    def test_k1_preference_keywords_are_attitude_verbs_only(self):
        """The config must not map behavioural nouns onto preference."""
        from mirror_memory.config.loader import load_config

        cfg = load_config("config/")
        preference_words = cfg.extraction.keywords.get("preference", [])
        behavioural = {"hiking", "coffee", "laptop", "museum", "workshop"}
        assert not (behavioural & {w.lower() for w in preference_words}), (
            "a behavioural noun mapped onto the preference dimension would "
            "turn every mention of it into a long-term preference"
        )


# ---------------------------------------------------------------------------
# P0-2-4: self-correction vs source conflict
# ---------------------------------------------------------------------------


class TestConflictKinds:
    """A retraction and a disagreement are different events.

    "Actually I don't like coffee" is the user changing their mind: the old
    state closes and the new one becomes current.  Two sources disagreeing is
    unresolved: confidence is attenuated and clarification is requested, and
    neither side is overwritten -- resolving it by whoever spoke last would
    silently pick a winner.
    """

    def test_self_correction_closes_the_old_state(self, session):
        from mirror_memory.core.repository import (
            CONFLICT_SELF_CORRECTION,
            record_claim,
        )

        set_memory_enabled(session, "u1", True)
        old, _ = record_claim(
            session, "u1", dimension="preference", key="likes_coffee",
            claim_text="User likes coffee", confidence=0.9,
            predicate="likes", object="coffee", polarity="positive",
        )
        _result, event = record_claim(
            session, "u1", dimension="preference", key="likes_coffee",
            claim_text="User does not like coffee", confidence=0.9,
            predicate="likes", object="coffee", polarity="positive",
            relation="contradicts", conflict_kind=CONFLICT_SELF_CORRECTION,
        )
        assert event == "self_corrected"
        assert old.status == "superseded"
        assert old.valid_to is not None
        # The retracted claim is not merely dimmed -- it is closed.
        assert old.confidence == 0.9

    def test_source_conflict_attenuates_and_flags(self, session):
        from mirror_memory.core.repository import (
            CONFLICT_SOURCE_CONFLICT,
            record_claim,
        )
        from mirror_memory.core.utils import safe_json

        set_memory_enabled(session, "u1", True)
        old, _ = record_claim(
            session, "u1", dimension="preference", key="likes_coffee",
            claim_text="User likes coffee", confidence=0.9,
            predicate="likes", object="coffee", polarity="positive",
        )
        _result, event = record_claim(
            session, "u1", dimension="preference", key="likes_coffee",
            claim_text="Another source says otherwise", confidence=0.9,
            predicate="likes", object="coffee", polarity="positive",
            relation="contradicts", conflict_kind=CONFLICT_SOURCE_CONFLICT,
        )
        assert event == "contradicted"
        # Still active -- neither side wins.
        assert old.status == "active"
        assert old.confidence < 0.9
        assert safe_json(old.value_json)["clarification_status"] == "needs_clarification"

    def test_default_is_source_conflict(self, session):
        """Backwards compatible: an unspecified kind is treated as unresolved."""
        from mirror_memory.core.repository import record_claim

        set_memory_enabled(session, "u1", True)
        old, _ = record_claim(
            session, "u1", dimension="topic", key="k",
            claim_text="User said X", confidence=0.9,
        )
        _result, event = record_claim(
            session, "u1", dimension="topic", key="k",
            claim_text="User said not-X", confidence=0.9,
            relation="contradicts",
        )
        assert event == "contradicted"
        assert old.status == "active"


# ---------------------------------------------------------------------------
# P1-6: raw predicate kept for audit
# ---------------------------------------------------------------------------


class TestRawPredicateAudit:
    def test_raw_predicate_stored_alongside_canonical(self, session):
        from mirror_memory.core.repository import record_claim

        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="works_at_globex",
            claim_text="User was hired by Globex", confidence=0.9,
            predicate="works_at", object="globex",
            raw_predicate="was_hired_by",
        )
        assert belief.predicate == "works_at"          # canonical
        assert belief.raw_predicate == "was_hired_by"  # as extracted

    def test_audit_reports_unmapped_predicates(self, session):
        from mirror_memory.core.repository import audit_predicate_coverage, record_claim

        set_memory_enabled(session, "u1", True)
        # One canonicalised (was_hired_by -> works_at), one the map has never
        # seen -- exactly the silent-split failure mode.
        record_claim(session, "u1", dimension="fact", key="k1",
                     claim_text="a", confidence=0.9,
                     predicate="works_at", object="globex",
                     raw_predicate="was_hired_by")
        record_claim(session, "u1", dimension="fact", key="k2",
                     claim_text="b", confidence=0.9,
                     predicate="transitioned_to", object="acme",
                     raw_predicate="transitioned_to")

        audit = audit_predicate_coverage(session, "u1")
        assert audit["beliefs_with_raw_predicate"] == 2
        assert "transitioned_to" in audit["unmapped_raw_predicates"]
        assert "was_hired_by" not in audit["unmapped_raw_predicates"]
        assert 0.0 < audit["coverage"] < 1.0

    def test_audit_full_coverage_when_all_mapped(self, session):
        from mirror_memory.core.repository import audit_predicate_coverage, record_claim

        set_memory_enabled(session, "u1", True)
        record_claim(session, "u1", dimension="fact", key="k",
                     claim_text="a", confidence=0.9,
                     predicate="works_at", object="globex",
                     raw_predicate="hired_by")
        audit = audit_predicate_coverage(session, "u1")
        assert audit["unmapped_raw_predicates"] == []
        assert audit["coverage"] == 1.0

    def test_audit_empty_user(self, session):
        from mirror_memory.core.repository import audit_predicate_coverage

        audit = audit_predicate_coverage(session, "nobody")
        assert audit["beliefs_with_raw_predicate"] == 0
        assert audit["coverage"] == 1.0


# ---------------------------------------------------------------------------
# P2-10: release gate
# ---------------------------------------------------------------------------


class TestReleaseGate:
    @staticmethod
    def _report(score, categories, tracks=None):
        from types import SimpleNamespace

        return SimpleNamespace(score=score, by_category=categories, tracks=tracks or {})

    def test_all_clear(self):
        from mirror_memory.bench.gate import check_release_gate

        result = check_release_gate(
            self._report(0.93, {"replacement": {"score": 0.9},
                                "round_trip": {"score": 0.95}},
                         {"recall": 0.95, "answer": 0.9}),
        )
        assert result.passed
        assert result.failures == []

    def test_overall_below_gate_fails(self):
        from mirror_memory.bench.gate import check_release_gate

        result = check_release_gate(self._report(0.71, {"replacement": {"score": 0.9}}))
        assert not result.passed
        assert any("state overall" in f for f in result.failures)

    def test_single_category_below_gate_fails(self):
        from mirror_memory.bench.gate import check_release_gate

        result = check_release_gate(
            self._report(0.95, {"replacement": {"score": 0.9},
                                "round_trip": {"score": 0.5}})
        )
        assert not result.passed
        assert any("round_trip" in f for f in result.failures)

    def test_recall_and_answer_gates(self):
        from mirror_memory.bench.gate import check_release_gate

        result = check_release_gate(
            self._report(0.95, {"replacement": {"score": 0.9}},
                         {"recall": 0.78, "answer": 0.6})
        )
        assert not result.passed
        assert any("recall" in f for f in result.failures)
        assert any("answer" in f for f in result.failures)

    def test_missing_manifest_fails(self, tmp_path):
        from mirror_memory.bench.gate import check_release_gate

        result = check_release_gate(
            self._report(0.95, {"replacement": {"score": 0.9}}),
            manifest_path=tmp_path / "absent.json",
        )
        assert not result.passed
        assert any("missing manifest" in f for f in result.failures)

    def test_present_manifest_passes(self, tmp_path):
        from mirror_memory.bench.gate import check_release_gate

        path = tmp_path / "present.manifest.json"
        path.write_text("{}")
        result = check_release_gate(
            self._report(0.95, {"replacement": {"score": 0.9}}),
            manifest_path=path,
        )
        assert result.passed

    def test_forced_production_gap(self):
        from mirror_memory.bench.gate import GATE_PRODUCTION_GAP, compare_forced_production

        forced = self._report(0.93, {"replacement": {"score": 0.95}})
        production = self._report(0.91, {"replacement": {"score": 0.93}})
        gap = compare_forced_production(forced, production)
        assert gap["gap"] <= GATE_PRODUCTION_GAP
        assert gap["gate"] is True
        assert gap["per_category_gap"]["replacement"] == 0.02

    def test_large_gap_fails(self):
        from mirror_memory.bench.gate import compare_forced_production

        forced = self._report(0.95, {"replacement": {"score": 0.95}})
        production = self._report(0.70, {"replacement": {"score": 0.7}})
        gap = compare_forced_production(forced, production)
        assert gap["gate"] is False
