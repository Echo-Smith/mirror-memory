"""Tests for the extraction → storage loss.

The funnel measured 31 facts reaching extraction but only 16 answers existing
in any stored belief.  Two defects accounted for it, and both are pinned here.

1. **Assembler truncation** — `assemble_claims` sorted by dimension *scarcity*
   as its primary key, so a fact in an already-populated dimension lost to
   low-confidence filler in an empty one.  The answer claim was dropped
   precisely because its dimension was well covered.

2. **claim_text overwrite** — SUPPORT replaced the stored claim_text with
   "the longest seen so far".  A later, longer sentence about a different
   aspect of the same belief key therefore erased the wording that answered
   the question.  Both `record_claim` and `support_belief_by_id` had a copy.
"""

import pytest

from mirror_memory.config.loader import load_config
from mirror_memory.core.repository import (
    _is_elaboration,
    record_claim,
    set_memory_enabled,
    support_belief_by_id,
)
from mirror_memory.extraction.assembler import assemble_claims


@pytest.fixture
def config():
    return load_config("config/")


def _claim(key, dim="fact", conf=0.9, triple=True):
    value = {"predicate": "ran", "object": key} if triple else {"via": "pattern"}
    return {
        "dimension": dim, "key": key, "claim_text": f"user {key}",
        "confidence": conf, "relation": "supports", "source": "extracted",
        "value": value,
    }


# ---------------------------------------------------------------------------
# 1. Assembler ordering
# ---------------------------------------------------------------------------


class TestAssemblerOrdering:
    def test_quality_beats_scarcity(self, config):
        """A structured high-confidence claim survives a populated dimension."""
        answer = _claim("ran_charity_race", dim="fact", conf=0.5, triple=False)
        filler = [_claim(f"goal_{i}", dim="goal", conf=0.9) for i in range(3)]
        # The answer's dimension is well covered; the filler's is empty.
        counts = {"fact": 5, "goal": 0}

        kept = assemble_claims([answer] + filler, config, active_dimension_counts=counts)
        keys = [c["key"] for c in kept]
        assert "ran_charity_race" in keys, "answer claim dropped for being in a populated dimension"

    def test_scarcity_still_breaks_ties(self, config):
        """Breadth-seeking survives between claims of equal quality."""
        a = _claim("likes_coffee", dim="preference", conf=0.9)
        b = _claim("likes_tea", dim="preference", conf=0.9)
        kept = assemble_claims([a, b], config, active_dimension_counts={
            "preference": 5, "topic": 0,
        })
        assert [c["key"] for c in kept] == ["likes_coffee", "likes_tea"]

    def test_higher_confidence_wins(self, config):
        strong = _claim("strong_claim", conf=0.95)
        weak = _claim("weak_claim", conf=0.3)
        kept = assemble_claims([weak, strong], config)
        assert kept[0]["key"] == "strong_claim"

    def test_triple_bearing_claim_wins_over_plain(self, config):
        structured = _claim("structured", conf=0.5, triple=True)
        plain = _claim("plain_text", conf=0.6, triple=False)
        kept = assemble_claims([plain, structured], config)
        assert kept[0]["key"] == "structured"

    def test_cap_still_applies(self, config):
        claims = [_claim(f"c{i}", conf=0.9) for i in range(10)]
        kept = assemble_claims(claims, config)
        assert len(kept) == config.extraction.max_claims_per_turn

    def test_empty_claims(self, config):
        assert assemble_claims([], config) == []


# ---------------------------------------------------------------------------
# 2. claim_text elaboration rule
# ---------------------------------------------------------------------------


class TestElaborationRule:
    def test_generic_placeholder_accepts_longer(self):
        """The documented behaviour: 'short' -> a fuller description."""
        assert _is_elaboration(
            "much longer and more detailed description", "short"
        )

    def test_same_fact_more_detail_accepted(self):
        assert _is_elaboration(
            "User ran a charity race for mental health last Saturday",
            "User ran a charity race",
        )

    def test_different_aspect_refused(self):
        """The regression: a longer text about something else must not replace."""
        assert not _is_elaboration(
            "User went for a long walk in the park with friends",
            "User ran a charity race",
        )

    def test_shorter_text_never_replaces(self):
        assert not _is_elaboration("short", "a much longer existing claim")

    def test_empty_old_accepts(self):
        assert _is_elaboration("anything", "")
        assert _is_elaboration("anything", None)

    def test_empty_new_refused(self):
        assert not _is_elaboration("", "existing text")

    def test_generic_word_overlap_is_not_enough(self):
        """"user" is in nearly every claim; it must not license a replacement."""
        assert not _is_elaboration(
            "user mentioned something entirely different about painting",
            "user ran a charity race for mental health",
        )


class TestClaimTextPreservedOnSupport:
    @pytest.fixture
    def session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from mirror_memory.core.models import Base

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)
        yield factory()
        Base.metadata.drop_all(engine)
        engine.dispose()

    def test_record_claim_keeps_answer_bearing_text(self, session):
        set_memory_enabled(session, "u1", True)
        record_claim(
            session, "u1", dimension="fact", key="race",
            claim_text="User ran a charity race for mental health",
            confidence=0.5, session_id="s1",
        )
        belief, event = record_claim(
            session, "u1", dimension="fact", key="race",
            claim_text="User went for a long walk in the park with friends "
                       "and family last sunday afternoon",
            confidence=0.5, session_id="s2", relation="supports",
        )
        assert event == "supported"
        assert "charity" in belief.claim_text.lower()

    def test_support_belief_by_id_keeps_answer_bearing_text(self, session):
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="race",
            claim_text="User ran a charity race for mental health",
            confidence=0.5, session_id="s1",
        )
        support_belief_by_id(
            session, belief.id,
            claim_text="User went for a long walk in the park with friends "
                       "and family last sunday afternoon",
            evidence_message_ids=[],
        )
        assert "charity" in belief.claim_text.lower()

    def test_genuine_elaboration_still_updates_both_paths(self, session):
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="race",
            claim_text="User ran a charity race",
            confidence=0.5, session_id="s1",
        )
        fuller = "User ran a charity race for mental health awareness last Saturday"
        record_claim(
            session, "u1", dimension="fact", key="race",
            claim_text=fuller, confidence=0.5, session_id="s2", relation="supports",
        )
        assert belief.claim_text == fuller

        support_belief_by_id(
            session, belief.id,
            claim_text="User ran a charity race for mental health awareness "
                       "last Saturday morning",
            evidence_message_ids=[],
        )
        assert "Saturday morning" in belief.claim_text
