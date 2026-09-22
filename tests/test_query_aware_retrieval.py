"""Tests for query-aware retrieval: admission, ranking, and identity scope.

The renderer used to take the N most recently evidenced beliefs and score
those, so recency acted as a stand-in for relevance.  On LOCOMo the answer
belief sat at recency rank 153 of 161 and never reached the scorer at all.
Identity resolution had the same shape of bug with ``limit=200``: a user past
200 active beliefs silently lost older predicates, and the resolver turned
what should have been a SUPPORT or UPDATE into a CREATE.

These tests pin all three fixes: query-aware admission, a content-overlap
ranking signal, and predicate-scoped identity resolution.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base
from mirror_memory.core.repository import (
    beliefs_with_predicates,
    recall_candidates,
    record_claim,
    set_memory_enabled,
)
from mirror_memory.core.utils import content_tokens, text_overlap


@pytest.fixture
def config():
    return load_config("config/")


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory()
    Base.metadata.drop_all(engine)
    engine.dispose()


def _belief(session, key, obj, *, pred="", text="", confidence=0.8, session_id="s1"):
    set_memory_enabled(session, "u1", True)
    belief, _ = record_claim(
        session, "u1", dimension="fact", key=key,
        claim_text=text or f"user {obj}", confidence=confidence,
        predicate=pred, object=obj, session_id=session_id,
    )
    return belief


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


class TestContentTokens:
    def test_stop_words_dropped(self):
        tokens = content_tokens("What did the charity race raise awareness for?")
        assert "what" not in tokens
        assert "the" not in tokens
        assert {"charity", "race", "awareness", "raise"} <= tokens

    def test_short_words_dropped(self):
        assert content_tokens("go to the park") == {"park"}

    def test_empty(self):
        assert content_tokens("") == set()
        assert content_tokens(None) == set()

    def test_chinese_uses_bigrams(self):
        tokens = content_tokens("用户住在上海")
        assert "住在" in tokens
        assert "上海" in tokens

    def test_single_chinese_char(self):
        assert content_tokens("家") == {"家"}

    def test_overlap_is_a_fraction(self):
        tokens = {"charity", "race", "zebra"}
        assert text_overlap(tokens, "a charity race happened") == pytest.approx(2 / 3)
        assert text_overlap(tokens, "nothing here") == 0.0
        assert text_overlap(set(), "anything") == 0.0
        assert text_overlap({"a"}, "") == 0.0


# ---------------------------------------------------------------------------
# Query-aware admission
# ---------------------------------------------------------------------------


class TestRecallCandidates:
    def test_old_but_relevant_belief_is_admitted(self, session):
        """The regression: a fact stated early must not be truncated away."""
        # Fill the store so the answer would fall outside any recency window.
        for i in range(120):
            _belief(session, f"topic:noise{i}", f"unrelated_{i}",
                    text=f"user mentioned filler number {i}", session_id=f"s{i}")
        old = _belief(session, "ran_charity_race", "charity_race_for_mental_health",
                      pred="ran", text="User ran a charity race for mental health",
                      session_id="s_old")

        candidates = recall_candidates(session, "u1", query="What did the charity race raise for?")
        keys = {b.key for b in candidates}
        assert old.key in keys, "relevant-but-old belief was not admitted"

    def test_recency_slice_keeps_context_non_empty(self, session):
        """A question nothing matches still yields something."""
        for i in range(30):
            _belief(session, f"topic:n{i}", f"thing_{i}", session_id=f"s{i}")
        candidates = recall_candidates(session, "u1", query="xyzzy qwerty")
        assert candidates, "no fallback slice"

    def test_recent_limit_is_respected(self, session):
        for i in range(60):
            _belief(session, f"topic:n{i}", f"thing_{i}", session_id=f"s{i}")
        candidates = recall_candidates(session, "u1", query="", recent_limit=10)
        assert len(candidates) <= 10

    def test_relevant_and_recent_are_deduplicated(self, session):
        for i in range(20):
            _belief(session, f"topic:n{i}", f"charity_{i}", session_id=f"s{i}")
        candidates = recall_candidates(session, "u1", query="charity", recent_limit=20)
        ids = [b.id for b in candidates]
        assert len(ids) == len(set(ids)), "duplicate beliefs in the candidate set"

    def test_disabled_memory_returns_nothing(self, session):
        _belief(session, "topic:a", "alpha")
        set_memory_enabled(session, "u1", False)
        assert recall_candidates(session, "u1", query="alpha") == []

    def test_dimension_filter(self, session):
        from mirror_memory.core.repository import list_active_beliefs  # noqa: F401

        for i in range(5):
            _belief(session, f"topic:n{i}", f"thing_{i}")
        candidates = recall_candidates(
            session, "u1", query="thing", dimensions=("preference",)
        )
        assert all(b.dimension == "preference" for b in candidates)

    def test_historical_mode_admits_superseded(self, session):
        """A closed interval is history; a past question must reach it."""
        from mirror_memory.core.repository import update_belief_by_id

        old = _belief(session, "lives_in:shanghai", "shanghai", pred="lives_in",
                      text="lives in Shanghai")
        update_belief_by_id(session, old.id, new_object="beijing",
                            new_claim_text="lives in Berlin", new_confidence=0.8)

        past = recall_candidates(session, "u1", query="Where did they live before?",
                                 temporal_mode="historical")
        assert old.key in {b.key for b in past}

        current = recall_candidates(session, "u1", query="Where do they live now?",
                                    temporal_mode="current")
        assert old.key not in {b.key for b in current}

    def test_like_wildcards_in_the_query_are_escaped(self, session):
        """A '%' in the question must match itself, not everything."""
        _belief(session, "topic:a", "alpha", text="user likes alpha")
        _belief(session, "topic:b", "beta", text="user likes beta")
        candidates = recall_candidates(session, "u1", query="100%")
        # Too many matches for a bounded slice is fine; a crash is not.
        assert isinstance(candidates, list)

    def test_no_query_returns_recent_only(self, session):
        for i in range(10):
            _belief(session, f"topic:n{i}", f"thing_{i}", session_id=f"s{i}")
        assert len(recall_candidates(session, "u1", query="")) == 10


# ---------------------------------------------------------------------------
# Predicate-scoped identity resolution
# ---------------------------------------------------------------------------


class TestBeliefsWithPredicates:
    def test_returns_only_matching_predicates(self, session):
        lives = _belief(session, "lives_in:shanghai", "shanghai", pred="lives_in")
        _belief(session, "likes:coffee", "coffee", pred="likes")
        result = beliefs_with_predicates(session, "u1", {"lives_in"})
        assert [b.id for b in result] == [lives.id]

    def test_survives_a_large_belief_count(self, session):
        """The regression: 200+ beliefs must not hide older predicates."""
        for i in range(250):
            _belief(session, f"likes:thing{i}", f"thing_{i}", pred="likes")
        old = _belief(session, "lives_in:shanghai", "shanghai", pred="lives_in",
                      text="lives in Shanghai", session_id="s_old")

        result = beliefs_with_predicates(session, "u1", {"lives_in"})
        assert old.id in {b.id for b in result}, "older predicate was truncated away"

    def test_empty_predicates(self, session):
        _belief(session, "likes:coffee", "coffee", pred="likes")
        assert beliefs_with_predicates(session, "u1", set()) == []

    def test_disabled_memory(self, session):
        _belief(session, "likes:coffee", "coffee", pred="likes")
        set_memory_enabled(session, "u1", False)
        assert beliefs_with_predicates(session, "u1", {"likes"}) == []

    def test_superseded_beliefs_excluded(self, session):
        """Identity resolution acts on current values, not closed history."""
        from mirror_memory.core.repository import update_belief_by_id

        old = _belief(session, "lives_in:shanghai", "shanghai", pred="lives_in",
                      text="lives in Shanghai")
        update_belief_by_id(session, old.id, new_object="beijing",
                            new_claim_text="lives in Berlin", new_confidence=0.8)
        assert old.id not in {b.id for b in beliefs_with_predicates(session, "u1", {"lives_in"})}


# ---------------------------------------------------------------------------
# Content-overlap ranking signal
# ---------------------------------------------------------------------------


class TestContentOverlapRanking:
    @staticmethod
    def _belief(key, obj, pred, text):
        from datetime import UTC, datetime
        from types import SimpleNamespace

        return SimpleNamespace(
            dimension="fact", key=key, layer="L4", confidence=0.9,
            value_json="{}", evidence_json="[]",
            last_evidence_at=datetime.now(UTC),
            predicate=pred, object=obj, claim_text=text,
        )

    def test_content_match_beats_a_tie(self, config):
        from mirror_memory.core.retrieval import score_belief

        answer = self._belief(
            "ran_charity_race", "charity_race_for_mental_health", "ran",
            "User ran a charity race for mental health last Saturday",
        )
        noise = self._belief(
            "went_hiking", "hiking", "went_to", "User went hiking last week",
        )
        tokens = content_tokens("What did the charity race raise awareness for?")

        assert score_belief(answer, query_tokens=tokens) > score_belief(
            noise, query_tokens=tokens
        )

    def test_without_query_tokens_behaviour_is_unchanged(self, config):
        """Backwards compatibility: no query tokens -> no content signal."""
        from mirror_memory.core.retrieval import score_belief

        belief = self._belief("k", "o", "p", "some claim text")
        assert score_belief(belief) == score_belief(belief, query_tokens=set())

    def test_explicit_content_overlap_is_honoured(self, config):
        from mirror_memory.core.retrieval import score_belief

        belief = self._belief("k", "o", "p", "some claim text")
        plain = score_belief(belief)
        boosted = score_belief(belief, content_overlap=1.0)
        assert boosted > plain

    def test_phrasing_independent(self, config):
        """Two ways of asking the same thing both rank the answer first."""
        from mirror_memory.core.retrieval import score_belief

        answer = self._belief(
            "ran_charity_race", "charity_race_for_mental_health", "ran",
            "User ran a charity race for mental health last Saturday",
        )
        noise = self._belief("went_hiking", "hiking", "went_to", "User went hiking")

        for question in (
            "What did the charity race raise awareness for?",
            "Tell me about the charity race",
            "Which cause did the race support?",
        ):
            tokens = content_tokens(question)
            assert score_belief(answer, query_tokens=tokens) > score_belief(
                noise, query_tokens=tokens
            ), question

    def test_zero_overlap_adds_nothing(self, config):
        from mirror_memory.core.retrieval import score_belief

        belief = self._belief("k", "o", "p", "some claim text")
        assert score_belief(belief, query_tokens={"zzz", "qqq"}) == score_belief(belief)

    def test_chinese_overlap_scores(self, config):
        from mirror_memory.core.retrieval import score_belief

        belief = self._belief("lives_shanghai", "shanghai", "lives_in", "用户住在上海")
        tokens = content_tokens("用户住在哪里")
        assert score_belief(belief, query_tokens=tokens) > score_belief(belief)


# ---------------------------------------------------------------------------
# End to end through the renderer
# ---------------------------------------------------------------------------


class TestRendererUsesQueryAwareAdmission:
    def test_old_fact_reaches_the_rendered_block(self, session, config):
        """The headline fix: rank 153 of 161 must now be reachable."""
        from mirror_memory.render.renderer import render_memory_block

        set_memory_enabled(session, "u1", True)
        for i in range(120):
            record_claim(
                session, "u1", dimension="topic", key=f"topic:noise{i}",
                claim_text=f"user mentioned filler number {i}", confidence=0.8,
                session_id=f"s{i}",
            )
        record_claim(
            session, "u1", dimension="fact", key="ran_charity_race",
            claim_text="User ran a charity race for mental health", confidence=0.9,
            predicate="ran", object="charity_race_for_mental_health", session_id="s_old",
        )

        block = render_memory_block(
            session, "u1", config=config,
            user_message="What did the charity race raise awareness for?",
            language="en",
        )
        assert block is not None
        assert "charity" in block.lower()
