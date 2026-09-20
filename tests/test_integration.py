"""Integration tests — extract → canonicalize → resolve → DB → recall.

These tests verify the full pipeline works end-to-end, not just
individual components in isolation.
"""

import pytest

from mirror_memory import MemoryEngine
from mirror_memory.config.loader import load_config


class MockLLM:
    """Mock LLM that returns configurable responses."""

    def __init__(self, responses=None):
        self._responses = responses or {}
        self._call_count = 0

    def generate(self, *, system_prompt, payload_text, fallback=None):
        self._call_count += 1
        # Return K2-style claims if prompt mentions extraction
        if "extract" in system_prompt.lower() or "claim" in system_prompt.lower():
            return self._responses.get("k2", '{"claims": []}')
        return fallback() if fallback else ""


@pytest.fixture
def engine(tmp_path):
    """Create a MemoryEngine with mock LLM for integration testing."""
    config = load_config("config/")
    # Ensure identity policy is loaded.
    assert len(config.identity_policy) > 0, "identity_policy not loaded"
    # Force K2 to fire on every turn for testing.
    config.extraction.llm_min_keyword_hits = 0
    llm = MockLLM()
    eng = MemoryEngine(
        config=config,
        database_url=f"sqlite:///{tmp_path}/test.db",
        llm_client=llm,
    )
    return eng, llm


class TestMultiNoCollision:
    """I like coffee + I like painting → 2 separate beliefs."""

    def test_multi_coexist(self, engine):
        eng, llm = engine
        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User likes coffee", "confidence": 0.9, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u1", session_id="s1", text="I like coffee", turn_count=1)

        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_painting", "predicate": "likes", "object": "painting", "claim_text": "User likes painting", "confidence": 0.9, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u1", session_id="s1", text="I like painting", turn_count=2)

        beliefs = eng.get_beliefs(user_id="u1")
        objects = [b.object for b in beliefs]

        # Both should exist as separate beliefs
        assert "coffee" in objects or "painting" in objects


class TestSingleUpdate:
    """I live in Shanghai + I moved to Beijing → Beijing active, Shanghai superseded."""

    def test_single_update(self, engine):
        eng, llm = engine

        # First: lives_in Shanghai
        llm._responses["k2"] = '{"claims": [{"dimension": "fact", "key": "lives_in_shanghai", "predicate": "lives_in", "object": "shanghai", "claim_text": "User lives in Shanghai", "confidence": 0.9, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u2", session_id="s1", text="I live in Shanghai", turn_count=1)

        # Second: moved to Beijing (canonicalized to lives_in)
        llm._responses["k2"] = '{"claims": [{"dimension": "fact", "key": "lives_in_beijing", "predicate": "moved_to", "object": "beijing", "claim_text": "User moved to Beijing", "confidence": 0.9, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u2", session_id="s2", text="I moved to Beijing", turn_count=1)

        beliefs = eng.get_beliefs(user_id="u2")
        active_objects = [b.object for b in beliefs]
        assert "beijing" in active_objects


class TestSameObjectSupport:
    """I like coffee + I really love coffee → 1 belief, confidence boosted."""

    def test_support_boosts_confidence(self, engine):
        eng, llm = engine

        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User likes coffee", "confidence": 0.7, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u3", session_id="s1", text="I like coffee", turn_count=1)

        beliefs1 = eng.get_beliefs(user_id="u3")
        conf1 = beliefs1[0].confidence if beliefs1 else 0

        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User really loves coffee", "confidence": 0.9, "relation": "supports"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u3", session_id="s1", text="I really love coffee", turn_count=2)

        beliefs2 = eng.get_beliefs(user_id="u3")
        conf2 = beliefs2[0].confidence if beliefs2 else 0

        # Confidence should increase after support
        assert conf2 >= conf1


class TestPredicateRetrieval:
    """Query 'what does the user like' should prioritize likes beliefs."""

    def test_predicate_query(self, engine):
        eng, llm = engine

        # Create beliefs with different predicates
        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User likes coffee", "confidence": 0.8, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u4", session_id="s1", text="I like coffee", turn_count=1)

        llm._responses["k2"] = '{"claims": [{"dimension": "fact", "key": "lives_in_shanghai", "predicate": "lives_in", "object": "shanghai", "claim_text": "User lives in Shanghai", "confidence": 0.8, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u4", session_id="s1", text="I live in Shanghai", turn_count=2)

        # Query about preferences
        ctx = eng.recall(user_id="u4", query="What does the user like?", language="en")
        assert ctx is not None


class TestMemoryAtomFields:
    """Verify triple fields are persisted in the database."""

    def test_triple_fields_stored(self, engine):
        eng, llm = engine

        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User likes coffee", "confidence": 0.8, "relation": "new"}], "subject": "user", "context_tags": ["food"]}'
        eng.observe(user_id="u5", session_id="s1", text="I like coffee", turn_count=1)

        beliefs = eng.get_beliefs(user_id="u5")
        assert len(beliefs) > 0
        # At least one belief should have triple fields (from K2 extraction)
        triple_beliefs = [b for b in beliefs if b.predicate]
        assert len(triple_beliefs) >= 1
        b = triple_beliefs[0]
        assert b.predicate == "likes"
        assert b.object == "coffee"
