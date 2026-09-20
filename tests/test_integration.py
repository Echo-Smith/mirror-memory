"""Integration tests — extract → canonicalize → resolve → DB → recall.

These tests verify the full pipeline works end-to-end, not just
individual components in isolation.
"""

import json

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
        likes_beliefs = [b for b in beliefs if b.predicate == "likes"]
        # BOTH must exist as separate beliefs
        objects = {b.object for b in likes_beliefs}
        assert "coffee" in objects
        assert "painting" in objects
        assert len(likes_beliefs) == 2


class TestSingleUpdate:
    """I live in Shanghai + I moved to Beijing → Beijing active, Shanghai superseded."""

    def test_single_update(self, engine):
        eng, llm = engine

        # First: lives_in Shanghai
        llm._responses["k2"] = '{"claims": [{"dimension": "fact", "key": "lives_in_shanghai", "predicate": "lives_in", "object": "shanghai", "claim_text": "User lives in Shanghai", "confidence": 0.9, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u2", session_id="s1", text="I live in Shanghai", turn_count=1)

        # Second: moved to Beijing (canonicalized to lives_in → UPDATE)
        llm._responses["k2"] = '{"claims": [{"dimension": "fact", "key": "lives_in_beijing", "predicate": "moved_to", "object": "beijing", "claim_text": "User moved to Beijing", "confidence": 0.9, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u2", session_id="s2", text="I moved to Beijing", turn_count=1)

        beliefs = eng.get_beliefs(user_id="u2")

        # Only Beijing should be active (Shanghai superseded)
        lives_in = [b for b in beliefs if b.predicate == "lives_in"]
        active_objects = [b.object for b in lives_in]
        assert active_objects == ["beijing"]


class TestSameObjectSupport:
    """I like coffee + I really love coffee → 1 belief, confidence boosted."""

    def test_support_boosts_confidence(self, engine):
        eng, llm = engine

        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User likes coffee", "confidence": 0.7, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u3", session_id="s1", text="I like coffee", turn_count=1)

        beliefs1 = eng.get_beliefs(user_id="u3")
        likes1 = [b for b in beliefs1 if b.predicate == "likes"]
        conf1 = likes1[0].confidence if likes1 else 0

        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User really loves coffee", "confidence": 0.9, "relation": "supports"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u3", session_id="s1", text="I really love coffee", turn_count=2)

        beliefs2 = eng.get_beliefs(user_id="u3")
        likes2 = [b for b in beliefs2 if b.predicate == "likes"]
        conf2 = likes2[0].confidence if likes2 else 0

        # Still 1 belief, confidence strictly increased
        assert len(likes2) == 1
        assert conf2 > conf1


class TestPredicateRetrieval:
    """Query 'what does the user like' should prioritize likes beliefs."""

    def test_predicate_query(self, engine):
        eng, llm = engine

        # Create a likes belief
        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User likes coffee", "confidence": 0.8, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u4", session_id="s1", text="I like coffee", turn_count=1)

        # Query about preferences
        ctx = eng.recall(user_id="u4", query="What does the user like?", language="en")
        assert ctx is not None
        assert "coffee" in ctx.lower()


class TestCityRoundTrip:
    """Shanghai → Beijing → Shanghai end-to-end."""

    def test_round_trip(self, engine):
        eng, llm = engine

        llm._responses["k2"] = '{"claims": [{"dimension": "fact", "key": "lives_in_shanghai", "predicate": "lives_in", "object": "shanghai", "claim_text": "User lives in Shanghai", "confidence": 0.9, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u6", session_id="s1", text="I live in Shanghai", turn_count=1)

        llm._responses["k2"] = '{"claims": [{"dimension": "fact", "key": "lives_in_beijing", "predicate": "moved_to", "object": "beijing", "claim_text": "User moved to Beijing", "confidence": 0.9, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u6", session_id="s2", text="I moved to Beijing", turn_count=1)

        lis3 = [
            {"dimension": "fact", "key": "lives_in_shanghai",
             "predicate": "lives_in", "object": "shanghai",
             "claim_text": "User moved back to Shanghai",
             "confidence": 0.9, "relation": "new"},
        ]
        llm._responses["k2"] = json.dumps({"claims": lis3, "subject": "user", "context_tags": []})
        eng.observe(user_id="u6", session_id="s3", text="I moved back to Shanghai", turn_count=1)

        beliefs = eng.get_beliefs(user_id="u6")
        lives_in = [b for b in beliefs if b.predicate == "lives_in"]
        assert len(lives_in) == 1
        assert lives_in[0].object == "shanghai"


class TestContradictionE2E:
    """Contradicting an existing belief does NOT strengthen it."""

    def test_contradiction_attenuates(self, engine):
        eng, llm = engine

        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "like_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User likes coffee", "confidence": 0.8, "relation": "new"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u7", session_id="s1", text="I like coffee", turn_count=1)

        # Get the K2-created belief's confidence
        likes_before = [b for b in eng.get_beliefs(user_id="u7") if b.key == "like_coffee"]
        conf_before = likes_before[0].confidence if likes_before else 0

        # Now say "I don't like coffee anymore" with relation=contradicts
        llm._responses["k2"] = '{"claims": [{"dimension": "preference", "key": "dislike_coffee", "predicate": "likes", "object": "coffee", "claim_text": "User no longer likes coffee", "confidence": 0.6, "relation": "contradicts"}], "subject": "user", "context_tags": []}'
        eng.observe(user_id="u7", session_id="s1", text="I don't like coffee anymore", turn_count=2)

        likes_after = [b for b in eng.get_beliefs(user_id="u7") if b.key == "like_coffee"]
        conf_after = likes_after[0].confidence if likes_after else 0
        # Confidence should strictly decrease from a contradiction
        assert conf_after < conf_before


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
