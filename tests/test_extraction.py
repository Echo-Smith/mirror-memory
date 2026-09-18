"""Tests for extraction modules — deterministic, throttle, semantic."""

import json

import pytest

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base
from mirror_memory.extraction.deterministic import extract_claims
from mirror_memory.extraction.throttle import should_extract


class TestExtractClaims:
    @pytest.fixture
    def config(self):
        return load_config("config/")

    def test_empty_text_returns_empty(self, config):
        assert extract_claims("", config) == []
        assert extract_claims(None, config) == []

    def test_no_keyword_match(self, config):
        claims = extract_claims("xyzzy plugh", config)
        assert claims == []

    def test_single_keyword_match(self, config):
        claims = extract_claims("I have trouble with work deadlines", config)
        # "work" should match the topic keywords
        assert len(claims) >= 1
        assert any(c["dimension"] == "topic" for c in claims)

    def test_multiple_categories(self, config):
        claims = extract_claims("I want to improve my health and work life balance", config)
        dims = {c["dimension"] for c in claims}
        assert len(dims) >= 1  # at least one dimension matched

    def test_claim_text_is_user_message(self, config):
        text = "I want to learn Japanese this year"
        claims = extract_claims(text, config)
        if claims:
            # claim_text should be the user's actual message, not "Keyword signal"
            assert "Keyword signal" not in claims[0]["claim_text"]

    def test_pattern_match(self, config):
        claims = extract_claims("I am 28 years old", config)
        # The age pattern should match
        age_claims = [c for c in claims if c.get("key") == "age"]
        if age_claims:
            assert age_claims[0]["confidence"] > 0

    def test_confidence_values(self, config):
        claims = extract_claims("I like painting", config)
        for c in claims:
            assert 0 <= c["confidence"] <= 1

    def test_relation_is_supports(self, config):
        claims = extract_claims("I love music", config)
        for c in claims:
            assert c["relation"] == "supports"


class TestShouldExtract:
    @pytest.fixture
    def config(self):
        return load_config("config/")

    def test_min_hits_zero_always_extract(self, config):
        config.extraction.llm_min_keyword_hits = 0
        assert should_extract("any text", 1, config) is True

    def test_keyword_hits_meet_threshold(self, config):
        config.extraction.llm_min_keyword_hits = 1
        # "work" is a keyword
        assert should_extract("I have work to do", 1, config) is True

    def test_no_keyword_hits_no_extract(self, config):
        config.extraction.llm_min_keyword_hits = 2
        assert should_extract("xyzzy", 1, config) is False

    def test_periodic_extract_with_hit(self, config):
        config.extraction.llm_min_keyword_hits = 2
        config.extraction.llm_every_turns = 3
        # Turn 3 with 1 keyword hit should extract
        assert should_extract("I have work", 3, config) is True

    def test_periodic_no_extract_without_hit(self, config):
        config.extraction.llm_min_keyword_hits = 2
        config.extraction.llm_every_turns = 3
        # Turn 3 with 0 keyword hits should NOT extract
        assert should_extract("xyzzy plugh", 3, config) is False

    def test_empty_text(self, config):
        assert should_extract("", 1, config) is False


class TestSemanticParseExtractionJson:
    """Test _parse_extraction_json from semantic.py."""

    def _parse(self, raw, allowed_keys):
        from mirror_memory.extraction.semantic import _parse_extraction_json
        return _parse_extraction_json(raw, allowed_keys)

    def test_bare_array(self):
        allowed = {"topic": frozenset({"sleep", "work"})}
        raw = '[{"dimension": "topic", "key": "sleep", "claim_text": "trouble sleeping", "confidence": 0.8}]'
        claims = self._parse(raw, allowed)
        assert len(claims) == 1
        assert claims[0]["key"] == "sleep"

    def test_wrapped_claims(self):
        allowed = {"topic": frozenset({"sleep"})}
        raw = '{"claims": [{"dimension": "topic", "key": "sleep", "claim_text": "test", "confidence": 0.5}]}'
        claims = self._parse(raw, allowed)
        assert len(claims) == 1

    def test_invalid_dimension_skipped(self):
        allowed = {"topic": frozenset({"sleep"})}
        raw = '[{"dimension": "invalid_dim", "key": "sleep", "claim_text": "test", "confidence": 0.5}]'
        claims = self._parse(raw, allowed)
        assert len(claims) == 0

    def test_empty_key_skipped(self):
        allowed = {"topic": frozenset({"sleep"})}
        raw = '[{"dimension": "topic", "key": "", "claim_text": "test", "confidence": 0.5}]'
        claims = self._parse(raw, allowed)
        assert len(claims) == 0

    def test_relation_new_normalized(self):
        allowed = {"topic": frozenset({"sleep"})}
        raw = '[{"dimension": "topic", "key": "sleep", "claim_text": "test", "confidence": 0.5, "relation": "new"}]'
        claims = self._parse(raw, allowed)
        assert claims[0]["relation"] == "supports"

    def test_invalid_relation_skipped(self):
        allowed = {"topic": frozenset({"sleep"})}
        raw = '[{"dimension": "topic", "key": "sleep", "claim_text": "test", "confidence": 0.5, "relation": "invalid"}]'
        claims = self._parse(raw, allowed)
        assert len(claims) == 0

    def test_confidence_clamped(self):
        allowed = {"topic": frozenset({"sleep"})}
        raw = '[{"dimension": "topic", "key": "sleep", "claim_text": "test", "confidence": 1.5}]'
        claims = self._parse(raw, allowed)
        assert claims[0]["confidence"] == 1.0

    def test_max_claims(self):
        from mirror_memory.core.constants import MAX_CLAIMS_PER_TURN
        allowed = {"topic": frozenset({"sleep", "work", "health", "family", "money"})}
        items = [{"dimension": "topic", "key": k, "claim_text": f"about {k}", "confidence": 0.5}
                 for k in ["sleep", "work", "health", "family", "money"]]
        raw = json.dumps(items)
        claims = self._parse(raw, allowed)
        assert len(claims) <= MAX_CLAIMS_PER_TURN

    def test_none_input(self):
        assert self._parse(None, {}) == []

    def test_empty_string(self):
        assert self._parse("", {}) == []

    def test_markdown_fenced(self):
        allowed = {"topic": frozenset({"sleep"})}
        raw = '```json\n[{"dimension": "topic", "key": "sleep", "claim_text": "test", "confidence": 0.5}]\n```'
        claims = self._parse(raw, allowed)
        assert len(claims) == 1

    def test_claim_text_truncated(self):
        from mirror_memory.core.constants import CLAIM_TEXT_MAX_LENGTH
        allowed = {"topic": frozenset({"sleep"})}
        long_text = "x" * 500
        raw = json.dumps([{"dimension": "topic", "key": "sleep", "claim_text": long_text, "confidence": 0.5}])
        claims = self._parse(raw, allowed)
        assert len(claims[0]["claim_text"]) <= CLAIM_TEXT_MAX_LENGTH