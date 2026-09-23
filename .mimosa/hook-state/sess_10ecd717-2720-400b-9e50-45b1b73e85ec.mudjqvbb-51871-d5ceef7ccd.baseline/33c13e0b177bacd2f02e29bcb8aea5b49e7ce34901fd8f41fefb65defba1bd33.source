"""Tests for core/utils.py — safe_json, parse_llm_json, truncate_id."""

import json
from mirror_memory.core.utils import safe_json, parse_llm_json, truncate_id


class TestSafeJson:
    def test_none_returns_fallback(self):
        assert safe_json(None) == {}

    def test_none_returns_custom_fallback(self):
        assert safe_json(None, {"default": True}) == {"default": True}

    def test_empty_string_returns_fallback(self):
        assert safe_json("") == {}

    def test_valid_dict(self):
        assert safe_json('{"a": 1}') == {"a": 1}

    def test_valid_list_returns_fallback(self):
        """Non-dict JSON (like a list) should return fallback."""
        assert safe_json("[1, 2]") == {}

    def test_malformed_json_returns_fallback(self):
        assert safe_json("{invalid") == {}

    def test_json_number_returns_fallback(self):
        assert safe_json("42") == {}

    def test_json_string_returns_fallback(self):
        assert safe_json('"hello"') == {}


class TestParseLlmJson:
    def test_none_returns_none(self):
        assert parse_llm_json(None) is None

    def test_empty_returns_none(self):
        assert parse_llm_json("") is None

    def test_bare_object(self):
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_bare_array_with_expect_array(self):
        result = parse_llm_json('[{"a": 1}]', expect_array=True)
        assert result == [{"a": 1}]

    def test_bare_array_without_expect_array(self):
        """Bare array without expect_array: finds inner object braces."""
        result = parse_llm_json('[{"a": 1}]', expect_array=False)
        # find("{") hits the inner {, rfind("}") hits the inner }
        assert result == {"a": 1}

    def test_markdown_fenced_object(self):
        raw = '```json\n{"key": "value"}\n```'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_markdown_fenced_without_json_prefix(self):
        raw = '```\n{"key": "value"}\n```'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_object_with_surrounding_text(self):
        raw = 'Here is the result: {"key": "value"} done.'
        result = parse_llm_json(raw)
        assert result == {"key": "value"}

    def test_no_json_found(self):
        assert parse_llm_json("no json here") is None

    def test_malformed_json(self):
        assert parse_llm_json("{invalid json}") is None


class TestTruncateId:
    def test_normal_string(self):
        assert truncate_id("abcdefghijklmnop", 8) == "abcdefgh"

    def test_short_string(self):
        assert truncate_id("abc", 8) == "abc"

    def test_none_returns_question_mark(self):
        assert truncate_id(None) == "?"

    def test_empty_returns_question_mark(self):
        assert truncate_id("") == "?"

    def test_default_length_is_8(self):
        assert truncate_id("abcdefghijklmnop") == "abcdefgh"
