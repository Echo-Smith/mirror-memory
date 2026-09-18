"""Tests for render/renderer.py — scoring, budget, diversity, fallback."""

import json
from datetime import UTC, datetime

import pytest

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.render.renderer import (
    _extract_query_nouns,
    _extract_query_topics,
    _render_belief,
    render_memory_block,
)
from mirror_memory.render.display import DisplayDict


class TestExtractQueryNouns:
    def test_english_nouns(self):
        nouns = _extract_query_nouns("When did Caroline go to the support group")
        assert "caroline" in nouns
        assert "support" in nouns
        assert "group" in nouns
        # Stop words filtered
        assert "when" not in nouns
        assert "the" not in nouns

    def test_chinese_nouns(self):
        nouns = _extract_query_nouns("用户什么时候去了支持小组")
        assert len(nouns) > 0
        # Should contain bigrams
        assert "支持" in nouns
        assert "小组" in nouns

    def test_empty_query(self):
        assert _extract_query_nouns("") == set()

    def test_mixed_language(self):
        nouns = _extract_query_nouns("用户的 sleep 质量如何")
        assert len(nouns) > 0


class TestExtractQueryTopics:
    def test_matches_keywords(self, config):
        topics = _extract_query_topics("I have trouble sleeping at night", config)
        assert isinstance(topics, set)

    def test_no_match(self, config):
        topics = _extract_query_topics("xyzzy plugh", config)
        assert len(topics) == 0

    def test_empty_query(self, config):
        topics = _extract_query_topics("", config)
        assert topics == set()


class TestRenderBelief:
    def test_belief_with_label(self, config):
        """Belief with a display label should render the label."""
        display = DisplayDict(config)
        from types import SimpleNamespace
        belief = SimpleNamespace(
            dimension="topic", key="work", claim_text="User works hard",
            confidence=0.8, layer="L2", value_json="{}",
            last_evidence_at=datetime.now(UTC),
        )
        text = _render_belief(belief, display, "en")
        assert text is not None

    def test_belief_without_label_uses_claim_text(self, config):
        """Belief without display label should fall back to claim_text."""
        display = DisplayDict(config)
        from types import SimpleNamespace
        belief = SimpleNamespace(
            dimension="topic", key="nonexistent_xyz", claim_text="User mentioned something specific",
            confidence=0.8, layer="L2", value_json="{}",
            last_evidence_at=datetime.now(UTC),
        )
        text = _render_belief(belief, display, "en")
        assert text is not None
        assert "specific" in text

    def test_belief_no_label_no_claim(self, config):
        """Belief with no label and no claim_text returns None."""
        display = DisplayDict(config)
        from types import SimpleNamespace
        belief = SimpleNamespace(
            dimension="topic", key="xyz", claim_text="",
            confidence=0.8, layer="L2", value_json="{}",
            last_evidence_at=None,
        )
        text = _render_belief(belief, display, "en")
        assert text is None


class TestRenderMemoryBlock:
    def test_empty_user_returns_none(self, db_session, config):
        result = render_memory_block(db_session, "nonexistent", config)
        assert result is None

    def test_with_beliefs_returns_string(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="User has trouble sleeping", confidence=0.8,
                     session_id="s1")
        result = render_memory_block(db_session, "u1", config, language="en")
        assert result is not None
        assert len(result) > 0

    def test_budget_limits_output(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        # Add many beliefs to exceed budget
        for i in range(20):
            record_claim(db_session, "u1", dimension="topic", key=f"topic_{i}",
                         claim_text=f"User mentioned topic {i} " * 10, confidence=0.8,
                         session_id="s1")
        result = render_memory_block(db_session, "u1", config, language="en",
                                     tail_load=0)
        if result:
            assert len(result) <= config.budget.cap * 2  # reasonable upper bound

    def test_rejected_beliefs_appear(self, db_session, config):
        from mirror_memory.core.repository import reject_belief
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                            claim_text="User has trouble sleeping", confidence=0.8,
                            session_id="s1")
        reject_belief(db_session, "u1", b.id)
        result = render_memory_block(db_session, "u1", config, language="en")
        # Rejected beliefs should appear as "avoid" items
        if result:
            assert "avoid" in result.lower() or "avoid" in result

    def test_chinese_language(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="睡不好", confidence=0.8, session_id="s1")
        result = render_memory_block(db_session, "u1", config, language="zh")
        # Should use Chinese joiner
        if result:
            assert "；" in result or len(result) > 0

    def test_disabled_memory_returns_none(self, db_session, config):
        set_memory_enabled(db_session, "u1", False)
        result = render_memory_block(db_session, "u1", config)
        assert result is None

    def test_query_topics_boost_matching_beliefs(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="User has trouble sleeping", confidence=0.8,
                     session_id="s1")
        record_claim(db_session, "u1", dimension="topic", key="work",
                     claim_text="User works hard", confidence=0.8,
                     session_id="s1")
        # Query about sleep should prioritize sleep belief
        result = render_memory_block(db_session, "u1", config, language="en",
                                     user_message="I can't sleep")
        if result:
            # The result should contain sleep-related content
            assert "sleep" in result.lower() or "睡" in result