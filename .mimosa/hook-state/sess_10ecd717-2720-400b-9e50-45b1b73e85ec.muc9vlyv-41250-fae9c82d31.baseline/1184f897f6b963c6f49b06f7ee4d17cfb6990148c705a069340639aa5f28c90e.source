"""Tests for core/retrieval.py — score_belief."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import json

from mirror_memory.core.retrieval import score_belief


def _belief(dimension="topic", key="sleep", confidence=0.8, layer="L2",
            last_evidence_at=None, value_json="{}"):
    return SimpleNamespace(
        dimension=dimension, key=key, confidence=confidence, layer=layer,
        last_evidence_at=last_evidence_at, value_json=value_json,
    )


class TestScoreBelief:
    def test_zero_score_no_signals(self):
        """Belief with no matching signals should have low score."""
        now = datetime.now(UTC)
        b = _belief(key="unknown", confidence=0.1, layer="L4", last_evidence_at=now - timedelta(days=30))
        s = score_belief(b, topics=set(), now=now)
        assert s >= 0  # activity contributes some score

    def test_topic_hit_bonus(self):
        now = datetime.now(UTC)
        b = _belief(key="sleep", last_evidence_at=now)
        s_with = score_belief(b, topics={"sleep"}, now=now)
        s_without = score_belief(b, topics=set(), now=now)
        assert s_with > s_without

    def test_layer_confirmed_bonus(self):
        now = datetime.now(UTC)
        b_l2 = _belief(layer="L2", last_evidence_at=now)
        b_l4 = _belief(layer="L4", last_evidence_at=now)
        assert score_belief(b_l2, topics=set(), now=now) > score_belief(b_l4, topics=set(), now=now)

    def test_activity_contribution(self):
        now = datetime.now(UTC)
        b_fresh = _belief(confidence=0.8, last_evidence_at=now)
        b_old = _belief(confidence=0.8, last_evidence_at=now - timedelta(days=60))
        assert score_belief(b_fresh, topics=set(), now=now) > score_belief(b_old, topics=set(), now=now)

    def test_freshness_bonus(self):
        now = datetime.now(UTC)
        b_fresh = _belief(last_evidence_at=now)
        b_stale = _belief(last_evidence_at=now - timedelta(hours=3))
        s_fresh = score_belief(b_fresh, topics=set(), now=now)
        s_stale = score_belief(b_stale, topics=set(), now=now)
        assert s_fresh > s_stale

    def test_multi_evidence_bonus(self):
        now = datetime.now(UTC)
        val_many = json.dumps({"evidence_ids": [1, 2, 3]})
        val_few = json.dumps({"evidence_ids": [1]})
        b_many = _belief(value_json=val_many, last_evidence_at=now)
        b_few = _belief(value_json=val_few, last_evidence_at=now)
        assert score_belief(b_many, topics=set(), now=now) > score_belief(b_few, topics=set(), now=now)

    def test_dimension_boost(self):
        now = datetime.now(UTC)
        b = _belief(dimension="D7", last_evidence_at=now)
        boosts = {"D7": 4.0}
        s_with = score_belief(b, topics=set(), dimension_boosts=boosts, now=now)
        s_without = score_belief(b, topics=set(), dimension_boosts={}, now=now)
        assert s_with > s_without

    def test_none_last_evidence_at(self):
        now = datetime.now(UTC)
        b = _belief(last_evidence_at=None)
        s = score_belief(b, topics=set(), now=now)
        assert s >= 0

    def test_none_topics(self):
        now = datetime.now(UTC)
        b = _belief(last_evidence_at=now)
        s = score_belief(b, topics=None, now=now)
        assert s >= 0

    def test_all_signals_combined(self):
        """Maximum score when all signals fire."""
        now = datetime.now(UTC)
        b = _belief(
            dimension="D7", key="sleep", confidence=0.9, layer="L2",
            last_evidence_at=now,
            value_json=json.dumps({"evidence_ids": [1, 2, 3]}),
        )
        s = score_belief(b, topics={"sleep"}, dimension_boosts={"D7": 4.0}, now=now)
        assert s > 10  # topic(5) + boost(4) + layer(3) + activity(~1.8) + evidence(1) + freshness(1)

    def test_empty_value_json(self):
        now = datetime.now(UTC)
        b = _belief(value_json="", last_evidence_at=now)
        s = score_belief(b, topics=set(), now=now)
        assert s >= 0

    def test_malformed_value_json(self):
        now = datetime.now(UTC)
        b = _belief(value_json="not json", last_evidence_at=now)
        s = score_belief(b, topics=set(), now=now)
        assert s >= 0
