"""Tests for core/activity.py — belief_activity."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from mirror_memory.core.activity import belief_activity


def _belief(confidence=0.8, layer="L2", last_evidence_at=None):
    return SimpleNamespace(confidence=confidence, layer=layer, last_evidence_at=last_evidence_at)


class TestBeliefActivity:
    def test_none_last_evidence_returns_zero(self):
        b = _belief(last_evidence_at=None)
        assert belief_activity(b) == 0.0

    def test_just_now_full_activity(self):
        now = datetime.now(UTC)
        b = _belief(confidence=0.8, layer="L2", last_evidence_at=now)
        activity = belief_activity(b, now=now)
        assert abs(activity - 0.8) < 0.01

    def test_half_life_decay(self):
        now = datetime.now(UTC)
        half_life = datetime.now(UTC) - timedelta(days=60)
        b = _belief(confidence=1.0, layer="L2", last_evidence_at=half_life)
        activity = belief_activity(b, now=now, half_life_days=60.0)
        assert abs(activity - 0.5) < 0.01

    def test_l1_layer_full_factor(self):
        now = datetime.now(UTC)
        b = _belief(confidence=0.8, layer="L1", last_evidence_at=now)
        activity = belief_activity(b, now=now)
        assert abs(activity - 0.8) < 0.01

    def test_l2_layer_full_factor(self):
        now = datetime.now(UTC)
        b = _belief(confidence=0.8, layer="L2", last_evidence_at=now)
        activity = belief_activity(b, now=now)
        assert abs(activity - 0.8) < 0.01

    def test_l4_layer_reduced_factor(self):
        now = datetime.now(UTC)
        b = _belief(confidence=0.8, layer="L4", last_evidence_at=now)
        activity = belief_activity(b, now=now)
        assert abs(activity - 0.56) < 0.01  # 0.8 * 0.7

    def test_zero_confidence_returns_zero(self):
        now = datetime.now(UTC)
        b = _belief(confidence=0.0, layer="L2", last_evidence_at=now)
        assert belief_activity(b, now=now) == 0.0

    def test_old_belief_low_activity(self):
        now = datetime.now(UTC)
        old = datetime.now(UTC) - timedelta(days=365)
        b = _belief(confidence=0.9, layer="L2", last_evidence_at=old)
        activity = belief_activity(b, now=now)
        assert activity < 0.1

    def test_naive_datetime_handled(self):
        """Naive datetime (no tzinfo) should be treated as UTC."""
        now = datetime.now(UTC)
        naive_now = now.replace(tzinfo=None)
        b = _belief(confidence=0.8, layer="L2", last_evidence_at=naive_now)
        activity = belief_activity(b, now=now)
        assert activity > 0.7  # should be close to full activity

    def test_custom_half_life(self):
        now = datetime.now(UTC)
        ten_days_ago = now - timedelta(days=10)
        b = _belief(confidence=1.0, layer="L2", last_evidence_at=ten_days_ago)
        activity_short = belief_activity(b, now=now, half_life_days=10.0)
        activity_long = belief_activity(b, now=now, half_life_days=100.0)
        assert activity_short < activity_long  # shorter half-life = faster decay
