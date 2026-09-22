"""Tests for core/confidence.py — compute_confidence_weight."""

from mirror_memory.core.confidence import compute_confidence_weight


class TestConfidenceWeight:
    def test_user_confirmed_l2_minimal_penalty(self):
        """Best case: user_confirmed + L2 + high support + high diversity."""
        w = compute_confidence_weight(
            source="user_confirmed", layer="L2", support_count=3, context_diversity=3,
        )
        assert w == 1.0

    def test_extracted_l4_max_static_penalty(self):
        """Worst static case: extracted + L4 + high support + high diversity."""
        w = compute_confidence_weight(
            source="extracted", layer="L4", support_count=3, context_diversity=3,
        )
        assert w == 0.8  # 1.0 - 0.1 (source) - 0.1 (layer)

    def test_unknown_source_penalized(self):
        w = compute_confidence_weight(source="unknown_source")
        assert w < 1.0

    def test_unknown_layer_penalized(self):
        w = compute_confidence_weight(layer="L99")
        assert w < 1.0

    def test_zero_support_count_penalized(self):
        w0 = compute_confidence_weight(support_count=0)
        w3 = compute_confidence_weight(support_count=3)
        assert w0 < w3

    def test_high_support_count_no_penalty(self):
        w = compute_confidence_weight(support_count=5)
        # No additional penalty beyond support_count=3
        w3 = compute_confidence_weight(support_count=3)
        assert w == w3

    def test_context_diversity_only_when_support_ge_2(self):
        """Context diversity penalty only applies when support_count >= 2."""
        w_low = compute_confidence_weight(support_count=1, context_diversity=0)
        w_high = compute_confidence_weight(support_count=1, context_diversity=3)
        # With support_count=1, diversity doesn't matter
        assert w_low == w_high

    def test_context_diversity_penalty_with_enough_support(self):
        w0 = compute_confidence_weight(support_count=3, context_diversity=0)
        w3 = compute_confidence_weight(support_count=3, context_diversity=3)
        assert w0 < w3

    def test_staleness_penalty(self):
        w_fresh = compute_confidence_weight(days_since_last_evidence=0)
        w_stale = compute_confidence_weight(days_since_last_evidence=30)
        assert w_fresh > w_stale

    def test_no_staleness_within_7_days(self):
        w0 = compute_confidence_weight(days_since_last_evidence=0)
        w7 = compute_confidence_weight(days_since_last_evidence=7)
        assert w0 == w7

    def test_contradiction_penalty(self):
        w0 = compute_confidence_weight(contradict_count=0)
        w2 = compute_confidence_weight(contradict_count=2)
        assert w0 > w2

    def test_clarification_penalty(self):
        import json
        w_normal = compute_confidence_weight(value_json='{}')
        w_clarify = compute_confidence_weight(value_json=json.dumps({"clarification_status": "needs_clarification"}))
        assert w_normal > w_clarify

    def test_floor_clamped_at_0_5(self):
        """Even with maximum penalties, weight should not go below 0.5."""
        w = compute_confidence_weight(
            source="extracted", layer="L4", support_count=0,
            contradict_count=10, context_diversity=0, days_since_last_evidence=100,
            value_json='{"clarification_status": "needs_clarification"}',
        )
        assert w >= 0.5

    def test_ceiling_clamped_at_1_0(self):
        w = compute_confidence_weight(source="user_confirmed", layer="L2", support_count=10)
        assert w <= 1.0
