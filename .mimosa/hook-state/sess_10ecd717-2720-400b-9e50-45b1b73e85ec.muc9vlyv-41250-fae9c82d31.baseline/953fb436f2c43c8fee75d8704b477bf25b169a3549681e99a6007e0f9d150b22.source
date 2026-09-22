"""Tests for core/budget.py — compute_profile_budget."""

from mirror_memory.core.budget import compute_profile_budget


class TestBudget:
    def test_zero_pressure(self):
        """No load, no topics → maximum budget (base * 1.2)."""
        b = compute_profile_budget(0, 0.0, base=480, floor=160, cap=720)
        assert b == 576  # 480 * 1.2

    def test_max_pressure(self):
        """Full load + full topics → minimum budget (base * 0.8)."""
        b = compute_profile_budget(1800, 1.0, base=480, floor=160, cap=720)
        assert b == 384  # 480 * 0.8

    def test_negative_tail_load_clamped(self):
        b = compute_profile_budget(-100, 0.0, base=480, floor=160, cap=720)
        b2 = compute_profile_budget(0, 0.0, base=480, floor=160, cap=720)
        assert b == b2

    def test_knowledge_proxy_clamped_above_1(self):
        b = compute_profile_budget(0, 2.0, base=480, floor=160, cap=720)
        b2 = compute_profile_budget(0, 1.0, base=480, floor=160, cap=720)
        assert b == b2

    def test_knowledge_proxy_clamped_below_0(self):
        b = compute_profile_budget(0, -0.5, base=480, floor=160, cap=720)
        b2 = compute_profile_budget(0, 0.0, base=480, floor=160, cap=720)
        assert b == b2

    def test_below_floor_returns_floor(self):
        b = compute_profile_budget(0, 0.0, base=100, floor=200, cap=500)
        assert b == 200

    def test_above_cap_returns_cap(self):
        b = compute_profile_budget(0, 0.0, base=1000, floor=100, cap=500)
        assert b == 500

    def test_floor_equals_cap(self):
        b = compute_profile_budget(500, 0.5, base=300, floor=300, cap=300)
        assert b == 300

    def test_mid_pressure(self):
        """Half pressure should be between min and max."""
        b_min = compute_profile_budget(1800, 1.0, base=480, floor=160, cap=720)
        b_mid = compute_profile_budget(900, 0.5, base=480, floor=160, cap=720)
        b_max = compute_profile_budget(0, 0.0, base=480, floor=160, cap=720)
        assert b_min < b_mid < b_max

    def test_returns_int(self):
        b = compute_profile_budget(100, 0.3, base=480, floor=160, cap=720)
        assert isinstance(b, int)
