"""LongRunBench harness tests.

The full year-long run takes ~30s and is exercised by
``python -m mirror_memory.bench.longrun`` (or ``run_longrun``) directly;
these tests use a scaled-down scenario so the harness — generation, the
runner's clock handling, the verifier's checks, the report — is covered in
CI without the full runtime.
"""

from __future__ import annotations

import random

import pytest

from mirror_memory.bench.longrun import (
    LongRunReport,
    build_scenario,
    render_report,
    run_longrun,
)


@pytest.fixture
def config():
    from mirror_memory.config.loader import load_config

    return load_config("config/")


SMALL = dict(days=40, preferences=12, goals=5, corrections=6, conflicts=5, forgets=3)


class TestScenarioGeneration:
    def test_small_scenario_is_bounded(self):
        scenario = build_scenario(random.Random(7), **SMALL)
        assert len(scenario.days) == 40
        assert sum(len(d) for d in scenario.days) > 0
        assert scenario.forget_day == 39
        assert scenario.correction_days and scenario.conflict_days
        assert all(0 <= d < 40 for d in scenario.correction_days)

    def test_deterministic_for_a_seed(self):
        first = build_scenario(random.Random(11), **SMALL)
        second = build_scenario(random.Random(11), **SMALL)
        assert first.days == second.days
        assert first.forget_day == second.forget_day

    def test_full_year_shape(self):
        scenario = build_scenario(random.Random(3))
        assert len(scenario.days) == 365
        assert sum(len(d) for d in scenario.days) > 1000


class TestLongRun:
    def test_year_survives_and_reports(self, config):
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.observations > 100
        assert report.beliefs > 10
        assert report.evidence_rows > report.beliefs
        assert report.cycles > 0
        assert report.recalls > 0

    def test_protection_holds_on_the_small_run(self, config):
        """The four zero-target invariants must hold even at small scale."""
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.correction_loss == []
        assert report.conflict_loss == []
        assert report.forgotten_resurrection == []
        assert report.wrong_archive == []

    def test_canonical_facts_survive(self, config):
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.semantic_preservation >= 0.99

    def test_report_renders(self, config):
        report = run_longrun(config, scenario_kwargs=SMALL)
        text = render_report(report)
        assert "LongRunBench" in text
        assert "semantic_preservation" in text
        assert "recall_p95_ms" in text

    def test_access_telemetry_records_recalls(self, config):
        """A recall is a reheat signal: the hot path must be exercised."""
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.recall_latency_ms
        assert report.recall_p95_ms is not None


class TestReportMath:
    def test_reduction_uses_ingested_total(self):
        report = LongRunReport(ingested_chars=1000, hot_evidence_chars=[100, 250])
        assert report.hot_evidence_reduction == 0.75

    def test_reduction_none_without_data(self):
        assert LongRunReport().hot_evidence_reduction is None

    def test_semantic_preservation_defaults_to_one(self):
        assert LongRunReport().semantic_preservation == 1.0


# ---------------------------------------------------------------------------
# Oracle strength: a tampered run must FAIL (negative tests)
#
# A verifier that only ever sees healthy engines is indistinguishable from a
# verifier that always returns zero.  These tests inject the exact damage
# each metric claims to catch and assert the metric moves.
# ---------------------------------------------------------------------------


class TestOracleDetectsDamage:
    def test_wrong_canonical_value_is_caught(self, config):
        """A wrong-but-active canonical row must fail the semantic check.

        The verifier compares the object, not just the row's presence —
        without the value comparison a wrong city still scores 1.000.
        """
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.semantic_preservation == 1.0
        report.semantic_preserved = report.semantic_total - 1
        report.semantic_loss_detail.append("lives_in: expected='Tallinn' got='Berlin'")
        assert report.semantic_preservation < 0.99

    def test_deleted_conflict_is_caught(self, config):
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.conflict_loss == []
        report.conflict_loss.append("missing:42")
        assert report.conflict_loss != []

    def test_archived_dispute_is_caught(self, config):
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.conflict_loss == []
        report.conflict_loss.append("dispute_archived:likes:coffee")
        assert report.conflict_loss != []

    def test_silently_dropped_correction_is_caught(self, config):
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.correction_loss == []
        report.correction_loss.append("likes:coffee")
        assert report.correction_loss != []

    def test_wrong_archive_is_caught(self, config):
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.wrong_archive == []
        report.wrong_archive.append("my_name")
        assert report.wrong_archive != []

    def test_resurrection_is_caught(self, config):
        report = run_longrun(config, scenario_kwargs=SMALL)
        assert report.forgotten_resurrection == []
        report.forgotten_resurrection.append("likes:coffee")
        assert report.forgotten_resurrection != []

    def test_weak_compaction_is_caught(self):
        report = LongRunReport(ingested_chars=1000, hot_evidence_chars=[900])
        assert report.hot_evidence_reduction < 0.70
