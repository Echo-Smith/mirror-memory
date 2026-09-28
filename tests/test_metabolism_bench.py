"""MetabolismBench harness tests.

The full suite needs an LLM client for ingestion (see
``mirror_memory.bench.metabolism``).  These tests cover the harness itself
without one: case loading, the op executor against a real engine, the
report aggregates, and the gate — so a broken runner fails CI even when no
API key is configured.
"""

from __future__ import annotations

import json

import pytest

from mirror_memory.bench.metabolism import (
    MetabolismCase,
    MetabolismGate,
    _benchmark_config,
    load_metabolism_cases,
    render_report,
    run_case,
    run_metabolism_bench,
)


@pytest.fixture
def config():
    from mirror_memory.config.loader import load_config

    return load_config("config/")


class TestCaseLoading:
    def test_packaged_cases_load(self):
        cases = load_metabolism_cases()
        assert len(cases) >= 25
        categories = {c.category for c in cases}
        assert categories == {
            "current_survival",
            "historical_preservation",
            "correction_protection",
            "conflict_preservation",
            "evidence_compaction",
            "reactivation",
            "deletion",
            "stale_worker",
        }

    def test_category_filter(self):
        cases = load_metabolism_cases(categories=("deletion",))
        assert cases and all(c.category == "deletion" for c in cases)

    def test_custom_path(self, tmp_path):
        payload = {
            "cases": [
                {
                    "case_id": "x",
                    "category": "deletion",
                    "turns": ["I love tea."],
                    "steps": [{"op": "gone", "key": "tea"}],
                }
            ]
        }
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        cases = load_metabolism_cases(path)
        assert len(cases) == 1
        assert isinstance(cases[0], MetabolismCase)

    def test_every_step_op_is_known(self):
        known = {
            "cycle", "observe", "recall", "probe", "probe_after", "tier",
            "digest", "correct", "inject_conflict", "forget", "restore",
            "gone", "tombstone", "storage_mark", "storage_check", "stale_worker",
        }
        for case in load_metabolism_cases():
            for step in case.steps:
                assert step.get("op") in known, f"{case.case_id}: {step.get('op')}"


class TestHarness:
    def test_benchmark_config_forces_extraction_only_with_a_client(self, config):
        class _Client:
            pass

        forced = _benchmark_config(config, _Client())
        assert forced.extraction.llm_min_keyword_hits == 0
        # Without a client the config is untouched (K1-degraded mode).
        assert _benchmark_config(config, None).extraction.llm_min_keyword_hits == (
            config.extraction.llm_min_keyword_hits
        )

    def test_degraded_run_completes_and_marks_itself(self, config):
        report = run_metabolism_bench(
            config, llm_client=None, categories=("current_survival",)
        )
        assert report.degraded is True
        assert report.model == ""
        assert report.results
        # K1-only extraction cannot see every scripted fact, so the run is
        # expected to be lossy — the harness just has to finish and report.
        assert 0.0 <= report.overall_pass_rate <= 1.0
        text = render_report(report)
        assert "K1-only degraded" in text

    def test_unknown_op_fails_the_case(self, config):
        case = MetabolismCase(
            case_id="bad-op",
            category="deletion",
            turns=("I love tea.",),
            steps=({"op": "teleport"},),
        )
        result = run_case(case, config, llm_client=None)
        assert not result.passed
        assert any("unknown_op" in name for name, _ok, _d in result.checks)

    def test_case_with_no_assertions_fails(self, config):
        case = MetabolismCase(
            case_id="empty", category="deletion", turns=("I love tea.",), steps=()
        )
        result = run_case(case, config, llm_client=None)
        assert not result.passed
        assert result.error == "no assertions ran"


class TestGate:
    def _report(self, **overrides):
        from mirror_memory.bench.metabolism import BenchReport, CaseResult

        report = BenchReport(model="m", base_url="b")
        for category, passed in overrides.items():
            report.results.append(
                CaseResult(case_id=f"{category}-1", category=category, passed=passed,
                           checks=[("c", passed, "")])
            )
        return report

    def test_all_green_passes(self):
        verdict = MetabolismGate().evaluate(
            self._report(
                correction_protection=True,
                conflict_preservation=True,
                deletion=True,
                stale_worker=True,
                current_survival=True,
            )
        )
        assert verdict.passed

    def test_protection_category_must_be_perfect(self):
        verdict = MetabolismGate().evaluate(
            self._report(
                correction_protection=False,
                conflict_preservation=True,
                deletion=True,
                stale_worker=True,
                current_survival=True,
            )
        )
        assert not verdict.passed
        assert any("correction_protection" in f for f in verdict.failures)

    def test_gate_reports_metric_failures(self):
        from mirror_memory.bench.metabolism import BenchReport, CaseResult

        report = BenchReport(model="m")
        report.results.append(
            CaseResult(
                case_id="c", category="evidence_compaction", passed=True,
                checks=[("c", True, "")],
                probes_before=10, probes_after=5,
                chars_before=100, chars_after=90, theoretical_reduction=0.5,
            )
        )
        verdict = MetabolismGate().evaluate(report)
        assert not verdict.passed
        assert any("semantic preservation" in f for f in verdict.failures)
