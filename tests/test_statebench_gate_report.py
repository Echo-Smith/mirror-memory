"""Offline tests for the StateBench gate four-state semantics and report chain.

Covers the section-2 requirements end to end without a network or API key:

* A — forced vs production semantics: forced demands every-turn K2; production
  reports true coverage without lowering the forced thresholds.
* B — per-track isolation: a completed track is not overwritten by another
  track's failure; ``error`` never becomes ``pass`` and is never relabelled
  ``not_run``.
* C — raw results / manifest / gate verdict / summary land in four distinct,
  non-overwriting paths, with the dataset version stated explicitly.
* D — fake extractor / answerer scenarios: success, missing key (not_run), one
  track failing (error), production throttling.
* E — the CI adapter + workflow expose multiple artifacts, the four states, and
  the dataset version label.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirror_memory.bench import gate as gate_mod
from mirror_memory.bench.gate import (
    GATE_STATE_OVERALL,
    TRACK_ERROR,
    TRACK_FAIL,
    TRACK_NOT_RUN,
    TRACK_PASS,
    check_release_gate,
    evaluate_tracks,
)
from mirror_memory.bench.statebench_v1_1 import STATEBENCH_V11_VERSION

import importlib.util

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CI_TOOL = _REPO_ROOT / "tools" / "m5_statebench_ci.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "statebench-gate.yml"


def _load_ci_tool():
    """Import tools/m5_statebench_ci.py (not on sys.path) as a module."""
    spec = importlib.util.spec_from_file_location("m5_statebench_ci", _CI_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic report builders (mirror the real v1.1 report shape)
# ---------------------------------------------------------------------------

_CATEGORIES = (
    "replacement", "round_trip", "multi_value", "contradiction",
    "temporal_event", "preference_evolution", "stale_state",
)


def _track_payload(score: float | None, total: int = 200) -> dict:
    return {"total": total, "passed": None, "score": score}


def _make_report(
    *,
    mode: str = "forced",
    state_score: float = 0.93,
    recall_score: float = 0.95,
    answer_score: float | None = 0.95,
    extraction_health: dict | None = None,
    full_coverage: bool = False,
) -> dict:
    cases: list[dict] = []
    if full_coverage:
        # Real case ids from the packaged v1.1 dataset, in dataset order, so
        # the forced run passes the exact-coverage check.
        from mirror_memory.bench.statebench_v1_1 import load_statebench_v1_1

        cases = [{"case_id": c.case_id} for c in load_statebench_v1_1()]
    return {
        "engine": "mirror1",
        "dataset_version": STATEBENCH_V11_VERSION,
        "dataset_sha256": "same-dataset-sha",
        "extraction_mode": mode,
        "extractor_model": "fake-model",
        "answerer_model": "fake-model",
        "extraction_health": extraction_health or {
            "k2_attempts": 10, "k2_failures": 0, "k2_empty_responses": 0,
            "k2_parse_failures": 0, "k2_schema_rejections": 0,
            "dataset_turns": 10,
        },
        "tracks": {
            "state": {
                "total": 200, "score": state_score,
                "by_category": {c: {"score": state_score} for c in _CATEGORIES},
            },
            "recall": _track_payload(recall_score),
            "answer": _track_payload(answer_score),
        },
        "cases": cases,
    }


def _gate_report_with_tracks(*, state, recall, answer) -> dict:
    report = _make_report()
    report["tracks"]["state"] = state
    report["tracks"]["recall"] = recall
    report["tracks"]["answer"] = answer
    return report


# ===========================================================================
# B — four-state classification: pass / fail / not_run / error
# ===========================================================================

class TestFourStateClassification:
    def test_pass_track(self):
        v = evaluate_tracks(_gate_report_with_tracks(
            state={"score": 0.99}, recall={"score": 0.99}, answer={"score": 0.99},
        ))
        assert v["state"].state == TRACK_PASS
        assert v["recall"].state == TRACK_PASS
        assert v["answer"].state == TRACK_PASS

    def test_fail_track_below_threshold(self):
        v = evaluate_tracks(_gate_report_with_tracks(
            state={"score": 0.50}, recall={"score": 0.99}, answer={"score": 0.99},
        ))
        assert v["state"].state == TRACK_FAIL
        assert v["state"].passed is False

    def test_not_run_track(self):
        v = evaluate_tracks(_gate_report_with_tracks(
            state={"score": 0.99}, recall={"score": 0.99},
            answer={"score": None, "not_run": True},
        ))
        assert v["answer"].state == TRACK_NOT_RUN

    def test_error_track_not_confused_with_not_run(self):
        v = evaluate_tracks(_gate_report_with_tracks(
            state={"score": 0.99}, recall={"score": 0.99},
            answer={"score": None, "error": "RuntimeError: boom"},
        ))
        assert v["answer"].state == TRACK_ERROR
        assert v["answer"].state != TRACK_NOT_RUN
        assert "boom" in v["answer"].error

    def test_error_beats_not_run_precedence(self):
        """A payload carrying both flags is an error, never a benign not_run."""
        v = evaluate_tracks(_gate_report_with_tracks(
            state={"score": 0.99}, recall={"score": 0.99},
            answer={"score": None, "not_run": True, "error": "OSError: disk"},
        ))
        assert v["answer"].state == TRACK_ERROR


# ===========================================================================
# B — isolation: one track's failure must not overwrite another's result
# ===========================================================================

class TestTrackIsolation:
    def test_error_on_one_track_does_not_overwrite_other_pass(self):
        report = _gate_report_with_tracks(
            state={"score": 0.98}, recall={"score": 0.98},
            answer={"score": None, "error": "TimeoutError: timeout"},
        )
        verdicts = evaluate_tracks(report)
        result = check_release_gate(report, track_verdicts=verdicts)
        # The failing track is an error, and the passing tracks keep passing.
        assert result.tracks["state"].state == TRACK_PASS
        assert result.tracks["recall"].state == TRACK_PASS
        assert result.tracks["answer"].state == TRACK_ERROR
        assert result.passed is False
        assert any("answer track error" in f for f in result.failures)

    def test_not_run_is_never_a_pass(self):
        report = _gate_report_with_tracks(
            state={"score": 0.98}, recall={"score": 0.98},
            answer={"score": None, "not_run": True},
        )
        result = check_release_gate(report, track_verdicts=evaluate_tracks(report))
        assert result.passed is False
        assert result.tracks["answer"].state == TRACK_NOT_RUN


# ===========================================================================
# A — forced vs production semantics
# ===========================================================================

class TestForcedVsProduction:
    def test_forced_requires_every_turn_k2(self):
        """A forced run that skipped turns fails the every-turn attempt check."""
        report = _make_report(
            mode="forced",
            extraction_health={
                "k2_attempts": 8, "k2_failures": 0, "k2_empty_responses": 0,
                "k2_parse_failures": 0, "k2_schema_rejections": 0,
                "dataset_turns": 10,
            },
        )
        result = check_release_gate(report)
        assert any("K2 attempted 8 of 10 turns" in f for f in result.failures)

    def test_production_does_not_need_every_turn_attempt(self):
        """Production is measured as configured: a throttle is not a gate fail
        of the every-turn kind; the gate reports the production gap instead."""
        forced = _make_report(mode="forced", state_score=0.93)
        production = _make_report(mode="production", state_score=0.90)
        # Production's k2_attempts can be far below dataset_turns.
        production["extraction_health"]["k2_attempts"] = 3
        result = check_release_gate(
            forced, production_report=production,
        )
        # No "K2 attempted ... of ... turns" failure from the PRODUCTION run.
        assert not any(
            "of 10 turns" in f and "production" in f for f in result.failures
        )
        # The gap check is present and passes (0.03 <= 0.05).
        gap_checks = [c for c in result.checks if c["check"] == "production_gap"]
        assert gap_checks and gap_checks[0]["passed"] is True

    def test_production_throttle_does_not_lower_forced_threshold(self):
        """A big forced-production gap fails the gap check without relaxing the
        fixed forced thresholds."""
        forced = _make_report(mode="forced", state_score=0.93)
        production = _make_report(mode="production", state_score=0.70)
        production["extraction_health"]["k2_attempts"] = 2
        result = check_release_gate(forced, production_report=production)
        assert any("production gap" in f for f in result.failures)
        # Forced thresholds are unchanged constants.
        assert GATE_STATE_OVERALL == 0.90
        assert gate_mod.GATE_PRODUCTION_GAP == 0.05

    def test_production_error_does_not_erase_forced_verdict(self):
        """If production errored, forced's already-computed pass must stand;
        the production problem is surfaced but does not overwrite forced."""
        forced = _make_report(mode="forced", state_score=0.95)
        verdicts = evaluate_tracks(forced)
        verdicts["production"] = gate_mod.TrackVerdict(
            track="production", state=TRACK_ERROR, error="RuntimeError: prod down",
        )
        result = check_release_gate(
            forced, production_report=None, track_verdicts=verdicts,
        )
        assert result.tracks["state"].state == TRACK_PASS
        assert result.tracks["production"].state == TRACK_ERROR
        # A missing production report still fails the gap check (fail-closed),
        # but the forced track's own result is intact.
        assert any("production gap not evaluated" in f for f in result.failures)


# ===========================================================================
# D — offline CI adapter scenarios with fake extractor/answerer
# ===========================================================================

def _fake_report(mode: str, *, state_score: float = 0.95,
                 full_coverage: bool = False) -> SimpleNamespace:
    """A minimal object shaped like StateBenchV11Report for the CI tool."""
    return SimpleNamespace(
        dataset_version=STATEBENCH_V11_VERSION,
        extraction_mode=mode,
        to_dict=lambda: _make_report(
            mode=mode, state_score=state_score, full_coverage=full_coverage,
        ),
    )


@pytest.fixture
def ci(monkeypatch):
    """The CI tool with all heavy runners replaced by offline fakes."""
    module = _load_ci_tool()

    # Deterministic base report: what a key-less local run yields.
    def _det_run(**kwargs):
        return _fake_report("deterministic", state_score=0.95)

    monkeypatch.setattr(module, "run_statebench_v1_1", _det_run)

    # The manifest writer: create a real file so manifest_present can pass.
    def _write(report, path, **kwargs):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict()), encoding="utf-8")
        manifest = out.with_name(f"{out.stem}.manifest.json")
        manifest.write_text(json.dumps({
            "commit": "deadbeef", "dataset_sha256": "same-dataset-sha",
            "config_hashes": {"a.yaml": "h"}, "extractor_model": "fake-model",
        }), encoding="utf-8")
        return out, manifest

    monkeypatch.setattr(module, "write_statebench_v1_1_report", _write)
    return module


def _run_main(ci, tmp_path, monkeypatch, *, env_key=None, allow_llm=False,
              forced_factory=None, production_factory=None):
    if env_key is None:
        monkeypatch.delenv("MM_LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("MM_LLM_API_KEY", env_key)
        monkeypatch.setenv("MM_LLM_MODEL", "fake-model")
        monkeypatch.delenv("MM_LLM_BASE_URL", raising=False)

    # Stub client builders so no network is touched.
    def _build_extractor(*a, **k):
        return object()

    def _build_answerer(*a, **k):
        return lambda q, c: ""

    import mirror_memory.bench.run as run_mod
    monkeypatch.setattr(run_mod, "_build_extractor", _build_extractor)
    monkeypatch.setattr(run_mod, "_build_answerer", _build_answerer)

    call_state = {"n": 0}

    def _dispatcher(**kwargs):
        mode = kwargs.get("extraction_mode")
        if mode == "forced" and forced_factory is not None:
            return forced_factory()
        if mode == "production" and production_factory is not None:
            return production_factory()
        return _fake_report(mode, state_score=0.95)

    # The deterministic base run keeps its own fake; LLM modes go through the
    # dispatcher.
    det = ci.run_statebench_v1_1

    def _run(**kwargs):
        mode = kwargs.get("extraction_mode", "deterministic")
        if mode == "deterministic":
            return det(**kwargs)
        return _dispatcher(**kwargs)

    monkeypatch.setattr(ci, "run_statebench_v1_1", _run)

    out = tmp_path / "statebench-gate.json"
    code = ci.main(["--output", str(out), "--allow-llm"] if allow_llm else ["--output", str(out)])
    return code, out


class TestCIAdapterScenarios:
    def test_success_writes_four_nonoverwriting_artifacts(self, ci, tmp_path, monkeypatch):
        code, out = _run_main(
            ci, tmp_path, monkeypatch, env_key="sk-test", allow_llm=True,
            forced_factory=lambda: _fake_report("forced", state_score=0.95, full_coverage=True),
            production_factory=lambda: _fake_report("production", state_score=0.93, full_coverage=True),
        )
        assert code == 0
        report = out
        manifest = out.with_name("statebench-gate.manifest.json")
        verdict = out.with_name("statebench-gate.verdict.json")
        summary = out.with_name("statebench-gate.summary.md")
        for p in (report, manifest, verdict, summary):
            assert p.exists(), p
        # Four DISTINCT paths.
        assert len({report, manifest, verdict, summary}) == 4

        vdata = json.loads(verdict.read_text())
        assert vdata["dataset_version"] == STATEBENCH_V11_VERSION
        assert "statebench_v1_1" in vdata["dataset_version_label"]
        assert vdata["llm_tracks_enabled"] is True
        # Forced tracks all pass.
        assert vdata["tracks"]["state"]["state"] == TRACK_PASS
        assert vdata["tracks"]["answer"]["state"] == TRACK_PASS

        sdata = summary.read_text()
        assert sdata.splitlines()[0].startswith("PASS:")
        assert STATEBENCH_V11_VERSION in sdata

    def test_missing_key_marks_not_run_never_pass(self, ci, tmp_path, monkeypatch):
        code, out = _run_main(ci, tmp_path, monkeypatch, env_key=None, allow_llm=False)
        assert code == 0
        verdict = json.loads(out.with_name("statebench-gate.verdict.json").read_text())
        assert verdict["llm_tracks_enabled"] is False
        # answer track is not_run, never pass.
        assert verdict["tracks"]["answer"]["state"] == TRACK_NOT_RUN
        assert verdict["passed"] is False
        summary = out.with_name("statebench-gate.summary.md").read_text()
        assert summary.splitlines()[0].startswith("FAIL:")
        assert "not_run" in summary

    def test_one_track_failing_is_error_not_not_run(self, ci, tmp_path, monkeypatch):
        """Forced extraction raises -> error state, never not_run, never pass.

        Production still succeeds, proving the error isolation: the failed
        forced track does not overwrite a completed production run into a
        bogus pass.
        """
        def _boom():
            raise RuntimeError("forced extractor exploded")

        code, out = _run_main(
            ci, tmp_path, monkeypatch, env_key="sk-test", allow_llm=True,
            forced_factory=_boom,
            production_factory=lambda: _fake_report("production", state_score=0.95),
        )
        assert code == 0
        verdict = json.loads(out.with_name("statebench-gate.verdict.json").read_text())
        assert verdict["tracks"]["forced"]["state"] == TRACK_ERROR
        assert "exploded" in verdict["tracks"]["forced"]["error"]
        assert verdict["passed"] is False
        summary = out.with_name("statebench-gate.summary.md").read_text()
        assert summary.splitlines()[0].startswith("FAIL:")
        # The summary must clearly show infrastructure error, not a benign
        # not_run for the forced track.
        assert "Infrastructure error" in summary or "error" in summary.lower()

    def test_production_throttling_is_reported_and_measured(self, ci, tmp_path, monkeypatch):
        """Production throttles its coverage; the gap is measured and visible."""
        forced = _fake_report("forced", state_score=0.95)
        production = _fake_report("production", state_score=0.80)

        code, out = _run_main(
            ci, tmp_path, monkeypatch, env_key="sk-test", allow_llm=True,
            forced_factory=lambda: forced,
            production_factory=lambda: production,
        )
        assert code == 0
        verdict = json.loads(out.with_name("statebench-gate.verdict.json").read_text())
        gap_checks = [c for c in verdict["checks"] if c["check"] == "production_gap"]
        assert gap_checks and gap_checks[0]["passed"] is False
        assert any("production gap" in f for f in verdict["failures"])

    def test_ci_never_writes_five_paths_into_one(self, ci, tmp_path, monkeypatch):
        """Re-running with a different stem must not clobber a prior run."""
        code1, out1 = _run_main(ci, tmp_path, monkeypatch, env_key=None, allow_llm=False)
        # second run with different stem
        monkeypatch.delenv("MM_LLM_API_KEY", raising=False)
        out2 = tmp_path / "statebench-gate-run2.json"
        code2 = ci.main(["--output", str(out2)])
        assert code1 == 0 and code2 == 0
        # Both verdicts survive with distinct paths.
        assert out1.with_name("statebench-gate.verdict.json").exists()
        assert out2.with_name("statebench-gate-run2.verdict.json").exists()


# ===========================================================================
# C/E — dataset version label and workflow contract
# ===========================================================================

class TestWorkflowContract:
    def test_workflow_yaml_parses_and_uploads_multiple_artifacts(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load(_WORKFLOW.read_text())
        assert data is not None
        jobs = data["jobs"]
        assert "gate" in jobs
        steps = jobs["gate"]["steps"]
        # Locate the upload-artifact step(s).
        uploads = [s for s in steps if "uses" in s and "upload-artifact" in s["uses"]]
        assert uploads, "workflow must upload artifacts"
        all_paths = "\n".join(u["with"]["path"] for u in uploads)
        # The four artifact kinds must all be uploaded.
        for suffix in (
            "statebench-gate.json",
            "statebench-gate.manifest.json",
            "statebench-gate.verdict.json",
            "statebench-gate-summary.md",
        ):
            assert suffix in all_paths, suffix

    def test_workflow_job_is_report_only(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load(_WORKFLOW.read_text())
        steps = data["jobs"]["gate"]["steps"]
        gate_step = next(s for s in steps if s.get("id") == "gate")
        assert gate_step.get("continue-on-error") is True

    def test_ci_tool_and_gate_expose_four_states(self):
        module = _load_ci_tool()
        # The CI tool imports the canonical constants from the gate module.
        assert module.TRACK_NOT_RUN == "not_run"
        assert module.TRACK_ERROR == "error"
        # The canonical four-state constants live in gate.py.
        assert gate_mod.TRACK_PASS == "pass"
        assert gate_mod.TRACK_FAIL == "fail"
        assert gate_mod.TRACK_NOT_RUN == "not_run"
        assert gate_mod.TRACK_ERROR == "error"
        assert gate_mod.TRACK_STATES == {"pass", "fail", "not_run", "error"}


# ===========================================================================
# Dataset version truth check
# ===========================================================================

def test_runner_loads_v1_1_dataset_and_reports_its_version():
    """The runner loads statebench_v1_1.json, not statebench_v1.json."""
    from mirror_memory.bench.statebench_v1_1 import statebench_v1_1_path

    path = statebench_v1_1_path()
    assert path.name == "statebench_v1_1.json"
    payload = json.loads(path.read_text())
    assert payload["version"] == STATEBENCH_V11_VERSION
    assert payload["case_count"] == 200
