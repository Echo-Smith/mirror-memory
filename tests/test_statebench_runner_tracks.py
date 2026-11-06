"""Real-runner track tests for StateBench v1.1 (handover item 1).

These tests drive the REAL ``run_statebench_v1_1`` runner with a fake LLM
client (small ``max_cases`` subsets) and assert on the report / manifest it
actually produces.  They do NOT replace the runner with a fake report: the
engine, extraction-call accounting, scoring, health, and manifest writing all
run for real.

The one seam these tests use is the *throttle decision predicate*
``mirror_memory.extraction.pipeline.should_extract`` in the "production
throttling" case, because the live heuristic's cold-start/novelty escapes make
production cover every turn on the short 2-turn cases.  Patching that single
predicate is a throttle seam, not a runner replacement: the runner still
executes every turn, counts K2 attempts, and emits the real report.  All other
scenarios (forced full coverage, error, not_run) run with the predicate
untouched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mirror_memory.bench.gate import TRACK_ERROR, TRACK_NOT_RUN, TRACK_PASS
from mirror_memory.bench.statebench_v1_1_runner import (
    run_statebench_v1_1,
    write_statebench_v1_1_report,
)
from mirror_memory.config.loader import load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CI_TOOL = _REPO_ROOT / "tools" / "m5_statebench_ci.py"

# A small deterministic subset (first N cases) keeps these tests fast.  These
# are a SUBSET, not the full 200-case dataset -- the coverage check therefore
# fails by design; the tests assert track-level behaviour, not full coverage.
_MAX_CASES = 6


def _load_ci_tool():
    spec = importlib.util.spec_from_file_location("m5_statebench_ci", _CI_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake LLM clients (deterministic; no network)
# ---------------------------------------------------------------------------

class _WorkingLLM:
    """A well-behaved client: every call succeeds and returns a valid payload."""

    model = "fake-working-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, system_prompt, payload_text, fallback=None):
        self.calls += 1
        return '{"claims": [], "subject": "user", "context_tags": []}'


class _FailingLLM:
    """Every call raises, like an exhausted quota / broken endpoint."""

    model = "fake-failing-model"

    def __init__(self) -> None:
        self.calls = 0
        self.call_failures = 0

    def generate(self, *, system_prompt, payload_text, fallback=None):
        self.calls += 1
        self.call_failures += 1
        raise RuntimeError("503 upstream unavailable")


# ===========================================================================
# A/D — forced full coverage (real runner, no seam)
# ===========================================================================

class TestForcedFullCoverage:
    def test_forced_attempts_every_turn(self):
        """Forced mode must attempt K2 on every dataset turn of the subset."""
        report = run_statebench_v1_1(
            config=load_config("config"),
            extraction_mode="forced",
            llm_client=_WorkingLLM(),
            max_cases=_MAX_CASES,
            database_url="sqlite://",
        )
        health = report.extraction_health
        assert health["dataset_turns"] == 2 * _MAX_CASES
        assert health["k2_attempts"] == health["dataset_turns"]
        assert health["k2_failures"] == 0
        assert report.extraction_mode == "forced"

    def test_forced_all_calls_failing_raises(self):
        """A forced run whose every K2 call fails must refuse to report."""
        with pytest.raises(RuntimeError, match="K2 calls failed"):
            run_statebench_v1_1(
                config=load_config("config"),
                extraction_mode="forced",
                llm_client=_FailingLLM(),
                max_cases=2,
                database_url="sqlite://",
            )


# ===========================================================================
# A/D — production throttling retains ACTUAL coverage + health
# ===========================================================================

class TestProductionThrottling:
    def test_production_partial_coverage_is_retained_not_raised(self, monkeypatch):
        """Under throttling, production keeps its real coverage/health and does
        NOT raise; forced would refuse the same inequality.

        The throttle predicate is patched to a deterministic even-turn schedule
        (a throttle seam only -- the runner, engine, accounting, report and
        manifest all run for real).  The live heuristic's cold-start/novelty
        escapes cover every turn on short cases, so an explicit schedule is the
        only way to exercise the partial-coverage write path deterministically.
        """
        import mirror_memory.extraction.pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "should_extract",
            lambda text, turn, config, *, session=None, user_id="": turn % 2 == 0,
        )

        report = run_statebench_v1_1(
            config=load_config("config"),
            extraction_mode="production",
            llm_client=_WorkingLLM(),
            max_cases=_MAX_CASES,
            database_url="sqlite://",
        )
        health = report.extraction_health
        # Half the turns fire -> real partial coverage retained.
        assert health["dataset_turns"] == 2 * _MAX_CASES
        assert health["k2_attempts"] == _MAX_CASES  # one per 2-turn case
        assert health["k2_attempts"] < health["dataset_turns"]
        assert health["k2_failures"] == 0
        assert report.extraction_mode == "production"

    def test_production_failures_are_retained_not_raised(self):
        """Real production call failures are kept in health (surfaced by the
        gate), never raised away and never counted as a pass."""
        # Patch throttle to fire on every turn so every call hits the failing
        # client; production must still return a report carrying k2_failures.
        import mirror_memory.extraction.pipeline as pipeline_mod

        original = pipeline_mod.should_extract
        pipeline_mod.should_extract = (
            lambda text, turn, config, *, session=None, user_id="": True
        )
        try:
            report = run_statebench_v1_1(
                config=load_config("config"),
                extraction_mode="production",
                llm_client=_FailingLLM(),
                max_cases=2,
                database_url="sqlite://",
            )
        finally:
            pipeline_mod.should_extract = original
        health = report.extraction_health
        assert health["k2_failures"] > 0
        # The report was produced (not raised) so the failure is visible.
        assert report.extraction_mode == "production"


# ===========================================================================
# B — per-track report + manifest saved separately, non-overlapping
# ===========================================================================

class TestSeparateTrackPersistence:
    def test_forced_and_production_have_distinct_report_and_manifest(self, tmp_path):
        forced = run_statebench_v1_1(
            config=load_config("config"),
            extraction_mode="forced",
            llm_client=_WorkingLLM(),
            max_cases=_MAX_CASES,
            database_url="sqlite://",
        )
        production = run_statebench_v1_1(
            config=load_config("config"),
            extraction_mode="production",
            llm_client=_WorkingLLM(),
            max_cases=_MAX_CASES,
            database_url="sqlite://",
        )
        f_report, f_manifest = write_statebench_v1_1_report(
            forced, tmp_path / "run.forced.json",
        )
        p_report, p_manifest = write_statebench_v1_1_report(
            production, tmp_path / "run.production.json",
        )
        # Four distinct files.
        paths = {f_report, f_manifest, p_report, p_manifest}
        assert len(paths) == 4
        for p in paths:
            assert p.exists()

        f_man = json.loads(f_manifest.read_text())
        p_man = json.loads(p_manifest.read_text())
        for man in (f_man, p_man):
            # Manifest carries provenance: commit, dataset sha, hashes, model.
            assert man["commit"]
            assert man["dataset_sha256"]
            assert man["extractor_model"] == "fake-working-model"
            assert "config_hashes" in man
            assert man["benchmark"].startswith("statebench-")
        # Each manifest identifies its own extraction mode.
        assert "forced" in f_man["notes"]
        assert "production" in p_man["notes"]

        f_rep = json.loads(f_report.read_text())
        p_rep = json.loads(p_report.read_text())
        assert f_rep["dataset_version"] == p_rep["dataset_version"]
        assert f_rep["extraction_mode"] == "forced"
        assert p_rep["extraction_mode"] == "production"


# ===========================================================================
# D — CI adapter drives the REAL runner (no fake reports)
# ===========================================================================

class TestCIAdapterRealRunner:
    def test_missing_key_marks_llm_tracks_not_run(self, tmp_path, monkeypatch):
        """No MM_LLM_API_KEY -> deterministic only; answer/forced not_run."""
        module = _load_ci_tool()
        monkeypatch.delenv("MM_LLM_API_KEY", raising=False)
        out = tmp_path / "gate.json"
        # Constrain to a small subset via the same dataset (fast, real runner).
        code = module.main(["--output", str(out), "--max-cases", str(_MAX_CASES)])
        assert code == 0
        verdict = json.loads(out.with_name("gate.verdict.json").read_text())
        assert verdict["llm_tracks_enabled"] is False
        # answer is not_run, never pass; gate overall not a pass.
        assert verdict["tracks"]["answer"]["state"] == TRACK_NOT_RUN
        assert verdict["passed"] is False
        summary = out.with_name("gate.summary.md").read_text()
        assert summary.splitlines()[0].startswith("FAIL:")

    def test_forced_error_keeps_surviving_production_output(
        self, tmp_path, monkeypatch,
    ):
        """If forced raises but production succeeds, forced is `error` and
        production's files/verdict survive -- neither discards the other.

        Both tracks run the REAL runner; only the client builders are stubbed
        to return a working client (production) vs. a failing one (forced).
        """
        module = _load_ci_tool()
        monkeypatch.setenv("MM_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("MM_LLM_MODEL", "fake-model")
        monkeypatch.delenv("MM_LLM_BASE_URL", raising=False)

        import mirror_memory.bench.run as run_mod

        # Answerer succeeds; extractor is chosen per mode.
        monkeypatch.setattr(
            run_mod, "_build_answerer",
            lambda *a, **k: (lambda q, c: ""),
        )

        def _build_extractor(model, api_key, base_url, extra_body=None):
            # The runner calls _build_extractor once per mode? No: CI builds one
            # extractor then reuses it for both forced and production, so make
            # the extractor fail only for forced by raising on its Nth call is
            # hard. Instead: build a client that fails only when the label says
            # forced is impossible here. Simpler: make the CI tool run forced
            # with a failing client and production with a working one by
            # patching run_statebench_v1_1 per mode.
            return _WorkingLLM()

        monkeypatch.setattr(run_mod, "_build_extractor", _build_extractor)

        real_run = module.run_statebench_v1_1

        def _run(**kwargs):
            mode = kwargs.get("extraction_mode", "deterministic")
            if mode == "forced":
                # Real runner + a client whose every call raises -> runner
                # raises "K2 calls failed" (genuine error path).
                return real_run(**{**kwargs, "llm_client": _FailingLLM()})
            if mode == "production":
                # Real runner + working client -> genuine production report.
                return real_run(**{**kwargs, "llm_client": _WorkingLLM()})
            return real_run(**kwargs)

        monkeypatch.setattr(module, "run_statebench_v1_1", _run)

        out = tmp_path / "gate.json"
        code = module.main(["--output", str(out), "--allow-llm", "--max-cases", str(_MAX_CASES)])
        assert code == 0
        verdict = json.loads(out.with_name("gate.verdict.json").read_text())
        # Forced errored (surfaced), production ran (pass integrity), and the
        # production report file survived.
        assert verdict["tracks"]["forced"]["state"] == TRACK_ERROR
        assert verdict["tracks"]["production"]["state"] == TRACK_PASS
        assert verdict["passed"] is False
        # Production's own report + manifest were written despite forced error.
        assert verdict["production_report"]
        assert Path(verdict["production_report"]).exists()
        assert Path(verdict["production_manifest"]).exists()

    def test_production_error_survives_forced_output(self, tmp_path, monkeypatch):
        """If production raises but forced succeeds, forced's report survives
        and production's error is surfaced (never silently dropped)."""
        module = _load_ci_tool()
        monkeypatch.setenv("MM_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("MM_LLM_MODEL", "fake-model")
        monkeypatch.delenv("MM_LLM_BASE_URL", raising=False)

        import mirror_memory.bench.run as run_mod
        monkeypatch.setattr(run_mod, "_build_extractor", lambda *a, **k: _WorkingLLM())
        monkeypatch.setattr(run_mod, "_build_answerer", lambda *a, **k: (lambda q, c: ""))

        real_run = module.run_statebench_v1_1

        def _run(**kwargs):
            mode = kwargs.get("extraction_mode", "deterministic")
            if mode == "production":
                return real_run(**{**kwargs, "llm_client": _FailingLLM()})
            if mode == "forced":
                return real_run(**{**kwargs, "llm_client": _WorkingLLM()})
            return real_run(**kwargs)

        monkeypatch.setattr(module, "run_statebench_v1_1", _run)

        out = tmp_path / "gate.json"
        code = module.main(["--output", str(out), "--allow-llm", "--max-cases", str(_MAX_CASES)])
        assert code == 0
        verdict = json.loads(out.with_name("gate.verdict.json").read_text())
        # Production raises -> runner raises on failed K2 -> surfaced as error,
        # never silently dropped.
        assert verdict["tracks"]["production"]["state"] == TRACK_ERROR
        # Forced succeeded: the gate subject is the real forced report, and its
        # files survived even though production errored.
        assert verdict["extraction_mode"] == "forced"
        assert verdict["forced_report"]
        assert Path(verdict["forced_report"]).exists()
        assert Path(verdict["forced_manifest"]).exists()
        # The forced subject passed its forced-mode / every-turn health checks.
        by_check = {c["check"]: c for c in verdict["checks"]}
        assert by_check["forced_mode"]["passed"] is True
        assert by_check["k2_attempted_every_turn"]["passed"] is True
        assert verdict["passed"] is False


# ===========================================================================
# §3 A — no key: forced AND production are EACH explicitly not_run
# ===========================================================================

class TestNoKeyBothTracksNotRun:
    def test_forced_and_production_are_separate_not_run(self, tmp_path, monkeypatch):
        """With no API key, forced and production each carry their OWN not_run
        verdict -- one must not stand in for the other (A)."""
        module = _load_ci_tool()
        monkeypatch.delenv("MM_LLM_API_KEY", raising=False)
        out = tmp_path / "gate.json"
        code = module.main(["--output", str(out), "--max-cases", str(_MAX_CASES)])
        assert code == 0

        verdict = json.loads(out.with_name("gate.verdict.json").read_text())
        tracks = verdict["tracks"]
        assert tracks["forced"]["state"] == TRACK_NOT_RUN
        assert tracks["production"]["state"] == TRACK_NOT_RUN
        assert tracks["answer"]["state"] == TRACK_NOT_RUN
        # Each is its own entry: distinct reason keys are present for both.
        assert "forced" in tracks and "production" in tracks
        # Neither is a pass.
        assert verdict["passed"] is False

        summary = out.with_name("gate.summary.md").read_text()
        assert summary.splitlines()[0].startswith("FAIL:")
        # Both not_run rows appear in the summary track table.
        assert "| forced |" in summary and "| production |" in summary
        forced_row = next(l for l in summary.splitlines() if l.startswith("| forced |"))
        prod_row = next(l for l in summary.splitlines() if l.startswith("| production |"))
        assert "not_run" in forced_row
        assert "not_run" in prod_row

        # The verdict self-reports false and no forced/production file pair is
        # fabricated (they genuinely never ran).
        assert verdict["forced_report"] == ""
        assert verdict["production_report"] == ""


# ===========================================================================
# §3 B — client-construction failure vs track-run failure are distinct
# ===========================================================================

class TestConstructionVsRunFailure:
    def test_construction_failure_is_error_with_init_phase(
        self, tmp_path, monkeypatch,
    ):
        """A client-construction failure marks the LLM tracks `error` naming
        client construction/init -- never not_run, never pass (B)."""
        module = _load_ci_tool()
        monkeypatch.setenv("MM_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("MM_LLM_MODEL", "fake-model")
        monkeypatch.delenv("MM_LLM_BASE_URL", raising=False)

        import mirror_memory.bench.run as run_mod

        def _boom(*a, **k):
            raise RuntimeError("bad credentials: cannot init client")

        monkeypatch.setattr(run_mod, "_build_extractor", _boom)

        out = tmp_path / "gate.json"
        code = module.main(["--output", str(out), "--allow-llm", "--max-cases", str(_MAX_CASES)])
        assert code == 0
        verdict = json.loads(out.with_name("gate.verdict.json").read_text())
        tracks = verdict["tracks"]
        # Every LLM-owned track is an `error` (construction), never not_run.
        for name in ("forced", "production", "answer"):
            assert tracks[name]["state"] == TRACK_ERROR, (name, tracks[name])
        # The reason explicitly names the construction/init phase.
        assert "client construction" in tracks["forced"]["reason"].lower() \
            or "client_init" in tracks["forced"]["reason"].lower() \
            or "init" in tracks["forced"]["reason"].lower()
        assert "not_run" not in tracks["forced"]["reason"].lower()
        assert verdict["passed"] is False
        summary = out.with_name("gate.summary.md").read_text()
        assert summary.splitlines()[0].startswith("FAIL:")
        assert "Infrastructure error" in summary or "error" in summary.lower()

    def test_run_failure_is_error_with_run_phase(self, tmp_path, monkeypatch):
        """A track that starts then raises is `error` naming a run failure --
        distinct from a construction failure (B)."""
        module = _load_ci_tool()
        monkeypatch.setenv("MM_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("MM_LLM_MODEL", "fake-model")
        monkeypatch.delenv("MM_LLM_BASE_URL", raising=False)

        import mirror_memory.bench.run as run_mod
        # Construction succeeds (working client built), the RUN fails.
        monkeypatch.setattr(run_mod, "_build_extractor", lambda *a, **k: _FailingLLM())
        monkeypatch.setattr(run_mod, "_build_answerer", lambda *a, **k: (lambda q, c: ""))

        out = tmp_path / "gate.json"
        code = module.main(["--output", str(out), "--allow-llm", "--max-cases", str(_MAX_CASES)])
        assert code == 0
        verdict = json.loads(out.with_name("gate.verdict.json").read_text())
        tracks = verdict["tracks"]
        assert tracks["forced"]["state"] == TRACK_ERROR
        # Forced ran then its every-K2 call failed -> run-failure reason.
        reason = tracks["forced"]["reason"].lower()
        assert "run failed" in reason
        # It must NOT be labelled a construction failure.
        assert "client construction" not in reason


# ===========================================================================
# §3 D — verdict / summary / files must be mutually consistent
# ===========================================================================

class TestArtifactConsistency:
    def test_success_verdict_summary_files_consistent(self, tmp_path, monkeypatch):
        """A successful forced+production run: verdict four-state, summary first
        line + checks, and each written file's dataset_version / manifest must
        agree (D)."""
        module = _load_ci_tool()
        monkeypatch.setenv("MM_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("MM_LLM_MODEL", "fake-model")
        monkeypatch.delenv("MM_LLM_BASE_URL", raising=False)

        import mirror_memory.bench.run as run_mod
        monkeypatch.setattr(run_mod, "_build_extractor", lambda *a, **k: _WorkingLLM())
        monkeypatch.setattr(run_mod, "_build_answerer", lambda *a, **k: (lambda q, c: ""))

        out = tmp_path / "gate.json"
        code = module.main(["--output", str(out), "--allow-llm", "--max-cases", str(_MAX_CASES)])
        assert code == 0

        verdict = json.loads(out.with_name("gate.verdict.json").read_text())
        summary = out.with_name("gate.summary.md").read_text()
        vversion = verdict["dataset_version"]

        # --- verdict <-> summary first line consistency ---------------------
        first = summary.splitlines()[0]
        assert first.startswith("PASS:" if verdict["passed"] else "FAIL:")

        # --- verdict four-state is a valid state for every track ------------
        valid = {"pass", "fail", "not_run", "error"}
        for name, v in verdict["tracks"].items():
            assert v["state"] in valid, (name, v)

        # --- each written file's dataset_version matches the verdict --------
        assert verdict["forced_report"] and verdict["production_report"]
        forced_rep = json.loads(Path(verdict["forced_report"]).read_text())
        prod_rep = json.loads(Path(verdict["production_report"]).read_text())
        assert forced_rep["dataset_version"] == vversion
        assert prod_rep["dataset_version"] == vversion
        assert forced_rep["extraction_mode"] == "forced"
        assert prod_rep["extraction_mode"] == "production"

        # --- each manifest carries provenance and matches its mode ----------
        for rep_path, man_path, mode in (
            (verdict["forced_report"], verdict["forced_manifest"], "forced"),
            (verdict["production_report"], verdict["production_manifest"], "production"),
        ):
            man = json.loads(Path(man_path).read_text())
            assert man["commit"]
            assert man["dataset_sha256"]
            assert man["extractor_model"] == "fake-working-model"
            assert man["config_hashes"]
            assert mode in man["notes"]
            # Manifest's dataset hash matches the report's dataset hash prefix.
            rep = json.loads(Path(rep_path).read_text())
            assert man["dataset_sha256"] == rep["dataset_sha256"][:16]

        # --- per-check table exists and every check has a pass column -------
        assert "| check | value | threshold | pass |" in summary
        # verdict checks and summary checks tell the same story for forced mode
        by_check = {c["check"]: c for c in verdict["checks"]}
        assert "forced_mode" in by_check and by_check["forced_mode"]["passed"] is True

    def test_production_error_preserves_forced_files_and_consistency(
        self, tmp_path, monkeypatch,
    ):
        """When production errors, forced's report+manifest are complete and the
        verdict/summary stay mutually consistent (C + D)."""
        module = _load_ci_tool()
        monkeypatch.setenv("MM_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("MM_LLM_MODEL", "fake-model")
        monkeypatch.delenv("MM_LLM_BASE_URL", raising=False)

        import mirror_memory.bench.run as run_mod
        monkeypatch.setattr(run_mod, "_build_extractor", lambda *a, **k: _WorkingLLM())
        monkeypatch.setattr(run_mod, "_build_answerer", lambda *a, **k: (lambda q, c: ""))

        real_run = module.run_statebench_v1_1

        def _run(**kwargs):
            mode = kwargs.get("extraction_mode", "deterministic")
            if mode == "production":
                return real_run(**{**kwargs, "llm_client": _FailingLLM()})
            if mode == "forced":
                return real_run(**{**kwargs, "llm_client": _WorkingLLM()})
            return real_run(**kwargs)

        monkeypatch.setattr(module, "run_statebench_v1_1", _run)

        out = tmp_path / "gate.json"
        code = module.main(["--output", str(out), "--allow-llm", "--max-cases", str(_MAX_CASES)])
        assert code == 0
        verdict = json.loads(out.with_name("gate.verdict.json").read_text())
        summary = out.with_name("gate.summary.md").read_text()

        # Production error is visible; forced succeeded and files are complete.
        assert verdict["tracks"]["production"]["state"] == TRACK_ERROR
        assert verdict["forced_report"] and Path(verdict["forced_report"]).exists()
        assert verdict["forced_manifest"] and Path(verdict["forced_manifest"]).exists()
        forced_rep = json.loads(Path(verdict["forced_report"]).read_text())
        forced_man = json.loads(Path(verdict["forced_manifest"]).read_text())
        assert forced_rep["dataset_version"] == verdict["dataset_version"]
        assert forced_man["commit"]
        assert forced_rep["extraction_mode"] == "forced"
        # First line reflects the failure and verdict.passed agrees.
        assert summary.splitlines()[0].startswith("FAIL:")
        assert verdict["passed"] is False
