"""Dataset and scoring contract tests for StateBench v1.1."""

from __future__ import annotations

import hashlib
from collections import Counter

from mirror_memory.bench.gate import check_release_gate, compare_forced_production
from mirror_memory.bench.statebench import load_statebench
from mirror_memory.bench.statebench_v1_1 import (
    STATEBENCH_V11_CASE_COUNT,
    STATEBENCH_V11_CATEGORIES,
    STATEBENCH_V11_SPLITS,
    STATEBENCH_V11_TRACKS,
    TRANSITION_RUNTIME_CONTRACT,
    load_statebench_v1_1,
    load_statebench_v1_1_payload,
)
from mirror_memory.bench.statebench_v1_1_run import main as statebench_v11_main
from mirror_memory.bench.statebench_v1_1_runner import (
    _score_state,
    _score_text,
    run_full_context_answer_baseline,
    run_statebench_v1_1,
    write_statebench_v1_1_report,
)
from mirror_memory.config.loader import load_config

EXPECTED_CATEGORIES = {
    "replacement": 30,
    "round_trip": 30,
    "multi_value": 30,
    "contradiction": 30,
    "temporal_event": 30,
    "preference_evolution": 30,
    "stale_state": 20,
}


def test_v1_dataset_is_frozen():
    path = "src/mirror_memory/bench/data/statebench_v1.json"
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert digest == "b4cb6d2032bda970c268a124e379c341824b3d52ed6302214c7b059c64dd8fe3"


def test_v11_has_200_stable_cases_and_declared_tracks():
    payload = load_statebench_v1_1_payload()
    cases = load_statebench_v1_1()
    legacy = load_statebench()

    assert payload["tracks"] == ["state", "recall", "answer"]
    assert set(payload["tracks"]) == STATEBENCH_V11_TRACKS
    assert len(cases) == STATEBENCH_V11_CASE_COUNT == 200
    assert {case.case_id for case in cases} == {case.case_id for case in legacy}
    assert Counter(case.category for case in cases) == EXPECTED_CATEGORIES
    assert {case.category for case in cases} == STATEBENCH_V11_CATEGORIES


def test_v11_uses_public_evaluation_not_held_out_wording():
    payload = load_statebench_v1_1_payload()
    cases = load_statebench_v1_1()

    assert {case.split for case in cases} == STATEBENCH_V11_SPLITS
    assert Counter(case.split for case in cases) == {
        "development": 140,
        "public_evaluation": 60,
    }
    assert "not a private held-out set" in payload["split_semantics"][
        "public_evaluation"
    ].casefold()


def test_every_case_has_three_nonempty_contracts():
    for case in load_statebench_v1_1():
        assert case.transition_type
        assert case.query_mode in {"current", "history", "all_occurrences"}
        assert case.state_contract.required
        assert (
            case.recall_contract.required_all
            or case.recall_contract.required_any
        )
        assert (
            case.answer_contract.required_all
            or case.answer_contract.required_any
        )


def test_every_benchmark_transition_has_a_runtime_contract():
    labels = {case.transition_type for case in load_statebench_v1_1()}
    assert labels == set(TRANSITION_RUNTIME_CONTRACT)
    assert TRANSITION_RUNTIME_CONTRACT["current_value_revival"] == "REVIVE"
    assert TRANSITION_RUNTIME_CONTRACT["goal_cancellation"] == "END_CURRENT"
    assert TRANSITION_RUNTIME_CONTRACT["query_time_selection"] == "READ_AT_TIME"


def test_historical_recall_may_include_current_state_but_answer_may_not():
    history_cases = [
        case for case in load_statebench_v1_1() if case.query_mode == "history"
    ]
    assert len(history_cases) == 5
    for case in history_cases:
        assert not case.recall_contract.forbidden
        assert case.answer_contract.forbidden


def test_temporal_events_assert_one_structural_occurrence_per_turn():
    cases = load_statebench_v1_1({"temporal_event"})
    for case in cases:
        assert len(case.state_contract.required) == len(case.turns)
        assert all(item.label.startswith("occurrence:") for item in case.state_contract.required)


def test_state_and_text_tracks_score_independently():
    case = load_statebench_v1_1({"replacement"})[0]
    beliefs = [
        {
            "status": "superseded",
            "claim_text": "User lived in Shanghai",
            "object": "Shanghai",
            "predicate": "lives_in",
            "valid_to": "2026-05-01T00:00:00+00:00",
            "value": {},
        },
        {
            "status": "active",
            "claim_text": "User lives in Berlin",
            "object": "Berlin",
            "predicate": "lives_in",
            "valid_to": None,
            "value": {},
        },
    ]

    assert _score_state(case, beliefs).passed
    assert _score_text(case.recall_contract, "User lives in Berlin").passed
    assert not _score_text(
        case.recall_contract, "User lived in Shanghai and lives in Berlin"
    ).passed


def test_raw_context_baseline_is_scored_only_after_answerer():
    calls: list[tuple[str, str]] = []

    def answerer(question: str, context: str) -> str:
        calls.append((question, context))
        return "Berlin"

    report = run_full_context_answer_baseline(
        answerer=answerer,
        categories={"replacement"},
        max_cases=1,
    )
    assert report["total"] == 1
    assert calls and "Shanghai" in calls[0][1]
    assert report["cases"][0]["answer"] == "Berlin"


def test_deterministic_smoke_run_and_manifest(tmp_path):
    report = run_statebench_v1_1(
        categories={"replacement"},
        config=load_config("config"),
        extraction_mode="deterministic",
        max_cases=1,
    )
    assert report.tracks["state"]["total"] == 1
    assert report.tracks["recall"]["total"] == 1
    assert report.tracks["answer"]["total"] == 0

    output, manifest = write_statebench_v1_1_report(
        report, tmp_path / "statebench-v1.1.json"
    )
    assert output.exists()
    assert manifest.exists()

    gate = check_release_gate(report, manifest_path=manifest)
    assert not gate.passed
    assert any("forced run required" in reason for reason in gate.failures)
    assert any("answer not evaluated" in reason for reason in gate.failures)


class _FailingLLM:
    """An LLM client whose every call raises, like an exhausted quota."""

    model = "failing-model"

    def generate(self, *, system_prompt, payload_text, fallback=None):
        raise RuntimeError("429 rate limit")


class _SwallowingLLM:
    """A client that answers with the fallback instead of raising.

    This is what ``OpenAILLM`` does: it catches the API error and returns the
    caller's fallback string.  The extractor must still count it as a failed
    call, or the run measures the fallback and reports it as the model.
    """

    model = "swallowing-model"

    def __init__(self):
        self.calls = 0
        self.call_failures = 0

    def generate(self, *, system_prompt, payload_text, fallback=None):
        self.calls += 1
        self.call_failures += 1
        return fallback() if fallback else ""


def test_forced_run_refuses_to_report_when_every_k2_call_fails():
    """A rate-limited forced run must not pass as a model measurement.

    K2 failures are swallowed per turn so the turn still yields K1 claims,
    which makes a fully-failed run numerically identical to a deterministic
    one.  The runner has to refuse it instead of labelling K1 numbers as a
    model result.
    """
    import pytest as _pytest

    from mirror_memory.bench.statebench_v1_1_runner import run_statebench_v1_1

    for client in (_FailingLLM(), _SwallowingLLM()):
        with _pytest.raises(RuntimeError, match="K2 calls failed"):
            run_statebench_v1_1(
                categories={"replacement"},
                config=load_config("config"),
                extraction_mode="forced",
                llm_client=client,
                max_cases=1,
            )


def test_forced_run_reports_k2_health_when_calls_succeed():
    """A healthy forced run records how many K2 calls it made."""
    from mirror_memory.bench.statebench_v1_1_runner import run_statebench_v1_1

    class _WorkingLLM:
        model = "working-model"

        def generate(self, *, system_prompt, payload_text, fallback=None):
            return '{"claims": [], "subject": "user", "context_tags": []}'

    report = run_statebench_v1_1(
        categories={"replacement"},
        config=load_config("config"),
        extraction_mode="forced",
        llm_client=_WorkingLLM(),
        max_cases=1,
    )
    assert report.extraction_health["k2_attempts"] >= 1
    assert report.extraction_health["k2_failures"] == 0


def test_case_id_filter_selects_the_same_case_for_paired_runs():
    selected = {"sb-replacement-001", "sb-contradiction-011"}
    report = run_statebench_v1_1(
        case_ids=selected,
        extraction_mode="deterministic",
    )
    assert {case.case_id for case in report.cases} == selected

    import pytest

    with pytest.raises(ValueError, match="unknown or filtered"):
        run_statebench_v1_1(case_ids={"does-not-exist"}, max_cases=1)


def _v11_gate_report(mode, score, *, answer=0.9, case_id="case-1", health=None):
    ids = [case.case_id for case in load_statebench_v1_1()]
    if case_id != "case-1":
        ids[0] = case_id
    if health is None:
        health = {
            "k2_attempts": 486,
            "k2_failures": 0,
            "k2_empty_responses": 0,
            "k2_parse_failures": 0,
            "k2_schema_rejections": 0,
            "dataset_turns": 486,
        }
    return {
        "extraction_mode": mode,
        "dataset_sha256": "same-dataset",
        "extractor_model": "same-model",
        "extraction_health": health,
        "cases": [{"case_id": selected} for selected in ids],
        "tracks": {
            "state": {
                "total": STATEBENCH_V11_CASE_COUNT,
                "score": score,
                "by_category": {
                    category: {"score": score} for category in STATEBENCH_V11_CATEGORIES
                },
            },
            "recall": {"total": STATEBENCH_V11_CASE_COUNT, "score": 0.9},
            "answer": {"total": STATEBENCH_V11_CASE_COUNT, "score": answer},
        },
    }


def test_v11_gate_checks_real_track_shape_and_production_gap(tmp_path):
    manifest = tmp_path / "forced.manifest.json"
    manifest.write_text("{}")
    forced = _v11_gate_report("forced", 0.93)
    production = _v11_gate_report("production", 0.90)

    result = check_release_gate(
        forced, manifest_path=manifest, production_report=production,
    )
    assert result.passed


def test_v11_gate_rejects_a_degraded_extraction_run(tmp_path):
    """Scores from a run whose K2 calls partly failed are not model numbers.

    A fully-failed run is refused by the runner; a partly-failed one still
    writes a report, so the gate has to refuse it -- otherwise the fallback's
    K1 numbers are averaged into the model's score.
    """
    manifest = tmp_path / "forced.manifest.json"
    manifest.write_text("{}")
    health = {
        "k2_attempts": 486,
        "k2_failures": 12,
        "k2_empty_responses": 0,
        "k2_parse_failures": 0,
        "k2_schema_rejections": 0,
        "dataset_turns": 486,
    }
    forced = _v11_gate_report("forced", 0.93, health=health)
    production = _v11_gate_report("production", 0.90)

    result = check_release_gate(
        forced, manifest_path=manifest, production_report=production,
    )
    assert not result.passed
    assert any("12 failed K2 calls" in reason for reason in result.failures)


def test_v11_gate_rejects_a_run_that_skipped_turns(tmp_path):
    """Forced means every turn got an extraction attempt."""
    manifest = tmp_path / "forced.manifest.json"
    manifest.write_text("{}")
    health = {
        "k2_attempts": 480,
        "k2_failures": 0,
        "k2_empty_responses": 0,
        "k2_parse_failures": 0,
        "k2_schema_rejections": 0,
        "dataset_turns": 486,
    }
    forced = _v11_gate_report("forced", 0.93, health=health)
    production = _v11_gate_report("production", 0.90)

    result = check_release_gate(
        forced, manifest_path=manifest, production_report=production,
    )
    assert not result.passed
    assert any("K2 attempted 480 of 486 turns" in reason for reason in result.failures)
    assert any(check["check"] == "production_gap" and check["value"] == 0.03
               for check in result.checks)

    low_production = _v11_gate_report("production", 0.70)
    gap = compare_forced_production(forced, low_production)
    assert gap["gap"] == 0.23
    assert gap["gate"] is False


def test_v11_pair_rejects_different_cases(tmp_path):
    manifest = tmp_path / "forced.manifest.json"
    manifest.write_text("{}")
    forced = _v11_gate_report("forced", 0.93)
    production = _v11_gate_report("production", 0.90, case_id="case-2")

    result = check_release_gate(
        forced, manifest_path=manifest, production_report=production,
    )
    assert not result.passed
    assert any("case sets or order differ" in reason for reason in result.failures)


def test_v11_gate_rejects_pilot_subset_even_when_scores_pass(tmp_path):
    manifest = tmp_path / "forced.manifest.json"
    manifest.write_text("{}")
    forced = _v11_gate_report("forced", 0.95)
    production = _v11_gate_report("production", 0.95)
    forced["cases"] = forced["cases"][:2]
    production["cases"] = production["cases"][:2]

    result = check_release_gate(
        forced, manifest_path=manifest, production_report=production,
    )
    assert not result.passed
    assert any("full v1.1 case coverage required" in reason for reason in result.failures)


def test_v11_gate_rejects_partial_answer_coverage(tmp_path):
    manifest = tmp_path / "forced.manifest.json"
    manifest.write_text("{}")
    forced = _v11_gate_report("forced", 0.95, answer=0.95)
    production = _v11_gate_report("production", 0.95, answer=0.95)
    forced["tracks"]["answer"]["total"] = 1

    result = check_release_gate(
        forced, manifest_path=manifest, production_report=production,
    )
    assert not result.passed
    assert any("answer evaluated 1/200 cases" in reason for reason in result.failures)


def test_v11_cli_writes_report_before_reporting_unmet_gate(tmp_path, capsys):
    output = tmp_path / "deterministic.json"
    code = statebench_v11_main([
        "--mode", "deterministic", "--max-cases", "1",
        "--output", str(output), "--gate",
    ])
    assert code == 1
    assert output.exists()
    assert output.with_name("deterministic.manifest.json").exists()
    assert "forced run required" in capsys.readouterr().out


def test_zero_keyword_threshold_is_a_valid_explicit_benchmark_config():
    config = load_config("config")
    extraction = config.extraction.model_copy(update={"llm_min_keyword_hits": 0})
    validated = type(config.extraction).model_validate(extraction.model_dump())
    assert validated.llm_min_keyword_hits == 0
