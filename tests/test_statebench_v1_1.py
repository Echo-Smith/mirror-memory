"""Dataset and scoring contract tests for StateBench v1.1."""

from __future__ import annotations

import hashlib
from collections import Counter

from mirror_memory.bench.statebench import load_statebench
from mirror_memory.bench.statebench_v1_1 import (
    STATEBENCH_V11_CASE_COUNT,
    STATEBENCH_V11_CATEGORIES,
    STATEBENCH_V11_SPLITS,
    STATEBENCH_V11_TRACKS,
    load_statebench_v1_1,
    load_statebench_v1_1_payload,
)
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


def test_zero_keyword_threshold_is_a_valid_explicit_benchmark_config():
    config = load_config("config")
    extraction = config.extraction.model_copy(update={"llm_min_keyword_hits": 0})
    validated = type(config.extraction).model_validate(extraction.model_dump())
    assert validated.llm_min_keyword_hits == 0
