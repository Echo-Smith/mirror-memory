"""Structural and scoring-contract tests for StateBench v1.0."""

from collections import Counter

import pytest

from mirror_memory.bench.statebench import (
    STATEBENCH_CASE_COUNT,
    STATEBENCH_CATEGORIES,
    STATEBENCH_SPLITS,
    StateCase,
    category_counts,
    load_statebench,
    load_statebench_payload,
    split_counts,
)
from mirror_memory.bench.statebench_runner import (
    _score,
    run_full_context_baseline,
)

EXPECTED_CATEGORY_COUNTS = {
    "replacement": 30,
    "round_trip": 30,
    "multi_value": 30,
    "contradiction": 30,
    "temporal_event": 30,
    "preference_evolution": 30,
    "stale_state": 20,
}
EXPECTED_SPLIT_COUNTS = {"development": 140, "evaluation": 60}


def test_statebench_has_exactly_200_unique_cases():
    cases = load_statebench()

    assert len(cases) == STATEBENCH_CASE_COUNT == 200
    assert len({case.case_id for case in cases}) == 200


def test_statebench_category_distribution_is_declared_and_balanced():
    cases = load_statebench()

    assert set(category_counts(cases)) == STATEBENCH_CATEGORIES
    assert category_counts(cases) == EXPECTED_CATEGORY_COUNTS


def test_statebench_has_public_development_and_evaluation_splits():
    cases = load_statebench()

    assert set(split_counts(cases)) == STATEBENCH_SPLITS
    assert split_counts(cases) == EXPECTED_SPLIT_COUNTS
    for category, expected_total in EXPECTED_CATEGORY_COUNTS.items():
        subset = [case for case in cases if case.category == category]
        assert len(subset) == expected_total
        assert {case.split for case in subset} == STATEBENCH_SPLITS


def test_dataset_envelope_matches_loaded_cases():
    payload = load_statebench_payload()
    cases = load_statebench()

    assert payload["case_count"] == len(cases)
    assert payload["category_counts"] == EXPECTED_CATEGORY_COUNTS
    assert payload["split_counts"] == EXPECTED_SPLIT_COUNTS


def test_every_case_has_a_question_turns_and_observable_assertion():
    for case in load_statebench():
        assert case.case_id
        assert case.category in STATEBENCH_CATEGORIES
        assert case.split in STATEBENCH_SPLITS
        assert case.difficulty in {"basic", "paraphrase", "distractor"}
        assert len(case.turns) >= 2
        assert all(turn.strip() for turn in case.turns)
        assert case.question.strip()
        assert (
            case.expect_current
            or case.expect_history
            or case.expect_any
            or case.forbidden
        ), case.case_id
        assert all(group for group in case.expect_any), case.case_id
        assert case.tags


def test_no_duplicate_scenarios():
    cases = load_statebench()
    scenarios = {(case.turns, case.question.casefold()) for case in cases}

    assert len(scenarios) == len(cases)


def test_no_required_literal_is_also_forbidden():
    for case in load_statebench():
        required = {
            value.casefold()
            for value in case.expect_current | case.expect_history
        }
        forbidden = {value.casefold() for value in case.forbidden}
        assert required.isdisjoint(forbidden), case.case_id


def test_difficulty_distribution_is_balanced():
    counts = Counter(case.difficulty for case in load_statebench())

    assert max(counts.values()) - min(counts.values()) <= 1
    assert set(counts) == {"basic", "paraphrase", "distractor"}


def test_loader_filters_category_and_split():
    cases = load_statebench(
        {"replacement"},
        splits={"evaluation"},
    )

    assert len(cases) == 9
    assert {case.category for case in cases} == {"replacement"}
    assert {case.split for case in cases} == {"evaluation"}


@pytest.mark.parametrize(
    "categories,splits",
    [
        ({"not_a_category"}, None),
        (None, {"secret"}),
    ],
)
def test_loader_rejects_unknown_filters(categories, splits):
    with pytest.raises(ValueError):
        load_statebench(categories, splits=splits)


def test_alternative_group_accepts_one_semantic_surface():
    case = StateCase(
        case_id="example",
        category="contradiction",
        turns=("I like coffee.", "I do not like coffee."),
        question="Do they like coffee?",
        expect_any=(frozenset({"do not like coffee", "dislike coffee"}),),
    )

    result = _score(case, "The user now dislikes coffee.")

    assert result.passed
    assert result.missing_any == []


def test_alternative_group_rejects_old_positive_state():
    case = StateCase(
        case_id="example",
        category="contradiction",
        turns=("I like coffee.", "I do not like coffee."),
        question="Do they like coffee?",
        expect_any=(frozenset({"do not like coffee", "dislike coffee"}),),
    )

    result = _score(case, "The user likes coffee.")

    assert not result.passed
    assert result.missing_any


def test_full_context_baseline_covers_all_200_cases():
    report = run_full_context_baseline()

    assert report.total == 200
    assert set(report.by_category) == STATEBENCH_CATEGORIES
    assert report.by_split["development"]["total"] == 140
    assert report.by_split["evaluation"]["total"] == 60


def test_required_alternatives_do_not_trigger_forbidden_assertions():
    """A valid expected phrase must be capable of passing the case."""
    from mirror_memory.bench.statebench_runner import _matches

    for case in load_statebench():
        required = list(case.expect_current | case.expect_history)
        required.extend(value for group in case.expect_any for value in group)
        for value in required:
            for forbidden in case.forbidden:
                assert not _matches(value, forbidden), (
                    case.case_id,
                    value,
                    forbidden,
                )
