"""Build the static StateBench v1.1 dataset from the frozen v1.0 corpus.

v1.1 keeps the scenarios and case ids stable so score changes can be
attributed to the evaluation contract.  It replaces the single rendered-text
assertion with independent state, recall, and answer contracts.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/mirror_memory/bench/data/statebench_v1.json"
TARGET = ROOT / "src/mirror_memory/bench/data/statebench_v1_1.json"


CONTRADICTION_TARGETS = [
    "coffee", "hiking", "jazz", "spicy food", "running",
    "horror movies", "crowded concerts", "early meetings", "flying",
    "public speaking", "learn Japanese", "move to Canada",
    "run a marathon", "management role", "buy a house", "study law",
    "visit Brazil", "start a podcast", "red bicycle", "Turin cabin",
    "film camera", "blue guitar", "old laptop", "Korean", "Rust",
    "certified diver", "Italian", "Maya", "Theo", "Professor Singh",
]

EVOLUTION_TARGETS = [
    "coffee", "running", "painting", "jazz", "hiking", "spicy food",
    "travel", "public speaking", "gardening", "cooking", "cycling", "tea",
    "museums", "dogs", "early mornings", "fiction", "swimming",
    "remote work", "documentaries", "language", "marathon", "Japanese",
    "novel", "Canada", "manager", "astronomy", "house", "podcast",
    "Iceland", "shelter",
]


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _contains(text: str, value: str) -> bool:
    return _norm(value) in _norm(text)


def _text_contract(
    *,
    required_all: list[str] | None = None,
    required_any: list[list[str]] | None = None,
    forbidden: list[str] | None = None,
) -> dict:
    return {
        "required_all": required_all or [],
        "required_any": required_any or [],
        "forbidden": forbidden or [],
    }


def _atom(
    label: str,
    *,
    statuses: list[str] | None = None,
    surface_all: list[str] | None = None,
    surface_any: list[str] | None = None,
    polarity: str = "any",
    temporal_mode: str = "any",
    min_count: int = 1,
    max_count: int | None = None,
) -> dict:
    atom = {
        "label": label,
        "statuses": statuses or [],
        "surface_all": surface_all or [],
        "surface_any": surface_any or [],
        "polarity": polarity,
        "temporal_mode": temporal_mode,
        "min_count": min_count,
    }
    if max_count is not None:
        atom["max_count"] = max_count
    return atom


def _state_contract(required: list[dict], forbidden: list[dict] | None = None) -> dict:
    return {"required": required, "forbidden": forbidden or []}


def _transition_type(case: dict, index: int) -> str:
    category = case["category"]
    tags = set(case["tags"])
    if category == "contradiction":
        if "skill_correction" in tags:
            return "explicit_correction"
        if "possession" in tags:
            return "state_termination"
        if "goal" in tags:
            return "goal_cancellation"
        if "relationship" in tags:
            return "relationship_termination"
        return "preference_reversal"
    if category == "preference_evolution":
        if index >= 20:
            return "goal_reactivation"
        if index >= 10:
            return "preference_reversal"
        return "preference_revival"
    return {
        "replacement": "current_value_replacement",
        "round_trip": "current_value_revival",
        "multi_value": "coexisting_values",
        "temporal_event": "distinct_occurrences",
        "stale_state": "query_time_selection",
    }[category]


def _convert_case(case: dict, category_index: int) -> dict:
    category = case["category"]
    current = list(case.get("expect_current") or [])
    history = list(case.get("expect_history") or [])
    alternatives = [list(group) for group in case.get("expect_any") or []]
    stale = list(case.get("forbidden") or [])
    query_mode = "history" if history else (
        "all_occurrences" if category in {"multi_value", "temporal_event"} else "current"
    )

    required_state: list[dict] = []
    forbidden_state: list[dict] = []

    if category in {"replacement", "round_trip"}:
        for value in current:
            required_state.append(_atom(
                f"current:{value}", statuses=["active"], surface_all=[value],
                polarity="positive", temporal_mode="current",
            ))
        for value in stale:
            required_state.append(_atom(
                f"history:{value}", statuses=["superseded"], surface_all=[value],
                polarity="positive", temporal_mode="historical",
            ))
            forbidden_state.append(_atom(
                f"not-current:{value}", statuses=["active"], surface_all=[value],
                polarity="positive", temporal_mode="current",
            ))
        recall = _text_contract(required_all=current, forbidden=stale)
        answer = _text_contract(required_all=current, forbidden=stale)

    elif category == "multi_value":
        for value in current:
            required_state.append(_atom(
                f"coexists:{value}", statuses=["active"], surface_all=[value],
                polarity="positive", temporal_mode="current",
            ))
        recall = _text_contract(required_all=current)
        answer = _text_contract(required_all=current)

    elif category == "contradiction":
        target = CONTRADICTION_TARGETS[category_index]
        negative_surfaces = alternatives[0]
        required_state.append(_atom(
            f"negative-current:{target}", statuses=["active"],
            surface_any=negative_surfaces, polarity="negative", temporal_mode="current",
        ))
        forbidden_state.append(_atom(
            f"positive-no-longer-current:{target}", statuses=["active"],
            surface_all=[target], polarity="positive", temporal_mode="current",
        ))
        recall = _text_contract(required_any=alternatives)
        answer = _text_contract(required_any=alternatives)

    elif category == "temporal_event":
        for turn_index, turn in enumerate(case["turns"]):
            matched = [value for value in current if _contains(turn, value)]
            if not matched:
                raise ValueError(f"{case['case_id']}: event turn has no observable assertion")
            required_state.append(_atom(
                f"occurrence:{turn_index + 1}", statuses=["active"],
                surface_all=matched, polarity="positive", temporal_mode="any",
            ))
        recall = _text_contract(required_all=current)
        answer = _text_contract(required_all=current)

    elif category == "preference_evolution":
        target = EVOLUTION_TARGETS[category_index]
        required_state.append(_atom(
            f"restored-current:{target}", statuses=["active"],
            surface_any=alternatives[0], polarity="positive", temporal_mode="current",
        ))
        for value in stale:
            forbidden_state.append(_atom(
                f"intermediate-no-longer-current:{value}", statuses=["active"],
                surface_all=[value], temporal_mode="current",
            ))
        recall = _text_contract(required_any=alternatives, forbidden=stale)
        answer = _text_contract(required_any=alternatives, forbidden=stale)

    elif category == "stale_state":
        if query_mode == "history":
            for value in history:
                required_state.append(_atom(
                    f"history:{value}", statuses=["superseded"], surface_all=[value],
                    polarity="positive", temporal_mode="historical",
                ))
            for value in stale:
                required_state.append(_atom(
                    f"current:{value}", statuses=["active"], surface_all=[value],
                    polarity="positive", temporal_mode="current",
                ))
            # A memory block may contain both periods. Only the final answer is
            # required to exclude the current value for a historical question.
            recall = _text_contract(required_all=history)
            answer = _text_contract(required_all=history, forbidden=stale)
        else:
            for value in current:
                required_state.append(_atom(
                    f"current:{value}", statuses=["active"], surface_all=[value],
                    polarity="positive", temporal_mode="current",
                ))
            for value in stale:
                required_state.append(_atom(
                    f"history:{value}", statuses=["superseded"], surface_all=[value],
                    polarity="positive", temporal_mode="historical",
                ))
                forbidden_state.append(_atom(
                    f"not-current:{value}", statuses=["active"], surface_all=[value],
                    polarity="positive", temporal_mode="current",
                ))
            recall = _text_contract(required_all=current, forbidden=stale)
            answer = _text_contract(required_all=current, forbidden=stale)
    else:
        raise ValueError(f"unsupported category: {category}")

    return {
        "case_id": case["case_id"],
        "category": category,
        "transition_type": _transition_type(case, category_index),
        "split": "public_evaluation" if case["split"] == "evaluation" else case["split"],
        "difficulty": case["difficulty"],
        "query_mode": query_mode,
        "turns": case["turns"],
        "question": case["question"],
        "state_contract": _state_contract(required_state, forbidden_state),
        "recall_contract": recall,
        "answer_contract": answer,
        "tags": case["tags"],
        "note": case["note"],
    }


def build() -> dict:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    positions: Counter[str] = Counter()
    cases: list[dict] = []
    for case in source["cases"]:
        category = case["category"]
        index = positions[category]
        positions[category] += 1
        cases.append(_convert_case(case, index))

    category_counts = Counter(case["category"] for case in cases)
    split_counts = Counter(case["split"] for case in cases)
    return {
        "version": "1.1.0",
        "derived_from": "1.0.0",
        "case_count": len(cases),
        "language": "en",
        "tracks": ["state", "recall", "answer"],
        "split_semantics": {
            "development": "Public cases for diagnosis and iteration.",
            "public_evaluation": (
                "Public reporting boundary; reproducible but not a private held-out set."
            ),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "cases": cases,
    }


def main() -> None:
    payload = build()
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {payload['case_count']} cases to {TARGET}")


if __name__ == "__main__":
    main()
