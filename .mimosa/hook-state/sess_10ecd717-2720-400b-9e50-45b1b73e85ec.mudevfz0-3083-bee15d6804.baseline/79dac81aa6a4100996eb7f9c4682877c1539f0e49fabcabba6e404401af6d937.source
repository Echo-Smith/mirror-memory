"""StateBench v1.1 dataset contract.

v1.1 separates three questions that v1.0 collapsed into one rendered-string
score:

1. Did Mirror persist the right state and lifecycle?
2. Did recall surface the state needed by the query?
3. Did an answerer use that context correctly?

The v1.0 dataset remains packaged and loadable through ``statebench.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

STATEBENCH_V11_VERSION = "1.1.0"
STATEBENCH_V11_CASE_COUNT = 200
STATEBENCH_V11_CATEGORIES = frozenset({
    "replacement",
    "round_trip",
    "multi_value",
    "contradiction",
    "temporal_event",
    "preference_evolution",
    "stale_state",
})
STATEBENCH_V11_SPLITS = frozenset({"development", "public_evaluation"})
STATEBENCH_V11_TRACKS = frozenset({"state", "recall", "answer"})


@dataclass(frozen=True)
class AtomAssertion:
    """A structural assertion over persisted belief rows."""

    label: str
    statuses: tuple[str, ...] = field(default_factory=tuple)
    predicates: tuple[str, ...] = field(default_factory=tuple)
    surface_all: tuple[str, ...] = field(default_factory=tuple)
    surface_any: tuple[str, ...] = field(default_factory=tuple)
    polarity: str = "any"
    temporal_mode: str = "any"
    min_count: int = 1
    max_count: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AtomAssertion:
        return cls(
            label=str(raw.get("label") or ""),
            statuses=tuple(str(value) for value in raw.get("statuses") or []),
            predicates=tuple(str(value) for value in raw.get("predicates") or []),
            surface_all=tuple(str(value) for value in raw.get("surface_all") or []),
            surface_any=tuple(str(value) for value in raw.get("surface_any") or []),
            polarity=str(raw.get("polarity") or "any"),
            temporal_mode=str(raw.get("temporal_mode") or "any"),
            min_count=int(raw.get("min_count", 1)),
            max_count=(
                int(raw["max_count"])
                if raw.get("max_count") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class StateContract:
    required: tuple[AtomAssertion, ...] = field(default_factory=tuple)
    forbidden: tuple[AtomAssertion, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StateContract:
        return cls(
            required=tuple(
                AtomAssertion.from_dict(item) for item in raw.get("required") or []
            ),
            forbidden=tuple(
                AtomAssertion.from_dict(item) for item in raw.get("forbidden") or []
            ),
        )


@dataclass(frozen=True)
class TextContract:
    required_all: tuple[str, ...] = field(default_factory=tuple)
    required_any: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    forbidden: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TextContract:
        return cls(
            required_all=tuple(str(value) for value in raw.get("required_all") or []),
            required_any=tuple(
                tuple(str(value) for value in group)
                for group in raw.get("required_any") or []
            ),
            forbidden=tuple(str(value) for value in raw.get("forbidden") or []),
        )


@dataclass(frozen=True)
class StateCaseV11:
    case_id: str
    category: str
    transition_type: str
    split: str
    difficulty: str
    query_mode: str
    turns: tuple[str, ...]
    question: str
    state_contract: StateContract
    recall_contract: TextContract
    answer_contract: TextContract
    tags: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StateCaseV11:
        return cls(
            case_id=str(raw.get("case_id") or ""),
            category=str(raw.get("category") or ""),
            transition_type=str(raw.get("transition_type") or ""),
            split=str(raw.get("split") or ""),
            difficulty=str(raw.get("difficulty") or "basic"),
            query_mode=str(raw.get("query_mode") or "current"),
            turns=tuple(str(turn) for turn in raw.get("turns") or []),
            question=str(raw.get("question") or ""),
            state_contract=StateContract.from_dict(raw.get("state_contract") or {}),
            recall_contract=TextContract.from_dict(raw.get("recall_contract") or {}),
            answer_contract=TextContract.from_dict(raw.get("answer_contract") or {}),
            tags=tuple(str(tag) for tag in raw.get("tags") or []),
            note=str(raw.get("note") or ""),
        )


def packaged_statebench_v1_1() -> resources.abc.Traversable:
    return resources.files("mirror_memory.bench.data").joinpath(
        "statebench_v1_1.json"
    )


def statebench_v1_1_path() -> Path:
    """Return the installed dataset path when backed by a real filesystem."""
    return Path(str(packaged_statebench_v1_1()))


def load_statebench_v1_1_payload(
    path: str | Path | None = None,
) -> dict[str, Any]:
    source = packaged_statebench_v1_1() if path is None else Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("StateBench v1.1 dataset must be a JSON object")
    if raw.get("version") != STATEBENCH_V11_VERSION:
        raise ValueError(
            f"unsupported StateBench version: {raw.get('version')!r}; "
            f"expected {STATEBENCH_V11_VERSION!r}"
        )
    if raw.get("case_count") != STATEBENCH_V11_CASE_COUNT:
        raise ValueError("StateBench v1.1 case_count must be 200")
    if set(raw.get("tracks") or []) != STATEBENCH_V11_TRACKS:
        raise ValueError("StateBench v1.1 must declare state, recall, and answer tracks")
    if not isinstance(raw.get("cases"), list):
        raise ValueError("StateBench v1.1 dataset must contain a cases list")
    return raw


def load_statebench_v1_1(
    categories: set[str] | None = None,
    *,
    splits: set[str] | None = None,
    path: str | Path | None = None,
) -> list[StateCaseV11]:
    unknown_categories = (categories or set()) - STATEBENCH_V11_CATEGORIES
    if unknown_categories:
        raise ValueError(
            f"unknown StateBench v1.1 categories: {sorted(unknown_categories)}"
        )
    unknown_splits = (splits or set()) - STATEBENCH_V11_SPLITS
    if unknown_splits:
        raise ValueError(
            f"unknown StateBench v1.1 splits: {sorted(unknown_splits)}"
        )

    payload = load_statebench_v1_1_payload(path)
    cases = [StateCaseV11.from_dict(item) for item in payload["cases"]]
    if categories is not None:
        cases = [case for case in cases if case.category in categories]
    if splits is not None:
        cases = [case for case in cases if case.split in splits]
    return cases


def counts(cases: Iterable[StateCaseV11], attribute: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for case in cases:
        value = str(getattr(case, attribute))
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


__all__ = [
    "AtomAssertion",
    "STATEBENCH_V11_CASE_COUNT",
    "STATEBENCH_V11_CATEGORIES",
    "STATEBENCH_V11_SPLITS",
    "STATEBENCH_V11_TRACKS",
    "STATEBENCH_V11_VERSION",
    "StateCaseV11",
    "StateContract",
    "TextContract",
    "counts",
    "load_statebench_v1_1",
    "load_statebench_v1_1_payload",
    "statebench_v1_1_path",
]
