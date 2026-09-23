"""StateBench v1.0 -- 200 cases for evolving memory state.

Public memory benchmarks mostly score whether an answer can be produced from a
long conversation. They do not isolate the state transitions that distinguish
Mirror from a text archive: replacement, revival, coexistence, contradiction,
repeated events, preference evolution, and stale-state filtering.

StateBench is a versioned, static JSON dataset packaged with the library. Its
cases are scored against Mirror's rendered state surface, so no answer-model or
LLM judge is required after ingestion. The development/evaluation split is a
workflow boundary, not a secret split: both are committed for reproducibility.
Do not tune on the evaluation split when reporting a final score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

STATEBENCH_VERSION = "1.0.0"
STATEBENCH_CASE_COUNT = 200
STATEBENCH_CATEGORIES = frozenset({
    "replacement",
    "round_trip",
    "multi_value",
    "contradiction",
    "temporal_event",
    "preference_evolution",
    "stale_state",
})
STATEBENCH_SPLITS = frozenset({"development", "evaluation"})


@dataclass(frozen=True)
class StateCase:
    """One scripted stateful-memory case.

    Turns are ingested in order and question is recalled afterwards.

    expect_current and expect_history are all-of assertions: every string must
    appear in the rendered block. expect_any is a list of alternative groups;
    at least one string in every group must appear. forbidden catches stale or
    contradicted state leaking into the answer surface.

    The values are observable text rather than internal row ids. This makes the
    benchmark exercise extraction, lifecycle, retrieval, and rendering as one
    stateful-memory contract.
    """

    case_id: str
    category: str
    turns: tuple[str, ...]
    question: str
    split: str = "development"
    difficulty: str = "basic"
    expect_current: frozenset[str] = field(default_factory=frozenset)
    expect_history: frozenset[str] = field(default_factory=frozenset)
    forbidden: frozenset[str] = field(default_factory=frozenset)
    expect_any: tuple[frozenset[str], ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StateCase:
        return cls(
            case_id=str(raw.get("case_id") or ""),
            category=str(raw.get("category") or ""),
            split=str(raw.get("split") or "development"),
            difficulty=str(raw.get("difficulty") or "basic"),
            turns=tuple(str(turn) for turn in (raw.get("turns") or [])),
            question=str(raw.get("question") or ""),
            expect_current=frozenset(
                str(value) for value in (raw.get("expect_current") or [])
            ),
            expect_history=frozenset(
                str(value) for value in (raw.get("expect_history") or [])
            ),
            forbidden=frozenset(str(value) for value in (raw.get("forbidden") or [])),
            expect_any=tuple(
                frozenset(str(value) for value in group)
                for group in (raw.get("expect_any") or [])
            ),
            tags=tuple(str(tag) for tag in (raw.get("tags") or [])),
            note=str(raw.get("note") or ""),
        )


def _packaged_dataset() -> resources.abc.Traversable:
    return resources.files("mirror_memory.bench.data").joinpath("statebench_v1.json")


def load_statebench_payload(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate StateBench's top-level dataset envelope."""
    if path is None:
        raw = json.loads(_packaged_dataset().read_text(encoding="utf-8"))
    else:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("StateBench dataset must be a JSON object")
    if raw.get("version") != STATEBENCH_VERSION:
        raise ValueError(
            f"unsupported StateBench version: {raw.get('version')!r}; "
            f"expected {STATEBENCH_VERSION!r}"
        )
    if not isinstance(raw.get("cases"), list):
        raise ValueError("StateBench dataset must contain a cases list")
    return raw


def load_statebench(
    categories: set[str] | None = None,
    *,
    splits: set[str] | None = None,
    path: str | Path | None = None,
) -> list[StateCase]:
    """Load cases, optionally filtered by category and public split."""
    unknown_categories = (categories or set()) - STATEBENCH_CATEGORIES
    if unknown_categories:
        raise ValueError(
            f"unknown StateBench categories: {sorted(unknown_categories)}"
        )
    unknown_splits = (splits or set()) - STATEBENCH_SPLITS
    if unknown_splits:
        raise ValueError(f"unknown StateBench splits: {sorted(unknown_splits)}")

    payload = load_statebench_payload(path)
    cases = [StateCase.from_dict(item) for item in payload["cases"]]
    if categories is not None:
        cases = [case for case in cases if case.category in categories]
    if splits is not None:
        cases = [case for case in cases if case.split in splits]
    return cases


def category_counts(cases: Iterable[StateCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return dict(sorted(counts.items()))


def split_counts(cases: Iterable[StateCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.split] = counts.get(case.split, 0) + 1
    return dict(sorted(counts.items()))
