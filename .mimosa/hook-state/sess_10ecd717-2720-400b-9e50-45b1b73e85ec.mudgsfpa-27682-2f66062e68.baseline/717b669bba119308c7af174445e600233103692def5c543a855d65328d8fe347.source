"""StateBench runner.

StateBench scores Mirror's rendered state surface rather than an answer model's
wording. A case passes when all required values are present, every alternative
assertion has at least one match, and stale or contradicted values are absent.

The raw full-context comparison is diagnostic. It is expected to do well on
coexistence and poorly on replacement or stale-state cases because it exposes
all turns without lifecycle semantics.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mirror_memory.bench.statebench import StateCase, load_statebench


@dataclass
class StateCaseResult:
    case_id: str
    category: str
    split: str
    difficulty: str
    passed: bool
    question: str
    expected_current: list[str]
    expected_history: list[str]
    expected_any: list[list[str]]
    forbidden: list[str]
    tags: list[str]
    block: str = ""
    missing_current: list[str] = field(default_factory=list)
    missing_history: list[str] = field(default_factory=list)
    missing_any: list[list[str]] = field(default_factory=list)
    leaked_forbidden: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StateBenchReport:
    engine: str
    cases: list[StateCaseResult]
    by_category: dict[str, dict]
    by_split: dict[str, dict]

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "total": self.total,
            "passed": self.passed,
            "score": round(self.score, 4),
            "by_category": self.by_category,
            "by_split": self.by_split,
            "cases": [case.to_dict() for case in self.cases],
        }


def _run_mirror(
    case: StateCase,
    config: Any,
    database_url: str,
    llm_client: Any | None = None,
) -> str:
    """Ingest one case, then recall its state surface."""
    from mirror_memory.api import MemoryEngine

    engine = MemoryEngine(
        config=config,
        database_url=database_url,
        llm_client=llm_client,
    )
    for index, turn in enumerate(case.turns):
        engine.observe(
            user_id=case.case_id,
            session_id=case.case_id,
            text=turn,
            turn_count=index,
        )
    return engine.recall(
        user_id=case.case_id,
        query=case.question,
        language="en",
    ) or ""


def _run_full_context(case: StateCase) -> str:
    """Diagnostic baseline: the complete conversation with no memory engine."""
    return "\n".join(case.turns)


def _normalise_surface(value: str) -> str:
    """Normalise punctuation and simple English inflection for assertions."""
    tokens = re.findall(r"[a-z0-9]+", (value or "").casefold())
    normalised: list[str] = []
    for token in tokens:
        # Renderer labels often use third-person verbs (likes/dislikes) while
        # case assertions use their lemma.  Strip only a plain trailing s;
        # semantic alternatives remain explicit in the dataset.
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalised.append(token)
    return " ".join(normalised)


def _matches(block: str, expected: str) -> bool:
    return _normalise_surface(expected) in _normalise_surface(block)


def _score(case: StateCase, block: str) -> StateCaseResult:
    missing_current = sorted(
        value for value in case.expect_current if not _matches(block, value)
    )
    missing_history = sorted(
        value for value in case.expect_history if not _matches(block, value)
    )
    missing_any = [
        sorted(group)
        for group in case.expect_any
        if not any(_matches(block, value) for value in group)
    ]
    leaked_forbidden = sorted(
        value for value in case.forbidden if _matches(block, value)
    )
    passed = not (
        missing_current
        or missing_history
        or missing_any
        or leaked_forbidden
    )
    return StateCaseResult(
        case_id=case.case_id,
        category=case.category,
        split=case.split,
        difficulty=case.difficulty,
        passed=passed,
        question=case.question,
        expected_current=sorted(case.expect_current),
        expected_history=sorted(case.expect_history),
        expected_any=[sorted(group) for group in case.expect_any],
        forbidden=sorted(case.forbidden),
        tags=list(case.tags),
        block=(block or "")[:1000],
        missing_current=missing_current,
        missing_history=missing_history,
        missing_any=missing_any,
        leaked_forbidden=leaked_forbidden,
        note=case.note,
    )


def _group_stats(results: list[StateCaseResult], attribute: str) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    values = sorted({str(getattr(case, attribute)) for case in results})
    for value in values:
        subset = [case for case in results if str(getattr(case, attribute)) == value]
        passed = sum(1 for case in subset if case.passed)
        grouped[value] = {
            "total": len(subset),
            "passed": passed,
            "score": round(passed / len(subset), 4),
        }
    return grouped


def _build_report(engine: str, results: list[StateCaseResult]) -> StateBenchReport:
    return StateBenchReport(
        engine=engine,
        cases=results,
        by_category=_group_stats(results, "category"),
        by_split=_group_stats(results, "split"),
    )


def run_statebench(
    *,
    categories: set[str] | None = None,
    splits: set[str] | None = None,
    config: Any | None = None,
    config_path: str | Path | None = "config/",
    dataset_path: str | Path | None = None,
    database_url: str = "sqlite://",
    engine_name: str = "mirror1",
    llm_client: Any | None = None,
    max_cases: int | None = None,
) -> StateBenchReport:
    """Run selected StateBench cases through Mirror."""
    from mirror_memory.config.loader import load_config

    cfg = config
    if cfg is None:
        cfg = load_config(config_path)

    cases = load_statebench(categories, splits=splits, path=dataset_path)
    if max_cases is not None:
        if max_cases < 0:
            raise ValueError("max_cases must be non-negative")
        cases = cases[:max_cases]

    results = [
        _score(case, _run_mirror(case, cfg, database_url, llm_client))
        for case in cases
    ]
    return _build_report(engine_name, results)


def run_full_context_baseline(
    *,
    categories: set[str] | None = None,
    splits: set[str] | None = None,
    dataset_path: str | Path | None = None,
    max_cases: int | None = None,
) -> StateBenchReport:
    """Score the same assertions against raw conversation context."""
    cases = load_statebench(categories, splits=splits, path=dataset_path)
    if max_cases is not None:
        if max_cases < 0:
            raise ValueError("max_cases must be non-negative")
        cases = cases[:max_cases]
    results = [_score(case, _run_full_context(case)) for case in cases]
    return _build_report("full_context", results)


def render_report(
    report: StateBenchReport,
    baseline: StateBenchReport | None = None,
) -> str:
    lines = [f"StateBench -- engine: {report.engine}"]
    lines.append(f"  overall: {report.passed}/{report.total} = {report.score:.3f}")
    if baseline is not None:
        lines.append(
            f"  full-context baseline: {baseline.passed}/{baseline.total} = "
            f"{baseline.score:.3f}"
        )
        lines.append(f"  delta over baseline: {report.score - baseline.score:+.3f}")

    lines.append("")
    lines.append("  by category")
    for category, stats in report.by_category.items():
        base = ""
        if baseline is not None and category in baseline.by_category:
            base = f"   (baseline {baseline.by_category[category]['score']:.3f})"
        lines.append(
            f"    {category:<22} {stats['passed']}/{stats['total']} = "
            f"{stats['score']:.3f}{base}"
        )

    lines.append("")
    lines.append("  by split")
    for split, stats in report.by_split.items():
        lines.append(
            f"    {split:<22} {stats['passed']}/{stats['total']} = "
            f"{stats['score']:.3f}"
        )

    failures = [case for case in report.cases if not case.passed]
    if failures:
        lines.append("")
        lines.append("  failures")
        for case in failures:
            lines.append(
                f"    [{case.category}/{case.split}] {case.case_id}: {case.question}"
            )
            if case.missing_current:
                lines.append(f"        missing current : {case.missing_current}")
            if case.missing_history:
                lines.append(f"        missing history : {case.missing_history}")
            if case.missing_any:
                lines.append(f"        missing any     : {case.missing_any}")
            if case.leaked_forbidden:
                lines.append(f"        leaked forbidden: {case.leaked_forbidden}")
    return "\n".join(lines)


def write_report(
    report: StateBenchReport,
    path: str | Path,
    baseline: StateBenchReport | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    if baseline is not None:
        payload["baseline"] = baseline.to_dict()
        payload["delta_over_baseline"] = round(report.score - baseline.score, 4)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out
