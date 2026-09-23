"""Three-track runner for StateBench v1.1."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from mirror_memory.bench.manifest import (
    RunManifest,
    manifest_path_for,
    write_manifest,
)
from mirror_memory.bench.statebench_v1_1 import (
    STATEBENCH_V11_VERSION,
    AtomAssertion,
    StateCaseV11,
    TextContract,
    load_statebench_v1_1,
    statebench_v1_1_path,
)

Answerer = Callable[[str, str], str]
EXTRACTION_MODES = frozenset({"deterministic", "production", "forced"})


@dataclass
class StateTrackResult:
    passed: bool
    missing: list[str] = field(default_factory=list)
    violated: list[str] = field(default_factory=list)
    match_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class TextTrackResult:
    passed: bool
    missing_all: list[str] = field(default_factory=list)
    missing_any: list[list[str]] = field(default_factory=list)
    leaked_forbidden: list[str] = field(default_factory=list)


@dataclass
class StateBenchV11CaseResult:
    case_id: str
    category: str
    transition_type: str
    split: str
    difficulty: str
    query_mode: str
    question: str
    tags: list[str]
    state: StateTrackResult
    recall: TextTrackResult
    answer: TextTrackResult | None
    beliefs: list[dict]
    recall_block: str
    answer_text: str | None
    extracted_claims_by_turn: list[int]
    note: str = ""

    @property
    def passed(self) -> bool:
        tracks = [self.state.passed, self.recall.passed]
        if self.answer is not None:
            tracks.append(self.answer.passed)
        return all(tracks)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass
class StateBenchV11Report:
    engine: str
    dataset_version: str
    dataset_path: str
    dataset_sha256: str
    extraction_mode: str
    extractor_model: str
    answerer_model: str
    cases: list[StateBenchV11CaseResult]
    tracks: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "dataset_version": self.dataset_version,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "extraction_mode": self.extraction_mode,
            "extractor_model": self.extractor_model,
            "answerer_model": self.answerer_model,
            "tracks": self.tracks,
            "cases": [case.to_dict() for case in self.cases],
        }


def _normalise(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", (value or "").casefold())
    normalised: list[str] = []
    for token in tokens:
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalised.append(token)
    return " ".join(normalised)


def _matches(surface: str, expected: str) -> bool:
    return _normalise(expected) in _normalise(surface)


def _score_text(contract: TextContract, text: str) -> TextTrackResult:
    missing_all = [
        value for value in contract.required_all if not _matches(text, value)
    ]
    missing_any = [
        list(group)
        for group in contract.required_any
        if not any(_matches(text, value) for value in group)
    ]
    leaked = [
        value for value in contract.forbidden if _matches(text, value)
    ]
    return TextTrackResult(
        passed=not (missing_all or missing_any or leaked),
        missing_all=sorted(missing_all),
        missing_any=[sorted(group) for group in missing_any],
        leaked_forbidden=sorted(leaked),
    )


_NEGATIVE_MARKERS = (
    " do not ", " don't ", " not ", " no longer ", " never ",
    " dislike", " hate", " avoid", " gave up", " cancelled", " canceled",
    " abandoned", " dropped", " sold", " stolen", " recycled",
    " gave away", " lost interest", " decided against", " cannot ",
)


def _polarity(atom: dict) -> str:
    text = f" {atom.get('predicate', '')} {atom.get('claim_text', '')} ".casefold()
    predicate = str(atom.get("predicate") or "").casefold()
    if predicate in {"dislikes", "hates", "avoids"}:
        return "negative"
    if any(marker in text for marker in _NEGATIVE_MARKERS):
        return "negative"
    return "positive"


def _is_current(atom: dict) -> bool:
    if atom.get("status") != "active":
        return False
    valid_to = atom.get("valid_to")
    if not valid_to:
        return True
    try:
        parsed = datetime.fromisoformat(str(valid_to).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed > datetime.now(UTC)
    except ValueError:
        return False


def _is_historical(atom: dict) -> bool:
    return atom.get("status") == "superseded" or not _is_current(atom)


def _surface(atom: dict) -> str:
    value = atom.get("value") or {}
    return " ".join([
        str(atom.get("key") or ""),
        str(atom.get("claim_text") or ""),
        str(atom.get("predicate") or ""),
        str(atom.get("object") or ""),
        json.dumps(value, ensure_ascii=False, sort_keys=True),
    ])


def _atom_matches(assertion: AtomAssertion, atom: dict) -> bool:
    if assertion.statuses and atom.get("status") not in assertion.statuses:
        return False
    if assertion.predicates and _normalise(str(atom.get("predicate") or "")) not in {
        _normalise(value) for value in assertion.predicates
    }:
        return False
    if assertion.polarity != "any" and _polarity(atom) != assertion.polarity:
        return False
    if assertion.temporal_mode == "current" and not _is_current(atom):
        return False
    if assertion.temporal_mode == "historical" and not _is_historical(atom):
        return False
    surface = _surface(atom)
    if any(not _matches(surface, value) for value in assertion.surface_all):
        return False
    if assertion.surface_any and not any(
        _matches(surface, value) for value in assertion.surface_any
    ):
        return False
    return True


def _score_state(case: StateCaseV11, beliefs: list[dict]) -> StateTrackResult:
    missing: list[str] = []
    violated: list[str] = []
    match_counts: dict[str, int] = {}
    for assertion in case.state_contract.required:
        count = sum(_atom_matches(assertion, atom) for atom in beliefs)
        match_counts[assertion.label] = count
        if count < assertion.min_count:
            missing.append(assertion.label)
        if assertion.max_count is not None and count > assertion.max_count:
            violated.append(assertion.label)
    for assertion in case.state_contract.forbidden:
        count = sum(_atom_matches(assertion, atom) for atom in beliefs)
        match_counts[assertion.label] = count
        if count >= assertion.min_count:
            violated.append(assertion.label)
    return StateTrackResult(
        passed=not (missing or violated),
        missing=sorted(missing),
        violated=sorted(violated),
        match_counts=match_counts,
    )


def _snapshot_beliefs(engine: Any, user_id: str) -> list[dict]:
    from sqlalchemy import select

    from mirror_memory.core.models import Belief
    from mirror_memory.core.utils import safe_json

    engine._ensure_db()
    with engine._session() as session:
        rows = list(session.scalars(
            select(Belief).where(Belief.user_id == user_id).order_by(Belief.id)
        ))
        return [
            {
                "id": row.id,
                "dimension": row.dimension,
                "key": row.key,
                "claim_text": row.claim_text,
                "status": row.status,
                "confidence": row.confidence,
                "subject": row.subject,
                "predicate": row.predicate,
                "object": row.object,
                "cardinality": row.cardinality,
                "superseded_by": row.superseded_by,
                "valid_from": row.valid_from.isoformat() if row.valid_from else None,
                "valid_to": row.valid_to.isoformat() if row.valid_to else None,
                "temporal_scope": row.temporal_scope,
                "value": safe_json(row.value_json),
            }
            for row in rows
        ]


def _model_name(client: Any | None) -> str:
    if client is None:
        return ""
    return str(
        getattr(client, "model", "")
        or getattr(client, "_model", "")
        or client.__class__.__name__
    )


def _config_for_mode(config: Any, mode: str, llm_client: Any | None) -> tuple[Any, Any | None]:
    if mode not in EXTRACTION_MODES:
        raise ValueError(f"unknown extraction mode: {mode!r}")
    if mode == "deterministic":
        return config.model_copy(update={"llm_client": None}), None

    effective_client = llm_client or config.llm_client
    if effective_client is None:
        raise ValueError(f"extraction mode {mode!r} requires an LLM client")
    if mode == "forced":
        extraction = config.extraction.model_copy(
            update={"llm_min_keyword_hits": 0}
        )
        config = config.model_copy(update={"extraction": extraction})
    return config, effective_client


def _run_case(
    case: StateCaseV11,
    *,
    config: Any,
    database_url: str,
    llm_client: Any | None,
    answerer: Answerer | None,
) -> StateBenchV11CaseResult:
    from mirror_memory.api import MemoryEngine

    engine = MemoryEngine(
        config=config,
        database_url=database_url,
        llm_client=llm_client,
    )
    extracted = [
        engine.observe(
            user_id=case.case_id,
            session_id=case.case_id,
            text=turn,
            turn_count=index,
        )
        for index, turn in enumerate(case.turns)
    ]
    beliefs = _snapshot_beliefs(engine, case.case_id)
    block = engine.recall(
        user_id=case.case_id,
        query=case.question,
        language="en",
    ) or ""
    answer_text = answerer(case.question, block) if answerer is not None else None
    return StateBenchV11CaseResult(
        case_id=case.case_id,
        category=case.category,
        transition_type=case.transition_type,
        split=case.split,
        difficulty=case.difficulty,
        query_mode=case.query_mode,
        question=case.question,
        tags=list(case.tags),
        state=_score_state(case, beliefs),
        recall=_score_text(case.recall_contract, block),
        answer=(
            _score_text(case.answer_contract, answer_text or "")
            if answer_text is not None
            else None
        ),
        beliefs=beliefs,
        recall_block=block[:2000],
        answer_text=answer_text,
        extracted_claims_by_turn=extracted,
        note=case.note,
    )


def _track_passed(case: StateBenchV11CaseResult, track: str) -> bool | None:
    result = getattr(case, track)
    return None if result is None else bool(result.passed)


def _track_summary(cases: list[StateBenchV11CaseResult], track: str) -> dict:
    available = [case for case in cases if _track_passed(case, track) is not None]

    def grouped(attribute: str) -> dict[str, dict]:
        output: dict[str, dict] = {}
        for value in sorted({str(getattr(case, attribute)) for case in available}):
            subset = [case for case in available if str(getattr(case, attribute)) == value]
            passed = sum(bool(_track_passed(case, track)) for case in subset)
            output[value] = {
                "total": len(subset),
                "passed": passed,
                "score": round(passed / len(subset), 4) if subset else None,
            }
        return output

    passed = sum(bool(_track_passed(case, track)) for case in available)
    return {
        "total": len(available),
        "passed": passed,
        "score": round(passed / len(available), 4) if available else None,
        "by_category": grouped("category"),
        "by_split": grouped("split"),
        "by_transition": grouped("transition_type"),
    }


def run_statebench_v1_1(
    *,
    categories: set[str] | None = None,
    splits: set[str] | None = None,
    config: Any | None = None,
    config_path: str | Path = "config/",
    dataset_path: str | Path | None = None,
    database_url: str = "sqlite://",
    engine_name: str = "mirror1",
    extraction_mode: str = "deterministic",
    llm_client: Any | None = None,
    answerer: Answerer | None = None,
    answerer_model: str = "",
    max_cases: int | None = None,
) -> StateBenchV11Report:
    from mirror_memory.config.loader import load_config

    cfg = config or load_config(config_path)
    cfg, effective_client = _config_for_mode(cfg, extraction_mode, llm_client)
    cases = load_statebench_v1_1(
        categories, splits=splits, path=dataset_path
    )
    if max_cases is not None:
        if max_cases < 0:
            raise ValueError("max_cases must be non-negative")
        cases = cases[:max_cases]
    results = [
        _run_case(
            case,
            config=cfg,
            database_url=database_url,
            llm_client=effective_client,
            answerer=answerer,
        )
        for case in cases
    ]
    source = Path(dataset_path) if dataset_path else statebench_v1_1_path()
    dataset_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    return StateBenchV11Report(
        engine=engine_name,
        dataset_version=STATEBENCH_V11_VERSION,
        dataset_path=str(source),
        dataset_sha256=dataset_hash,
        extraction_mode=extraction_mode,
        extractor_model=_model_name(effective_client),
        answerer_model=answerer_model,
        cases=results,
        tracks={
            track: _track_summary(results, track)
            for track in ("state", "recall", "answer")
        },
    )


def run_full_context_answer_baseline(
    *,
    answerer: Answerer,
    categories: set[str] | None = None,
    splits: set[str] | None = None,
    dataset_path: str | Path | None = None,
    max_cases: int | None = None,
) -> dict:
    """Evaluate raw context through the same answerer used for Mirror context."""
    cases = load_statebench_v1_1(categories, splits=splits, path=dataset_path)
    if max_cases is not None:
        cases = cases[:max_cases]
    results = []
    for case in cases:
        answer = answerer(case.question, "\n".join(case.turns))
        score = _score_text(case.answer_contract, answer)
        results.append({
            "case_id": case.case_id,
            "category": case.category,
            "split": case.split,
            "answer": answer,
            "score": asdict(score),
        })
    passed = sum(item["score"]["passed"] for item in results)
    return {
        "engine": "full_context_answerer",
        "total": len(results),
        "passed": passed,
        "score": round(passed / len(results), 4) if results else None,
        "cases": results,
    }


def render_statebench_v1_1_report(report: StateBenchV11Report) -> str:
    lines = [
        f"StateBench {report.dataset_version} -- engine: {report.engine}",
        f"  extraction mode: {report.extraction_mode}",
    ]
    for track in ("state", "recall", "answer"):
        stats = report.tracks[track]
        score = "not run" if stats["score"] is None else f"{stats['score']:.3f}"
        lines.append(
            f"  {track:<7}: {stats['passed']}/{stats['total']} = {score}"
        )
    return "\n".join(lines)


def write_statebench_v1_1_report(
    report: StateBenchV11Report,
    path: str | Path,
    *,
    config_dir: str | Path = "config/",
    base_url: str = "",
    thinking: str = "",
    generation_params: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a report and its mandatory provenance manifest."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = RunManifest.build(
        run_id=out.stem,
        benchmark=f"statebench-{report.dataset_version}",
        dataset_path=report.dataset_path,
        cases=len(report.cases),
        categories=sorted({case.category for case in report.cases}),
        answerer_model=report.answerer_model,
        extractor_model=report.extractor_model,
        base_url=base_url,
        thinking=thinking,
        config_dir=config_dir,
        generation_params=generation_params,
        notes=f"extraction_mode={report.extraction_mode}",
    )
    manifest_path = write_manifest(manifest, manifest_path_for(out))
    return out, manifest_path


__all__ = [
    "EXTRACTION_MODES",
    "StateBenchV11CaseResult",
    "StateBenchV11Report",
    "render_statebench_v1_1_report",
    "run_full_context_answer_baseline",
    "run_statebench_v1_1",
    "write_statebench_v1_1_report",
]
