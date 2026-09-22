"""Benchmark runner -- ingest, recall, and attribute every failure to a stage.

The runner is deliberately dataset-agnostic.  A *case* is a dict::

    {
        "case_id": "conv0_q12",
        "category": "single-hop",          # free-form, used for grouping
        "turns": ["I moved to Berlin in May.", ...],   # ingested before the question
        "question": "Where does the user live now?",
        "answer": "Berlin",                # ground truth
        "evidence": "Berlin",              # the fact the answer must come from
    }

``evidence`` is what the funnel is scored against.  It is separate from
``answer`` on purpose: the fact can be correctly stored and retrieved while
the answering model still words its reply differently, and those are
different failures.

Usage::

    from mirror_memory.bench.runner import run_cases, render_report

    report = run_cases(cases, config_path="config/", db_url="sqlite://")
    print(render_report(report))
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mirror_memory.bench.metrics import (
    CaseTrace,
    compute_case_metrics,
    contains_fact,
    normalise,
    summarise,
    token_f1,
)
from mirror_memory.bench.trace import TraceCollector

logger = logging.getLogger(__name__)

RETRIEVE_TOP_K = 5


@dataclass
class BenchCase:
    """One benchmark question plus the fact it depends on.

    ``answer`` is what every funnel stage is scored against -- the content the
    answer needs.  ``evidence`` is the source turn(s) the dataset attributes
    the answer to; it is provenance, not the scoring target, because a K1
    claim stores its input snippet verbatim and would otherwise match the
    evidence turn tautologically.

    ``group`` is the unit of ingestion: every case in a group shares one
    ingested conversation and therefore one ``user_id``.  Ingesting per
    question instead would replay the same conversation hundreds of times.
    """

    case_id: str
    category: str = "uncategorised"
    turns: list[str] = field(default_factory=list)
    question: str = ""
    answer: str = ""
    evidence: str = ""
    group: str = ""

    def __post_init__(self) -> None:
        if not self.group:
            self.group = self.case_id

    @classmethod
    def from_dict(cls, raw: dict) -> BenchCase:
        return cls(
            case_id=str(raw.get("case_id") or raw.get("id") or ""),
            category=str(raw.get("category") or "uncategorised"),
            turns=list(raw.get("turns") or []),
            question=str(raw.get("question") or ""),
            answer=str(raw.get("answer") or ""),
            evidence=str(raw.get("evidence") or raw.get("answer") or ""),
            group=str(raw.get("group") or ""),
        )


def load_cases(path: str | Path) -> list[BenchCase]:
    """Load cases from a JSON file (a list of case dicts)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases", [])
    return [BenchCase.from_dict(item) for item in raw]


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    ground_truth: str
    stages: dict[str, bool]
    attribution: str
    answer: str
    answer_f1: float
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _ingest(engine, group: str, turns: list[str], collector: TraceCollector) -> None:
    """Feed a group's turns through the engine once, under the group's user id.

    ``turn_count`` is the index within the group: the K2 throttle's
    every-N-turns rule keys off it, so passing a constant would silently
    disable that rule and understate how much K2 fires in production.
    """
    for index, turn in enumerate(turns):
        collector.set_case(group)
        engine.observe(user_id=group, session_id=group, text=turn, turn_count=index)


def _score_target(case: BenchCase) -> str:
    """The content each funnel stage is scored against.

    The answer is the target.  When the answer carries no significant tokens
    of its own (``"yes"``, ``"no"``, a bare number), the attributed evidence
    turn is used instead so the stage is still measurable.
    """
    if normalise(case.answer):
        return case.answer
    return case.evidence or case.answer


def _score_case(
    case: BenchCase,
    collector: TraceCollector,
    context: str,
    answer: str,
) -> CaseTrace:
    """Turn the collected trace into per-stage hits for one case.

    The ingest stages (extraction / identity / store) are read from the
    *group* trace, because the conversation is ingested once for every
    question it carries.  The query stages (retrieve / context) are read from
    the case's own trace.
    """
    target = _score_target(case)
    trace = CaseTrace(
        case_id=case.case_id,
        question=case.question,
        ground_truth=case.answer,
        answer=answer,
        category=case.category,
    )

    extraction_events = collector.all_of(case.group, "extraction")
    extracted_text = " ".join(
        " ".join(str(v) for v in (e.detail.get("claim_texts") or []))
        for e in extraction_events
    )
    extraction_hit = bool(target) and contains_fact(extracted_text, target)
    trace.record(
        "extraction",
        extraction_hit,
        target=target,
        events=len(extraction_events),
        extracted=extracted_text[:400],
        attributed_evidence=case.evidence[:400],
    )

    identity_events = collector.all_of(case.group, "identity")
    identity_hit = False
    if identity_events:
        # The fact was extracted and resolved to a live belief rather than
        # dropped, mis-merged, or blocked by the resurrection guard.
        identity_hit = any(
            e.detail.get("action") in {"CREATE", "SUPPORT", "UPDATE", "CONTRADICT"}
            for e in identity_events
        )
    trace.record(
        "identity",
        identity_hit,
        actions=[e.detail.get("action") for e in identity_events],
    )

    store_events = collector.all_of(case.group, "store")
    stored_hit = any(bool(e.detail.get("persisted")) for e in store_events)
    trace.record(
        "store",
        stored_hit,
        actions=[e.detail.get("action") for e in store_events],
        belief_ids=[e.detail.get("belief_id") for e in store_events],
    )

    retrieve_events = collector.all_of(case.case_id, "retrieve")
    scored_keys: list[str] = []
    for e in retrieve_events:
        scored_keys.extend(str(k) for k in (e.detail.get("scored") or []))
    retrieve_hit = bool(retrieve_events) and len(scored_keys) > 0
    retrieve_detail = retrieve_events[0].detail if retrieve_events else {}
    trace.record(
        "retrieve",
        retrieve_hit,
        top_k=scored_keys[:RETRIEVE_TOP_K],
        scored=len(scored_keys),
        render=retrieve_detail,
    )

    context_events = collector.all_of(case.case_id, "context")
    # Pass the renderer's own trace through verbatim rather than cherry-picking
    # fields: a signal the runner forgets to copy is a signal the report loses.
    render_detail = context_events[0].detail if context_events else {}
    context_hit = contains_fact(context or "", target)
    trace.record(
        "context",
        context_hit,
        target=target,
        chars=len(context or ""),
        render=render_detail,
    )

    trace.record("answer", token_f1(answer, case.answer) >= 0.5, answer=answer)
    return trace


def run_cases(
    cases: Iterable[BenchCase | dict],
    *,
    config_path: str | Path | None = "config/",
    config: Any | None = None,
    database_url: str = "sqlite://",
    answerer: Any | None = None,
    language: str = "en",
) -> dict:
    """Run every case through the engine and attribute each failure.

    Parameters
    ----------
    cases:
        ``BenchCase`` instances or raw case dicts.
    config_path / config:
        Engine configuration (one of the two).
    database_url:
        SQLAlchemy URL.  Defaults to a private in-memory database per run.
    answerer:
        Optional callable ``answerer(question, context) -> str``.  Without
        one the answer stage is scored as a miss, and the funnel still
        reports extraction/identity/store/retrieve/context -- which is the
        point: those stages are measurable without an answering model.
    language:
        Render language passed to the engine.

    Returns
    -------
    dict
        ``{"summary": ..., "by_category": ..., "cases": [...]}``
    """
    from mirror_memory.api import MemoryEngine

    normalised = [
        c if isinstance(c, BenchCase) else BenchCase.from_dict(c) for c in cases
    ]
    if not normalised:
        return {"summary": summarise([]), "by_category": {}, "cases": []}

    collector = TraceCollector()
    engine = MemoryEngine(
        config=config,
        config_path=None if config is not None else config_path,
        database_url=database_url,
    )
    pipeline_hook = collector.hook
    # Route every funnel stage through the collector.  The pipeline is created
    # lazily by the first observe(); building it here lets us attach the hook.
    engine._get_pipeline()._trace_hook = pipeline_hook  # noqa: SLF001 - instrumentation

    results: list[CaseResult] = []
    traces: list[CaseTrace] = []

    # Ingest once per group, then ask every question that group carries.  The
    # ingest trace is attributed to the group so a case's own extraction trace
    # is not polluted by its siblings' questions.
    groups: dict[str, list[BenchCase]] = {}
    for case in normalised:
        groups.setdefault(case.group, []).append(case)

    for group, group_cases in groups.items():
        _ingest(engine, group, group_cases[0].turns, collector)

        for case in group_cases:
            collector.set_case(case.case_id)
            context = engine.recall(
                user_id=group,
                query=case.question,
                language=language,
                trace_hook=pipeline_hook,
            ) or ""

            answer = ""
            if answerer is not None:
                try:
                    answer = str(answerer(case.question, context))
                except Exception:
                    logger.warning("bench: answerer failed for case %s", case.case_id)

            trace = _score_case(case, collector, context, answer)
            metrics = compute_case_metrics(trace)
            traces.append(trace)
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    question=case.question,
                    ground_truth=case.answer,
                    stages={name: stage.hit for name, stage in trace.stages.items()},
                    attribution=metrics.attribution,
                    answer=answer,
                    answer_f1=metrics.answer_f1,
                    detail={
                        name: stage.detail for name, stage in trace.stages.items()
                    },
                )
            )

    by_category: dict[str, dict] = {}
    for category in sorted({t.category for t in traces}):
        subset = [t for t in traces if t.category == category]
        by_category[category] = summarise(subset)

    return {
        "summary": summarise(traces),
        "by_category": by_category,
        "cases": [r.to_dict() for r in results],
    }


def render_report(report: dict) -> str:
    """Render a run as plain text, leading with the attribution histogram."""
    lines: list[str] = []
    summary = report.get("summary", {})
    lines.append(f"cases: {summary.get('cases', 0)}")

    lines.append("")
    lines.append("funnel metrics")
    for name, info in (summary.get("metrics") or {}).items():
        lines.append(
            f"  {name:<24} {info['value']:<8} "
            f"({info['numerator']}/{info['denominator']})  {info['definition']}"
        )

    attribution = summary.get("attribution") or {}
    if attribution:
        lines.append("")
        lines.append("failure attribution")
        share = summary.get("attribution_share_of_failures") or {}
        for stage, count in attribution.items():
            pct = share.get(stage)
            suffix = "" if pct is None else f"  {pct:.0%} of failures"
            lines.append(f"  {stage:<24} {count}{suffix}")

    by_category = report.get("by_category") or {}
    if by_category:
        lines.append("")
        lines.append("by category")
        for category, cat_summary in by_category.items():
            metrics = cat_summary.get("metrics") or {}
            extraction = metrics.get("extraction_recall", {}).get("value", 0.0)
            retrieval = metrics.get("retrieval_recall_at_5", {}).get("value", 0.0)
            f1 = metrics.get("answer_f1", {}).get("value", 0.0)
            lines.append(
                f"  {category:<24} n={cat_summary.get('cases', 0):<5} "
                f"extraction={extraction:<8} retrieval={retrieval:<8} answer_f1={f1}"
            )

    return "\n".join(lines)
