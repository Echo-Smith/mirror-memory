"""Per-case funnel capture — every stage, saved, not inferred.

The plan's P2-9 asks for the whole chain per case: per-turn extraction counts,
candidate atoms, identity decisions, persistence actions, final belief rows,
recall candidates, gate reasons, the rendered block, and the answer.  A score
says a case failed; this says which stage dropped it.

The capture reuses the engine's existing ``trace_hook`` (the same one the
LOCOMO funnel uses) rather than reimplementing the pipeline, so what is
recorded is what actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mirror_memory.bench.trace import TraceCollector


@dataclass
class CaseFunnel:
    """The full funnel for one benchmark case."""

    case_id: str
    # Per-turn extraction counts: {"k1": n, "k2": m} per turn index.
    extraction_per_turn: list[dict] = field(default_factory=list)
    # Candidate atoms the extractor produced, with their canonical triples.
    candidate_atoms: list[dict] = field(default_factory=list)
    # Identity decisions: action / lifecycle / target / reason per atom.
    identity_decisions: list[dict] = field(default_factory=list)
    # Persistence actions: what the writer actually did.
    persistence_actions: list[dict] = field(default_factory=list)
    # Final belief rows for the case's user.
    belief_rows: list[dict] = field(default_factory=list)
    # Recall candidates admitted for the question.
    recall_candidates: list[str] = field(default_factory=list)
    # Why candidates were dropped (l4_watermark / needs_confirmation /
    # zero_score) and why rendered items were cut (budget / diversity).
    gate_reasons: list[str] = field(default_factory=list)
    # The rendered block that reached the answerer.
    rendered_block: str = ""
    # The generated answer, when an answerer was supplied.
    answer: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "extraction_per_turn": self.extraction_per_turn,
            "candidate_atoms": self.candidate_atoms,
            "identity_decisions": self.identity_decisions,
            "persistence_actions": self.persistence_actions,
            "belief_rows": self.belief_rows,
            "recall_candidates": self.recall_candidates,
            "gate_reasons": self.gate_reasons,
            "rendered_block": self.rendered_block,
            "answer": self.answer,
        }


def capture_funnel(
    engine: Any,
    *,
    case_id: str,
    turns: list[str],
    question: str,
    session_id: str | None = None,
    language: str = "en",
    answerer: Any | None = None,
) -> tuple[CaseFunnel, str]:
    """Ingest *turns*, recall *question*, and record every stage on the way.

    Returns ``(funnel, block)``.  The trace hook is attached to the engine's
    pipeline for the duration and detached afterwards, so a capture never
    leaks into a later case.
    """
    sid = session_id or case_id
    collector = TraceCollector()
    pipeline = engine._get_pipeline()  # noqa: SLF001
    previous_hook = getattr(pipeline, "_trace_hook", None)
    pipeline._trace_hook = collector.hook  # noqa: SLF001
    try:
        for index, turn in enumerate(turns):
            collector.set_case(f"{case_id}:{index}")
            engine.observe(user_id=case_id, session_id=sid, text=turn, turn_count=index)
    finally:
        pipeline._trace_hook = previous_hook  # noqa: SLF001

    funnel = CaseFunnel(case_id=case_id)
    funnel.extraction_per_turn = _extraction_per_turn(collector, len(turns))
    funnel.candidate_atoms = _candidate_atoms(collector)
    funnel.identity_decisions = _identity_decisions(collector)
    funnel.persistence_actions = _persistence_actions(collector)

    block = engine.recall(
        user_id=case_id, query=question, language=language,
        trace_hook=collector.hook,
    ) or ""
    funnel.recall_candidates = _recall_candidates(collector, case_id)
    funnel.gate_reasons = _gate_reasons(collector, case_id)
    funnel.rendered_block = block
    funnel.belief_rows = _belief_rows(engine, case_id)

    if answerer is not None:
        try:
            funnel.answer = str(answerer(question, block))
        except Exception:
            funnel.answer = ""

    return funnel, block


def _events(collector: TraceCollector, case_id: str):
    """Every event whose case id is the case itself or one of its turns."""
    prefix = f"{case_id}:"
    return [
        e for e in collector.events
        if e.case_id == case_id or e.case_id.startswith(prefix)
    ]


def _extraction_per_turn(collector: TraceCollector, turn_count: int) -> list[dict]:
    per_turn = [{"k1": 0, "k2": 0} for _ in range(turn_count)]
    for event in collector.events:
        if event.stage != "extraction":
            continue
        _case, _, index = event.case_id.partition(":")
        if not index.isdigit():
            continue
        index = int(index)
        if index >= len(per_turn):
            continue
        source = event.detail.get("source", "")
        if source == "k1":
            per_turn[index]["k1"] += event.detail.get("count", 0)
        elif source == "k2":
            per_turn[index]["k2"] += event.detail.get("count", 0)
    return per_turn


def _candidate_atoms(collector: TraceCollector) -> list[dict]:
    atoms = []
    for event in collector.events:
        if event.stage != "extraction":
            continue
        detail = event.detail
        predicates = detail.get("predicates") or []
        objects = detail.get("objects") or []
        texts = detail.get("claim_texts") or []
        for i in range(max(len(predicates), len(objects), len(texts))):
            atoms.append({
                "source": detail.get("source", ""),
                "predicate": predicates[i] if i < len(predicates) else "",
                "object": objects[i] if i < len(objects) else "",
                "claim_text": texts[i] if i < len(texts) else "",
            })
    return atoms


def _identity_decisions(collector: TraceCollector) -> list[dict]:
    return [
        {
            "action": e.detail.get("action", ""),
            "lifecycle": e.detail.get("lifecycle", ""),
            "temporal_relation": e.detail.get("temporal_relation", ""),
            "predicate": e.detail.get("predicate", ""),
            "object": e.detail.get("object", ""),
            "target_belief_id": e.detail.get("target_belief_id"),
            "reason": e.detail.get("reason", ""),
        }
        for e in collector.events
        if e.stage == "identity"
    ]


def _persistence_actions(collector: TraceCollector) -> list[dict]:
    return [
        {
            "action": e.detail.get("action", ""),
            "belief_id": e.detail.get("belief_id"),
            "persisted": e.detail.get("persisted"),
            "predicate": e.detail.get("predicate", ""),
            "object": e.detail.get("object", ""),
        }
        for e in collector.events
        if e.stage == "store"
    ]


def _recall_candidates(collector: TraceCollector, case_id: str) -> list[str]:
    for event in collector.events:
        if event.stage == "retrieve" and event.case_id == case_id:
            return [str(k) for k in (event.detail.get("scored") or [])]
    return []


def _gate_reasons(collector: TraceCollector, case_id: str) -> list[str]:
    reasons: list[str] = []
    for event in collector.events:
        if event.stage == "retrieve" and event.case_id == case_id:
            reasons.extend(str(g) for g in (event.detail.get("gated_out") or []))
        elif event.stage == "context" and event.case_id == case_id:
            dropped = event.detail.get("dropped_for_budget")
            if dropped:
                reasons.append(f"dropped_for_budget:{dropped}")
    return reasons


def _belief_rows(engine: Any, case_id: str) -> list[dict]:
    from mirror_memory.core.models import Belief

    with engine._session() as session:  # noqa: SLF001
        rows = session.query(Belief).filter_by(user_id=case_id).all()
        return [
            {
                "id": b.id,
                "key": b.key,
                "predicate": b.predicate,
                "raw_predicate": b.raw_predicate,
                "object": b.object,
                "status": b.status,
                "polarity": b.polarity,
                "lifecycle_state": b.lifecycle_state,
                "confidence": b.confidence,
                "valid_from": str(b.valid_from) if b.valid_from else None,
                "valid_to": str(b.valid_to) if b.valid_to else None,
                "claim_text": b.claim_text,
            }
            for b in rows
        ]
