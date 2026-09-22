"""Benchmark metrics -- five funnel metrics plus failure attribution.

Each metric measures one link in the chain, so a low score localises the
problem instead of hiding it inside an aggregate:

``Extraction Recall``
    Of the ground-truth facts that should have been extracted, how many were?
``Identity Accuracy``
    Of the extracted facts, how many were resolved to the right belief
    (support/update/create as appropriate) rather than mis-merged?
``Retrieval Recall@5``
    Of the stored facts, how many appear in the top-5 retrieved set?
``Context Coverage``
    Of the retrieved facts, how many survive into the rendered context?
``Answer F1``
    Token-level F1 between the generated answer and the ground truth.

The funnel is deliberately ordered: ``attribute_failure`` names the *first*
stage that dropped the fact, which is the only attribution that is actionable
-- a fact that was never extracted cannot also be a retrieval failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Stage names, in the order the chain runs.  Used for attribution so that a
# miss is always charged to the earliest stage that lost the fact.
ATTRIBUTION_ORDER = (
    "extraction",
    "identity",
    "store",
    "retrieve",
    "context",
    "answer",
)

# Denominator for each metric: which cases count.
_IDENTITY_DENOMINATOR = "extracted"


@dataclass
class Stage:
    """One measured link for one case."""

    name: str
    hit: bool
    detail: dict = field(default_factory=dict)


@dataclass
class CaseTrace:
    """Everything measured for a single benchmark question."""

    case_id: str
    question: str = ""
    ground_truth: str = ""
    answer: str = ""
    category: str = "uncategorised"
    stages: dict[str, Stage] = field(default_factory=dict)

    def record(self, name: str, hit: bool, **detail: object) -> Stage:
        stage = Stage(name=name, hit=bool(hit), detail=dict(detail))
        self.stages[name] = stage
        return stage

    def hit(self, name: str) -> bool:
        stage = self.stages.get(name)
        return bool(stage and stage.hit)


@dataclass
class CaseMetrics:
    extraction_hit: bool
    identity_hit: bool
    stored: bool
    retrieved_top_k: bool
    context_hit: bool
    answer_f1: float
    attribution: str


def normalise(text: str) -> list[str]:
    """Lower-case tokenisation shared by F1 and containment checks."""
    return re.findall(r"\w+", (text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-level F1 in [0, 1]."""
    pred = normalise(prediction)
    ref = normalise(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    counts: dict[str, int] = {}
    for token in pred:
        counts[token] = counts.get(token, 0) + 1

    overlap = 0
    for token in ref:
        if counts.get(token, 0) > 0:
            counts[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_fact(haystack: str, fact: str) -> bool:
    """Does *haystack* carry the ground-truth fact?

    Matches on the fact's significant tokens rather than exact substrings, so
    "She moved to Berlin in May" satisfies the fact "Berlin".
    """
    fact_tokens = [t for t in normalise(fact) if len(t) > 2]
    if not fact_tokens:
        return False
    haystack_tokens = set(normalise(haystack))
    return all(token in haystack_tokens for token in fact_tokens)


def attribute_failure(case: CaseTrace) -> str:
    """Name the earliest stage that dropped the fact for *case*.

    Returns ``"none"`` when every stage hit, and ``"answer"`` when the fact
    survived the whole chain but the generated answer was still wrong.
    """
    for name in ATTRIBUTION_ORDER:
        stage = case.stages.get(name)
        if stage is not None and not stage.hit:
            return name
    return "none"


def compute_case_metrics(case: CaseTrace) -> CaseMetrics:
    """Score one case and attribute its failure."""
    answer_f1 = token_f1(case.answer, case.ground_truth)
    # The answer stage is a hit only when the answer actually matches; the
    # ground truth reaching the context is necessary but not sufficient.
    attribution = attribute_failure(case)
    if attribution == "none" and answer_f1 < 0.5:
        attribution = "answer"

    return CaseMetrics(
        extraction_hit=case.hit("extraction"),
        identity_hit=case.hit("identity"),
        stored=case.hit("store"),
        retrieved_top_k=case.hit("retrieve"),
        context_hit=case.hit("context"),
        answer_f1=round(answer_f1, 4),
        attribution=attribution,
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarise(cases: list[CaseTrace]) -> dict:
    """Aggregate per-case metrics into the five headline funnel metrics.

    Returns a dict with each metric's value, its numerator/denominator (so a
    small denominator is visible rather than silently rounding to 0.0), and a
    failure-attribution histogram.
    """
    if not cases:
        return {
            "cases": 0,
            "metrics": {},
            "attribution": {},
        }

    per_case = [(case, compute_case_metrics(case)) for case in cases]

    extracted = [m for _c, m in per_case if m.extraction_hit]
    stored = [m for _c, m in per_case if m.stored]
    retrieved = [m for _c, m in per_case if m.retrieved_top_k]

    metrics = {
        "extraction_recall": {
            "value": _rate(len(extracted), len(per_case)),
            "numerator": len(extracted),
            "denominator": len(per_case),
            "definition": "ground-truth facts that reached extraction",
        },
        "identity_accuracy": {
            "value": _rate(sum(1 for m in extracted if m.identity_hit), len(extracted)),
            "numerator": sum(1 for m in extracted if m.identity_hit),
            "denominator": len(extracted),
            "definition": f"extracted facts resolved to the right belief (denominator: {_IDENTITY_DENOMINATOR})",
        },
        "retrieval_recall_at_5": {
            "value": _rate(sum(1 for m in stored if m.retrieved_top_k), len(stored)),
            "numerator": sum(1 for m in stored if m.retrieved_top_k),
            "denominator": len(stored),
            "definition": "stored facts present in the top-5 retrieved set",
        },
        "context_coverage": {
            "value": _rate(sum(1 for m in retrieved if m.context_hit), len(retrieved)),
            "numerator": sum(1 for m in retrieved if m.context_hit),
            "denominator": len(retrieved),
            "definition": "retrieved facts that survived into the rendered context",
        },
        "answer_f1": {
            "value": round(sum(m.answer_f1 for _c, m in per_case) / len(per_case), 4),
            "numerator": round(sum(m.answer_f1 for _c, m in per_case), 4),
            "denominator": len(per_case),
            "definition": "mean token-level F1 against the ground truth",
        },
    }

    attribution: dict[str, int] = {}
    for _case, m in per_case:
        attribution[m.attribution] = attribution.get(m.attribution, 0) + 1

    failed = sum(count for stage, count in attribution.items() if stage != "none")
    return {
        "cases": len(per_case),
        "metrics": metrics,
        "attribution": dict(
            sorted(attribution.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "failed_cases": failed,
        "attribution_share_of_failures": {
            stage: _rate(count, failed)
            for stage, count in sorted(attribution.items(), key=lambda kv: (-kv[1], kv[0]))
            if stage != "none"
        },
    }
