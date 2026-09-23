"""Benchmark observability -- turn the recall chain into measurable stages.

A benchmark score on its own says nothing about *where* a system lost the
answer.  This package splits the chain

    Ground Truth → Extraction → Identity → Store → Retrieve → Context → Answer

into individually measured stages, so a failure can be attributed instead of
guessed at.  The headline question it exists to answer is of the form::

    "Of the Single-hop failures, 58% were lost at extraction, 22% at
     retrieval" -- not "Single-hop scored 0.042".

Nothing here imports an LLM or a dataset: ``metrics`` is pure functions,
``trace`` is a data collector, ``runner`` glues them to the engine.  That
keeps the funnel testable without network access.
"""

from mirror_memory.bench.metrics import (
    ATTRIBUTION_ORDER,
    CaseTrace,
    Stage,
    attribute_failure,
    compute_case_metrics,
    summarise,
)
from mirror_memory.bench.statebench import (
    STATEBENCH_CASE_COUNT,
    STATEBENCH_CATEGORIES,
    STATEBENCH_SPLITS,
    StateCase,
    load_statebench,
)
from mirror_memory.bench.statebench_v1_1 import (
    STATEBENCH_V11_CASE_COUNT,
    STATEBENCH_V11_CATEGORIES,
    STATEBENCH_V11_SPLITS,
    STATEBENCH_V11_TRACKS,
    STATEBENCH_V11_VERSION,
    StateCaseV11,
    load_statebench_v1_1,
)
from mirror_memory.bench.trace import TraceCollector, TraceEvent

__all__ = [
    "ATTRIBUTION_ORDER",
    "CaseTrace",
    "Stage",
    "TraceCollector",
    "TraceEvent",
    "attribute_failure",
    "compute_case_metrics",
    "summarise",
    "STATEBENCH_CASE_COUNT",
    "STATEBENCH_CATEGORIES",
    "STATEBENCH_SPLITS",
    "StateCase",
    "load_statebench",
    "STATEBENCH_V11_CASE_COUNT",
    "STATEBENCH_V11_CATEGORIES",
    "STATEBENCH_V11_SPLITS",
    "STATEBENCH_V11_TRACKS",
    "STATEBENCH_V11_VERSION",
    "StateCaseV11",
    "load_statebench_v1_1",
]
