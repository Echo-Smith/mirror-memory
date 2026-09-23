"""Stage trace collector.

The engine emits stage events through an optional ``trace_hook`` callable so
that instrumentation never changes behaviour.  ``TraceCollector`` is the
bench-side receiver: it groups events by case and answers the funnel
questions -- was the fact extracted, was identity resolved correctly, was it
stored, was it retrieved in the top K, did it reach the rendered context.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceEvent:
    """One observation of the pipeline, tagged with the stage that produced it."""

    stage: str
    case_id: str
    detail: dict = field(default_factory=dict)


class TraceCollector:
    """Collects stage events and indexes them by case.

    A *case* is one benchmark question: the ground-truth fact is known, so
    each stage can be scored as a hit or a miss for that case.
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._by_case: dict[str, list[TraceEvent]] = {}
        self._current_case: str = ""

    # -- wiring ------------------------------------------------------------

    def hook(self, stage: str, **detail: object) -> None:
        """Trace-hook callable handed to the engine."""
        self.emit(stage, self._current_case, **detail)

    def emit(self, stage: str, case_id: str, **detail: object) -> None:
        event = TraceEvent(stage=stage, case_id=case_id, detail=dict(detail))
        self.events.append(event)
        self._by_case.setdefault(case_id, []).append(event)

    def set_case(self, case_id: str) -> None:
        """Mark subsequent hook emissions as belonging to *case_id*."""
        self._current_case = case_id

    # -- queries -----------------------------------------------------------

    def for_case(self, case_id: str) -> list[TraceEvent]:
        return list(self._by_case.get(case_id, []))

    def stages(self, case_id: str) -> list[str]:
        return [e.stage for e in self.for_case(case_id)]

    def first(self, case_id: str, stage: str) -> TraceEvent | None:
        for event in self.for_case(case_id):
            if event.stage == stage:
                return event
        return None

    def all_of(self, case_id: str, stage: str) -> list[TraceEvent]:
        return [e for e in self.for_case(case_id) if e.stage == stage]

    def __len__(self) -> int:
        return len(self.events)
