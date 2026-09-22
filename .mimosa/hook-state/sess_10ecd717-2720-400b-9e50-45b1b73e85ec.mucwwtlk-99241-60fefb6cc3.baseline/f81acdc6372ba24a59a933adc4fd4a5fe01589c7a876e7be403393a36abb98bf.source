"""Public data models for MemoryEngine return values.

These are lightweight dataclasses (not Pydantic models) so the library
layer has no dependency on Pydantic.  The server layer converts them to
Pydantic response models for HTTP serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class BeliefInfo:
    """A single belief returned by ``MemoryEngine.get_beliefs()``."""

    dimension: str
    key: str
    claim_text: str
    confidence: float
    layer: str
    source: str
    last_evidence_at: datetime | None = None
    # Cognitive triple fields
    subject: str = "user"
    predicate: str = ""
    object: str = ""
    cardinality: str = "multi"


@dataclass(frozen=True)
class DeleteResult:
    """Result of ``MemoryEngine.delete_memories()``."""

    beliefs: int = 0
    belief_events: int = 0
    extraction_stats: int = 0
    evolution_jobs: int = 0
    snapshots: int = 0
    intervention_events: int = 0
    consent_grants: int = 0
    session_summaries: int = 0
    evidence: int = 0
    belief_evidence_links: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return as a plain dict for backward compatibility."""
        return {
            "beliefs": self.beliefs,
            "belief_events": self.belief_events,
            "extraction_stats": self.extraction_stats,
            "evolution_jobs": self.evolution_jobs,
            "snapshots": self.snapshots,
            "intervention_events": self.intervention_events,
            "consent_grants": self.consent_grants,
            "session_summaries": self.session_summaries,
            "evidence": self.evidence,
            "belief_evidence_links": self.belief_evidence_links,
        }


@dataclass
class PanelData:
    """Structured panel data returned by ``MemoryEngine.panel()``."""

    memory: dict = field(default_factory=lambda: {"enabled": True, "has_data": False})
    sections: list[dict] = field(default_factory=list)
    avatar: dict = field(default_factory=lambda: {"familiarity": 0, "known_dimensions": []})
