"""CandidateAtom — an extracted but not-yet-persisted cognitive triple.

A CandidateAtom is the output of K1/K2 extraction before it passes through
canonicalization and identity resolution.  It becomes a persisted Atom
(or supports/updates an existing one) only after the IdentityResolver
decides its action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateAtom:
    """A structured observation awaiting identity resolution.

    Fields
    ------
    subject:
        Who the claim is about (``"user"`` or a specific name).
    predicate:
        The relationship verb (``likes``, ``lives_in``, ``went_to``).
    object:
        What the predicate applies to (``coffee``, ``Shanghai``).
    dimension:
        Classification category (``preference``, ``fact``, ``event``, etc.).
    claim_text:
        The original user text or a natural-language summary.
    confidence:
        Extraction confidence in [0, 1].
    evidence_ids:
        References to source messages or evidence rows.
    context_tags:
        Life-context tags (``work``, ``family``, etc.).
    temporal:
        Temporal qualifier (``current``, ``past``, a date string, or ``None``).
    observed_at:
        When the engine learned this fact (ingestion time).
    valid_from / valid_to:
        When the fact itself was true.  ``None`` means open-ended: no
        ``valid_to`` is "still true", which is what lets a later fact close
        the interval instead of replacing the row.
    temporal_scope:
        One of ``current_state`` / ``persistent`` / ``episodic`` from the
        predicate policy.  Drives the lifecycle transition together with the
        temporal relation.
    source:
        How this atom was produced (``k1_keyword``, ``k1_regex``, ``k2_llm``).
    """

    subject: str = "user"
    predicate: str = ""
    object: str = ""
    dimension: str = ""
    claim_text: str = ""
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    temporal: str | None = None
    observed_at: Any | None = None
    valid_from: Any | None = None
    valid_to: Any | None = None
    temporal_scope: str = "current_state"
    source: str = "extracted"
    relation: str = "supports"  # supports / contradicts / updates / new


# ── Atom state transitions ────────────────────────────────────────────────

# Actions the IdentityResolver can return.
ACTION_CREATE = "CREATE"          # New atom, no existing match
ACTION_SUPPORT = "SUPPORT"        # Same predicate+object, strengthen evidence
ACTION_UPDATE = "UPDATE"          # SINGLE cardinality, new object supersedes old
ACTION_CONTRADICT = "CONTRADICT"  # Explicitly conflicting claim
ACTION_NOOP = "NOOP"             # Rejected / resurrection guard / no change


@dataclass
class Resolution:
    """Result of identity resolution for a single CandidateAtom.

    ``lifecycle`` is the action to take; ``action`` is kept as an alias so
    existing callers keep working.  ``temporal_relation`` records how the
    candidate's validity window related to the target's, which is what makes
    a TEMPORAL_UPDATE auditable rather than a guess.
    """

    action: str  # One of ACTION_CREATE / SUPPORT / UPDATE / CONTRADICT / NOOP
    target_belief_id: int | None = None  # Existing belief to act on (None for CREATE)
    reason: str = ""  # Human-readable explanation
    lifecycle: str = ""  # LifecycleAction value; defaults to `action`
    temporal_relation: str = ""  # TemporalRelation value, when decided
    # Extra wiring for the writer.  The round-trip revival path uses it to
    # name the same-attribute beliefs whose intervals must close; most
    # resolutions leave it empty.
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lifecycle:
            self.lifecycle = self.action


# ── Cardinality types ─────────────────────────────────────────────────────

CARDINALITY_SINGLE = "single"  # lives_in, works_at, age — only one active value
CARDINALITY_MULTI = "multi"    # likes, has, skills — multiple values coexist
CARDINALITY_EVENT = "event"    # went_to, attended — each occurrence is unique
