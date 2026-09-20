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
    """Result of identity resolution for a single CandidateAtom."""

    action: str  # One of ACTION_CREATE / SUPPORT / UPDATE / CONTRADICT / NOOP
    target_belief_id: int | None = None  # Existing belief to act on (None for CREATE)
    reason: str = ""  # Human-readable explanation


# ── Cardinality types ─────────────────────────────────────────────────────

CARDINALITY_SINGLE = "single"  # lives_in, works_at, age — only one active value
CARDINALITY_MULTI = "multi"    # likes, has, skills — multiple values coexist
CARDINALITY_EVENT = "event"    # went_to, attended — each occurrence is unique
