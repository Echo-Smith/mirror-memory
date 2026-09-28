"""Protected invariants — what the runtime may never cool, compact, or purge.

This is written *before* any planner exists, on purpose.  A long-lived
memory system's first question is not "what can I delete?" but "what must I
never delete by accident?".  Every rule here answers the second question; a
user's explicit signal always outranks an age heuristic::

    "the user corrected this"    >   "this hasn't been used in months"
    "this is under dispute"      >   "storage reduction"

The function is pure: the caller assembles the evidence-relation counts and
the pending-verification flag (the repository already knows how), and this
module only decides.  The metabolism planner must consult it before
proposing any tier demotion, compaction, or archive.
"""

from __future__ import annotations

from mirror_memory.core.utils import safe_json

# A protected belief's heat score is floored here, which keeps it in the hot
# scan set no matter how old it is.
PROTECTED_HEAT_FLOOR = 0.70

# A canonical belief at or above this confidence is treated as settled fact.
CANONICAL_PROTECT_CONFIDENCE = 0.8

# Reasons (symbols, not prose — they are stored and compared).
REASON_UNRESOLVED_CONFLICT = "unresolved_conflict"
REASON_USER_CORRECTION = "user_correction"
REASON_PENDING_VERIFICATION = "pending_verification"
REASON_USER_CONFIRMED = "user_confirmed"
REASON_CANONICAL_HIGH_CONFIDENCE = "canonical_high_confidence"
REASON_THIN_EVIDENCE = "thin_evidence"


def protection_reason(
    belief,
    *,
    support_count: int = 0,
    correct_count: int = 0,
    pending_verification: bool = False,
) -> str | None:
    """Return the first reason *belief* is protected, or ``None``.

    Parameters
    ----------
    belief:
        The belief row (or any stand-in with the same attributes).
    support_count / correct_count:
        Typed evidence-link counts by relation.  ``support`` counts as
        evidence strength; a ``correct`` link means the user explicitly
        corrected around this belief.
    pending_verification:
        Whether a verification question about this belief awaits an answer.
    """
    # An unresolved source conflict: both sides stay alive until the user
    # clarifies.  Compacting either side would flatten the dispute.
    val = safe_json(getattr(belief, "value_json", "{}") or "{}")
    if val.get("clarification_status") == "needs_clarification":
        return REASON_UNRESOLVED_CONFLICT

    # The user explicitly corrected this belief — the strongest retention
    # signal the system has.
    if correct_count > 0 or getattr(belief, "source", "") == "user_corrected":
        return REASON_USER_CORRECTION

    # A verification question is in flight; moving the belief mid-question
    # would strand the pending answer.
    if pending_verification:
        return REASON_PENDING_VERIFICATION

    # User-confirmed beliefs (L1/L2 or confirmed source).
    if getattr(belief, "layer", "") in ("L1", "L2") or getattr(belief, "source", "") == "user_confirmed":
        return REASON_USER_CONFIRMED

    # Settled canonical fact (name, employer, long-lived state) with real
    # confidence behind it.
    if (
        getattr(belief, "retention_class", "") == "canonical"
        and float(getattr(belief, "confidence", 0.0) or 0.0) >= CANONICAL_PROTECT_CONFIDENCE
    ):
        return REASON_CANONICAL_HIGH_CONFIDENCE

    # Thin evidence: a belief with at most one supporting observation cannot
    # afford to lose any of it to compaction.
    if support_count <= 1:
        return REASON_THIN_EVIDENCE

    return None


def is_protected(
    belief,
    *,
    support_count: int = 0,
    correct_count: int = 0,
    pending_verification: bool = False,
) -> bool:
    """Convenience wrapper: is *belief* protected for any reason?"""
    return (
        protection_reason(
            belief,
            support_count=support_count,
            correct_count=correct_count,
            pending_verification=pending_verification,
        )
        is not None
    )


def compaction_protection_reason(
    belief,
    *,
    pending_verification: bool = False,
) -> str | None:
    """Protection that applies to *evidence compaction* specifically.

    A user correction is deliberately not in this set: corrected rows are
    always kept by the compaction keep rules (``correct`` links are never
    folded), so aggregating the redundant support bulk loses nothing about
    the correction.  What compaction must not touch is a live dispute —
    where the shape of the evidence *is* the dispute — or a belief with a
    verification question in flight.

    Tier cooling, by contrast, keeps the stricter :func:`protection_reason`:
    demoting a corrected belief's storage tier serves nothing.
    """
    val = safe_json(getattr(belief, "value_json", "{}") or "{}")
    if val.get("clarification_status") == "needs_clarification":
        return REASON_UNRESOLVED_CONFLICT
    if pending_verification:
        return REASON_PENDING_VERIFICATION
    return None
