"""xfail(strict=True) stub for the ONE invariant v1 does not implement literally.

Context: the six runtime invariants I1–I6 are defined in
``docs/v2-inherited/invariants.md``; the mapping (re-audited against the real
test code) lives in ``docs/invariants.md``.

**I4 (v2 wording).**  *The same ``source_event_id`` is globally unique within
one scope (idempotency).*  v2 enforces this with a ``source_event_id`` column
plus a claim-time ``UNIQUE`` constraint.

**What v1 actually has.**  v1 has **no** ``source_event_id`` column and no
claim-time constraint on that name.  It achieves the *same behavioural
guarantee* at the ingestion layer instead:

* ``mm_evidence.ref`` is the caller's identifier for the source (message id,
  document path, event id) and is protected by a real database constraint,
  ``uq_mm_evidence_user_ref`` = UNIQUE(tenant_id, app_id, user_id, ref);
  ``record_evidence`` is idempotent on ``(user_id, ref)`` — re-observing the
  same source returns the existing row, never a duplicate.  That is
  scope-relative global uniqueness of a source identifier, i.e. I4's shape.
* publish-time idempotency is additionally enforced by
  ``mm_proposal_log.proposal_id`` (UNIQUE) and
  ``mm_evolution_jobs`` UNIQUE(tenant_id, app_id, idempotency_key).

So the *behaviour* is present (and a real dual-leg test asserts it — see
``tests/test_invariant_mapping.py::test_i4_v1_equivalent_source_ref_uniqueness``);
only the *literal v2 column name/site* is missing.

**This file** holds the strict-xfail guard for the literal v2 shape.  It is
CONDITIONAL: it asserts the v2 property **only through the mechanism v1 would
have to grow** — ``Evidence.source_event_id`` existing **and** being backed by
a uniqueness constraint.  If that column (with enforcing semantics) ever
appears, the body passes and strict mode reports **XPASS** (a failure), forcing
the mapping to be updated instead of silently drifting.  While the column is
absent the body fails, so strict mode reports **XFAIL**.  There is deliberately
**no unconditional ``pytest.fail``** — the pass/fail decision comes from the
real assertion, not a hard-coded fail.
"""

from __future__ import annotations

import pytest


def _source_event_id_uniqueness_enforced() -> tuple[bool, str]:
    """Is the v2 ``source_event_id`` column present AND uniquely constrained?

    Returns ``(enforced, why)``.  This is real introspection of the model —
    nothing hard-coded: it becomes True the moment v1 grows the column with a
    unique constraint over it, i.e. the day I4 is ported.
    """
    from mirror_memory.core.models import Evidence

    if not hasattr(Evidence, "source_event_id"):
        return False, "Evidence has no source_event_id column (v2 shape absent)"

    # The column exists — require a real uniqueness guarantee over it, so a
    # bare column that does NOT enforce the invariant still fails (XFAIL).
    unique_sets = [
        frozenset(col.name for col in c.columns)
        for c in Evidence.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    for cols in unique_sets:
        if "source_event_id" in cols:
            required = {"tenant_id", "app_id"}  # "within one scope"
            if required <= cols:
                return True, f"unique constraint over {sorted(cols)}"
    return False, "source_event_id exists but is not uniquely constrained within a scope"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "I4 in v2 shape (a unique source_event_id column) is not implemented in "
        "v1; v1 carries the EQUIVALENT guarantee on mm_evidence.ref "
        "(uq_mm_evidence_user_ref) and on proposal_id/idempotency_key. If the "
        "source_event_id column ever gains in-scope uniqueness this body passes "
        "-> XPASS -> failure, forcing docs/invariants.md to be updated."
    ),
)
def test_i4_source_event_id_global_uniqueness():
    """Assert the literal v2 I4 shape: ``Evidence.source_event_id`` globally
    unique within a scope.

    Conditional by construction: passes iff v1 actually grows the column with
    an in-scope unique constraint (then strict xfail XPASSes); otherwise the
    assertion fails and the test stays XFAIL.
    """
    enforced, why = _source_event_id_uniqueness_enforced()
    assert enforced, f"I4 (v2 source_event_id shape) not implemented: {why}"
