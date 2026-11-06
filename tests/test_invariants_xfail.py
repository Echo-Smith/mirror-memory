"""STRUCTURAL WARNING (not a behaviour test) for I4's one v2-only shape.

Context: the six runtime invariants I1–I6 are defined in
``docs/v2-inherited/invariants.md``; the mapping (re-audited against the real
test code) lives in ``docs/invariants.md``.

**Scope of this file — read this first.**
This module holds a **structural warning**, NOT a behaviour test.  It inspects
the *shape* of the ``Evidence`` model (does a column / unique constraint of the
right shape exist?) and never exercises runtime behaviour.  The behaviour that
v1 *does* provide is tested separately, for real, in
``tests/test_invariant_mapping.py`` (serial idempotency, concurrent single row,
scope isolation).  Do **not** read this file as evidence that I4's behaviour is
implemented.

**I4 (v2 wording).**  *The same ``source_event_id`` is globally unique within
one scope (idempotency).*  v2 enforces this at **CLAIM time** with a
``source_event_id`` column plus a claim-time ``UNIQUE`` constraint over
``(tenant_id, app_id, subject_id, source_event_id)``.  ``Scope`` is the trusted
``(tenant_id, app_id, subject_id)`` triple; ``source_event_id`` is the caller's
event identifier, written when the event is claimed.

**What v1 has — partial coverage, a real gap.**  v1 has **no**
``source_event_id`` column and **no claim phase** for evidence.  Its closest
analogue is ``mm_evidence.ref`` (the caller's source identifier, e.g. the
pipeline's ``evidence_message_ids``) protected by
``uq_mm_evidence_user_ref`` = UNIQUE(tenant_id, app_id, user_id, ref) with
``record_evidence`` idempotent on ``(user_id, ref)``.  That gives serial
idempotency and scope isolation, and a constraint-backed single-row outcome
under concurrency — but the **claim-time** semantics and the **event-identity
mapping** (``source_event_id`` → atom) have **no v1 counterpart**.  Verdict:
**partial coverage, gap exists** — never "equivalent".

**What the structural warning checks (precisely).**
``_source_event_id_uniqueness_enforced`` returns True ONLY if BOTH hold:

1. ``Evidence`` has a ``source_event_id`` **column**; AND
2. some ``UniqueConstraint`` on ``Evidence`` has a column set **exactly equal**
   to the target in-scope identity
   ``{tenant_id, app_id, <subject>, source_event_id}``,
   where ``<subject>`` must be the subject dimension — ``user_id`` in v1 (the
   v2 ``subject_id`` role) — and the constraint must contain **no extra
   columns**.

The **exact-set** requirement is the point: a composite unique constraint that
*contains* ``source_event_id`` but has **extra columns** (a superset, e.g.
``(tenant_id, app_id, user_id, source_event_id, observed_at)``) is **weaker**
than the target (it permits many rows per ``source_event_id`` as the extra
column varies).  Such a superset must therefore be judged **NOT implemented** —
never accepted as proof.  A bare existence check ("some column named
source_event_id exists") is also rejected.  The counterexample test
``tests/test_invariant_mapping.py::test_i4_structural_warning_rejects_superset_constraint``
pins exactly this.

**The strict xfail below** is CONDITIONAL: it passes the body only through that
precise structural check.  If v1 ever grows the exact v2 shape the body passes
and strict mode reports **XPASS** (a failure), forcing the mapping/doc to be
updated instead of silently drifting.  Today the shape is absent, so the body
fails and strict mode reports **XFAIL**.  There is deliberately **no
unconditional ``pytest.fail``** — the pass/fail decision is the real structural
assertion.  (This remains a structural warning even when it XPASSes: XPASS
means "the *shape* now exists", which must then be *behaviourally* validated.)
"""

from __future__ import annotations

import pytest

#: The v2 claim-time in-scope identity for I4, in v1 column names.
#: tenant + app + the subject dimension (v1 calls the v2 ``subject_id``
#: ``user_id``) + the event identifier itself.
_SCOPE_COLUMNS = frozenset({"tenant_id", "app_id"})
_SUBJECT_COLUMNS = frozenset({"user_id", "subject_id"})  # v1 uses user_id
_TARGET_COLUMN = "source_event_id"


def _unique_column_sets(table) -> list[frozenset[str]]:
    """Every UNIQUE constraint's column-name set on *table*."""
    out: list[frozenset[str]] = []
    for c in table.constraints:
        if c.__class__.__name__ != "UniqueConstraint":
            continue
        out.append(frozenset(col.name for col in c.columns))
    return out


def source_event_id_uniqueness_shape(table) -> tuple[bool, str]:
    """STRUCTURAL check: does *table* have the exact v2 I4 constraint shape?

    This is a *shape* test, not a behaviour test.  It is deliberately strict:

    * the ``source_event_id`` column must exist;
    * a UNIQUE constraint's column set must EQUAL
      ``tenant_id + app_id + <subject> + source_event_id`` — the subject
      dimension (``user_id`` in v1 / ``subject_id`` in v2) must be present, and
      there must be **no extra columns** (a superset is weaker than the target
      and must be judged "not implemented").

    Returns ``(shape_present, why)``.  Exposed (no leading underscore) so the
    counterexample test can drive it against a fabricated table.
    """
    col_names = {col.name for col in table.columns}
    if _TARGET_COLUMN not in col_names:
        return False, (
            f"no {_TARGET_COLUMN!r} column on {table.name} "
            "(v2 claim-time shape absent)"
        )

    target_shape_variants = [
        _SCOPE_COLUMNS | {subject} | {_TARGET_COLUMN}
        for subject in _SUBJECT_COLUMNS
    ]

    superset_reasons: list[str] = []
    for cols in _unique_column_sets(table):
        if _TARGET_COLUMN not in cols:
            continue
        # exact target shape: scope + subject + target, nothing more.
        # Accept if ANY constraint is exact, regardless of order (constraint
        # iteration order is not guaranteed, so scan all before deciding).
        if cols in target_shape_variants:
            return True, f"exact in-scope unique constraint over {sorted(cols)}"
        # a constraint that CONTAINS the target but is a superset (extra cols)
        # is WEAKER than the target — record it, but keep looking for an exact
        # match elsewhere on the table.
        if not (_SCOPE_COLUMNS <= cols):
            continue  # missing scope columns: not even scope-scoped
        if not (_SUBJECT_COLUMNS & cols):
            continue  # missing subject dimension
        extra = sorted(cols - (_SCOPE_COLUMNS | (_SUBJECT_COLUMNS & cols) | {_TARGET_COLUMN}))
        superset_reasons.append(
            f"unique constraint {sorted(cols)} CONTAINS {_TARGET_COLUMN!r} but "
            f"has extra column(s) {extra} — a superset constraint is weaker "
            "than the exact in-scope identity and does NOT implement the target"
        )

    if superset_reasons:
        # no exact match, but a look-alike superset exists -> explicitly reject.
        return False, superset_reasons[0]

    return False, (
        f"{_TARGET_COLUMN!r} exists but no UNIQUE constraint matches the exact "
        "in-scope identity (tenant_id, app_id, <subject>, source_event_id)"
    )


def _source_event_id_uniqueness_enforced() -> tuple[bool, str]:
    """Convenience wrapper: run the structural check on the real Evidence model."""
    from mirror_memory.core.models import Evidence

    return source_event_id_uniqueness_shape(Evidence.__table__)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "STRUCTURAL WARNING (not a behaviour test): I4's v2 claim-time shape "
        "(a source_event_id column with an EXACT in-scope unique constraint "
        "over tenant+app+subject+source_event_id) is not present on v1's "
        "Evidence model. v1 only PARTIALLY covers I4 — behaviourally via "
        "mm_evidence.ref, tested in test_invariant_mapping.py. If the exact "
        "shape ever appears this body passes -> XPASS -> failure, forcing "
        "docs/invariants.md to be updated."
    ),
)
def test_i4_source_event_id_global_uniqueness():
    """STRUCTURAL WARNING — NOT a behaviour test.

    Asserts that the real ``Evidence`` model carries the *exact* v2 I4 shape
    (a ``source_event_id`` column plus an in-scope unique constraint with no
    extra columns).  Conditional by construction: passes iff that exact shape
    exists (then strict xfail XPASSes); otherwise the assertion fails and the
    test stays XFAIL.  It never inspects runtime behaviour — see
    ``test_invariant_mapping.py`` for the behaviour tests.
    """
    enforced, why = _source_event_id_uniqueness_enforced()
    assert enforced, f"I4 (v2 source_event_id shape) not present: {why}"
