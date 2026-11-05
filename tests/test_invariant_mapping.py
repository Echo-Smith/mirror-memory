"""Invariant → test mapping checks that the real code must PASS (not xfail).

Companion to ``docs/invariants.md``.  Where ``test_invariants_xfail.py`` guards
the ONE literal v2 shape v1 lacks (``source_event_id``), this module pins the
behavioural guarantee v1 *does* provide, so the "I4 has a v1 equivalent"
statement in the docs is backed by a runnable, dual-leg test rather than prose.

I4 (v2): *the same source identifier is globally unique within one scope.*

v1 realises this at the evidence layer:
``mm_evidence`` carries ``uq_mm_evidence_user_ref`` =
UNIQUE(tenant_id, app_id, user_id, ref) and ``record_evidence`` is idempotent
on ``(user_id, ref)``.  Recording the same source twice in one scope yields ONE
row, never two — the I4 shape, expressed on ``ref`` instead of
``source_event_id``.

The test uses the shared ``db_session`` fixture, so it runs on BOTH legs
(SQLite in-memory by default; PostgreSQL when ``MM_DATABASE_URL`` is set — the
session is opened through the app role under RLS).
"""

from __future__ import annotations

from mirror_memory.core.models import Evidence
from mirror_memory.core.repository import record_evidence
from mirror_memory.core.tenancy import tenant_scope


def _evidence_ref_unique_constraint_present() -> bool:
    """The DB-level constraint that makes I4's shape real in v1."""
    for c in Evidence.__table__.constraints:
        if c.__class__.__name__ != "UniqueConstraint":
            continue
        cols = {col.name for col in c.columns}
        if cols == {"tenant_id", "app_id", "user_id", "ref"}:
            return True
    return False


def test_i4_v1_equivalent_source_ref_uniqueness(db_session):
    """I4 equivalence (v1): the same source ``ref`` in one scope is one row.

    Runs on both SQLite and PostgreSQL via ``db_session``.  This is the
    behavioural guarantee that stands in for v2's ``source_event_id`` global
    uniqueness.
    """
    assert _evidence_ref_unique_constraint_present(), (
        "mm_evidence must keep uq_mm_evidence_user_ref(tenant_id, app_id, "
        "user_id, ref) — it is the v1 realisation of I4"
    )

    with tenant_scope("default", "default"):
        first = record_evidence(db_session, "u-i4", ref="evt-123", content="hello")
        db_session.commit()
        second = record_evidence(db_session, "u-i4", ref="evt-123", content="hello again")
        db_session.commit()

        rows = (
            db_session.query(Evidence)
            .filter_by(user_id="u-i4", ref="evt-123")
            .all()
        )

    # Same source -> same row, exactly one: the source identifier is unique
    # within the scope (this is I4 expressed on ``ref``).
    assert first.id == second.id
    assert len(rows) == 1, [r.id for r in rows]

    # A different ref in the same scope is a distinct source (uniqueness is
    # per source identifier, not a blanket "one row per user").
    with tenant_scope("default", "default"):
        third = record_evidence(db_session, "u-i4", ref="evt-999", content="other")
        db_session.commit()
        count = (
            db_session.query(Evidence)
            .filter_by(user_id="u-i4")
            .count()
        )
    assert third.id != first.id
    assert count == 2
