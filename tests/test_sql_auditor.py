"""M1 task 3 -- test-time SQL auditor.

Rule tests run the auditor directly on statement text (fast, precise);
the counterexamples prove the auditor catches the classic leak shapes.
A separate integration test proves the installed listener hooks real
SQLAlchemy execution.
"""

from __future__ import annotations

import pytest

from mirror_memory.core.sql_audit import AuditViolation, audit_sql


def test_select_with_both_predicates_passes():
    audit_sql(
        "SELECT id FROM mm_beliefs WHERE user_id = ? AND tenant_id = ? AND app_id = ?"
    )


def test_update_with_both_predicates_passes():
    audit_sql(
        "UPDATE mm_beliefs SET confidence = ? WHERE id = ? AND tenant_id = ? AND app_id = ?"
    )


def test_delete_with_both_predicates_passes():
    audit_sql("DELETE FROM mm_beliefs WHERE id = ? AND tenant_id = ? AND app_id = ?")


def test_insert_with_both_columns_passes():
    audit_sql(
        "INSERT INTO mm_beliefs (user_id, tenant_id, app_id, key) VALUES (?, ?, ?, ?)"
    )


def test_join_both_sides_passes():
    audit_sql(
        "SELECT * FROM mm_beliefs AS b JOIN mm_evidence AS e "
        "ON e.belief_id = b.id "
        "WHERE b.tenant_id = ? AND b.app_id = ? AND e.tenant_id = ? AND e.app_id = ?"
    )


def test_non_tenant_table_unaffected():
    audit_sql("SELECT id FROM some_other_table WHERE x = ?")


def test_transaction_control_exempt():
    audit_sql("SAVEPOINT sa_savepoint_1")
    audit_sql("RELEASE sa_savepoint_1")
    audit_sql("SELECT 1")
    audit_sql("PRAGMA table_info(mm_beliefs)")


# ---------------------------------------------------------------------------
# Counterexamples: each of these MUST be caught
# ---------------------------------------------------------------------------


def test_counterexample_select_missing_predicates():
    with pytest.raises(AuditViolation):
        audit_sql("SELECT * FROM mm_beliefs WHERE user_id = ?")


def test_counterexample_join_filters_one_side_only():
    with pytest.raises(AuditViolation):
        audit_sql(
            "SELECT * FROM mm_beliefs AS b JOIN mm_evidence AS e "
            "ON e.belief_id = b.id "
            "WHERE b.tenant_id = ? AND b.app_id = ?"
        )


def test_counterexample_subquery_missing_filter():
    with pytest.raises(AuditViolation):
        audit_sql(
            "SELECT * FROM mm_beliefs AS b WHERE b.tenant_id = ? AND b.app_id = ? "
            "AND b.id IN (SELECT belief_id FROM mm_belief_events WHERE event_type = ?)"
        )


def test_counterexample_raw_text_sql():
    with pytest.raises(AuditViolation):
        audit_sql("SELECT * FROM mm_beliefs")


def test_counterexample_insert_missing_tenant_columns():
    with pytest.raises(AuditViolation):
        audit_sql("INSERT INTO mm_beliefs (user_id, key) VALUES (?, ?)")


def test_counterexample_filtered_update_missing_predicates():
    # A WHERE over non-PK columns without tenant predicates is a genuine
    # cross-tenant write surface (PK-only WHERE is the ORM unit-of-work
    # shape, exempted by design -- its safety lives in the write guard).
    with pytest.raises(AuditViolation):
        audit_sql("UPDATE mm_beliefs SET confidence = ? WHERE user_id = ?")


def test_counterexample_pk_delete_of_another_shape():
    # DELETE with a non-PK filter and no tenant predicates: caught.
    with pytest.raises(AuditViolation):
        audit_sql("DELETE FROM mm_beliefs WHERE session_id = ?")


def test_counterexample_literal_predicate_not_bound():
    # A literal in the SQL text is not a bound parameter: it cannot be
    # proven tenant-scoped per execution and must fail the audit.
    with pytest.raises(AuditViolation):
        audit_sql(
            "SELECT * FROM mm_beliefs WHERE tenant_id = 'tA' AND app_id = 'app'"
        )


def test_counterexample_unparseable_fails_closed():
    with pytest.raises(AuditViolation):
        audit_sql("SELCT * FRMM mm_beliefs WHERE tenant_id = ?")


# ---------------------------------------------------------------------------
# Integration: the installed listener catches a real leak through SQLAlchemy
# ---------------------------------------------------------------------------


def test_installed_listener_catches_real_leak():
    import os

    if os.environ.get("MM_SQL_AUDIT", "1") == "0":
        pytest.skip("auditor disabled for this run (MM_SQL_AUDIT=0)")
    from sqlalchemy import create_engine, text as sa_text

    from mirror_memory.core.models import Base, Belief
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    # Legitimate: ORM query under a tenant scope gets loader criteria.
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=engine)
    with tenant_scope("tA", "app"), fixture_write_scope("auditor integration seed"):
        s = factory()
        s.add(Belief(user_id="u1", dimension="d", key="k", claim_text="ok"))
        s.commit()
        assert s.query(Belief).all()  # auditor sees tenant_id=? AND app_id=?
        s.close()

    # Leak: raw SQL without predicates must raise through the listener.
    with pytest.raises(AuditViolation):
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT id FROM mm_beliefs")).all()


# ---------------------------------------------------------------------------
# Task 1a: the verdict cache must never cache a context "pass"
# ---------------------------------------------------------------------------

LEAK_SQL = "SELECT id FROM mm_beliefs WHERE id = ? AND user_id = ?"  # no tenant predicates


def test_cache_context_pass_not_stored_scope_first():
    """Order 1: the same SQL first runs inside system_scope (passed),
    then outside -- the cached structural verdict must still fail it."""
    from sqlalchemy import create_engine

    from mirror_memory.core.tenancy import system_scope

    from mirror_memory.core.models import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with system_scope(reason="task-1a: first run inside system scope"):
        with engine.connect() as conn:
            conn.exec_driver_sql(LEAK_SQL, (1, "u1")).all()

    # outside any scope the same statement must raise
    with pytest.raises(AuditViolation):
        with engine.connect() as conn:
            conn.exec_driver_sql(LEAK_SQL, (1, "u1")).all()


def test_cache_context_pass_not_stored_outside_first():
    """Order 2: the same SQL first fails outside, then runs inside
    system_scope -- the cached *violation* must not leak into the scoped
    run (and afterwards must still fail outside)."""
    from sqlalchemy import create_engine

    from mirror_memory.core.tenancy import system_scope

    from mirror_memory.core.models import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with pytest.raises(AuditViolation):
        with engine.connect() as conn:
            conn.exec_driver_sql(LEAK_SQL, (1, "u1")).all()

    with system_scope(reason="task-1a: scoped run after an out-of-scope failure"):
        with engine.connect() as conn:
            conn.exec_driver_sql(LEAK_SQL, (1, "u1")).all()

    with pytest.raises(AuditViolation):
        with engine.connect() as conn:
            conn.exec_driver_sql(LEAK_SQL, (1, "u1")).all()


def test_audit_sql_refuses_to_run_inside_context_pass():
    """Hard guard: if a future refactor routes scoped statements into
    audit_sql, the invariant becomes a loud error instead of a cached
    'pass'."""
    from mirror_memory.core.sql_audit import audit_sql
    from mirror_memory.core.tenancy import system_scope

    with system_scope(reason="task-1a invariant guard"):
        with pytest.raises(RuntimeError):
            audit_sql(LEAK_SQL)
