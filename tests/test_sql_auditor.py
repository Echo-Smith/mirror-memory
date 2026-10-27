"""M1 task 3 -- test-time SQL auditor.

Rule tests run the auditor directly on statement text (fast, precise);
the counterexamples prove the auditor catches the classic leak shapes.
A separate integration test proves the installed listener hooks real
SQLAlchemy execution.
"""

from __future__ import annotations

import pytest

from mirror_memory.core.sql_audit import AuditViolation, _dialect_var, audit_sql


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


# ---------------------------------------------------------------------------
# M2 patch: the catalog whitelist is structural, not textual
#
# The old text patterns exempted any statement containing ``FROM
# pg_catalog.`` ANYWHERE -- which whitelisted tenant reads wrapped in a
# catalog subquery -- and ``SELECT pg_catalog.`` as a prefix, which
# whitelisted ``SELECT pg_catalog.version() FROM mm_beliefs``.  Catalog
# statements are now recognised from the parse tree (every relation must
# be pg_catalog / information_schema); a mixed statement falls through to
# the normal tenant rules.  The other whitelist patterns were tightened
# the same way: a prefix anchor alone is a hole, because psycopg runs
# multi-statement query strings (``BEGIN; SELECT ...``).
# ---------------------------------------------------------------------------


def _audit_as(sql: str, dialect: str):
    token = _dialect_var.set(dialect)
    try:
        audit_sql(sql)
        return None
    except AuditViolation as exc:
        return exc


@pytest.mark.parametrize(
    "label, sql",
    [
        ("catalog subquery", "SELECT * FROM mm_beliefs WHERE id IN (SELECT 1 FROM pg_catalog.pg_class)"),
        ("catalog join", "SELECT c.relname FROM pg_catalog.pg_class c JOIN mm_beliefs b ON b.key = c.relname"),
        ("catalog union", "SELECT id FROM mm_beliefs UNION SELECT 1 FROM pg_catalog.pg_class"),
        ("catalog cte", "WITH c AS (SELECT 1 FROM pg_catalog.pg_class) SELECT * FROM mm_beliefs, c"),
        ("catalog function + tenant FROM", "SELECT pg_catalog.version() FROM mm_beliefs"),
        ("bare pg_ prefix mix", "SELECT * FROM mm_beliefs WHERE EXISTS (SELECT 1 FROM pg_tables)"),
    ],
)
@pytest.mark.parametrize("dialect", ["postgresql", "sqlite"])
def test_counterexample_catalog_mix_is_a_violation(label, sql, dialect):
    assert _audit_as(sql, dialect) is not None, (label, dialect)


@pytest.mark.parametrize(
    "label, sql",
    [
        ("reflection join", "SELECT c.relname FROM pg_catalog.pg_class c "
                            "JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid"),
        ("catalog function only", "SELECT pg_catalog.version()"),
        ("information_schema columns", "SELECT column_name FROM information_schema.columns "
                                       "WHERE table_name = 'mm_beliefs'"),
        ("bare pg_ relation", "SELECT relname FROM pg_class ORDER BY relname LIMIT 1"),
    ],
)
def test_catalog_only_statements_pass(label, sql):
    """SQLAlchemy's own create_all / reflection statements: every relation
    is a catalog relation, so they carry no tenant data and stay exempt."""
    assert _audit_as(sql, "postgresql") is None


@pytest.mark.parametrize(
    "label, sql",
    [
        ("BEGIN smuggle", "BEGIN; SELECT * FROM mm_beliefs"),
        ("COMMIT smuggle", "COMMIT; SELECT * FROM mm_beliefs"),
        ("SAVEPOINT smuggle", "SAVEPOINT sa_1; SELECT * FROM mm_beliefs"),
        ("PRAGMA smuggle", "PRAGMA table_info(mm_beliefs); SELECT * FROM mm_beliefs"),
        ("CREATE TABLE smuggle", "CREATE TABLE mm_tmp (id int); SELECT * FROM mm_beliefs"),
        ("DROP TABLE smuggle", "DROP TABLE IF EXISTS mm_tmp; SELECT * FROM mm_beliefs"),
        ("ALTER TABLE smuggle", "ALTER TABLE mm_tmp ADD COLUMN c int; SELECT * FROM mm_beliefs"),
    ],
)
def test_counterexample_multi_statement_smuggle_is_a_violation(label, sql):
    """A whitelist prefix must not whitewash the rest of a multi-statement
    query string (psycopg executes them all)."""
    assert _audit_as(sql, "postgresql") is not None


def test_counterexample_create_table_as_select_is_a_violation():
    """CTAS reads every row of the source table and parks them in a new
    one -- the old ``create table`` prefix exempted it wholesale."""
    assert _audit_as("CREATE TABLE stolen AS SELECT * FROM mm_beliefs", "postgresql") is not None


def test_single_statement_ddl_still_exempt():
    assert _audit_as("CREATE TABLE mm_tmp (id int)", "postgresql") is None
    assert _audit_as("DROP TABLE IF EXISTS mm_tmp", "postgresql") is None
    assert _audit_as("ALTER TABLE mm_tmp ADD COLUMN c VARCHAR(64)", "postgresql") is None
    assert _audit_as("SAVEPOINT sa_1", "postgresql") is None
    assert _audit_as("ROLLBACK TO SAVEPOINT sa_1", "postgresql") is None
    assert _audit_as("RELEASE SAVEPOINT sa_1", "postgresql") is None


# ---------------------------------------------------------------------------
# M2 batch-2 task 0: tenant table names must not look like catalog relations
# ---------------------------------------------------------------------------


def test_counterexample_pg_prefixed_tenant_table_would_escape_the_audit(monkeypatch):
    """The hazard the metadata check exists for: an unqualified ``pg_*``
    relation is classified as PostgreSQL catalog metadata, so a tenant
    table with that prefix would be read without any tenant predicate and
    the auditor would wave the statement through."""
    import mirror_memory.core.sql_audit as audit

    monkeypatch.setattr(audit, "_tenant_table_names", lambda: {"pg_beliefs_t0", "mm_beliefs"})
    # same shape against a real tenant table is a violation ...
    assert _audit_as("SELECT * FROM mm_beliefs WHERE user_id = ?", "postgresql") is not None
    # ... but the reserved prefix makes the identical statement pass
    assert _audit_as("SELECT * FROM pg_beliefs_t0 WHERE user_id = ?", "postgresql") is None


def test_counterexample_pg_prefixed_tenant_table_name_is_refused():
    from mirror_memory.core.tenancy import TenantError, assert_tenant_table_names

    with pytest.raises(TenantError, match="pg_beliefs"):
        assert_tenant_table_names(["mm_beliefs", "pg_beliefs"])
    with pytest.raises(TenantError):
        assert_tenant_table_names(["PG_Shadow"])  # case-insensitive, like the auditor


def test_live_metadata_passes_the_table_name_check():
    from mirror_memory.core.models import Base
    from mirror_memory.core.tenancy import assert_tenant_table_names

    assert_tenant_table_names(Base.metadata.tables)
    assert not any(name.startswith("pg_") for name in Base.metadata.tables)


# ---------------------------------------------------------------------------
# M2 batch-2: the cast tail must stop before a JOIN (reflection shapes)
# ---------------------------------------------------------------------------


def test_cast_tail_stops_before_a_join_so_reflection_parses():
    """SQLAlchemy's foreign-key reflection: ``%(c)s::VARCHAR LEFT OUTER JOIN
    pg_catalog.pg_class`` -- the old keyword stop-list did not know JOIN,
    swallowed the join into the cast, and the statement failed to parse
    (a false fail-closed verdict on a catalog-only statement)."""
    sql = (
        "SELECT pg_catalog.pg_class.relname FROM pg_catalog.pg_class "
        "LEFT OUTER JOIN pg_catalog.pg_constraint ON pg_catalog.pg_class.oid = "
        "pg_catalog.pg_constraint.conrelid AND pg_catalog.pg_constraint.contype = "
        "%(contype_1)s::VARCHAR LEFT OUTER JOIN pg_catalog.pg_class AS cls_ref "
        "ON cls_ref.oid = pg_catalog.pg_constraint.confrelid"
    )
    assert _audit_as(sql, "postgresql") is None
    # The same shape over tenant tables keeps every table visible: the
    # JOINed table without predicates is still a violation.
    leaky = (
        "SELECT b.id FROM mm_beliefs AS b LEFT OUTER JOIN mm_evidence AS e "
        "ON b.id = e.belief_id AND e.kind = %(k)s::VARCHAR "
        "WHERE b.tenant_id = %(t)s AND b.app_id = %(a)s"
    )
    assert _audit_as(leaky, "postgresql") is not None


@pytest.mark.parametrize(
    "cast",
    [
        "TIMESTAMP WITHOUT TIME ZONE",
        "TIMESTAMP WITH TIME ZONE",
        "DOUBLE PRECISION",
        "CHARACTER VARYING(64)",
        "VARCHAR(64)",
        "TEXT[]",
        "INTEGER",
    ],
)
def test_multi_word_cast_types_still_normalise(cast):
    sql = (
        f"SELECT id FROM mm_beliefs WHERE created_at = %(c)s::{cast} "
        "AND tenant_id = %(t)s AND app_id = %(a)s"
    )
    assert _audit_as(sql, "postgresql") is None
