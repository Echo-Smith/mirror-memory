"""M2 task 3 -- tenant composite primary keys, and the auditor's
PK-addressing exemption fully retired.

On every dialect the tenant tables' primary key leads with
``(tenant_id, app_id)`` at the ORM level, so an ORM unit-of-work statement
names the tenant columns in its WHERE clause.  That let the auditor drop
its PK-addressing exemption -- first on PostgreSQL, then on SQLite too
once the composite mapping proved every UOW statement there carries the
tenant columns as well.  SQLite's physical key stays single-column
(autoincrement limitation); the runtime layers in
``tests/test_pk_addressing.py`` all remain as defense in depth.

The tests here run on whichever database the suite targets, so the same
file proves both dialects.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, inspect, update
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief
from mirror_memory.core.sql_audit import AuditViolation, audit_sql, _dialect_var
from mirror_memory.core.tenancy import (
    TenantMismatchError,
    WriteGuardViolation,
    fixture_write_scope,
    tenant_scope,
)

PG_URL = os.environ.get("MM_DATABASE_URL", "")
IS_PG = PG_URL.startswith("postgres")


def _engine():
    if IS_PG:
        eng = create_engine(PG_URL)
        Base.metadata.drop_all(eng)
        Base.metadata.create_all(eng)
        return eng
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


def _dialect(sql: str, dialect: str):
    token = _dialect_var.set(dialect)
    try:
        audit_sql(sql)
        return None
    except AuditViolation as exc:
        return exc
    finally:
        _dialect_var.reset(token)


# ---------------------------------------------------------------------------
# Primary-key shape
# ---------------------------------------------------------------------------


def test_every_tenant_table_has_a_tenant_composite_primary_key():
    """The mapping-level key is composite everywhere, on both dialects.

    The physical constraint differs (SQLite narrows the autoincrement
    tables), but the ORM key must not: ``session.get``, the identity map
    and the PG auditor all read the mapping.
    """
    for name, table in Base.metadata.tables.items():
        pk = [c.name for c in table.primary_key.columns]
        assert pk[:2] == ["tenant_id", "app_id"], (name, pk)
        assert len(pk) >= 3, (name, pk)


def test_pg_physical_primary_key_is_composite():
    if not IS_PG:
        pytest.skip("physical composite PK only exists on PostgreSQL")
    eng = _engine()
    insp = inspect(eng)
    names = sorted(Base.metadata.tables)
    assert names
    for name in names:
        cols = insp.get_pk_constraint(name)["constrained_columns"]
        assert cols[:2] == ["tenant_id", "app_id"], (name, cols)


def test_sqlite_physical_primary_key_stays_single_column_on_autoincrement_tables():
    if IS_PG:
        pytest.skip("SQLite DDL narrowing is SQLite-specific")
    eng = _engine()
    insp = inspect(eng)
    # mm_beliefs has an autoincrement id: narrowed to the rowid alias.
    assert insp.get_pk_constraint("mm_beliefs")["constrained_columns"] == ["id"]
    # A string-keyed tenant table keeps the full composite constraint.
    assert insp.get_pk_constraint("mm_snapshots")["constrained_columns"] == [
        "tenant_id", "app_id", "id",
    ]


def test_identity_map_key_includes_the_tenant():
    """The identity map is keyed by the full composite PK, so two tenants'
    rows with the same surrogate id are distinct entries."""
    eng = _engine()
    factory = sessionmaker(bind=eng)
    with tenant_scope("tA", "x"), fixture_write_scope("seed tA"):
        s = factory()
        a = Belief(user_id="u1", dimension="d", key="k", claim_text="A")
        s.add(a)
        s.commit()
        a_id = a.id
        s.close()
    with tenant_scope("tB", "x"), fixture_write_scope("seed tB"):
        s = factory()
        b = Belief(user_id="u1", dimension="d", key="k", claim_text="B")
        s.add(b)
        s.commit()
        b_id = b.id
        s.close()
    # Each tenant resolves only its own row through the identity-keyed read
    # path, even when handed the other tenant's key.
    with tenant_scope("tA", "x"):
        s = factory()
        assert s.get(Belief, ("tA", "x", a_id)).claim_text == "A"
        assert s.get(Belief, ("tB", "x", b_id)) is None
        s.close()
    with tenant_scope("tB", "x"):
        s = factory()
        assert s.get(Belief, ("tB", "x", b_id)).claim_text == "B"
        assert s.get(Belief, ("tA", "x", a_id)) is None
        s.close()


def test_session_get_accepts_the_legacy_single_value_key():
    """~40 call sites pass only the business id; the wrapper prefixes the
    ambient tenant instead of requiring every caller to change."""
    eng = _engine()
    factory = sessionmaker(bind=eng)
    with tenant_scope("tA", "x"), fixture_write_scope("seed"):
        s = factory()
        row = Belief(user_id="u1", dimension="d", key="k", claim_text="legacy")
        s.add(row)
        s.commit()
        row_id = row.id
        s.close()
    with tenant_scope("tA", "x"):
        s = factory()
        assert s.get(Belief, row_id).claim_text == "legacy"
        s.close()
    with tenant_scope("tB", "x"):
        s = factory()
        assert s.get(Belief, row_id) is None
        s.close()


# ---------------------------------------------------------------------------
# Auditor: exemption retired on PG, kept on SQLite
# ---------------------------------------------------------------------------

PK_ONLY = "UPDATE mm_beliefs SET claim_text = ? WHERE id = ?"
PK_WITH_TENANT = (
    "UPDATE mm_beliefs SET claim_text = ? "
    "WHERE mm_beliefs.tenant_id = ? AND mm_beliefs.app_id = ? AND mm_beliefs.id = ?"
)


def test_pg_auditor_rejects_pk_only_update():
    if not IS_PG:
        pytest.skip("the exemption is retired on PostgreSQL only")
    assert _dialect(PK_ONLY, "postgresql") is not None


def test_pg_auditor_accepts_pk_update_with_tenant_predicates():
    assert _dialect(PK_WITH_TENANT, "postgresql") is None


def test_pk_only_update_violates_on_every_dialect():
    """The PK-addressing exemption is retired on BOTH dialects: the
    composite ORM mapping puts the tenant columns into every unit-of-work
    statement on SQLite too, so nothing needs the exemption any more."""
    for dialect in ("postgresql", "sqlite"):
        assert _dialect(PK_ONLY, dialect) is not None, dialect


def test_verdict_cache_shares_one_verdict_per_text():
    """The judge is dialect-independent now: the same statement text yields
    the same verdict no matter which dialect audited it first."""
    for dialect in ("sqlite", "postgresql", "sqlite"):
        assert _dialect(PK_ONLY, dialect) is not None
    for dialect in ("postgresql", "sqlite"):
        assert _dialect(PK_WITH_TENANT, dialect) is None


# ---------------------------------------------------------------------------
# Counterexample: a PK-addressed write cannot reach another tenant's row
# ---------------------------------------------------------------------------


def test_counterexample_orm_pk_update_cannot_cross_tenants():
    """Writing through an ORM object built in tenant A must not touch
    tenant B's row, even when the surrogate id is fabricated to match."""
    eng = _engine()
    factory = sessionmaker(bind=eng)
    with tenant_scope("tB", "x"), fixture_write_scope("seed tB"):
        s = factory()
        victim = Belief(user_id="u1", dimension="d", key="k", claim_text="TB-SECRET")
        s.add(victim)
        s.commit()
        victim_id = victim.id
        s.close()

    with tenant_scope("tA", "x"):
        s = factory()
        try:
            # Construction of a foreign-PK object trips the write guard
            # before any statement reaches the database.
            with pytest.raises((WriteGuardViolation, TenantMismatchError)):
                hostile = Belief(id=victim_id, user_id="u1", dimension="d", key="k")
                s.add(hostile)
                hostile.claim_text = "HIJACKED"
                s.commit()
        finally:
            s.rollback()
            s.close()

    with tenant_scope("tB", "x"):
        s = factory()
        assert s.get(Belief, victim_id).claim_text == "TB-SECRET"
        s.close()


def test_counterexample_core_update_matching_zero_rows():
    """A core UPDATE addressed by primary key alone cannot match another
    tenant's row: the loader criteria adds the tenant predicate, so the
    statement affects 0 rows and the target row is unchanged."""
    eng = _engine()
    factory = sessionmaker(bind=eng)
    with tenant_scope("tB", "x"), fixture_write_scope("seed tB"):
        s = factory()
        victim = Belief(user_id="u1", dimension="d", key="k", claim_text="TB-SECRET")
        s.add(victim)
        s.commit()
        victim_id = victim.id
        s.close()

    with tenant_scope("tA", "x"):
        s = factory()
        result = s.execute(
            update(Belief)
            .where(Belief.id == victim_id)
            .values(claim_text="HIJACKED")
            .execution_options(synchronize_session=False)
        )
        s.commit()
        assert result.rowcount == 0
        s.close()

    with tenant_scope("tB", "x"):
        s = factory()
        assert s.get(Belief, victim_id).claim_text == "TB-SECRET"
        s.close()
