"""M5 §2 patch: ``storage_stats`` authorization & tenancy boundaries.

The report is scoped by the **bound tenancy context**, never by a free caller
argument.  These tests pin that down on both backends:

* no bound context         -> fail closed (``TenantContextMissing``)
* bound context            -> only that (tenant, app) is reported
* same tenant, other app   -> separate key, never merged or overwritten
* same user in two tenants -> two distinct keys
* PostgreSQL app role+RLS  -> explicit predicate AND row security both apply
* no privileged API        -> there is no cross-tenant report to bypass with,
                              and the function takes no tenant argument
"""

from __future__ import annotations

import inspect
import os

import pytest
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import Session

from mirror_memory.core import metrics
from mirror_memory.core.models import Base, Belief as CoreBelief
from mirror_memory.core.tenancy import (
    TenantContextMissing,
    no_tenant_scope,
    tenant_scope,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _belief_row(n: int, *, tenant: str, app: str, user: str, text: str) -> dict:
    return dict(
        id=n, global_id=f"g-{tenant}-{app}-{user}-{n}",
        tenant_id=tenant, app_id=app, user_id=user,
        dimension="d", key="k", predicate="p", object="o",
        claim_text=text, polarity="pos", status="active", confidence=0.9,
    )


def _sqlite_engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'store.db'}")
    Base.metadata.create_all(eng)
    return eng


# ---------------------------------------------------------------------------
# SQLite leg
# ---------------------------------------------------------------------------


def test_no_bound_context_fails_closed(tmp_path):
    """No tenant context -> TenantContextMissing, not a default tenant."""
    eng = _sqlite_engine(tmp_path)
    with eng.begin() as c:
        c.execute(insert(CoreBelief.__table__).values(
            _belief_row(1, tenant="T1", app="a", user="u1", text="x")))
    # the suite autouse-binds a default tenant; clear it explicitly so this
    # test really exercises the unbound path (fail-closed, no default).
    with no_tenant_scope():
        with pytest.raises(TenantContextMissing):
            metrics.storage_stats(eng)


def test_bound_context_scopes_the_report(tmp_path):
    eng = _sqlite_engine(tmp_path)
    with eng.begin() as c:
        for n, (t, a, u) in enumerate(
            [("T1", "a", "u1"), ("T1", "a", "u2"), ("T2", "a", "u1"),
             ("T1", "other-app", "u1")], start=1
        ):
            c.execute(insert(CoreBelief.__table__).values(
                _belief_row(n, tenant=t, app=a, user=u, text="marked-ABC")))

    with tenant_scope("T1", "a"):
        stats = metrics.storage_stats(eng)
    # only the bound (tenant, app) appears
    assert set(stats) == {("T1", "a", "u1"), ("T1", "a", "u2")}
    assert stats[("T1", "a", "u1")]["mm_beliefs"]["rows"] == 1
    # same tenant, DIFFERENT app, same user id -> must be a separate key
    with tenant_scope("T1", "other-app"):
        other_app = metrics.storage_stats(eng)
    assert set(other_app) == {("T1", "other-app", "u1")}
    # same user id in another tenant -> separate key, and not visible from T1
    with tenant_scope("T2", "a"):
        t2 = metrics.storage_stats(eng)
    assert set(t2) == {("T2", "a", "u1")}
    assert ("T2", "a", "u1") not in stats


def test_report_keys_carry_tenant_app_user(tmp_path):
    """Keys must include all three components so equal user ids in different
    (tenant, app) pairs can never overwrite each other in the map."""
    eng = _sqlite_engine(tmp_path)
    with eng.begin() as c:
        c.execute(insert(CoreBelief.__table__).values(
            _belief_row(1, tenant="T1", app="a", user="same", text="one")))
        c.execute(insert(CoreBelief.__table__).values(
            _belief_row(2, tenant="T1", app="b", user="same", text="two")))
    with tenant_scope("T1", "a"):
        a = metrics.storage_stats(eng)
    with tenant_scope("T1", "b"):
        b = metrics.storage_stats(eng)
    assert ("T1", "a", "same") in a and ("T1", "b", "same") not in a
    assert ("T1", "b", "same") in b and ("T1", "a", "same") not in b


# ---------------------------------------------------------------------------
# API surface: no free tenant argument, no cross-tenant escape hatch
# ---------------------------------------------------------------------------


def test_storage_stats_takes_no_tenant_argument():
    params = inspect.signature(metrics.storage_stats).parameters
    assert "tenant" not in params, "tenant must come from the bound context"
    assert set(params) == {"engine"}


def test_no_cross_tenant_storage_api():
    assert not hasattr(metrics, "storage_stats_all_tenants"), (
        "a cross-tenant capacity report would need its own verifiable admin "
        "authorization and privileged connection; it must not come back as a "
        "system_scope-based bypass"
    )


def test_storage_stats_uses_no_system_scope():
    import ast

    tree = ast.parse(inspect.getsource(metrics.storage_stats).lstrip())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Name) and n.func.id == "system_scope")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "system_scope"))
    ]
    assert not calls, "storage_stats must not rely on system_scope as authorization"


def test_storage_stats_binds_values_not_identifiers(tmp_path):
    """The SQL carries the tenant as BOUND parameters, not interpolated text."""
    eng = _sqlite_engine(tmp_path)
    with eng.begin() as c:
        c.execute(insert(CoreBelief.__table__).values(
            _belief_row(1, tenant="T1", app="a", user="u1", text="x")))
    # a tenant id with a quote/SQL fragment must be treated as data
    hostile = "T1' OR '1'='1"
    with tenant_scope(hostile, "a"):
        stats = metrics.storage_stats(eng)
    assert stats == {}, "a hostile tenant id must not match rows by injection"


# ---------------------------------------------------------------------------
# PostgreSQL leg: real app role under RLS
# ---------------------------------------------------------------------------


def _pg_seed(engine, rows):
    """Seed under each row's OWN bound tenant: the RLS WITH CHECK policy refuses
    a row whose tenant does not match the transaction's tenant, so the write
    scope and tenant scope must both be open per row."""
    from mirror_memory.core.tenancy import fixture_write_scope as _fws
    for n, (t, a, u) in enumerate(rows, start=1):
        with _fws("storage_stats test: seed"):
            with tenant_scope(t, a):
                with engine.begin() as c:
                    c.execute(insert(CoreBelief.__table__).values(
                        _belief_row(n, tenant=t, app=a, user=u, text="marked-ABC")))


def _pg_engine(pg_app_url):
    from mirror_memory.core.rls import prepare_runtime_engine

    eng = create_engine(pg_app_url)
    prepare_runtime_engine(eng)
    return eng


@pytest.mark.skipif(
    not (os.environ.get("MM_DATABASE_URL") or "").startswith("postgres"),
    reason="needs PostgreSQL (MM_DATABASE_URL)",
)
def test_pg_app_role_rls_storage_stats(pg_app_url):
    """Real PG app role under RLS: bound tenant sees its rows, another tenant
    does not, and a raw unpredicated read is still narrowed by RLS."""
    from sqlalchemy import text

    eng = _pg_engine(pg_app_url)
    try:
        _pg_seed(eng, [("tA", "app", "uSame"), ("tB", "app", "uSame")])

        with tenant_scope("tA", "app"):
            stats = metrics.storage_stats(eng)
        assert set(stats) == {("tA", "app", "uSame")}
        assert stats[("tA", "app", "uSame")]["mm_beliefs"]["rows"] >= 1

        with tenant_scope("tB", "app"):
            stats_b = metrics.storage_stats(eng)
        assert set(stats_b) == {("tB", "app", "uSame")}

        # RLS itself: a deliberately unpredicated read under tenant A must not
        # see tenant B's row -- this is the database policy, not our WHERE.
        # (The probe is intentionally predicate-free, so the SQL auditor --
        # which exists to catch engine code doing that -- is switched off for
        # this one diagnostic statement only.)
        from mirror_memory.core.sql_audit import no_audit_scope

        with no_audit_scope(), tenant_scope("tA", "app"):
            with eng.connect() as conn:
                tx = conn.begin()
                try:
                    seen = {
                        (r[0], r[1])
                        for r in conn.execute(
                            text("SELECT DISTINCT tenant_id, app_id FROM mm_beliefs")
                        )
                    }
                finally:
                    tx.rollback()
        assert seen == {("tA", "app")}, f"RLS did not narrow the read: {seen}"
    finally:
        eng.dispose()


@pytest.mark.skipif(
    not (os.environ.get("MM_DATABASE_URL") or "").startswith("postgres"),
    reason="needs PostgreSQL (MM_DATABASE_URL)",
)
def test_pg_no_bound_context_fails_closed(pg_app_url):
    eng = _pg_engine(pg_app_url)
    try:
        with no_tenant_scope():
            with pytest.raises(TenantContextMissing):
                metrics.storage_stats(eng)
    finally:
        eng.dispose()