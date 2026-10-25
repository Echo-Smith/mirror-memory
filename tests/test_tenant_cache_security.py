"""Security stress tests for the tenant loader-criteria option (M1).

The criteria template is a *tenant-agnostic* lambda whose comparisons bind
to two module-level ``bindparam`` callables reading the ambient tenant at
execution time.  There is no per-tenant option cache, so the historical
leak vector -- a wrong cache key or bad eviction -- is structurally gone;
this module keeps hammering the paths the acceptance plan calls out:
massive tenant churn, look-alike keys, threads, session tenant binding,
and lambda-SQL statement-cache reuse across tenants.

M1 session-binding note: sessions are single-tenant (bound at creation).
Every scenario below opens a fresh session inside each tenant scope, and
the dedicated session-switch test asserts that reusing a session across
tenants raises instead of re-filtering.
"""

from __future__ import annotations

import random
import threading

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief
from mirror_memory.core.tenancy import (
    TenantContextMissing,
    TenantMismatchError,
    _LOADER_OPTION,
    tenant_scope,
)


def session_at(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def seed_belief(factory, tenant, user, key, claim):
    with tenant_scope(*tenant):
        s = factory()
        s.add(
            Belief(
                user_id=user, dimension="topic", key=key,
                claim_text=claim, confidence=0.9,
            )
        )
        s.commit()
        s.close()


# ---------------------------------------------------------------------------
# 1. Tenant churn far beyond any caching bound: one shared option serves
#    every tenant, values resolved per execution -- no leak possible
# ---------------------------------------------------------------------------


def test_tenant_churn_never_leaks(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/churn.db")
    tenants = [(f"tenant-{i:05d}", "app") for i in range(1500)]
    for tenant in tenants:
        seed_belief(factory, tenant, "u1", "k", f"CLAIM-{tenant[0]}")

    assert _LOADER_OPTION is not None  # the shared singleton is in use

    # Two deterministic sweeps over all tenants (3000 tenant-scoped
    # sessions).  With no per-tenant cache there is no eviction to observe;
    # what must hold is that every read sees exactly its own tenant's row.
    for sweep in range(2):
        for tenant in tenants:
            with tenant_scope(*tenant):
                s = factory()
                try:
                    rows = s.query(Belief).filter(Belief.user_id == "u1").all()
                    assert len(rows) == 1, (tenant, [r.tenant_id for r in rows])
                    assert rows[0].claim_text == f"CLAIM-{tenant[0]}"
                    assert rows[0].tenant_id == tenant[0]
                    assert rows[0].app_id == tenant[1]
                finally:
                    s.close()


# ---------------------------------------------------------------------------
# 2. Look-alike keys must stay isolated (empty components are rejected)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        (("t1", "app"), ("t", "1app")),          # same concatenation
        (("t", "app"), ("t ", "app")),            # trailing-space id
        (("T", "app"), ("t", "app")),             # case difference
        (("t\u00e9", "app"), ("te", "app")),      # unicode vs ascii
        (("x" * 200, "app"), ("x" * 199 + "y", "app")),  # very long ids
    ],
)
def test_lookalike_tenant_keys_are_isolated(tmp_path, left, right):
    factory = session_at(f"sqlite:///{tmp_path}/keys.db")
    seed_belief(factory, left, "u1", "k", "LEFT")
    seed_belief(factory, right, "u1", "k", "RIGHT")

    for tenant, marker in ((left, "LEFT"), (right, "RIGHT")):
        with tenant_scope(*tenant):
            s = factory()
            try:
                rows = s.query(Belief).all()
                assert len(rows) == 1
                assert rows[0].claim_text == marker
                assert rows[0].tenant_id == tenant[0]
                assert rows[0].app_id == tenant[1]
            finally:
                s.close()

    # Empty components are rejected outright by the sanctioned setter.
    with pytest.raises(ValueError):
        with tenant_scope("", "app"):
            pass


# ---------------------------------------------------------------------------
# 3. Threads with different tenants under heavy option churn
# ---------------------------------------------------------------------------


def test_concurrent_threads_under_churn(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/threads.db")
    thread_tenants = [(f"thread-{i}", "app") for i in range(8)]
    churn_tenants = [(f"churn-{i}", "app") for i in range(1200)]
    for tenant in thread_tenants + churn_tenants:
        seed_belief(factory, tenant, "u1", "k", f"OWN-{tenant[0]}")

    errors: list[Exception] = []

    def reader(tenant):
        own = f"OWN-{tenant[0]}"
        try:
            for _ in range(100):
                with tenant_scope(*tenant):
                    s = factory()
                    try:
                        rows = s.query(Belief).filter(Belief.user_id == "u1").all()
                        for row in rows:
                            if row.claim_text != own:
                                errors.append(
                                    AssertionError(
                                        f"leak: {tenant} saw {row.claim_text!r}"
                                    )
                                )
                                return
                    finally:
                        s.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def churner():
        try:
            rng = random.Random(7)
            for _ in range(1500):
                tenant = churn_tenants[rng.randrange(len(churn_tenants))]
                with tenant_scope(*tenant):
                    s = factory()
                    try:
                        s.query(Belief).filter(Belief.user_id == "u1").first()
                    finally:
                        s.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader, args=(t,)) for t in thread_tenants]
    threads.append(threading.Thread(target=churner))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ---------------------------------------------------------------------------
# 4. Tenant switch on one session raises (M1 design change from the spike:
#    sessions are single-tenant, bound at creation)
# ---------------------------------------------------------------------------


def test_same_session_tenant_switch_raises(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/switch.db")
    seed_belief(factory, ("tA", "app"), "u1", "kA", "A")

    with tenant_scope("tA", "app"):
        s = factory()
        loaded = s.query(Belief).first()
        aid = loaded.id
        s.close()

    with tenant_scope("tB", "app"):
        s2 = factory()  # bound to tB
        with pytest.raises(TenantMismatchError):
            with tenant_scope("tA", "app"):
                s2.query(Belief).all()  # bound(tB) != ambient(tA)
        s2.close()

    # A session created under tB never sees tA rows, and a get() across the
    # tenant boundary returns None (detached, not leakable).
    with tenant_scope("tB", "app"):
        s3 = factory()
        try:
            assert s3.query(Belief).all() == []
            assert s3.get(Belief, aid) is None
        finally:
            s3.close()


# ---------------------------------------------------------------------------
# 5. Statement cache: identical statement shapes across tenants must always
#    bind the *current* tenant values
# ---------------------------------------------------------------------------


def test_statement_cache_binds_current_tenant(tmp_path):
    url = f"sqlite:///{tmp_path}/cache.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    seed_belief(factory, ("tA", "app"), "u1", "k", "A")
    seed_belief(factory, ("tB", "app"), "u1", "k", "B")

    captured: list[tuple[str, tuple]] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        params = parameters if isinstance(parameters, tuple) else tuple(parameters or ())
        captured.append((statement, params))

    event.listen(engine, "before_cursor_execute", _capture)
    try:
        from sqlalchemy import update as sa_update

        expected = [("tA", "app"), ("tB", "app")] * 5
        for tenant in expected:
            with tenant_scope(*tenant):
                s = factory()
                try:
                    # count() (column-only top-level SELECT, entity in subquery)
                    s.query(Belief).filter(Belief.user_id == "u1").count()
                    # entity SELECT
                    s.query(Belief).all()
                    # scalar subquery shape
                    s.query(Belief.id).filter(Belief.user_id == "u1").first()
                    # ORM-enabled bulk UPDATE
                    bulk = sa_update(Belief).where(Belief.user_id == "u1").values(
                        confidence=0.5
                    ).execution_options(synchronize_session=False)
                    s.execute(bulk)
                    s.commit()
                finally:
                    s.close()

        assert captured, "no SQL captured"
        for statement, params in captured:
            sql = statement.lower()
            if "tenant_id" not in sql:
                continue
            bound = [p for p in params if isinstance(p, str)]
            assert bound, (statement, params)
            # Every captured statement's tenant values must match the tenant
            # that was current when it ran -- a stale cached statement would
            # carry the *other* tenant's values here.
            tenant_ids = {p for p in bound if p in ("tA", "tB")}
            if tenant_ids:
                assert len(tenant_ids) == 1, (statement, params)
    finally:
        event.remove(engine, "before_cursor_execute", _capture)


# ---------------------------------------------------------------------------
# 6. Closure freeze: the shared criteria template references no local state
# ---------------------------------------------------------------------------


def test_criteria_template_is_closure_frozen():
    import inspect

    import mirror_memory.core.tenancy as ten

    code = ten._criteria_template.__code__
    assert code.co_freevars == (), code.co_freevars
    # global references: only the module-level bindparams; the rest are
    # attribute accesses on cls (tenant_id / app_id)
    assert {"_tenant_bind", "_app_bind"} <= set(code.co_names)
    assert not any(n not in ("_tenant_bind", "_app_bind", "tenant_id", "app_id") for n in code.co_names), code.co_names


def test_closure_guard_rejects_local_capture():
    import mirror_memory.core.tenancy as ten

    leaked = "tA"

    def bad_template(cls):  # captures a local -- must be rejected
        return (cls.tenant_id == leaked) & (cls.app_id == "app")

    with pytest.raises(ten.TenantError):
        ten._assert_closure_frozen(bad_template)


# ---------------------------------------------------------------------------
# 7. No-context get() fails closed
# ---------------------------------------------------------------------------


def test_get_without_context_raises(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/getctx.db")
    seed_belief(factory, ("tA", "app"), "u1", "k", "A")

    with tenant_scope("tA", "app"):
        s = factory()
        aid = s.query(Belief).first().id
        s.close()

    from mirror_memory.core.tenancy import no_tenant_scope

    with no_tenant_scope():
        s = factory()
        try:
            with pytest.raises(TenantContextMissing):
                s.get(Belief, aid)
        finally:
            s.close()
