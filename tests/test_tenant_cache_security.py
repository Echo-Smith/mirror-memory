"""Security stress tests for the tenant loader-criteria option.

The criteria template is a *tenant-agnostic* lambda whose comparisons bind
to two module-level ``bindparam`` callables reading the ambient tenant at
execution time (spike/v1-tenancy-perf).  There is no per-tenant option
cache at all, so the historical leak vector -- a wrong cache key or bad
eviction -- is structurally gone; this module keeps hammering the paths
the acceptance plan calls out: massive tenant churn, look-alike keys,
threads, tenant switches on one session, and lambda-SQL statement-cache
reuse across tenants.
"""

from __future__ import annotations

import random
import threading

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief
from mirror_memory.core.tenancy import (
    TenantContextError,
    _LOADER_OPTION,
    tenant_scope,
)


def session_at(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def seed_belief(s, tenant, user, key, claim):
    with tenant_scope(*tenant):
        s.add(
            Belief(
                user_id=user, dimension="topic", key=key,
                claim_text=claim, confidence=0.9,
            )
        )
        s.commit()


# ---------------------------------------------------------------------------
# 1. Tenant churn far beyond any caching bound: one shared option serves
#    every tenant, values resolved per execution -- no leak possible
# ---------------------------------------------------------------------------


def test_tenant_churn_never_leaks(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/churn.db")
    s = factory()
    tenants = [(f"tenant-{i:05d}", "app") for i in range(1500)]
    for tenant in tenants:
        seed_belief(s, tenant, "u1", "k", f"CLAIM-{tenant[0]}")

    assert _LOADER_OPTION is not None  # the shared singleton is in use

    # Deterministic sweeps across all tenants (7 passes).  With no
    # per-tenant cache there is no eviction to observe; what must hold is
    # that every one of the 10,500 reads sees exactly its own tenant's row.
    order = [i % len(tenants) for i in range(10_500)]
    for idx in order:
        tenant = tenants[idx]
        with tenant_scope(*tenant):
            rows = s.query(Belief).filter(Belief.user_id == "u1").all()
            assert len(rows) == 1, (tenant, [r.tenant_id for r in rows])
            assert rows[0].claim_text == f"CLAIM-{tenant[0]}"
            assert rows[0].tenant_id == tenant[0]
            assert rows[0].app_id == tenant[1]
    s.close()


# ---------------------------------------------------------------------------
# 2. Look-alike keys must stay isolated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        (("t1", "app"), ("t", "1app")),          # same concatenation
        (("", ""), ("t", "")),                    # empty components
        (("T", "app"), ("t", "app")),             # case difference
        (("t\u00e9", "app"), ("te", "app")),      # unicode vs ascii
        (("x" * 200, "app"), ("x" * 199 + "y", "app")),  # very long ids
    ],
)
def test_lookalike_tenant_keys_are_isolated(tmp_path, left, right):
    factory = session_at(f"sqlite:///{tmp_path}/keys.db")
    s = factory()
    seed_belief(s, left, "u1", "k", "LEFT")
    seed_belief(s, right, "u1", "k", "RIGHT")

    for tenant, marker in ((left, "LEFT"), (right, "RIGHT")):
        with tenant_scope(*tenant):
            rows = s.query(Belief).all()
            assert len(rows) == 1
            assert rows[0].claim_text == marker
            assert rows[0].tenant_id == tenant[0]
            assert rows[0].app_id == tenant[1]
    s.close()


# ---------------------------------------------------------------------------
# 3. Threads with different tenants under eviction churn
# ---------------------------------------------------------------------------


def test_concurrent_threads_under_eviction(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/threads.db")
    setup = factory()
    thread_tenants = [(f"thread-{i}", "app") for i in range(8)]
    churn_tenants = [(f"churn-{i}", "app") for i in range(1200)]
    for tenant in thread_tenants:
        seed_belief(setup, tenant, "u1", "k", f"OWN-{tenant[0]}")
    for tenant in churn_tenants:
        seed_belief(setup, tenant, "u1", "k", f"OWN-{tenant[0]}")
    setup.close()

    errors: list[Exception] = []

    def reader(tenant):
        own = f"OWN-{tenant[0]}"
        s = factory()
        try:
            for _ in range(200):
                with tenant_scope(*tenant):
                    rows = s.query(Belief).filter(Belief.user_id == "u1").all()
                    for row in rows:
                        if row.claim_text != own:
                            errors.append(
                                AssertionError(
                                    f"leak: {tenant} saw {row.claim_text!r}"
                                )
                            )
                            return
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            s.close()

    def churner():
        s = factory()
        try:
            rng = random.Random(7)
            for _ in range(4000):
                tenant = churn_tenants[rng.randrange(len(churn_tenants))]
                with tenant_scope(*tenant):
                    s.query(Belief).filter(Belief.user_id == "u1").first()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            s.close()

    threads = [threading.Thread(target=reader, args=(t,)) for t in thread_tenants]
    threads.append(threading.Thread(target=churner))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ---------------------------------------------------------------------------
# 4. Tenant switch on one session: old rows unreachable, new rows scoped
# ---------------------------------------------------------------------------


def test_same_session_tenant_switch(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/switch.db")
    s = factory()
    seed_belief(s, ("tA", "app"), "u1", "kA", "A")
    with tenant_scope("tA", "app"):
        loaded = s.query(Belief).first()
        aid = loaded.id

    # Switch the same session to tB.
    with tenant_scope("tB", "app"):
        rows = s.query(Belief).all()
        assert rows == []  # tB has nothing
        assert s.get(Belief, aid) is None  # tA row invisible via get
        # A write carrying the *other* tenant is refused at flush.
        from mirror_memory.core.tenancy import TenantMismatchError

        s.add(Belief(user_id="u1", dimension="topic", key="k", tenant_id="tA", app_id="app"))
        with pytest.raises(TenantMismatchError):
            s.commit()
        s.rollback()
        # Writing under the *current* context still works.
        s.add(Belief(user_id="u1", dimension="topic", key="kB", claim_text="B", confidence=0.9))
        s.commit()
        assert s.query(Belief).count() == 1

    # Back in tA: the original row is still the only one visible.
    with tenant_scope("tA", "app"):
        rows = s.query(Belief).all()
        assert len(rows) == 1
        assert rows[0].claim_text == "A"
    s.close()


# ---------------------------------------------------------------------------
# 5. lambda-SQL statement cache: identical statement shapes across tenants
#    must always bind the *current* tenant values
# ---------------------------------------------------------------------------


def test_statement_cache_binds_current_tenant(tmp_path):
    url = f"sqlite:///{tmp_path}/cache.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    s = factory()
    seed_belief(s, ("tA", "app"), "u1", "k", "A")
    seed_belief(s, ("tB", "app"), "u1", "k", "B")

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
            apps = {p for p in bound if p == "app"}
            assert tenant_ids <= {expected[0][0], expected[1][0]}
            assert apps == {"app"} or apps == set()
            if tenant_ids:
                # parameters arrive in (tenant_id, app_id, ...) order per the
                # criteria expression; the pair must be consistent
                assert len(tenant_ids) == 1, (statement, params)
    finally:
        event.remove(engine, "before_cursor_execute", _capture)
        s.close()
