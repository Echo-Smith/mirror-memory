"""Variant experiment: tenant-agnostic loader criteria options.

V0: current per-tenant lambda option (scalar closure, tracked)
V1: singleton plain-expression criteria + bindparam(callable_)
V2: singleton lambda criteria + track_closure_variables=False + bindparam(callable_)

Each variant: SELECT isolation across tenants, ORM UPDATE scoping, then an
interleaved timing comparison against inject-off.
"""
import statistics
import time

from sqlalchemy import bindparam, create_engine, event, update as sa_update
from sqlalchemy.orm import Session, sessionmaker, with_loader_criteria

import mirror_memory.core.tenancy as ten
from mirror_memory.core.models import Base, Belief, MemoryPreference
from mirror_memory.core.repository import record_claim, set_memory_enabled

TENANT = ("default", "default")
OTHER = ("other", "app")
USER = "u1"

engine = create_engine("sqlite://")
Base.metadata.create_all(engine)
factory = sessionmaker(bind=engine)

with ten.tenant_scope(*TENANT):
    s = factory()
    set_memory_enabled(s, USER, True)
    s.commit()
    for i in range(30):
        record_claim(s, USER, dimension="topic", key=f"k{i}",
                     claim_text=f"claim {i}", confidence=0.8, session_id=f"s{i}")
    s.commit()
    s.close()
with ten.tenant_scope(*OTHER):
    s = factory()
    set_memory_enabled(s, USER, True)
    s.commit()
    record_claim(s, USER, dimension="topic", key="k0",
                 claim_text="OTHER-ROW", confidence=0.9, session_id="sx")
    s.commit()
    s.close()

_tenant_bind = bindparam("mm_spike_tenant", callable_=lambda: ten._tenant_var.get()[0])
_app_bind = bindparam("mm_spike_app", callable_=lambda: ten._tenant_var.get()[1])

V1 = with_loader_criteria(
    ten.TenantScoped,
    (Belief.__table__.c.tenant_id == _tenant_bind) & (Belief.__table__.c.app_id == _app_bind),
    include_aliases=True,
)


def _v2_crit(cls):
    return (cls.tenant_id == _tenant_bind) & (cls.app_id == _app_bind)


V2 = with_loader_criteria(ten.TenantScoped, _v2_crit, include_aliases=True,
                          track_closure_variables=False)

_orig_inject = ten._inject_tenant_criteria


def make_inject(option):
    def inject(state):
        if state.is_column_load or state.is_relationship_load:
            return
        if not state.is_orm_statement:
            return
        if not (state.is_select or state.is_update or state.is_delete):
            return
        if ten.in_system_scope():
            return
        if ten._tenant_var.get() is None:
            raise ten.TenantContextError("no ctx")
        state.statement = state.statement.options(option)
    return inject


inj_v1 = make_inject(V1)
inj_v2 = make_inject(V2)


_current_listener = None


def install(inject):
    global _current_listener
    if _current_listener is not None:
        event.remove(Session, "do_orm_execute", _current_listener)
        _current_listener = None
    if inject is not None:
        event.listen(Session, "do_orm_execute", inject)
        _current_listener = inject


def check_isolation(tag):
    with ten.tenant_scope(*TENANT):
        s = factory()
        rows = s.query(Belief).filter(Belief.user_id == USER).all()
        assert len(rows) == 30 and all(r.tenant_id == "default" for r in rows), (tag, len(rows))
        assert s.query(Belief).count() == 30, (tag, s.query(Belief).count())
        s.close()
    with ten.tenant_scope(*OTHER):
        s = factory()
        rows = s.query(Belief).filter(Belief.user_id == USER).all()
        assert len(rows) == 1 and rows[0].claim_text == "OTHER-ROW", (tag, rows)
        s.close()


def check_update(tag):
    # one MemoryPreference row per tenant exists; ORM UPDATE scoped to the
    # current tenant must touch exactly one row.
    with ten.tenant_scope(*TENANT):
        s = factory()
        upd = sa_update(MemoryPreference).where(MemoryPreference.user_id == USER).values(
            state_revision=MemoryPreference.state_revision + 1
        ).execution_options(synchronize_session=False)
        result = s.execute(upd)
        s.commit()
        s.close()
        assert result.rowcount == 1, (tag, "rowcount", result.rowcount)


for tag, inj in (("V1", inj_v1), ("V2", inj_v2)):
    install(inj)
    try:
        check_isolation(tag)
        check_update(tag)
        print(f"{tag}: SELECT isolation OK, ORM UPDATE rowcount==1 OK")
    except Exception as exc:
        print(f"{tag}: FAILED - {type(exc).__name__}: {str(exc)[:120]}")
    finally:
        install(None)

# --- interleaved timing ------------------------------------------------------
N = 300


def loop():
    s = factory()
    try:
        for _ in range(N):
            s.query(Belief).filter(Belief.user_id == USER, Belief.status == "active").all()
            s.query(Belief).filter(Belief.user_id == USER).count()
    finally:
        s.close()


def timed():
    t0 = time.perf_counter()
    with ten.tenant_scope(*TENANT):
        loop()
    return time.perf_counter() - t0


res = {"V0": [], "V1": [], "V2": [], "off": []}
timed()
for cycle in range(9):
    install(None)
    res["off"].append(timed())
    install(inj_v1)
    res["V1"].append(timed())
    install(inj_v2)
    res["V2"].append(timed())
    install(_orig_inject)
    res["V0"].append(timed())
install(_orig_inject)

m = {k: statistics.median(v) * 1000 for k, v in res.items()}
for k in ("off", "V1", "V2", "V0"):
    print(f"{k:4s} {m[k]:8.1f} ms  runs={[round(r * 1000) for r in res[k]]}")
base = m["off"]
for k in ("V1", "V2", "V0"):
    print(f"{k} vs off: {100 * (m[k] / base - 1):+.1f}%")
