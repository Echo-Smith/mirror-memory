"""In-process A/B micro-benchmark for the tenancy hooks.

The host machine's wall-clock drifts up to +/-100% on minute scales, which
makes cross-run suite comparisons meaningless.  This benchmark instead
alternates hook variants inside ONE process over identical DB work, so
system drift cancels out of the ratios.

Modes:
  fixed    - current spike/v1-tenancy-perf implementation (cached option,
             no traversal on the fast path)
  orig     - cfd14d9 implementation (fresh lambda per execute + mapper/
             traversal check on every statement), re-created here
  noinject - read-hook injection removed entirely (write-path validation
             and Session.get wrapper stay); gives the read-hook cost floor
"""

import statistics
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import mirror_memory.core.tenancy as ten
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import (
    get_state_revision,
    list_active_beliefs,
    record_claim,
    set_memory_enabled,
)

TENANT = ("default", "default")
USER = "u1"

engine = create_engine("sqlite://")
Base.metadata.create_all(engine)
factory = sessionmaker(bind=engine)

with ten.tenant_scope(*TENANT):
    s = factory()
    set_memory_enabled(s, USER, True)
    s.commit()
    for i in range(50):
        record_claim(
            s, USER, dimension="topic", key=f"k{i}",
            claim_text=f"claim {i}", confidence=0.8,
            session_id=f"s{i}",
        )
    s.commit()
    SOME_ID = s.query(Belief).first().id
    s.close()


def iteration(s):
    """One mixed read/write workload round (read-heavy like recall)."""
    set_memory_enabled(s, USER, True)
    s.commit()
    for i in range(3):
        record_claim(
            s, USER, dimension="topic", key=f"k{i}",
            claim_text=f"claim {i} again", confidence=0.8,
            session_id="sBench",
        )
    s.commit()
    list_active_beliefs(s, USER, limit=50)
    for _ in range(5):
        get_state_revision(s, USER)
    s.query(Belief).filter(Belief.user_id == USER, Belief.status == "active").count()
    s.get(Belief, SOME_ID)


_orig_listener = None


def _remove_inject():
    from sqlalchemy import event

    try:
        event.remove(Session, "do_orm_execute", ten._inject_tenant_criteria)
    except Exception:  # noqa: BLE001
        pass
    _restore_orig_listener()


def _restore_orig_listener():
    from sqlalchemy import event

    global _orig_listener
    if _orig_listener is not None:
        try:
            event.remove(Session, "do_orm_execute", _orig_listener)
        except Exception:  # noqa: BLE001
            pass
        _orig_listener = None


def _install_orig_listener():
    from sqlalchemy import event
    from sqlalchemy.orm import with_loader_criteria as _wlc

    _restore_orig_listener()

    def orig_inject(state):
        if state.is_column_load or state.is_relationship_load:
            return
        if not ten._statement_mappers(state) and not ten._statement_touches_tenant_table(
            state.statement
        ):
            return
        if ten.in_system_scope():
            return
        tenant = ten._tenant_var.get()
        if tenant is None:
            raise ten.TenantContextError("no ctx")
        if not (state.is_select or state.is_update or state.is_delete):
            return
        t, a = tenant
        state.statement = state.statement.options(
            _wlc(
                ten.TenantScoped,
                lambda cls: (cls.tenant_id == t) & (cls.app_id == a),
                include_aliases=True,
            )
        )

    global _orig_listener
    _orig_listener = orig_inject
    event.listen(Session, "do_orm_execute", orig_inject)


def bench_round(mode, n_iter=60):
    if mode == "noinject":
        _remove_inject()
        Session.get = ten._tenant_aware_get
    elif mode == "orig":
        _remove_inject()
        _install_orig_listener()
        Session.get = ten._tenant_aware_get
    # mode == "fixed": module state as imported

    t0 = time.perf_counter()
    with ten.tenant_scope(*TENANT):
        s = factory()
        try:
            for _ in range(n_iter):
                iteration(s)
        finally:
            s.close()
    return time.perf_counter() - t0


results = {"noinject": [], "fixed": [], "orig": []}
bench_round("fixed", 10)  # warm-up
for cycle in range(7):
    for mode in ("noinject", "fixed", "orig"):
        results[mode].append(bench_round(mode))

med = {m: statistics.median(v) for m, v in results.items()}
print("median per 60 iterations (lower is better):")
for m in ("noinject", "fixed", "orig"):
    print(f"  {m:9s} {med[m] * 1000:8.1f} ms   runs={[round(r * 1000) for r in results[m]]}")
print(f"fixed vs noinject (read-hook cost floor) : {100 * (med['fixed'] / med['noinject'] - 1):+.1f}%")
print(f"orig   vs noinject                       : {100 * (med['orig'] / med['noinject'] - 1):+.1f}%")
print(f"fixed vs orig   (improvement)            : {100 * (med['fixed'] / med['orig'] - 1):+.1f}%")
