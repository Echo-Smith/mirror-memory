"""Cache diagnostics for the tenancy read-injection (measure-only).

1. How many times per statement execution is the with_loader_criteria
   lambda invoked?  (If ~1 per execution per entity, the tenant predicate
   is rebuilt every time and never served from a cache.)
2. Who calls enum.__and__ / coercions.expect?  (caller chains)
3. Compile-cache health: "[generated in" vs "[cached since" counts.
"""
import io
import logging
import sys
import time

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker, with_loader_criteria

import mirror_memory.core.tenancy as ten
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import record_claim, set_memory_enabled

TENANT = ("default", "default")
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

# ---------------------------------------------------------------------------
# 1. Lambda invocation count: replace the cached option with an instrumented
#    lambda, run N executions of one statement shape, count invocations.
#    Counting is done OUTSIDE the lambda (patching _invoke_user_fn) because
#    a counter inside the lambda body breaks lambda-SQL closure analysis.
# ---------------------------------------------------------------------------
import sqlalchemy.sql.lambdas as _lam

calls = {"lambda": 0}
N = 300

_orig_invoke = _lam.LambdaElement._invoke_user_fn


def _counting_invoke(self, fn):
    calls["lambda"] += 1
    return _orig_invoke(self, fn)


def instrumented_option(tenant):
    t, a = tenant

    def crit(cls):
        return (cls.tenant_id == t) & (cls.app_id == a)

    return with_loader_criteria(ten.TenantScoped, crit, include_aliases=True)


ten._option_cache.clear()
for k in (TENANT,):
    ten._option_cache[k] = instrumented_option(k)

_lam.LambdaElement._invoke_user_fn = _counting_invoke
try:
    with ten.tenant_scope(*TENANT):
        s = factory()
        try:
            # warm-up (compilation, relationship setup)
            s.query(Belief).filter(Belief.user_id == USER, Belief.status == "active").all()
            calls["lambda"] = 0
            t0 = time.perf_counter()
            for _ in range(N):
                s.query(Belief).filter(Belief.user_id == USER, Belief.status == "active").all()
            wall = time.perf_counter() - t0
        finally:
            s.close()
finally:
    _lam.LambdaElement._invoke_user_fn = _orig_invoke
print(f"1. lambda invocations for {N} executions of 1 statement shape: {calls['lambda']}"
      f"  ({calls['lambda'] / N:.2f} per execution)  wall={wall * 1000:.0f}ms")

# ---------------------------------------------------------------------------
# 1b. Inject-on vs inject-off call-count diff: WHERE do the extra
#     elements.__init__ / expect / _clone calls come from?
# ---------------------------------------------------------------------------
import cProfile
import pstats

from sqlalchemy import event as sa_event


def profile_loop():
    s = factory()
    try:
        for _ in range(N):
            s.query(Belief).filter(Belief.user_id == USER, Belief.status == "active").all()
            s.query(Belief).count()
    finally:
        s.close()


def run_profile():
    p = cProfile.Profile()
    with ten.tenant_scope(*TENANT):
        p.enable()
        profile_loop()
        p.disable()
    return p


st_on = pstats.Stats(run_profile())
event.remove(Session, "do_orm_execute", ten._inject_tenant_criteria)
st_off = pstats.Stats(run_profile())
event.listen(Session, "do_orm_execute", ten._inject_tenant_criteria)


def counts(st, pattern):
    out = {}
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        if pattern in func[2]:
            key = (func[0].split("site-packages/")[-1], func[2])
            t, n = out.get(key, (0.0, 0))
            out[key] = (t + tt, n + nc)
    return out


print("1b. call-count diff (inject ON vs OFF) for construction hotspots:")
for pat in ("__init__", "expect", "_clone", "_boolean_compare", "__and__", "traverse", "_deep_annotate"):
    on = counts(st_on, pat)
    off = counts(st_off, pat)
    for key in sorted(set(on) | set(off), key=lambda k: -(on.get(k, (0, 0))[1] - off.get(k, (0, 0))[1])):
        d = on.get(key, (0, 0))[1] - off.get(key, (0, 0))[1]
        if abs(d) > 200:
            print(f"   {pat:20s} {key[0][:60]}:{key[1]}  on={on.get(key, (0,0))[1]:8d} off={off.get(key, (0,0))[1]:8d} delta={d:+8d}")

# ---------------------------------------------------------------------------
# 2. cProfile caller chains for enum.__and__ and coercions.expect
# ---------------------------------------------------------------------------
import cProfile
import pstats

calls["lambda"] = 0
prof = cProfile.Profile()
with ten.tenant_scope(*TENANT):
    s = factory()
    try:
        prof.enable()
        for _ in range(N):
            s.query(Belief).filter(Belief.user_id == USER, Belief.status == "active").all()
        prof.disable()
    finally:
        s.close()
print(f"   (profiled lambda invocations: {calls['lambda']})")
out = io.StringIO()
st = pstats.Stats(prof, stream=out)
st.print_callers("__and__")
block = out.getvalue()
idx = block.find("Function")
print("2. FULL callers block for __and__:")
print(block[block.rfind("was called by"):][:2000])
out = io.StringIO()
st2 = pstats.Stats(prof, stream=out)
st2.print_callers("_boolean_compare")
block = out.getvalue()
print("FULL callers block for _boolean_compare:")
print(block[block.rfind("was called by"):][:2000])

# ---------------------------------------------------------------------------
# 3. Compile-cache health via the SQLAlchemy engine log
# ---------------------------------------------------------------------------
logbuf = io.StringIO()
handler = logging.StreamHandler(logbuf)
sqllog = logging.getLogger("sqlalchemy.engine")
sqllog.addHandler(handler)
sqllog.setLevel(logging.INFO)

try:
    with ten.tenant_scope(*TENANT):
        s = factory()
        try:
            for _ in range(50):
                s.query(Belief).filter(Belief.user_id == USER, Belief.status == "active").all()
                s.query(Belief).count()
        finally:
            s.close()
finally:
    sqllog.removeHandler(handler)
    sqllog.setLevel(logging.NOTSET)

text = logbuf.getvalue()
generated = text.count("[generated in")
cached = text.count("[cached since")
nokey = text.count("no key")
print(f"3. engine log over 100 executions: generated={generated} cached={cached} no-key={nokey}")
