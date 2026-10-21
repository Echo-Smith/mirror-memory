"""Bisect the pollution source: run selected isolation tests, then the
longrun test, printing SQLAlchemy/GC state before and after."""
import gc
import sys
import time

import pytest


def sqla_state(tag):
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    import mirror_memory.core.tenancy as ten

    lines = [f"--- state {tag} ---"]
    lines.append(
        "listeners: do_orm_execute=%s before_flush=%s"
        % (
            event.contains(Session, "do_orm_execute", ten._inject_tenant_criteria),
            event.contains(Session, "before_flush", ten._enforce_tenant_on_flush),
        )
    )
    try:
        from sqlalchemy.sql import lambdas

        cache = getattr(lambdas, "_closure_per_cache_key", None)
        if cache is not None:
            lines.append(
                "lambda _closure_per_cache_key: len=%d (LRU cap 1000)"
                % len(cache)
            )
        else:
            lines.append("no _closure_per_cache_key global")
    except Exception as e:  # noqa: BLE001
        lines.append(f"lambda cache introspection failed: {e}")
    g = gc.get_stats()
    lines.append(
        "gc collections: gen0=%d gen1=%d gen2=%d  objects=%d"
        % (g[0]["collections"], g[1]["collections"], g[2]["collections"], len(gc.get_objects()))
    )
    out = "\n".join(lines)
    sys.stdout.write(out + "\n")
    sys.stdout.flush()


def main():
    subset = sys.argv[1].split() if len(sys.argv) > 1 else []
    target = "tests/test_longrun_bench.py::TestOracleDetectsDamage::test_resurrection_is_caught"
    sqla_state("before")
    t0 = time.monotonic()
    rc = pytest.main(["-q", "-p", "no:cacheprovider", target] + subset)
    wall = time.monotonic() - t0
    sqla_state("after")
    print(f"COMBO wall={wall:.1f}s rc={rc}")


if __name__ == "__main__":
    main()
