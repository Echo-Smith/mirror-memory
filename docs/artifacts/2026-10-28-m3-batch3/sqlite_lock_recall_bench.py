"""SQLite: recall latency (p50/p99) while a worker publish transaction
holds the file's write lock -- with the access audit ON vs OFF.

Measurement only (M3 batch-3 task 3); no design change.  The audit-off
control disables exactly the per-read audit INSERT (the same code path
otherwise), via monkeypatching -- mirroring what a deployment without
the audit row would pay.

Method: a lock-holder thread emulates the worker's publish transaction
(BEGIN IMMEDIATE + writes + hold 40 ms + commit, cycling with short
gaps).  The measuring thread issues recall() in a loop; per-recall latency
is recorded.  200 samples per configuration, 3 repeats, min taken across
repeats (the machine-noise floor is the enemy, not the audit).
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import sqlalchemy  # noqa: E402
from sqlalchemy import create_engine, event, text  # noqa: E402

from mirror_memory.api import MemoryEngine  # noqa: E402
from mirror_memory.config.loader import load_config  # noqa: E402
from mirror_memory.core import access_audit  # noqa: E402
from mirror_memory.core.models import Base, Belief  # noqa: E402
from mirror_memory.core.repository import set_memory_enabled  # noqa: E402
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope  # noqa: E402
from mirror_memory.core import authorization  # noqa: E402


def build(url):
    eng0 = create_engine(url)
    Base.metadata.create_all(eng0)
    F = sqlalchemy.orm.sessionmaker(bind=eng0)
    with tenant_scope("default", "default"), fixture_write_scope("seed"):
        s = F()
        set_memory_enabled(s, "u1", True)
        authorization.grant_purpose(s, "u1", "api", operations=list(authorization.OPERATIONS), issuer="bench")
        for i in range(6):
            s.add(Belief(user_id="u1", dimension="topic", key=f"k{i}",
                         claim_text=f"CLAIM-{i} about topics", confidence=0.9, layer="L2"))
        s.commit()
        s.close()
    eng0.dispose()


def lock_holder(url, stop: threading.Event):
    """Emulate the worker's publish transaction: hold the SQLite write lock
    in a repeating cycle (40 ms hold, 15 ms gap)."""
    conn = create_engine(url, connect_args={"timeout": 30.0}).connect()
    try:
        conn.execute(text("BEGIN IMMEDIATE"))
        while not stop.is_set():
            conn.execute(text(
                "UPDATE mm_memory_preferences SET updated_at = updated_at WHERE user_id = 'u1'"))
            time.sleep(0.040)  # the publish transaction's work
            conn.execute(text("COMMIT"))
            time.sleep(0.015)
            conn.execute(text("BEGIN IMMEDIATE"))
    except Exception:
        pass
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


def measure(url, *, disable_audit: bool, samples: int) -> dict:
    me = MemoryEngine(config=load_config("config/"), database_url=url,
                      tenant_id="default", app_id="default")
    me._ensure_db()
    if disable_audit:
        orig = access_audit.record_access
        access_audit.record_access = lambda *a, **k: None
    latencies: list[float] = []
    stop = threading.Event()
    t = threading.Thread(target=lock_holder, args=(url, stop))
    t.start()
    time.sleep(0.2)
    try:
        for _ in range(samples):
            t0 = time.perf_counter()
            try:
                me.recall(user_id="u1", query="CLAIM", purpose="api")
            except Exception as exc:
                if "locked" in str(exc):
                    continue  # a refused attempt is a lock outcome, not a latency
                raise
            latencies.append((time.perf_counter() - t0) * 1000)
    finally:
        stop.set()
        t.join(timeout=30)
        if disable_audit:
            access_audit.record_access = orig
    latencies.sort()

    def pct(p):
        if not latencies:
            return float("nan")
        idx = min(len(latencies) - 1, max(0, round(p / 100 * len(latencies)) - 1))
        return latencies[idx]

    return {
        "samples": len(latencies),
        "lock_skips": samples - len(latencies),
        "p50_ms": round(pct(50), 2),
        "p90_ms": round(pct(90), 2),
        "p99_ms": round(pct(99), 2),
        "median_ms": round(statistics.median(latencies), 2) if latencies else float("nan"),
    }


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    results = {}
    for label, disable in (("audit_on", False), ("audit_off", True)):
        best = None
        for _ in range(repeats):
            d = tempfile.mkdtemp()
            url = f"sqlite:///{d}/bench.db"
            build(url)
            r = measure(url, disable_audit=disable, samples=samples)
            if best is None or r["p99_ms"] < best["p99_ms"]:
                best = r
        results[label] = best
    results["note"] = ("min across repeats per configuration; lock_skips = recalls "
                       "refused by the busy timeout (lock outcomes, not latencies)")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
