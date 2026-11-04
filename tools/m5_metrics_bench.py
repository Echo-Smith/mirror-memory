#!/usr/bin/env python
"""M5 metrics bench (Task 2): recall latency baselines + metrics overhead.

Two measurements, both required by the observability task:

1. **Recall p50/p95 baseline** on SQLite and PostgreSQL, at 100 / 1k / 10k
   seeded beliefs.  Seeds direct belief rows (no LLM, no extraction) so the
   measurement is the recall PATH, not ingestion.  These numbers feed Task
   5's latency budget.

2. **Metrics overhead**: the same recall loop with metrics ON vs OFF, so the
   telemetry cost is the only delta.  Reported as a percentage; the task's
   bar is <= 1%.

Usage::

    .venv/bin/python tools/m5_metrics_bench.py            # sqlite only
    MM_BENCH_PG=postgresql://postgres:mm@127.0.0.1:5437/mm_test \\
        .venv/bin/python tools/m5_metrics_bench.py        # sqlite + pg

The PG leg uses the raw URL for a throwaway schema per size; it is created
and dropped by this script (an admin connection, not the app role).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import uuid

SIZES = (100, 1000, 10000)
RECALL_SAMPLES = 40


def _pct(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = min(int(round(q * (len(s) - 1))), len(s) - 1)
    return round(s[idx], 3)


def _seed_beliefs(engine, user_id: str, n: int) -> None:
    """Seed *n* belief rows directly (no LLM, no extraction)."""
    from mirror_memory.core.models import Evidence
    from mirror_memory.core.repository import record_claim, set_memory_enabled
    from mirror_memory.core.tenancy import fixture_write_scope

    dims = ["fact", "preference", "goal", "event", "topic"]
    with fixture_write_scope(reason="m5_metrics_bench: seed beliefs for recall baseline"):
        with engine._session() as s:
            set_memory_enabled(s, user_id, True)
            for i in range(n):
                record_claim(
                    s, user_id,
                    dimension=dims[i % len(dims)],
                    key=f"key_{i}",
                    claim_text=f"seeded belief number {i} about topic {i % 97}",
                    confidence=0.6 + (i % 40) / 100.0,
                    predicate="mentions", object=f"topic_{i % 97}",
                    cardinality="multi",
                )
            s.commit()


def _recall_samples(engine, user_id: str, n_samples: int) -> list[float]:
    samples: list[float] = []
    for i in range(n_samples):
        q = f"topic question number {i}"
        t0 = time.monotonic()
        engine.recall(user_id=user_id, query=q, purpose="benchmark")
        samples.append((time.monotonic() - t0) * 1000.0)
    return samples


def _build_engine(url: str, tmp_dir: str):
    from mirror_memory.api import MemoryEngine
    from mirror_memory.config.loader import load_config

    cfg = load_config("config/")
    cfg.deletion_log.directory = os.path.join(tmp_dir, "del_" + uuid.uuid4().hex[:8])
    eng = MemoryEngine(config=cfg, database_url=url, default_purpose="benchmark")
    eng._ensure_db()
    return eng


def bench_one(url: str, n: int, tmp_dir: str, *, metrics_on: bool) -> dict:
    """Seed *n* beliefs on *url*, measure recall latency with metrics
    enabled or disabled.  Returns a result dict."""
    from mirror_memory.core import metrics

    if metrics_on:
        metrics.reset_for_tests()
    # "OFF" is approximated by the engine WITHOUT the new wrappers?  No --
    # we need the SAME code path.  Instead we flip the module's overhead
    # switch: the metric calls are the only difference, so we measure the
    # real wrapped path ON and a no-op counter module OFF.
    user_id = "bench-u-" + uuid.uuid4().hex[:6]
    eng = _build_engine(url, tmp_dir)
    from mirror_memory.bench import grant_bench_purpose

    grant_bench_purpose(eng, [user_id])

    t_seed = time.monotonic()
    _seed_beliefs(eng, user_id, n)
    seed_ms = (time.monotonic() - t_seed) * 1000.0

    # warm a little so the first-call schema/pool cost is not in the samples
    eng.recall(user_id=user_id, query="warmup", purpose="benchmark")

    samples = _recall_samples(eng, user_id, RECALL_SAMPLES)
    return {
        "n": n,
        "seed_ms": round(seed_ms, 1),
        "recall_p50_ms": _pct(samples, 0.50),
        "recall_p95_ms": _pct(samples, 0.95),
        "samples": len(samples),
    }


def _overhead_run(url: str, n: int, tmp_dir: str) -> dict:
    """Measure the metrics-on vs metrics-off delta on the recall loop.

    Both runs use the identical engine + recall code.  "OFF" is produced by
    monkeypatching the module-level recorders the wrappers call to no-ops --
    the same approach a counterfactual needs: only the telemetry work
    differs, the request path is byte-for-byte the same otherwise."""
    from mirror_memory.core import metrics

    results = {}
    for label in ("off", "on"):
        user_id = "ov-" + label + "-" + uuid.uuid4().hex[:6]
        eng = _build_engine(url, tmp_dir)
        from mirror_memory.bench import grant_bench_purpose

        grant_bench_purpose(eng, [user_id])
        _seed_beliefs(eng, user_id, n)

        saved = {}
        if label == "off":
            # neutralise the three recorders exercised on the recall path
            saved = {
                "observe_latency": metrics.observe_latency,
                "record_sql": metrics.record_sql,
                "record_llm": metrics.record_llm,
            }
            metrics.observe_latency = lambda *a, **k: None
            metrics.record_sql = lambda *a, **k: None
            metrics.record_llm = lambda *a, **k: None
        try:
            eng.recall(user_id=user_id, query="warmup", purpose="benchmark")
            # more samples for a tighter overhead estimate
            samples = _recall_samples(eng, user_id, 200)
            results[label] = statistics.mean(samples)
        finally:
            for name, fn in saved.items():
                setattr(metrics, name, fn)
    off, on = results["off"], results["on"]
    pct = (on - off) / off * 100.0 if off else 0.0
    return {
        "mean_off_ms": round(off, 4),
        "mean_on_ms": round(on, 4),
        "delta_pct": round(pct, 3),
    }


def main() -> int:
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="m5m_")
    report: dict = {"sizes": list(SIZES), "baselines": {}, "overhead": {}}

    # SQLite
    sqlite_url = "sqlite:///" + os.path.join(tmp_dir, "bench.db")
    report["baselines"]["sqlite"] = {}
    for n in SIZES:
        print(f"[sqlite] seeding {n} beliefs and measuring recall...", file=sys.stderr)
        report["baselines"]["sqlite"][str(n)] = bench_one(sqlite_url, n, tmp_dir, metrics_on=False)
    print("[sqlite] measuring metrics overhead...", file=sys.stderr)
    report["overhead"]["sqlite"] = _overhead_run(sqlite_url, 1000, tmp_dir)

    # PostgreSQL
    pg_url = os.environ.get("MM_BENCH_PG")
    if pg_url:
        report["baselines"]["postgresql"] = {}
        for n in SIZES:
            print(f"[pg] seeding {n} beliefs and measuring recall...", file=sys.stderr)
            # Fresh schema per size so the table is not cumulative.
            schema = "mm_bench_" + uuid.uuid4().hex[:8]
            try:
                report["baselines"]["postgresql"][str(n)] = _bench_pg(
                    pg_url, schema, n, tmp_dir
                )
            finally:
                _drop_pg_schema(pg_url, schema)
        print("[pg] measuring metrics overhead...", file=sys.stderr)
        schema = "mm_bench_ov_" + uuid.uuid4().hex[:6]
        try:
            report["overhead"]["postgresql"] = _overhead_pg(pg_url, schema, 1000, tmp_dir)
        finally:
            _drop_pg_schema(pg_url, schema)
    else:
        report["baselines"]["postgresql"] = "skipped (set MM_BENCH_PG)"

    print(json.dumps(report, indent=2))
    _print_table(report)
    return 0


def _schema_url(pg_url: str, schema: str) -> str:
    sep = "&" if "?" in pg_url else "?"
    return f"{pg_url}{sep}options=-csearch_path%3D{schema}"


def _admin_conn(pg_url: str):
    import psycopg

    return psycopg.connect(pg_url, autocommit=True)


def _drop_pg_schema(pg_url: str, schema: str) -> None:
    try:
        with _admin_conn(pg_url) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    except Exception as exc:  # noqa: BLE001
        print(f"warning: dropping schema {schema} failed: {exc}", file=sys.stderr)


def _bench_pg(pg_url: str, schema: str, n: int, tmp_dir: str) -> dict:
    with _admin_conn(pg_url) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    return bench_one(_schema_url(pg_url, schema), n, tmp_dir, metrics_on=False)


def _overhead_pg(pg_url: str, schema: str, n: int, tmp_dir: str) -> dict:
    with _admin_conn(pg_url) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    return _overhead_run(_schema_url(pg_url, schema), n, tmp_dir)


def _print_table(report: dict) -> None:
    print("\n=== Recall latency baseline (p50/p95 ms) ===")
    print(f"{'backend':<12}{'beliefs':>8}{'p50':>10}{'p95':>10}")
    for backend, rows in report["baselines"].items():
        if not isinstance(rows, dict):
            continue
        for n, r in rows.items():
            print(f"{backend:<12}{n:>8}{r['recall_p50_ms']:>10}{r['recall_p95_ms']:>10}")
    print("\n=== Metrics overhead (ON vs OFF) ===")
    for backend, r in report["overhead"].items():
        print(f"{backend}: off={r['mean_off_ms']}ms on={r['mean_on_ms']}ms "
              f"delta={r['delta_pct']}%")


if __name__ == "__main__":
    raise SystemExit(main())
