"""M5 §5 patch: recall latency baseline on PostgreSQL as the APP ROLE under
RLS (the production shape), at 100 / 1k / 10k beliefs.

This is a measurement harness, not a behavioural test: it reuses the repo's own
PostgreSQL fixtures (``pg_app_url`` / ``pg_admin_engine``) so the schema, the
row-level-security posture and the login are EXACTLY what the suite exercises.
Building them by hand in a standalone script would not be the hosted shape.

It runs only when ``MM_RECALL_BASELINE=1`` (plus PostgreSQL via
``MM_DATABASE_URL``), so normal CI is unaffected::

    MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test \\
    MM_PG_RLS=1 MM_RECALL_BASELINE=1 MM_RECALL_OUT=/tmp/m5p2/recall_pg_approle.json \\
    .venv/bin/python -m pytest tests/test_m5_recall_baseline_pg.py -q -s -p no:cacheprovider

Reported per size: samples, p50, p95, min, max (milliseconds) plus the role/RLS
condition, so the number is attributable.
"""

from __future__ import annotations

import json
import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MM_RECALL_BASELINE") != "1",
    reason="measurement harness; set MM_RECALL_BASELINE=1 to run",
)

SIZES = (100, 1000, 10000)
SAMPLES = 40


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def test_pg_app_role_rls_recall_baseline(pg_app_url, tmp_path):
    from sqlalchemy import insert

    from mirror_memory.api import MemoryEngine
    from mirror_memory.config.loader import load_config
    from mirror_memory.core.models import Belief as CoreBelief
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.rls import app_role, rls_enabled
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from sqlalchemy.orm import Session

    assert rls_enabled(), "this baseline must run with RLS on (mode A)"

    cfg = load_config("config/")
    cfg.deletion_log.directory = str(tmp_path / "deletions")
    (tmp_path / "deletions").mkdir(parents=True, exist_ok=True)

    eng = MemoryEngine(
        config=cfg, database_url=pg_app_url, default_purpose="benchmark",
        tenant_id="default", app_id="default",
    )
    eng._ensure_db()
    user = "baseline-u1"
    with tenant_scope(*eng._tenant), fixture_write_scope("recall baseline: enable"):
        s = Session(eng._engine)
        try:
            set_memory_enabled(s, user, True)
            s.commit()
        finally:
            s.close()

    out: dict = {
        "backend": "postgresql",
        "role": app_role(),
        "rls": rls_enabled(),
        "samples_per_size": SAMPLES,
        "sizes": {},
    }

    table = CoreBelief.__table__
    next_id = 1
    seeded = 0
    for n in SIZES:
        to_seed = n - seeded
        batch = 500
        for start in range(0, to_seed, batch):
            rows = []
            for i in range(start, min(start + batch, to_seed)):
                idx = next_id + i
                rows.append(dict(
                    id=idx, global_id=f"g-base-{idx}",
                    tenant_id="default", app_id="default", user_id=user,
                    # uq_mm_beliefs_user_key is UNIQUE(tenant,app,user,key),
                    # so every seeded belief needs its own key.
                    dimension="preference", key=f"topic-{idx}",
                    predicate="likes", object=f"thing-{idx}",
                    claim_text=f"belief number {idx} about topic {idx}",
                    polarity="pos", status="active", confidence=0.7,
                    memory_tier="hot", layer="L4", source="extracted",
                ))
            with tenant_scope(*eng._tenant), fixture_write_scope("recall baseline: seed"):
                with eng._engine.begin() as conn:
                    conn.execute(insert(table), rows)
        next_id += to_seed
        seeded = n

        # warm-up so the first-call pool/schema cost is not in the samples
        eng.recall(user_id=user, query="warmup topic", purpose="benchmark")

        samples: list[float] = []
        for i in range(SAMPLES):
            t0 = time.monotonic()
            eng.recall(user_id=user, query=f"topic {i % 37}", purpose="benchmark")
            samples.append((time.monotonic() - t0) * 1000.0)

        out["sizes"][str(n)] = {
            "beliefs": n,
            "samples": len(samples),
            "p50_ms": _pct(samples, 0.50),
            "p95_ms": _pct(samples, 0.95),
            "min_ms": round(min(samples), 2),
            "max_ms": round(max(samples), 2),
        }
        print(f"[pg app role + RLS] {n} beliefs: {out['sizes'][str(n)]}", flush=True)

    eng._engine.dispose()

    path = os.environ.get("MM_RECALL_OUT", "/tmp/m5p2/recall_pg_approle.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n# recall baseline -> {path}")
    assert out["sizes"]["10000"]["beliefs"] == 10000