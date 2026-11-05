"""M5 §5a diagnostic: capture EXPLAIN (ANALYZE, BUFFERS) for the hottest
statements of one full M4 ``run_longrun`` on the APP role with RLS active.

This is NOT a behavioural test: it is a measurement harness that reuses the
repo's own PostgreSQL fixtures (``pg_app_url`` / ``pg_admin_engine``) so the
schema, RLS posture and app role are EXACTLY what the suite exercises --
building them by hand in a standalone script would not be the hosted shape.

It runs only when ``MM_EXPLAIN_CAPTURE=1`` (plus PostgreSQL via
``MM_DATABASE_URL``); otherwise it is skipped, so normal CI is unaffected.

    MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test \\
    MM_PG_RLS=1 MM_EXPLAIN_CAPTURE=1 \\
    .venv/bin/python -m pytest tests/test_m5_explain_capture.py -q -s -p no:cacheprovider

Artifacts are written to ``$MM_EXPLAIN_OUT`` (default
``/tmp/m5_perf_bal/explain_top5.json``).
"""

from __future__ import annotations

import json
import os
import re

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MM_EXPLAIN_CAPTURE") != "1",
    reason="diagnostic capture; set MM_EXPLAIN_CAPTURE=1 to run",
)


def test_capture_top_statements_plans(pg_admin_engine, pg_app_url, tmp_path):
    """Populate real data with one longrun on the app role, then EXPLAIN the
    five most frequent statements inside a tenant-scoped transaction."""
    from sqlalchemy import create_engine, event, text

    from mirror_memory.bench.longrun import run_longrun
    from mirror_memory.config.loader import load_config

    out_path = os.environ.get("MM_EXPLAIN_OUT", "/tmp/m5_perf_bal/explain_top5.json")

    from sqlalchemy.engine import Engine as _SAEngine
    engine = create_engine(pg_app_url, echo=False)
    counts: dict[str, int] = {}
    first_params: dict[str, tuple] = {}

    def _hook(conn, cursor, statement, parameters, context, executemany):
        key = re.sub(r"\s+", " ", statement).strip()
        counts[key] = counts.get(key, 0) + 1
        # keep the bound params exactly as the driver will receive them:
        # a dict for named style (%(name)s), a tuple for positional style.
        if key not in first_params and isinstance(parameters, (dict, tuple)):
            first_params[key] = parameters

    cfg = load_config("config/")
    cfg.deletion_log.directory = str(tmp_path / "deletions")
    (tmp_path / "deletions").mkdir(parents=True, exist_ok=True)

    # run_longrun builds its OWN engine internally, so listen on the Engine
    # CLASS (the only way to observe the statements it issues).
    event.listen(_SAEngine, "before_cursor_execute", _hook)
    try:
        run_longrun(cfg, user_id="explain-u1", database_url=pg_app_url)
    finally:
        event.remove(_SAEngine, "before_cursor_execute", _hook)

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:5]

    from mirror_memory.core.tenancy import tenant_scope

    result: dict = {"dialect": engine.dialect.name, "statements": []}
    with tenant_scope("default", "default"):
        with engine.connect() as conn:
            for stmt, n in ranked:
                entry: dict = {"statement": stmt[:400], "calls": n}
                try:
                    # The captured statement is in DBAPI paramstyle
                    # (%(name)s + dict params); exec_driver_sql passes those
                    # through unchanged, whereas text() would expect :name.
                    params = first_params.get(stmt)
                    plan = conn.exec_driver_sql(
                        "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + stmt,
                        params,
                    ).fetchall()
                    entry["plan"] = [row[0] for row in plan]
                    times = [ln for ln in entry["plan"] if "Execution Time" in ln]
                    entry["execution_ms"] = (
                        times[-1].split(":")[-1].strip() if times else "?"
                    )
                    entry["has_rls_predicate"] = any(
                        "current_setting('mm." in ln for ln in entry["plan"]
                    )
                except Exception as exc:  # noqa: BLE001
                    entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
                result["statements"].append(entry)
    engine.dispose()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=1)[:4000])
    print(f"\n# explain artifact -> {out_path}")
    assert ranked, "no statements captured"