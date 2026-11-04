"""M5 Task 0 -- PostgreSQL performance diagnosis across three revisions.

Diagnosis only: this script measures and decomposes; it never modifies
``src/`` or ``tests/`` for any purpose other than temporarily materialising
a baseline revision in the worktree (restored to the starting branch at the
end).  See ``docs/reports/2026-11-04-m5-pg-perf-diag.md`` for the findings.

What it does (one process, one run per measurement so noise is shared):

  measure     interleaved three-point timing.  For round in 1..R: for
              rev in [M2=403f38d, M3=aed0639, M4=d61e7b7]: materialise the
              revision (git checkout rev -- src tests pyproject.toml), run
              ``tests/test_longrun_bench.py`` against PostgreSQL mode A
              (MM_PG_RLS=1, app read filter, MM_SQL_AUDIT=0), record wall
              time and junit.  Prints median and min per revision, and
              saves the raw rounds to JSON.

  profile     per-revision cProfile of ``run_longrun`` through the same
              small scenario the longrun tests use, so Python-side time is
              attributed to functions (sqlalchemy execute, authorization,
              access audit, deletion guard, ...).

  statements  per-revision SQL statement COUNTS from a live longrun run,
              via a SQLAlchemy ``before_cursor_execute`` listener attached
              to the Engine class.  Reports total statements plus
              per-category call counts (read gate / visibility filter /
              audit writes / deletion checks / sync catch-up / other) by
              the stack frame that issued them, and the top-5 hottest
              individual statements.  This is COUNT-only (no timing):
              per-statement TIMING is not feasible without an APM inside
              every execute, so it is stated as a limitation, not faked.

  restore     materialise the starting branch back into the worktree and
              print ``git status`` for src/tests/pyproject.toml.

Methodology reused from ``tools/perf_ids.py`` (``_use_rev``) and
``tools/m2_mode_perf.py`` (``mode_env``, junit accounting).  Run from the
repo root with the project venv, e.g.::

    MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test \
    MM_PG_RLS=1 MM_TENANT_READ_FILTER=app \
    .venv/bin/python tools/m5_perf_diag.py measure --rounds 3

Caveats (repeat in the report): local Docker PG 16 is smaller than a CI
runner; each ``test_longrun_bench.py`` file run executes all 18 tests and
takes minutes, so the three-point sweep is 3*3 runs of that length.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib
import io
import json
import os
import pstats
import re
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# revision label -> commit
REVS: dict[str, str] = {"M2": "403f38d", "M3": "aed0639", "M4": "d61e7b7"}
REV_ORDER = ("M2", "M3", "M4")
BRANCH = "feat/m5-readiness"
MANAGED_PATHS = ["src", "tests", "pyproject.toml"]
LONGRUN = "tests/test_longrun_bench.py"
# tests/test_longrun_bench.py has 18 tests; recorded for the validity check.
EXPECTED_TESTS = 18

_ADMIN_URL = os.environ.get("MM_DATABASE_URL", "postgresql://postgres:mm@127.0.0.1:5437/mm_test")

# Mode A = production today: RLS on + application read filter, SQL auditor off
# for measurement (the M1 acceptance figure disables it in every leg).
MODE_A_ENV: dict[str, str] = {
    "MM_PG_RLS": "1",
    "MM_TENANT_READ_FILTER": "app",
    "MM_SQL_AUDIT": "0",
}
_MODE_ENV_KEYS = {"MM_PG_RLS", "MM_TENANT_READ_FILTER", "MM_ALLOW_UNFILTERED_MEASUREMENT", "MM_SQL_AUDIT"}
# Env keys a revision sweep must clear so nothing leaks between revisions.
_LEAK_KEYS = _MODE_ENV_KEYS | {"MM_LONGRUN_DATABASE_URL"}


def mode_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _LEAK_KEYS}
    env.update(MODE_A_ENV)
    env["MM_DATABASE_URL"] = _ADMIN_URL
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Revision materialisation (reuses perf_ids._use_rev semantics; kept local so
# this tool has no import-time dependency on the tools package).
# ---------------------------------------------------------------------------


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def _use_rev(rev: str) -> None:
    """Make worktree src/tests/pyproject.toml match *rev* exactly.

    ``git checkout <rev> -- path`` never deletes files that exist in the
    worktree but not in *rev*.  Remove worktree files under src/tests that
    the revision's tree does not contain first, then materialise.  This is
    the same logic as ``tools/perf_ids.py::_use_rev``.
    """
    keep = set(
        _git("ls-tree", "-r", "--name-only", rev, "--", "src", "tests").stdout.splitlines()
    )
    listed = _git("ls-files", "--", "src", "tests").stdout.splitlines()
    for path in listed:
        if path not in keep and os.path.exists(path):
            os.remove(path)
    for root, _dirs, files in os.walk("tests"):
        for name in files:
            if name.endswith(".py"):
                rel = os.path.join(root, name)
                if rel not in keep:
                    os.remove(rel)
    _git("checkout", "-q", rev, "--", *MANAGED_PATHS)
    _git("reset", "-q", "HEAD", "--", "src", "tests", check=False)
    _purge_bytecode()


def _purge_bytecode() -> None:
    for root, dirs, _files in os.walk("src"):
        for d in list(dirs):
            if d == "__pycache__":
                for f in Path(os.path.join(root, d)).glob("*.pyc"):
                    f.unlink(missing_ok=True)


def _restore(work: Path) -> None:
    print(f"# restoring branch {BRANCH} into the worktree", file=sys.stderr)
    _git("checkout", "-q", BRANCH, "--", *MANAGED_PATHS)
    _git("reset", "-q", "HEAD", "--", "src", "tests", check=False)
    _purge_bytecode()
    status = _git("status", "--porcelain", "--", *MANAGED_PATHS)
    changed = [l for l in status.stdout.splitlines() if l.strip()]
    print(f"# src/tests/pyproject.toml changes after restore: {len(changed)}")
    for line in changed:
        print("  " + line)


# ---------------------------------------------------------------------------
# junit helpers (same shape as m2_mode_perf)
# ---------------------------------------------------------------------------


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return {
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _run_pytest(env: dict[str, str], junit: Path, out: Path) -> float:
    cmd = [
        sys.executable, "-m", "pytest", LONGRUN, "-q",
        "-p", "no:cacheprovider", f"--junitxml={junit}",
    ]
    start = time.monotonic()
    with open(out, "w", encoding="utf-8") as fh:
        subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return time.monotonic() - start


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def cmd_measure(rounds: int, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"branch": BRANCH, "mode": "A", "expected_tests": EXPECTED_TESTS, "rounds": []}
    for r in range(1, rounds + 1):
        row: dict = {}
        for label in REV_ORDER:
            rev = REVS[label]
            _use_rev(rev)
            junit = work / f"measure_{label}_r{r}.xml"
            out = work / f"measure_{label}_r{r}.txt"
            secs = _run_pytest(mode_env(), junit, out)
            counts = junit_counts(junit)
            valid = (
                counts["failed"] == 0
                and counts["errors"] == 0
                and counts["passed"] == EXPECTED_TESTS
            )
            row[label] = {"rev": rev, "secs": round(secs, 1), **counts, "valid": valid}
            print(
                f"round {r} {label} ({rev}): {row[label]['secs']}s "
                f"passed={counts['passed']} failed={counts['failed']} "
                f"errors={counts['errors']} {'VALID' if valid else 'INVALID'}",
                flush=True,
            )
        row["valid"] = all(row[t]["valid"] for t in REV_ORDER)
        # interleaving frame: a round is usable only if all three passed
        report["rounds"].append(row)
        print(f"round {r}: {'VALID' if row['valid'] else 'INVALID'}\n", flush=True)

    valid_rows = [row for row in report["rounds"] if row["valid"]]
    stats: dict = {"valid_rounds": len(valid_rows), "per_rev": {}}
    for label in REV_ORDER:
        samples = [row[label]["secs"] for row in valid_rows]
        if not samples:
            stats["per_rev"][label] = {"n": 0}
            continue
        stats["per_rev"][label] = {
            "n": len(samples),
            "min": round(min(samples), 1),
            "median": round(statistics.median(samples), 1),
            "all": samples,
        }
    base = stats["per_rev"].get("M2", {})
    for label in ("M3", "M4"):
        cur = stats["per_rev"].get(label, {})
        if base.get("min") and cur.get("min"):
            cur["min_ratio_vs_M2"] = round(cur["min"] / base["min"], 3)
            cur["median_ratio_vs_M2"] = round(cur["median"] / base["median"], 3)
    report["stats"] = stats
    (work / "measure.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_stats(stats)
    return report


def _print_stats(stats: dict) -> None:
    print("### three-point longrun timing (PG mode A, SQL auditor off)")
    print(f"valid rounds: {stats['valid_rounds']}")
    print("| rev | n | min s | median s | min ratio vs M2 | median ratio vs M2 |")
    print("|---|---:|---:|---:|---:|---:|")
    for label in REV_ORDER:
        s = stats["per_rev"][label]
        if not s.get("n"):
            print(f"| {label} | 0 | - | - | - | - |")
            continue
        mn = s.get("min_ratio_vs_M2", 1.0)
        md = s.get("median_ratio_vs_M2", 1.0)
        print(
            f"| {label} ({REVS[label]}) | {s['n']} | {s['min']:.1f} | "
            f"{s['median']:.1f} | {mn:.3f} | {md:.3f} |"
        )


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

SMALL = dict(days=40, preferences=12, goals=5, corrections=6, conflicts=5, forgets=3)


def cmd_profile(work: Path) -> dict:
    """cProfile one small run_longrun per revision, on the PG bench database.

    The bench builds its own engine inside run_longrun, so we call it direct
    with an explicit PG URL rather than going through pytest.  Correctness
    isn't the point here (timing is), and the harness verifies invariants
    anyway.
    """
    work.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("MM_LONGRUN_DATABASE_URL") or os.environ.get("MM_M5_LONGRUN_URL")
    if not url:
        raise SystemExit(
            "profile needs MM_LONGRUN_DATABASE_URL pointing at a ready PG schema "
            "(build one with the pg harness, or run measure first)."
        )
    sys.path.insert(0, os.path.abspath("src"))
    result: dict = {"per_rev": {}, "scenario": SMALL}
    for label in REV_ORDER:
        _use_rev(REVS[label])
        # fresh import so we exercise the materialised revision
        for mod in [m for m in sys.modules if m.startswith("mirror_memory")]:
            del sys.modules[mod]
        longrun = importlib.import_module("mirror_memory.bench.longrun")
        from mirror_memory.config.loader import load_config

        cfg = load_config("config/")
        if hasattr(cfg, "deletion_log"):
            cfg.deletion_log.directory = str(work / f"deletion_{label}")
        prof = cProfile.Profile()
        prof.enable()
        t0 = time.monotonic()
        rep = longrun.run_longrun(cfg, scenario_kwargs=SMALL, database_url=url)
        wall = time.monotonic() - t0
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(45)
        (work / f"profile_{label}.txt").write_text(buf.getvalue(), encoding="utf-8")
        # top cumulative self functions (python time attribution)
        stats = pstats.Stats(prof)
        stats.sort_stats("cumulative")
        top: list[dict] = []
        for func, (_cc, _nc, _tt, ct, callers) in stats.stats.items():
            top.append({"func": f"{func[2]}:{func[0].split('/')[-1]}:{func[1]}",
                        "cum_s": round(ct, 3), "calls": _nc})
        top.sort(key=lambda d: d["cum_s"], reverse=True)
        result["per_rev"][label] = {
            "rev": REVS[label],
            "wall_s": round(wall, 3),
            "top_cumulative": top[:30],
        }
        print(f"profile {label} ({REVS[label]}): wall {wall:.1f}s -> {work}/profile_{label}.txt", flush=True)
    (work / "profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# statement counts
# ---------------------------------------------------------------------------


def _classify(sql: str, chain: str) -> str:
    """Bucket a statement+call-site into one increment source.

    Both the SQL text and the Python call chain are used: the same SQL
    shape can be issued by the authorization gate, a visibility filter or a
    deletion check, and only the call site distinguishes them.  Order
    matters; the first match wins.  ``chain`` is the lowercased
    ``file:function`` chain of the issuing frames (mirror_memory frames
    only, nearest last).
    """
    low = sql.lower()
    if "grant_purpose" in chain or "grant_bench_purpose" in chain:
        return "bench_grant_setup"
    if "mm_access_audit" in low or "access_audit" in chain:
        return "access_audit_writes"
    if "authorization.py" in chain or "authorize_or_raise" in chain or "authorize<" in chain or ":authorize" in chain:
        return "purpose_auth_gate"
    if "deletion.py" in chain or "tombstones.py" in chain or "replay.py" in chain \
            or "mm_deletion" in low or "mm_tombstone" in low:
        return "deletion_snapshot_checks"
    if "watermark" in low:
        return "sync_catchup_gate"
    if "cross_purpose" in chain or "visibility" in chain or "purpose" in low:
        return "cross_purpose_visibility"
    return "other"


def _build_pg_env(work: Path) -> tuple[str, dict[str, str]]:
    """Create the PG login roles and a fresh RLS schema, register the
    maintenance routing, and return (app_runtime_url, env).  Mirrors the
    test harness fixtures (tests/conftest.py) so standalone runs exercise
    the identical hosted shape."""
    os.environ.update(MM_DATABASE_URL=_ADMIN_URL, MM_PG_RLS="1",
                      MM_TENANT_READ_FILTER="app", MM_SQL_AUDIT="0")
    tests_dir = os.path.abspath("tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    if os.path.abspath("src") not in sys.path:
        sys.path.insert(0, os.path.abspath("src"))
    import conftest
    import psycopg
    from psycopg import sql as psql
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url

    from mirror_memory.core import rls

    admin = _ADMIN_URL
    conninfo = make_url(admin).set(drivername="postgresql").render_as_string(hide_password=False)
    roles = (
        (rls.owner_role(), False, None),
        (rls.app_role(), True, conftest.PG_TEST_ROLE_PASSWORDS["app"]),
        (rls.maintenance_role(), True, conftest.PG_TEST_ROLE_PASSWORDS["maintenance"]),
    )
    with psycopg.connect(conninfo, autocommit=True) as conn:
        for name, login, pw in roles:
            exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (name,)).fetchone()
            if exists is None:
                stmt = (
                    psql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(psql.Identifier(name), psql.Literal(pw))
                    if login else psql.SQL("CREATE ROLE {} NOLOGIN").format(psql.Identifier(name))
                )
                conn.execute(stmt)
            elif login:
                conn.execute(
                    psql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(psql.Identifier(name), psql.Literal(pw))
                )
    engine = create_engine(admin)
    conftest._pg_fresh_schema(engine)
    engine.dispose()
    app_url = conftest._pg_role_url("app")
    maint_url = conftest._pg_role_url("maintenance")
    rls.register_maintenance_url(admin, maint_url)
    rls.register_maintenance_url(app_url, maint_url)
    return app_url, mode_env()


def _reload_mirror_memory() -> None:
    for mod in [m for m in sys.modules if m.startswith("mirror_memory")]:
        del sys.modules[mod]


def cmd_profile(work: Path) -> dict:
    """cProfile one small run_longrun per revision on a fresh PG schema."""
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"per_rev": {}, "scenario": SMALL}
    for label in REV_ORDER:
        _use_rev(REVS[label])
        _reload_mirror_memory()
        app_url, env = _build_pg_env(work)
        for k, v in env.items():
            if k.startswith("MM_"):
                os.environ[k] = v
        longrun = importlib.import_module("mirror_memory.bench.longrun")
        from mirror_memory.config.loader import load_config

        cfg = load_config("config/")
        if hasattr(cfg, "deletion_log"):
            cfg.deletion_log.directory = str(work / f"deletion_{label}")
        prof = cProfile.Profile()
        prof.enable()
        t0 = time.monotonic()
        longrun.run_longrun(cfg, scenario_kwargs=SMALL, database_url=app_url)
        wall = time.monotonic() - t0
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(50)
        (work / f"profile_{label}.txt").write_text(buf.getvalue(), encoding="utf-8")
        stats = pstats.Stats(prof)
        stats.sort_stats("cumulative")
        top = []
        for func, (_cc, nc, _tt, ct, _callers) in stats.stats.items():
            top.append({"func": f"{func[2]} ({os.path.basename(func[0])}:{func[1]})",
                        "cum_s": round(ct, 3), "calls": nc})
        top.sort(key=lambda d: d["cum_s"], reverse=True)
        result["per_rev"][label] = {"rev": REVS[label], "wall_s": round(wall, 3),
                                    "top_cumulative": top[:30]}
        print(f"profile {label} ({REVS[label]}): wall {wall:.1f}s -> {work}/profile_{label}.txt", flush=True)
    (work / "profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cmd_statements(work: Path) -> dict:
    """Count SQL statements issued during one small longrun per revision,
    bucketed by the increment source that issued them."""
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"per_rev": {}, "scenario": SMALL}
    for label in REV_ORDER:
        _use_rev(REVS[label])
        _reload_mirror_memory()
        app_url, env = _build_pg_env(work)
        for k, v in env.items():
            if k.startswith("MM_"):
                os.environ[k] = v
        from sqlalchemy import event
        from sqlalchemy.engine import Engine

        counts: dict[str, int] = {}
        cats: dict[str, int] = {}
        total = [0]

        def _on_exec(conn, cursor, statement, parameters, context, executemany):
            import traceback

            total[0] += 1
            frames = traceback.extract_stack(limit=80)
            chain = " ".join(
                f"{os.path.basename(f.filename)}:{f.name}"
                for f in frames if "/mirror_memory/" in f.filename or "/tests/" in f.filename
            )
            cat = _classify(statement, chain)
            cats[cat] = cats.get(cat, 0) + 1
            key = re.sub(r"\s+", " ", statement.strip())[:200]
            counts[key] = counts.get(key, 0) + 1

        event.listen(Engine, "before_cursor_execute", _on_exec)
        try:
            from mirror_memory.bench import longrun
            from mirror_memory.config.loader import load_config

            cfg = load_config("config/")
            if hasattr(cfg, "deletion_log"):
                cfg.deletion_log.directory = str(work / f"stmt_deletion_{label}")
            longrun.run_longrun(cfg, scenario_kwargs=SMALL, database_url=app_url)
            # the bench grants the `benchmark` purpose for its own users;
            # exclude those issuance statements (one-time setup per run),
            # keeping only the steady-state per-entry statement load.
            cats.pop("bench_grant_setup", None)
        finally:
            event.remove(Engine, "before_cursor_execute", _on_exec)
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        result["per_rev"][label] = {
            "rev": REVS[label],
            "total_statements": total[0],
            "by_category": cats,
            "top_statements": [{"sql": s, "count": c} for s, c in top],
        }
        print(f"statements {label} ({REVS[label]}): total={total[0]} cats={cats}", flush=True)
    (work / "statements.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cmd_restore() -> None:
    _restore(Path("."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["measure", "profile", "statements", "restore"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--work", default="/tmp/m5_perf_diag")
    args = ap.parse_args()
    work = Path(args.work)
    try:
        if args.command == "measure":
            cmd_measure(args.rounds, work)
        elif args.command == "profile":
            cmd_profile(work)
        elif args.command == "statements":
            cmd_statements(work)
        else:
            cmd_restore()
    finally:
        # always leave the worktree on the starting branch
        if args.command != "restore":
            _restore(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
