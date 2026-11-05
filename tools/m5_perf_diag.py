"""M5 Task 0 -- PostgreSQL performance diagnosis across three revisions.

DIAGNOSIS ONLY.  This script measures and decomposes; it never edits the
shared checkout's ``src/``/``tests/``.  Each revision is materialised in
its OWN detached git worktree (``/tmp/mm-perf-<short>``), so three agents
can share one checkout without anyone swapping files under the others:

    git worktree add --detach /tmp/mm-perf-403 403f38d   # M2
    git worktree add --detach /tmp/mm-perf-aed aed0639   # M3
    git worktree add --detach /tmp/mm-perf-d61 d61e7b7   # M4

Every run imports the revision under test through ``PYTHONPATH=<wt>/src``
and executes from inside that worktree directory.  The shared checkout is
only ever READ.

What it measures (all on PostgreSQL, mode A = RLS on + application read
filter, SQL auditor off -- production's shape today):

  measure     Interleaved three-point wall time.  For round in 1..R:
              for rev in [M2, M3, M4]: run tests/test_longrun_bench.py
              (18 tests) in that rev's worktree, record wall time + junit.
              Prints median and min per revision; saves raw rounds.
  statements  Per-revision SQL statement COUNTS from one representative
              longrun test via a before_cursor_execute listener, bucketed
              into the increment sources by the issuing call site.  COUNT
              only (no per-statement timing -- see caveats).
  profile     Per-revision cProfile of run_longrun (Python-time
              attribution) on a fresh PG schema built from that revision.

  worktrees   Create the three worktrees (measure/statements/profile do this
              automatically).
  cleanup     Remove the worktrees this tool created.

Methodology borrowed from tools/perf_ids.py (worktree isolation) and
tools/m2_mode_perf.py (mode_env, junit accounting).  Run from the shared
checkout root with the project venv::

    MM_DATABASE_URL=postgresql://postgres:mm@127.0.0.1:5437/mm_test \
    .venv/bin/python tools/m5_perf_diag.py measure --rounds 3

Limitations (repeated in the report): local Docker PG 16 is smaller than a
CI runner; each test_longrun_bench.py run is minutes long, so a 3-point
sweep is 9 runs of that length; per-statement TIMING is not captured (only
counts) -- there is no APM inside every execute, so it is stated as a
limitation rather than fabricated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# revision label -> commit; worktree dir derived from the short sha.
REVS: dict[str, str] = {"M2": "403f38d", "M3": "aed0639", "M4": "d61e7b7"}
REV_ORDER = ("M2", "M3", "M4")
LONGRUN = "tests/test_longrun_bench.py"
EXPECTED_TESTS = 18             # full file: tests/test_longrun_bench.py has 18 tests
#: When MEASURE_TEST is a single test id, a leg is valid iff 1 test passes.
def _expected_tests() -> int:
    return EXPECTED_TESTS if os.environ.get("MEASURE_TEST", "single") == "all" else 1
SHORT = {"M2": "403", "M3": "aed", "M4": "d61"}
WT_ROOT = Path("/tmp")
WT = {label: WT_ROOT / f"mm-perf-{SHORT[label]}" for label in REV_ORDER}

_ADMIN_URL = os.environ.get("MM_DATABASE_URL", "postgresql://postgres:mm@127.0.0.1:5437/mm_test")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable

# Mode A = production today: RLS on + application read filter, SQL auditor
# off for measurement (the M1 acceptance figure disables it in every leg).
MODE_A_ENV: dict[str, str] = {
    "MM_PG_RLS": "1",
    "MM_TENANT_READ_FILTER": "app",
    "MM_SQL_AUDIT": "0",
}
_LEAK_KEYS = {
    "MM_PG_RLS", "MM_TENANT_READ_FILTER", "MM_ALLOW_UNFILTERED_MEASUREMENT",
    "MM_SQL_AUDIT", "MM_LONGRUN_DATABASE_URL", "PYTHONPATH",
}


def mode_env(wt: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _LEAK_KEYS}
    env.update(MODE_A_ENV)
    env["MM_DATABASE_URL"] = _ADMIN_URL
    env["PYTHONPATH"] = str(wt / "src")
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# worktrees
# ---------------------------------------------------------------------------


def ensure_worktrees() -> None:
    existing = subprocess.run(
        ["git", "worktree", "list", "--porcelain"], cwd=_REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    for label in REV_ORDER:
        wt = WT[label]
        if str(wt) in existing or str(wt.resolve()) in existing:
            continue
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), REVS[label]],
            cwd=_REPO_ROOT, check=True, capture_output=True,
        )
        print(f"# worktree {label} -> {wt} @ {REVS[label]}")


def cmd_cleanup() -> None:
    for label in REV_ORDER:
        wt = WT[label]
        if wt.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt)],
                cwd=_REPO_ROOT, check=False, capture_output=True,
            )
            print(f"# removed worktree {wt}")


# ---------------------------------------------------------------------------
# junit (same accounting as m2_mode_perf)
# ---------------------------------------------------------------------------


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return {"tests": tests, "passed": tests - failures - errors - skipped,
            "failed": failures, "errors": errors, "skipped": skipped}


def _run_pytest(wt: Path, env: dict[str, str], junit: Path, out: Path,
                extra_args: list[str] | None = None) -> float:
    # MEASURE_TEST (env) selects a representative subset instead of the full
    # 18-test file: the full file runs the longrun scenario ~18x, which is
    # fine on a quiet runner but multiplies wall time by 18 on a contended
    # workstation.  ``test_year_survives_and_reports`` runs ONE full scenario
    # and is what the statements/profile passes use, so it is the faithful
    # unit for a wall-clock comparison.  Set MEASURE_TEST=all for the whole
    # file (the CI-shaped run).
    target = os.environ.get("MEASURE_TEST", "tests/test_longrun_bench.py::TestLongRun::test_year_survives_and_reports")
    want = LONGRUN if target == "all" else target
    cmd = [_PY, "-m", "pytest", want, "-q", "-p", "no:cacheprovider",
           f"--junitxml={junit}", *(extra_args or [])]
    start = time.monotonic()
    with open(out, "w", encoding="utf-8") as fh:
        subprocess.run(cmd, cwd=wt, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return time.monotonic() - start


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def cmd_measure(rounds: int, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    ensure_worktrees()
    report: dict = {"mode": "A", "expected_tests": _expected_tests(), "order": "balanced/rotated",
                    "rounds": []}
    base_order = list(REV_ORDER)
    for r in range(1, rounds + 1):
        # BALANCED / ROTATED order: shift the base order by (r-1) so each
        # revision occupies a different POSITION each round.  This removes the
        # last-run/position bias a fixed M2->M3->M4 order injects on a shared
        # (contended) machine.
        shift = (r - 1) % len(base_order)
        order = base_order[shift:] + base_order[:shift]
        row: dict = {"order": order}
        for label in order:
            wt = WT[label]
            junit = work / f"measure_{label}_r{r}.xml"
            out = work / f"measure_{label}_r{r}.txt"
            secs = _run_pytest(wt, mode_env(wt), junit, out)
            counts = junit_counts(junit)
            valid = counts["failed"] == 0 and counts["errors"] == 0 and counts["passed"] == _expected_tests()
            row[label] = {"rev": REVS[label], "secs": round(secs, 1), "position": order.index(label),
                          **counts, "valid": valid}
            print(f"round {r} {label} ({REVS[label]}) [pos {order.index(label)}]: "
                  f"{row[label]['secs']}s passed={counts['passed']} failed={counts['failed']} "
                  f"errors={counts['errors']} {'VALID' if valid else 'INVALID'}", flush=True)
        row["valid"] = all(row[t]["valid"] for t in REV_ORDER)
        report["rounds"].append(row)
        print(f"round {r} order={order}: {'VALID' if row['valid'] else 'INVALID'}\n", flush=True)
    report["stats"] = _stats(report["rounds"])
    (work / "measure.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_stats(report["stats"])
    return report


def _stats(rounds: list[dict]) -> dict:
    valid_rows = [row for row in rounds if row["valid"]]
    stats: dict = {"valid_rounds": len(valid_rows), "per_rev": {}}
    for label in REV_ORDER:
        samples = [row[label]["secs"] for row in valid_rows]
        if not samples:
            stats["per_rev"][label] = {"n": 0}
            continue
        stats["per_rev"][label] = {"n": len(samples), "min": round(min(samples), 1),
                                   "median": round(statistics.median(samples), 1), "all": samples}
    base = stats["per_rev"].get("M2", {})
    for label in ("M3", "M4"):
        cur = stats["per_rev"].get(label, {})
        if base.get("min") and cur.get("min"):
            cur["min_ratio_vs_M2"] = round(cur["min"] / base["min"], 3)
            cur["median_ratio_vs_M2"] = round(cur["median"] / base["median"], 3)
    return stats


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
        print(f"| {label} ({REVS[label]}) | {s['n']} | {s['min']:.1f} | "
              f"{s['median']:.1f} | {mn:.3f} | {md:.3f} |")


# ---------------------------------------------------------------------------
# statements  (via a tiny pytest plugin counting SQL on the bench engine)
# ---------------------------------------------------------------------------

COUNTER_PLUGIN = r'''
"""pytest plugin: count SQL statements per increment source during the bench.

Attaches before_cursor_execute to the SQLAlchemy Engine class (the bench
builds its engine through pg_engine.build_engine, which is sqlalchemy.create_engine
under the hood and therefore covered).  Buckets by issuing call site, then
writes a JSON blob to $M5_STMT_OUT at session finish.
"""
import json, os, re, traceback

def _classify(sql, chain):
    low = sql.lower()
    if "grant_purpose" in chain or "grant_bench_purpose" in chain:
        return "bench_grant_setup"
    if "mm_access_audit" in low or "access_audit" in chain:
        return "access_audit_writes"
    if "authorization.py" in chain or "authorize_or_raise" in chain or ":authorize" in chain:
        return "purpose_auth_gate"
    if "deletion.py" in chain or "tombstones.py" in chain or "replay.py" in chain \
            or "mm_deletion" in low or "mm_tombstone" in low:
        return "deletion_snapshot_checks"
    if "watermark" in low:
        return "sync_catchup_gate"
    if "cross_purpose" in chain or "visibility" in chain or "purpose" in low:
        return "cross_purpose_visibility"
    return "other"

def pytest_configure(config):
    from sqlalchemy import event
    from sqlalchemy.engine import Engine
    state = {"total": 0, "cats": {}, "stmts": {}}

    def _on(conn, cursor, statement, parameters, context, executemany):
        state["total"] += 1
        frames = traceback.extract_stack(limit=80)
        chain = " ".join(
            os.path.basename(f.filename) + ":" + f.name
            for f in frames
            if "/mirror_memory/" in f.filename or "/tests/" in f.filename
        )
        cat = _classify(statement, chain)
        state["cats"][cat] = state["cats"].get(cat, 0) + 1
        key = re.sub(r"\s+", " ", statement.strip())[:200]
        state["stmts"][key] = state["stmts"].get(key, 0) + 1

    event.listen(Engine, "before_cursor_execute", _on)
    config._m5_state = state

def pytest_sessionfinish(session, exitstatus):
    state = getattr(session.config, "_m5_state", None)
    out = os.environ.get("M5_STMT_OUT")
    if state is None or not out:
        return
    top = sorted(state["stmts"].items(), key=lambda kv: kv[1], reverse=True)[:10]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"total_statements": state["total"], "by_category": state["cats"],
                   "top_statements": [{"sql": s, "count": c} for s, c in top]}, fh, indent=2)
'''


def cmd_statements(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    ensure_worktrees()
    result: dict = {"per_rev": {}}
    for label in REV_ORDER:
        wt = WT[label]
        plugin = wt / "m5_sqlcount.py"
        plugin.write_text(COUNTER_PLUGIN, encoding="utf-8")
        out_json = work / f"statements_{label}.json"
        junit = work / f"statements_{label}.xml"
        env = mode_env(wt, {"M5_STMT_OUT": str(out_json),
                            "PYTHONPATH": str(wt / "src") + os.pathsep + str(wt)})
        args = ["-p", "m5_sqlcount", "-k", "test_year_survives_and_reports"]
        _run_pytest(wt, env, junit, work / f"statements_{label}.txt", extra_args=args)
        plugin.unlink(missing_ok=True)
        if out_json.exists():
            data = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            data = {"total_statements": None, "by_category": {}, "top_statements": [],
                    "error": "plugin produced no output"}
        data["rev"] = REVS[label]
        result["per_rev"][label] = data
        print(f"statements {label} ({REVS[label]}): total={data.get('total_statements')} "
              f"cats={data.get('by_category')}", flush=True)
    (work / "statements.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

SMALL = dict(days=40, preferences=12, goals=5, corrections=6, conflicts=5, forgets=3)


def _prepare_pg_schema(work: Path, label: str) -> str:
    """Build the PG login roles + a fresh RLS schema using the revision's own
    tests/conftest.py, and return the app runtime URL.  Runs in the
    revision's worktree so the harness matches the code under test."""
    wt = WT[label]
    code = (
        "import os,sys\n"
        "sys.path.insert(0,os.path.abspath('tests'));sys.path.insert(0,os.path.abspath('src'))\n"
        "import conftest\n"
        "from sqlalchemy import create_engine\n"
        "from mirror_memory.core import rls\n"
        "from psycopg import sql as psql\n"
        "from sqlalchemy.engine import make_url\n"
        "admin=os.environ['MM_DATABASE_URL']\n"
        "ci=make_url(admin).set(drivername='postgresql').render_as_string(hide_password=False)\n"
        "roles=((rls.owner_role(),False,None),(rls.app_role(),True,conftest.PG_TEST_ROLE_PASSWORDS['app']),"
        "(rls.maintenance_role(),True,conftest.PG_TEST_ROLE_PASSWORDS['maintenance']))\n"
        "import psycopg as P\n"
        "for name,login,pw in roles:\n"
        "    with P.connect(ci,autocommit=True) as c:\n"
        "        ex=c.execute('SELECT 1 FROM pg_roles WHERE rolname=%s',(name,)).fetchone()\n"
        "        if ex is None:\n"
        "            stmt=(psql.SQL('CREATE ROLE {} LOGIN PASSWORD {}').format(psql.Identifier(name),psql.Literal(pw)) if login else psql.SQL('CREATE ROLE {} NOLOGIN').format(psql.Identifier(name)))\n"
        "            c.execute(stmt)\n"
        "        elif login:\n"
        "            c.execute(psql.SQL('ALTER ROLE {} WITH LOGIN PASSWORD {}').format(psql.Identifier(name),psql.Literal(pw)))\n"
        "e=create_engine(admin); conftest._pg_fresh_schema(e); e.dispose()\n"
        "print(conftest._pg_role_url('app'))\n"
    )
    env = mode_env(wt)
    proc = subprocess.run([_PY, "-c", code], cwd=wt, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"schema prep failed for {label}:\n{proc.stderr[-1500:]}")
    return proc.stdout.strip().splitlines()[-1]


STATEMENTS_PROFILE = (
    "import os,sys,cProfile,pstats,json,io\n"
    "sys.path.insert(0,os.path.abspath('src'))\n"
    "from mirror_memory.bench import longrun\n"
    "from mirror_memory.config.loader import load_config\n"
    "cfg=load_config('config/')\n"
    "if hasattr(cfg,'deletion_log'):\n"
    "    cfg.deletion_log.directory=os.environ['M5_DELDIR']\n"
    "kwargs=json.loads(os.environ['M5_SCENARIO'])\n"
    "prof=cProfile.Profile();prof.enable()\n"
    "longrun.run_longrun(cfg,scenario_kwargs=kwargs,database_url=os.environ['M5_APP_URL'])\n"
    "prof.disable()\n"
    "buf=io.StringIO();pstats.Stats(prof,stream=buf).sort_stats('cumulative').print_stats(50)\n"
    "open(os.environ['M5_PROFILE_OUT'],'w').write(buf.getvalue())\n"
    "stats=pstats.Stats(prof);stats.sort_stats('cumulative')\n"
    "top=[]\n"
    "for func,(cc,nc,tt,ct,callers) in stats.stats.items():\n"
    "    top.append({'func':func[2]+' ('+os.path.basename(func[0])+':'+str(func[1])+')','cum_s':round(ct,3),'calls':nc})\n"
    "top.sort(key=lambda d:d['cum_s'],reverse=True)\n"
    "open(os.environ['M5_PROFILE_JSON'],'w').write(json.dumps(top[:30]))\n"
)


def cmd_profile(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    ensure_worktrees()
    result: dict = {"per_rev": {}, "scenario": SMALL}
    for label in REV_ORDER:
        wt = WT[label]
        try:
            app_url = _prepare_pg_schema(work, label)
        except Exception as exc:  # noqa: BLE001 -- record, don't fabricate
            result["per_rev"][label] = {"rev": REVS[label], "error": str(exc)[-800:]}
            print(f"profile {label}: SKIPPED ({exc})", flush=True)
            continue
        env = mode_env(wt, {
            "M5_APP_URL": app_url, "M5_DELDIR": str(work / f"del_{label}"),
            "M5_SCENARIO": json.dumps(SMALL),
            "M5_PROFILE_OUT": str(work / f"profile_{label}.txt"),
            "M5_PROFILE_JSON": str(work / f"profile_{label}.json"),
        })
        t0 = time.monotonic()
        proc = subprocess.run([_PY, "-c", STATEMENTS_PROFILE], cwd=wt, env=env,
                              capture_output=True, text=True)
        wall = time.monotonic() - t0
        if proc.returncode != 0:
            result["per_rev"][label] = {"rev": REVS[label], "error": proc.stderr[-1200:]}
            print(f"profile {label}: FAILED\n{proc.stderr[-600:]}", flush=True)
            continue
        top = json.loads((work / f"profile_{label}.json").read_text(encoding="utf-8"))
        result["per_rev"][label] = {"rev": REVS[label], "wall_s": round(wall, 3), "top_cumulative": top}
        print(f"profile {label} ({REVS[label]}): wall {wall:.1f}s", flush=True)
    (work / "profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["measure", "statements", "profile", "cleanup", "worktrees"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--work", default="/tmp/m5_perf_diag")
    args = ap.parse_args()
    work = Path(args.work)
    if args.command == "measure":
        cmd_measure(args.rounds, work)
    elif args.command == "statements":
        cmd_statements(work)
    elif args.command == "profile":
        cmd_profile(work)
    elif args.command == "cleanup":
        cmd_cleanup()
    else:
        ensure_worktrees()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
