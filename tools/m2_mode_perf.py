"""M2 batch-2 task 5: the three-mode performance final on PostgreSQL.

Same revision, three read-filter modes, interleaved rounds::

    C  measurement-only baseline -- no tenant read filter at all
       (MM_TENANT_READ_FILTER=none, MM_ALLOW_UNFILTERED_MEASUREMENT=1,
       MM_PG_RLS=0: no policies, no per-transaction setting, superuser
       connection).  Never shipped.
    A  application-layer read injection + RLS (production today)
    B  RLS carries the read filter; the application keeps the write checks

Methodology = M1 task 0 / M2 batch-1 patch, with "mode" where they had
"revision":

- the measured set is the ID INTERSECTION of the tests that PASS in every
  mode's calibration run (tenant-isolation assertions fail in C by design
  and drop out of all three);
- every run writes junit XML; a round is VALID only when all three modes
  pass exactly |ids| tests with zero failures/errors;
- the SQL auditor is OFF in every leg (the M1 acceptance figure).

Subcommands: ``calibrate``, ``measure``, ``gate``, ``decide``, ``all``.
``--files`` restricts calibration/measurement to a few test files for a
local smoke run of the tool itself.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

MODES: dict[str, dict[str, str]] = {
    "C": {"MM_PG_RLS": "0", "MM_TENANT_READ_FILTER": "none", "MM_ALLOW_UNFILTERED_MEASUREMENT": "1"},
    "A": {"MM_PG_RLS": "1", "MM_TENANT_READ_FILTER": "app"},
    "B": {"MM_PG_RLS": "1", "MM_TENANT_READ_FILTER": "rls"},
}
ORDER = ("C", "A", "B")
_MODE_KEYS = {k for m in MODES.values() for k in m}

GATE_FILES = ["tests/test_rls.py", "tests/test_concurrency_ported.py"]


def mode_env(tag: str, admin_url: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _MODE_KEYS}
    env.update(MODES[tag])
    env["MM_SQL_AUDIT"] = "0"
    env["MM_DATABASE_URL"] = admin_url
    env.pop("MM_LONGRUN_DATABASE_URL", None)
    return env


def _pytest(args: list[str], env: dict[str, str], junit: Path, out: Path) -> float:
    cmd = [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"]
    start = time.monotonic()
    with open(out, "w", encoding="utf-8") as fh:
        subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return time.monotonic() - start


def _run_ids(ids_file: Path, env: dict[str, str], junit: Path, out: Path) -> float:
    cmd = [sys.executable, "tools/run_ids.py", str(ids_file), "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"]
    start = time.monotonic()
    with open(out, "w", encoding="utf-8") as fh:
        subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return time.monotonic() - start


def collect_ids(env: dict[str, str], files: list[str] | None) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *(files or [])]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    ids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and line.split("::")[0].endswith(".py"):
            ids.append(line)
    return ids


def node_key(nodeid: str) -> tuple[str, str]:
    """pytest node id -> (junit classname, junit name)."""
    path, _, rest = nodeid.partition("::")
    module = path[:-3].replace("/", ".")
    parts = rest.split("::")
    name = parts[-1]
    classes = parts[:-1]
    return (".".join([module, *classes]), name)


def junit_outcomes(path: Path) -> dict[tuple[str, str], str]:
    root = ET.parse(path).getroot()
    out: dict[tuple[str, str], str] = {}
    for case in root.iter("testcase"):
        key = (case.get("classname", ""), case.get("name", ""))
        verdict = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                verdict = child.tag
                break
        out[key] = verdict
    return out


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return {"tests": tests, "passed": tests - failures - errors - skipped, "failed": failures,
            "errors": errors, "skipped": skipped}


def _emit(summary: Path | None, text: str) -> None:
    print(text)
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


# ---------------------------------------------------------------------------


def cmd_calibrate(admin_url: str, out: Path, work: Path, files: list[str] | None, summary: Path | None) -> dict:
    ids = collect_ids(mode_env("A", admin_url), files)
    keys = [node_key(i) for i in ids]
    per_mode: dict[str, dict] = {}
    passed_sets: dict[str, set] = {}
    for tag in ORDER:
        junit = work / f"calib_{tag}.xml"
        secs = _pytest(files or [], mode_env(tag, admin_url), junit, work / f"calib_{tag}.txt")
        outcomes = junit_outcomes(junit)
        passed_sets[tag] = {k for k, v in outcomes.items() if v == "passed"}
        per_mode[tag] = {"secs": round(secs, 1), **junit_counts(junit)}
    common = [i for i, k in zip(ids, keys) if all(k in passed_sets[t] for t in ORDER)]
    dropped: dict[str, dict[str, int]] = {}
    for i, k in zip(ids, keys):
        missing = [t for t in ORDER if k not in passed_sets[t]]
        if missing:
            f = i.split("::")[0]
            dropped.setdefault(f, {}).setdefault(",".join(missing), 0)
            dropped[f][",".join(missing)] += 1
    out.write_text("\n".join(common) + "\n", encoding="utf-8")
    result = {"collected": len(ids), "intersection": len(common), "per_mode": per_mode, "dropped_by_file": dropped}
    _emit(summary, "### calibration")
    _emit(summary, f"collected={len(ids)} intersection={len(common)}")
    for tag in ORDER:
        m = per_mode[tag]
        _emit(summary, f"- {tag}: {m['secs']}s passed={m['passed']} failed={m['failed']} errors={m['errors']} skipped={m['skipped']}")
    if dropped:
        _emit(summary, "dropped from the measured set (file: not-passing-in -> count):")
        for f in sorted(dropped):
            _emit(summary, f"- {f}: " + ", ".join(f"{modes}->{n}" for modes, n in sorted(dropped[f].items())))
    (work / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cmd_measure(admin_url: str, ids_file: Path, rounds: int, work: Path, summary: Path | None) -> dict:
    ids = [l for l in ids_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    lr_file = work / "longrun_ids.txt"
    lr_ids = [i for i in ids if "test_longrun_bench" in i]
    lr_file.write_text("\n".join(lr_ids) + "\n", encoding="utf-8")
    report: dict = {"ids": len(ids), "longrun_ids": len(lr_ids), "rounds": [], "longrun_rounds": []}

    def one_round(kind: str, source: Path, expected: int) -> dict:
        row: dict = {}
        for tag in ORDER:
            junit = work / f"{kind}_{tag}_{len(report[kind + '_rounds' if kind == 'longrun' else 'rounds']) + 1}.xml"
            out = work / (junit.stem + ".txt")
            secs = _run_ids(source, mode_env(tag, admin_url), junit, out)
            counts = junit_counts(junit)
            valid = counts["failed"] == 0 and counts["errors"] == 0 and counts["passed"] == expected
            row[tag] = {"secs": round(secs, 1), **counts, "valid": valid}
        row["valid"] = all(row[t]["valid"] for t in ORDER)
        return row

    _emit(summary, f"### full suite ({len(ids)} ids), interleaved x{rounds}, auditor off")
    for r in range(1, rounds + 1):
        row = one_round("full", ids_file, len(ids))
        report["rounds"].append(row)
        _emit(summary, f"round {r}: " + " | ".join(f"{t} {row[t]['secs']}s passed={row[t]['passed']}" for t in ORDER)
              + f" | {'VALID' if row['valid'] else 'INVALID'}")
    _emit(summary, f"### longrun ({len(lr_ids)} ids), interleaved x{rounds}")
    for r in range(1, rounds + 1):
        row = one_round("longrun", lr_file, len(lr_ids))
        report["longrun_rounds"].append(row)
        _emit(summary, f"round {r}: " + " | ".join(f"{t} {row[t]['secs']}s passed={row[t]['passed']}" for t in ORDER)
              + f" | {'VALID' if row['valid'] else 'INVALID'}")
    report["stats"] = _stats(report)
    (work / "measure.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _emit_stats(summary, report["stats"])
    return report


def _stats(report: dict) -> dict:
    stats: dict = {}
    for kind in ("rounds", "longrun_rounds"):
        valid_rows = [row for row in report[kind] if row["valid"]]
        if not valid_rows:
            stats[kind] = {"valid_rounds": 0}
            continue
        secs = {t: [row[t]["secs"] for row in valid_rows] for t in ORDER}
        agg = {t: {"min": min(v), "median": statistics.median(v), "all": v} for t, v in secs.items()}
        ratios = {}
        for t in ("A", "B"):
            ratios[f"{t}/C"] = {
                "min": round(agg[t]["min"] / agg["C"]["min"], 3),
                "median": round(agg[t]["median"] / agg["C"]["median"], 3),
            }
        ratios["B/A"] = {
            "min": round(agg["B"]["min"] / agg["A"]["min"], 3),
            "median": round(agg["B"]["median"] / agg["A"]["median"], 3),
        }
        stats[kind] = {"valid_rounds": len(valid_rows), "secs": agg, "ratios": ratios}
    return stats


def _emit_stats(summary: Path | None, stats: dict) -> None:
    for kind, label in (("rounds", "full suite"), ("longrun_rounds", "longrun")):
        s = stats.get(kind, {})
        if not s.get("valid_rounds"):
            _emit(summary, f"### {label}: no valid rounds")
            continue
        _emit(summary, f"### {label}: {s['valid_rounds']} valid round(s)")
        _emit(summary, "| mode | min s | median s | min ratio vs C | median ratio vs C |")
        _emit(summary, "|---|---:|---:|---:|---:|")
        for t in ORDER:
            r = s["ratios"].get(f"{t}/C", {"min": 1.0, "median": 1.0})
            _emit(summary, f"| {t} | {s['secs'][t]['min']:.1f} | {s['secs'][t]['median']:.1f} | {r['min']:.3f} | {r['median']:.3f} |")
        _emit(summary, f"B/A: min {s['ratios']['B/A']['min']:.3f}, median {s['ratios']['B/A']['median']:.3f}")


def cmd_gate(admin_url: str, work: Path, summary: Path | None) -> dict:
    junit = work / "gate_B.xml"
    secs = _pytest(GATE_FILES, mode_env("B", admin_url), junit, work / "gate_B.txt")
    counts = junit_counts(junit)
    ok = counts["failed"] == 0 and counts["errors"] == 0 and counts["passed"] > 0
    result = {"secs": round(secs, 1), **counts, "ok": ok, "files": GATE_FILES}
    _emit(summary, f"### B isolation gate (mode B, auditor off): {'PASS' if ok else 'FAIL'} "
                   f"passed={counts['passed']} failed={counts['failed']} errors={counts['errors']} skipped={counts['skipped']}")
    (work / "gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cmd_longrun_ac(admin_url: str, rounds: int, work: Path, summary: Path | None) -> dict:
    """A-vs-C longrun only (M2 batch-2 patch acceptance): for when the
    worker's transaction structure changed but everything else did not.
    The longrun ids pass in every mode (both prior full finals), so the
    set is simply tests/test_longrun_bench.py; three interleaved rounds,
    validated the same way."""
    ids = [i for i in collect_ids(mode_env("A", admin_url), ["tests/test_longrun_bench.py"])]
    ids_file = work / "longrun_ac_ids.txt"
    ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    report: dict = {"ids": len(ids), "modes": ["C", "A"], "rounds": []}
    _emit(summary, f"### longrun A vs C ({len(ids)} ids), interleaved x{rounds}, auditor off")
    for r in range(1, rounds + 1):
        row: dict = {}
        for tag in ("C", "A"):
            junit = work / f"ac_{tag}_{r}.xml"
            secs = _run_ids(ids_file, mode_env(tag, admin_url), junit, work / (junit.stem + ".txt"))
            counts = junit_counts(junit)
            row[tag] = {"secs": round(secs, 1), **counts, "valid": counts["failed"] == 0 and counts["errors"] == 0 and counts["passed"] == len(ids)}
        row["valid"] = all(row[t]["valid"] for t in ("C", "A"))
        report["rounds"].append(row)
        _emit(summary, f"round {r}: C {row['C']['secs']}s | A {row['A']['secs']}s | "
                       f"{'VALID' if row['valid'] else 'INVALID'}")
    valid = [r for r in report["rounds"] if r["valid"]]
    if valid:
        c = min(r["C"]["secs"] for r in valid); a = min(r["A"]["secs"] for r in valid)
        report["ratios"] = {"A/C_min": round(a / c, 3), "rounds_valid": len(valid)}
        _emit(summary, f"min ratio A/C = {a / c:.3f} over {len(valid)} valid round(s)")
    (work / "longrun_ac.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def cmd_decide(measure: dict, gate: dict, summary: Path | None) -> dict:
    """Pre-agreed rule: B passes the task-1 isolation gate, longrun B/C
    <= 1.15 and B at least 5 points below A on longrun -> recommend B for
    PostgreSQL (SQLite keeps A); otherwise keep A.  Evaluated on the MIN
    ratios (the M1 acceptance figure); medians are reported alongside."""
    lr = measure["stats"].get("longrun_rounds", {})
    full = measure["stats"].get("rounds", {})
    if not lr.get("valid_rounds"):
        decision = {"recommend": "A", "reason": "no valid longrun round"}
    else:
        b_c = lr["ratios"]["B/C"]["min"]
        a_c = lr["ratios"]["A/C"]["min"]
        gap_points = round((a_c - b_c) * 100, 1)
        conditions = {
            "gate_ok": bool(gate.get("ok")),
            "longrun_B_over_C_le_1.15": b_c <= 1.15,
            "B_at_least_5_points_below_A": gap_points >= 5.0,
        }
        recommend = "B" if all(conditions.values()) else "A"
        decision = {"recommend": recommend, "conditions": conditions,
                    "longrun_min_ratios": {"A/C": a_c, "B/C": b_c, "A-B points": gap_points},
                    "longrun_median_ratios": {k: v["median"] for k, v in lr["ratios"].items()},
                    "full_min_ratios": {k: v["min"] for k, v in full.get("ratios", {}).items()},
                    "full_median_ratios": {k: v["median"] for k, v in full.get("ratios", {}).items()}}
    _emit(summary, "### decision")
    _emit(summary, "```json\n" + json.dumps(decision, indent=2) + "\n```")
    return decision


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["calibrate", "measure", "gate", "decide", "all", "longrun-ac"])
    ap.add_argument("--admin-url", required=True)
    ap.add_argument("--work", default="/tmp/m2_mode_perf")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--files", nargs="*", default=None, help="restrict to these test files (tool smoke run)")
    ap.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY") or None)
    ap.add_argument("--out", default=None, help="final JSON report path (all/decide)")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    summary = Path(args.summary) if args.summary else None
    ids_file = work / "mode_ids.txt"

    if args.command == "calibrate":
        cmd_calibrate(args.admin_url, ids_file, work, args.files, summary)
    elif args.command == "measure":
        cmd_measure(args.admin_url, ids_file, args.rounds, work, summary)
    elif args.command == "gate":
        cmd_gate(args.admin_url, work, summary)
    elif args.command == "decide":
        measure = json.loads((work / "measure.json").read_text(encoding="utf-8"))
        gate = json.loads((work / "gate.json").read_text(encoding="utf-8"))
        cmd_decide(measure, gate, summary)
    elif args.command == "longrun-ac":
        cmd_longrun_ac(args.admin_url, args.rounds, work, summary)
    else:
        calib = cmd_calibrate(args.admin_url, ids_file, work, args.files, summary)
        measure = cmd_measure(args.admin_url, ids_file, args.rounds, work, summary)
        gate = cmd_gate(args.admin_url, work, summary)
        decision = cmd_decide(measure, gate, summary)
        final = {"calibration": calib, "measure": measure, "gate": gate, "decision": decision}
        target = Path(args.out) if args.out else work / "report.json"
        target.write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(f"report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
