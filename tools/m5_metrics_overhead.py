"""M5 §5b: CONTROLLED longrun experiment — metrics ENABLED vs DISABLED.

Requirement (handover §5): "用真实开启/关闭指标的受控实验验证 longrun 增量
≤1%".  The µs-extrapolation in ``tools/m5_metrics_bench.py`` is a fallback;
this script is the controlled experiment it asks for.

Design (why this is a fair comparison):

* The SAME process, the SAME scenario, the SAME database shape — only the
  metric recording switch (``metrics.set_enabled``) changes.
* Runs are INTERLEAVED and paired (ON, OFF, OFF, ON, ... / alternating) so
  machine-load drift cancels within each pair rather than biasing one arm.
* Each arm is a full ``run_longrun`` (one simulated year, ~510 observe
  entries) so the measured path is the product path (extraction + recall +
  persistence + audit), not a micro-benchmark.
* Reports per-arm wall time, the per-pair ratio, median ratio, and the
  overall delta as a percentage — the number the ≤1% gate is judged on.

If the wall-clock noise swamps the (genuinely sub-percent) effect, the script
says so explicitly and reports the paired median rather than pretending a
single run is decisive.

Usage::

    .venv/bin/python tools/m5_metrics_overhead.py --pairs 3
    .venv/bin/python tools/m5_metrics_overhead.py --pairs 3 --database-url sqlite://
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path


def _one_run(database_url: str, tmp: Path, *, metrics_on: bool) -> tuple[float, float]:
    """One full longrun with metrics ON or OFF; returns (wall_s, cpu_s).

    CPU time (``process_time``) is the robust signal on a contended machine:
    wall clock swings by tens of percent from scheduler noise, while CPU time
    is stable to a few percent.  Both are reported; the CPU-time ratio is the
    number the ≤1% question is answered on, with wall clock alongside for
    transparency."""
    from mirror_memory.bench.longrun import run_longrun
    from mirror_memory.config.loader import load_config
    from mirror_memory.core import metrics

    metrics.set_enabled(metrics_on)
    try:
        cfg = load_config("config/")
        cfg.deletion_log.directory = str(tmp / ("del-" + uuid.uuid4().hex[:6]))
        (tmp / "deletions").mkdir(parents=True, exist_ok=True)
        wall0 = time.monotonic()
        cpu0 = time.process_time()
        run_longrun(
            cfg,
            user_id="bench-overhead-" + uuid.uuid4().hex[:6],
            database_url=database_url,
        )
        return time.monotonic() - wall0, time.process_time() - cpu0
    finally:
        metrics.set_enabled(True)
        metrics.reset_for_tests()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=3,
                    help="number of ON/OFF pairs (each pair = one ON + one OFF run)")
    ap.add_argument("--database-url", default="sqlite://",
                    help="backend for the experiment (default in-memory SQLite)")
    ap.add_argument("--out", default="", help="write the JSON result here")
    ap.add_argument("--blocks", type=int, default=0,
                    help="paired-block mode: number of alternating blocks (0 = full longrun pairs)")
    ap.add_argument("--observes-per-block", type=int, default=50,
                    help="paired-block mode: observe() calls per block")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="m5ovh-"))
    if args.blocks:
        result = _block_experiment(args.database_url, tmp, blocks=args.blocks,
                                   observes_per_block=args.observes_per_block)
        out = Path(args.out) if args.out else tmp / "overhead_block.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nreport -> {out}")
        return 0
    report: dict = {
        "database_url": args.database_url,
        "pairs": [],
        "note": "same process, same scenario, interleaved ON/OFF pairs",
    }
    print(f"# {args.pairs} ON/OFF pairs on {args.database_url}", file=sys.stderr)

    for i in range(1, args.pairs + 1):
        # alternate which arm runs first so ordering bias cancels across pairs
        if i % 2 == 1:
            on_s, on_c = _one_run(args.database_url, tmp, metrics_on=True)
            off_s, off_c = _one_run(args.database_url, tmp, metrics_on=False)
        else:
            off_s, off_c = _one_run(args.database_url, tmp, metrics_on=False)
            on_s, on_c = _one_run(args.database_url, tmp, metrics_on=True)
        ratio = on_s / off_s if off_s else float("inf")
        cpu_ratio = on_c / off_c if off_c else float("inf")
        delta_pct = (ratio - 1.0) * 100.0
        report["pairs"].append({
            "pair": i, "on_s": round(on_s, 2), "off_s": round(off_s, 2),
            "ratio": round(ratio, 4), "delta_pct": round(delta_pct, 3),
            "on_cpu_s": round(on_c, 2), "off_cpu_s": round(off_c, 2),
            "cpu_ratio": round(cpu_ratio, 4),
            "cpu_delta_pct": round((cpu_ratio - 1.0) * 100.0, 3),
        })
        print(f"pair {i}: wall ON {on_s:.2f}s OFF {off_s:.2f}s ({delta_pct:+.2f}%) | "
              f"cpu ON {on_c:.2f}s OFF {off_c:.2f}s "
              f"({(cpu_ratio - 1.0) * 100.0:+.2f}%)", file=sys.stderr)

    ratios = [p["ratio"] for p in report["pairs"]]
    deltas = [p["delta_pct"] for p in report["pairs"]]
    median_ratio = statistics.median(ratios)
    # Paired median is the headline; the range shows whether noise dominates.
    report["median_ratio"] = round(median_ratio, 4)
    report["median_delta_pct"] = round((median_ratio - 1.0) * 100.0, 3)
    report["min_delta_pct"] = round(min(deltas), 3)
    report["max_delta_pct"] = round(max(deltas), 3)
    cpu_ratios = [p["cpu_ratio"] for p in report["pairs"]]
    cpu_median = statistics.median(cpu_ratios)
    report["cpu_median_ratio"] = round(cpu_median, 4)
    report["cpu_median_delta_pct"] = round((cpu_median - 1.0) * 100.0, 3)
    report["cpu_range_pct"] = [round((min(cpu_ratios) - 1.0) * 100.0, 3),
                               round((max(cpu_ratios) - 1.0) * 100.0, 3)]
    # The GATE is judged on CPU time -- wall clock is noise-dominated here.
    report["gate_le_1pct_cpu"] = bool(cpu_median <= 1.01)
    report["gate_le_1pct_wall"] = bool(median_ratio <= 1.01)

    print("\n=== metrics overhead, controlled longrun ON vs OFF ===")
    print(f"{'pair':>4}{'ON s':>10}{'OFF s':>10}{'ratio':>9}{'delta %':>10}")
    for p in report["pairs"]:
        print(f"{p['pair']:>4}{p['on_s']:>10}{p['off_s']:>10}"
              f"{p['ratio']:>9.4f}{p['delta_pct']:>+10.2f}")
    print("\n  wall (noise-dominated): "
          f"median ratio {median_ratio:.4f} ({report['median_delta_pct']:+.2f}%), "
          f"range [{report['min_delta_pct']:+.2f}%, {report['max_delta_pct']:+.2f}%]")
    print(f"  CPU  (robust): median ratio {cpu_median:.4f} "
          f"({report['cpu_median_delta_pct']:+.3f}%), "
          f"range [{report['cpu_range_pct'][0]:+.2f}%, {report['cpu_range_pct'][1]:+.2f}%]")
    print(f"\n  GATE (CPU median delta <= 1%): "
          f"{'PASS' if report['gate_le_1pct_cpu'] else 'FAIL'}")
    print("  (wall-clock gate: "
          f"{'PASS' if report['gate_le_1pct_wall'] else 'FAIL'} -- not decisive on a "
          "contended machine)")

    out = Path(args.out) if args.out else tmp / "overhead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0



# ---------------------------------------------------------------------------
# Paired-block experiment (the low-variance variant)
#
# Full longrun runs are 50-70 s; two of them per pair leaves too few samples to
# beat machine noise.  This variant runs MANY short blocks of real
# ``engine.observe()`` calls, alternating metrics ON / OFF block by block in the
# SAME process against the SAME database, and takes the median of the paired
# CPU-time ratios.  More pairs -> the noise averages out while the workload
# stays the product path.
# ---------------------------------------------------------------------------


def _block_experiment(database_url: str, tmp: Path, *, blocks: int,
                      observes_per_block: int) -> dict:
    import statistics as _st

    from mirror_memory.api import MemoryEngine
    from mirror_memory.config.loader import load_config
    from mirror_memory.core import metrics
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from sqlalchemy.orm import Session

    class _Stub:
        def generate(self, **kw):
            return '{"patterns": [], "support_policy": {}}'

    metrics.set_enabled(True)
    cfg = load_config("config/")
    cfg.llm_client = _Stub()
    (tmp / "del-b").mkdir(parents=True, exist_ok=True)
    cfg.deletion_log.directory = str(tmp / "del-b")
    eng = MemoryEngine(config=cfg, database_url=database_url, llm_client=_Stub())
    eng._ensure_db()
    uid = "block-" + uuid.uuid4().hex[:6]
    with tenant_scope(*eng._tenant), fixture_write_scope("overhead block: enable"):
        s = Session(eng._engine)
        try:
            set_memory_enabled(s, uid, True)
            s.commit()
        finally:
            s.close()
    from mirror_memory.bench import grant_bench_purpose
    grant_bench_purpose(eng, [uid])

    # warm both paths once so first-call costs are not in the samples
    eng.observe(user_id=uid, session_id="w0", text="warm up one.", purpose="benchmark")
    eng.observe(user_id=uid, session_id="w1", text="warm up two.", purpose="benchmark")

    pairs: list[dict] = []
    for b in range(1, blocks + 1):
        arms = (True, False) if b % 2 == 1 else (False, True)
        times: dict[bool, float] = {}
        for on in arms:
            metrics.set_enabled(on)
            c0 = time.process_time()
            w0 = time.monotonic()
            for j in range(observes_per_block):
                eng.observe(user_id=uid, session_id=f"b{b}-{j}",
                            text=f"block {b} sample {j} about tea and mornings.",
                            purpose="benchmark")
            times[on] = (time.process_time() - c0, time.monotonic() - w0)
        metrics.set_enabled(True)
        on_c, on_w = times[True]
        off_c, off_w = times[False]
        pairs.append({
            "block": b,
            "on_cpu_ms": round(on_c * 1000, 1), "off_cpu_ms": round(off_c * 1000, 1),
            "cpu_ratio": round(on_c / off_c, 4) if off_c else None,
            "cpu_delta_pct": round((on_c / off_c - 1) * 100, 3) if off_c else None,
            "wall_delta_pct": round((on_w / off_w - 1) * 100, 3) if off_w else None,
        })
        d = pairs[-1]
        print(f"block {b}: cpu ON {d['on_cpu_ms']}ms OFF {d['off_cpu_ms']}ms "
              f"({d['cpu_delta_pct']:+.2f}%)  wall ({d['wall_delta_pct']:+.2f}%)",
              file=sys.stderr)

    cpu = [p["cpu_delta_pct"] for p in pairs if p["cpu_delta_pct"] is not None]
    wall = [p["wall_delta_pct"] for p in pairs if p["wall_delta_pct"] is not None]
    result = {
        "mode": "paired-block (real engine.observe, alternating ON/OFF)",
        "backend": database_url, "blocks": blocks,
        "observes_per_block": observes_per_block,
        "pairs": pairs,
        "cpu_median_delta_pct": round(_st.median(cpu), 3),
        "cpu_min_delta_pct": round(min(cpu), 3),
        "cpu_max_delta_pct": round(max(cpu), 3),
        "wall_median_delta_pct": round(_st.median(wall), 3),
        "gate_le_1pct": bool(_st.median(cpu) <= 1.0),
    }
    print("\n=== paired-block overhead (metrics ON vs OFF, real observe) ===")
    print(f"blocks={blocks} x {observes_per_block} observes")
    print(f"CPU  median delta {result['cpu_median_delta_pct']:+.3f}%  "
          f"range [{result['cpu_min_delta_pct']:+.2f}%, {result['cpu_max_delta_pct']:+.2f}%]")
    print(f"wall median delta {result['wall_median_delta_pct']:+.3f}%")
    print(f"GATE (CPU median <= 1%): {'PASS' if result['gate_le_1pct'] else 'FAIL'}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
