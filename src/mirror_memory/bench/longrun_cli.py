"""LongRunBench CLI — one user, one simulated year, measured.

Usage::

    python -m mirror_memory.bench.longrun_cli [--json OUT] [--seed N]

Deterministic (K1 extraction, no LLM): the point is the lifecycle under
load, not extraction quality.  Reports the seven targets and exits non-zero
when any of them misses, so it can gate a release the way MetabolismBench
and StateBench do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mirror_memory.bench.longrun import _TARGETS, render_report, run_longrun
from mirror_memory.config.loader import load_config


def _meets(value, op: str, target: float) -> bool:
    if value is None:
        return False
    if op == ">=":
        return value >= target
    if op == "<":
        return value < target
    return value == target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LongRunBench v1")
    parser.add_argument("--config", default="config/", help="config directory")
    parser.add_argument("--json", default=None, help="write the full report to this path")
    parser.add_argument("--seed", type=int, default=None, help="override the scenario seed")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    kwargs = {"seed": args.seed} if args.seed is not None else {}
    report = run_longrun(config, **kwargs)

    print(render_report(report))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nreport written to {args.json}")

    values = {
        "semantic_preservation": report.semantic_preservation,
        "correction_loss": len(report.correction_loss),
        "conflict_loss": len(report.conflict_loss),
        "forgotten_resurrection": len(report.forgotten_resurrection),
        "wrong_archive": len(report.wrong_archive),
        "hot_evidence_reduction": report.hot_evidence_reduction,
        "recall_p95_ms": report.recall_p95_ms,
    }
    failures = [
        f"{name}={value} (target {op} {target})"
        for name, (op, target) in _TARGETS.items()
        for value in (values[name],)
        if not _meets(value, op, target)
    ]
    if failures:
        print("\nLongRunBench FAILED: " + "; ".join(failures))
        return 1
    print("\nLongRunBench PASSED: all seven targets met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
