#!/usr/bin/env python3
"""M4 batch-2 patch 2, task 4: per-test overhead recomputed from junit.

Given the base- and head-revision junit XML of one round, compute the
per-test time ratio (head/base) for every test that passed on BOTH
sides, EXCLUDING tests whose file changed in the git range (their
content differs, so their ratio measures the change, not the overhead).

Printed, with at least 3 significant digits:
- the number of excluded test FILES (and their names, one per line into
  the step summary);
- the number of tests remaining in the ratio;
- the per-test ratios' median and the totals ratio (sum head / sum base).

Usage:
    python tools/perf_junit_ratio.py base.xml head.xml \
        [--git-range aed0639..HEAD]
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET


def _cases(path: str) -> dict[str, float]:
    """testcase id -> seconds, for passed tests only."""
    out: dict[str, float] = {}
    for _, elem in ET.iterparse(path):
        if elem.tag != "testcase":
            continue
        status = "passed"
        for child in elem:
            if child.tag in ("failure", "error"):
                status = child.tag
        if status != "passed":
            elem.clear()
            continue
        classname = elem.get("classname", "")
        name = elem.get("name", "")
        time_attr = elem.get("time", "0")
        out[f"{classname}::{name}"] = float(time_attr)
        elem.clear()
    return out


def _changed_test_files(git_range: str) -> set[str]:
    diff = subprocess.run(
        ["git", "diff", "--name-only", git_range, "--", "tests"],
        capture_output=True, text=True, check=True).stdout
    return {line.strip() for line in diff.splitlines() if line.strip()}


def _file_of(test_id: str) -> str:
    # classname looks like "tests.test_mesh" or
    # "tests.test_mesh.Class" (class-grouped cases)
    parts = test_id.split("::", 1)[0].split(".")
    if parts and parts[0] == "tests":
        return "/".join(parts[:-1] if len(parts) > 2 else parts) + ".py"
    return "/".join(parts) + ".py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_xml")
    parser.add_argument("head_xml")
    parser.add_argument("--git-range", default="aed0639..HEAD")
    args = parser.parse_args()

    base = _cases(args.base_xml)
    head = _cases(args.head_xml)
    excluded = _changed_test_files(args.git_range)

    ratios: list[float] = []
    sum_base = sum_head = 0.0
    excluded_tests = 0
    for test_id, head_t in head.items():
        if test_id not in base:
            continue
        if _file_of(test_id) in excluded:
            excluded_tests += 1
            continue
        base_t = base[test_id]
        if base_t <= 0.0:
            continue
        ratios.append(head_t / base_t)
        sum_base += base_t
        sum_head += head_t

    med = statistics.median(ratios)
    totals = sum_head / sum_base if sum_base else float("nan")
    print(f"excluded_test_files={len(excluded)}")
    for f in sorted(excluded):
        print(f"  excluded: {f}")
    print(f"tests_compared={len(ratios)} (changed-file tests skipped: "
          f"{excluded_tests})")
    print(f"median_per_test_ratio={med:.4g}")
    print(f"totals_ratio={totals:.4g}")


if __name__ == "__main__":
    sys.exit(main())
