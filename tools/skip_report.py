"""Skip inventory from a junit XML run (M3 batch-2 task 0: report numbers
come from the junit, never from hand counts).

Usage:
    python tools/skip_report.py RUN.xml                 # per-file table + reasons
    python tools/skip_report.py RUN.xml --markdown      # same, as markdown tables
    python tools/skip_report.py RUN.xml --diff BASE.xml # skips in RUN absent from BASE

The reasons section lists every (file, skip message) pair with its count,
so a report can quote the declared reason verbatim instead of paraphrasing.
``--diff`` keys skipped testcases by (classname, name) and prints the ones
that are skipped in RUN but not skipped (or not present) in BASE -- the
"new skips since <revision>" list.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter


def _cases(path: str):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        yield from suite.iter("testcase")


def _skip_message(case) -> str | None:
    for child in case:
        if child.tag == "skipped":
            return (child.get("message") or child.text or "").strip()
    return None


def _file_for(classname: str) -> str:
    module = classname
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    if not module.startswith("tests."):
        return module
    parts = module.split(".")
    return f"{parts[0]}/{parts[1]}.py" if len(parts) >= 2 else module


def inventory(path: str):
    per_file_total: Counter = Counter()
    per_file_skip: Counter = Counter()
    reasons: Counter = Counter()
    for case in _cases(path):
        file_key = _file_for(case.get("classname", ""))
        per_file_total[file_key] += 1
        message = _skip_message(case)
        if message is not None:
            per_file_skip[file_key] += 1
            reasons[(file_key, message)] += 1
    return per_file_total, per_file_skip, reasons


def skipped_ids(path: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for case in _cases(path):
        message = _skip_message(case)
        if message is not None:
            out[(case.get("classname", ""), case.get("name", ""))] = message
    return out


def print_inventory(path: str, markdown: bool) -> None:
    per_file_total, per_file_skip, reasons = inventory(path)
    total = sum(per_file_total.values())
    skipped = sum(per_file_skip.values())
    if markdown:
        print("| file | total | skipped |")
        print("|---|---:|---:|")
        for file_key in sorted(per_file_total):
            print(f"| {file_key} | {per_file_total[file_key]} | {per_file_skip[file_key]} |")
        print(f"| **TOTAL** | **{total}** | **{skipped}** |")
        print()
        print("| file | skipped | declared reason |")
        print("|---|---:|---|")
        for (file_key, message), n in sorted(reasons.items()):
            print(f"| {file_key} | {n} | `{message}` |")
        return
    print(f"{'file':48s} {'total':>5s} {'skipped':>7s}")
    for file_key in sorted(per_file_total):
        print(f"{file_key:48s} {per_file_total[file_key]:>5d} {per_file_skip[file_key]:>7d}")
    print(f"{'TOTAL':48s} {total:>5d} {skipped:>7d}")
    print()
    print("skip reasons (file, count, declared message):")
    for (file_key, message), n in sorted(reasons.items()):
        print(f"  {file_key:46s} {n:>3d}  {message}")


def print_diff(run: str, base: str, markdown: bool) -> None:
    now = skipped_ids(run)
    before = skipped_ids(base)
    new = {k: v for k, v in now.items() if k not in before}
    gone = {k: v for k, v in before.items() if k not in now}
    if markdown:
        print(f"| new skip ({len(new)}) | declared reason |")
        print("|---|---|")
        for (cls, name), message in sorted(new.items()):
            print(f"| {_file_for(cls)}::{name} | `{message}` |")
    else:
        print(f"skips in {run} not in {base}: {len(new)}")
        for (cls, name), message in sorted(new.items()):
            print(f"  {_file_for(cls)}::{name}  --  {message}")
    print()
    print(f"skips in base no longer skipped (or removed): {len(gone)}")
    for (cls, name), message in sorted(gone.items()):
        print(f"  {_file_for(cls)}::{name}  --  {message}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("junit")
    parser.add_argument("--diff", metavar="BASE", help="baseline junit to compare against")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    if args.diff:
        print_diff(args.junit, args.diff, args.markdown)
    else:
        print_inventory(args.junit, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
