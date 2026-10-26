"""Task 0 (M1 batch 2): strict performance measurement methodology.

Collects the test-ID intersection between two revisions so both sides run
the identical test set, and verifies from junit XML that both sides
actually passed the same number of tests (a round is invalid otherwise).

Usage (from repo root, with both revisions available in git):
    python tools/perf_ids.py collect BASE FIX --out ids.txt
    python tools/perf_ids.py verify base.xml fix.xml   # exit 1 on mismatch
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET


def _use_rev(rev: str) -> None:
    """Make worktree src/tests match *rev* exactly.

    ``git checkout <rev> -- path`` never deletes files that exist in the
    worktree but not in *rev* -- a lingering file (e.g. one deleted by a
    later revision) silently changes the collected test set.  Remove any
    tracked-or-untracked file under src/tests that is not in *rev*'s tree
    first, then materialise the revision.
    """
    keep = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", rev, "--", "src", "tests"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    keep_set = set(keep)
    listed = subprocess.run(
        ["git", "ls-files", "--", "src", "tests"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    import os

    for path in listed:
        if path not in keep_set and os.path.exists(path):
            os.remove(path)
    # untracked leftovers under src/tests (e.g. from an aborted run)
    for root, _dirs, files in os.walk("tests"):
        for name in files:
            if name.endswith(".py"):
                rel = os.path.join(root, name)
                if rel not in keep_set:
                    os.remove(rel)
    subprocess.run(
        ["git", "checkout", "-q", rev, "--", "src", "tests", "pyproject.toml"],
        check=True,
    )
    # Drop index entries for files not in HEAD (e.g. restored-from-base
    # files the revision switch just removed from the worktree).
    subprocess.run(["git", "reset", "-q", "HEAD", "--", "src", "tests"], check=False)


def _collect_ids(rev: str) -> set[str]:
    _use_rev(rev)
    import os

    # The revision switch moves src/ in the WORKTREE, but a venv that
    # installed the package editable still points at whichever checkout it
    # was created from.  Collection has to import the revision under test,
    # or both sides collect the same (wrong) set of parametrizations.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath("src")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    ids = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        # --collect-only -q prints one test id per line ("path::Class::test").
        if "::" in line and line.split("::")[0].endswith(".py"):
            ids.add(line)
    return ids


def cmd_collect(base: str, fix: str, out: str) -> None:
    base_ids = _collect_ids(base)
    fix_ids = _collect_ids(fix)
    common = sorted(base_ids & fix_ids)
    only_fix = sorted(fix_ids - base_ids)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(common) + "\n")
    print(
        f"collect: base={len(base_ids)} fix={len(fix_ids)} "
        f"intersection={len(common)} fix-only={len(only_fix)} -> {out}"
    )


def _read_counts(path: str) -> tuple[int, int, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    passed = failed = errors = 0
    for suite in suites:
        # junit counts skips in `skipped`; passed = tests - failures - errors - skipped
        total = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errs = int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        failed += failures
        errors += errs
        passed += total - failures - errs - skipped
    return passed, failed, errors


def cmd_verify(base_xml: str, fix_xml: str) -> None:
    bp, bf, be = _read_counts(base_xml)
    fp, ff, fe = _read_counts(fix_xml)
    print(f"base: passed={bp} failed={bf} errors={be}")
    print(f"fix : passed={fp} failed={ff} errors={fe}")
    if bf or be or ff or fe:
        print("INVALID round: failures/errors present")
        sys.exit(1)
    if bp != fp:
        print(f"INVALID round: passed counts differ ({bp} vs {fp})")
        sys.exit(1)
    print(f"VALID round: {bp} tests passed on both sides")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--out"]
    if not args:
        print(__doc__)
        sys.exit(2)
    cmd = args[0]
    if cmd == "collect":
        cmd_collect(args[1], args[2], args[3])
    elif cmd == "verify":
        cmd_verify(args[1], args[2])
    else:
        print(f"unknown command {cmd!r}")
        sys.exit(2)


if __name__ == "__main__":
    main()
