"""Run exactly the test IDs listed in a file (one per line), plus any
extra pytest arguments.  Used by the task-0 measurement methodology so
parametrized IDs containing spaces survive intact (shell word-splitting
mangles them)."""

from __future__ import annotations

import sys

import pytest


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python tools/run_ids.py IDS_FILE [pytest args...]")
        raise SystemExit(2)
    with open(sys.argv[1], encoding="utf-8") as fh:
        ids = [line.strip() for line in fh if line.strip()]
    raise SystemExit(pytest.main(ids + sys.argv[2:]))


if __name__ == "__main__":
    main()
