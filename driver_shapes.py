"""Capture the top statement shapes executed by the longrun benchmark."""
import re
import sys

import pytest
from sqlalchemy import create_engine, event

import mirror_memory.core.tenancy as ten
from mirror_memory.core.tenancy import tenant_scope


def normalize(sql: str) -> str:
    try:
        sql = re.sub(r"\s+", " ", sql).strip().lower()
        sql = re.sub(r" in \([^\)]*\)", " in (?...)", sql)
        sql = re.sub(r"from (.+?) (where|order|group|limit|$)", r"from <tables> \1 \2", sql)
        return sql[:160]
    except Exception:  # noqa: BLE001
        return "<unparseable>"


shapes = {}

orig_create = create_engine


def capturing_create_engine(*args, **kwargs):
    eng = orig_create(*args, **kwargs)

    def cap(conn, cursor, statement, parameters, context, executemany):
        key = normalize(statement)
        shapes[key] = shapes.get(key, 0) + 1

    event.listen(eng, "before_cursor_execute", cap)
    return eng


import sqlalchemy
sqlalchemy.create_engine = capturing_create_engine

with tenant_scope("default", "default"):
    rc = pytest.main(["tests/test_longrun_bench.py", "-q", "-p", "no:cacheprovider"])
    del rc

total = sum(shapes.values())
print(f"\ntotal statements: {total}, distinct shapes: {len(shapes)}")
print("top 12 shapes:")
for shape, n in sorted(shapes.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {n:6d}  {shape}")
