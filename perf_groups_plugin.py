"""Measurement-only pytest plugin for the perf decomposition groups.

Usage: pytest -p perf_groups_plugin  (with env MM_GROUP=schema)

MM_GROUP=schema disables the tenancy read-injection and flush validation
while keeping the composite-PK session.get wrapper (which is structurally
required -- disabling it fails 11 tests).  Measurement only; it does not
touch any feature code path when MM_GROUP is unset.
"""


def pytest_configure(config):
    import os

    if os.environ.get("MM_GROUP") != "schema":
        return
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    import mirror_memory.core.tenancy as ten

    event.remove(Session, "do_orm_execute", ten._inject_tenant_criteria)
    event.remove(Session, "before_flush", ten._enforce_tenant_on_flush)
