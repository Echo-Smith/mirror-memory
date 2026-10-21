"""Pytest plugin: disable the tenancy read-hook injection for A/B tests.

Load with:  pytest -p noinject_plugin ...
Only removes do_orm_execute; before_flush and the Session.get wrapper
stay, so the composite-PK schema keeps working.
"""


def pytest_configure(config):
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    import mirror_memory.core.tenancy as ten

    event.remove(Session, "do_orm_execute", ten._inject_tenant_criteria)
