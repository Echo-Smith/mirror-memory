"""Audit retention maintenance (M3 batch 3 task 3): the worker's periodic
pass drains the access audit once its retention window has passed.

Per tenant, in batches: tenants with audit rows are enumerated once under
``system_scope`` (a READ of the audit table's tenant columns -- a new
system_scope use point, reviewer 待评审), then each tenant's rows are
deleted through the sanctioned writer inside that tenant's scope, in
bounded batches that commit between them so a backlog never holds one
long transaction.  Never runs for a tenant it has not seen, and the
engine never enumerates tenants for a host's own call.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: The system_scope reason for the enumeration read.
_ENUMERATE_REASON = "audit maintenance: list tenants holding access-audit rows"


def purge_expired_audit(session_factory: Any, config: Any, *, now: float | None = None) -> int:
    """Run the retention purge for every tenant that has audit rows.

    Returns the total number of rows removed.  Safe to run on any tick:
    a tenant with nothing expired costs one bounded batch probe.
    """
    from mirror_memory.core import access_audit
    from mirror_memory.core.models import AccessAudit
    from mirror_memory.core.tenancy import system_scope, tenant_scope

    cutoff = access_audit.retention_cutoff(config.access_audit.retention_days)
    batch = config.access_audit.purge_batch_size

    with system_scope(reason=_ENUMERATE_REASON):
        session = session_factory() if callable(session_factory) else session_factory
        try:
            tenants = (
                session.query(AccessAudit.tenant_id, AccessAudit.app_id)
                .distinct()
                .all()
            )
        finally:
            session.close()

    total = 0
    for tenant_id, app_id in tenants:
        while True:
            with tenant_scope(tenant_id, app_id):
                session = session_factory() if callable(session_factory) else session_factory
                try:
                    removed = access_audit.purge_access_audit(
                        session, older_than=cutoff, limit=batch,
                    )
                    session.commit()
                finally:
                    session.close()
            total += removed
            if removed < batch:
                break
    return total


def next_due(state: dict[str, float], config: Any, *, now: float | None = None) -> bool:
    """Is the maintenance pass due?  *state* is the worker loop's small
    dict (``{"next": monotonic}``); the first call is always due."""
    current = time.monotonic() if now is None else now
    if current >= state.get("next", 0.0):
        state["next"] = current + config.access_audit.purge_interval_seconds
        return True
    return False
