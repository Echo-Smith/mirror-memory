"""Session-level multi-tenant isolation for mirror-memory v1 (spike).

Design (spike/v1-tenancy): tenant isolation is enforced by SQLAlchemy
session-wide events, not by touching the ~22 per-user call sites.

- Tenant context is carried in a ``ContextVar``: ``(tenant_id, app_id)``.
  Entry points (``MemoryEngine``, the FastAPI server, background workers,
  test fixtures) set it explicitly; the hooks never fall back to a default.
- **Fail closed**: any ORM statement that touches a tenant-scoped entity
  without a tenant context raises :class:`TenantContextError`.
- Reads (SELECT) and ORM-enabled bulk UPDATE/DELETE get
  ``with_loader_criteria`` injected.
- Writes: ``before_flush`` fills ``tenant_id``/``app_id`` on new objects,
  and refuses any new/dirty/deleted tenant-scoped object whose tenant
  disagrees with the ambient context.
- ``Session.get`` bypasses ``with_loader_criteria`` (identity map / PK
  lookup), so it is wrapped: legacy business-key gets get the tenant prefix
  prepended, and a fetched row from another tenant is detached and reported
  as not found.
- Cross-tenant system work (job-queue scans) must go through the explicit,
  grep-able :func:`system_scope` context manager.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator

from sqlalchemy import Column, String, event
from sqlalchemy.orm import Mapped, Session, with_loader_criteria
from sqlalchemy.orm.decl_api import declared_attr
from sqlalchemy.sql import visitors

_tenant_var: ContextVar[tuple[str, str] | None] = ContextVar(
    "mm_tenant", default=None
)
_system_var: ContextVar[bool] = ContextVar("mm_system_scope", default=False)


class TenantError(RuntimeError):
    """Base class for tenant-isolation failures."""


class TenantContextError(TenantError):
    """A tenant-scoped statement ran with no tenant context set (fail closed)."""


class TenantMismatchError(TenantError):
    """An object's tenant disagrees with the ambient tenant context."""


def current_tenant() -> tuple[str, str]:
    """Return the ambient ``(tenant_id, app_id)`` or raise."""
    tenant = _tenant_var.get()
    if tenant is None:
        raise TenantContextError(
            "no tenant context: set one with tenant_scope()/set_tenant() "
            "at the entry point before touching tenant-scoped tables"
        )
    return tenant


def peek_tenant() -> tuple[str, str] | None:
    """Return the ambient tenant tuple, or ``None`` when unset."""
    return _tenant_var.get()


def set_tenant(tenant_id: str, app_id: str) -> object:
    """Explicitly set the tenant context; returns a token for reset_tenant."""
    return _tenant_var.set((tenant_id, app_id))


def reset_tenant(token: object) -> None:
    _tenant_var.reset(token)  # type: ignore[arg-type]


@contextlib.contextmanager
def tenant_scope(tenant_id: str, app_id: str) -> Iterator[None]:
    """Run the block under an explicit tenant context."""
    token = _tenant_var.set((tenant_id, app_id))
    try:
        yield
    finally:
        _tenant_var.reset(token)


@contextlib.contextmanager
def system_scope() -> Iterator[None]:
    """Explicit system-level bypass for cross-tenant maintenance work.

    Every use must be a grep-able, reviewable call site.  Under this scope
    the read criteria are not injected and flush-time tenant validation is
    skipped; it is reserved for cross-tenant queue/maintenance scans.
    """
    token = _system_var.set(True)
    try:
        yield
    finally:
        _system_var.reset(token)


@contextlib.contextmanager
def no_tenant_scope() -> Iterator[None]:
    """Run the block with the tenant context explicitly absent.

    Test-only helper simulating a missing entry point: the hooks still
    fail closed, so statements touching tenant tables raise.
    """
    token = _tenant_var.set(None)
    try:
        yield
    finally:
        _tenant_var.reset(token)


def in_system_scope() -> bool:
    return _system_var.get()


class TenantScoped:
    """Mixin for tables partitioned by ``(tenant_id, app_id)``.

    ``user_id`` keeps its existing meaning (the subject within the tenant);
    tenancy is the pair of columns this mixin adds.
    """

    @declared_attr
    def tenant_id(cls) -> Mapped[str]:  # noqa: N805
        return Column(String(64), nullable=False, default="", index=True)

    @declared_attr
    def app_id(cls) -> Mapped[str]:  # noqa: N805
        return Column(String(64), nullable=False, default="", index=True)


# ---------------------------------------------------------------------------
# Read path: inject tenant criteria into SELECT / bulk UPDATE / bulk DELETE
# ---------------------------------------------------------------------------


def _statement_mappers(state) -> list[type]:
    mappers = []
    for mapper in getattr(state, "all_mappers", ()) or ():
        cls = getattr(mapper, "class_", None)
        if isinstance(cls, type) and issubclass(cls, TenantScoped):
            mappers.append(cls)
    return mappers


_tenant_tables: set | None = None


def _tenant_table_set() -> set:
    """Every table owned by a TenantScoped subclass (stable after import)."""
    global _tenant_tables
    if _tenant_tables is None:
        _tenant_tables = {
            cls.__table__ for cls in TenantScoped.__subclasses__() if hasattr(cls, "__table__")
        }
    return _tenant_tables


def _statement_touches_tenant_table(statement) -> bool:
    """True when any table (including inside subqueries) is tenant-scoped.

    Needed because aggregate wrappers such as ``Query.count()`` produce a
    column-only top-level SELECT whose ``all_mappers`` is empty while the
    tenant entity hides in the subquery.
    """
    tables = _tenant_table_set()
    if not tables:
        return False
    found = []

    def _visit_table(obj):
        if obj in tables:
            found.append(obj)

    visitors.traverse(statement, {}, {"table": _visit_table})
    return bool(found)


def _is_tenant_statement(state) -> bool:
    if _statement_mappers(state):
        return True
    return _statement_touches_tenant_table(state.statement)


@event.listens_for(Session, "do_orm_execute")
def _inject_tenant_criteria(state) -> None:
    # Column/relationship loads serve an already-validated persistent object;
    # the owning statement was filtered when it ran.
    if state.is_column_load or state.is_relationship_load:
        return
    if not _is_tenant_statement(state):
        return
    if in_system_scope():
        return
    tenant_id, app_id = current_tenant()
    if not (state.is_select or state.is_update or state.is_delete):
        return
    # The closure captures the tenant values themselves: lambda-SQL tracks
    # closure scalar values as part of its cache key, so a different tenant
    # compiles a different cached statement -- no cross-tenant cache bleed.
    state.statement = state.statement.options(
        with_loader_criteria(
            TenantScoped,
            lambda cls: (cls.tenant_id == tenant_id) & (cls.app_id == app_id),
            include_aliases=True,
        )
    )


# ---------------------------------------------------------------------------
# Write path: fill and validate tenant columns at flush
# ---------------------------------------------------------------------------


@event.listens_for(Session, "before_flush")
def _enforce_tenant_on_flush(session, flush_context, instances) -> None:
    pending = [
        obj
        for collection in (session.new, session.dirty, session.deleted)
        for obj in collection
        if isinstance(obj, TenantScoped)
    ]
    if not pending or in_system_scope():
        return
    tenant_id, app_id = current_tenant()

    for obj in session.new:
        if not isinstance(obj, TenantScoped):
            continue
        if obj.tenant_id is None and obj.app_id is None:
            obj.tenant_id = tenant_id
            obj.app_id = app_id
        elif (obj.tenant_id, obj.app_id) != (tenant_id, app_id):
            raise TenantMismatchError(
                f"new {type(obj).__name__} carries tenant "
                f"({obj.tenant_id!r}, {obj.app_id!r}) but context is "
                f"({tenant_id!r}, {app_id!r})"
            )

    for obj in session.dirty:
        if isinstance(obj, TenantScoped) and (obj.tenant_id, obj.app_id) != (
            tenant_id,
            app_id,
        ):
            raise TenantMismatchError(
                f"modified {type(obj).__name__} belongs to tenant "
                f"({obj.tenant_id!r}, {obj.app_id!r}) but context is "
                f"({tenant_id!r}, {app_id!r})"
            )

    for obj in session.deleted:
        if isinstance(obj, TenantScoped) and (obj.tenant_id, obj.app_id) != (
            tenant_id,
            app_id,
        ):
            raise TenantMismatchError(
                f"deleted {type(obj).__name__} belongs to tenant "
                f"({obj.tenant_id!r}, {obj.app_id!r}) but context is "
                f"({tenant_id!r}, {app_id!r})"
            )


# ---------------------------------------------------------------------------
# Session.get: identity-map / PK lookups bypass with_loader_criteria
# ---------------------------------------------------------------------------

_original_get = Session.get


def _tenant_aware_get(self, entity, primary_key, *args, **kwargs):
    if not (isinstance(entity, type) and issubclass(entity, TenantScoped)):
        return _original_get(self, entity, primary_key, *args, **kwargs)
    if in_system_scope():
        return _original_get(self, entity, primary_key, *args, **kwargs)

    from sqlalchemy import inspect as sa_inspect

    mapper = sa_inspect(entity)
    pk_cols = mapper.primary_key

    pk = primary_key if isinstance(primary_key, tuple) else (primary_key,)
    if len(pk_cols) - len(pk) == 2 and pk_cols[0].key == "tenant_id":
        # Legacy business key (pre-tenancy PK): the caller addresses a row
        # inside the ambient tenant, so the tenant prefix is supplied here
        # rather than at each of the get() call sites.
        tenant_id, app_id = current_tenant()
        pk = (tenant_id, app_id) + tuple(pk)
        primary_key = pk

    obj = _original_get(self, entity, primary_key, *args, **kwargs)
    if obj is None:
        return None
    tenant_id, app_id = current_tenant()
    if (obj.tenant_id, obj.app_id) != (tenant_id, app_id):
        self.expunge(obj)
        return None
    return obj


Session.get = _tenant_aware_get  # type: ignore[assignment]
