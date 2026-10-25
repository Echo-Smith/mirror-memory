"""Production multi-tenant isolation for mirror-memory (M1 batch 1).

Design (merge-plan M1; replaces the spike implementation):

- Tenant context is carried in a ``ContextVar`` ``(tenant_id, app_id)``.
  It may only be set through :func:`tenant_scope`; there is no bare setter.
  Exiting restores the previous value on every path, including exceptions.
- **Sessions are bound at creation**: every ``Session`` snapshots the
  ambient tenant into ``session.info`` at ``__init__`` time.  Executing a
  statement when the ambient tenant differs from the session's bound
  tenant raises :class:`TenantMismatchError` -- a session may not be
  reused across tenants, and a tenant switch can never silently re-filter
  an existing session.
- **Fail closed**: any tenant-scoped statement without a tenant context
  raises :class:`TenantContextMissing` (never None, never empty string,
  never another tenant's data).
- Reads (SELECT) and ORM-enabled bulk UPDATE/DELETE are filtered by one
  *tenant-agnostic* :func:`with_loader_criteria` option: the criteria
  template is a lambda over the entity class whose comparisons bind to two
  module-level ``bindparam`` callables reading the ambient tenant at
  execution time, with ``track_closure_variables=False`` so the template
  is built exactly once.  The lambda is closure-frozen (see
  ``_assert_closure_frozen``) -- a template that captures local state is a
  cross-tenant leak waiting to happen and refuses to import.
- Writes: ``before_flush`` fills ``tenant_id``/``app_id`` on new objects,
  refuses new objects that contradict the ambient tenant, refuses
  modifying the tenant columns of existing rows, and refuses any tenant
  object whose tenant disagrees with either the ambient context or the
  session's bound tenant.
- ``Session.get`` cannot see loader criteria (identity map / PK lookup),
  so it is wrapped: legacy business keys get the tenant prefix prepended,
  and a row from another tenant is detached and reported as missing.
- Cross-tenant system work (job-queue scans) must go through
  :func:`system_scope`, the only bypass, which requires a *reason* string.
- Thread pools must propagate context explicitly via
  :func:`submit_with_tenant` / :class:`TenantAwareExecutor`
  (:func:`contextvars.copy_context`); bare pools fail closed.
- See ``docs/tenancy-context.md`` for the usage rules.
"""

from __future__ import annotations

import contextlib
import functools
from contextvars import ContextVar
from typing import Any, Callable, Iterator

from sqlalchemy import Column, String, bindparam, event
from sqlalchemy.orm import Mapped, Session, with_loader_criteria
from sqlalchemy.orm.decl_api import declared_attr
from sqlalchemy.sql import visitors


class TenantError(RuntimeError):
    """Base class for tenant-isolation failures."""


class TenantContextMissing(TenantError):
    """A tenant-scoped operation ran with no tenant context set."""


#: Historical name kept as an alias -- the fail-closed path raises
#: :class:`TenantContextMissing`.
TenantContextError = TenantContextMissing


class TenantMismatchError(TenantError):
    """A session or object's tenant disagrees with the ambient context."""


_tenant_var: ContextVar[tuple[str, str] | None] = ContextVar("mm_tenant", default=None)
_system_var: ContextVar[tuple[bool, str] | None] = ContextVar("mm_system", default=None)

#: Name ``session.info`` holds the tenant bound at creation time.
BOUND_TENANT_KEY = "mm_bound_tenant"


def peek_tenant() -> tuple[str, str] | None:
    """Return the ambient tenant tuple, or ``None`` when unset."""
    return _tenant_var.get()


def current_tenant() -> tuple[str, str]:
    """Return the ambient tenant or raise :class:`TenantContextMissing`."""
    tenant = _tenant_var.get()
    if tenant is None:
        raise TenantContextMissing(
            "no tenant context: open one with tenant_scope(tenant_id, app_id) "
            "at the entry point before touching tenant-scoped tables"
        )
    return tenant


@contextlib.contextmanager
def tenant_scope(tenant_id: str, app_id: str) -> Iterator[None]:
    """Run the block under an explicit tenant context.

    The only sanctioned way to set a tenant.  The previous value (including
    "unset") is restored on every exit path.
    """
    if not tenant_id or not app_id:
        raise ValueError("tenant_scope requires non-empty tenant_id and app_id")
    token = _tenant_var.set((tenant_id, app_id))
    try:
        yield
    finally:
        _tenant_var.reset(token)


@contextlib.contextmanager
def system_scope(reason: str) -> Iterator[None]:
    """Cross-tenant bypass for system maintenance work (queue scans).

    The only sanctioned bypass.  *reason* is mandatory and recorded; the
    test-time SQL auditor logs every statement executed under it together
    with this reason, and the second M1 batch adds a CI whitelist check so
    new call sites cannot appear unreviewed.
    """
    if not reason or not reason.strip():
        raise ValueError("system_scope requires a non-empty reason")
    token = _system_var.set((True, reason))
    try:
        yield
    finally:
        _system_var.reset(token)


def in_system_scope() -> bool:
    state = _system_var.get()
    return bool(state and state[0])


def system_reason() -> str | None:
    state = _system_var.get()
    return state[1] if state else None


@contextlib.contextmanager
def no_tenant_scope() -> Iterator[None]:
    """Run the block with the tenant context explicitly absent.

    Test-only helper simulating a missing entry point; the hooks still
    fail closed, so statements touching tenant tables raise.
    """
    token = _tenant_var.set(None)
    try:
        yield
    finally:
        _tenant_var.reset(token)


# ---------------------------------------------------------------------------
# Thread / executor helpers
# ---------------------------------------------------------------------------


def submit_with_tenant(executor: Any, fn: Callable, /, *args: Any, **kwargs: Any) -> Any:
    """Submit *fn* to *executor* with the current tenant context attached.

    Worker threads do not inherit ``ContextVar`` state, so a bare pool would
    run tenant work without a context (fail closed) -- or worse, with a
    stale one if the callable itself sets one.  This wrapper copies the
    submitting context, which carries the tenant (and system scope).
    """
    ctx = contextvars_copy()
    return executor.submit(ctx.run, functools.partial(fn, *args, **kwargs))


def contextvars_copy() -> Any:
    import contextvars

    return contextvars.copy_context()


class TenantAwareExecutor:
    """Wrapper over a ``concurrent.futures`` executor that propagates the
    submitting context (tenant, system scope) into every submitted call."""

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def submit(self, fn: Callable, /, *args: Any, **kwargs: Any) -> Any:
        return submit_with_tenant(self._executor, fn, *args, **kwargs)

    def map(self, fn: Callable, *iterables: Any, **kwargs: Any):
        ctx = contextvars_copy()
        return self._executor.map(lambda *a, **kw: ctx.run(fn, *a, **kw), *iterables, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)


# ---------------------------------------------------------------------------
# Tenant-scoped tables
# ---------------------------------------------------------------------------


class TenantScoped:
    """Mixin for tables partitioned by ``(tenant_id, app_id)``.

    ``user_id`` keeps its existing meaning (the subject within the tenant).
    The columns are deliberately not single-column indexed: tenants are
    few (low cardinality), every tenant predicate pairs the two columns,
    and the composite unique/PK constraints that lead with them already
    serve point lookups.
    """

    @declared_attr
    def tenant_id(cls) -> Mapped[str]:  # noqa: N805
        return Column(String(64), nullable=False, default="")

    @declared_attr
    def app_id(cls) -> Mapped[str]:  # noqa: N805
        return Column(String(64), nullable=False, default="")


# ---------------------------------------------------------------------------
# Read path: one tenant-agnostic loader-criteria option for every statement
# ---------------------------------------------------------------------------

_tenant_bind = bindparam("mm_tenant_id", callable_=lambda: current_tenant()[0])
_app_bind = bindparam("mm_app_id", callable_=lambda: current_tenant()[1])


def _criteria_template(cls):
    return (cls.tenant_id == _tenant_bind) & (cls.app_id == _app_bind)


def _assert_closure_frozen(fn: Callable) -> None:
    """Refuse criteria templates that capture local state.

    A template whose ``__code__.co_freevars`` is non-empty depends on
    closure cells -- i.e. on values captured where it was defined.  For the
    tenant filter that means a stale tenant could be baked into the cached
    expression and served to another tenant.  The template must reference
    only its ``cls`` argument and module-level globals (the bindparams).
    """
    freevars = getattr(getattr(fn, "__code__", None), "co_freevars", ())
    if freevars:
        raise TenantError(
            f"tenant criteria template {fn!r} captures closure variables "
            f"{freevars!r}; it must be closure-frozen (module-level "
            "bindparams only) or a stale tenant could be cached into the "
            "shared criteria"
        )


_assert_closure_frozen(_criteria_template)

_LOADER_OPTION = with_loader_criteria(
    TenantScoped,
    _criteria_template,
    include_aliases=True,
    track_closure_variables=False,
)


def _session_bound_tenant(session: Session) -> tuple[str, str] | None:
    return session.info.get(BOUND_TENANT_KEY)


def _check_session_tenant(session: Session, tenant: tuple[str, str] | None) -> None:
    """The ambient tenant must match the tenant the session was created in."""
    if tenant is None:
        return
    bound = _session_bound_tenant(session)
    if bound is not None and bound != tenant:
        raise TenantMismatchError(
            f"session was created under tenant {bound!r} but the statement "
            f"runs under {tenant!r}; sessions are single-tenant -- open a "
            "new session per tenant"
        )
    if bound is None:
        raise TenantMismatchError(
            "session has no bound tenant (it was created outside "
            "tenant_scope); create sessions inside the tenant context they "
            "will serve"
        )


@event.listens_for(Session, "do_orm_execute")
def _inject_tenant_criteria(state) -> None:
    # Column/relationship loads serve an already-validated persistent object;
    # the owning statement was filtered when it ran.
    if state.is_column_load or state.is_relationship_load:
        return
    if not state.is_orm_statement:
        return  # text()/Core statements are not injected; the SQL auditor covers them
    if not (state.is_select or state.is_update or state.is_delete):
        return
    if in_system_scope():
        return
    tenant = _tenant_var.get()
    if tenant is not None:
        _check_session_tenant(state.session, tenant)
        # Fast path: attach the shared tenant-agnostic criteria.  It only
        # binds to TenantScoped entities (values resolved at execution), so
        # statements over other tables are unaffected, and it reaches
        # entities inside subqueries too (Query.count()).
        state.statement = state.statement.options(_LOADER_OPTION)
        return
    # Fail-closed decision path (no tenant context): only statements that
    # actually touch tenant tables must raise; the rest pass through.
    if _statement_mappers(state):
        raise TenantContextMissing(_NO_CONTEXT_MSG)
    if _statement_touches_tenant_table(state.statement):
        raise TenantContextMissing(_NO_CONTEXT_MSG)


_NO_CONTEXT_MSG = (
    "no tenant context: open one with tenant_scope(tenant_id, app_id) at "
    "the entry point before touching tenant-scoped tables"
)


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
            cls.__table__
            for cls in _iter_tenant_classes()
            if hasattr(cls, "__table__")
        }
    return _tenant_tables


def _iter_tenant_classes():
    stack = list(TenantScoped.__subclasses__())
    while stack:
        cls = stack.pop()
        yield cls
        stack.extend(cls.__subclasses__())


def _statement_touches_tenant_table(statement) -> bool:
    """True when any table (including inside subqueries) is tenant-scoped.

    Needed because aggregate wrappers such as ``Query.count()`` produce a
    column-only top-level SELECT whose ``all_mappers`` is empty while the
    tenant entity hides in the subquery.  Only consulted on the
    fail-closed path (no ambient tenant), never on the fast path.
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


# ---------------------------------------------------------------------------
# Write path: fill and validate tenant columns at flush
# ---------------------------------------------------------------------------


@event.listens_for(Session, "before_flush")
def _enforce_tenant_on_flush(session, flush_context, instances) -> None:
    if in_system_scope():
        return
    new, dirty, deleted = session.new, session.dirty, session.deleted
    if not (new or dirty or deleted):
        return
    classes = frozenset(_iter_tenant_classes())
    tenant = _tenant_var.get()

    if tenant is not None:
        _check_session_tenant(session, tenant)
    elif any(obj.__class__ in classes for collection in (new, dirty, deleted) for obj in collection):
        raise TenantContextMissing(
            "flushing tenant-scoped rows requires a tenant context"
        )

    from sqlalchemy import inspect as sa_inspect

    for obj in new:
        if obj.__class__ not in classes:
            continue
        if obj.tenant_id is None and obj.app_id is None:
            obj.tenant_id, obj.app_id = tenant
        elif (obj.tenant_id, obj.app_id) != tenant:
            raise TenantMismatchError(
                f"new {type(obj).__name__} carries tenant "
                f"({obj.tenant_id!r}, {obj.app_id!r}) but context is {tenant!r}"
            )

    for obj in dirty:
        if obj.__class__ not in classes:
            continue
        if (obj.tenant_id, obj.app_id) != tenant:
            raise TenantMismatchError(
                f"modified {type(obj).__name__} belongs to tenant "
                f"({obj.tenant_id!r}, {obj.app_id!r}) but context is {tenant!r}"
            )
        state = sa_inspect(obj)
        for col in ("tenant_id", "app_id"):
            hist = state.attrs[col].history
            if hist.has_changes():
                raise TenantMismatchError(
                    f"tenant column {type(obj).__name__}.{col} is immutable "
                    "(move the data, do not re-tenant the row)"
                )

    for obj in deleted:
        if obj.__class__ in classes and (obj.tenant_id, obj.app_id) != tenant:
            raise TenantMismatchError(
                f"deleted {type(obj).__name__} belongs to tenant "
                f"({obj.tenant_id!r}, {obj.app_id!r}) but context is {tenant!r}"
            )


# ---------------------------------------------------------------------------
# Session creation binding + Session.get guard
# ---------------------------------------------------------------------------

_original_session_init = Session.__init__
_original_get = Session.get


def _tenant_bound_session_init(self, *args: Any, **kwargs: Any) -> None:
    _original_session_init(self, *args, **kwargs)
    # Bind the session to the tenant that was ambient at creation time.
    # A session created outside any scope is bound to None and may only be
    # used for non-tenant work or under system_scope.
    self.info[BOUND_TENANT_KEY] = _tenant_var.get()


def _tenant_aware_get(self, entity, primary_key, *args, **kwargs):
    if not (isinstance(entity, type) and issubclass(entity, TenantScoped)):
        return _original_get(self, entity, primary_key, *args, **kwargs)
    if in_system_scope():
        return _original_get(self, entity, primary_key, *args, **kwargs)

    tenant = _tenant_var.get()
    if tenant is None:
        raise TenantContextMissing(
            "session.get() on a tenant-scoped entity requires a tenant context"
        )
    _check_session_tenant(self, tenant)

    mapper = entity.__mapper__
    pk_cols = mapper.primary_key

    pk = primary_key if isinstance(primary_key, tuple) else (primary_key,)
    if len(pk_cols) - len(pk) == 2 and pk_cols[0].key == "tenant_id":
        # Legacy business key (pre-tenancy PK): the caller addresses a row
        # inside the ambient tenant, so the tenant prefix is supplied here
        # rather than at each of the get() call sites.
        pk = tenant + tuple(pk)
        primary_key = pk

    obj = _original_get(self, entity, primary_key, *args, **kwargs)
    if obj is None:
        return None
    if (obj.tenant_id, obj.app_id) != tenant:
        self.expunge(obj)
        return None
    return obj


def install_session_hooks() -> None:
    """Idempotently install the session-level hooks.

    Called on import (below); kept as a function so hosts that construct
    the module dynamically (tests that reload it) can re-apply.
    """
    Session.__init__ = _tenant_bound_session_init  # type: ignore[assignment]
    Session.get = _tenant_aware_get  # type: ignore[assignment]


install_session_hooks()
