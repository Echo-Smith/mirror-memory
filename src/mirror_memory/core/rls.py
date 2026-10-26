"""PostgreSQL row-level security: the second tenant defence (M2 batch-2 task 1).

Roles (names configurable through ``MM_PG_*_ROLE``; defaults in brackets):

- **owner** [``mm_owner``] owns every table and runs DDL / migrations.  It
  is NOT exempt from the policies -- ``FORCE ROW LEVEL SECURITY`` applies
  them to the owner too -- so it is admitted by the maintenance policy.
- **app** [``mm_app``] is the runtime login: not a table owner, no
  ``BYPASSRLS``, cannot ``ALTER TABLE``; it reaches rows only through the
  tenant policy.
- **maintenance** [``mm_maintenance``] is the login the ``system_scope``
  call sites run on, through a SEPARATE engine and pool (the runtime
  allowlist check in ``system_scope`` still applies).  It is deliberately
  NOT ``BYPASSRLS``: it is admitted by a named policy (``USING (true) WITH
  CHECK (true)``) instead.  A policy keeps the role an ordinary login whose
  reach is listed in ``pg_policies`` and can be withdrawn table by table;
  ``BYPASSRLS`` would silently bypass every future policy on every table.

Policies, on every tenant table with ``ENABLE`` + ``FORCE ROW LEVEL
SECURITY``:

- ``mm_tenant_rows`` FOR ALL TO app -- USING and WITH CHECK are both
  ``tenant_id = current_setting('mm.tenant_id', true) AND app_id =
  current_setting('mm.app_id', true)``.  Without a setting in the current
  transaction ``current_setting(..., true)`` is NULL, the predicate is
  NULL, and no row is visible or writable: reads return 0 rows, writes
  fail (fail closed).
- ``mm_maintenance_rows`` FOR ALL TO maintenance, owner -- USING (true)
  WITH CHECK (true).

Tenant setting: a ``begin`` listener on the ``Engine`` class runs
``SELECT set_config('mm.tenant_id', %(tenant_id)s, true), set_config(
'mm.app_id', %(app_id)s, true)`` -- bound parameters, transaction-local
(``is_local = true``), so PostgreSQL itself discards the setting at
COMMIT / ROLLBACK and a pooled connection can never carry one tenant's
setting into the next transaction.  SQLAlchemy 2.x emits ``begin`` for
autobegin as well, on raw connections and Session transactions alike, so
Core statements through ``engine.connect()`` are covered by the same hook
as ORM sessions, and a Session that commits and keeps working begins a new
transaction and the hook fires again.  Autocommit connections are not
covered (a transaction-local setting has nowhere to live); tenant work
must not use them, and the policies fail closed if it does.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import re
import threading
from typing import Iterator

from sqlalchemy import event, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import DDLElement

from mirror_memory.core.tenancy import TenantError, peek_tenant

logger = logging.getLogger(__name__)


class RowSecurityError(TenantError):
    """The row-level-security posture is missing or wrong (fail closed)."""


#: Transaction-local settings the tenant policy reads.
TENANT_GUC = "mm.tenant_id"
APP_GUC = "mm.app_id"

#: Policy names (one pair per tenant table).
TENANT_POLICY = "mm_tenant_rows"
MAINTENANCE_POLICY = "mm_maintenance_rows"

_ROLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _role(env: str, default: str) -> str:
    name = os.environ.get(env, default)
    if not _ROLE_NAME.match(name):
        raise RowSecurityError(
            f"{env} must be a lower-case PostgreSQL identifier "
            f"({_ROLE_NAME.pattern}); got {name!r}"
        )
    return name


def owner_role() -> str:
    return _role("MM_PG_OWNER_ROLE", "mm_owner")


def app_role() -> str:
    return _role("MM_PG_APP_ROLE", "mm_app")


def maintenance_role() -> str:
    return _role("MM_PG_MAINTENANCE_ROLE", "mm_maintenance")


# ---------------------------------------------------------------------------
# Mode switches
# ---------------------------------------------------------------------------

_rls_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "mm_rls_override", default=None
)


def rls_enabled() -> bool:
    """Whether PostgreSQL tenant tables are expected to be RLS-protected.

    On by default (``MM_PG_RLS`` != "0").  Off only for the measurement-only
    baseline (perf mode C): the engine then neither sets the tenant GUC nor
    checks the posture.  A production host must never run that way.
    """
    override = _rls_override.get()
    if override is not None:
        return override
    return os.environ.get("MM_PG_RLS", "1") != "0"


@contextlib.contextmanager
def rls_override(enabled: bool) -> Iterator[None]:
    """Test helper: force :func:`rls_enabled` for the block."""
    token = _rls_override.set(enabled)
    try:
        yield
    finally:
        _rls_override.reset(token)


def is_postgres(bind) -> bool:
    dialect = getattr(bind, "dialect", None)
    return getattr(dialect, "name", None) == "postgresql"


# ---------------------------------------------------------------------------
# Tenant setting: one transaction-local set_config per transaction
# ---------------------------------------------------------------------------

#: psycopg named placeholders; the two values are bound parameters.
SET_TENANT_SQL = (
    "SELECT set_config('" + TENANT_GUC + "', %(tenant_id)s, true), "
    "set_config('" + APP_GUC + "', %(app_id)s, true)"
)

_hooks_installed = False


def _set_tenant_on_begin(conn) -> None:
    if not is_postgres(conn) or not rls_enabled():
        return
    if conn.get_execution_options().get("isolation_level") == "AUTOCOMMIT":
        return
    tenant = peek_tenant()
    if tenant is None:
        return  # no tenant -> no setting -> the policies see nothing
    conn.exec_driver_sql(SET_TENANT_SQL, {"tenant_id": tenant[0], "app_id": tenant[1]})


def install_rls_hooks() -> None:
    """Idempotently attach the per-transaction tenant setting to every Engine."""
    global _hooks_installed
    if _hooks_installed:
        return
    event.listen(Engine, "begin", _set_tenant_on_begin)
    _hooks_installed = True


# ---------------------------------------------------------------------------
# Maintenance engine registry: system_scope work on PostgreSQL runs on a
# separate engine logged in as the maintenance role
# ---------------------------------------------------------------------------

_registry_lock = threading.Lock()
_maintenance_urls: dict[str, str] = {}
_maintenance_engines: dict[str, Engine] = {}


def _url_key(url) -> str:
    """Identity of an application database minus its password."""
    return make_url(url).set(password=None).render_as_string(hide_password=False)


def register_maintenance_url(app_url, maintenance_url) -> None:
    """Pair an application database URL with the maintenance login's URL.

    Keyed by the application URL without its password, so every engine
    created against that database -- the server's shared engine, a worker's
    own engine, a test's ad-hoc one -- resolves to the same maintenance
    engine.
    """
    if not is_postgres_url(maintenance_url):
        raise RowSecurityError("the maintenance URL must be a PostgreSQL URL")
    with _registry_lock:
        _maintenance_urls[_url_key(app_url)] = str(
            make_url(maintenance_url).render_as_string(hide_password=False)
        )


def is_postgres_url(url) -> bool:
    try:
        return make_url(url).get_backend_name() == "postgresql"
    except Exception:  # noqa: BLE001 -- not a URL at all
        return False


def maintenance_engine_for(bind) -> Engine | None:
    """The maintenance engine registered for *bind*'s database, or None.

    The engine is created lazily on first use with a small pool
    (``MM_PG_MAINTENANCE_POOL_SIZE``, default 2): maintenance traffic is a
    queue scan and a startup probe, not request fan-out.
    """
    url = getattr(bind, "url", None) or getattr(getattr(bind, "engine", None), "url", None)
    if url is None:
        return None
    key = _url_key(url)
    with _registry_lock:
        engine = _maintenance_engines.get(key)
        if engine is not None:
            return engine
        maint_url = _maintenance_urls.get(key)
        if maint_url is None:
            return None
    import sqlalchemy

    size = max(1, int(os.environ.get("MM_PG_MAINTENANCE_POOL_SIZE", "2")))
    engine = sqlalchemy.create_engine(
        maint_url,
        pool_size=size,
        max_overflow=size,
        pool_pre_ping=True,
        connect_args={"application_name": "mirror-memory-maintenance"},
    )
    with _registry_lock:
        existing = _maintenance_engines.get(key)
        if existing is not None:
            engine.dispose()
            return existing
        _maintenance_engines[key] = engine
    return engine


def system_engine_for(bind) -> Engine:
    """The engine ``system_scope`` work must run on.

    SQLite, or PostgreSQL with RLS switched off: *bind* itself.  PostgreSQL
    with RLS: the registered maintenance engine -- and a hard error when
    none is registered, because the alternative is a queue scan that runs
    as the app role, sees 0 rows and reports an empty queue.
    """
    if not is_postgres(bind) or not rls_enabled():
        return bind
    engine = maintenance_engine_for(bind)
    if engine is None:
        raise RowSecurityError(
            "system_scope on PostgreSQL needs the maintenance role: set "
            "MM_MAINTENANCE_DATABASE_URL (or call "
            "register_maintenance_url) -- without it the cross-tenant scan "
            "would run as the app role and see no rows at all"
        )
    return engine


def dispose_maintenance_engines() -> None:
    """Drop every cached maintenance engine (tests / shutdown)."""
    with _registry_lock:
        engines = list(_maintenance_engines.values())
        _maintenance_engines.clear()
    for engine in engines:
        engine.dispose()


def clear_maintenance_registry() -> None:
    dispose_maintenance_engines()
    with _registry_lock:
        _maintenance_urls.clear()


def register_maintenance_from_env(app_url) -> bool:
    """Register ``MM_MAINTENANCE_DATABASE_URL`` for *app_url* when set."""
    maint = os.environ.get("MM_MAINTENANCE_DATABASE_URL", "")
    if not maint:
        return False
    register_maintenance_url(app_url, maint)
    return True


# ---------------------------------------------------------------------------
# DDL constructs (PostgreSQL only; identifiers quoted by the dialect)
# ---------------------------------------------------------------------------


def _pg_only(compiler, what: str) -> None:
    if compiler.dialect.name != "postgresql":
        raise RowSecurityError(what + " is PostgreSQL DDL; SQLite has no row-level security")


def _roles(compiler, roles) -> str:
    return ", ".join(compiler.preparer.quote(r) for r in roles)


class CreateRole(DDLElement):
    """``CREATE ROLE <name> LOGIN|NOLOGIN`` -- no password: credentials are
    the operator's (``ALTER ROLE ... PASSWORD`` or external auth)."""

    inherit_cache = True

    def __init__(self, name: str, login: bool) -> None:
        self.name = name
        self.login = login


@compiles(CreateRole)
def _compile_create_role(element, compiler, **kw):
    _pg_only(compiler, "CREATE ROLE")
    return "CREATE ROLE %s %s" % (
        compiler.preparer.quote(element.name),
        "LOGIN" if element.login else "NOLOGIN",
    )


class TransferOwnership(DDLElement):
    inherit_cache = True

    def __init__(self, table: str, role: str) -> None:
        self.table = table
        self.role = role


@compiles(TransferOwnership)
def _compile_transfer_ownership(element, compiler, **kw):
    _pg_only(compiler, "ALTER TABLE OWNER TO")
    return "ALTER TABLE %s OWNER TO %s" % (
        compiler.preparer.quote(element.table),
        compiler.preparer.quote(element.role),
    )


class GrantSchemaUsage(DDLElement):
    inherit_cache = True

    def __init__(self, schema: str, roles: list[str]) -> None:
        self.schema = schema
        self.roles = list(roles)


@compiles(GrantSchemaUsage)
def _compile_grant_schema_usage(element, compiler, **kw):
    _pg_only(compiler, "GRANT USAGE ON SCHEMA")
    return "GRANT USAGE ON SCHEMA %s TO %s" % (
        compiler.preparer.quote(element.schema),
        _roles(compiler, element.roles),
    )


class GrantTableAccess(DDLElement):
    """Row access only -- never ownership, never DDL."""

    inherit_cache = True

    def __init__(self, table: str, roles: list[str]) -> None:
        self.table = table
        self.roles = list(roles)


@compiles(GrantTableAccess)
def _compile_grant_table_access(element, compiler, **kw):
    _pg_only(compiler, "GRANT ON TABLE")
    return "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE %s TO %s" % (
        compiler.preparer.quote(element.table),
        _roles(compiler, element.roles),
    )


class GrantSequenceAccess(DDLElement):
    """Serial columns draw from sequences the app must be able to use."""

    inherit_cache = True

    def __init__(self, schema: str, roles: list[str]) -> None:
        self.schema = schema
        self.roles = list(roles)


@compiles(GrantSequenceAccess)
def _compile_grant_sequence_access(element, compiler, **kw):
    _pg_only(compiler, "GRANT ON ALL SEQUENCES")
    return "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA %s TO %s" % (
        compiler.preparer.quote(element.schema),
        _roles(compiler, element.roles),
    )


class EnableForceRowSecurity(DDLElement):
    inherit_cache = True

    def __init__(self, table: str) -> None:
        self.table = table


@compiles(EnableForceRowSecurity)
def _compile_enable_force_row_security(element, compiler, **kw):
    _pg_only(compiler, "ROW LEVEL SECURITY")
    return "ALTER TABLE %s ENABLE ROW LEVEL SECURITY, FORCE ROW LEVEL SECURITY" % (
        compiler.preparer.quote(element.table),
    )


class DropPolicy(DDLElement):
    inherit_cache = True

    def __init__(self, table: str, policy: str) -> None:
        self.table = table
        self.policy = policy


@compiles(DropPolicy)
def _compile_drop_policy(element, compiler, **kw):
    _pg_only(compiler, "DROP POLICY")
    return "DROP POLICY IF EXISTS %s ON %s" % (
        compiler.preparer.quote(element.policy),
        compiler.preparer.quote(element.table),
    )


class CreateTenantPolicy(DDLElement):
    """The tenant predicate, as USING and WITH CHECK, for the app role."""

    inherit_cache = True

    def __init__(self, table: str, role: str) -> None:
        self.table = table
        self.role = role


def _tenant_predicate(compiler) -> str:
    q = compiler.preparer.quote
    return "%s = current_setting('%s', true) AND %s = current_setting('%s', true)" % (
        q("tenant_id"), TENANT_GUC, q("app_id"), APP_GUC,
    )


@compiles(CreateTenantPolicy)
def _compile_create_tenant_policy(element, compiler, **kw):
    _pg_only(compiler, "CREATE POLICY")
    predicate = _tenant_predicate(compiler)
    return "CREATE POLICY %s ON %s FOR ALL TO %s USING (%s) WITH CHECK (%s)" % (
        compiler.preparer.quote(TENANT_POLICY),
        compiler.preparer.quote(element.table),
        compiler.preparer.quote(element.role),
        predicate,
        predicate,
    )


class CreateMaintenancePolicy(DDLElement):
    """Full access for the maintenance and owner roles -- a named policy,
    not BYPASSRLS (see the module docstring)."""

    inherit_cache = True

    def __init__(self, table: str, roles: list[str]) -> None:
        self.table = table
        self.roles = list(roles)


@compiles(CreateMaintenancePolicy)
def _compile_create_maintenance_policy(element, compiler, **kw):
    _pg_only(compiler, "CREATE POLICY")
    return "CREATE POLICY %s ON %s FOR ALL TO %s USING (true) WITH CHECK (true)" % (
        compiler.preparer.quote(MAINTENANCE_POLICY),
        compiler.preparer.quote(element.table),
        _roles(compiler, element.roles),
    )


# ---------------------------------------------------------------------------
# Catalog probes (bound parameters over pg_catalog only)
# ---------------------------------------------------------------------------


def tenant_table_names() -> list[str]:
    from mirror_memory.core.models import Base

    return [table.name for table in Base.metadata.sorted_tables]


def schema_name() -> str:
    from mirror_memory.core.models import Base

    return Base.metadata.schema or "public"


_ROLE_EXISTS = text("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :name")

_ROLE_ATTRS = text(
    "SELECT rolsuper, rolbypassrls FROM pg_catalog.pg_roles WHERE rolname = :name"
)

_CURRENT_ROLE = text(
    "SELECT current_user, rolsuper, rolbypassrls, "
    "pg_catalog.pg_has_role(current_user, :app, 'USAGE') "
    "FROM pg_catalog.pg_roles WHERE rolname = current_user"
)

_TABLE_STATE = text(
    "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, "
    "pg_catalog.pg_get_userbyid(c.relowner) "
    "FROM pg_catalog.pg_class AS c "
    "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
    "WHERE c.relkind = 'r' AND n.nspname = :schema"
)

_POLICIES = text(
    "SELECT tablename, policyname, roles::text, cmd, qual, with_check "
    "FROM pg_catalog.pg_policies WHERE schemaname = :schema"
)


def missing_roles(engine: Engine) -> list[tuple[str, bool]]:
    """(name, login) for every configured role that does not exist yet."""
    wanted = [(owner_role(), False), (app_role(), True), (maintenance_role(), True)]
    missing = []
    with engine.connect() as conn:
        for name, login in wanted:
            if conn.execute(_ROLE_EXISTS, {"name": name}).first() is None:
                missing.append((name, login))
    return missing


def table_states(engine: Engine) -> dict[str, tuple[bool, bool, str]]:
    """relname -> (rowsecurity, forcerowsecurity, owner) for the schema."""
    with engine.connect() as conn:
        rows = conn.execute(_TABLE_STATE, {"schema": schema_name()}).all()
    return {r[0]: (bool(r[1]), bool(r[2]), r[3]) for r in rows}


def policies(engine: Engine) -> dict[str, dict[str, str]]:
    """tablename -> {policyname: roles-text}."""
    out: dict[str, dict[str, str]] = {}
    with engine.connect() as conn:
        for table, name, roles, _cmd, _qual, _check in conn.execute(
            _POLICIES, {"schema": schema_name()}
        ).all():
            out.setdefault(table, {})[name] = roles
    return out


def _policy_roles(roles_text: str) -> set[str]:
    return {r.strip().strip('"') for r in roles_text.strip("{}").split(",") if r.strip()}


def rls_plan(engine: Engine, tables: list[str] | None = None) -> list[DDLElement]:
    """The DDL that brings *tables* (default: every tenant table) to the
    documented posture.  Every element is idempotent, so the plan is
    re-entrant: ownership and GRANTs are no-ops when already in place,
    ENABLE/FORCE are idempotent, and policies are dropped-if-exists and
    recreated.  Role creation is NOT in the plan -- see
    :func:`missing_roles`; it needs CREATEROLE and is emitted separately
    so a privilege failure happens before anything else is touched."""
    names = tenant_table_names() if tables is None else list(tables)
    owner, app, maint = owner_role(), app_role(), maintenance_role()
    schema = schema_name()
    states = table_states(engine)
    plan: list[DDLElement] = [GrantSchemaUsage(schema, [app, maint])]
    for name in names:
        state = states.get(name)
        if state is None:
            continue  # not created yet (older schema); the step verifies later
        if state[2] != owner:
            plan.append(TransferOwnership(name, owner))
        plan.append(GrantTableAccess(name, [app, maint]))
        plan.append(EnableForceRowSecurity(name))
        plan.append(DropPolicy(name, TENANT_POLICY))
        plan.append(DropPolicy(name, MAINTENANCE_POLICY))
        plan.append(CreateTenantPolicy(name, app))
        plan.append(CreateMaintenancePolicy(name, [maint, owner]))
    plan.append(GrantSequenceAccess(schema, [app, maint]))
    return plan


def verify_posture(engine: Engine, tables: list[str] | None = None) -> None:
    """Structural check (any role): every tenant table is ENABLE+FORCE RLS,
    owned by the owner role, with both policies bound to the right roles;
    the app role exists and is neither superuser nor BYPASSRLS."""
    names = tenant_table_names() if tables is None else list(tables)
    owner, app, maint = owner_role(), app_role(), maintenance_role()
    states = table_states(engine)
    pols = policies(engine)
    problems: list[str] = []
    for name in names:
        state = states.get(name)
        if state is None:
            problems.append(f"{name}: table missing")
            continue
        rowsec, force, table_owner = state
        if not rowsec or not force:
            problems.append(f"{name}: row security enable={rowsec} force={force}")
        if table_owner != owner:
            problems.append(f"{name}: owner {table_owner!r} != {owner!r}")
        table_pols = pols.get(name, {})
        if _policy_roles(table_pols.get(TENANT_POLICY, "")) != {app}:
            problems.append(f"{name}: policy {TENANT_POLICY} not bound to {app!r}")
        if _policy_roles(table_pols.get(MAINTENANCE_POLICY, "")) != {maint, owner}:
            problems.append(
                f"{name}: policy {MAINTENANCE_POLICY} not bound to {sorted((maint, owner))}"
            )
    with engine.connect() as conn:
        attrs = conn.execute(_ROLE_ATTRS, {"name": app}).first()
    if attrs is None:
        problems.append(f"app role {app!r} does not exist")
    elif attrs[0] or attrs[1]:
        problems.append(f"app role {app!r} is superuser={attrs[0]} bypassrls={attrs[1]}")
    if problems:
        raise RowSecurityError("row-level security posture: " + "; ".join(problems))


def verify_runtime_connection(engine: Engine) -> None:
    """The connection a runtime engine actually gets must be SUBJECT to the
    policies: logged in as (or a member of) the app role, not superuser,
    not BYPASSRLS, and not the owner of the tables.  A superuser passes
    every policy silently, which would make the second defence a no-op
    while looking enabled."""
    app, owner = app_role(), owner_role()
    with engine.connect() as conn:
        row = conn.execute(_CURRENT_ROLE, {"app": app}).first()
    if row is None:
        raise RowSecurityError("cannot resolve current_user in pg_roles")
    user, superuser, bypass, member = row
    if superuser or bypass:
        raise RowSecurityError(
            f"runtime login {user!r} is superuser={superuser} bypassrls={bypass}: "
            "row-level security would not apply; connect as the app role"
        )
    if user == owner:
        raise RowSecurityError(
            f"runtime login {user!r} is the table owner; connect as the app role"
        )
    if not member:
        raise RowSecurityError(
            f"runtime login {user!r} is not the app role {app!r} (or a member): "
            "the tenant policy would not admit it"
        )


def prepare_runtime_engine(engine: Engine) -> None:
    """Checks a PostgreSQL runtime engine must pass before serving tenants.

    - the schema exists (the app role never runs DDL; an owner-side
      ``upgrade_database`` must have created it);
    - the posture is in place (:func:`verify_posture`);
    - the login is subject to the policies (:func:`verify_runtime_connection`);
    - the maintenance URL from the environment is registered when present.
    """
    from sqlalchemy import inspect

    missing = [t for t in tenant_table_names() if not inspect(engine).has_table(t)]
    if missing:
        raise RowSecurityError(
            "PostgreSQL schema is not initialised for " + ", ".join(sorted(missing))
            + ": run upgrade_database(...) with the owner role first (the "
            "app role does not create tables)"
        )
    verify_posture(engine)
    verify_runtime_connection(engine)
    register_maintenance_from_env(engine.url)
