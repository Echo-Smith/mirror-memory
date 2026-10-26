"""Test-time SQL auditor (M1 task 3).

Listens to every statement actually sent to the database and fails closed:

- SELECT / UPDATE / DELETE: every tenant table referenced (JOINs, subqueries,
  aliases included) must carry ``tenant_id = ?`` **and** ``app_id = ?``
  equality predicates with *bound* parameters somewhere in the statement.
- INSERT into a tenant table must write both tenant columns.
- Statements inside ``system_scope`` are allowed but recorded with their
  declared reason.
- Statements that cannot be parsed are errors, unless they match the small
  transaction-control / health-check whitelist below.

The auditor is installed by the test suite (conftest) on the ``Engine``
class so it sees every engine; production hosts may opt in the same way.
Disable for performance measurement with ``MM_SQL_AUDIT=0``.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import re
import threading

from sqlalchemy import event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class AuditViolation(AssertionError):
    """A statement violated the tenant-filter audit rules."""


# ---------------------------------------------------------------------------
# Whitelist -- every entry documents why it is safe
# ---------------------------------------------------------------------------

_EXEMPT_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*select\s+1\s*;?\s*$",  # health check: no tables
        r"^\s*begin\b",              # transaction control, no tables
        r"^\s*commit\b",
        r"^\s*rollback\b",
        r"^\s*savepoint\b",          # publisher savepoints, no tables
        r"^\s*release\b",
        r"^\s*pragma\b",             # sqlite introspection (table_info etc.)
        r"^\s*create\s+(table|index|unique\s+index|savepoint)\b",  # DDL in create_all
        r"^\s*drop\s+table\b",       # DDL in teardown
        r"^\s*alter\s+table\b",      # additive migration DDL (check_schema), no row access
        # PostgreSQL catalog introspection issued by SQLAlchemy itself
        # (create_all / drop_all / has_table): metadata queries over the
        # pg_catalog relations, never over tenant data.
        r"^\s*select\s+[\s\S]*?\bfrom\s+pg_catalog\.",
        r"^\s*select\s+[\s\S]*?\bfrom\s+pg_[a-z_]+\b",
        r"^\s*select\s+pg_catalog\.",
    )
)

# SQL text of every exempt statement, for reporting.
_exemptions_seen: list[str] = []
_violations_lock = threading.Lock()


def _exempt(sql: str) -> bool:
    for pat in _EXEMPT_PATTERNS:
        if pat.match(sql):
            return True
    return False


# ---------------------------------------------------------------------------
# sqlglot-based analysis
# ---------------------------------------------------------------------------


def _tenant_table_names() -> set[str]:
    from mirror_memory.core.models import Base

    return {
        t for t in Base.metadata.tables
    }


def _is_bound(value) -> bool:
    import sqlglot.expressions as exp

    return isinstance(value, (exp.Placeholder, exp.Parameter))


def _analyse_statement(stmt, tenant_tables: set[str]) -> None:
    import sqlglot.expressions as exp

    # Map alias-or-name -> real table name for every table reference
    # (FROM, JOIN, subqueries).
    refs: dict[str, str] = {}
    for table in stmt.find_all(exp.Table):
        name = table.name
        refs[table.alias_or_name] = name

    touched = {alias: real for alias, real in refs.items() if real in tenant_tables}
    if not touched:
        return

    insert = stmt.find(exp.Insert)
    if insert is not None:
        # Covers both INSERT ... VALUES and INSERT ... SELECT: the target
        # column list is what matters (the PG dialect renders executemany
        # inserts as INSERT ... SELECT CAST(...), same column list).
        target = insert.this  # a Schema(table, columns) for INSERT INTO t (...)
        table = target.this if isinstance(target, exp.Schema) else target
        real = table.name if isinstance(table, exp.Table) else None
        if real in tenant_tables:
            columns = set()
            if isinstance(target, exp.Schema):
                for expr in target.expressions:
                    name = expr.name if hasattr(expr, "name") else str(expr)
                    columns.add(name)
            missing = {"tenant_id", "app_id"} - columns
            if missing:
                raise AuditViolation(
                    f"INSERT into tenant table {real!r} is missing tenant "
                    f"columns {sorted(missing)}: {stmt.sql()[:200]}"
                )
        return

    # Identity-keyed exemption -- SQLite only (M2 task 3 retired it on
    # PostgreSQL).  ORM unit-of-work statements address rows by primary
    # key; on PostgreSQL every tenant table's primary key now leads with
    # ``(tenant_id, app_id)``, so those statements carry the tenant
    # predicates natively and need no exemption.  SQLite cannot host that
    # constraint on its autoincrement tables (the DDL narrows back to a
    # single column), so the UOW WHERE there is PK-only and the surface is
    # closed by dedicated layers, each with its own test
    # (tests/test_pk_addressing.py): the whitelist write guard intercepts
    # foreign-PK objects at construction, the merge wrapper refuses
    # cross-tenant merges, and the flush checks refuse dirty/deleted rows
    # from another tenant.
    if (
        len(touched) == 1
        and _identity_exemption_dialect()
        and _is_identity_keyed(stmt, touched, tenant_tables, exp)
    ):
        return

    # Collect the tenant columns constrained by bound equalities anywhere
    # in the statement (WHERE, JOIN ON, subquery WHERE, HAVING).
    # Note: there is NO identity-keyed exemption.  Every tenant table's
    # primary key leads with (tenant_id, app_id), so ORM unit-of-work
    # statements carry the tenant predicates natively; a PK-addressed
    # statement without them (raw SQL, merge-by-id of another tenant) is a
    # violation.  UOW statements' tenant safety additionally rests on
    # load-time filtering and the authoring-time write guard.
    covered: dict[str, set[str]] = {}

    def _attribute(table_ref: str) -> str | None:
        real = refs.get(table_ref)
        if real is None:
            # unqualified tenant column: attribute it to every tenant table
            # in the statement (conservative over-approximation; ORM SQL
            # always qualifies).
            return "*"
        return real if real in tenant_tables else None

    for eq in stmt.find_all(exp.EQ):
        left, right = eq.this, eq.expression
        for col, val in ((left, right), (right, left)):
            if isinstance(col, exp.Column) and _is_bound(val):
                target = _attribute(col.table or "")
                if target:
                    covered.setdefault(target, set()).add(col.name)

    for alias, real in touched.items():
        got = covered.get(real, set()) | covered.get("*", set())
        missing = {"tenant_id", "app_id"} - got
        if missing:
            raise AuditViolation(
                f"tenant table {real!r} (alias {alias!r}) is referenced "
                f"without equality predicates on {sorted(missing)} bound to "
                f"parameters: {stmt.sql()[:200]}"
            )


def _is_identity_keyed(stmt, touched: dict[str, str], tenant_tables: set[str], exp) -> bool:
    """True for a unit-of-work statement: one tenant table, WHERE is only
    ``AND``-joined equalities between that table's PK columns and bound
    parameters.

    See the caller for why identity-keyed statements are exempt and which
    layers close the residual surface instead.
    """
    if len(touched) != 1:
        return False
    real = next(iter(touched.values()))
    pk_cols = set(tenant_pk_columns().get(real, ()))
    if not pk_cols:
        return False

    where = stmt.args.get("where")
    if where is None:
        return False
    condition = where.this

    def _pk_chain(node) -> bool:
        if isinstance(node, exp.And):
            return _pk_chain(node.this) and _pk_chain(node.expression)
        if isinstance(node, exp.EQ):
            col, val = node.this, node.expression
            if not (isinstance(col, exp.Column) and _is_bound(val)):
                return False
            if col.name in pk_cols and (not col.table or col.table in touched):
                return True
            return False
        return False

    return _pk_chain(condition)


_pk_columns_cache: dict[str, set[str]] | None = None


def tenant_pk_columns() -> dict[str, set[str]]:
    """Primary-key column names per tenant table."""
    global _pk_columns_cache
    if _pk_columns_cache is None:
        from mirror_memory.core.models import Base

        _pk_columns_cache = {
            name: {col.name for col in table.primary_key.columns}
            for name, table in Base.metadata.tables.items()
        }
    return _pk_columns_cache


_verdict_cache: dict[tuple[str, bool], AuditViolation | None] = {}
_VERDICT_CACHE_CAP = 100_000
_cache_hits = 0
_cache_misses = 0

#: The dialect of the statement currently being audited.  Set by the
#: execution listener right before :func:`audit_sql` and read by the
#: structural judge.  ``audit_sql`` called directly (unit tests) defaults
#: to SQLite rules -- the historical behaviour.
_dialect_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mm_audit_dialect", default="sqlite"
)


def auditor_cache_stats() -> tuple[int, int, int]:
    """(distinct entries, cache hits, misses) for the report."""
    return len(_verdict_cache), _cache_hits, _cache_misses


def _identity_exemption_dialect() -> bool:
    """Whether the PK-addressing exemption applies to the current dialect.

    M2 task 3: PostgreSQL tenant tables carry ``PRIMARY KEY (tenant_id,
    app_id, ...)``, so their unit-of-work statements name the tenant
    columns and the exemption is retired there.  SQLite physically keeps a
    single-column PK (autoincrement limitation) and keeps the exemption,
    backed by the three fallback layers.
    """
    return _dialect_var.get() != "postgresql"


def audit_sql(sql: str) -> None:
    """Audit one statement; raise :class:`AuditViolation` on any rule breach.

    Verdicts are cached by SQL text: the different statement shapes in a
    test run number in the hundreds while executions number in the
    hundreds of thousands, and the verdict depends only on the statement
    text (bound parameters never affect the rules).  First-seen text pays
    one sqlglot parse; repeats are a dict hit.

    Task-1a invariant: the cache holds ONLY the context-independent
    structural verdict (which tenant tables appear, whether each carries
    the tenant predicates, whether the statement is PK-addressed).  All
    context decisions -- system_scope, no_audit_scope, audit enabled --
    are re-evaluated on EVERY execution by the listener, before this
    function is called; a cached "clean" can therefore never turn into an
    out-of-scope pass.  The guard below makes that contract a hard error
    if a future refactor moves a context check in here.
    """
    global _cache_hits, _cache_misses

    if in_system_scope() or _disabled_var.get():
        raise RuntimeError(
            "audit_sql() must never run inside system_scope/no_audit_scope: "
            "context pass/fail decisions are re-made per execution by the "
            "listener and must stay out of the verdict cache"
        )

    if _exempt(sql):
        with _violations_lock:
            _exemptions_seen.append(sql.strip()[:120])
        return

    # The verdict depends on the statement text AND on the dialect (the
    # PK-addressing exemption is PostgreSQL-only), so the cache key carries
    # both.  Cross-dialect text (``UPDATE t SET v=? WHERE tenant_id=? AND
    # app_id=? AND id=?``) therefore never serves a SQLite verdict to a
    # PostgreSQL run or the reverse.
    cache_key = (sql, _identity_exemption_dialect())

    if cache_key in _verdict_cache:
        _cache_hits += 1
        verdict = _verdict_cache[cache_key]
    else:
        _cache_misses += 1
        try:
            verdict = _judge(sql)
        except AuditViolation as exc:
            verdict = exc
        if len(_verdict_cache) >= _VERDICT_CACHE_CAP:
            _verdict_cache.clear()
        _verdict_cache[cache_key] = verdict
    if verdict is not None:
        raise verdict


_PG_PLACEHOLDER_RE = re.compile(
    # PostgreSQL driver placeholders, optionally with an explicit cast:
    #   %(name)s::TIMESTAMP WITHOUT TIME ZONE  ->  ?
    #   %(name)s::VARCHAR(64)                  ->  ?
    # sqlglot cannot parse the psycopg placeholder form at all, and the
    # cast is irrelevant to the tenant rules (it only pins the bind value's
    # type).  The cast tail is a sequence of type words (each optionally
    # with a parenthesised length), stopped by any SQL keyword so it never
    # swallows a following predicate.
    r"%\([A-Za-z0-9_]+\)s"
    r"(?:::"
    r"[A-Za-z0-9_\[\]]+(?:\([0-9, ]+\))?"
    r"(?:[ ](?!AND\b|OR\b|WHERE\b|FROM\b|GROUP\b|ORDER\b|LIMIT\b|RETURNING\b|VALUES\b|SET\b)"
    r"[A-Za-z0-9_\[\]]+(?:\([0-9, ]+\))?)*"
    r")?"
)


_LIKE_ESCAPE_RE = re.compile(
    # PostgreSQL renders the LIKE escape character as a bare inline literal
    # right after the bind placeholder:   claim_text ILIKE ? '\'
    # (there is no bind parameter for ESCAPE and this dialect omits the
    # keyword).  sqlglot rejects that shape; the escape clause is
    # irrelevant to the tenant rules, so it is dropped before parsing.
    # The literal is: apostrophe, backslash, apostrophe.
    r"(?<=\?)[ ]+(?:'\\\\'|'\\'|'')"
)


def _normalise_placeholders(sql: str) -> str:
    """Rewrite driver placeholders / dialect-specific noise to one form."""
    normalised = _PG_PLACEHOLDER_RE.sub("?", sql)
    return _LIKE_ESCAPE_RE.sub("", normalised)


def _judge(sql: str) -> AuditViolation | None:
    import sqlglot

    normalised = _normalise_placeholders(sql)
    # Dialect fallback: SQLite SQL (the default) and PostgreSQL SQL differ
    # in ways sqlglot cares about -- ILIKE, ::casts, RETURNING placement.
    # Parse with the matching dialect and fall back to the other one; the
    # structural rules below are dialect-neutral.
    stmt = None
    last_exc: Exception | None = None
    for dialect in ("sqlite", "postgres"):
        try:
            stmt = sqlglot.parse_one(normalised, read=dialect)
            if stmt is not None:
                break
        except Exception as exc:  # noqa: BLE001 -- try the next dialect
            last_exc = exc
    if stmt is None:
        return AuditViolation(
            f"SQL auditor could not parse statement (fail closed): {sql[:200]!r} ({last_exc})"
        )
    if stmt is None:
        return AuditViolation(f"SQL auditor parsed an empty statement: {sql[:200]!r}")
    try:
        _analyse_statement(stmt, _tenant_table_names())
        return None
    except AuditViolation as exc:
        return exc


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

_installed = False
_disabled_var: contextvars.ContextVar[bool] = contextvars.ContextVar("mm_audit_off", default=False)


@contextlib.contextmanager
def no_audit_scope():
    """Test-only escape hatch for tests that deliberately exercise raw /
    mechanic SQL (mirrors core.tenancy.no_tenant_scope).  Never use in
    engine code."""
    token = _disabled_var.set(True)
    try:
        yield
    finally:
        _disabled_var.reset(token)


def install_sql_auditor() -> None:
    """Install the auditor on the Engine class (idempotent)."""
    global _installed
    if _installed:
        return

    def _audit(conn, cursor, statement, parameters, context, executemany) -> None:
        if not in_audit_mode():
            return
        if _disabled_var.get():
            return
        if _single_tenant_active():
            # Task 6: single-tenant mode skips per-statement injection by
            # design; the engine's startup check refused any database with
            # more than one tenant, and before_flush refuses foreign-tenant
            # writes, so an unpredicated read cannot cross tenants.
            return
        if in_system_scope():
            logger.info(
                "sql-audit: system_scope(reason=%r) statement: %s",
                system_reason(), statement[:160],
            )
            return
        token = _dialect_var.set(conn.dialect.name)
        try:
            audit_sql(statement)
        finally:
            _dialect_var.reset(token)

    event.listen(Engine, "before_cursor_execute", _audit)
    _installed = True


def _single_tenant_active() -> bool:
    from mirror_memory.core.tenancy import _single_tenant_mode

    return _single_tenant_mode.get()


def in_system_scope() -> bool:
    from mirror_memory.core.tenancy import in_system_scope as _iss

    return _iss()


def system_reason() -> str | None:
    from mirror_memory.core.tenancy import system_reason as _sr

    return _sr()


def in_audit_mode() -> bool:
    return os.environ.get("MM_SQL_AUDIT", "1") != "0"
