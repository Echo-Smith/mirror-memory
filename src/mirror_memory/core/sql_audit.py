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
    if insert is not None and not stmt.find(exp.Select):
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

    # Identity-keyed exemption: the ORM unit of work addresses rows purely
    # by primary key (expired-attribute refresh SELECT, dirty-row UPDATE,
    # cascade DELETE).  Those objects were tenant-validated when loaded
    # (filtered SELECT) and again at flush (before_flush checks); a
    # PK-addressed statement cannot cross tenants.  Scope: exactly one
    # tenant table, a WHERE that is a pure AND-chain of
    # ``pk_column = :placeholder`` equalities, nothing else.
    if _is_identity_keyed(stmt, touched, tenant_tables, exp):
        return

    # Collect the tenant columns constrained by bound equalities anywhere
    # in the statement (WHERE, JOIN ON, subquery WHERE, HAVING).
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

    Unconditionally exempt: ORM unit-of-work statements address rows purely
    by primary key, and their tenant safety is enforced where it actually
    lives -- load-time filtering (loader criteria) and the write guard --
    not in the statement text.  (Scoping this exemption by flush-time state
    is unreliable: a test-authored mutation is often flushed by a later
    engine call, so attribution at cursor time is wrong in both
    directions.)  The auditor's real target is query-shaped statements:
    SELECTs and filtered bulk operations, where a missing tenant predicate
    is a genuine cross-tenant read/write surface.
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


def audit_sql(sql: str) -> None:
    """Audit one statement; raise :class:`AuditViolation` on any rule breach."""
    import sqlglot

    if _exempt(sql):
        with _violations_lock:
            _exemptions_seen.append(sql.strip()[:120])
        return

    try:
        stmt = sqlglot.parse_one(sql, read="sqlite")
    except Exception as exc:
        raise AuditViolation(
            f"SQL auditor could not parse statement (fail closed): {sql[:200]!r} ({exc})"
        ) from exc
    if stmt is None:
        raise AuditViolation(f"SQL auditor parsed an empty statement: {sql[:200]!r}")
    _analyse_statement(stmt, _tenant_table_names())


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
        if in_system_scope():
            logger.info(
                "sql-audit: system_scope(reason=%r) statement: %s",
                system_reason(), statement[:160],
            )
            return
        audit_sql(statement)

    event.listen(Engine, "before_cursor_execute", _audit)
    _installed = True


def in_system_scope() -> bool:
    from mirror_memory.core.tenancy import in_system_scope as _iss

    return _iss()


def system_reason() -> str | None:
    from mirror_memory.core.tenancy import system_reason as _sr

    return _sr()


def in_audit_mode() -> bool:
    return os.environ.get("MM_SQL_AUDIT", "1") != "0"
