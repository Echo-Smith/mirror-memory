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
#
# Shape rules (M2 patch, after the pg_catalog incident): a whitelist pattern
# must describe the WHOLE statement.  A prefix anchor alone is a hole --
# ``^\s*begin\b`` also matches ``BEGIN; SELECT * FROM mm_beliefs`` sent as
# one query string (psycopg runs multi-statement queries), and
# ``^\s*create\s+table\b`` also matched ``CREATE TABLE stolen AS SELECT *
# FROM mm_beliefs``.  Every pattern therefore either anchors the end
# (``;?\s*$``) or rejects statement separators, and DDL that can read rows
# (CREATE TABLE ... AS SELECT) is explicitly excluded.
#
# PostgreSQL catalog introspection is NOT in this list any more: it used to
# be three prefix regexes, and ``FROM pg_catalog.`` appearing anywhere in
# the text whitelisted the whole statement -- including
# ``SELECT * FROM mm_beliefs WHERE id IN (SELECT 1 FROM pg_catalog.pg_class)``.
# Catalog statements are now recognised structurally after parsing (see
# ``_is_catalog_only``), and mixed statements go through the normal rules.
# ---------------------------------------------------------------------------

_EXEMPT_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*select\s+1\s*;?\s*$",  # health check: no tables
        r"^\s*begin\s*(work|transaction)?\s*;?\s*$",  # transaction control
        r"^\s*commit\s*(work|transaction)?\s*;?\s*$",
        r'^\s*rollback\s*(work\s*|transaction\s*|to\s+savepoint\s+[a-z0-9_"]+\s*)?;?\s*$',
        r'^\s*savepoint\s+[a-z0-9_"]+\s*;?\s*$',  # publisher savepoints
        r'^\s*release\s*(savepoint\s*)?[a-z0-9_"]+\s*;?\s*$',
        # sqlite introspection: single pragma, single statement
        r"^\s*pragma\s+[a-z0-9_.]+\s*(\([^);]*\)|=\s*[^;]*)?\s*;?\s*$",
        # DDL in create_all / teardown.  CTAS reads rows, so CREATE TABLE
        # ... AS SELECT must NOT match (the negative lookahead rejects it).
        r"^\s*create\s+(temp(?:orary)?\s+)?table\b(?![\s\S]*\bas\s+select\b)[^;]*;?\s*$",
        r"^\s*create\s+(unique\s+)?index\b[^;]*;?\s*$",
        r"^\s*create\s+savepoint\b[^;]*;?\s*$",
        r"^\s*drop\s+table\s+(if\s+exists\s+)?[^;]*;?\s*$",
        # additive migration DDL (check_schema): no row access, one statement
        r"^\s*alter\s+table\s+[^;]*;?\s*$",
        # Row-level-security posture DDL (M2 batch-2 task 1).  None of these
        # can read or write a row: GRANT changes privileges, CREATE/DROP
        # POLICY changes the catalog, CREATE ROLE creates a login.  Each is
        # one statement (no separator) and anchored at both ends.  Note the
        # role model, not the auditor, is what stops the app role from
        # issuing them: they fail with "must be owner" at the database.
        r"^\s*grant\s+[^;]*;?\s*$",
        r"^\s*create\s+policy\s+[^;]*;?\s*$",
        r"^\s*drop\s+policy\s+(if\s+exists\s+)?[^;]*;?\s*$",
        r"^\s*create\s+role\s+[^;]*;?\s*$",
        # The engine's own per-transaction tenant setting (core.rls), exactly
        # as rendered -- both values are bound parameters.  Any OTHER
        # statement touching the mm.* settings is a hard violation below.
        r"^\s*select\s+set_config\('mm\.tenant_id',\s*%\(tenant_id\)s,\s*true\),\s*"
        r"set_config\('mm\.app_id',\s*%\(app_id\)s,\s*true\)\s*;?\s*$",
    )
)

# ---------------------------------------------------------------------------
# Hard violations (M2 batch-2 task 1): statements that would re-point or
# switch off PostgreSQL row-level security from INSIDE an audited
# connection.  Checked after the exact whitelist and before anything else;
# a match anywhere in the text counts (``(^|;)`` catches a smuggled second
# statement), and the verdict does not depend on parsing.
#
# - set_config('mm.*' ...) / SET [LOCAL|SESSION] mm.* : the tenant policy
#   reads exactly these settings; re-pointing them is choosing another
#   tenant's rows.  Only the engine's own statement above may set them.
# - SET/RESET ROLE, SET SESSION AUTHORIZATION: switching to a role the
#   maintenance policy admits.
# - row_security: SET row_security=off makes a non-exempt role's queries
#   error instead of bypassing, but it is never legitimate engine traffic.
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bset_config\s*\(\s*'mm\.",
        r"(?:^|;)\s*set\s+(?:local\s+|session\s+)?mm\.",
        r"(?:^|;)\s*(?:set|reset)\s+(?:local\s+|session\s+)?role\b",
        r"(?:^|;)\s*(?:set|reset)\s+session\s+authorization\b",
        r"(?:^|;)\s*(?:set|reset)\s+(?:local\s+|session\s+)?row_security\b",
        r"\bset_config\s*\(\s*'row_security'",
    )
)


def _forbidden(sql: str) -> str | None:
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(sql):
            return pat.pattern
    return None

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

    # There is NO identity-keyed exemption (M2 task 3 retired it on
    # PostgreSQL; the M2 patch then retired it on SQLite too, after the
    # composite mapping proved every unit-of-work statement there carries
    # the tenant columns as well).  Every tenant table's primary key leads
    # with (tenant_id, app_id), so ORM unit-of-work statements carry the
    # tenant predicates natively; a PK-addressed statement without them
    # (raw SQL, merge-by-id of another tenant) is a violation.  Runtime
    # safety additionally rests on load-time filtering, the authoring-time
    # write guard, the merge wrapper and the flush checks -- all retained.
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


_verdict_cache: dict[str, AuditViolation | None] = {}
_VERDICT_CACHE_CAP = 100_000
_cache_hits = 0
_cache_misses = 0

#: The dialect of the statement currently being audited.  Set by the
#: execution listener right before :func:`audit_sql`.  The judge itself is
#: dialect-independent (the PK-addressing exemption is retired on every
#: dialect); tests use the same var to pin a dialect for statement probes.
_dialect_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mm_audit_dialect", default="sqlite"
)


def auditor_cache_stats() -> tuple[int, int, int]:
    """(distinct entries, cache hits, misses) for the report."""
    return len(_verdict_cache), _cache_hits, _cache_misses


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

    forbidden = _forbidden(sql)
    if forbidden is not None:
        raise AuditViolation(
            "statement would re-point or switch off row-level security "
            f"(matched {forbidden!r}): {sql[:200]!r}"
        )

    # The verdict depends only on the statement text: the judge is
    # dialect-independent (the PK-addressing exemption is retired on every
    # dialect), bound parameters never affect the rules, and no context
    # reaches this function.
    cache_key = sql

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
    #   %(name)s::TEXT[] (inside ARRAY[...]')  ->  ?
    # sqlglot cannot parse the psycopg placeholder form at all, and the
    # cast is irrelevant to the tenant rules (it only pins the bind value's
    # type).  The cast tail is ONE type word (optionally with a
    # parenthesised length and/or an ``[]`` array suffix, as one unit so a
    # word never swallows the ``]`` closing an enclosing ARRAY[...])
    # followed only by the continuation words multi-word PostgreSQL types
    # are built from.  M2 batch-2: the tail used to be "any word not in a
    # keyword stop-list", and ``%(c)s::VARCHAR LEFT OUTER JOIN
    # pg_catalog.pg_class`` (SQLAlchemy's foreign-key reflection) lost its
    # JOIN to the cast -- a parse failure, i.e. a false fail-closed verdict.
    r"%\([A-Za-z0-9_]+\)s"
    r"(?:::"
    r"[A-Za-z0-9_]+(?:\([0-9, ]+\))?(?:\[\])?"
    r"(?:[ ](?:WITHOUT|WITH|TIME|ZONE|PRECISION|VARYING)\b(?:\([0-9, ]+\))?(?:\[\])?)*"
    r")?",
    re.IGNORECASE,
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
        if _is_catalog_only(stmt):
            with _violations_lock:
                _exemptions_seen.append(sql.strip()[:120])
            return None
        _analyse_statement(stmt, _tenant_table_names())
        return None
    except AuditViolation as exc:
        return exc


# ---------------------------------------------------------------------------
# PostgreSQL catalog introspection (structural, M2 patch)
#
# The old text whitelist exempted any statement with ``FROM pg_catalog.``
# anywhere in it, which also whitelisted
# ``SELECT * FROM mm_beliefs WHERE id IN (SELECT 1 FROM pg_catalog.pg_class)``.
# Catalog statements are now recognised from the parse tree instead: every
# relation the statement touches (FROM, JOIN, subquery, UNION branch, CTE
# body) must be a catalog relation.  Anything mixed with a real table falls
# through to the normal tenant rules, and a statement that cannot be parsed
# is still a violation (fail closed).
# ---------------------------------------------------------------------------

#: Schemas whose contents are engine metadata, never tenant data.
_CATALOG_SCHEMAS = ("pg_catalog", "information_schema")


def _is_catalog_relation(table) -> bool:
    schema = (table.db or "").lower()
    name = (table.name or "").lower()
    if schema in _CATALOG_SCHEMAS:
        return True
    # unqualified system relations (search_path-resolved); PostgreSQL
    # reserves the pg_ prefix, so a bare pg_* relation cannot be user data
    return not schema and name.startswith("pg_")


def _is_catalog_only(stmt) -> bool:
    """True when every relation in the statement is a catalog relation.

    A statement with no FROM at all (e.g. ``SELECT pg_catalog.version()``)
    is handled by the caller: it references no table, so the tenant rules
    pass it without a catalog exemption.
    """
    import sqlglot.expressions as exp

    tables = list(stmt.find_all(exp.Table))
    if not tables:
        return False
    return all(_is_catalog_relation(table) for table in tables)


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
