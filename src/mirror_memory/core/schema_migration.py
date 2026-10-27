"""Versioned schema migration (M2 task 2).

Design choice: a self-managed ``mm_schema_version`` table plus an ordered
list of step objects, NOT Alembic.  Reasons:

- This is a library, not an application: it cannot impose an ``alembic/``
  env or its own connection lifecycle on the host.  What M3/M4 actually
  need is incremental, re-entrant steps -- the version table is the whole
  requirement.
- Steps are dialect-aware: PostgreSQL uses ``ALTER TABLE ADD COLUMN`` +
  batched backfill + ``SET NOT NULL`` (no table rewrite, so a big table is
  never exclusively locked for the whole operation); SQLite keeps the
  create-copy-drop-rename rebuild that its ALTER limitations force.

Statement construction: DML is built from SQLAlchemy expression constructs
(``Table.update().where(...).values(...)``, ``select(func.count())``) and
DDL from small ``DDLElement`` subclasses whose compiler quotes identifiers
through ``compiler.preparer.quote()``.  Values are always bound
parameters.  No identifier or value is ever interpolated into SQL text by
hand, and every name originates from ``Base.metadata`` or a step's own
declaration.

Contract per step:

- ``apply(engine)`` must be RE-ENTRANT: re-running after a partial failure
  either continues (the batched backfill resumes at the first unstamped
  row) or is a no-op.
- ``verify(engine)`` must pass before the version row is written; a raise
  leaves the version unadvanced so the next call retries the same step.

Backups: SQLite gets an automatic file backup (restored on failure).
PostgreSQL has no safe file to copy -- :func:`upgrade_database` REFUSES to
run without ``confirm_backup=True`` and prints a ready ``pg_dump`` command.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    func,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateIndex, DDLElement, DropIndex

from mirror_memory.core.models import Base, utcnow
from mirror_memory.core.tenancy import system_scope

logger = logging.getLogger(__name__)

_TEMP_SUFFIX = "__mm_migrate"

_MIGRATION_REASON = (
    "schema migration: whole-table backfill/verification is inherently "
    "cross-tenant"
)

#: Backfill batch size (configurable per the task's "batched backfill").
DEFAULT_BACKFILL_BATCH = 5000

#: How long any single migration DDL waits for its lock before failing
#: (``MM_MIGRATION_LOCK_TIMEOUT``, e.g. "250ms", "3s", "1min").  Failing
#: fast and retrying beats queueing behind a long-running reader: a queued
#: ``ACCESS EXCLUSIVE`` request blocks every later reader on that table.
DEFAULT_LOCK_TIMEOUT = "3s"

_LOCK_TIMEOUT_PATTERN = re.compile(r"^\d+(\.\d+)?(ms|s|min|h)$")

#: How many times a lock-starved step is retried
#: (``MM_MIGRATION_LOCK_RETRIES``).
DEFAULT_LOCK_RETRIES = 3

_RETRY_SLEEP_SECONDS = 2.0

#: Suffix of the throwaway unique index the composite primary key is built
#: from.  ``ADD CONSTRAINT ... PRIMARY KEY USING INDEX`` renames it to the
#: constraint name, so the build-time name never collides with it.
_PKEY_BUILD_SUFFIX = "__pkey_build"


class MigrationError(RuntimeError):
    """A step failed; the version was not advanced."""


class MigrationRefused(MigrationError):
    """Preconditions not met; nothing was touched."""


# ---------------------------------------------------------------------------
# DDL constructs (identifiers quoted by the dialect)
# ---------------------------------------------------------------------------


class AddColumn(DDLElement):
    inherit_cache = True

    def __init__(self, table: str, column: str, coltype: str) -> None:
        self.table = table
        self.column = column
        self.coltype = coltype


@compiles(AddColumn)
def _compile_add_column(element, compiler, **kw):
    preparer = compiler.preparer
    return "ALTER TABLE %s ADD COLUMN %s %s" % (
        preparer.quote(element.table),
        preparer.quote(element.column),
        element.coltype,
    )


class PrimaryKeySwap(DDLElement):
    """Attach a table's build index as its composite primary key (PG).

    Emits ``ALTER TABLE ... DROP CONSTRAINT IF EXISTS <old>, ADD CONSTRAINT
    <table>_pkey PRIMARY KEY USING INDEX <build index>``.  The USING INDEX
    form is the point of the whole design: PostgreSQL adopts the already
    built index instead of scanning the table again, so the swap holds its
    brief lock for a catalog update only -- no scan, no rewrite, and reads
    are never blocked behind an index build.

    Identifiers come from ``Base.metadata`` at compile time; nothing is
    interpolated from the database or from a caller.
    """

    inherit_cache = True

    def __init__(self, table: str) -> None:
        self.table = table


@compiles(PrimaryKeySwap)
def _compile_primary_key_swap(element, compiler, **kw):
    if compiler.dialect.name != "postgresql":
        raise MigrationError(
            "PrimaryKeySwap is a PostgreSQL catalog ALTER; SQLite keeps its "
            "single-column physical key"
        )
    preparer = compiler.preparer
    model_table = Base.metadata.tables[element.table]
    constraint = preparer.quote(model_table.name + "_pkey")
    build = preparer.quote(model_table.name + _PKEY_BUILD_SUFFIX)
    return "ALTER TABLE %s DROP CONSTRAINT IF EXISTS %s, ADD CONSTRAINT %s PRIMARY KEY USING INDEX %s" % (
        preparer.format_table(model_table),
        constraint,
        constraint,
        build,
    )


class SetLockTimeout(DDLElement):
    """Bound the wait for any one migration DDL lock.

    A queued ``ACCESS EXCLUSIVE`` request blocks every later reader on the
    table, so the migration fails fast on ``lock_timeout`` and retries
    instead of queueing indefinitely.  The value is an operator setting
    (``MM_MIGRATION_LOCK_TIMEOUT``), validated against a strict duration
    grammar before it ever reaches statement text.
    """

    inherit_cache = True

    def __init__(self, value: str) -> None:
        if not _LOCK_TIMEOUT_PATTERN.match(value):
            raise MigrationRefused(
                "MM_MIGRATION_LOCK_TIMEOUT must match "
                + _LOCK_TIMEOUT_PATTERN.pattern
                + "; got " + repr(value)
            )
        self.value = value


@compiles(SetLockTimeout)
def _compile_set_lock_timeout(element, compiler, **kw):
    return "SET lock_timeout = '%s'" % (element.value,)


class AddNotNullCheck(DDLElement):
    """Add a NOT VALID ``CHECK (col IS NOT NULL)`` (PostgreSQL).

    NOT VALID means existing rows are not scanned at ADD time -- the
    constraint only gates new writes, and the ADD holds its lock for a
    catalog update only.
    """

    inherit_cache = True

    def __init__(self, table: str, column: str) -> None:
        self.table = table
        self.column = column

    @property
    def constraint_name(self) -> str:
        return "chk_" + self.table + "__mm_notnull_" + self.column


@compiles(AddNotNullCheck)
def _compile_add_not_null_check(element, compiler, **kw):
    preparer = compiler.preparer
    return "ALTER TABLE %s ADD CONSTRAINT %s CHECK (%s IS NOT NULL) NOT VALID" % (
        preparer.quote(element.table),
        preparer.quote(element.constraint_name),
        preparer.quote(element.column),
    )


class ValidateConstraint(DDLElement):
    """Scan once to prove every existing row satisfies the constraint.

    Runs under ``SHARE UPDATE EXCLUSIVE``: reads and writes continue
    throughout.  After it succeeds, ``SET NOT NULL`` skips its own full
    scan on PostgreSQL 12+.
    """

    inherit_cache = True

    def __init__(self, table: str, constraint: str) -> None:
        self.table = table
        self.constraint = constraint


@compiles(ValidateConstraint)
def _compile_validate_constraint(element, compiler, **kw):
    preparer = compiler.preparer
    return "ALTER TABLE %s VALIDATE CONSTRAINT %s" % (
        preparer.quote(element.table),
        preparer.quote(element.constraint),
    )


class DropConstraintByName(DDLElement):
    """``ALTER TABLE ... DROP CONSTRAINT IF EXISTS <name>``."""

    inherit_cache = True

    def __init__(self, table: str, constraint: str) -> None:
        self.table = table
        self.constraint = constraint


@compiles(DropConstraintByName)
def _compile_drop_constraint_by_name(element, compiler, **kw):
    preparer = compiler.preparer
    return "ALTER TABLE %s DROP CONSTRAINT IF EXISTS %s" % (
        preparer.quote(element.table),
        preparer.quote(element.constraint),
    )


class AlterColumnClause(DDLElement):
    """ALTER TABLE t ALTER COLUMN c <clause> (clause is a fixed keyword
    string chosen by the step, never caller input)."""

    inherit_cache = True

    def __init__(self, table: str, column: str, clause: str) -> None:
        if clause not in ("SET NOT NULL", "SET DEFAULT ''"):
            raise MigrationError("unsupported ALTER COLUMN clause: " + repr(clause))
        self.table = table
        self.column = column
        self.clause = clause


@compiles(AlterColumnClause)
def _compile_alter_column(element, compiler, **kw):
    preparer = compiler.preparer
    return "ALTER TABLE %s ALTER COLUMN %s %s" % (
        preparer.quote(element.table),
        preparer.quote(element.column),
        element.clause,
    )


class RenameTable(DDLElement):
    inherit_cache = True

    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new


@compiles(RenameTable)
def _compile_rename_table(element, compiler, **kw):
    preparer = compiler.preparer
    return "ALTER TABLE %s RENAME TO %s" % (
        preparer.quote(element.old),
        preparer.quote(element.new),
    )


class DropTable(DDLElement):
    inherit_cache = True

    def __init__(self, table: str) -> None:
        self.table = table


@compiles(DropTable)
def _compile_drop_table(element, compiler, **kw):
    return "DROP TABLE %s" % (compiler.preparer.quote(element.table),)


# ---------------------------------------------------------------------------
# Version table
# ---------------------------------------------------------------------------


def _version_table(metadata: MetaData) -> Table:
    return Table(
        "mm_schema_version",
        metadata,
        Column("version", Integer, primary_key=True),
        Column("applied_at", DateTime, default=utcnow),
        Column("note", String(200), default=""),
    )


def _ensure_version_table(engine: Engine) -> None:
    _version_table(MetaData()).create(engine, checkfirst=True)


def current_version(engine: Engine) -> int:
    """Highest applied version (0 when the table is absent/empty)."""
    if not inspect(engine).has_table("mm_schema_version"):
        return 0
    table = _version_table(MetaData())
    with engine.connect() as conn:
        return int(conn.execute(select(func.max(table.c.version))).scalar() or 0)


def _record_version(engine: Engine, version: int, note: str) -> None:
    table = _version_table(MetaData())
    with engine.begin() as conn:
        conn.execute(table.insert().values(version=version, note=note[:200]))


# ---------------------------------------------------------------------------
# Dialect helpers
# ---------------------------------------------------------------------------


def _is_postgres(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


def _sqlite_file(engine: Engine) -> Path | None:
    url = str(engine.url)
    if not url.startswith("sqlite:///"):
        return None
    path = url[len("sqlite:///"):]
    if not path or path == ":memory:":
        return None
    return Path(path)


def _backup_sqlite(engine: Engine) -> Path | None:
    path = _sqlite_file(engine)
    if path is None:
        return None
    backup = path.with_name(path.name + ".pre-migration.bak")
    shutil.copy2(path, backup)
    return backup


def _restore_sqlite(engine: Engine, backup: Path | None) -> None:
    path = _sqlite_file(engine)
    if path is None or backup is None:
        return
    engine.dispose()
    shutil.copy2(backup, path)
    engine.dispose()


def _table_columns(engine: Engine, table: str) -> set[str]:
    insp = inspect(engine)
    if not insp.has_table(table):
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def _live_table(engine: Engine, name: str) -> Table | None:
    """Reflect one existing table so DML can be expressed as constructs."""
    if not inspect(engine).has_table(name):
        return None
    return Table(name, MetaData(), autoload_with=engine)


def _count_rows(engine: Engine, name: str) -> int:
    table = _live_table(engine, name)
    if table is None:
        return 0
    with system_scope(reason=_MIGRATION_REASON):
        with engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def _row_counts(engine: Engine, tables: list[str]) -> dict[str, int]:
    counts = {}
    insp = inspect(engine)
    for name in tables:
        if not insp.has_table(name):
            continue
        counts[name] = _count_rows(engine, name)
    return counts


def _backfill_column(
    engine: Engine,
    name: str,
    assignments: dict[str, str],
    null_predicate_names: list[str],
    batch: int,
) -> int:
    """Batched, re-entrant backfill of *assignments* on rows where any of
    *null_predicate_names* is NULL.

    Expressed with SQLAlchemy constructs: the batch subquery selects primary
    keys, the outer statement updates only those rows, every assigned value
    is a bound parameter.
    """
    table = _live_table(engine, name)
    if table is None:
        return 0
    pk = list(table.primary_key.columns)
    if len(pk) != 1:
        raise MigrationError(
            "batched backfill needs a single-column primary key on " + name
        )
    pk_col = pk[0]
    null_filter = None
    for col_name in null_predicate_names:
        condition = table.c[col_name].is_(None)
        null_filter = condition if null_filter is None else (null_filter | condition)

    total = 0
    while True:
        batch_ids = (
            select(pk_col).where(null_filter).limit(batch).scalar_subquery()
        )
        stmt = (
            update(table)
            .where(pk_col.in_(batch_ids))
            .values(assignments)
        )
        with system_scope(reason=_MIGRATION_REASON):
            with engine.begin() as conn:
                result = conn.execute(stmt)
                if result.rowcount == 0:
                    return total
                total += result.rowcount


# ---------------------------------------------------------------------------
# Step 1: tenancy (the M1 migration, now versioned)
# ---------------------------------------------------------------------------


@dataclass
class Step1Tenancy:
    """Base schema -> tenant-scoped schema.

    PostgreSQL: ADD COLUMN (nullable) -> batched backfill -> SET NOT NULL
    + defaults.  No table rewrite.  SQLite: create-copy-drop-rename (its
    ALTER limitations force the rebuild).
    """

    version: int = 1
    name: str = "tenancy (tenant columns, backfill, NOT NULL)"
    default_tenant: tuple[str, str] = ("", "")

    def needed(self, engine: Engine) -> bool:
        for table in Base.metadata.sorted_tables:
            cols = _table_columns(engine, table.name)
            if not cols:
                continue  # absent table: a fresh DB gets create_all
            if "tenant_id" not in cols or "app_id" not in cols:
                return True
        return False

    def apply(self, engine: Engine) -> None:
        if _is_postgres(engine):
            self._apply_postgres(engine)
        else:
            self._apply_sqlite(engine)

    def _apply_postgres(self, engine: Engine) -> None:
        """Tenant columns without a blocking NOT NULL.

        1. add the columns nullable (brief metadata-only locks, bounded by
           ``lock_timeout``);
        2. backfill in batches, one transaction per batch;
        3. prove NOT NULL without blocking anyone: add the CHECK constraint
           ``NOT VALID`` (catalog update -- existing rows are not scanned),
           ``VALIDATE`` it (``SHARE UPDATE EXCLUSIVE``, reads and writes
           continue), then ``SET NOT NULL`` -- PostgreSQL 12+ uses the
           validated CHECK to skip its own full scan -- and finally drop
           the helper CHECK.
        """
        insp = inspect(engine)
        existing = [t.name for t in Base.metadata.sorted_tables if insp.has_table(t.name)]
        if not existing:
            return
        tenant_id, app_id = self.default_tenant

        add = [_lock_timeout_element()]
        for name in existing:
            cols = _table_columns(engine, name)
            if "tenant_id" not in cols:
                add.append(AddColumn(name, "tenant_id", "VARCHAR(64)"))
            if "app_id" not in cols:
                add.append(AddColumn(name, "app_id", "VARCHAR(64)"))
        _DDLPlan(add).run(engine)

        for name in existing:
            _backfill_column(
                engine, name,
                {"tenant_id": tenant_id, "app_id": app_id},
                ["tenant_id", "app_id"],
                DEFAULT_BACKFILL_BATCH,
            )

        notnull = [_lock_timeout_element()]
        for name in existing:
            live_checks = {
                c["name"] for c in inspect(engine).get_check_constraints(name)
            }
            for col in ("tenant_id", "app_id"):
                check = AddNotNullCheck(name, col)
                if check.constraint_name in live_checks:
                    continue  # re-entry: the constraint survived a retry
                notnull.append(check)
                notnull.append(ValidateConstraint(name, check.constraint_name))
                notnull.append(AlterColumnClause(name, col, "SET NOT NULL"))
                notnull.append(AlterColumnClause(name, col, "SET DEFAULT ''"))
                notnull.append(DropConstraintByName(name, check.constraint_name))
        _DDLPlan(notnull).run(engine)

    def _apply_sqlite(self, engine: Engine) -> None:
        tenant_id, app_id = self.default_tenant
        fresh_meta = MetaData()
        insp = inspect(engine)
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            cols = _table_columns(engine, table.name)
            if "tenant_id" in cols and "app_id" in cols:
                continue
            new = table.to_metadata(fresh_meta, name=table.name + _TEMP_SUFFIX)
            # Copy only columns that exist on BOTH sides: a base-schema row
            # may predate columns added by later steps (lease_token etc.),
            # and the new table's defaults fill those in.
            live_cols = _table_columns(engine, table.name)
            copied = [
                c for c in table.columns
                if c.name not in ("tenant_id", "app_id") and c.name in live_cols
            ]
            insert = new.insert().from_select(
                ["tenant_id", "app_id"] + [c.name for c in copied],
                select(
                    *[
                        _literal(tenant_id).label("tenant_id"),
                        _literal(app_id).label("app_id"),
                        *copied,
                    ]
                ).select_from(table),
            )
            with engine.begin() as conn:
                new.create(conn)
                conn.execute(insert)
                conn.execute(DropTable(table.name))
                conn.execute(RenameTable(new.name, table.name))

    def verify(self, engine: Engine) -> None:
        _verify_tenancy(engine)


def _literal(value: str):
    """A bound-parameter literal for a SELECT list."""
    from sqlalchemy import literal

    return literal(value)


# ---------------------------------------------------------------------------
# Generic incremental step (the M3/M4 shape; also the tests' "fake step 2")
# ---------------------------------------------------------------------------


@dataclass
class StepAddColumnBackfill:
    """Add a column, then backfill it in batches.  Re-entrant."""

    table: str
    column: str
    coltype: str = "VARCHAR(64)"
    value: str = ""
    version: int = 2
    name: str = "add column + backfill"

    def needed(self, engine: Engine) -> bool:
        return self.column not in _table_columns(engine, self.table)

    def apply(self, engine: Engine, batch: int = DEFAULT_BACKFILL_BATCH) -> None:
        if self.needed(engine):
            with engine.begin() as conn:
                conn.execute(AddColumn(self.table, self.column, self.coltype))
        _backfill_column(
            engine, self.table, {self.column: self.value}, [self.column], batch
        )

    def verify(self, engine: Engine) -> None:
        _verify_column(engine, self.table, self.column)


# ---------------------------------------------------------------------------
# Shared verification helpers (module level: the system_scope allowlist
# scanner records call sites by function, and class methods are not sites)
# ---------------------------------------------------------------------------


def _verify_tenancy(engine: Engine) -> None:
    for table in Base.metadata.sorted_tables:
        cols = _table_columns(engine, table.name)
        if not cols:
            continue
        missing = {"tenant_id", "app_id"} - cols
        if missing:
            raise MigrationError(
                "verification failed: " + table.name + " missing " + str(sorted(missing))
            )
        live = _live_table(engine, table.name)
        with system_scope(reason=_MIGRATION_REASON):
            with engine.connect() as conn:
                nulls = conn.execute(
                    select(func.count())
                    .select_from(live)
                    .where(live.c.tenant_id.is_(None) | live.c.app_id.is_(None))
                ).scalar_one()
        if nulls:
            raise MigrationError(
                "verification failed: " + table.name + " has "
                + str(nulls) + " rows without tenant columns"
            )


def _verify_column(engine: Engine, table_name: str, column: str) -> None:
    if column not in _table_columns(engine, table_name):
        raise MigrationError(
            "verification failed: " + table_name + "." + column + " missing"
        )
    live = _live_table(engine, table_name)
    with system_scope(reason=_MIGRATION_REASON):
        with engine.connect() as conn:
            nulls = conn.execute(
                select(func.count())
                .select_from(live)
                .where(live.c[column].is_(None))
            ).scalar_one()
    if nulls:
        raise MigrationError(
            "verification failed: " + str(nulls) + " rows with NULL " + column
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 2: tenant composite primary keys (M2 task 3)
# ---------------------------------------------------------------------------


@dataclass
class Step2TenantPrimaryKeys:
    """Install ``PRIMARY KEY (tenant_id, app_id, ...)`` on tenant tables.

    A separate step from the tenancy conversion, not part of it: a database
    already migrated to version 1 by an earlier release has the tenant
    columns but a single-column key, and it must be able to advance without
    re-running the step that already succeeded.

    PostgreSQL: each table's PK constraint is dropped and re-added with the
    tenant columns leading.  That is a catalog change -- the replacement
    index is built while the old one still serves reads -- so the table is
    never rewritten and never held under an exclusive lock for the whole
    operation.  Re-entrant: tables whose key already matches are skipped,
    so a re-run after a partial failure continues where it stopped.

    SQLite: nothing is altered, by design.  A composite key over an
    autoincrement column is impossible there (``CompileError``), so the
    physical key stays single-column and write safety comes from the three
    fallback layers (see ``tests/test_pk_addressing.py``).  The step still
    records its version, so both databases advance through the same
    sequence.
    """

    version: int = 2
    name: str = "tenant composite primary keys"

    def needed(self, engine: Engine) -> bool:
        if not _is_postgres(engine):
            return False  # SQLite's physical key is intentionally single-column
        insp = inspect(engine)
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            wanted = [column.name for column in table.primary_key.columns]
            if list(insp.get_pk_constraint(table.name)["constrained_columns"]) != wanted:
                return True
        return False

    def apply(self, engine: Engine) -> None:
        if not _is_postgres(engine):
            return
        existing = [
            t.name for t in Base.metadata.sorted_tables if inspect(engine).has_table(t.name)
        ]
        _replace_primary_keys(engine, existing)

    def verify(self, engine: Engine) -> None:
        _verify_tenant_primary_keys(engine)


def _literal(value: str):
    """A bound-parameter literal for a SELECT list."""
    from sqlalchemy import literal

    return literal(value)


def _replace_primary_keys(engine: Engine, names: list) -> None:
    """Convert the named tables to the tenant composite primary key (PG).

    Tables whose live key already matches are skipped, which is what makes
    a re-run after a partial failure continue rather than fail.  Each
    conversion is retried on lock timeout (``MM_MIGRATION_LOCK_RETRIES``)
    -- see :func:`_swap_primary_key` for the per-table recipe.
    """
    if not _is_postgres(engine):
        return
    insp = inspect(engine)
    for name in names:
        wanted = [column.name for column in Base.metadata.tables[name].primary_key.columns]
        if list(insp.get_pk_constraint(name)["constrained_columns"]) == wanted:
            continue
        _swap_primary_key(engine, name)


def _emit_primary_key_swaps(engine: Engine, names: list) -> None:
    """Run the conversion inside the migration's cross-tenant scope.

    The DDL names no rows -- it is a catalog operation over a whole table --
    so there is no tenant to scope it to; the explicit scope plus its
    allowlist entry is how that is stated rather than implied.
    """
    elements = [_lock_timeout_element()]
    for name in names:
        build = _pkey_build_index(name)
        elements.append(_drop_build_index(build))
        elements.append(_create_build_index(build))
        elements.append(PrimaryKeySwap(name))
    with system_scope(reason=_MIGRATION_REASON):
        _DDLPlan(elements, autocommit=True).run(engine)


def _pkey_build_index(name: str) -> Index:
    """The throwaway unique index the composite PK is adopted from.

    ``postgresql_concurrently`` makes both DDL forms render ``CONCURRENTLY``
    (``CreateIndex``/``DropIndex`` read the index's dialect options); the
    build name differs from the final constraint name because ``PRIMARY KEY
    USING INDEX`` renames the adopted index.
    """
    # Built on a detached copy of the table: an Index over the LIVE table's
    # columns attaches itself to Base.metadata, and every later create_all
    # in the process would then emit "CREATE UNIQUE INDEX CONCURRENTLY ...
    # __pkey_build" inside its transaction and fail (M2 batch-2 finding).
    table = Base.metadata.tables[name].to_metadata(MetaData())
    return Index(
        name + _PKEY_BUILD_SUFFIX,
        *[table.c[column.name] for column in table.primary_key.columns],
        unique=True,
        postgresql_concurrently=True,
    )


def _drop_build_index(build: Index) -> DropIndex:
    """Drop a leftover build index -- a failed CONCURRENTLY attempt leaves
    an INVALID index behind, and a fresh build must not adopt or keep it."""
    return DropIndex(build, if_exists=True)


def _create_build_index(build: Index) -> CreateIndex:
    """Build the unique index without blocking reads or writes."""
    return CreateIndex(build)


def _lock_timeout_element() -> SetLockTimeout:
    return SetLockTimeout(_migration_lock_timeout())


def _migration_lock_timeout() -> str:
    return os.environ.get("MM_MIGRATION_LOCK_TIMEOUT", DEFAULT_LOCK_TIMEOUT)


def _migration_lock_retries() -> int:
    raw = os.environ.get("MM_MIGRATION_LOCK_RETRIES", str(DEFAULT_LOCK_RETRIES))
    try:
        retries = int(raw)
    except ValueError as exc:
        raise MigrationRefused(
            "MM_MIGRATION_LOCK_RETRIES must be an integer; got " + repr(raw)
        ) from exc
    if retries < 1:
        raise MigrationRefused("MM_MIGRATION_LOCK_RETRIES must be >= 1")
    return retries


def _is_lock_not_available(exc: OperationalError) -> bool:
    """True for a lock_timeout failure (SQLSTATE 55P03) -- the shape the
    retry loop exists for.  Anything else is a real error."""
    text = str(exc)
    return "55P03" in text or "lock timeout" in text.lower()


def _swap_primary_key(engine: Engine, name: str) -> None:
    """Convert one table's primary key without a blocking build.

    1. drop any leftover build index -- a failed CONCURRENTLY attempt
       leaves an INVALID index behind, and ``IF NOT EXISTS`` would silently
       keep it;
    2. ``CREATE UNIQUE INDEX CONCURRENTLY`` (outside any transaction) --
       reads and writes continue throughout; the build waits for open
       snapshots instead of blocking them;
    3. ``ALTER TABLE ... ADD CONSTRAINT ... PRIMARY KEY USING INDEX``
       adopts the built index: the catalog update is the only lock held,
       so no scan of the table ever runs under the exclusive lock.

    Retried on lock timeout; every DDL waits at most
    ``MM_MIGRATION_LOCK_TIMEOUT`` before failing instead of queueing.
    """
    retries = _migration_lock_retries()
    for attempt in range(1, retries + 1):
        try:
            _emit_primary_key_swaps(engine, [name])
            return
        except OperationalError as exc:
            if not _is_lock_not_available(exc) or attempt == retries:
                raise
            logger.warning(
                "migration: lock timeout on primary key swap for %r "
                "(attempt %d/%d); retrying",
                name, attempt, retries,
            )
            time.sleep(_RETRY_SLEEP_SECONDS)


class _DDLPlan:
    """A batch of DDL elements emitted through SQLAlchemy's own pipeline.

    The elements ride as ``after_create`` hooks on a throwaway
    ``MetaData``, in registration order, so they execute on one connection
    under one isolation level without this module assembling statements
    itself.  ``autocommit`` selects a connection where every statement
    commits immediately -- required by ``CREATE INDEX CONCURRENTLY``,
    which cannot run inside a transaction block.  ``checkfirst`` keeps
    ``create_all`` from touching the real tables: only the hooks fire.
    """

    def __init__(self, elements: list, autocommit: bool = False) -> None:
        self.elements = elements
        self.autocommit = autocommit

    def run(self, engine: Engine) -> None:
        from sqlalchemy import event

        metadata = MetaData()
        bind = engine
        conn = None
        if self.autocommit:
            conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
            bind = conn
        try:
            for element in self.elements:
                event.listen(metadata, "after_create", element)
            metadata.create_all(bind=bind, checkfirst=True)
        finally:
            if conn is not None:
                conn.close()


def _verify_tenant_primary_keys(engine: Engine) -> None:
    """The composite key must be in place wherever the dialect can host it."""
    if not _is_postgres(engine):
        return
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        wanted = [column.name for column in table.primary_key.columns]
        current = list(insp.get_pk_constraint(table.name)["constrained_columns"])
        if current != wanted:
            raise MigrationError(
                "verification failed: primary key columns differ from the model"
            )


# ---------------------------------------------------------------------------
# Step 3: row-level security (M2 batch-2 task 1)
# ---------------------------------------------------------------------------


@dataclass
class Step3RowLevelSecurity:
    """Install the PostgreSQL second defence: roles, ownership, grants,
    ``ENABLE`` + ``FORCE ROW LEVEL SECURITY`` and the two policies on every
    tenant table (see ``core.rls`` for the role model and why the
    maintenance role is a policy rather than BYPASSRLS).

    Re-entrant by construction: roles are created only when missing,
    ownership transfers and GRANTs are idempotent, ENABLE/FORCE are
    idempotent, and each policy is dropped-if-exists and recreated.  The
    connected role must own the tables (or be superuser) and must hold
    CREATEROLE if the roles do not exist yet; a privilege failure on role
    creation aborts BEFORE any table is touched.

    SQLite has no row-level security: nothing is altered and the version
    still advances so both databases share one version sequence.
    """

    version: int = 3
    name: str = "row-level security"

    def needed(self, engine: Engine) -> bool:
        if not _is_postgres(engine):
            return False
        from mirror_memory.core import rls

        if rls.missing_roles(engine):
            return True
        try:
            rls.verify_posture(engine)
        except rls.RowSecurityError:
            return True
        return False

    def apply(self, engine: Engine) -> None:
        if not _is_postgres(engine):
            return
        install_row_level_security(engine, verify=False)

    def verify(self, engine: Engine) -> None:
        if not _is_postgres(engine):
            return
        from mirror_memory.core import rls

        rls.verify_posture(engine)


def install_row_level_security(engine: Engine, *, verify: bool = True) -> None:
    """Bring a PostgreSQL database to the RLS posture (idempotent).

    Roles first, so an insufficient-privilege failure leaves the tables
    untouched; then the table plan in one transaction.  Hosts call this
    through :func:`upgrade_database` (step 3); the test harness calls it
    directly on each freshly created schema.
    """
    from sqlalchemy.exc import ProgrammingError

    from mirror_memory.core import rls

    if not _is_postgres(engine):
        raise MigrationRefused("row-level security is PostgreSQL only")
    missing = rls.missing_roles(engine)
    if missing:
        try:
            _DDLPlan([rls.CreateRole(name, login) for name, login in missing]).run(engine)
        except ProgrammingError as exc:
            wanted = ", ".join(
                "CREATE ROLE %s %s" % (name, "LOGIN" if login else "NOLOGIN")
                for name, login in missing
            )
            raise MigrationRefused(
                "the migration login may not create roles; create them first "
                "(" + wanted + ") and set their credentials, then re-run"
            ) from exc
    _DDLPlan(rls.rls_plan(engine)).run(engine)
    if verify:
        rls.verify_posture(engine)


# ---------------------------------------------------------------------------
# Step 4: partial index for the maintenance queue scan (M2 batch-2 task 4)
# ---------------------------------------------------------------------------

#: Declared on the model (fresh databases get it from create_all); this
#: step adds it to databases created before it existed.
PENDING_JOBS_INDEX = "ix_mm_evolution_jobs_pending"

_INDEX_VALIDITY = text(
    "SELECT i.indisvalid FROM pg_catalog.pg_index AS i "
    "JOIN pg_catalog.pg_class AS c ON c.oid = i.indexrelid "
    "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
    "WHERE c.relname = :name AND n.nspname = current_schema()"
)


def _index_state(engine: Engine, table: str, name: str) -> str | None:
    """None (absent), "valid" or "invalid" (a failed CONCURRENTLY build)."""
    if _is_postgres(engine):
        with engine.connect() as conn:
            row = conn.execute(_INDEX_VALIDITY, {"name": name}).first()
        if row is None:
            return None
        return "valid" if row[0] else "invalid"
    insp = inspect(engine)
    if not insp.has_table(table):
        return None
    return "valid" if any(ix.get("name") == name for ix in insp.get_indexes(table)) else None


def _pending_jobs_index(concurrently: bool) -> Index:
    """The model's partial index, rebuilt on a DETACHED table copy (an Index
    on the live table would join Base.metadata) with the CONCURRENTLY flag
    the migration needs and ``create_all`` must not use."""
    table = Base.metadata.tables["mm_evolution_jobs"].to_metadata(MetaData())
    declared = next(ix for ix in table.indexes if ix.name == PENDING_JOBS_INDEX)
    table.indexes.discard(declared)
    return Index(
        PENDING_JOBS_INDEX,
        table.c.created_at,
        postgresql_where=declared.dialect_options["postgresql"]["where"],
        sqlite_where=declared.dialect_options["sqlite"]["where"],
        postgresql_concurrently=concurrently,
    )


@dataclass
class Step4PendingJobsIndex:
    """Add ``ix_mm_evolution_jobs_pending`` to databases that predate it.

    Why: the EXPLAIN review (100 tenants x 10k beliefs, 10k jobs) showed
    every other key query already served by the composite primary keys and
    unique constraints, and exactly one sequential scan -- the maintenance
    role's ``status = 'pending' ORDER BY created_at LIMIT n`` queue scan,
    which has no tenant to index by.  A partial index over the pending rows
    is tiny and turns it into an index scan with nothing filtered.

    PostgreSQL: ``CREATE INDEX CONCURRENTLY`` outside any transaction, so
    the queue stays readable and writable while it builds; a failed build
    leaves an INVALID index behind, so the step drops any leftover first
    (re-entrant).  SQLite: a plain ``CREATE INDEX`` (``checkfirst``).
    """

    version: int = 4
    name: str = "pending-jobs partial index"

    def needed(self, engine: Engine) -> bool:
        return _index_state(engine, "mm_evolution_jobs", PENDING_JOBS_INDEX) != "valid"

    def apply(self, engine: Engine) -> None:
        if _is_postgres(engine):
            build = _pending_jobs_index(concurrently=True)
            _DDLPlan(
                [_lock_timeout_element(), DropIndex(build, if_exists=True), CreateIndex(build)],
                autocommit=True,
            ).run(engine)
            return
        _pending_jobs_index(concurrently=False).create(engine, checkfirst=True)

    def verify(self, engine: Engine) -> None:
        if _index_state(engine, "mm_evolution_jobs", PENDING_JOBS_INDEX) != "valid":
            raise MigrationError(
                "verification failed: " + PENDING_JOBS_INDEX + " is missing or invalid"
            )


# ---------------------------------------------------------------------------
# Step 5: (tenant_id, app_id, user_id) replaces the single-column user_id
# indexes (M2 batch-2 patch task 3)
# ---------------------------------------------------------------------------


def _tenant_user_index_pair(table_name: str, concurrently: bool):
    """(new composite index, legacy single-column index) for *table_name*,
    both on a DETACHED copy of the table so neither touches Base.metadata
    (the model already carries the composite one)."""
    from mirror_memory.core.tenancy import legacy_user_index_name, tenant_user_index_name

    table = Base.metadata.tables[table_name].to_metadata(MetaData())
    for index in list(table.indexes):
        if index.name in (tenant_user_index_name(table_name), legacy_user_index_name(table_name)):
            table.indexes.discard(index)
    new = Index(
        tenant_user_index_name(table_name),
        table.c.tenant_id, table.c.app_id, table.c.user_id,
        postgresql_concurrently=concurrently,
    )
    old = Index(legacy_user_index_name(table_name), table.c.user_id, postgresql_concurrently=concurrently)
    return new, old


@dataclass
class Step5TenantUserIndexes:
    """Replace ``ix_<table>_user_id`` with ``ix_<table>_tenant_user`` on
    ``(tenant_id, app_id, user_id)`` for every tenant table with a
    ``user_id`` column -- a replacement, not an addition: the index count
    per table does not grow.

    Why: user ids repeat across tenants, so the bare ``user_id`` index
    matches every tenant's rows for an id and the planner intersects it
    with the primary key (EXPLAIN review, M2 batch-2 task 4: the delete-by-
    user path was a BitmapAnd over 10k candidates at 100 tenants).  Every
    by-user statement carries the tenant predicates, so the prefixed index
    serves it exactly.  No ``system_scope`` (cross-tenant) query filters by
    ``user_id`` -- the queue scan is by status / created_at -- so nothing
    loses an index it relied on.

    PostgreSQL per table: ``CREATE INDEX CONCURRENTLY`` the new index (an
    INVALID leftover of a failed build is dropped first), then ``DROP INDEX
    CONCURRENTLY`` the old one; both outside transactions, both
    re-entrant.  SQLite: plain create / drop with ``checkfirst``.
    """

    version: int = 5
    name: str = "tenant-prefixed user_id indexes"

    def tables(self) -> list[str]:
        from mirror_memory.core.models import TENANT_USER_INDEXED_TABLES

        return list(TENANT_USER_INDEXED_TABLES)

    def _states(self, engine: Engine, table_name: str):
        from mirror_memory.core.tenancy import legacy_user_index_name, tenant_user_index_name

        return (
            _index_state(engine, table_name, tenant_user_index_name(table_name)),
            _index_state(engine, table_name, legacy_user_index_name(table_name)),
        )

    def needed(self, engine: Engine) -> bool:
        insp = inspect(engine)
        for name in self.tables():
            if not insp.has_table(name):
                continue
            new_state, old_state = self._states(engine, name)
            if new_state != "valid" or old_state is not None:
                return True
        return False

    def apply(self, engine: Engine) -> None:
        insp = inspect(engine)
        pg = _is_postgres(engine)
        for name in self.tables():
            if not insp.has_table(name):
                continue
            new_state, old_state = self._states(engine, name)
            new, old = _tenant_user_index_pair(name, concurrently=pg)
            if pg:
                elements: list = [_lock_timeout_element()]
                if new_state == "invalid":
                    elements.append(DropIndex(new, if_exists=True))
                if new_state != "valid":
                    elements.append(CreateIndex(new))
                if old_state is not None:
                    elements.append(DropIndex(old, if_exists=True))
                if len(elements) > 1:
                    _DDLPlan(elements, autocommit=True).run(engine)
                continue
            if new_state != "valid":
                new.create(engine, checkfirst=True)
            if old_state is not None:
                old.drop(engine, checkfirst=True)

    def verify(self, engine: Engine) -> None:
        insp = inspect(engine)
        for name in self.tables():
            if not insp.has_table(name):
                continue
            new_state, old_state = self._states(engine, name)
            if new_state != "valid":
                raise MigrationError(
                    "verification failed: " + name + " lacks a valid (tenant_id, app_id, user_id) index"
                )
            if old_state is not None:
                raise MigrationError(
                    "verification failed: " + name + " still has its single-column user_id index"
                )


# ---------------------------------------------------------------------------
# Step 6: purpose authorization tables (M3 batch-1 task 1)
# ---------------------------------------------------------------------------

PURPOSE_TABLES = ("mm_purpose_grants", "mm_purpose_grant_events")


@dataclass
class Step6PurposeAuthorization:
    """Create the purpose-authorization tables and add ``mm_evolution_jobs
    .purpose``.

    Fresh databases get everything from ``create_all``; this step brings
    databases already at version 5 up.  Re-entrant: tables are created
    with ``checkfirst``, the column addition resumes its backfill, and the
    RLS plan is idempotent.

    - The two tables are created FROM THE MODEL METADATA, so every dialect
      gets the same shape ``create_all`` would have made (the composite
      tenant primary key included -- both their keys are all-string and
      need no SQLite narrowing).
    - ``mm_evolution_jobs.purpose``: ``ADD COLUMN`` (nullable) -> batched
      backfill to ``''`` -> non-blocking ``SET NOT NULL`` (the same
      ``NOT VALID`` CHECK / ``VALIDATE`` / ``SET NOT NULL`` recipe as
      step 1).  The empty purpose marks pre-M3 jobs; the worker refuses
      them with ``missing_purpose`` instead of running them ungated.
    - PostgreSQL: the RLS posture is applied to the two new tables (the
      databases this step runs on already have step 3's roles and the
      tenant tables protected; a fresh database gets everything at once
      from ``install_row_level_security``).
    """

    version: int = 6
    name: str = "purpose authorization tables"

    def needed(self, engine: Engine) -> bool:
        insp = inspect(engine)
        if any(not insp.has_table(name) for name in PURPOSE_TABLES):
            return True
        return "purpose" not in _table_columns(engine, "mm_evolution_jobs")

    def apply(self, engine: Engine) -> None:
        for name in PURPOSE_TABLES:
            Base.metadata.tables[name].create(engine, checkfirst=True)
        if "purpose" not in _table_columns(engine, "mm_evolution_jobs"):
            if _is_postgres(engine):
                _DDLPlan([AddColumn("mm_evolution_jobs", "purpose", "VARCHAR(64)")]).run(engine)
                _backfill_column(
                    engine, "mm_evolution_jobs", {"purpose": ""}, ["purpose"],
                    DEFAULT_BACKFILL_BATCH,
                )
                notnull = [_lock_timeout_element()]
                check = AddNotNullCheck("mm_evolution_jobs", "purpose")
                notnull.append(check)
                notnull.append(ValidateConstraint("mm_evolution_jobs", check.constraint_name))
                notnull.append(AlterColumnClause("mm_evolution_jobs", "purpose", "SET NOT NULL"))
                notnull.append(AlterColumnClause("mm_evolution_jobs", "purpose", "SET DEFAULT ''"))
                notnull.append(DropConstraintByName("mm_evolution_jobs", check.constraint_name))
                _DDLPlan(notnull).run(engine)
            else:
                # SQLite allows ADD COLUMN with a NOT NULL constant default
                _DDLPlan([AddColumn(
                    "mm_evolution_jobs", "purpose", "VARCHAR(64) NOT NULL DEFAULT ''",
                )]).run(engine)
        if _is_postgres(engine):
            from mirror_memory.core import rls

            _DDLPlan(rls.rls_plan(engine, tables=list(PURPOSE_TABLES))).run(engine)

    def verify(self, engine: Engine) -> None:
        if "purpose" not in _table_columns(engine, "mm_evolution_jobs"):
            raise MigrationError("verification failed: mm_evolution_jobs.purpose missing")
        if _is_postgres(engine):
            from mirror_memory.core import rls

            rls.verify_posture(engine, tables=list(PURPOSE_TABLES))


#: Ordered migration steps.  A database advances by running every step
#: whose version exceeds its recorded version, so an older database (say,
#: one already at version 1) picks up only what it is missing.
DEFAULT_STEPS: list = [
    Step1Tenancy(),
    Step2TenantPrimaryKeys(),
    Step3RowLevelSecurity(),
    Step4PendingJobsIndex(),
    Step5TenantUserIndexes(),
    Step6PurposeAuthorization(),
]

#: The version a freshly upgraded database records.
LATEST_VERSION: int = max(step.version for step in DEFAULT_STEPS)


def upgrade_database(
    engine: Engine,
    *,
    default_tenant: str,
    default_app: str,
    steps: list | None = None,
    confirm_backup: bool = False,
    batch: int = DEFAULT_BACKFILL_BATCH,
) -> dict[str, int]:
    """Advance the schema to the latest version; returns pre-run row counts.

    SQLite: automatic file backup, restored on failure.
    PostgreSQL: requires ``confirm_backup=True`` (no safe file to copy).
    """
    if not default_tenant or not default_app:
        raise MigrationRefused(
            "upgrade_database requires an explicit default tenant "
            "(default_tenant=..., default_app=...); existing rows are "
            "assigned to it, so it must be a deliberate configuration value"
        )
    if _is_postgres(engine) and not confirm_backup:
        raise MigrationRefused(
            "PostgreSQL migration requires confirm_backup=True.  Take a backup "
            "first, e.g.:  pg_dump --no-owner --format=custom "
            "--file=pre-migration.dump <database-url>"
        )

    plan = steps if steps is not None else DEFAULT_STEPS
    backup = None if _is_postgres(engine) else _backup_sqlite(engine)
    with _unbounded_statement_timeout(engine):
        _ensure_version_table(engine)
        start_version = current_version(engine)

        tables = [t.name for t in Base.metadata.sorted_tables]
        before = _row_counts(engine, tables)

        try:
            for step in plan:
                if step.version <= start_version:
                    continue
                if hasattr(step, "default_tenant"):
                    step.default_tenant = (default_tenant, default_app)
                if step.needed(engine):
                    logger.info("migration: applying step %d (%s)", step.version, step.name)
                    if isinstance(step, StepAddColumnBackfill):
                        step.apply(engine, batch=batch)
                    else:
                        step.apply(engine)
                step.verify(engine)
                _record_version(engine, step.version, step.name)
        except Exception as exc:
            _restore_sqlite(engine, backup)
            if isinstance(exc, MigrationError):
                raise
            raise MigrationError(
                "migration step failed (" + type(exc).__name__ + ": " + str(exc) + ")"
            ) from exc

    return before


@contextlib.contextmanager
def _unbounded_statement_timeout(engine: Engine):
    """PostgreSQL: the migration's own connections run without the
    application's ``statement_timeout`` (M2 batch-2 patch task 2).

    The owner-side engine is normally built by ``pg_engine.build_engine``,
    which pins ``statement_timeout`` for request traffic (30 s default);
    ``VALIDATE CONSTRAINT``, backfills and ``CREATE INDEX CONCURRENTLY`` on
    a large table legitimately run longer.  While the migration is active
    every connection it checks out gets ``SET statement_timeout = 0`` --
    session level, because CONCURRENTLY runs outside any transaction, so a
    ``SET LOCAL`` could not cover it -- and ``RESET statement_timeout`` on
    the way back into the pool restores the session default, i.e. the value
    the connect-time libpq options established.  Both run on the fresh /
    returned DBAPI connection and are committed at once, so the connection
    is idle (not in a transaction) when SQLAlchemy takes it over or an
    autocommit isolation level is applied.  ``lock_timeout`` is deliberately
    NOT touched: the migration sets its own (``SetLockTimeout``) and must
    keep failing fast rather than queueing behind readers.

    Backfill batches never idle inside a transaction: each batch is one
    UPDATE in its own ``engine.begin()`` block, and the Python loop runs
    BETWEEN transactions, so ``idle_in_transaction_session_timeout`` cannot
    fire during a backfill.
    """
    if not _is_postgres(engine):
        yield
        return
    from sqlalchemy import event

    def _checkout(dbapi_connection, connection_record, connection_proxy):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SET statement_timeout = 0")
        finally:
            cursor.close()
        dbapi_connection.commit()

    def _checkin(dbapi_connection, connection_record):
        if dbapi_connection is None:
            return  # the pool discarded a broken connection
        try:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("RESET statement_timeout")
            finally:
                cursor.close()
            dbapi_connection.commit()
        except Exception:  # noqa: BLE001 -- a dead connection is the pool's to drop
            logger.debug("migration: could not reset statement_timeout on checkin", exc_info=True)

    event.listen(engine, "checkout", _checkout)
    event.listen(engine, "checkin", _checkin)
    try:
        yield
    finally:
        event.remove(engine, "checkout", _checkout)
        event.remove(engine, "checkin", _checkin)


def downgrade_database(engine: Engine) -> bool:
    """SQLite only: restore the pre-migration backup.

    Documented loss: every write made after the upgrade.  PostgreSQL has no
    file to restore -- use the caller's pg_dump/pg_restore.
    """
    if _is_postgres(engine):
        raise MigrationRefused(
            "downgrade requires the caller's PostgreSQL backup (pg_restore); "
            "the library never rewrites a live database"
        )
    path = _sqlite_file(engine)
    if path is None:
        raise MigrationRefused("downgrade requires a SQLite file database")
    pre = path.with_name(path.name + ".pre-migration.bak")
    if not pre.exists():
        raise MigrationRefused("no pre-migration backup found at " + str(pre))
    _restore_sqlite(engine, pre)
    logger.warning("migration: downgraded by restoring %s", pre)
    return True


def verify_schema_parity(engine: Engine) -> list[str]:
    """Reflect the database and compare against Base.metadata."""
    insp = inspect(engine)
    diffs: list[str] = []
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            diffs.append("missing table " + table.name)
            continue
        db_cols = {c["name"]: c for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in db_cols:
                diffs.append(table.name + "." + col.name + " missing in database")
                continue
            db_col = db_cols[col.name]
            if db_col.get("nullable", True) != col.nullable:
                diffs.append(
                    table.name + "." + col.name + " nullable "
                    + str(db_col.get("nullable")) + " != " + str(col.nullable)
                )
        extra = set(db_cols) - {c.name for c in table.columns}
        if extra:
            diffs.append(table.name + " extra columns " + str(sorted(extra)))
        db_uniques = {
            uq.get("name") for uq in insp.get_unique_constraints(table.name)
        } | {
            ix.get("name") for ix in insp.get_indexes(table.name) if ix.get("unique")
        }
        for constraint in table.constraints:
            name = getattr(constraint, "name", None)
            if name is None:
                continue
            if type(constraint).__name__ == "UniqueConstraint" and name not in db_uniques:
                diffs.append(table.name + ": unique constraint " + name + " missing")
    return diffs
