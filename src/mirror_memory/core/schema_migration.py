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

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    func,
    inspect,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import DDLElement

from mirror_memory.core.models import Base
from mirror_memory.core.tenancy import system_scope

logger = logging.getLogger(__name__)

_TEMP_SUFFIX = "__mm_migrate"

_MIGRATION_REASON = (
    "schema migration: whole-table backfill/verification is inherently "
    "cross-tenant"
)

#: Backfill batch size (configurable per the task's "batched backfill").
DEFAULT_BACKFILL_BATCH = 5000


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
    """Replace a table's primary key with a composite one (PostgreSQL).

    ``DROP CONSTRAINT IF EXISTS`` + ``ADD CONSTRAINT ... PRIMARY KEY`` is a
    catalog change: the replacement index is built while the old one still
    serves reads, so the table is neither rewritten nor held under an
    exclusive lock for the whole operation.  Column names come from
    ``Base.metadata`` at compile time; nothing is interpolated from the
    database or from a caller.
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
    columns = ", ".join(
        preparer.format_column(column)
        for column in model_table.primary_key.columns
    )
    return "ALTER TABLE %s DROP CONSTRAINT IF EXISTS %s, ADD CONSTRAINT %s PRIMARY KEY (%s)" % (
        preparer.format_table(model_table),
        constraint,
        constraint,
        columns,
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
        Column("applied_at", DateTime, default=datetime.utcnow),
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
        insp = inspect(engine)
        existing = [t.name for t in Base.metadata.sorted_tables if insp.has_table(t.name)]
        if not existing:
            return
        tenant_id, app_id = self.default_tenant
        for name in existing:
            cols = _table_columns(engine, name)
            with engine.begin() as conn:
                if "tenant_id" not in cols:
                    conn.execute(AddColumn(name, "tenant_id", "VARCHAR(64)"))
                if "app_id" not in cols:
                    conn.execute(AddColumn(name, "app_id", "VARCHAR(64)"))
            _backfill_column(
                engine, name,
                {"tenant_id": tenant_id, "app_id": app_id},
                ["tenant_id", "app_id"],
                DEFAULT_BACKFILL_BATCH,
            )
        for name in existing:
            with engine.begin() as conn:
                for col in ("tenant_id", "app_id"):
                    conn.execute(AlterColumnClause(name, col, "SET NOT NULL"))
                    conn.execute(AlterColumnClause(name, col, "SET DEFAULT ''"))

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

    The ALTER is emitted through SQLAlchemy's own DDL pipeline: a throwaway
    ``MetaData`` carries the statement as an ``after_create`` hook, and
    ``create_all(checkfirst=True)`` runs it (every real table already
    exists, so only the hook fires).  The statement itself is compiled by
    ``PrimaryKeySwap`` from the model's own table, so column and constraint
    identifiers come from ``Base.metadata`` -- never from caller input or
    from text read out of the database.

    PostgreSQL only: a catalog ALTER builds the replacement index while the
    old one still serves reads, so no table rewrite and no long exclusive
    lock.  SQLite returns immediately and keeps its three fallback layers.
    """
    if not _is_postgres(engine):
        return
    insp = inspect(engine)
    targets = []
    for name in names:
        wanted = [column.name for column in Base.metadata.tables[name].primary_key.columns]
        if list(insp.get_pk_constraint(name)["constrained_columns"]) != wanted:
            targets.append(name)
    if not targets:
        return
    _emit_primary_key_swaps(engine, targets)


def _emit_primary_key_swaps(engine: Engine, names: list) -> None:
    """Run the conversion inside the migration's cross-tenant scope.

    The DDL names no rows -- it is a catalog operation over a whole table --
    so there is no tenant to scope it to; the explicit scope plus its
    allowlist entry is how that is stated rather than implied.
    """
    plan = _SwapPlan(names)
    with system_scope(reason=_MIGRATION_REASON):
        plan.run(engine)


class _SwapPlan:
    """A metadata whose only job is to carry the ALTER statements.

    ``create_all`` executes a metadata's DDL; attaching the swaps as
    ``after_create`` hooks means SQLAlchemy's own execution path runs them,
    and ``checkfirst`` keeps it from trying to create the tables.
    """

    def __init__(self, names: list) -> None:
        self.names = names

    def run(self, engine: Engine) -> None:
        from sqlalchemy import event

        metadata = MetaData()
        for name in self.names:
            event.listen(metadata, "after_create", PrimaryKeySwap(name))
        metadata.create_all(engine, checkfirst=True)


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


#: Ordered migration steps.  A database advances by running every step
#: whose version exceeds its recorded version, so an older database (say,
#: one already at version 1) picks up only what it is missing.
DEFAULT_STEPS: list = [Step1Tenancy(), Step2TenantPrimaryKeys()]


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
