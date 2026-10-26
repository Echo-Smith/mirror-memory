"""Schema migration: base (7f783bc) -> multi-tenant schema (M1 batch-2 task 4).

Approach: a metadata-driven table-rebuild runner embedded in the library.
For every table that lacks the tenant columns it creates the CURRENT ORM
shape (``Base.metadata`` via ``Table.tometadata``) under a temporary name,
copies the rows across while stamping the configured default tenant,
drops the old table and renames the new one.  This is the same mechanism
Alembic's SQLite ``batch`` mode uses (create-copy-drop-rename), with two
differences documented in the batch-2 report:

- the target schema comes from ``Base.metadata`` itself, so
  ORM-vs-migration drift is impossible by construction (the parity test
  reflects the migrated database back against the metadata);
- it is an embedded API (``upgrade_database`` / ``downgrade_database``)
  rather than an alembic env, because this is a library, not a project.
  The operations are portable SQL (no SQLite-only syntax) apart from the
  file-level backup, which is SQLite-file specific; PostgreSQL backup
  strategy lands with M2.

Safety: refuses to run without an explicit default tenant; backs the
database file up before touching it; verifies row counts and tenant
column completeness after the copy; restores the backup automatically
when verification fails.  ``downgrade`` restores the pre-upgrade backup
(data written after the upgrade is lost -- that is the documented cost).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from mirror_memory.core.models import Base

logger = logging.getLogger(__name__)

_TEMP_SUFFIX = "__mm_migrate"


class MigrationError(RuntimeError):
    """The migration could not be completed; the backup was restored."""


class MigrationRefused(MigrationError):
    """Precondition failed; nothing was touched."""


def _tenant_tables(metadata: MetaData) -> list:
    return [t for t in metadata.sorted_tables if "tenant_id" in t.c and "app_id" in t.c]


def _needs_upgrade(engine: Engine) -> bool:
    insp = inspect(engine)
    for table in _tenant_tables(Base.metadata):
        if not insp.has_table(table.name):
            continue  # fresh database: create_all handles it
        cols = {c["name"] for c in insp.get_columns(table.name)}
        if "tenant_id" not in cols or "app_id" not in cols:
            return True
    return False


def _sqlite_file(engine: Engine) -> Path | None:
    url = str(engine.url)
    if not url.startswith("sqlite:///"):
        return None
    path = url[len("sqlite:///"):]
    if not path or path == ":memory:":
        return None
    return Path(path)


def _backup(engine: Engine) -> Path | None:
    path = _sqlite_file(engine)
    if path is None:
        return None
    backup = path.with_name(path.name + ".pre-tenancy.bak")
    shutil.copy2(path, backup)
    return backup


def _restore(engine: Engine, backup: Path | None) -> None:
    path = _sqlite_file(engine)
    if path is None or backup is None:
        return
    engine.dispose()  # pooled connections would keep reading the old file
    shutil.copy2(backup, path)
    engine.dispose()


def _row_counts(engine: Engine, tables: list[str]) -> dict[str, int]:
    counts = {}
    with engine.connect() as conn:
        for name in tables:
            stmt = text(f'SELECT COUNT(*) FROM "{name}"')  # table names are ours, not user input
            counts[name] = conn.execute(stmt).scalar_one()
    return counts


def upgrade_database(
    engine: Engine,
    *,
    default_tenant: str,
    default_app: str,
) -> dict[str, int]:
    """Migrate a base (pre-tenancy) database to the tenant schema.

    Returns per-table copied row counts.  Refuses without an explicit
    default tenant; verifies counts + tenant completeness after the copy;
    restores the backup automatically on any failure.
    """
    if not default_tenant or not default_app:
        raise MigrationRefused(
            "upgrade_database requires an explicit default tenant "
            "(default_tenant=..., default_app=...); existing rows are "
            "assigned to it, so it must be a deliberate configuration value"
        )
    if not _needs_upgrade(engine):
        logger.info("migration: nothing to do (already tenanted or empty)")
        return {}

    tables = [t.name for t in _tenant_tables(Base.metadata)]
    insp = inspect(engine)
    existing = [t for t in tables if insp.has_table(t)]
    from mirror_memory.core.tenancy import system_scope

    with system_scope(reason="schema migration: pre-copy row counts"):
        before = _row_counts(engine, existing)

    backup = _backup(engine)
    fresh_meta = MetaData()
    try:
        from mirror_memory.core.tenancy import system_scope

        with system_scope(
            reason="schema migration: whole-table backfill/verification "
            "is inherently cross-tenant"
        ), engine.begin() as conn:
            for table in _tenant_tables(Base.metadata):
                if not insp.has_table(table.name):
                    continue
                # 1) current ORM shape under a temporary name
                new = table.tometadata(fresh_meta, name=table.name + _TEMP_SUFFIX)
                new.create(conn)
                cols = [c.name for c in table.columns if c.name not in ("tenant_id", "app_id")]
                col_list = ", ".join(f'"{c}"' for c in cols)
                # 2) copy rows, stamping the configured default tenant
                conn.execute(
                    text(
                        f'INSERT INTO "{new.name}" (tenant_id, app_id, {col_list}) '
                        f'SELECT :t, :a, {col_list} FROM "{table.name}"'
                    ).bindparams(t=default_tenant, a=default_app)
                )
                # 3) swap
                conn.execute(text(f'DROP TABLE "{table.name}"'))
                conn.execute(
                    text(f'ALTER TABLE "{new.name}" RENAME TO "{table.name}"')
                )
    except Exception as exc:  # noqa: BLE001
        _restore(engine, backup)
        raise MigrationError(
            f"migration failed ({type(exc).__name__}: {exc}); "
            "the pre-migration backup was restored"
        ) from exc

    # verify: row counts unchanged, tenant columns complete
    try:
        from mirror_memory.core.tenancy import system_scope

        with system_scope(
            reason="schema migration: post-copy verification"
        ):
            after = _row_counts(engine, existing)
            for name in existing:
                if after[name] != before[name]:
                    raise MigrationError(
                        f"verification failed: {name} row count "
                        f"{before[name]} -> {after[name]}"
                    )
            with engine.connect() as conn:
                for name in existing:
                    nulls = conn.execute(
                        text(
                            f'SELECT COUNT(*) FROM "{name}" '
                            'WHERE tenant_id IS NULL OR app_id IS NULL'
                        )
                    ).scalar_one()
                    if nulls:
                        raise MigrationError(
                            f"verification failed: {name} has {nulls} rows "
                            "without tenant columns"
                        )
    except MigrationError:
        _restore(engine, backup)
        raise
    except Exception as exc:  # noqa: BLE001 -- verification crashed: restore
        _restore(engine, backup)
        raise MigrationError(
            f"verification crashed ({type(exc).__name__}: {exc}); "
            "the pre-migration backup was restored"
        ) from exc

    logger.info(
        "migration: upgraded %d table(s) to tenant %r/%r (backup: %s)",
        len(existing), default_tenant, default_app, backup,
    )
    return before


def downgrade_database(engine: Engine) -> bool:
    """Restore the pre-upgrade backup.

    The tenancy upgrade is a table rebuild, so downgrade = backup restore.
    Documented data loss: everything written between upgrade and
    downgrade.  Returns True when a backup was restored.
    """
    backup = _sqlite_file(engine)
    if backup is None:
        raise MigrationRefused(
            "downgrade requires a SQLite file database (in-memory or "
            "non-SQLite backup strategies are M2 scope)"
        )
    pre = backup.with_name(backup.name + ".pre-tenancy.bak")
    if not pre.exists():
        raise MigrationRefused(f"no pre-upgrade backup found at {pre}")
    _restore(engine, pre)
    logger.warning(
        "migration: downgraded by restoring %s (writes made after the "
        "upgrade are lost)", pre,
    )
    return True


def verify_schema_parity(engine: Engine) -> list[str]:
    """Reflect the database and compare against Base.metadata.

    Returns a list of differences (empty == parity).  Because the
    migration builds tables FROM the metadata, any difference indicates
    drift introduced after the migration (e.g. a model change without a
    migration).
    """
    insp = inspect(engine)
    diffs: list[str] = []
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            diffs.append(f"missing table {table.name}")
            continue
        db_cols = {c["name"]: c for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in db_cols:
                diffs.append(f"{table.name}.{col.name} missing in database")
                continue
            db_col = db_cols[col.name]
            if db_col.get("nullable", True) != col.nullable:
                diffs.append(
                    f"{table.name}.{col.name} nullable "
                    f"{db_col.get('nullable')} != {col.nullable}"
                )
        extra = set(db_cols) - {c.name for c in table.columns}
        if extra:
            diffs.append(f"{table.name} extra columns {sorted(extra)}")
        db_uniques = {
            uq.get("name") for uq in insp.get_unique_constraints(table.name)
        } | {ix.get("name") for ix in insp.get_indexes(table.name) if ix.get("unique")}
        for constraint in table.constraints:
            name = getattr(constraint, "name", None)
            if name is None:
                continue
            kind = type(constraint).__name__
            if kind == "UniqueConstraint" and name not in db_uniques:
                diffs.append(f"{table.name}: unique constraint {name} missing")
    return diffs
