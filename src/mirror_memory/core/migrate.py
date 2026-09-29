"""Schema guard and migration SQL for existing databases.

``Base.metadata.create_all`` creates missing tables but never adds columns
to tables that already exist.  Once the models grew the metabolism columns,
every ORM query references them — so an engine opened against a database
written by an older mirror-memory would fail mid-query with an obscure
"no such column" error.

Two entry points:

- :func:`check_schema` — the startup guard.  Detects a legacy database and
  raises :class:`ConfigError` with actionable instructions.
- :func:`legacy_migration_sql` — returns the exact additive
  ``ALTER TABLE ... ADD COLUMN`` statements still needed.  The *operator*
  runs them (sqlite3 CLI, Alembic, a DBA change window — whatever the
  deployment uses); this library never executes DDL itself.  That is a
  deliberate boundary: auto-mutating a production database on import is
  not something a library should do on its own, and it keeps every SQL
  string the engine executes behind SQLAlchemy's parameterised layer.

Each statement carries a default so existing rows backfill without a
rewrite.  Drops and type changes are out of scope by design: this is a
one-way, additive migration, and anything destructive belongs in a real
migration tool.
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from mirror_memory.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Columns each existing table must have for the current models to be
# queryable.  Kept in sync with mirror_memory.core.models.
_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "mm_beliefs": (
        "memory_tier",
        "metabolism_state",
        "retention_class",
        "importance_score",
        "access_count",
        "last_accessed_at",
        "last_supported_at",
        "protected_reason",
        "compacted_into",
    ),
    "mm_evidence": (
        "retention_state",
        "compaction_group_id",
    ),
}
# (table, column) -> additive statement.  Static, hand-written, and never
# executed by this module — see the module docstring.
_MIGRATION_STATEMENTS: dict[tuple[str, str], str] = {
    ("mm_beliefs", "memory_tier"):
        "ALTER TABLE mm_beliefs ADD COLUMN memory_tier VARCHAR(16) NOT NULL DEFAULT 'hot'",
    ("mm_beliefs", "metabolism_state"):
        "ALTER TABLE mm_beliefs ADD COLUMN metabolism_state VARCHAR(16) NOT NULL DEFAULT 'active'",
    ("mm_beliefs", "retention_class"):
        "ALTER TABLE mm_beliefs ADD COLUMN retention_class VARCHAR(16) NOT NULL DEFAULT 'preference'",
    ("mm_beliefs", "importance_score"):
        "ALTER TABLE mm_beliefs ADD COLUMN importance_score FLOAT NOT NULL DEFAULT 0.5",
    ("mm_beliefs", "access_count"):
        "ALTER TABLE mm_beliefs ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
    ("mm_beliefs", "last_accessed_at"):
        "ALTER TABLE mm_beliefs ADD COLUMN last_accessed_at DATETIME",
    ("mm_beliefs", "last_supported_at"):
        "ALTER TABLE mm_beliefs ADD COLUMN last_supported_at DATETIME",
    ("mm_beliefs", "protected_reason"):
        "ALTER TABLE mm_beliefs ADD COLUMN protected_reason VARCHAR(64) NOT NULL DEFAULT ''",
    ("mm_beliefs", "compacted_into"):
        "ALTER TABLE mm_beliefs ADD COLUMN compacted_into INTEGER",
    ("mm_evidence", "retention_state"):
        "ALTER TABLE mm_evidence ADD COLUMN retention_state VARCHAR(16) NOT NULL DEFAULT 'hot'",
    ("mm_evidence", "compaction_group_id"):
        "ALTER TABLE mm_evidence ADD COLUMN compaction_group_id VARCHAR(64)",
}

_LEGACY_HINT = (
    "this database was created by an older mirror-memory and is missing "
    "the memory-metabolism columns {detail}. Get the additive migration "
    "statements from mirror_memory.core.migrate.legacy_migration_sql(engine) "
    "and run them with your deployment's migration tooling, or recreate the "
    "database."
)


def missing_columns(engine: Engine) -> list[tuple[str, str]]:
    """``(table, column)`` pairs the current models expect but the DB lacks."""
    inspector = inspect(engine)
    missing: list[tuple[str, str]] = []
    for table, required in _REQUIRED_COLUMNS.items():
        if not inspector.has_table(table):
            continue
        existing = {col["name"] for col in inspector.get_columns(table)}
        for column in required:
            if column not in existing:
                missing.append((table, column))
    return missing


def check_schema(engine: Engine) -> None:
    """Raise :class:`ConfigError` if an existing table predates metabolism.

    Fresh databases (created by ``create_all`` with the current model) and
    databases without any tables yet pass silently.
    """
    missing = missing_columns(engine)
    if missing:
        detail = "; ".join(f"{table}: {column}" for table, column in missing)
        raise ConfigError(_LEGACY_HINT.format(detail=detail))


def legacy_migration_sql(engine: Engine) -> list[str]:
    """The additive statements still needed to bring *engine* up to date.

    Returned, not executed — the operator runs them (see the module
    docstring).  Empty list for a fresh or already-current database.
    """
    statements: list[str] = []
    for table, column in missing_columns(engine):
        statement = _MIGRATION_STATEMENTS.get((table, column))
        if statement is not None:
            statements.append(statement)
    return statements
