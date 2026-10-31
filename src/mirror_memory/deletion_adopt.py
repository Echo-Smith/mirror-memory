"""Operator entry point: claim a pre-M4-restored database for an existing
deletion log (M4 batch-1 patch task 1; lineage-aware migration in patch 2).

Usage:
    python -m mirror_memory.deletion_adopt --database-url <url> --db-id <db_id> \
        [--config-dir config/] [--default-tenant default] [--default-app default]

On a database still BELOW the M4 schema version, the migration runs FIRST
with minting suppressed (the operator's db_id becomes the watermark row,
not a freshly minted one), then the row is written.  The next startup's
reconciliation replays every entry of that log; the fingerprint matching
makes it safe to apply them onto whatever the restored backup contains.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    from mirror_memory.core.schema_migration import (
        LATEST_VERSION,
        current_version,
        upgrade_database,
    )

    parser = argparse.ArgumentParser(
        description="Claim a restored database for an existing deletion log "
                    "(writes the watermark row; the startup replay then "
                    "re-applies the log's deletions).  A database below the "
                    "M4 schema version is migrated first, with minting "
                    "suppressed -- the given db_id becomes the identity.")
    parser.add_argument("--database-url", required=True,
                        help="the main database URL to claim (the OWNER url "
                             "when the schema still needs the migration)")
    parser.add_argument("--db-id", required=True,
                        help="the db_id of the deletion log to adopt")
    parser.add_argument("--config-dir", default="config/",
                        help="the configuration directory the deletion-log "
                             "settings are read from (default: config/)")
    parser.add_argument("--default-tenant", default="default",
                        help="the default tenant for the pre-M4 migration "
                             "(default: default)")
    parser.add_argument("--default-app", default="default",
                        help="the default app for the pre-M4 migration "
                             "(default: default)")
    parser.add_argument("--new-lineage", action="store_true",
                        help="with --db-id of a brand-new log: start a NEW "
                             "deployment line instead of claiming the old "
                             "one (the old log's deletions will NOT be "
                             "replayed)")
    args = parser.parse_args()

    from sqlalchemy import create_engine

    engine = create_engine(args.database_url)
    try:
        version = current_version(engine)
        if version < LATEST_VERSION:
            from mirror_memory.config.loader import load_config

            print(f"schema is at version {version}: running the migration "
                  f"first (minting suppressed -- {args.db_id[:8]}... becomes "
                  "the watermark row)")
            upgrade_database(
                engine,
                default_tenant=args.default_tenant,
                default_app=args.default_app,
                config=load_config(args.config_dir),
                mint_watermark=False,
                # the operator is claiming an already-restored backup: the
                # pre-migration backup requirement is satisfied by the
                # restore itself
                confirm_backup=True,
                new_lineage=args.new_lineage,
            )
        db_id, applied = adopt_watermark(engine, args.db_id)
        print(f"watermark adopted: db_id={db_id} applied_seq={applied}")
        # best-effort informational tail (any config/repair problem here
        # must not fail the adoption itself)
        try:
            from mirror_memory.config.loader import load_config
            from mirror_memory.core.deletion import deletion_log_from_config

            log = deletion_log_from_config(
                load_config(args.config_dir).deletion_log,
                args.database_url)
            tail, _tail_hash = log.head(db_id)
            if tail:
                print(f"deletion log: {tail} entrie(s) under this db_id "
                      "will be replayed at next startup")
            else:
                print("deletion log: no entries under this db_id (nothing "
                      "will be replayed)")
        except Exception as exc:  # noqa: BLE001
            print(f"deletion log: tail not shown ({exc})")
        return 0
    finally:
        engine.dispose()


def adopt_watermark(engine, db_id: str) -> tuple[str, int]:
    from mirror_memory.core.schema_migration import adopt_watermark

    return adopt_watermark(engine, db_id)


if __name__ == "__main__":
    sys.exit(main())
