"""Operator entry point: claim a pre-M4-restored database for an existing
deletion log (M4 batch-1 patch task 1).

Usage:
    python -m mirror_memory.deletion_adopt --database-url <url> --db-id <db_id>

Writes the watermark row (db_id, applied_seq = 0).  The next startup's
reconciliation replays every entry of that log; the fingerprint matching
makes it safe to apply them onto whatever the restored backup contains.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    from mirror_memory.core.schema_migration import adopt_watermark
    from mirror_memory.core.deletion import FileDeletionLog

    parser = argparse.ArgumentParser(
        description="Claim a restored database for an existing deletion log "
                    "(writes the watermark row; the startup replay then "
                    "re-applies the log's deletions).")
    parser.add_argument("--database-url", required=True,
                        help="the main database URL to claim")
    parser.add_argument("--db-id", required=True,
                        help="the db_id of the deletion log to adopt")
    args = parser.parse_args()

    from sqlalchemy import create_engine

    engine = create_engine(args.database_url)
    try:
        db_id, applied = adopt_watermark(engine, args.db_id)
        log = FileDeletionLog(_default_directory(args.database_url))
        tail, _tail_hash = log.head(db_id)
        print(f"watermark adopted: db_id={db_id} applied_seq={applied}")
        if tail:
            print(f"deletion log: {tail} entrie(s) under this db_id will be "
                  "replayed at next startup")
        else:
            print("deletion log: no entries under this db_id (nothing will "
                  "be replayed)")
        return 0
    finally:
        engine.dispose()


def _default_directory(database_url: str) -> str:
    from mirror_memory.core.deletion import default_directory

    return default_directory(database_url)


if __name__ == "__main__":
    sys.exit(main())
