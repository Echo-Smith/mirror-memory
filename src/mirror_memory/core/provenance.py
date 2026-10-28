"""Transaction-level write provenance: the purpose under which the rows an
entry writes were collected (M3 batch-2 task 1).

Every row that records *where memory came from* -- an evidence row, a
belief-ledger event -- carries ``source_purpose``: the purpose the entry
that wrote it was authorized for.  ``observe`` writes the purpose of the
current call, the worker writes the purpose of the job it is running,
``correct`` / ``forget`` write theirs.  The value is bound to the
*session* once by the entry (right after it authorizes, before any
business statement) and read by the two writers at insert time, so the
thirteen call sites that append ledger events and the one that records
evidence need no extra parameter and cannot disagree with the entry.

Rows written without a binding -- pre-M3 rows, and rows produced by direct
repository use outside an authorized entry (bench seeding, tests) -- carry
``UNKNOWN`` (the empty string), the same marker migration step 7 backfills:
"source not recorded".  Nothing filters on the value yet; the recall
semantics for cross-purpose evidence are a separate design decision
(docs/plans, M3 batch-2 task 4).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

#: Source not recorded: pre-M3 rows and rows written outside an authorized
#: entry.  The empty string, not NULL -- the column is NOT NULL so a row can
#: never be "half stamped".
UNKNOWN: str = ""

_KEY = "mm_source_purpose"


def bind(session: Session, purpose: str) -> None:
    """Bind *purpose* as the source purpose of everything this session
    writes from now on.  The entry calls this once per transaction with the
    purpose it just authorized (a non-empty, whitespace-free string)."""
    if not isinstance(purpose, str) or not purpose.strip() or any(c.isspace() for c in purpose):
        raise ValueError("source purpose must be a non-empty string without whitespace")
    session.info[_KEY] = purpose


def unbind(session: Session) -> None:
    session.info.pop(_KEY, None)


def current(session: Session) -> str:
    """The purpose bound to *session*, or ``UNKNOWN`` when the session is
    not inside an authorized entry."""
    return session.info.get(_KEY, UNKNOWN)
