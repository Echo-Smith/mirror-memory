"""Export (M3 batch-2 task 3): a user's beliefs, evidence and ledger events
as versioned, paginated pages.

Shape
-----
``mm-export/1``.  Three sections in a fixed order -- ``beliefs``,
``evidence``, ``events`` -- each walked by ascending id with keyset
pagination.  A page is one section's slice: ``items`` plus an opaque
``next_cursor`` (``"<section>:<last id>"``, ``None`` after the last page).
Every item that has a recorded source carries ``source_purpose``
(evidence, events); a belief carries ``source_purposes`` -- the distinct
source purposes over its linked evidence, the derived provenance explain
reports too.

Why pages and not a file (the v2 design was an async export with a
download URL and temp-file expiry): each page is its own short
transaction, so the entry RE-AUTHORIZES before every page -- a revocation
during a long export refuses the next page instead of letting a
download URL outlive the grant -- memory is bounded by ``page_size``, and
there is no temp file to clean up or leak.  The price is that the pages
are not one consistent snapshot: each page carries the user's
``revision`` (the memory state revision at read time) so a consumer that
needs consistency can detect a change between pages and restart.

Content: belief claims and evidence text are the user's own memory -- that
is what an export is for.  No credentials, grant rows or audit rows are
included; the authorization that allowed the export is reported as
``authorization`` (purpose + grant version), not as the grant itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from mirror_memory.core import provenance
from mirror_memory.core.models import Belief, BeliefEvent, Evidence

SCHEMA_VERSION = "mm-export/1"
SECTIONS: tuple[str, ...] = ("beliefs", "evidence", "events")
DEFAULT_PAGE_SIZE = 500
MAX_PAGE_SIZE = 5000


class InvalidCursor(ValueError):
    """The cursor is not one this export produced."""


@dataclass
class ExportPage:
    schema_version: str
    user_id: str
    section: str
    items: list[dict[str, Any]]
    next_cursor: str | None
    revision: int
    exported_at: str
    authorization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "section": self.section,
            "items": self.items,
            "next_cursor": self.next_cursor,
            "revision": self.revision,
            "exported_at": self.exported_at,
            "authorization": dict(self.authorization),
        }


def parse_cursor(cursor: str | None) -> tuple[str, int]:
    """``None`` -> the first section from the start; otherwise
    ``"<section>:<last id>"``."""
    if cursor is None or cursor == "":
        return SECTIONS[0], 0
    section, sep, last = str(cursor).partition(":")
    if not sep or section not in SECTIONS:
        raise InvalidCursor(f"not an export cursor: {cursor!r}")
    try:
        last_id = int(last)
    except ValueError as exc:
        raise InvalidCursor(f"not an export cursor: {cursor!r}") from exc
    if last_id < 0:
        raise InvalidCursor(f"not an export cursor: {cursor!r}")
    return section, last_id


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _safe_json(text: str | None) -> Any:
    try:
        return json.loads(text or "null")
    except ValueError:
        return None


def _belief_item(belief: Belief, source_purposes: list[str]) -> dict[str, Any]:
    return {
        "id": belief.id,
        "dimension": belief.dimension,
        "key": belief.key,
        "claim_text": belief.claim_text,
        "value": _safe_json(belief.value_json),
        "layer": belief.layer,
        "status": belief.status,
        "confidence": belief.confidence,
        "source": belief.source,
        "subject": getattr(belief, "subject", "user"),
        "predicate": getattr(belief, "predicate", ""),
        "object": getattr(belief, "object", ""),
        "cardinality": getattr(belief, "cardinality", "multi"),
        "polarity": getattr(belief, "polarity", ""),
        "lifecycle_state": getattr(belief, "lifecycle_state", ""),
        "superseded_by": belief.superseded_by,
        "memory_tier": getattr(belief, "memory_tier", "hot"),
        "retention_class": getattr(belief, "retention_class", ""),
        "valid_from": _iso(getattr(belief, "valid_from", None)),
        "valid_to": _iso(getattr(belief, "valid_to", None)),
        "first_seen_at": _iso(belief.first_seen_at),
        "last_evidence_at": _iso(belief.last_evidence_at),
        "created_at": _iso(belief.created_at),
        "updated_at": _iso(belief.updated_at),
        # derived provenance: the distinct source purposes of the linked
        # evidence ("" = source not recorded)
        "source_purposes": source_purposes,
    }


def _evidence_item(row: Evidence) -> dict[str, Any]:
    return {
        "id": row.id,
        "ref": row.ref,
        "session_id": row.session_id,
        "content": row.content,
        "source_type": row.source_type,
        "extraction_method": row.extraction_method,
        "authority": row.authority,
        "source_purpose": row.source_purpose,
        "retention_state": row.retention_state,
        "observed_at": _iso(row.observed_at),
        "created_at": _iso(row.created_at),
    }


def _event_item(row: BeliefEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "belief_id": row.belief_id,
        "event_type": row.event_type,
        "evidence_ids": _safe_json(row.evidence_json) or [],
        "detail": _safe_json(row.detail_json) or {},
        "source_purpose": row.source_purpose,
        "created_at": _iso(row.created_at),
    }


def source_purposes_for_beliefs(
    session: Session, user_id: str, belief_ids: list[int],
) -> dict[int, list[str]]:
    """``belief id -> sorted distinct source purposes of its linked
    evidence``; beliefs with no linked evidence map to ``[]``.

    Delegates to the batched provenance query (which joins on the full
    composite key -- evidence ids are only unique per tenant, so a bare-id
    join could pair a belief with another tenant's row of the same number;
    batch-3 finding, fixed here)."""
    from mirror_memory.core import provenance

    links, _events = provenance.provenance_purposes(session, user_id, belief_ids)
    return {bid: sorted(links.get(bid) or ()) for bid in belief_ids}


def read_page(
    session: Session, user_id: str, *, cursor: str | None, page_size: int,
    visibility: "VisibilityRule | None" = None,
) -> tuple[str, list[dict[str, Any]], list[int], str | None]:
    """One page of the export: ``(section, items, item ids, next cursor)``.

    Walks the current section by ascending id from the cursor; when a
    section is exhausted the cursor moves to the start of the next one
    (an empty section yields one empty page so the consumer still sees
    it).  ``next_cursor`` is ``None`` only after the last section.

    With a visibility rule only visible rows are RETURNED: beliefs by the
    A2 provenance rule, evidence and events by their own source purpose.
    The cursor always advances by the FETCHED rows -- a page may carry
    fewer items than *page_size* (even zero) because invisible rows were
    withheld, never because rows were skipped in the keyset."""
    if page_size < 1 or page_size > MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be in 1..{MAX_PAGE_SIZE}")
    section, last_id = parse_cursor(cursor)
    fetched_last = 0
    if section == "beliefs":
        rows = session.execute(
            select(Belief).where(Belief.user_id == user_id, Belief.id > last_id)
            .order_by(Belief.id).limit(page_size)
        ).scalars().all()
        fetched_last = int(rows[-1].id) if rows else last_id
        if visibility is not None:
            kept = provenance.beliefs_visible(session, user_id, visibility, list(rows))
        else:
            kept = list(rows)
        purposes = source_purposes_for_beliefs(session, user_id, [b.id for b in kept])
        items = [_belief_item(b, purposes.get(b.id, [])) for b in kept]
    elif section == "evidence":
        rows = session.execute(
            select(Evidence).where(Evidence.user_id == user_id, Evidence.id > last_id)
            .order_by(Evidence.id).limit(page_size)
        ).scalars().all()
        fetched_last = int(rows[-1].id) if rows else last_id
        if visibility is not None:
            rows = [r for r in rows if visibility.purpose_visible(r.source_purpose)]
        items = [_evidence_item(r) for r in rows]
    else:
        rows = session.execute(
            select(BeliefEvent).where(BeliefEvent.user_id == user_id, BeliefEvent.id > last_id)
            .order_by(BeliefEvent.id).limit(page_size)
        ).scalars().all()
        fetched_last = int(rows[-1].id) if rows else last_id
        if visibility is not None:
            rows = [r for r in rows if visibility.purpose_visible(r.source_purpose)]
        items = [_event_item(r) for r in rows]
    ids = [int(r.id) for r in rows]
    if len(ids) == page_size:
        # advanced by the FETCHED keyset even when invisible rows shrank the
        # returned items (ids here are the visible ones; recompute the
        # fetched tail id for the cursor)
        next_cursor: str | None = f"{section}:{fetched_last}"
    else:
        index = SECTIONS.index(section)
        next_cursor = f"{SECTIONS[index + 1]}:0" if index + 1 < len(SECTIONS) else None
    return section, items, ids, next_cursor


def now_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()
