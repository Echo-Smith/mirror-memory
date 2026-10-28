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

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
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


# ---------------------------------------------------------------------------
# Cross-purpose visibility (M3 batch 3, the A2 rule)
# ---------------------------------------------------------------------------

#: The ledger events that record where a belief CAME FROM.  Support,
#: contradiction, confirmation and the worker's publish events are
#: consequences, not provenance -- they never make a belief visible.
PROVENANCE_EVENT_TYPES: tuple[str, ...] = ("created", "corrected")


@dataclass(frozen=True)
class VisibilityRule:
    """Which source purposes the current request may read memory from.

    ``live`` is the set of the user's purposes with a grant that is neither
    revoked nor expired; the empty marker counts when ``unknown_visible``
    (the global memory switch -- every consumer of a rule runs behind
    authorize, which has already verified it, so it is True in product
    paths; tests may pin the off-branch directly).
    """

    live: frozenset[str]
    unknown_visible: bool = True

    def purpose_visible(self, source_purpose: str) -> bool:
        if source_purpose == UNKNOWN:
            return self.unknown_visible
        return source_purpose in self.live

    def visible_purposes(self) -> frozenset[str]:
        return self.live | ({UNKNOWN} if self.unknown_visible else frozenset())


def visibility(session: Session, user_id: str, purpose: str | None = None) -> VisibilityRule:
    """The visibility rule for one request: the user's live purpose set,
    read ONCE per request (callers pass the same rule to every filter).

    A purpose is live while its grant is neither revoked nor expired -- the
    same criterion authorize() applies to the request's own purpose,
    widened to every purpose whose collected memory may be shown.  No
    caching across transactions: the query runs inside the caller's
    transaction, so a revocation that committed before this SELECT is
    already excluded (revocation stays immediate).

    *purpose* is the reading purpose (already authorized, so it is in
    ``live`` by construction); passing it validates the shape and documents
    the request.  On PostgreSQL the query is tenant-scoped twice over:
    loader criteria inject the tenant predicates and the grants table's
    primary key is the composite one.
    """
    from mirror_memory.core.authorization import _now
    from mirror_memory.core.models import PurposeGrant
    from sqlalchemy import or_

    if purpose is not None:
        resolved = str(purpose).strip()
        if not resolved or any(c.isspace() for c in resolved):
            raise ValueError("reading purpose must be a non-empty string without whitespace")
    rows = session.query(PurposeGrant.purpose).filter(
        PurposeGrant.user_id == user_id,
        PurposeGrant.revoked_at.is_(None),
        or_(PurposeGrant.expires_at.is_(None), PurposeGrant.expires_at > _now()),
    ).all()
    return VisibilityRule(live=frozenset(r[0] for r in rows if r[0]))


def provenance_purposes(
    session: Session, user_id: str, belief_ids: list[int],
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    """Per belief id: the source purposes of its linked evidence, and of its
    provenance events (created / corrected).  Two indexed queries for the
    whole candidate set -- never per belief.

    The evidence join carries the full composite key: evidence ids are only
    unique per tenant, so matching on the bare id could pair a belief with
    another tenant's evidence row of the same number.
    """
    from mirror_memory.core.models import BeliefEvent, BeliefEvidenceLink, Evidence

    if not belief_ids:
        return {}, {}
    link_purposes: dict[int, set[str]] = {b: set() for b in belief_ids}
    rows = session.execute(
        select(BeliefEvidenceLink.belief_id, Evidence.source_purpose)
        .join(
            Evidence,
            (Evidence.tenant_id == BeliefEvidenceLink.tenant_id)
            & (Evidence.app_id == BeliefEvidenceLink.app_id)
            & (Evidence.id == BeliefEvidenceLink.evidence_id),
        )
        .where(
            BeliefEvidenceLink.belief_id.in_(belief_ids),
            Evidence.user_id == user_id,
        )
    ).all()
    for belief_id, source_purpose in rows:
        link_purposes.setdefault(belief_id, set()).add(source_purpose or UNKNOWN)
    event_purposes: dict[int, set[str]] = {b: set() for b in belief_ids}
    rows = session.execute(
        select(BeliefEvent.belief_id, BeliefEvent.source_purpose)
        .where(
            BeliefEvent.belief_id.in_(belief_ids),
            BeliefEvent.user_id == user_id,
            BeliefEvent.event_type.in_(PROVENANCE_EVENT_TYPES),
        )
    ).all()
    for belief_id, source_purpose in rows:
        event_purposes.setdefault(belief_id, set()).add(source_purpose or UNKNOWN)
    return link_purposes, event_purposes


def belief_is_visible(
    session: Session, user_id: str, rule: VisibilityRule, belief_id: int,
) -> bool:
    """The A2 rule for one belief id (see :func:`beliefs_visible`)."""
    link_purposes, event_purposes = provenance_purposes(session, user_id, [belief_id])
    links = link_purposes.get(belief_id) or set()
    if links:
        return any(rule.purpose_visible(sp) for sp in links)
    events = event_purposes.get(belief_id) or set()
    if events:
        return any(rule.purpose_visible(sp) for sp in events)
    # no provenance recorded at all (pre-M3 shape, direct seeding): the
    # unknown marker -- visible while the global switch is on
    return rule.unknown_visible


def beliefs_visible(
    session: Session, user_id: str, rule: VisibilityRule, beliefs: list,
) -> list:
    """The beliefs whose provenance passes *rule*.

    - With evidence links: visible iff ANY linked evidence was collected
      under a live purpose (or under the unknown marker while the global
      switch is on).
    - Without evidence links: visible iff any created / corrected event was
      recorded under a visible purpose.  Other events never count -- a
      support or publish event written under Q must not launder memory
      collected under a since-revoked P (pinned by test).
    - With neither (no links, no provenance events): no source was ever
      recorded -- the pre-M3 / direct-seeding shape.  It behaves as the
      unknown marker: visible while the global switch is on.  A belief
      whose provenance EXISTS and is entirely invisible stays hidden; only
      the truly unrecorded fall back to the marker.
    """
    ids = [b.id for b in beliefs]
    link_purposes, event_purposes = provenance_purposes(session, user_id, ids)
    out = []
    for belief in beliefs:
        if _rule_passes(rule, link_purposes.get(belief.id), event_purposes.get(belief.id)):
            out.append(belief)
    return out


def _rule_passes(rule: VisibilityRule, links: set[str] | None, events: set[str] | None) -> bool:
    links = links or set()
    if links:
        return any(rule.purpose_visible(sp) for sp in links)
    events = events or set()
    if events:
        return any(rule.purpose_visible(sp) for sp in events)
    return rule.unknown_visible
