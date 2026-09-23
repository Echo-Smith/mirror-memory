"""Belief repository -- merge strategy, confidence update, resurrection guard.

Merge strategy (generic):
- ``(user_id, key)`` has exactly one current row; ``relation=supports``
  pushes confidence and appends evidence, ``relation=contradicts`` attenuates
  confidence.
- After rejection (``reject_belief``) the same key is never resurrected:
  new claims return ``(None, "resurrection_guard")`` with no event logged.
- ``source="extracted"`` rows are forced to ``layer="L4"``; ``L2`` can only
  come from user confirmation (``confirm_belief``) or program hard evidence.

All queries use SQLAlchemy 2.0 parameterised construction (bound parameters,
no string concatenation).

Domain overrides:
- ``blocked_key_prefixes``: list of key prefixes that are structurally
  forbidden from persisting (e.g. ``["distortion."]`` in a psychology
  domain).  Defaults to empty.
- ``question_value_tiers``: mapping of dimension -> int tier for question
  candidate prioritisation.  Dimensions not in the map get
  ``DEFAULT_QUESTION_TIER``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import desc, false, func, or_, select
from sqlalchemy.orm import Session

from mirror_memory.core.confidence import compute_confidence_weight
from mirror_memory.core.utils import (
    belief_evidence_ids,
    content_tokens,
    coerce_datetime,
    safe_json,
)
from mirror_memory.core.constants import (
    CONFIDENCE_CEILING,
    DEFAULT_QUESTION_TIER,
    L4_QUESTION_THRESHOLD,
    LIFE_EVENT_EXPIRY_DAYS,
    MAX_EVIDENCE_REFS,
    QUESTION_TIER_WEIGHT_MAX,
    QUESTION_TIER_WEIGHT_MIN,
    SUPPORT_GAIN,
)
from mirror_memory.core.models import (
    Belief,
    BeliefEvidenceLink,
    BeliefEvent,
    ConsentGrant,
    EvolutionJob,
    Evidence,
    ExtractionStats,
    InterventionEvent,
    MemoryPreference,
    SessionSummary,
    Snapshot,
    utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory preference
# ---------------------------------------------------------------------------


def is_memory_enabled(session: Session, user_id: str) -> bool:
    """Return the user's memory preference; missing rows = enabled (legacy-on)."""
    pref = session.get(MemoryPreference, user_id)
    return pref.enabled if pref is not None else True


def set_memory_enabled(session: Session, user_id: str, enabled: bool) -> MemoryPreference:
    """Enable or disable memory for a user.

    Toggling the switch is a state change, so it bumps the state revision:
    a job that read the user's evidence while memory was still enabled must
    not be allowed to publish after the switch was turned off.  A no-op call
    does not bump -- including writing ``enabled=True`` for a user with no
    preference row, since a missing row already means enabled.
    """
    pref = session.get(MemoryPreference, user_id)
    changed = (pref is None and not enabled) or (
        pref is not None and bool(pref.enabled) != bool(enabled)
    )
    if pref is None:
        pref = MemoryPreference(user_id=user_id)
        session.add(pref)
    pref.enabled = enabled
    pref.disabled_at = None if enabled else utcnow()
    pref.updated_at = utcnow()
    session.flush()
    if changed:
        bump_state_revision(session, user_id)
    return pref


def get_state_revision(session: Session, user_id: str) -> int:
    """Return the current memory state revision for a user.

    Uses a direct SELECT to bypass SQLAlchemy identity map cache,
    ensuring we always read the latest committed value even if
    another session has bumped the revision.

    Missing rows return revision 1 (default).
    """
    result = session.execute(
        select(MemoryPreference.state_revision).where(
            MemoryPreference.user_id == user_id
        )
    ).scalar()
    return result if result is not None else 1


def bump_state_revision(session: Session, user_id: str) -> int:
    """Atomic increment of the memory state revision.

    Uses SQL UPDATE ... SET state_revision = state_revision + 1
    to guarantee no lost updates when two sessions bump concurrently.
    Row creation uses INSERT OR IGNORE to handle concurrent first-touch.
    """
    from sqlalchemy import update as sa_update
    # Ensure row exists.  The insert runs inside a savepoint so that losing a
    # race with a concurrent first-touch cannot roll back the caller's whole
    # pending transaction.
    existing = session.get(MemoryPreference, user_id)
    if existing is None:
        with session.begin_nested():
            session.add(MemoryPreference(user_id=user_id, state_revision=1))
    # Atomic increment at SQL level.
    result = session.execute(
        sa_update(MemoryPreference)
        .where(MemoryPreference.user_id == user_id)
        .values(state_revision=MemoryPreference.state_revision + 1, updated_at=utcnow())
        .returning(MemoryPreference.state_revision)
    )
    new_rev = result.scalar_one_or_none()
    session.flush()
    return new_rev


def touch_memory_state(session: Session, user_id: str) -> int:
    """Bump the memory state revision after any state-changing mutation.

    Call this at the end of: CREATE, SUPPORT, UPDATE, CONTRADICT,
    CONFIRM, REJECT, CORRECT, FORGET operations.
    """
    return bump_state_revision(session, user_id)


def is_feature_enabled(session: Session, user_id: str, feature: str) -> bool:
    """Check feature-level consent; falls back to global memory preference."""
    grant = session.get(ConsentGrant, (user_id, feature))
    if grant is not None:
        return grant.granted
    return is_memory_enabled(session, user_id)


def set_feature_consent(session: Session, user_id: str, feature: str, granted: bool) -> None:
    """Set feature-level consent for a user."""
    existing = session.get(ConsentGrant, (user_id, feature))
    if existing is not None:
        existing.granted = granted
        existing.updated_at = utcnow()
    else:
        session.add(ConsentGrant(user_id=user_id, feature=feature, granted=granted))
    session.flush()


# ---------------------------------------------------------------------------
# Evidence -- first-class provenance
# ---------------------------------------------------------------------------

# Valid link relations.  ``support`` and ``contradict`` are opposites on the
# same axis; ``verify`` is neutral confirmation; ``correct`` is an explicit
# user override that supersedes whatever the evidence originally implied.
EVIDENCE_RELATIONS = ("support", "contradict", "verify", "correct")

# Recognised evidence authorities, strongest first.  A user's own statement
# outranks a derived one; the engine never invents authority on its own.
EVIDENCE_AUTHORITIES = ("user", "assistant", "system", "derived")


def record_evidence(
    session: Session,
    user_id: str,
    *,
    ref: str,
    content: str = "",
    session_id: str | None = None,
    source_type: str = "message",
    extraction_method: str = "",
    authority: str = "user",
    observed_at: datetime | None = None,
) -> Evidence:
    """Record (or re-find) one piece of evidence for *user_id*.

    Idempotent on ``(user_id, ref)``: observing the same message twice returns
    the existing row with its provenance refreshed, so evidence is never
    duplicated by re-ingestion.

    Returns ``None``-free: always an :class:`Evidence`.
    """
    existing = session.scalar(
        select(Evidence).where(Evidence.user_id == user_id, Evidence.ref == ref)
    )
    if existing is not None:
        # Refresh the mutable provenance fields; identity fields stay put.
        if content and len(content) > len(existing.content or ""):
            existing.content = content
        if session_id:
            existing.session_id = session_id
        if extraction_method:
            existing.extraction_method = extraction_method
        if authority:
            existing.authority = authority
        if observed_at is not None:
            existing.observed_at = observed_at
        session.flush()
        return existing

    row = Evidence(
        user_id=user_id,
        session_id=session_id,
        ref=ref,
        content=content,
        source_type=source_type,
        extraction_method=extraction_method,
        authority=authority,
        observed_at=observed_at or utcnow(),
    )
    session.add(row)
    session.flush()
    return row


def link_evidence(
    session: Session,
    belief_id: int,
    evidence_id: int,
    *,
    relation: str = "support",
) -> BeliefEvidenceLink | None:
    """Attach one evidence row to one belief with a typed relation.

    Idempotent on ``(belief_id, evidence_id, relation)``.  Returns the link,
    or ``None`` if the relation is not recognised.
    """
    if relation not in EVIDENCE_RELATIONS:
        raise ValueError(f"invalid evidence relation: {relation!r}")

    existing = session.scalar(
        select(BeliefEvidenceLink).where(
            BeliefEvidenceLink.belief_id == belief_id,
            BeliefEvidenceLink.evidence_id == evidence_id,
            BeliefEvidenceLink.relation == relation,
        )
    )
    if existing is not None:
        return existing

    link = BeliefEvidenceLink(
        belief_id=belief_id,
        evidence_id=evidence_id,
        relation=relation,
    )
    session.add(link)
    session.flush()
    return link


def evidence_for_belief(
    session: Session,
    belief_id: int,
    *,
    relations: tuple[str, ...] | None = None,
) -> list[tuple[Evidence, str]]:
    """Return ``(evidence, relation)`` pairs attached to *belief_id*.

    Ordered newest-observed first.  *relations* narrows the result to a subset
    of link types (e.g. only ``("support",)`` to see what backs a belief).
    """
    stmt = (
        select(Evidence, BeliefEvidenceLink.relation)
        .join(BeliefEvidenceLink, BeliefEvidenceLink.evidence_id == Evidence.id)
        .where(BeliefEvidenceLink.belief_id == belief_id)
        .order_by(desc(Evidence.observed_at), desc(Evidence.id))
    )
    if relations is not None:
        stmt = stmt.where(BeliefEvidenceLink.relation.in_(relations))
    return [(row[0], row[1]) for row in session.execute(stmt).all()]


def evidence_summary(session: Session, user_id: str, belief_id: int) -> dict:
    """Explainability view of a belief's evidence, grouped by relation.

    Returns counts per relation plus the supporting evidence's provenance, so
    a caller can answer "what backs this belief, and how was it obtained?"
    without walking the link table itself.
    """
    pairs = evidence_for_belief(session, belief_id)
    grouped: dict[str, list[dict]] = {}
    for evidence, relation in pairs:
        grouped.setdefault(relation, []).append(
            {
                "evidence_id": evidence.id,
                "ref": evidence.ref,
                "source_type": evidence.source_type,
                "extraction_method": evidence.extraction_method,
                "authority": evidence.authority,
                "observed_at": evidence.observed_at.isoformat() if evidence.observed_at else None,
                "session_id": evidence.session_id,
            }
        )
    return {
        "belief_id": belief_id,
        "user_id": user_id,
        "counts": {relation: len(items) for relation, items in grouped.items()},
        "by_relation": grouped,
    }


def _prune_orphan_evidence(session: Session, user_id: str) -> int:
    """Delete evidence rows for *user_id* that no belief points at any more.

    Evidence is shared: one message can back several beliefs, so a row only
    becomes garbage once its last link is gone.
    """
    linked_ids = select(BeliefEvidenceLink.evidence_id).where(
        BeliefEvidenceLink.evidence_id.in_(
            select(Evidence.id).where(Evidence.user_id == user_id)
        )
    )
    deleted = (
        session.query(Evidence)
        .filter(Evidence.user_id == user_id, Evidence.id.not_in(linked_ids))
        .delete(synchronize_session=False)
    )
    return int(deleted)


def evidence_counts_for_beliefs(session: Session, belief_ids: list[int]) -> dict[int, int]:
    """Return ``{belief_id: evidence_count}`` for the given beliefs.

    Counts links, so one message that both supports and contradicts a belief
    counts once per typed edge -- which is the point of typed links.
    """
    if not belief_ids:
        return {}
    rows = session.execute(
        select(BeliefEvidenceLink.belief_id, func.count(BeliefEvidenceLink.id))
        .where(BeliefEvidenceLink.belief_id.in_(belief_ids))
        .group_by(BeliefEvidenceLink.belief_id)
    ).all()
    return {row[0]: int(row[1]) for row in rows}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_event(
    session: Session,
    belief: Belief,
    event_type: str,
    *,
    evidence: list[int],
    detail: dict | None = None,
    stats_id: int | None = None,
) -> None:
    session.add(
        BeliefEvent(
            belief_id=belief.id,
            user_id=belief.user_id,
            event_type=event_type,
            evidence_json=json.dumps(evidence),
            detail_json=json.dumps(detail or {}, ensure_ascii=False),
            origin_stats_id=stats_id,
        )
    )


# ---------------------------------------------------------------------------
# Belief CRUD
# ---------------------------------------------------------------------------


def get_belief(session: Session, user_id: str, key: str) -> Belief | None:
    """Return the current row for ``(user_id, key)`` (including rejected)."""
    return session.scalar(select(Belief).where(Belief.user_id == user_id, Belief.key == key))


def get_active_belief(session: Session, user_id: str, key: str) -> Belief | None:
    belief = get_belief(session, user_id, key)
    if belief is not None and belief.status == "active":
        return belief
    return None


def list_active_beliefs(
    session: Session,
    user_id: str,
    *,
    dimensions: tuple[str, ...] | None = None,
    limit: int = 50,
    temporal_mode: str = "all",
) -> list[Belief]:
    """Active beliefs, ordered by most-recent evidence.  Returns [] if memory
    is disabled for the user.

    ``temporal_mode`` filters on the belief's validity interval, which is what
    separates "where do they live now?" from "where did they live before?":

    - ``"all"`` (default): every active belief, current and historical.
    - ``"current"``: only beliefs whose interval covers now (``valid_to`` is
      NULL or in the future).
    - ``"historical"``: only beliefs whose interval has closed.  Superseded
      rows are included here -- a closed interval *is* the history.
    """
    if not is_memory_enabled(session, user_id):
        return []
    conditions = [Belief.user_id == user_id]
    if temporal_mode == "historical":
        # A superseded belief is exactly a closed interval; excluding it would
        # make "where did they live before?" unanswerable.
        conditions.append(Belief.status.in_(("active", "superseded")))
    else:
        conditions.append(Belief.status == "active")
    if dimensions:
        conditions.append(Belief.dimension.in_(dimensions))
    stmt = select(Belief).where(*conditions).order_by(desc(Belief.last_evidence_at)).limit(limit)
    results = list(session.scalars(stmt))
    # Filter out expired life events.
    results = [b for b in results if not _is_expired(b)]
    return _filter_by_temporal_mode(results, temporal_mode)


def _filter_by_temporal_mode(beliefs: list[Belief], temporal_mode: str) -> list[Belief]:
    """Keep the beliefs whose validity interval matches *temporal_mode*.

    A belief with no interval information at all (no ``valid_from`` and no
    ``valid_to``) is *undated*, not current-or-historical.  It passes both
    filters: an undated episodic event ("went to the museum") is exactly what
    a question about the past needs, and excluding it would make every
    historical query return nothing.
    """
    if temporal_mode == "all":
        return beliefs
    now = datetime.now(UTC)
    if temporal_mode == "current":
        return [
            b for b in beliefs
            if _is_undated(b) or _interval_is_current(b, now)
        ]
    if temporal_mode == "historical":
        return [
            b for b in beliefs
            if _is_undated(b) or not _interval_is_current(b, now)
        ]
    return beliefs


def _is_undated(belief: Belief) -> bool:
    """Does the belief carry no validity interval at all?"""
    return belief.valid_from is None and belief.valid_to is None


def recall_candidates(
    session: Session,
    user_id: str,
    *,
    query: str = "",
    dimensions: tuple[str, ...] | None = None,
    temporal_mode: str = "all",
    relevant_limit: int = 150,
    recent_limit: int = 25,
) -> list[Belief]:
    """Query-aware admission: which beliefs could matter for *query*?

    The old renderer path took the N most recently evidenced beliefs and
    scored those.  That makes recency a stand-in for relevance, so a fact
    stated early in a long conversation is discarded before any relevance
    signal runs -- measured on LOCOMo, the answer belief sat at recency rank
    153 of 161 and never reached the scorer at all.

    Admission here is the union of two bounded slices:

    - **relevant** -- beliefs whose key, object, or claim_text contains one of
      the query's content tokens.
    - **recent** -- the most recently evidenced beliefs, so the context is
      never empty for a question nothing lexically matches.

    Both slices are bounded; the character budget downstream is what actually
    limits what reaches the prompt, and it is enforced independently.
    """
    if not is_memory_enabled(session, user_id):
        return []

    # A superseded row is exactly a closed interval, so a question about the
    # past must be able to reach it -- same rule list_active_beliefs follows.
    statuses = ("active", "superseded") if temporal_mode == "historical" else ("active",)

    tokens = content_tokens(query)
    relevant: list[Belief] = []
    if tokens and relevant_limit > 0:
        # Parameterised LIKE -- no string concatenation into SQL.
        pattern = or_(*[
            Belief.claim_text.ilike(f"%{_escape_like(token)}%", escape="\\")
            for token in sorted(tokens)
        ] + [
            Belief.object.ilike(f"%{_escape_like(token)}%", escape="\\")
            for token in sorted(tokens)
        ] + [
            Belief.key.ilike(f"%{_escape_like(token)}%", escape="\\")
            for token in sorted(tokens)
        ])
        conditions = [Belief.user_id == user_id, Belief.status.in_(statuses), pattern]
        if dimensions:
            conditions.append(Belief.dimension.in_(dimensions))
        relevant = list(session.scalars(
            select(Belief).where(*conditions)
            .order_by(desc(Belief.last_evidence_at))
            .limit(relevant_limit)
        ))

    recent: list[Belief] = []
    if recent_limit > 0:
        conditions = [Belief.user_id == user_id, Belief.status == "active"]
        if dimensions:
            conditions.append(Belief.dimension.in_(dimensions))
        recent = list(session.scalars(
            select(Belief).where(*conditions)
            .order_by(desc(Belief.last_evidence_at))
            .limit(recent_limit)
        ))

    # Union, preserving recency order and dropping duplicates by identity.
    seen: set[int] = set()
    merged: list[Belief] = []
    for belief in sorted(relevant + recent, key=_recency_key, reverse=True):
        if belief.id in seen:
            continue
        seen.add(belief.id)
        merged.append(belief)

    merged = [b for b in merged if not _is_expired(b)]
    merged = _filter_by_temporal_mode(merged, temporal_mode)
    merged = _keep_newest_per_single_slot(merged, temporal_mode, _SINGLE_VALUED_PREDICATES)
    return _filter_withdrawn(merged, temporal_mode)


def _recency_key(belief: Belief) -> tuple[int, datetime]:
    """Sort key that never mixes naive and aware datetimes.

    SQLite hands back naive datetimes while ``utcnow()`` is aware, so comparing
    them directly raises.  Everything is normalised to naive UTC, and a belief
    with no timestamp sorts below those that have one -- the leading flag keeps
    it from ever being compared against a datetime constant.
    """
    ts = belief.last_evidence_at
    if ts is None:
        return (0, datetime.min)
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC).replace(tzinfo=None)
    return (1, ts)


# Predicates that hold at most one value at a time.  Used at the read side to
# drop stale values regardless of which verb the extractor used for the update:
# "I am at Cedar Bank" and "I began working at Orion Foods" must not both
# render as current, even when their surface predicates never matched and the
# resolver therefore could not close the earlier one.
_SINGLE_VALUED_PREDICATES = frozenset({
    "lives_in", "works_at", "works_as", "profession", "age",
    "studies_at", "lives_at", "resides_in",
})


def _keep_newest_per_single_slot(
    beliefs: list[Belief], temporal_mode: str, single_valued: frozenset[str]
) -> list[Belief]:
    """For single-valued slots, keep only the newest active value.

    A "now?" query must not see two values of an attribute that holds one at a
    time.  The read-side rule closes the gap the write side cannot: extraction
    invents verbs ("at", "began working at") that never match the existing
    belief's predicate, so no UPDATE fires and the stale value stays active
    with an open interval.  History queries keep everything.
    """
    if temporal_mode == "historical":
        return beliefs
    newest: dict[str, tuple[int, datetime]] = {}
    for belief in beliefs:
        predicate = (belief.predicate or "").strip().lower()
        if predicate not in single_valued or belief.status != "active":
            continue
        ts = belief.last_evidence_at
        if ts is not None and ts.tzinfo is not None:
            ts = ts.astimezone(UTC).replace(tzinfo=None)
        ts = ts or datetime.min
        current = newest.get(predicate)
        if current is None or ts >= current[1]:
            newest[predicate] = (belief.id, ts)
    keep = {item[0] for item in newest.values()}
    if not keep:
        return beliefs
    return [
        b for b in beliefs
        if b.predicate.strip().lower() not in single_valued
        or b.status != "active"
        or b.id in keep
    ]


# Words that mark a preference or goal as ended.  A claim carrying one is a
# state termination (P0-2): it must never surface as current state, even
# though the belief row itself may still be active -- K1 rows have no
# predicate to supersede, so the write side cannot close them.
_WITHDRAWAL_MARKERS = (
    "avoided", "avoid", "stopped", "gave up", "dropped", "lost interest",
    "cancelled", "canceled", "no longer", "not want", "don't want",
    "do not want", "no longer want", "quit", "ended", "discontinued",
    "stopped enjoying", "don't like", "do not like", "don't enjoy",
)


def _filter_withdrawn(beliefs: list[Belief], temporal_mode: str) -> list[Belief]:
    """Drop claims that state a preference or goal as ended.

    Only current-state queries filter: a historical question is entitled to
    see that the user *used to* avoid spicy food.  The markers are matched
    against the claim text, since a K1 row has no predicate to key on.
    """
    if temporal_mode != "current":
        return beliefs
    kept = []
    for belief in beliefs:
        text = (belief.claim_text or "").lower()
        if any(marker in text for marker in _WITHDRAWAL_MARKERS):
            continue
        kept.append(belief)
    return kept


def _escape_like(token: str) -> str:
    """Escape LIKE wildcards so a token matches itself, not a pattern."""
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def beliefs_with_predicates(
    session: Session,
    user_id: str,
    predicates: set[str] | tuple[str, ...],
    *,
    candidate_objects: set[str] | tuple[str, ...] = (),
    limit: int = 500,
) -> list[Belief]:
    """Active beliefs whose predicate is in *predicates*.

    Identity resolution needs the beliefs a candidate could possibly match,
    and that set is determined by the predicate -- not by how recently the
    belief was touched.  The old code asked for the 200 most recent beliefs,
    so a user past 200 active beliefs silently lost the older predicates: the
    resolver could not see them and turned what should have been a SUPPORT or
    UPDATE into a CREATE.
    """
    wanted = {p for p in predicates if p}
    if not wanted or not is_memory_enabled(session, user_id):
        return []
    stmt = (
        select(Belief)
        .where(
            Belief.user_id == user_id,
            # Superseded rows are included on purpose: a return to a previous
            # value ("I moved back to Shanghai") is a revival of exactly such
            # a row, and the resolver cannot offer the revival path if the
            # history is filtered out here.
            Belief.status.in_(("active", "superseded")),
            or_(
                Belief.predicate.in_(wanted),
                # Same value under a different verb ("returned to Shanghai"
                # against a lives_in row) is the same attribute ...
                Belief.object.in_(candidate_objects),
                # ... and the values the candidate is returning *past* must
                # also be visible, because the revival has to close them.
                (
                    Belief.predicate.in_(
                        select(Belief.predicate).where(
                            Belief.user_id == user_id,
                            Belief.object.in_(candidate_objects),
                        )
                    )
                    if candidate_objects
                    else false()
                ),
            ),
        )
        .order_by(desc(Belief.last_evidence_at))
        .limit(limit)
    )
    results = list(session.scalars(stmt))
    return [b for b in results if not _is_expired(b)]


def _interval_is_current(belief: Belief, now: datetime) -> bool:
    """Does the belief's validity interval cover *now*?

    An open ``valid_to`` means "still true".  A NULL ``valid_from`` means
    "since always", so it never excludes a belief.
    """
    valid_from = belief.valid_from
    if valid_from is not None:
        if valid_from.tzinfo is None:
            valid_from = valid_from.replace(tzinfo=UTC)
        if now < valid_from:
            return False
    valid_to = belief.valid_to
    if valid_to is not None:
        if valid_to.tzinfo is None:
            valid_to = valid_to.replace(tzinfo=UTC)
        if now >= valid_to:
            return False
    return True


def list_rejected_beliefs(session: Session, user_id: str, *, limit: int = 5) -> list[Belief]:
    """Rejected beliefs, most-recently updated first."""
    if not is_memory_enabled(session, user_id):
        return []
    stmt = (
        select(Belief)
        .where(Belief.user_id == user_id, Belief.status == "rejected")
        .order_by(desc(Belief.updated_at))
        .limit(limit)
    )
    return list(session.scalars(stmt))


def get_belief_events(session: Session, user_id: str, belief_id: int, *, limit: int = 50) -> list[BeliefEvent]:
    stmt = (
        select(BeliefEvent)
        .where(BeliefEvent.user_id == user_id, BeliefEvent.belief_id == belief_id)
        .order_by(desc(BeliefEvent.id))
        .limit(limit)
    )
    return list(session.scalars(stmt))


# ---------------------------------------------------------------------------
# record_claim -- the main write path
# ---------------------------------------------------------------------------


def _event_key(base_key: str, claim_text: str) -> str:
    """Derive a unique event key by appending a content hash suffix.

    Life events with the same base key (e.g. ``"trip"``) but different
    content are distinct memories, not duplicates of one belief.
    """
    import hashlib

    digest = hashlib.sha256(claim_text.encode()).hexdigest()[:8]
    return f"{base_key}:{digest}"


def _is_expired(belief: Belief) -> bool:
    """Check if a life-event belief has expired (expires_at in the past)."""
    if belief.dimension != "event":
        return False
    val = safe_json(belief.value_json)
    expires_at = val.get("expires_at")
    if not expires_at:
        return False
    try:
        expiry = datetime.fromisoformat(expires_at)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return datetime.now(UTC) > expiry
    except (ValueError, TypeError):
        return False


def _is_elaboration(new_text: str, old_text: str | None) -> bool:
    """Is *new_text* a fuller statement of what *old_text* already says?

    Requires both that it is longer and that it does not drop the old text's
    distinctive words.  Length alone is not enough, and neither is sharing a
    generic word: "user" appears in nearly every claim, so matching on it let
    a longer sentence about a different aspect of the same belief key pass as
    an elaboration and throw away the wording that answered the question.

    Distinctive means six characters or more.  Below that, ordinary English
    words ("short", "likes", "works") look specific without carrying any
    content, and treating them as content is what made a generic placeholder
    block a legitimate update.  Real specifics -- charity, berlin, painting,
    insomnia -- are all longer.
    """
    if not new_text:
        return False
    if not old_text:
        return True
    if len(new_text) <= len(old_text):
        return False
    old_distinctive = content_tokens(old_text, min_length=6)
    if not old_distinctive:
        # Nothing specific to preserve; take the fuller wording.
        return True
    return bool(old_distinctive & content_tokens(new_text, min_length=6))


def record_claim(
    session: Session,
    user_id: str,
    *,
    dimension: str,
    key: str,
    claim_text: str,
    value: dict | None = None,
    relation: str = "supports",
    confidence: float = 0.0,
    layer: str = "L4",
    source: str = "extracted",
    session_id: str | None = None,
    evidence_message_ids: list[int] | None = None,
    stats_id: int | None = None,
    origin_slice_id: str | None = None,
    context_tags: list[str] | None = None,
    blocked_key_prefixes: tuple[str, ...] = (),
    allowed_dimensions: set[str] | None = None,
    # Cognitive triple fields (Phase 1: Memory Atom).
    subject: str = "user",
    predicate: str = "",
    object: str = "",
    cardinality: str = "multi",
    superseded_by: int | None = None,
) -> tuple[Belief | None, str]:
    """Write a claim, applying the merge strategy.

    Returns ``(belief, event_type)`` where *event_type* is one of:
    ``"created"``, ``"supported"``, ``"contradicted"``, ``"resurrection_guard"``,
    ``"profile_memory_disabled"``.

    Parameters
    ----------
    blocked_key_prefixes:
        Key prefixes that are structurally forbidden from persisting
        (e.g. ``("distortion.",)``).  The caller supplies this list so
        the repository layer stays domain-agnostic.
    allowed_dimensions:
        If provided, *dimension* must be in this set or a ``ValueError``
        is raised (strict closed-set validation).
    subject:
        Who the claim is about (default ``"user"``).
    predicate:
        The relationship verb (``likes``, ``lives_in``, etc.).
    object:
        What the predicate applies to (``coffee``, ``Shanghai``, etc.).
    cardinality:
        ``"single"`` / ``"multi"`` / ``"event"`` — controls identity resolution.
    superseded_by:
        If this claim supersedes an existing belief, set to the old belief's ID.
    """
    if not is_memory_enabled(session, user_id):
        return None, "profile_memory_disabled"

    if relation not in {"supports", "contradicts"}:
        raise ValueError(f"invalid claim relation: {relation!r}")

    if allowed_dimensions is not None and dimension not in allowed_dimensions:
        raise ValueError(
            f"dimension {dimension!r} not in allowed set {sorted(allowed_dimensions)}"
        )

    for prefix in blocked_key_prefixes:
        if key.startswith(prefix):
            raise ValueError(f"blocked key prefix {prefix!r} cannot be persisted: {key!r}")

    evidence = [int(mid) for mid in (evidence_message_ids or [])]

    # Extracted claims are always L4.
    if source == "extracted" and layer != "L4":
        logger.warning("Clamping non-L4 layer from extracted claim (layer=%s)", layer)
        layer = "L4"

    # Life events: derive a hash-suffixed key so distinct events don't collide,
    # and stamp expires_at so the renderer can filter stale events.
    # An occurrence time makes an event a distinct fact even when the key and
    # object are shared: "saw Dr. Patel on June 2" and "on June 16" must be
    # two rows, not one claim_text replaced by the other.  Hashing the claim
    # text into the key is what keeps them apart.
    has_occurrence_time = bool((value or {}).get("temporal"))
    if dimension == "event" or has_occurrence_time:
        key = _event_key(key, claim_text)
        if value is None:
            value = {}
        value.setdefault(
            "expires_at",
            (utcnow() + timedelta(days=LIFE_EVENT_EXPIRY_DAYS)).isoformat(),
        )

    existing = get_belief(session, user_id, key)
    if existing is not None and existing.status == "rejected":
        return None, "resurrection_guard"

    if existing is None:
        # New belief.
        merged_value = dict(value or {})
        if context_tags:
            existing_tags = merged_value.get("context_tags", [])
            merged_value["context_tags"] = sorted(set(existing_tags) | set(context_tags))
        belief = Belief(
            user_id=user_id,
            dimension=dimension,
            key=key,
            claim_text=claim_text,
            value_json=json.dumps(merged_value, ensure_ascii=False),
            layer=layer,
            status="active",
            confidence=min(CONFIDENCE_CEILING, max(0.0, confidence)),
            source=source,
            evidence_json=json.dumps(evidence[-MAX_EVIDENCE_REFS:]),
            # Cognitive triple fields.
            subject=subject,
            predicate=predicate,
            object=object,
            cardinality=cardinality,
            superseded_by=superseded_by,
            # Provenance.
            origin_stats_id=stats_id,
            origin_slice_id=origin_slice_id,
            origin_session_id=session_id,
            last_evidence_session_id=session_id,
        )
        session.add(belief)
        session.flush()
        _append_event(session, belief, "created", evidence=evidence, stats_id=stats_id)
        touch_memory_state(session, user_id)
        return belief, "created"

    if relation == "supports":
        # Merge supporting evidence.
        merged_evidence = json.loads(existing.evidence_json or "[]")
        for mid in evidence:
            if mid not in merged_evidence:
                merged_evidence.append(mid)
        existing.evidence_json = json.dumps(merged_evidence[-MAX_EVIDENCE_REFS:])
        existing.last_evidence_session_id = session_id
        existing.last_evidence_at = datetime.now(UTC)

        # Update claim_text when the new version is a genuine elaboration of
        # what is already stored -- later mentions often carry the date, name
        # or number an earlier one left out.  Replacing unconditionally with
        # "the longest so far" silently swapped in a longer text about a
        # different aspect of the same key, and the answer-bearing wording
        # vanished from the rendered surface.
        if claim_text and _is_elaboration(claim_text, existing.claim_text):
            existing.claim_text = claim_text

        if context_tags:
            val = safe_json(existing.value_json)
            old_tags = val.get("context_tags", [])
            val["context_tags"] = sorted(set(old_tags) | set(context_tags))
            existing.value_json = json.dumps(val, ensure_ascii=False)

        # Multi-factor confidence: base gain * 7-factor weight.
        now = datetime.now(UTC)
        last_ev = existing.last_evidence_at
        if last_ev.tzinfo is None:
            last_ev = last_ev.replace(tzinfo=UTC)
        days_since = max(0.0, (now - last_ev).total_seconds() / 86400)
        val = safe_json(existing.value_json)
        ctx_div = len(val.get("context_tags", []))
        events = get_belief_events(session, existing.user_id, existing.id)
        contradict_count = sum(1 for e in events if e.event_type in {"contradicted", "downgraded"})
        support_count = len(merged_evidence)

        weight = compute_confidence_weight(
            source=existing.source,
            layer=existing.layer,
            support_count=support_count,
            contradict_count=contradict_count,
            context_diversity=ctx_div,
            days_since_last_evidence=days_since,
            value_json=existing.value_json,
        )
        existing.confidence = min(CONFIDENCE_CEILING, existing.confidence + SUPPORT_GAIN * weight)
        _append_event(
            session,
            existing,
            "supported",
            evidence=evidence,
            detail={"confidence": existing.confidence},
            stats_id=stats_id,
        )
        touch_memory_state(session, user_id)
        return existing, "supported"

    # Contradicts: attenuate confidence, mark needs_clarification.
    from mirror_memory.core.constants import CONTRADICT_FACTOR

    existing.confidence = round(existing.confidence * CONTRADICT_FACTOR, 4)
    val = safe_json(existing.value_json)
    val["clarification_status"] = "needs_clarification"
    existing.value_json = json.dumps(val, ensure_ascii=False)
    detail: dict = {"confidence": existing.confidence, "clarification_status": "needs_clarification"}
    _append_event(session, existing, "contradicted", evidence=evidence, detail=detail, stats_id=stats_id)
    touch_memory_state(session, user_id)
    return existing, "contradicted"


# ---------------------------------------------------------------------------
# confirm / reject
# ---------------------------------------------------------------------------


def confirm_belief(session: Session, user_id: str, belief_id: int) -> Belief | None:
    """User confirmation: L4 -> L2 (questioning loop upgrade primitive)."""
    belief = session.get(Belief, belief_id)
    if belief is None or belief.user_id != user_id or belief.status != "active":
        return None
    belief.layer = "L2"
    belief.source = "user_confirmed"
    belief.confidence = max(belief.confidence, 0.9)
    _append_event(session, belief, "confirmed", evidence=[], detail={"layer": "L2"})
    touch_memory_state(session, user_id)
    return belief


def reject_belief(session: Session, user_id: str, belief_id: int) -> Belief | None:
    """User rejection: status -> rejected (same key never resurrected)."""
    belief = session.get(Belief, belief_id)
    if belief is None or belief.user_id != user_id or belief.status != "active":
        return None
    belief.status = "rejected"
    belief.confidence = 0.0
    _append_event(session, belief, "rejected", evidence=[], detail={"reason": "user_denied"})
    touch_memory_state(session, user_id)
    return belief


# ---------------------------------------------------------------------------
# ID-based operations (for IdentityResolver integration)
# ---------------------------------------------------------------------------


def support_belief_by_id(
    session: Session,
    belief_id: int,
    *,
    claim_text: str = "",
    confidence_gain: float | None = None,
    session_id: str | None = None,
    evidence_message_ids: list[int] | None = None,
) -> Belief | None:
    """Strengthen an existing belief by ID (SUPPORT action).

    Unlike ``record_claim(relation='supports')`` which matches by
    ``(user_id, key)``, this operates directly on a belief ID as
    determined by the IdentityResolver.
    """
    from mirror_memory.core.constants import SUPPORT_GAIN

    belief = session.get(Belief, belief_id)
    if belief is None or belief.status != "active":
        return None

    evidence = [int(mid) for mid in (evidence_message_ids or [])]
    merged_evidence = json.loads(belief.evidence_json or "[]")
    for mid in evidence:
        if mid not in merged_evidence:
            merged_evidence.append(mid)
    belief.evidence_json = json.dumps(merged_evidence[-MAX_EVIDENCE_REFS:])
    belief.last_evidence_session_id = session_id
    belief.last_evidence_at = datetime.now(UTC)

    # Same elaboration rule as record_claim: replace only when the new text is
    # a fuller statement of the same fact, never merely because it is longer.
    if claim_text and _is_elaboration(claim_text, belief.claim_text):
        belief.claim_text = claim_text

    # Confidence boost with 7-factor weighting (shared with record_claim).
    if confidence_gain is None:
        now = datetime.now(UTC)
        last_ev = belief.last_evidence_at
        if last_ev is not None and last_ev.tzinfo is None:
            last_ev = last_ev.replace(tzinfo=UTC)
        days_since = max(0.0, (now - last_ev).total_seconds() / 86400) if last_ev else 0.0
        val = safe_json(belief.value_json)
        ctx_div = len(val.get("context_tags", []))
        events = get_belief_events(session, belief.user_id, belief.id)
        contradict_count = sum(1 for e in events if e.event_type in {"contradicted", "downgraded"})
        weight = compute_confidence_weight(
            source=belief.source,
            layer=belief.layer,
            support_count=len(merged_evidence),
            contradict_count=contradict_count,
            context_diversity=ctx_div,
            days_since_last_evidence=days_since,
            value_json=belief.value_json,
        )
        confidence_gain = SUPPORT_GAIN * weight
    belief.confidence = min(CONFIDENCE_CEILING, belief.confidence + confidence_gain)
    session.flush()
    _append_event(session, belief, "supported", evidence=evidence)
    touch_memory_state(session, belief.user_id)
    return belief


def _derive_updated_key(old: Belief, new_object: str) -> str:
    """Pick the key for a belief whose object changed.

    ``(user_id, key)`` is unique regardless of status, and superseding is a
    status change rather than a delete -- so the new row can never reuse
    ``old.key``.  The base namespace is kept (``lives_in`` stays ``lives_in``)
    and the object becomes the suffix; when the object is unchanged the
    claim's own content hash disambiguates instead.
    """
    base = old.key.split(":", 1)[0]
    if new_object and new_object != old.object:
        return f"{base}:{new_object.strip().lower().replace(' ', '_')}"
    # Same or unknown object: the new row still needs its own key.
    return f"{base}:~{_short_hash(str(new_object) + str(old.id))}"


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:6]


def _target_key_for_update(old: Belief, new_key: str, new_object: str) -> str:
    """The key the superseding row should take.

    A caller-supplied ``new_key`` is normally respected, but not when it is the
    key the row being superseded still holds: ``(user_id, key)`` is unique
    regardless of status, so reusing it collides on insert.  That is exactly
    what happened when the extractor invented the same key for two different
    values of a single-cardinality predicate -- 19 LongMemEval users were lost
    to it.
    """
    target = new_key or old.key
    if target == old.key:
        # The UNIQUE constraint is on (user_id, key) with no status filter,
        # and superseding keeps the old row -- so the replacement must never
        # take the key the old row still holds.
        return _derive_updated_key(old, new_object)
    return target


def _disambiguate_key(session: Session, user_id: str, key: str) -> str:
    """Return a key that is free for *user_id*, suffixing if necessary.

    The UNIQUE constraint ignores ``status``, so a key taken by any row at all
    -- active, superseded, or rejected -- cannot be reused.  A superseding
    replacement therefore always gets a fresh key; the row it replaces keeps
    its own for historical lookup.
    """
    candidate = key
    suffix = 2
    while True:
        taken = session.scalar(
            select(Belief.id).where(
                Belief.user_id == user_id,
                Belief.key == candidate,
            )
        )
        if taken is None:
            return candidate
        candidate = f"{key}~{suffix}"
        suffix += 1


def update_belief_by_id(
    session: Session,
    old_belief_id: int,
    *,
    new_subject: str = "user",
    new_predicate: str = "",
    new_object: str = "",
    new_dimension: str = "",
    new_key: str = "",
    new_claim_text: str = "",
    new_confidence: float = 0.0,
    new_cardinality: str = "multi",
    session_id: str | None = None,
    evidence_message_ids: list[int] | None = None,
    new_value: dict | None = None,
    observed_at: datetime | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    temporal_scope: str = "",
) -> tuple[Belief | None, Belief | None]:
    """SINGLE cardinality update: supersede old belief, create new one.

    When the new fact carries a ``valid_from``, the old belief's interval is
    **closed** at that instant (``valid_to = valid_from``) rather than
    discarded.  That is what keeps "where did they live before?" answerable
    after "they live in Beijing now" arrives -- the old value becomes history
    instead of being deleted.

    If the target key already exists as a superseded/rejected belief
    (e.g. Shanghai → Beijing → Shanghai), the old row is revived instead
    of inserted, avoiding a (user_id, key) collision.

    Returns ``(old_belief, new_belief)``.  Old belief gets status='superseded'
    and superseded_by points to new belief's ID.
    """
    old = session.get(Belief, old_belief_id)
    if old is None:
        return None, None
    # A round-trip revival targets a *superseded* row -- that is what makes it
    # a revival.  Rejected rows stay untouchable (the resurrection guard is a
    # user decision), but superseded history may be reopened when the user
    # states the value again.
    if old.status == "rejected":
        return None, None

    evidence = [int(mid) for mid in (evidence_message_ids or [])]
    new_from = _as_utc(valid_from)
    target_key = _target_key_for_update(old, new_key, new_object)

    # Close the old interval when the new fact starts a new period.  Without
    # this the superseded row stays "current" forever and temporal questions
    # cannot tell history from the present.
    if new_from is not None and old.valid_to is None:
        old.valid_to = new_from

    # Check for an existing superseded belief with the same key (revival).
    existing_row = (
        session.query(Belief)
        .filter(
            Belief.user_id == old.user_id,
            Belief.key == target_key,
            Belief.status == "superseded",
        )
        .first()
    )

    if existing_row:
        # Revive: reactivate the old row, supersede current.
        existing_row.status = "active"
        existing_row.superseded_by = None
        existing_row.confidence = min(CONFIDENCE_CEILING, max(0.0, new_confidence))
        existing_row.last_evidence_at = datetime.now(UTC)
        existing_row.last_evidence_session_id = session_id
        existing_row.evidence_json = json.dumps(evidence[-MAX_EVIDENCE_REFS:])
        existing_row.valid_from = new_from or existing_row.valid_from
        existing_row.valid_to = _as_utc(valid_to)
        existing_row.observed_at = observed_at or utcnow()
        if temporal_scope:
            existing_row.temporal_scope = temporal_scope
        if new_claim_text:
            existing_row.claim_text = new_claim_text
        if new_value:
            existing_row.value_json = json.dumps(new_value, ensure_ascii=False)
        session.flush()

        old.status = "superseded"
        old.superseded_by = existing_row.id
        session.flush()

        _append_event(session, old, "superseded", evidence=evidence,
                      detail={"superseded_by": existing_row.id})
        _append_event(session, existing_row, "created", evidence=evidence,
                      detail={"supersedes": old.id, "revived": True})
        touch_memory_state(session, old.user_id)
        return old, existing_row

    # Normal path: create new belief.  The key must be free for this user
    # regardless of status -- the UNIQUE constraint does not care that the
    # row currently holding it is about to be superseded.
    # `old` still holds its key in the database at insert time -- superseding
    # is a status change, not a delete -- so the UNIQUE constraint will fire
    # on any new row that reuses it.  exclude_id must therefore NOT skip
    # `old`; the key has to be free of every row, including the one being
    # replaced.
    target_key = _disambiguate_key(session, old.user_id, target_key)
    new_belief = Belief(
        user_id=old.user_id,
        dimension=new_dimension or old.dimension,
        key=target_key,
        claim_text=new_claim_text,
        value_json=json.dumps(new_value or {}, ensure_ascii=False),
        layer="L4",
        status="active",
        confidence=min(CONFIDENCE_CEILING, max(0.0, new_confidence)),
        source="extracted",
        evidence_json=json.dumps(evidence[-MAX_EVIDENCE_REFS:]),
        subject=new_subject,
        predicate=new_predicate or old.predicate,
        object=new_object,
        cardinality=new_cardinality,
        origin_session_id=session_id,
        last_evidence_session_id=session_id,
        observed_at=observed_at or utcnow(),
        valid_from=new_from,
        valid_to=_as_utc(valid_to),
        temporal_scope=temporal_scope or old.temporal_scope,
    )
    session.add(new_belief)
    session.flush()

    old.status = "superseded"
    old.superseded_by = new_belief.id
    session.flush()

    _append_event(session, old, "superseded", evidence=evidence,
                  detail={"superseded_by": new_belief.id, "valid_to": _iso(old.valid_to)})
    _append_event(session, new_belief, "created", evidence=evidence,
                  detail={"supersedes": old.id, "valid_from": _iso(new_belief.valid_from)})
    touch_memory_state(session, old.user_id)
    return old, new_belief


def _as_utc(value) -> datetime | None:
    """Normalise a datetime to naive UTC for consistent storage and comparison.

    Extraction hands back ISO strings from the LLM, so anything that is not
    already a datetime is coerced rather than rejected: a string reaching here
    used to raise AttributeError and abort the whole UPDATE, which left the
    superseded value in place and the new one never created.
    """
    value = coerce_datetime(value)
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def update_belief_value(
    session: Session,
    user_id: str,
    key: str,
    value: dict,
    *,
    stats_id: int | None = None,
    evidence: list[int] | None = None,
) -> Belief | None:
    """Update a belief's structured value (value_json), preserving the
    previous value in the event log.
    """
    belief = get_belief(session, user_id, key)
    if belief is None or belief.status != "active":
        return None
    previous = belief.value_json
    belief.value_json = json.dumps(value, ensure_ascii=False)
    _append_event(
        session,
        belief,
        "value_updated",
        evidence=evidence or [],
        detail={"previous_value": previous, "value": value},
        stats_id=stats_id,
    )
    return belief


# ---------------------------------------------------------------------------
# Question candidates
# ---------------------------------------------------------------------------


def _question_confirm_rate_weight(rate: float) -> float:
    """Confirm-rate [0,1] -> weight [MIN, MAX] linear map; rate=0.5 -> 1.0."""
    clamped = min(max(rate, 0.0), 1.0)
    weight = QUESTION_TIER_WEIGHT_MIN + (QUESTION_TIER_WEIGHT_MAX - QUESTION_TIER_WEIGHT_MIN) * clamped
    return round(weight, 4)


def _needs_clarification_boost(belief: Belief) -> int:
    """Needs-clarification beliefs get priority in the question queue."""
    val = safe_json(belief.value_json)
    return 1 if val.get("clarification_status") == "needs_clarification" else 0


def list_question_candidates(
    session: Session,
    user_id: str,
    *,
    limit: int = 1,
    question_value_tiers: dict[str, int] | None = None,
) -> list[Belief]:
    """Question-candidate beliefs: L4, confidence >= threshold, cross-session
    evidence.

    Sorting = (clarification boost, tier * learn_weight, confidence, recency).

    Parameters
    ----------
    question_value_tiers:
        Mapping of dimension -> int priority tier.  Dimensions absent from
        the map fall back to ``DEFAULT_QUESTION_TIER``.
    """
    if not is_memory_enabled(session, user_id):
        return []

    tiers = question_value_tiers or {}

    conditions = (
        Belief.user_id == user_id,
        Belief.status == "active",
        Belief.layer == "L4",
    )
    rows = list(session.scalars(select(Belief).where(*conditions)))

    # Filter: confidence >= threshold AND cross-session evidence.
    pending = [
        b
        for b in rows
        if b.confidence >= L4_QUESTION_THRESHOLD
        and (
            not b.origin_session_id
            or not b.last_evidence_session_id
            or b.origin_session_id != b.last_evidence_session_id
        )
    ]

    pending.sort(
        key=lambda b: (
            _needs_clarification_boost(b),
            tiers.get(b.dimension, DEFAULT_QUESTION_TIER) * _question_confirm_rate_weight(
                safe_json(b.value_json).get("confirm_rate", 0.5)
            ),
            b.confidence,
            b.last_evidence_at,
        ),
        reverse=True,
    )
    return pending[:limit]


# ---------------------------------------------------------------------------
# Intervention events (verification loop)
# ---------------------------------------------------------------------------

QUESTION_INJECTED_KIND = "question_injected"
QUESTION_ANSWERED_KIND = "question_answered"


def record_intervention_event(
    session: Session,
    user_id: str,
    session_id: str,
    *,
    kind: str,
    detail: dict | None = None,
) -> InterventionEvent | None:
    """Record a question injection or answer event.

    Only action metadata is stored (belief label/key, verdict) -- never the
    conversation content.  Returns ``None`` when memory is disabled for the
    user.
    """
    if not is_memory_enabled(session, user_id):
        return None
    row = InterventionEvent(
        user_id=user_id,
        session_id=session_id,
        kind=kind,
        detail_json=json.dumps(detail or {}, ensure_ascii=False),
    )
    session.add(row)
    session.flush()
    return row


def get_pending_verification(
    session: Session,
    user_id: str,
    session_id: str,
    *,
    max_window_turns: int = 3,
) -> InterventionEvent | None:
    """Get the pending question injection within the N-turn window.

    Returns the InterventionEvent with kind='question_injected' that:
    - is the most recent injection for the user,
    - has NOT been answered yet (no later ``question_answered`` event),
    - is within the last N sessions (approximate turn window): the injection
      was made in the current session or in one of the last ``N-1`` sessions
      with recorded activity.  Intervention events are the generic
      session-activity proxy (this library has no message table).

    Returns None if no pending verification exists.
    """
    injection = session.scalar(
        select(InterventionEvent)
        .where(InterventionEvent.user_id == user_id, InterventionEvent.kind == QUESTION_INJECTED_KIND)
        .order_by(desc(InterventionEvent.id))
        .limit(1)
    )
    if injection is None:
        return None

    answered = session.scalar(
        select(InterventionEvent)
        .where(InterventionEvent.user_id == user_id, InterventionEvent.kind == QUESTION_ANSWERED_KIND)
        .order_by(desc(InterventionEvent.id))
        .limit(1)
    )
    if answered is not None and answered.id > injection.id:
        return None

    if injection.session_id == session_id:
        return injection

    if max_window_turns <= 0:
        return None

    # Collect the most recent session ids from the event stream; the current
    # session counts as the first entry of the window.
    window: list[str] = [session_id]
    seen = {session_id}
    for sid in session.scalars(
        select(InterventionEvent.session_id)
        .where(InterventionEvent.user_id == user_id)
        .order_by(desc(InterventionEvent.id))
    ):
        if len(window) >= max_window_turns:
            break
        if sid not in seen:
            seen.add(sid)
            window.append(sid)
    return injection if injection.session_id in seen else None


def has_unanswered_injection(session: Session, user_id: str) -> bool:
    """Check if there's an unanswered question injection (prevents double-inject).

    Only one hypothesis may be pending at a time: when the latest
    ``question_injected`` has no later ``question_answered``, injecting a new
    question would shadow the previous one and the verification loop could
    never close.
    """
    last_injected = session.scalar(
        select(InterventionEvent)
        .where(InterventionEvent.user_id == user_id, InterventionEvent.kind == QUESTION_INJECTED_KIND)
        .order_by(desc(InterventionEvent.id))
        .limit(1)
    )
    if last_injected is None:
        return False
    last_answered = session.scalar(
        select(InterventionEvent)
        .where(InterventionEvent.user_id == user_id, InterventionEvent.kind == QUESTION_ANSWERED_KIND)
        .order_by(desc(InterventionEvent.id))
        .limit(1)
    )
    return last_answered is None or last_answered.id < last_injected.id


# ---------------------------------------------------------------------------
# Extraction stats
# ---------------------------------------------------------------------------


def record_extraction_stats(
    session: Session,
    user_id: str,
    *,
    session_id: str | None = None,
    trigger: str = "",
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    claims_out: int = 0,
    status: str = "ok",
    error: str = "",
) -> ExtractionStats:
    """Record one extraction-call statistics row."""
    row = ExtractionStats(
        user_id=user_id,
        session_id=session_id,
        trigger=trigger,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        claims_out=claims_out,
        status=status,
        error=error,
    )
    session.add(row)
    session.flush()
    return row


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def delete_user_memories(session: Session, user_id: str) -> dict[str, int]:
    """Cascade-delete all memory data for a user.

    Deletes beliefs, events, stats, snapshots, intervention events,
    evolution jobs, consent grants, and session summaries.

    Returns counts of deleted rows per table.
    """
    # Order matters: child tables first.
    events_deleted = (
        session.query(BeliefEvent)
        .filter(BeliefEvent.user_id == user_id)
        .delete(synchronize_session=False)
    )
    interventions_deleted = (
        session.query(InterventionEvent)
        .filter(InterventionEvent.user_id == user_id)
        .delete(synchronize_session=False)
    )
    beliefs_deleted = (
        session.query(Belief).filter(Belief.user_id == user_id).delete(synchronize_session=False)
    )
    # Evidence links hang off belief ids, so they go with the beliefs.
    evidence_ids = [
        row[0]
        for row in session.execute(
            select(Evidence.id).where(Evidence.user_id == user_id)
        ).all()
    ]
    links_deleted = 0
    if evidence_ids:
        links_deleted = (
            session.query(BeliefEvidenceLink)
            .filter(BeliefEvidenceLink.evidence_id.in_(evidence_ids))
            .delete(synchronize_session=False)
        )
    evidence_deleted = (
        session.query(Evidence).filter(Evidence.user_id == user_id).delete(synchronize_session=False)
    )
    stats_deleted = (
        session.query(ExtractionStats)
        .filter(ExtractionStats.user_id == user_id)
        .delete(synchronize_session=False)
    )
    jobs_deleted = (
        session.query(EvolutionJob)
        .filter(EvolutionJob.user_id == user_id)
        .delete(synchronize_session=False)
    )
    snapshots_deleted = (
        session.query(Snapshot).filter(Snapshot.user_id == user_id).delete(synchronize_session=False)
    )
    consent_deleted = (
        session.query(ConsentGrant)
        .filter(ConsentGrant.user_id == user_id)
        .delete(synchronize_session=False)
    )
    summaries_deleted = (
        session.query(SessionSummary)
        .filter(SessionSummary.user_id == user_id)
        .delete(synchronize_session=False)
    )
    touch_memory_state(session, user_id)
    return {
        "beliefs": int(beliefs_deleted),
        "belief_events": int(events_deleted),
        "extraction_stats": int(stats_deleted),
        "evolution_jobs": int(jobs_deleted),
        "snapshots": int(snapshots_deleted),
        "intervention_events": int(interventions_deleted),
        "consent_grants": int(consent_deleted),
        "session_summaries": int(summaries_deleted),
        "evidence": int(evidence_deleted),
        "belief_evidence_links": int(links_deleted),
    }


def forget_belief(session: Session, user_id: str, belief_id: int) -> bool:
    """Targeted forget: delete a single belief and its event history.

    Also invalidates derived state: snapshots, pending evolution jobs,
    and bumps the state revision to block stale worker writes.

    Returns ``True`` if the belief was found and deleted.
    """
    belief = session.get(Belief, belief_id)
    if belief is None or belief.user_id != user_id:
        return False

    # Delete events first.
    session.query(BeliefEvent).filter(BeliefEvent.belief_id == belief_id).delete(
        synchronize_session=False
    )
    # Drop this belief's edges into the evidence graph.
    session.query(BeliefEvidenceLink).filter(
        BeliefEvidenceLink.belief_id == belief_id
    ).delete(synchronize_session=False)
    session.delete(belief)
    session.flush()
    # Evidence no belief points at is garbage, not provenance.
    _prune_orphan_evidence(session, user_id)

    # Invalidate derived state: source memory was deleted, so
    # snapshots (derived understanding) and pending evolution jobs
    # are no longer valid.
    session.query(Snapshot).filter(Snapshot.user_id == user_id).delete(synchronize_session=False)
    session.query(EvolutionJob).filter(
        EvolutionJob.user_id == user_id,
        EvolutionJob.status.in_(("pending", "running")),
    ).update(
        {"status": "cancelled", "error_code": "belief_forgotten"},
        synchronize_session=False,
    )
    touch_memory_state(session, user_id)
    return True


def forget_session_summary(session: Session, user_id: str, session_id: str) -> bool:
    """Targeted forget: delete a session summary by session_id.

    Returns ``True`` if the summary was found and deleted.
    """
    summary = (
        session.query(SessionSummary)
        .filter(
            SessionSummary.user_id == user_id,
            SessionSummary.session_id == session_id,
        )
        .first()
    )
    if summary is None:
        return False
    session.delete(summary)
    session.flush()
    return True


# ---------------------------------------------------------------------------
# Correct + Provenance
# ---------------------------------------------------------------------------


def correct_belief(
    session: Session,
    user_id: str,
    belief_id: int,
    *,
    new_claim_text: str,
    correction_note: str = "",
    new_predicate: str | None = None,
    new_object: str | None = None,
    new_value: dict | None = None,
) -> Belief | None:
    """User-initiated correction: supersede old belief with corrected version.

    Unlike a normal UPDATE (which is driven by the IdentityResolver),
    ``correct_belief`` is an explicit user action: "this memory is wrong,
    it should be X."

    The old belief is superseded and a correction event is logged.
    The new belief carries the corrected claim_text and optionally
    updated predicate, object, and value_json.

    Returns the new (corrected) belief, or ``None`` if the target is invalid.
    """
    old = session.get(Belief, belief_id)
    if old is None or old.user_id != user_id or old.status != "active":
        return None

    # For structured triple beliefs, correction must update all fields.
    has_triple = getattr(old, "predicate", "") and getattr(old, "object", "")
    if has_triple and new_object is None:
        logger.warning("correct_belief: structured belief requires new_object; refusing inconsistent correction")
        return None

    # Save before state for the event log.
    before = {
        "claim_text": old.claim_text,
        "predicate": getattr(old, "predicate", ""),
        "object": getattr(old, "object", ""),
        "value_json": old.value_json,
    }

    # In-place correction: update claim_text and mark as user_corrected.
    # The UNIQUE constraint on (user_id, key) prevents creating a second
    # row with the same key, so we update the existing one and log the
    # correction as a BeliefEvent for provenance.
    old.claim_text = new_claim_text
    old.source = "user_corrected"
    old.confidence = min(CONFIDENCE_CEILING, max(old.confidence, 0.5))
    old.last_evidence_at = datetime.now(UTC)

    # Update cognitive triple fields if provided.
    if new_predicate is not None:
        old.predicate = new_predicate
    if new_object is not None:
        old.object = new_object
    if new_value is not None:
        old.value_json = json.dumps(new_value, ensure_ascii=False)

    session.flush()

    after = {
        "claim_text": old.claim_text,
        "predicate": getattr(old, "predicate", ""),
        "object": getattr(old, "object", ""),
        "value_json": old.value_json,
    }
    detail: dict = {"before": before, "after": after, "correction": new_claim_text[:200]}
    if correction_note:
        detail["note"] = correction_note
    _append_event(session, old, "corrected", evidence=[], detail=detail)
    touch_memory_state(session, user_id)
    return old


def explain_belief(session: Session, user_id: str, belief_id: int) -> dict | None:
    """Return the provenance chain for a belief.

    Returns a dict with:
    - belief: current state
    - events: chronological event history
    - source_evidence: evidence IDs
    - evidence_graph: typed ``(evidence, relation)`` links
    - superseded_by: ID of the belief that replaced this one (if any)
    - correction_of: ID of the belief this one corrects (if any)

    Returns ``None`` if the belief is not found.
    """
    belief = session.get(Belief, belief_id)
    if belief is None or belief.user_id != user_id:
        return None

    events = get_belief_events(session, user_id, belief_id)
    return {
        "belief_id": belief.id,
        "dimension": belief.dimension,
        "key": belief.key,
        "claim_text": belief.claim_text,
        "predicate": getattr(belief, "predicate", ""),
        "object": getattr(belief, "object", ""),
        "confidence": belief.confidence,
        "layer": belief.layer,
        "status": belief.status,
        "source": belief.source,
        "subject": getattr(belief, "subject", "user"),
        "evidence_ids": belief_evidence_ids(belief),
        "evidence_graph": evidence_summary(session, user_id, belief_id),
        "superseded_by": belief.superseded_by,
        "first_seen_at": belief.first_seen_at.isoformat() if belief.first_seen_at else None,
        "last_evidence_at": belief.last_evidence_at.isoformat() if belief.last_evidence_at else None,
        "events": [
            {
                "event_type": e.event_type,
                "detail": safe_json(e.detail_json),
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    }


def store_session_summary(
    session: Session,
    user_id: str,
    session_id: str,
    text: str,
    turn_count: int,
    *,
    max_length: int = 3000,
    snippet_max: int = 500,
) -> None:
    """Append a conversation snippet to the session summary.

    Preserves specific details (dates, names, numbers) that structured
    beliefs may lose.  Creates a new row on first call, appends on
    subsequent calls.
    """
    from mirror_memory.core.models import SessionSummary

    snippet = text.strip()[:snippet_max]

    existing = (
        session.query(SessionSummary)
        .filter(
            SessionSummary.user_id == user_id,
            SessionSummary.session_id == session_id,
        )
        .first()
    )
    if existing:
        combined = existing.summary_text + " | " + snippet
        if len(combined) > max_length:
            combined = combined[-max_length:]
        existing.summary_text = combined
        existing.turn_count = turn_count
    else:
        session.add(SessionSummary(
            user_id=user_id,
            session_id=session_id,
            summary_text=snippet,
            turn_count=turn_count,
        ))
    session.flush()
