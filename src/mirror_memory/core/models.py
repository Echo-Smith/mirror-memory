"""Mirror Memory ORM models (SQLAlchemy 2.0 mapped_column style).

All tables use the ``mm_`` prefix.  Dimensions are free-form strings
(not hard-coded D1-D8).  Domain-specific profile fields are intentionally
excluded; the verification-loop event log (:class:`InterventionEvent`) is
generic and ships with the engine.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base shared by all Mirror Memory models."""

    pass


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------


class User(Base):
    """Minimal user record.  Extend in your application layer as needed."""

    __tablename__ = "mm_users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Memory preference
# ---------------------------------------------------------------------------


class MemoryPreference(Base):
    """User-owned switch for long-term memory.

    A missing row means *enabled* so existing accounts keep their current
    behaviour.  Disabling pauses collection and use without deleting data;
    deletion is a separate explicit action.
    """

    __tablename__ = "mm_memory_preferences"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    state_revision: Mapped[int] = mapped_column(Integer, default=1)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ConsentGrant(Base):
    """Feature-level consent grant (finer-grained than MemoryPreference).

    A missing row means the feature inherits the global MemoryPreference
    setting.  Feature names are free-form strings (e.g. ``"extraction"``,
    ``"recall"``, ``"verification"``, ``"panel"``).
    """

    __tablename__ = "mm_consent_grants"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    feature: Mapped[str] = mapped_column(String(64), primary_key=True)
    granted: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------------------
# Belief
# ---------------------------------------------------------------------------


class Belief(Base):
    """A single belief row -- the "current state" of one inferred claim.

    Full migration history lives in :class:`BeliefEvent` (append-only,
    auditable).  Deleting a user cascades across both tables.

    Structural invariants (enforced in code, not in the DB):
    - ``(user_id, key)`` is unique.
    - ``source="extracted"`` rows are forced to ``layer="L4"`` by the
      repository; ``L2`` can only come from user confirmation.
    - ``status="rejected"`` rows are never resurrected (resurrection guard).
    """

    __tablename__ = "mm_beliefs"
    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_mm_beliefs_user_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    dimension: Mapped[str] = mapped_column(String(64), index=True)
    key: Mapped[str] = mapped_column(String(128))
    claim_text: Mapped[str] = mapped_column(Text, default="")
    value_json: Mapped[str] = mapped_column(Text, default="{}")
    layer: Mapped[str] = mapped_column(String(8), default="L4")
    status: Mapped[str] = mapped_column(String(16), default="active")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32), default="extracted")
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    # Cognitive triple fields (Phase 1: Memory Atom)
    subject: Mapped[str] = mapped_column(String(64), default="user")
    predicate: Mapped[str] = mapped_column(String(64), default="")
    object: Mapped[str] = mapped_column(String(256), default="")
    cardinality: Mapped[str] = mapped_column(String(16), default="multi")
    superseded_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Temporal validity (Temporal V2).  ``observed_at`` is when the engine
    # learned the fact; ``valid_from``/``valid_to`` delimit when the fact
    # itself was true.  A NULL ``valid_to`` means "still true" -- which is
    # what makes "where do they live now?" answerable separately from
    # "where did they live before?".
    observed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    temporal_scope: Mapped[str] = mapped_column(String(16), default="current_state")
    # The predicate exactly as the extractor emitted it, kept alongside the
    # canonical ``predicate``.  Canonicalisation folds many surface forms onto
    # one slot ("was_hired_by", "joined", "started_at" -> works_at); without
    # the raw form there is no way to audit which spelling produced a belief,
    # or to notice an extractor inventing verbs the synonym map has never
    # seen.
    raw_predicate: Mapped[str] = mapped_column(String(64), default="")
    # Structured polarity: positive / negative / neutral.  A claim and its
    # withdrawal about the same object are two rows with opposite polarity,
    # and only one of them may be current -- that is what makes "I liked X"
    # then "I avoid X" answerable as a state change rather than a word-list
    # guess at read time.
    polarity: Mapped[str] = mapped_column(String(16), default="neutral")
    # Goal lifecycle: active / paused / cancelled / resumed.  Empty for
    # non-goal predicates.  A cancelled goal is closed, not deleted, so
    # "I gave up" followed by "I started again" is a resumption of the same
    # goal rather than a new one.
    lifecycle_state: Mapped[str] = mapped_column(String(16), default="")
    # Existing provenance fields
    origin_stats_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    origin_slice_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    origin_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_evidence_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_evidence_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    # Memory metabolism (lifecycle runtime).  These are orthogonal to the
    # truth state above: "superseded + warm" is a perfectly good combination
    # ("no longer the current value, but historically true and plausibly
    # useful soon"), and mixing the two dimensions is exactly the confusion
    # the split prevents.  The tier is changed only by maintenance proposals
    # committed through the Publisher -- never by the read path.
    memory_tier: Mapped[str] = mapped_column(String(16), default="hot")
    metabolism_state: Mapped[str] = mapped_column(String(16), default="active")
    retention_class: Mapped[str] = mapped_column(String(16), default="preference")
    importance_score: Mapped[float] = mapped_column(Float, default=0.5)
    # Access telemetry: how often recall has actually surfaced this belief.
    # Recording an access is not a state transition -- it deliberately does
    # not bump the state revision, so a recall can never invalidate an
    # in-flight proposal.
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_supported_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Why this belief is exempt from automatic cooling/compaction (see
    # mirror_memory.metabolism.protection); empty when it is not.
    protected_reason: Mapped[str] = mapped_column(String(64), default="")
    # EvidenceDigest id this belief's raw evidence was folded into (PR3).
    compacted_into: Mapped[int | None] = mapped_column(Integer, nullable=True)


# ---------------------------------------------------------------------------
# Evidence (first-class provenance entity)
# ---------------------------------------------------------------------------


class Evidence(Base):
    """A first-class piece of provenance a belief was derived from.

    Before this table, a belief carried a bare list of message ids in
    ``evidence_json`` -- enough to say *which* messages, but nothing about
    *what kind* of source they were, who said it, or how it was extracted.
    Evidence as its own row carries that, so the same message can support one
    belief and contradict another, and so "where did this belief come from"
    is answerable without re-parsing the belief.

    ``ref`` is the caller's own identifier for the source (a message id, a
    document path, an event id).  It is unique per user, so re-observing the
    same message yields the same Evidence row rather than a duplicate.
    """

    __tablename__ = "mm_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True)
    ref: Mapped[str] = mapped_column(String(128), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    # What kind of source this is: "message" / "session" / "document" / "system".
    source_type: Mapped[str] = mapped_column(String(32), default="message")
    # How the belief was pulled out of it: "k1_keyword" / "k1_regex" /
    # "k2_llm" / "k3_synthesis" / "user_confirmed" / "user_corrected".
    extraction_method: Mapped[str] = mapped_column(String(32), default="")
    # Who the statement is attributable to: "user" / "assistant" / "system" /
    # "derived".  Drives how much weight the evidence can carry.
    authority: Mapped[str] = mapped_column(String(32), default="user")
    observed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # Memory metabolism (PR3).  ``hot`` rows are live provenance;
    # ``representative`` rows are the sample kept after compaction;
    # ``compacted`` rows were folded into an EvidenceDigest (their content
    # is cleared — the digest carries the counts, the representatives carry
    # the wording); ``archived`` is the PR4 cold store.
    retention_state: Mapped[str] = mapped_column(String(16), default="hot")
    # Which digest group folded this row (EvidenceDigest.id as string).
    compaction_group_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (UniqueConstraint("user_id", "ref", name="uq_mm_evidence_user_ref"),)


# ---------------------------------------------------------------------------
# Evidence digest (compaction aggregate)
# ---------------------------------------------------------------------------


class EvidenceDigest(Base):
    """The aggregate that survives evidence compaction.

    Fifty "I like coffee" messages are one fact with fifty observations.
    Compaction folds the redundant bulk into this row — counts by relation,
    time span, source/authority distributions, and the surviving
    representative sample — so the hot store carries ~6 rows plus one
    digest while the conclusion ("likes coffee, 48 observations, never
    contradicted") stays exactly as answerable.

    One digest per belief (upserted across repeated compaction runs; counts
    always reflect the full link graph, not just the latest run's fold).
    """

    __tablename__ = "mm_evidence_digests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    belief_id: Mapped[int] = mapped_column(Integer, index=True, unique=True)
    # Total typed-link counts across the belief's whole history — including
    # representatives and already-folded rows.
    support_count: Mapped[int] = mapped_column(Integer, default=0)
    contradict_count: Mapped[int] = mapped_column(Integer, default=0)
    verify_count: Mapped[int] = mapped_column(Integer, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # JSON list of the evidence ids kept as the surviving sample.
    representative_ids: Mapped[str] = mapped_column(Text, default="[]")
    # Rule-generated one-liner (no model): "48 observations since 2025-03,
    # sources: message, authority: user".
    summary: Mapped[str] = mapped_column(Text, default="")
    source_distribution: Mapped[str] = mapped_column(Text, default="{}")
    authority_distribution: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------------------
# Belief <-> Evidence link (typed relation, not a blob of ids)
# ---------------------------------------------------------------------------


class BeliefEvidenceLink(Base):
    """A typed edge between a belief and one piece of evidence.

    The relation is what makes this more than a join table: the same evidence
    can ``support`` one belief and ``contradict`` another, which is the
    primitive conflict resolution needs.  A belief no longer owns a list of
    message ids -- it owns the set of edges that point at it.

    relation values: ``support`` / ``contradict`` / ``verify`` / ``correct``.
    """

    __tablename__ = "mm_belief_evidence_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    belief_id: Mapped[int] = mapped_column(Integer, index=True)
    evidence_id: Mapped[int] = mapped_column(Integer, index=True)
    relation: Mapped[str] = mapped_column(String(16), default="support")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("belief_id", "evidence_id", "relation",
                         name="uq_mm_belief_evidence_link"),
    )


# ---------------------------------------------------------------------------
# Belief event (append-only audit log)
# ---------------------------------------------------------------------------


class BeliefEvent(Base):
    """Append-only event log for a belief.

    event_type values: created / supported / contradicted / downgraded /
    confirmed / rejected / value_updated.  Only inserted, never updated.
    """

    __tablename__ = "mm_belief_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    belief_id: Mapped[int] = mapped_column(Integer, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    detail_json: Mapped[str] = mapped_column(Text, default="{}")
    origin_stats_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Memory transition (metabolism audit log)
# ---------------------------------------------------------------------------


class MemoryTransition(Base):
    """Audit row for one committed storage-tier transition.

    Written only by the Publisher (via ``set_belief_tier``) when a
    TIER_TRANSITION proposal commits.  This is what makes "why was this
    memory archived?" answerable months later: the decision's reason,
    heat score, and resulting revision are kept alongside the change.

    Truth-state events (created/supported/...) live in
    :class:`BeliefEvent`; storage-tier events live here.  Two orthogonal
    state dimensions, two logs.
    """

    __tablename__ = "mm_memory_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(16), default="belief")
    entity_id: Mapped[int] = mapped_column(Integer, index=True)
    from_tier: Mapped[str] = mapped_column(String(16))
    to_tier: Mapped[str] = mapped_column(String(16))
    # Why the transition fired: "cooling", "reheat", "manual", ...
    reason: Mapped[str] = mapped_column(String(64), default="")
    # The heat score the decision was computed from.
    score: Mapped[float] = mapped_column(Float, default=0.0)
    # Log-safe proposal description (proposals carry no id of their own).
    proposal_desc: Mapped[str] = mapped_column(String(128), default="")
    # State revision at commit time.  Note this is the *observed* revision,
    # not one the transition caused -- tier moves deliberately leave the
    # revision alone (storage state, not truth state).
    publisher_revision: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Deletion tombstone (forget audit + generation barrier)
# ---------------------------------------------------------------------------


class DeletionTombstone(Base):
    """The record that a deletion happened — deliberately content-free.

    A forget is a deletion *transaction*: the belief, its events, evidence
    links, digest, and tier-audit rows go in one commit, and this row is
    what survives it.  It carries no claim text, no evidence content, and
    no key — only a hash of the deletion scope, so "was this thing
    deleted?" is answerable while "what did it say?" is not.

    ``generation`` increments per user on every deletion.  It is the
    generation half of the revision barrier: a stale worker holding a
    pre-deletion revision cannot write the data back, and a re-observation
    of the same message creates *new* rows rather than resurrecting the
    deleted ones.
    """

    __tablename__ = "mm_deletion_tombstones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    generation: Mapped[int] = mapped_column(Integer, default=1)
    # "belief" / "session" / "user" — what kind of scope was deleted.
    scope: Mapped[str] = mapped_column(String(16), default="belief")
    # Hash of the scope identity (never the content itself).
    scope_hash: Mapped[str] = mapped_column(String(64), default="")
    # Per-table deleted row counts — structure, not content.
    counts_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Extraction stats
# ---------------------------------------------------------------------------


class ExtractionStats(Base):
    """Per-extraction-call statistics (one row per call, including skips)."""

    __tablename__ = "mm_extraction_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trigger: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    claims_out: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default="ok")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


# ---------------------------------------------------------------------------
# Snapshot (versioned, rollback-capable)
# ---------------------------------------------------------------------------


class Snapshot(Base):
    """Versioned snapshot of a user's structured understanding."""

    __tablename__ = "mm_snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="shadow")
    content_json: Mapped[str] = mapped_column(Text, default="{}")
    support_policy_json: Mapped[str] = mapped_column(Text, default="{}")
    evidence_watermark: Mapped[str] = mapped_column(Text, default="{}")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Evolution job (async synthesis)
# ---------------------------------------------------------------------------


class EvolutionJob(Base):
    """Async per-user synthesis job.  Idempotent, single-user mutex."""

    __tablename__ = "mm_evolution_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    evidence_watermark: Mapped[str] = mapped_column(Text, default="{}")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)
    claimed_revision: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Session summary (unstructured fallback for factual recall)
# ---------------------------------------------------------------------------


class SessionSummary(Base):
    """Per-session text summary for fallback retrieval.

    Structured beliefs capture "understanding the person"; session summaries
    preserve specific facts (dates, names, numbers) that beliefs may lose.
    When recall finds no highly relevant beliefs, session summaries are
    searched as a fallback.
    """

    __tablename__ = "mm_session_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    summary_text: Mapped[str] = mapped_column(Text, default="")
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Intervention event (verification loop)
# ---------------------------------------------------------------------------


class InterventionEvent(Base):
    """Intervention event for verification loop (question injection/answer).

    kind values: ``question_injected`` (a verification question was surfaced
    to the user; detail carries ``belief_label`` / ``belief_key``) and
    ``question_answered`` (the reply was judged; detail carries ``verdict``).
    Only inserted, never updated -- the pending state is derived from the
    event sequence, not from a mutable flag.
    """

    __tablename__ = "mm_intervention_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32))  # "question_injected" / "question_answered"
    detail_json: Mapped[str] = mapped_column(Text, default="{}")  # belief_label, belief_key, verdict
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
