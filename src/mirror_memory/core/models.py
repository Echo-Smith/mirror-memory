"""Mirror Memory ORM models (SQLAlchemy 2.0 mapped_column style).

All tables use the ``mm_`` prefix.  Dimensions are free-form strings
(not hard-coded D1-D8).  Domain-specific models (intervention events,
domain profile fields) are intentionally excluded.
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
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
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
    origin_stats_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    origin_slice_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    origin_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_evidence_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_evidence_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


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
