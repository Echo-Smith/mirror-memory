"""Async evolution worker -- domain-agnostic.

Six-node pipeline:
1. load_evidence    -- read active beliefs, background, session count
2. formulate        -- one LLM call for candidate understanding
3. validate         -- deterministic checks (forbidden labels, third-party)
4. compile_policy   -- compile support policy from candidate
5. synthesize       -- K3 understanding synthesis
6. persist_snapshot  -- save versioned snapshot

All domain-specific content (forbidden labels, third-party markers,
system prompts) comes from ``config``.  Zero domain coupling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import (
    FORMULATE_MIN_CONFIDENCE,
    JOB_BATCH_LIMIT,
    LOAD_EVIDENCE_LIMIT,
)
from mirror_memory.core.utils import truncate_id
from mirror_memory.core.models import Belief, EvolutionJob, Snapshot
from mirror_memory.worker.snapshot import persist_snapshot, promote_shadow_if_ready

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job enqueue
# ---------------------------------------------------------------------------


def enqueue_job(session: object, user_id: str, config: MemoryConfig) -> EvolutionJob | None:
    """Write or merge an evolution job for *user_id*.

    Idempotent: same user + same evidence watermark produces at most
    one pending job.  Returns the job row, or ``None`` if the user
    has no active beliefs.
    """
    from mirror_memory.core.repository import is_memory_enabled

    if not is_memory_enabled(session, user_id):
        return None

    watermark = _current_watermark(session, user_id)
    idempotency_key = f"{user_id}:{watermark}:{config.worker.prompt_version}"

    # Check for existing pending/running job.
    existing = (
        session.query(EvolutionJob)
        .filter(
            EvolutionJob.user_id == user_id,
            EvolutionJob.status.in_(("pending", "running")),
        )
        .first()
    )
    if existing:
        existing.evidence_watermark = json.dumps({"latest_update": watermark})
        existing.idempotency_key = idempotency_key
        session.commit()
        return existing

    # Promote shadow if ready.
    promote_shadow_if_ready(session, user_id)

    # Reuse stale job with same idempotency key.
    stale = (
        session.query(EvolutionJob)
        .filter(
            EvolutionJob.user_id == user_id,
            EvolutionJob.idempotency_key == idempotency_key,
            EvolutionJob.status.in_(("completed", "failed")),
        )
        .first()
    )
    if stale:
        stale.status = "pending"
        stale.attempt_count = 0
        stale.error_code = ""
        stale.started_at = None
        stale.completed_at = None
        stale.evidence_watermark = json.dumps({"latest_update": watermark})
        session.commit()
        return stale

    job = EvolutionJob(
        id=uuid.uuid4().hex[:16],
        user_id=user_id,
        evidence_watermark=json.dumps({"latest_update": watermark}),
        prompt_version=config.worker.prompt_version,
        idempotency_key=idempotency_key,
        status="pending",
    )
    session.add(job)
    session.commit()
    return job


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_pending(session_factory: object, config: MemoryConfig) -> None:
    """Process all pending evolution jobs.

    Parameters
    ----------
    session_factory:
        Callable returning a new SQLAlchemy session.
    config:
        The ``MemoryConfig``.
    """
    session = session_factory() if callable(session_factory) else session_factory

    try:
        # Crash recovery: expire stale running jobs.
        lease_cutoff = datetime.now(UTC) - timedelta(seconds=config.worker.job_lease_seconds)
        session.query(EvolutionJob).filter(
            EvolutionJob.status == "running",
            EvolutionJob.started_at < lease_cutoff,
        ).update({"status": "pending"}, synchronize_session=False)
        session.commit()

        jobs = (
            session.query(EvolutionJob)
            .filter(EvolutionJob.status == "pending")
            .order_by(EvolutionJob.created_at)
            .limit(JOB_BATCH_LIMIT)
            .all()
        )
        if not jobs:
            return

        for job in jobs:
            # Atomic claim: only pending jobs can be claimed.
            updated = (
                session.query(EvolutionJob)
                .filter(
                    EvolutionJob.id == job.id,
                    EvolutionJob.status == "pending",
                )
                .update(
                    {
                        EvolutionJob.status: "running",
                        EvolutionJob.started_at: datetime.now(UTC),
                        EvolutionJob.attempt_count: EvolutionJob.attempt_count + 1,
                    },
                    synchronize_session=False,
                )
            )
            session.commit()
            if updated == 0:
                continue  # claimed by another instance
            _process_single_job(session, job, config)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Single job processing (6-node pipeline)
# ---------------------------------------------------------------------------


def _process_single_job(
    session: object,
    job: EvolutionJob,
    config: MemoryConfig,
) -> None:
    """Process a single evolution job through the 6-node pipeline."""
    try:
        from mirror_memory.core.repository import get_state_revision
        # We intentionally ignore job.claimed_revision here (a snapshot from
        # enqueue time).  The revision may have been bumped by state mutations
        # between enqueue and compute, so we re-read the current value to
        # compare against the post-compute revision at Node 6.
        claimed_revision = get_state_revision(session, job.user_id)

        # Node 1: load_evidence
        evidence = _load_evidence(session, job.user_id)

        # Node 2: formulate (LLM)
        candidate = _formulate(session, job.user_id, evidence, config)

        # Node 3: validate
        validated = _validate(candidate, evidence, config)

        # Node 4: compile_policy
        policy = _compile_policy(validated)

        # Node 5: synthesize (K3 -- fail-open)
        try:
            from mirror_memory.extraction.synthesis import Synthesizer

            synthesizer = Synthesizer(config)
            synthesizer.synthesize(session, job.user_id, config)
        except Exception:
            logger.warning("evolution: K3 synthesis skipped in worker (job=%s)", truncate_id(job.id))

        # Node 6: persist_snapshot with revision guard
        from mirror_memory.core.repository import get_state_revision
        current_revision = get_state_revision(session, job.user_id)
        if current_revision != claimed_revision:
            logger.warning("evolution: stale write blocked (job=%s claimed=%d current=%d)",
                           truncate_id(job.id), claimed_revision, current_revision)
            job.status = "completed"  # mark done but don't persist
            job.completed_at = datetime.now(UTC)
            job.error_code = "stale_revision"
            session.commit()
            return

        is_shadow = evidence.get("distinct_sessions", 1) < 2
        persist_snapshot(
            session,
            job.user_id,
            validated,
            policy,
            job.evidence_watermark,
            shadow=is_shadow,
        )

        job.status = "completed"
        job.completed_at = datetime.now(UTC)
        job.error_code = ""
        session.commit()
        logger.info("evolution: completed job=%s", truncate_id(job.id))

    except Exception as exc:
        job.status = "failed" if job.attempt_count >= config.worker.max_attempts else "pending"
        job.error_code = type(exc).__name__
        session.commit()
        logger.warning("evolution: failed job=%s error=%s", truncate_id(job.id), type(exc).__name__)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _current_watermark(session: object, user_id: str) -> str:
    """Get the latest belief update timestamp as watermark."""
    latest = (
        session.query(Belief.updated_at)
        .filter(Belief.user_id == user_id, Belief.status == "active")
        .order_by(Belief.updated_at.desc())
        .limit(1)
        .scalar()
    )
    return latest.isoformat() if latest else "none"


def _load_evidence(session: object, user_id: str) -> dict:
    """Node 1: Read active beliefs, background, and session count."""
    beliefs = (
        session.query(Belief)
        .filter(Belief.user_id == user_id, Belief.status == "active")
        .order_by(Belief.last_evidence_at.desc())
        .limit(LOAD_EVIDENCE_LIMIT)
        .all()
    )
    belief_data = []
    for b in beliefs:
        belief_data.append({
            "dimension": b.dimension,
            "key": b.key,
            "claim_text": b.claim_text,
            "confidence": b.confidence,
            "layer": b.layer,
            "value": b.value_json,
        })

    distinct_sessions = (
        session.query(Belief.origin_session_id)
        .filter(Belief.user_id == user_id, Belief.status == "active")
        .distinct()
        .count()
    )

    return {
        "beliefs": belief_data,
        "user_id": user_id,
        "distinct_sessions": distinct_sessions,
    }


def _formulate(
    session: object,
    user_id: str,
    evidence: dict,
    config: MemoryConfig,
) -> dict:
    """Node 2: One LLM call for candidate understanding."""
    llm = config.llm_client
    if llm is None:
        return {"patterns": [], "support_policy": {}}

    system_prompt = config.prompts.formulate
    if not system_prompt:
        return {"patterns": [], "support_policy": {}}

    beliefs_text = "\n".join(
        f"- [{b['dimension']}] {b['key']} (confidence={b['confidence']:.2f}): {b['claim_text']}"
        for b in evidence["beliefs"]
        if b["confidence"] >= FORMULATE_MIN_CONFIDENCE
    )

    user_content = f"[Beliefs]\n{beliefs_text or '(none yet)'}"
    try:
        raw = llm.generate(
            system_prompt=system_prompt,
            payload_text=user_content,
        )
    except Exception:
        logger.warning("evolution: formulate LLM call failed")
        return {"patterns": [], "support_policy": {}}

    return _parse_formulate_json(raw)


def _parse_formulate_json(raw: str) -> dict:
    """Parse the formulate LLM output."""
    from mirror_memory.core.utils import parse_llm_json

    result = parse_llm_json(raw)
    if isinstance(result, dict):
        return result
    return {"patterns": [], "support_policy": {}}


def _validate(
    candidate: dict,
    evidence: dict | None,
    config: MemoryConfig,
) -> dict:
    """Node 3: Deterministic validation.

    Checks:
    - Forbidden labels (from config.worker.forbidden_labels)
    - Third-party attribution (from config.worker.third_party_markers)
    - Evidence existence
    - Contradiction forcing needs_verification
    """
    patterns = candidate.get("patterns", [])
    validated = []

    forbidden = {label.lower() for label in config.worker.forbidden_labels}
    third_party = {marker.lower() for marker in config.worker.third_party_markers}

    # Build evidence index.
    belief_keys: set[str] = set()
    contradicted_keys: set[str] = set()
    distinct_sessions = 1
    if evidence and "beliefs" in evidence:
        for b in evidence["beliefs"]:
            belief_keys.add(b.get("key", ""))
            # Counter-evidence: needs_clarification marks a contradicted belief.
            try:
                val = json.loads(b.get("value", "{}") or "{}")
                if val.get("clarification_status") == "needs_clarification":
                    contradicted_keys.add(b.get("key", ""))
            except (TypeError, ValueError):
                pass
        distinct_sessions = evidence.get("distinct_sessions", 1)

    for p in patterns:
        desc = p.get("description", "")
        desc_lower = desc.lower()

        # Forbidden labels
        if any(label in desc_lower for label in forbidden):
            continue
        # Empty description
        if not desc.strip():
            continue
        # Third-party attribution
        if any(marker in desc_lower for marker in third_party):
            continue
        # Evidence existence check
        p_keys = set(p.get("evidence_keys") or [])
        if p_keys and not p_keys.issubset(belief_keys):
            continue
        # Counter-evidence: any contradicted belief forces needs_verification
        if contradicted_keys:
            p["needs_verification"] = True
        # Single-session evidence forces needs_verification
        if distinct_sessions < 2:
            p["needs_verification"] = True
        validated.append(p)

    candidate["patterns"] = validated[:3]
    return candidate


_DEFAULT_POLICY = {
    "response_length": "normal",
    "pacing": "validate_before_suggestions",
    "question_budget": 2,
    "avoid_topics": [],
    "preferred_knowledge_paths": [],
}


def _compile_policy(candidate: dict) -> dict:
    """Node 4: Compile support policy from candidate."""
    raw_policy = candidate.get("support_policy", {})
    policy = dict(_DEFAULT_POLICY)
    if isinstance(raw_policy, dict):
        if raw_policy.get("response_length") in ("brief", "normal"):
            policy["response_length"] = raw_policy["response_length"]
        if raw_policy.get("pacing"):
            policy["pacing"] = raw_policy["pacing"]
    return policy


# ---------------------------------------------------------------------------
# Async worker entry point
# ---------------------------------------------------------------------------


async def evolution_worker(
    stop: asyncio.Event,
    config: MemoryConfig,
) -> None:
    """Persistent background worker that polls for pending jobs.

    Parameters
    ----------
    stop:
        An ``asyncio.Event`` that, when set, signals the worker to
        stop after the current poll cycle.
    config:
        The ``MemoryConfig`` providing session factory and worker
        parameters.
    """
    session_factory = config.session_factory
    poll_interval = config.worker.poll_interval_seconds

    while not stop.is_set():
        try:
            await asyncio.to_thread(process_pending, session_factory, config)
        except Exception:
            logger.warning("evolution: worker will retry on next tick", exc_info=True)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_interval)
