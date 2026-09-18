"""Snapshot management -- persist and promote shadow snapshots.

Domain-agnostic.  Uses the ORM models from ``core.models`` directly.
"""

from __future__ import annotations

import json
import logging
import uuid

from mirror_memory.core.models import Belief, Snapshot
from mirror_memory.core.utils import truncate_id

logger = logging.getLogger(__name__)


def persist_snapshot(
    session: object,
    user_id: str,
    content: dict,
    policy: dict,
    watermark: str,
    *,
    shadow: bool = False,
) -> Snapshot:
    """Save a versioned snapshot for *user_id*.

    Old active/shadow snapshots are retired.  The new snapshot is
    ``"shadow"`` if *shadow* is ``True`` (single-session evidence),
    otherwise ``"active"``.

    Parameters
    ----------
    session:
        SQLAlchemy database session.
    user_id:
        The user identifier.
    content:
        Structured understanding dict.
    policy:
        Support policy dict.
    watermark:
        Evidence watermark string for idempotency.
    shadow:
        If ``True``, the snapshot starts as ``"shadow"`` and must be
        promoted later.

    Returns
    -------
    Snapshot
        The newly created snapshot row.
    """
    # Compute new version number.
    latest = (
        session.query(Snapshot)
        .filter(Snapshot.user_id == user_id)
        .order_by(Snapshot.version.desc())
        .limit(1)
        .first()
    )
    new_version = (latest.version + 1) if latest else 1

    # Retire old active/shadow snapshots.
    session.query(Snapshot).filter(
        Snapshot.user_id == user_id,
        Snapshot.status.in_(("active", "shadow")),
    ).update({"status": "retired"}, synchronize_session=False)

    snapshot = Snapshot(
        id=uuid.uuid4().hex[:16],
        user_id=user_id,
        version=new_version,
        status="shadow" if shadow else "active",
        content_json=json.dumps(content, ensure_ascii=False),
        support_policy_json=json.dumps(policy, ensure_ascii=False),
        evidence_watermark=watermark,
        prompt_version="v1",
    )
    session.add(snapshot)
    session.flush()
    logger.info(
        "snapshot: persisted user=%s version=%d status=%s",
        truncate_id(user_id),
        new_version,
        snapshot.status,
    )
    return snapshot


def promote_shadow_if_ready(session: object, user_id: str) -> bool:
    """Promote the latest shadow snapshot to active if evidence
    spans multiple sessions.

    Returns ``True`` if promotion happened, ``False`` otherwise.
    """
    latest_snap = (
        session.query(Snapshot)
        .filter(Snapshot.user_id == user_id, Snapshot.status == "shadow")
        .order_by(Snapshot.version.desc())
        .limit(1)
        .first()
    )
    if not latest_snap:
        return False

    distinct_sessions = (
        session.query(Belief.origin_session_id)
        .filter(Belief.user_id == user_id, Belief.status == "active")
        .distinct()
        .count()
    )
    if distinct_sessions >= 2:
        latest_snap.status = "active"
        session.flush()
        logger.info(
            "snapshot: shadow promoted user=%s version=%d",
            truncate_id(user_id),
            latest_snap.version,
        )
        return True
    return False
