"""K3 synthesis -- build structured understanding from beliefs.

Domain-agnostic.  The system prompt is loaded from ``config.prompts.k3_system``.
Zero psychology-specific terminology in engine code.

K3 is a **Compute** node: it reads beliefs and summaries and returns a
structured understanding dict.  It never writes to the database -- persisting
that understanding is the Publisher's job (see ``worker.evolution``), which
alone may commit a snapshot and which does so only after the revision guard.
"""

from __future__ import annotations

import logging

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.utils import truncate_id

logger = logging.getLogger(__name__)

# Default max beliefs and summaries to feed into K3.
_MAX_BELIEFS = 30
_MAX_SUMMARIES = 5


def _format_beliefs(beliefs: list) -> str:
    """Format beliefs into a text block for the K3 prompt."""
    lines: list[str] = []
    for b in beliefs:
        if b.confidence < 0.3:
            continue
        lines.append(
            f"- [{b.dimension}] {b.key} "
            f"(confidence={b.confidence:.2f}, layer={b.layer}): {b.claim_text}"
        )
    return "\n".join(lines)


def _build_k3_payload(
    beliefs_text: str,
    slice_summaries: str,
    checkin_trend: str,
    current_understanding: str,
) -> str:
    """Assemble the user-content payload for K3."""
    parts = [
        "[Observations -- beliefs with confidence and evidence]",
        beliefs_text or "(no beliefs yet)",
        "",
        "[Recent conversation summaries]",
        slice_summaries or "(no summaries yet)",
        "",
        "[Checkin trends]",
        checkin_trend or "(no checkin data)",
        "",
        "[Current understanding -- update or replace as needed]",
        current_understanding or "(no prior understanding)",
    ]
    return "\n".join(parts)


class Synthesizer:
    """K3 synthesis engine.

    Parameters
    ----------
    config:
        The ``MemoryConfig`` providing the K3 system prompt and LLM
        client.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config

    def synthesize(
        self,
        session: object,
        user_id: str,
        config: MemoryConfig,
    ) -> dict | None:
        """Run K3 synthesis for *user_id*.

        Reads active beliefs, slice summaries, and check-in trends from
        *session*, then calls the LLM to produce a structured
        understanding dict.

        Parameters
        ----------
        session:
            SQLAlchemy database session.  **Read-only** -- this method
            performs no writes and issues no commits.
        user_id:
            The user identifier.
        config:
            The ``MemoryConfig`` (may differ from constructor config).

        Returns
        -------
        dict or None
            The structured understanding, or ``None`` if data is
            insufficient or the LLM call fails.  Persisting the result is
            the caller's responsibility.
        """
        # Lazy imports to avoid circular dependencies.
        from mirror_memory.core.repository import list_active_beliefs

        llm = config.llm_client
        if llm is None:
            logger.debug("synthesis: no LLM client, skipping")
            return None

        system_prompt = config.prompts.k3_system
        if not system_prompt:
            logger.warning("synthesis: k3_system prompt is empty, skipping")
            return None

        beliefs = list_active_beliefs(session, user_id, limit=_MAX_BELIEFS)
        if len(beliefs) < 2:
            logger.debug("synthesis: insufficient beliefs (%d), skipping", len(beliefs))
            return None

        beliefs_text = _format_beliefs(beliefs)
        if not beliefs_text:
            return None

        # Slice summaries are optional.  Checkin trends are not implemented
        # in the base engine (no check-in model defined) -- callers that
        # have check-in data can override _build_k3_payload.
        slice_summaries = _get_slice_summaries(session, user_id)
        current = _get_current_understanding(session, user_id)

        user_content = _build_k3_payload(
            beliefs_text, slice_summaries, "", current
        )

        try:
            raw = llm.generate(
                system_prompt=system_prompt,
                payload_text=user_content,
            )
        except Exception:
            logger.warning("synthesis: LLM call failed for user %s", truncate_id(user_id))
            return None

        return _parse_understanding(raw)


def _parse_understanding(raw: str) -> dict | None:
    """Parse the LLM output into a structured understanding dict."""
    from mirror_memory.core.utils import parse_llm_json

    result = parse_llm_json(raw)
    return result if isinstance(result, dict) else None


def _get_slice_summaries(session: object, user_id: str) -> str:
    """Get recent slice summaries as text.  Returns empty string if
    the session does not provide the required model."""
    try:
        from mirror_memory.core.models import Snapshot

        rows = (
            session.query(Snapshot)
            .filter(Snapshot.user_id == user_id)
            .order_by(Snapshot.created_at.desc())
            .limit(_MAX_SUMMARIES)
            .all()
        )
        lines = []
        for r in rows:
            lines.append(f"- {r.created_at.strftime('%Y-%m-%d') if r.created_at else '?'}: {r.content_json[:200]}")
        return "\n".join(lines)
    except Exception:
        return ""


def _get_current_understanding(session: object, user_id: str) -> str:
    """Read the user's current understanding JSON from the latest
    active snapshot."""
    try:
        from mirror_memory.core.models import Snapshot

        snap = (
            session.query(Snapshot)
            .filter(Snapshot.user_id == user_id, Snapshot.status.in_(("active", "shadow")))
            .order_by(Snapshot.version.desc())
            .limit(1)
            .first()
        )
        if snap and snap.content_json and snap.content_json != "{}":
            return snap.content_json
    except Exception:
        pass
    return ""
