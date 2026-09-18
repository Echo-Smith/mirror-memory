"""K2 LLM semantic extraction -- domain-agnostic.

Allowed keys and system prompt are loaded from ``config``.  The LLM
client is injected via ``config.llm_client``.  Zero domain coupling.
"""

from __future__ import annotations

import logging
import time

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import CLAIM_TEXT_MAX_LENGTH, MAX_CLAIMS_PER_TURN

logger = logging.getLogger(__name__)


def _build_allowed_keys(config: MemoryConfig) -> dict[str, frozenset[str]]:
    """Build dimension -> allowed keys mapping from config anchors."""
    allowed: dict[str, set[str]] = {}
    for anchor in config.anchors:
        if anchor.identify_only:
            continue
        allowed.setdefault(anchor.dimension, set()).add(anchor.key)
    return {dim: frozenset(keys) for dim, keys in allowed.items()}


def _anchor_prompt_lines(config: MemoryConfig) -> list[str]:
    """Build the allowed-keys section for the extraction prompt."""
    lines: list[str] = []
    for anchor in config.anchors:
        if anchor.identify_only:
            continue
        # Find a display label for this anchor
        label = ""
        for lbl in config.display_labels:
            if lbl.key == anchor.key and lbl.dimension == anchor.dimension:
                label = lbl.zh or lbl.en or ""
                break
        sample = anchor.phrases[0] if anchor.phrases else ""
        lines.append(f"{anchor.dimension} {anchor.key} = {label} | e.g. {sample}")
    return lines


def _parse_extraction_json(raw: str, allowed_keys: dict[str, frozenset[str]]) -> list[dict]:
    """Parse LLM output into validated claim dicts.

    Validation rules:
    - dimension/key must be in allowed_keys (closed set).
    - relation must be "supports" or "contradicts".
    - confidence clamped to [0, 1].
    - Maximum 3 claims.
    """
    from mirror_memory.core.utils import parse_llm_json

    data = parse_llm_json(raw, expect_array=True)
    if data is None:
        return []

    # Accept both {"claims": [...]} and bare [...] formats.
    if isinstance(data, list):
        raw_claims = data
    elif isinstance(data, dict):
        raw_claims = data.get("claims")
    else:
        raw_claims = None
    if not isinstance(raw_claims, list):
        return []

    validated: list[dict] = []
    for item in raw_claims:
        if not isinstance(item, dict) or len(validated) >= MAX_CLAIMS_PER_TURN:
            break
        dimension = str(item.get("dimension") or "")
        key = str(item.get("key") or "")
        relation = str(item.get("relation") or "supports")
        # Dimension must be one of the configured dimensions (strict).
        # Key can be any non-empty string within a valid dimension (lenient --
        # LLM may extract novel keys beyond the anchor vocabulary).
        if dimension not in allowed_keys or not key:
            continue
        # Accept "new" as equivalent to "supports" (LLM often uses "new" for first-time facts).
        if relation in {"new", "first_mention"}:
            relation = "supports"
        if relation not in {"supports", "contradicts"}:
            continue
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        validated.append({
            "dimension": dimension,
            "key": key,
            "claim_text": str(item.get("claim_text") or item.get("claim_zh") or "")[:CLAIM_TEXT_MAX_LENGTH],
            "confidence": confidence,
            "relation": relation,
            "source": "extracted",
            "value": {"via": "llm_semantic"},
        })
    return validated


class SemanticExtractor:
    """K2 LLM semantic extractor.

    Parameters
    ----------
    config:
        The ``MemoryConfig`` providing prompts, anchors, and the LLM
        client.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._allowed_keys = _build_allowed_keys(config)

    def extract(
        self,
        text: str,
        session: object | None,
        user_id: str,
        config: MemoryConfig,
    ) -> list[dict]:
        """Run LLM semantic extraction on *text*.

        Parameters
        ----------
        text:
            The user's current message.
        session:
            Database session (for reading current beliefs, if needed).
        user_id:
            The user identifier.
        config:
            The ``MemoryConfig`` (may differ from constructor config if
            the caller overrides at call time).

        Returns
        -------
        list[dict]
            Validated claim dicts (up to 3).
        """
        llm = config.llm_client
        if llm is None:
            logger.debug("semantic: no LLM client configured, skipping")
            return []

        system_prompt = config.prompts.k2_system
        if not system_prompt:
            logger.warning("semantic: k2_system prompt is empty, skipping")
            return []

        # Build payload: allowed keys + user text
        anchor_lines = _anchor_prompt_lines(config)
        payload = (
            "[Allowed keys]\n"
            + "\n".join(anchor_lines)
            + "\n\n[User utterance]\n"
            + (text or "").strip()
        )

        started = time.monotonic()
        try:
            raw = llm.generate(
                system_prompt=system_prompt,
                payload_text=payload,
                fallback=lambda: '{"claims": []}',
            )
        except Exception:
            logger.warning("semantic: LLM call failed, returning empty claims")
            return []

        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info("semantic: LLM call completed in %dms", elapsed_ms)

        claims = _parse_extraction_json(raw, self._allowed_keys)
        logger.info("semantic: extracted %d claims", len(claims))
        return claims
