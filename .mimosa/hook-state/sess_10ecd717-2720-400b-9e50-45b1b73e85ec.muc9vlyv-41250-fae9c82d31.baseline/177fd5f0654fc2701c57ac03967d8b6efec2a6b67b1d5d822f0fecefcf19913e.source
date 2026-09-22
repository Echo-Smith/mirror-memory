"""K1 deterministic extraction -- keyword and regex matching.

Zero domain coupling.  All keywords and patterns are loaded from
``config.extraction``.  Returns standardised claim dicts that the
pipeline can persist via ``record_claim``.
"""

from __future__ import annotations

import logging
import re

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import (
    CLAIM_TEXT_MAX_LENGTH,
    KEYWORD_CONFIDENCE,
    PATTERN_CONFIDENCE,
)

logger = logging.getLogger(__name__)


def extract_claims(text: str, config: MemoryConfig) -> list[dict]:
    """Extract claims from *text* using keyword and regex matching.

    Parameters
    ----------
    text:
        The user's message text.
    config:
        The ``MemoryConfig`` providing ``extraction.keywords`` (mapping
        of ``category -> list[str]``) and ``extraction.patterns`` (list
        of ``PatternRule`` with compiled regex, dimension, key).

    Returns
    -------
    list[dict]
        Each dict has keys: ``dimension``, ``key``, ``claim_text``,
        ``confidence``, ``relation``, ``source``, ``value``.
    """
    if not text or not text.strip():
        return []

    lowered = text.lower()
    claims: list[dict] = []

    # -- Keyword matching -------------------------------------------------------
    # Truncate text for claim storage (keep it meaningful but bounded).
    snippet = text.strip()[:CLAIM_TEXT_MAX_LENGTH]
    for category, keywords in config.extraction.keywords.items():
        for kw in keywords:
            if kw.lower() in lowered:
                parts = category.split(".", 1)
                dimension = parts[0] if len(parts) > 1 else "topic"
                # Content-specific key: "category:keyword" instead of generic category.
                base_key = parts[1] if len(parts) > 1 else parts[0]
                key = f"{base_key}:{kw.replace(' ', '_')}"
                claims.append({
                    "dimension": dimension,
                    "key": key,
                    "claim_text": snippet,
                    "confidence": KEYWORD_CONFIDENCE,
                    "relation": "supports",
                    "source": "extracted",
                    "value": {"via": "keyword", "keyword": kw},
                })
                break  # one claim per category per turn

    # -- Regex pattern matching -------------------------------------------------
    for pattern_rule in config.extraction.patterns:
        try:
            m = re.search(pattern_rule.regex, text)
            if m:
                import hashlib

                match_text = m.group(0)[:CLAIM_TEXT_MAX_LENGTH // 2] if m.group(0) else snippet
                # Content-specific key: "pattern_key:content_hash".
                content_hash = hashlib.sha256(match_text.encode()).hexdigest()[:6]
                key = f"{pattern_rule.key}:{content_hash}"
                claims.append({
                    "dimension": pattern_rule.dimension,
                    "key": key,
                    "claim_text": match_text,
                    "confidence": PATTERN_CONFIDENCE,
                    "relation": "supports",
                    "source": "extracted",
                    "value": {"via": "pattern", "match": match_text},
                })
        except re.error:
            logger.warning("Invalid regex in config: %s", pattern_rule.regex)

    return claims
