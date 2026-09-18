"""Value-driven throttle for LLM semantic extraction (K2).

Decides whether the current turn warrants an LLM extraction call.
All thresholds come from ``config.extraction`` -- zero domain coupling.
"""

from __future__ import annotations

import logging

from mirror_memory.config.schema import MemoryConfig

logger = logging.getLogger(__name__)


def _count_keyword_hits(text: str, config: MemoryConfig) -> int:
    """Count how many distinct keyword categories are hit in *text*."""
    if not text:
        return 0
    lowered = text.lower()
    hits = 0
    for _category, keywords in config.extraction.keywords.items():
        if any(kw.lower() in lowered for kw in keywords):
            hits += 1
    return hits


def should_extract(
    text: str,
    turn_count: int,
    config: MemoryConfig,
) -> bool:
    """Decide whether the current turn warrants LLM extraction.

    Rules (from config):
    - If keyword hits >= ``llm_min_keyword_hits``: always extract.
    - Otherwise: extract every ``llm_every_turns`` turns when there is
      at least one keyword hit.

    Parameters
    ----------
    text:
        The current user message.
    turn_count:
        The number of the current turn (0-indexed).
    config:
        The ``MemoryConfig`` providing throttle parameters.

    Returns
    -------
    bool
        ``True`` if LLM extraction should fire.
    """
    keyword_hits = _count_keyword_hits(text, config)
    min_hits = config.extraction.llm_min_keyword_hits
    every_n = max(1, config.extraction.llm_every_turns)

    # min_hits=0 means "always extract" (useful for benchmarks).
    if min_hits == 0:
        logger.info("throttle: min_hits=0 -> always extract")
        return True

    if keyword_hits >= min_hits:
        logger.info("throttle: keyword_hits=%d >= min=%d -> extract", keyword_hits, min_hits)
        return True

    if keyword_hits >= 1 and turn_count % every_n == 0:
        logger.info(
            "throttle: keyword_hits=%d, turn=%d, every=%d -> extract",
            keyword_hits,
            turn_count,
            every_n,
        )
        return True

    logger.debug(
        "throttle: keyword_hits=%d, turn=%d -> skip",
        keyword_hits,
        turn_count,
    )
    return False
