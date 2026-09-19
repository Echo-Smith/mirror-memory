"""Value-driven throttle for LLM semantic extraction (K2).

Decides whether the current turn warrants an LLM extraction call.
All thresholds come from ``config.extraction`` -- zero domain coupling.

Throttling = base rules (uniform discipline) + information-gain gates:

- Base rules (kept for backward compatibility): keyword hits >=
  ``llm_min_keyword_hits`` always extract; otherwise extract every
  ``llm_every_turns`` turns when there is at least one keyword hit.
- High-value gate: when the information-gain score from
  :func:`compute_extraction_value` reaches ``extraction_value_threshold``,
  extraction fires early -- breaking the uniform schedule for turns that
  actually teach the system something new.
- Suppression gate: when every keyword dimension hit this turn is already
  covered by a high-confidence active belief, extraction is suppressed --
  the marginal information gain of re-extracting known information is ~0.
"""

from __future__ import annotations

import logging

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import L4_RENDER_THRESHOLD

logger = logging.getLogger(__name__)

# -- Information-gain scoring -------------------------------------------------
# Component weights sum to 1.0 so the score stays in [0, 1].
W_NOVELTY = 0.45   # new topics not covered by existing active beliefs
W_SCARCITY = 0.25  # configured dimensions with no active belief yet
W_VALUE = 0.30     # hit on high-value dimensions

# A hit dimension counts as "already known" only when covered by a belief at
# or above this confidence -- reusing the render watermark (roughly two
# cross-session evidence items).  Below it, the belief is still a hypothesis
# and more evidence remains valuable.
_MATURE_COVERAGE_CONFIDENCE = L4_RENDER_THRESHOLD


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


def _hit_dimensions(text: str, config: MemoryConfig) -> set[str]:
    """Dimensions hit by extraction keywords in *text*.

    A keyword category maps to its dimension prefix (``"fact.age"`` ->
    ``"fact"``); plain category names are dimension ids themselves
    (``"topic"``, ``"preference"``, ...).
    """
    if not text:
        return set()
    lowered = text.lower()
    dims: set[str] = set()
    for category, keywords in config.extraction.keywords.items():
        if any(kw.lower() in lowered for kw in keywords):
            dims.add(category.split(".", 1)[0])
    return dims


def _belief_dimensions(active_belief_keys: set[str] | None, config: MemoryConfig) -> set[str]:
    """Map active belief keys onto the set of dimensions they cover.

    Generic key -> dimension matching (no domain knowledge required):
    - the key is a configured anchor key (anchors carry the dimension);
    - the key itself is a dimension id (``"topic"``);
    - the key is namespaced under a dimension (``"fact.age"`` covers ``fact``).
    """
    if not active_belief_keys:
        return set()
    anchor_dim_by_key = {a.key: a.dimension for a in config.anchors}
    covered: set[str] = set()
    for key in active_belief_keys:
        dim = anchor_dim_by_key.get(key)
        if dim:
            covered.add(dim)
        covered.add(key.split(".", 1)[0])
    return covered


def compute_extraction_value(
    *,
    text: str,
    config: MemoryConfig,
    active_belief_keys: set[str] | None = None,
    high_value_dimensions: set[str] | None = None,
) -> float:
    """Compute the information-gain value of extracting from this turn.

    Pure function (no I/O).  Three components with configurable weights:

    - Novelty (0.45): fraction of keyword-hit dimensions with no
      corresponding active belief (see :func:`_belief_dimensions` for the
      generic key -> dimension matching).
    - Scarcity (0.25): fraction of configured dimensions
      (``config.dimensions``) that have no active belief yet -- thin profiles
      are worth filling.
    - Value (0.30): fraction of keyword-hit dimensions present in the
      high-value set (``high_value_dimensions``, falling back to
      ``config.extraction.high_value_dimensions`` when ``None``).

    Returns a float in [0, 1].
    """
    hit_dims = _hit_dimensions(text, config)
    if not hit_dims:
        # No keyword signal this turn: the LLM has nothing to anchor on, so
        # extraction value is zero regardless of profile scarcity.
        return 0.0
    covered = _belief_dimensions(active_belief_keys, config)

    # Novelty: hit dimensions not covered by any active belief.
    novelty = len(hit_dims - covered) / len(hit_dims)

    # Scarcity: configured dimensions without any active belief.
    config_dims = {d.dimension_id for d in config.dimensions}
    scarcity = len(config_dims - covered) / len(config_dims) if config_dims else 0.0

    # Value: hit dimensions flagged as high-value for extraction priority.
    hv = high_value_dimensions if high_value_dimensions is not None else set(config.extraction.high_value_dimensions)
    value = len(hit_dims & hv) / len(hit_dims)

    return round(min(1.0, max(0.0, W_NOVELTY * novelty + W_SCARCITY * scarcity + W_VALUE * value)), 4)


def should_extract(
    text: str,
    turn_count: int,
    config: MemoryConfig,
    *,
    session: object | None = None,
    user_id: str = "",
) -> bool:
    """Decide whether the current turn warrants LLM extraction.

    Base rules (unchanged when no *session*/*user_id* is given -- cold paths
    and unit tests keep their exact behaviour):

    - If keyword hits >= ``llm_min_keyword_hits``: extract.
    - Otherwise: extract every ``llm_every_turns`` turns when there is at
      least one keyword hit.
    - ``llm_min_keyword_hits=0`` means "always extract" (benchmark escape
      hatch) and is never overridden by the value gates.

    With a *session* (production path), two information-gain gates are
    applied on top of the base rules:

    - Suppression: all keyword-hit dimensions are already covered by
      high-confidence active beliefs -> skip (avoid re-extracting known
      information; leaves the budget to high-gain turns).
    - Early trigger: :func:`compute_extraction_value` >=
      ``extraction_value_threshold`` -> extract immediately, breaking the
      uniform every-N schedule.

    Parameters
    ----------
    text:
        The current user message.
    turn_count:
        The number of the current turn (0-indexed).
    config:
        The ``MemoryConfig`` providing throttle parameters.
    session:
        Optional SQLAlchemy session (enables the value gates).
    user_id:
        Optional user id (required together with *session*).

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

    base = keyword_hits >= min_hits or (keyword_hits >= 1 and turn_count % every_n == 0)

    # Value gates need belief state; without it, fall back to base rules.
    # Zero-signal turns can neither trigger nor be suppressed by the gates.
    if session is None or not user_id or keyword_hits < 1:
        if base:
            logger.info("throttle: base rule -> extract (hits=%d turn=%d)", keyword_hits, turn_count)
        else:
            logger.debug("throttle: base rule -> skip (hits=%d turn=%d)", keyword_hits, turn_count)
        return base

    try:
        from mirror_memory.core.repository import list_active_beliefs

        active = list_active_beliefs(session, user_id)
    except Exception:
        logger.warning("throttle: belief lookup failed; falling back to base rules", exc_info=True)
        return base

    belief_keys = {b.key for b in active}
    value = compute_extraction_value(text=text, config=config, active_belief_keys=belief_keys)

    hit_dims = _hit_dimensions(text, config)
    mature_keys = {b.key for b in active if float(b.confidence) >= _MATURE_COVERAGE_CONFIDENCE}
    mature_dims = _belief_dimensions(mature_keys, config)
    fully_covered = bool(hit_dims) and hit_dims <= mature_dims

    logger.info(
        "throttle: value=%.2f threshold=%.2f hits=%d hit_dims=%s fully_covered=%s base=%s turn=%d",
        value,
        config.extraction.extraction_value_threshold,
        keyword_hits,
        sorted(hit_dims),
        fully_covered,
        base,
        turn_count,
    )

    # Suppression first: if everything this turn touches is already known at
    # high confidence, the marginal gain is ~0 even when the scalar score is
    # inflated by scarcity elsewhere.
    if fully_covered:
        logger.info("throttle: all hit dimensions mature-covered -> suppress extraction")
        return False

    # High-value gate: break the uniform schedule for high-gain turns.
    if value >= config.extraction.extraction_value_threshold:
        logger.info("throttle: value >= threshold -> extract early")
        return True

    return base
