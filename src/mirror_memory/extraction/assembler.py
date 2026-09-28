"""Claim priority assembler -- scarce dimensions first, cap fill.

Reorders and truncates the combined K1+K2 claim set before persistence,
activating ``config.extraction.max_claims_per_turn`` and
``DimensionConfig.render_priority`` (both previously dead config).
"""

from __future__ import annotations

import logging

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.memory.polarity import object_tokens

logger = logging.getLogger(__name__)


def assemble_claims(
    claims: list[dict],
    config: MemoryConfig,
    *,
    active_dimension_counts: dict[str, int] | None = None,
) -> list[dict]:
    """Assemble the final claim set for persistence.

    Priority order:
    1. Scarce dimensions first (fewer active beliefs = more novel information)
    2. Higher render_priority dimensions first within the same scarcity band
    3. Higher confidence first as tiebreaker

    Truncates to ``config.extraction.max_claims_per_turn``.

    Parameters
    ----------
    claims:
        Combined K1+K2 claims for this turn.
    config:
        The ``MemoryConfig``.
    active_dimension_counts:
        Mapping of dimension -> count of active beliefs (from DB).
        If None, all dimensions treated as equally scarce.

    Returns
    -------
    list[dict]
        Ordered, truncated claim list.
    """
    if not claims:
        return []

    counts = active_dimension_counts or {}
    max_claims = config.extraction.max_claims_per_turn

    # Build render_priority lookup.
    priorities = {d.dimension_id: d.render_priority for d in config.dimensions}

    def _sort_key(claim: dict) -> tuple:
        dim = claim.get("dimension", "")
        scarcity = counts.get(dim, 0)
        priority = priorities.get(dim, 0)
        confidence = claim.get("confidence", 0.0)
        # Bonus for cognitive triple structure (predicate+object present).
        value = claim.get("value") or {}
        has_triple = 1 if (value.get("predicate") and value.get("object")) else 0
        # Quality first: a structured, high-confidence claim is the one worth
        # persisting.  Scarcity used to be the primary key, which meant a fact
        # in an already-populated dimension lost to low-confidence filler in an
        # empty one -- the answer claim was dropped precisely because its
        # dimension was well covered.  Scarcity now only breaks ties between
        # claims of equal quality, which is where breadth-seeking belongs.
        return (-has_triple, -confidence, -priority, scarcity)

    ordered = sorted(claims, key=_sort_key)
    result = ordered[:max_claims]
    if len(ordered) > max_claims:
        logger.info("assembler: %d claims -> %d (max_claims_per_turn)", len(ordered), max_claims)
    return result


def _object_key(claim: dict) -> str:
    """The identity a claim is about, for same-turn arbitration.

    Falls back to the object's significant tokens rather than the raw string:
    K1's regex object capture swallows trailing adverbs ("coffee again" from
    "I like coffee again"), and an exact-string key would let that duplicate
    escape arbitration against the model's "coffee".
    """
    value = claim.get("value") or {}
    obj = (value.get("object") or "").strip().lower()
    if obj:
        tokens = sorted(object_tokens(obj))
        if tokens:
            return " ".join(tokens)
    # A claim with no object falls back to its own text, so two claims only
    # collide when they are literally about the same words.
    return (claim.get("claim_text") or "").strip().lower()


def _same_object_group(a: str, b: str) -> bool:
    """Do two arbitration keys name the same object?

    Token intersection, not equality: ``coffee`` and ``coffee_again`` share
    the significant token and are the same object said twice, while
    ``berlin`` and ``bicycle`` share nothing.
    """
    if not a or not b:
        return False
    sa, sb = set(a.split()), set(b.split())
    return bool(sa & sb)


def arbitrate_claims(
    claims: list[dict],
    policy: dict,
) -> list[dict]:
    """Resolve same-turn conflicts between K1 and K2 claims.

    Both extractors run every turn and can disagree about what a sentence
    means.  "My employer is Northstar." followed by "I moved to Redwood
    Health." is a change of employer, but K1's ``moved_to`` pattern also
    reads it as a change of residence: left alone, the K2 ``works_at`` claim
    and the K1 ``lives_in`` claim both persist, and the cross-predicate
    revival path then closes the *correct* K2 row in favour of the wrong K1
    one.

    The rule: when the two stages disagree about the same object in the same
    turn, the stage that names a configured predicate wins and the other is
    demoted to provenance.  When they agree (same predicate family) the K2
    claim's structure and temporal fields are kept and the K1 duplicate is
    dropped.  When only K1 speaks about an object -- K2 failed, was
    throttled, or found nothing there -- K1 stays as the fallback.

    Parameters
    ----------
    claims:
        Combined K1+K2 claims for this turn, each tagged with
        ``extractor_stage`` (``"k1"`` or ``"k2"``).
    policy:
        The identity policy mapping (predicate -> cardinality/scope config);
        a predicate present here is a configured slot.
    """
    if not claims:
        return []

    # Cluster by object identity: exact token match, else token intersection.
    groups: list[list[dict]] = []
    for claim in claims:
        key = _object_key(claim)
        for group in groups:
            if _same_object_group(_object_key(group[0]), key):
                group.append(claim)
                break
        else:
            groups.append([claim])

    kept: list[dict] = []
    for group in groups:
        k2_claims = [c for c in group if c.get("extractor_stage") == "k2"]
        k1_claims = [c for c in group if c.get("extractor_stage") != "k2"]
        if not k2_claims:
            # No model opinion about this object: K1 is the only voice.
            kept.extend(k1_claims)
            continue
        if not k1_claims:
            kept.extend(k2_claims)
            continue

        k2_predicates = {
            (c.get("value") or {}).get("predicate", "") for c in k2_claims
        }
        k2_configured = any(p in policy for p in k2_predicates if p)
        for claim in k1_claims:
            k1_predicate = (claim.get("value") or {}).get("predicate", "")
            if not k1_predicate:
                # A triple-less K1 claim (keyword hit) asserts no slot, so it
                # never conflicts with one; persistence skips it anyway.
                kept.append(claim)
                continue
            if k1_predicate in k2_predicates:
                # Same slot, two stages: the model's structure and temporal
                # fields are the better record of the same assertion.
                logger.info(
                    "assembler: arbitration merged k1 %r into k2",
                    k1_predicate,
                )
                continue
            if k2_configured:
                logger.info(
                    "assembler: arbitration demoted k1 %r "
                    "(k2 asserts a configured predicate)",
                    k1_predicate,
                )
                continue
            # The model named no configured predicate for this object, so its
            # claim is not authoritative over the pattern's reading.
            kept.append(claim)
        kept.extend(k2_claims)
    return kept
