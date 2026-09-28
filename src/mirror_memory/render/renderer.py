"""Memory block renderer -- domain-agnostic.

Renders beliefs into a compact text block for prompt injection.
All dimension-specific rendering is driven by ``config.display_labels``.
Uses ``core.activity``, ``core.retrieval``, and ``core.budget`` for
scoring and budget computation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import (
    CLAIM_TEXT_MAX_LENGTH,
    QUERY_NOUN_MATCH_THRESHOLD,
    SNIPPET_BUDGET_OVERFLOW_FACTOR,
    TIME_LABEL_ACTIVE_DAYS,
    TIME_LABEL_MONTHLY_DAYS,
    TIME_LABEL_WEEKLY_DAYS,
)
from mirror_memory.core.utils import content_tokens
from mirror_memory.render.display import DisplayDict

logger = logging.getLogger(__name__)


def _is_en(language: str) -> bool:
    return language.strip().lower() == "en"


_STOP_WORDS_EN = frozenset({
    "when", "where", "which", "what", "does", "did", "have", "been",
    "would", "could", "should", "about", "their", "there", "these",
    "those", "after", "before", "first", "last", "also", "more",
    "from", "with", "this", "that", "were", "been", "some", "them",
})


def _is_rejected_line(text: str) -> bool:
    """Check if a rendered item is the rejected (D8 boundary) line."""
    return text.startswith("Avoid:") or text.startswith("\u5e94\u56de\u907f")


def _extract_query_nouns(query: str) -> set[str]:
    """Extract meaningful query terms for relevance matching.

    Handles both English (space-separated words > 4 chars) and
    Chinese (character bigrams, since CJK has no spaces).
    """
    if not query:
        return set()
    # Detect CJK characters.
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in query)
    if has_cjk:
        # Chinese: extract character bigrams as query terms.
        clean = "".join(ch for ch in query if "\u4e00" <= ch <= "\u9fff")
        return {clean[i:i+2] for i in range(len(clean) - 1)} if len(clean) > 1 else {clean}
    # English: split and filter stop words.
    return {w for w in query.lower().split() if len(w) > 4 and w not in _STOP_WORDS_EN}


def _extract_query_topics(query: str, config: MemoryConfig) -> set[str]:
    """Extract topic keywords from a query string for relevance matching.

    Returns a set of keyword categories that appear in the query,
    used by ``score_belief`` to boost topic-matching beliefs.
    """
    if not query:
        return set()
    lowered = query.lower()
    topics: set[str] = set()
    for category, keywords in config.extraction.keywords.items():
        for kw in keywords:
            if kw.lower() in lowered:
                topics.add(category)
                break
    return topics


# Query-to-predicate mapping for cognitive triple retrieval.
_QUERY_PREDICATE_PATTERNS: dict[str, set[str]] = {
    # English patterns
    "what does": {"likes", "prefers", "has", "wants_to"},
    "what do": {"likes", "prefers", "has"},
    "what did": {"went_to", "attended", "bought", "experienced"},
    "what is": {"is", "has", "lives_in", "works_at"},
    "where does": {"lives_in", "works_at", "went_to"},
    "where did": {"went_to", "attended"},
    "who does": {"knows", "friends_with"},
    "what are": {"likes", "skills", "has"},
    "what's": {"is", "has", "likes"},
    "likes": {"likes"},
    "loves": {"likes"},
    "enjoys": {"likes"},
    "prefers": {"prefers"},
    "wants": {"wants_to"},
    "needs": {"has"},
    "lives": {"lives_in"},
    "works": {"works_at"},
    "went": {"went_to"},
    "bought": {"bought"},
    # Chinese patterns
    "喜欢": {"likes"},
    "爱": {"likes"},
    "住": {"lives_in"},
    "工作": {"works_at"},
    "去了": {"went_to"},
    "买了": {"bought"},
    "想要": {"wants_to"},
    "需要": {"has"},
}


def _extract_query_triples(
    query: str,
    config: MemoryConfig,
) -> tuple[set[str], set[str]]:
    """Extract predicate and object signals from a query.

    Returns ``(predicates, objects)`` sets for ``score_belief`` matching.
    """
    if not query:
        return set(), set()

    lowered = query.lower().strip()
    predicates: set[str] = set()
    objects: set[str] = set()

    # Pattern-based predicate extraction.
    for pattern, preds in _QUERY_PREDICATE_PATTERNS.items():
        if pattern in lowered:
            predicates.update(preds)

    # Object extraction: check if any anchor key appears in the query.
    for anchor in config.anchors:
        for phrase in anchor.phrases:
            if phrase.lower() in lowered:
                objects.add(anchor.key)
                break

    return predicates, objects


# Query intent → which validity intervals and storage tiers to draw from.
# "Where do they live now?" and "where did they live before?" are different
# questions about the same predicate (interval filter), and a "now?"
# question must not pay for scanning cold storage (tier filter).  The
# detection itself lives in mirror_memory.metabolism.eligibility so it is
# public, testable, and reusable; the aliases below keep the historical
# renderer-level entry points working.
from mirror_memory.metabolism.eligibility import (  # noqa: E402
    detect_query_mode as _detect_query_mode_impl,
    is_enumeration_query as _is_enumeration_query_impl,
)


def _detect_temporal_mode(query: str) -> str:
    """Map a query onto a retrieval mode (current / historical / evidence /
    experience / all_occurrences / all).  Historical wins on a tie;
    all-occurrences wins over both."""
    return _detect_query_mode_impl(query)


def _is_enumeration_query(query: str) -> bool:
    """Does this query ask for every matching item rather than one value?"""
    return _is_enumeration_query_impl(query)


def render_memory_block(
    session: object,
    user_id: str,
    config: MemoryConfig,
    *,
    user_message: str = "",
    language: str = "zh",
    tail_load: int = 0,
    trace_hook: object | None = None,
    now: datetime | None = None,
) -> str | None:
    """Render a memory block for prompt injection.

    Parameters
    ----------
    session:
        SQLAlchemy database session.
    user_id:
        The user identifier.
    config:
        The ``MemoryConfig`` providing render parameters, display
        labels, and dimension definitions.
    user_message:
        The current user message (used for topic matching).
    language:
        ``"zh"`` or ``"en"``.
    tail_load:
        Character count of content already in the prompt.
    trace_hook:
        Optional observability callable ``trace_hook(stage, **fields)``.
        Purely additive: it observes the retrieve/context stages and never
        changes what is rendered.
    now:
        Reference time for freshness scoring, time labels, and the access
        telemetry this call records.  Defaults to the current UTC time;
        benchmarks and tests pass a simulated clock so a recall can be
        positioned on the same timeline as the metabolism cycle that
        follows it.

    Returns
    -------
    str or None
        The rendered memory block, or ``None`` if there is nothing to
        render.
    """
    def _trace(stage: str, **fields: object) -> None:
        if trace_hook is None:
            return
        try:
            trace_hook(stage, **fields)
        except Exception:
            logger.debug("render: trace hook failed at stage=%s", stage, exc_info=True)
    # Lazy imports to avoid circular dependencies.
    from mirror_memory.core.activity import belief_activity
    from mirror_memory.core.budget import compute_profile_budget
    from mirror_memory.core.repository import list_rejected_beliefs
    from mirror_memory.core.retrieval import score_belief

    display = DisplayDict(config)
    is_en = _is_en(language)
    joiner = "; " if is_en else "\uff1b"

    # Compute budget
    render_cfg = config.render
    budget = compute_profile_budget(
        tail_load,
        0.0,  # knowledge_proxy -- caller may inject
        base=render_cfg.base,
        floor=render_cfg.floor,
        cap=render_cfg.cap,
    )

    # Each rendered line carries the belief it came from (None for the
    # rejected-line banner) so access telemetry can be recorded for exactly
    # the beliefs that reached the prompt.
    items: list[tuple[str, object | None]] = []

    # -- Rejected beliefs (D8 equivalent) -- fixed first slot ----------------
    rejected = list_rejected_beliefs(session, user_id, limit=5)
    if rejected:
        rejected_labels = [
            label
            for label in (display.friendly_label(b.key, language) for b in rejected)
            if label
        ]
        if rejected_labels:
            if is_en:
                items.append((f"Avoid: {', '.join(rejected_labels)}", None))
            else:
                joined = "\u3001".join(rejected_labels)
                items.append((f"\u5e94\u56de\u907f\uff08\u7528\u6237\u5df2\u660e\u786e\u6401\u7f6e\uff09\uff1a{joined}", None))

    # -- Active beliefs: scored and sorted -----------------------------------
    # Extract topics, predicates, and objects from user_message for
    # query-aware relevance scoring.
    query_topics = _extract_query_topics(user_message, config) if user_message else set()
    query_predicates: set[str] = set()
    query_objects: set[str] = set()
    if user_message:
        query_predicates, query_objects = _extract_query_triples(user_message, config)

    # Build dimension gate lookup (requires_user_confirmation per dimension).
    confirm_gates = {d.dimension_id: d.requires_user_confirmation for d in config.dimensions}
    l4_threshold = render_cfg.l4_render_threshold

    # Query-aware admission.  The old path took the N most recent beliefs and
    # scored those, which makes recency a stand-in for relevance and discards
    # a fact stated early in a long conversation before any relevance signal
    # runs.  ``recall_candidates`` unions the lexically relevant slice with a
    # small recent floor, so the context is never empty either.
    from mirror_memory.core.repository import recall_candidates

    query_tokens = content_tokens(user_message)
    all_beliefs = recall_candidates(
        session, user_id, query=user_message,
        temporal_mode=_detect_temporal_mode(user_message),
    )
    now = now or datetime.now(UTC)
    # Evidence is a first-class entity now, so the multi-evidence bonus counts
    # typed links rather than a blob of ids on the belief row.
    from mirror_memory.core.repository import evidence_counts_for_beliefs

    evidence_counts = evidence_counts_for_beliefs(session, [b.id for b in all_beliefs])
    scored: list[tuple[float, object]] = []
    gated_out: list[str] = []
    for belief in all_beliefs:
        # Gate 1: L4 render watermark — only render extracted beliefs above threshold.
        if belief.layer == "L4" and belief.confidence < l4_threshold:
            gated_out.append(f"l4_watermark:{belief.key}")
            continue
        # Gate 2: dimension requires user confirmation — skip unconfirmed beliefs.
        if confirm_gates.get(belief.dimension, False) and belief.source != "user_confirmed":
            gated_out.append(f"needs_confirmation:{belief.key}")
            continue
        s = score_belief(
            belief,
            topics=query_topics,
            query_predicates=query_predicates,
            query_objects=query_objects,
            now=now,
            evidence_count=evidence_counts.get(belief.id),
            query_tokens=query_tokens,
        )
        if s > 0:
            scored.append((s, belief))
        else:
            gated_out.append(f"zero_score:{belief.key}")
    scored.sort(key=lambda x: (x[0], getattr(x[1], "last_evidence_at", None) or ""), reverse=True)
    _trace(
        "retrieve",
        candidates=len(all_beliefs),
        temporal_mode=_detect_temporal_mode(user_message),
        scored=[b.key for _s, b in scored],
        gated_out=gated_out,
    )

    # Diversity: max N items per dimension.  An enumeration query ("what
    # languages do they speak") needs every matching value, so the per-
    # dimension cap lifts for it; the character budget below remains the hard
    # limit either way, so a wider cap costs prompt space, not correctness.
    max_per = render_cfg.max_per_dimension
    if _is_enumeration_query(user_message):
        max_per = max(max_per, render_cfg.enumeration_max_per_dimension)
    dim_counts: dict[str, int] = {}
    for _score, belief in scored:
        dim = belief.dimension
        count = dim_counts.get(dim, 0)
        if count >= max_per:
            continue
        text = _render_belief(belief, display, language, now=now)
        if text:
            items.append((text, belief))
            dim_counts[dim] = count + 1

    # -- Budget packing: whole-item, never truncate --------------------------
    # The rejected (D8 boundary) line in items[0] is exempt from the budget —
    # it is the strongest user signal and must never be dropped for space.
    rendered: list[tuple[str, object | None]] = []
    used = 0
    for i, (text, belief) in enumerate(items):
        is_exempt = i == 0 and _is_rejected_line(items[0][0])
        if not is_exempt:
            prefix_cost = len(joiner) if rendered else 0
            if used + prefix_cost + len(text) > budget:
                continue
            used += prefix_cost + len(text)
        rendered.append((text, belief))

    # Access telemetry: a belief that reached the prompt is, by definition,
    # a live belief — this is the reheat signal the heat score consumes.
    # Purely additive: it never changes what is rendered, and the caller
    # owns the transaction (MemoryEngine.recall commits it).
    from mirror_memory.core.repository import mark_belief_accessed

    accessed_ids = [
        getattr(belief, "id") for _text, belief in rendered if belief is not None
    ]
    try:
        mark_belief_accessed(session, accessed_ids, now=now)
    except Exception:
        logger.debug("render: access telemetry failed", exc_info=True)

    block = joiner.join(text for text, _belief in rendered)

    # -- Evidence intent: attach provenance density -------------------------
    # A "why / based on what" question deserves the evidence picture, not
    # just the conclusion: the digest summary for compacted beliefs, plain
    # observation counts otherwise.  Only for beliefs that made it into the
    # block, bounded by the same character budget.
    if user_message and _detect_temporal_mode(user_message) == "evidence":
        from mirror_memory.core.repository import digests_for_beliefs

        evidence_beliefs = [b for _t, b in rendered if b is not None][:3]
        digests = digests_for_beliefs(session, [b.id for b in evidence_beliefs])
        for belief in evidence_beliefs:
            digest = digests.get(belief.id)
            if digest is not None and digest.summary:
                hint = f"[evidence] {belief.key}: {digest.summary}"
            else:
                count = evidence_counts.get(belief.id) or 0
                hint = f"[evidence] {belief.key}: {count} observation(s)"
            prefix_cost = len(joiner) if block else 0
            if len(block or "") + prefix_cost + len(hint) <= budget:
                block = (block + joiner + hint) if block else hint

    # -- Fallback: session summary retrieval --------------------------------
    # If beliefs don't contain the key nouns from the query, search session
    # summaries for specific facts (dates, names, numbers) that beliefs lose.
    # Only available when session_summary_enabled=True in config.
    if user_message and config.session_summary_enabled:
        query_nouns = _extract_query_nouns(user_message)
        block_lower = (block or "").lower()
        # Check if key query terms appear in the block.
        matched = sum(1 for w in query_nouns if w in block_lower)
        if query_nouns and matched < len(query_nouns) * QUERY_NOUN_MATCH_THRESHOLD:
            snippets = _retrieve_session_snippets(session, user_id, user_message, limit=3)
            if snippets:
                snippet_text = " | ".join(snippets)
                # Allow 2x budget for snippet overflow.
                remaining = budget * SNIPPET_BUDGET_OVERFLOW_FACTOR - len(block or "")
                if len(snippet_text) > remaining:
                    snippet_text = snippet_text[:remaining] + "..."
                if block:
                    block = block + "; [context] " + snippet_text
                else:
                    block = snippet_text

    logger.info("render: chars=%d budget=%d items=%d", len(block or ""), budget, len(rendered))
    _trace(
        "context",
        rendered_keys=[b.key for _s, b in scored][: len(rendered)],
        rendered_items=len(rendered),
        dropped_for_budget=len(scored) - len(rendered),
        chars=len(block or ""),
        budget=budget,
        used_snippet_fallback=bool(user_message and config.session_summary_enabled and "[context]" in (block or "")),
    )
    return block or None


def _retrieve_session_snippets(
    session: object,
    user_id: str,
    query: str,
    limit: int = 3,
) -> list[str]:
    """Search session summaries for snippets matching query keywords.

    Returns the most relevant snippets (by keyword hit count), preserving
    specific details like dates, names, and numbers.
    """
    from mirror_memory.core.models import SessionSummary

    summaries = (
        session.query(SessionSummary)
        .filter(SessionSummary.user_id == user_id)
        .order_by(SessionSummary.created_at.desc())
        .limit(20)
        .all()
    )
    if not summaries or not query:
        return []

    # Extract query keywords (simple approach: split and filter).
    query_words = set(query.lower().split())
    # Remove common stop words.
    stop_words = {"the", "a", "an", "is", "was", "were", "are", "do", "did", "does",
                  "what", "when", "where", "who", "how", "which", "that", "this",
                  "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
                  "her", "its", "our", "their", "me", "him", "us", "them"}
    query_keywords = query_words - stop_words

    scored: list[tuple[int, str]] = []
    for s in summaries:
        text_lower = s.summary_text.lower()
        hits = sum(1 for kw in query_keywords if kw in text_lower)
        if hits > 0:
            scored.append((hits, s.summary_text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:limit]]


def _render_belief(belief: object, display: DisplayDict, language: str, now: datetime | None = None) -> str | None:
    """Render a single belief into a display string.

    Uses the display dictionary for label lookup.  If no label is
    found, returns ``None`` (the belief is skipped -- never leak
    raw keys into the prompt).
    """
    is_en = _is_en(language)
    label = display.friendly_label(belief.key, language)

    # Dimension-specific rendering hooks
    dim = belief.dimension

    # For beliefs with structured value_json, try to enrich the display
    from mirror_memory.core.utils import safe_json

    val = safe_json(belief.value_json)

    # Prefer claim_text (contains actual information) over generic label.
    # Labels like "Possessions" or "Hobbies" are category names that lose
    # the specific content needed for question answering.
    claim = belief.claim_text[:CLAIM_TEXT_MAX_LENGTH] if belief.claim_text else None
    display_text = claim or label
    if not display_text:
        return None

    # Add time annotation if recent
    time_tag = _time_ago_label(belief, is_en, now=now)
    if time_tag:
        return f"{display_text} ({time_tag})" if is_en else f"{display_text}\uff08{time_tag}\uff09"
    return display_text


def _time_ago_label(belief: object, is_en: bool, now: datetime | None = None) -> str:
    """Compact time annotation based on last_evidence_at."""
    ts = getattr(belief, "last_evidence_at", None)
    if ts is None:
        return ""
    now = now or datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    days = max(0, (now - ts).days)
    if days > TIME_LABEL_MONTHLY_DAYS:
        return ""
    if days <= TIME_LABEL_ACTIVE_DAYS:
        return "active" if is_en else "\u6d3b\u8dc3"
    if days <= TIME_LABEL_WEEKLY_DAYS:
        weeks = max(1, days // 7)
        return f"{weeks}w ago" if is_en else f"{weeks}\u5468\u524d"
    months = max(1, days // 30)
    return f"{months}mo ago" if is_en else f"{months}\u6708\u524d"
