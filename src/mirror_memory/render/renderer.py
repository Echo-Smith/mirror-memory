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


def render_memory_block(
    session: object,
    user_id: str,
    config: MemoryConfig,
    *,
    user_message: str = "",
    language: str = "zh",
    tail_load: int = 0,
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

    Returns
    -------
    str or None
        The rendered memory block, or ``None`` if there is nothing
        to render.
    """
    # Lazy imports to avoid circular dependencies.
    from mirror_memory.core.activity import belief_activity
    from mirror_memory.core.budget import compute_profile_budget
    from mirror_memory.core.repository import list_active_beliefs, list_rejected_beliefs
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

    items: list[str] = []

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
                items.append(f"Avoid: {', '.join(rejected_labels)}")
            else:
                joined = "\u3001".join(rejected_labels)
                items.append(f"\u5e94\u56de\u907f\uff08\u7528\u6237\u5df2\u660e\u786e\u6401\u7f6e\uff09\uff1a{joined}")

    # -- Active beliefs: scored and sorted -----------------------------------
    # Extract topics from user_message for query-aware relevance scoring.
    query_topics = _extract_query_topics(user_message, config) if user_message else set()

    all_beliefs = list_active_beliefs(session, user_id)
    now = datetime.now(UTC)
    scored: list[tuple[float, object]] = []
    for belief in all_beliefs:
        s = score_belief(belief, topics=query_topics, now=now)
        if s > 0:
            scored.append((s, belief))
    scored.sort(key=lambda x: (x[0], getattr(x[1], "last_evidence_at", None) or ""), reverse=True)

    # Diversity: max N items per dimension
    max_per = render_cfg.max_per_dimension
    dim_counts: dict[str, int] = {}
    for _score, belief in scored:
        dim = belief.dimension
        count = dim_counts.get(dim, 0)
        if count >= max_per:
            continue
        text = _render_belief(belief, display, language)
        if text:
            items.append(text)
            dim_counts[dim] = count + 1

    # -- Budget packing: whole-item, never truncate --------------------------
    rendered: list[str] = []
    used = 0
    for text in items:
        prefix_cost = len(joiner) if rendered else 0
        if used + prefix_cost + len(text) > budget:
            continue
        used += prefix_cost + len(text)
        rendered.append(text)

    block = joiner.join(rendered)

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


def _render_belief(belief: object, display: DisplayDict, language: str) -> str | None:
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
    time_tag = _time_ago_label(belief, is_en)
    if time_tag:
        return f"{display_text} ({time_tag})" if is_en else f"{display_text}\uff08{time_tag}\uff09"
    return display_text


def _time_ago_label(belief: object, is_en: bool) -> str:
    """Compact time annotation based on last_evidence_at."""
    ts = getattr(belief, "last_evidence_at", None)
    if ts is None:
        return ""
    now = datetime.now(UTC)
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
