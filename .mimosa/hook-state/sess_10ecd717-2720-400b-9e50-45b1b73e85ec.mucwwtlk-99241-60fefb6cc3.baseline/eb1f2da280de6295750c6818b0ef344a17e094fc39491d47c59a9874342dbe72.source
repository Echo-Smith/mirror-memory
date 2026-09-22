"""Shared utility functions for mirror-memory.

Extracted to eliminate duplication across modules:
- ``safe_json`` — parse JSON with fallback (used 6+ times)
- ``safe_json_list`` — parse a JSON array with fallback
- ``parse_llm_json`` — parse LLM output that may contain markdown fences (used 3+ times)
- ``truncate_id`` — truncate identifiers for logging (used 4+ times)
- ``belief_evidence_ids`` — read a belief's evidence refs (used 3+ times)
- ``content_tokens`` — significant tokens for lexical matching
- ``coerce_datetime`` — datetime / ISO string / None -> datetime | None
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Words that carry no matching signal.  Matching on them makes every belief
# look relevant to every question.
_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "did", "its", "let", "may", "new", "now", "old", "see", "two",
    "who", "boy", "did", "man", "run", "she", "too", "use", "that", "this",
    "with", "have", "from", "they", "been", "said", "each", "which", "their",
    "will", "many", "some", "them", "then", "than", "into", "over", "only",
    "also", "back", "after", "what", "when", "where", "does", "were",
})

_CJK_RANGES = ("\u4e00", "\u9fff")


def safe_json(text: str | None, fallback: dict | None = None) -> dict:
    """Parse a JSON string, returning *fallback* on any error.

    Handles ``None``, empty strings, and malformed JSON gracefully.
    Never raises.
    """
    if not text:
        return dict(fallback) if fallback else {}
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else (dict(fallback) if fallback else {})
    except (TypeError, ValueError):
        return dict(fallback) if fallback else {}


def safe_json_list(text: str | None) -> list:
    """Parse a JSON array string, returning ``[]`` on any error.

    Handles ``None``, empty strings, malformed JSON, and non-array JSON.
    Never raises.
    """
    if not text:
        return []
    try:
        result = json.loads(text)
    except (TypeError, ValueError):
        return []
    return result if isinstance(result, list) else []


def _coerce_evidence_ids(raw) -> list[int]:
    """Coerce a raw JSON value into a list of integer evidence IDs."""
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def belief_evidence_ids(belief) -> list[int]:
    """Return the evidence message IDs recorded on *belief*.

    ``Belief.evidence_json`` is the only field the repository writes, so it is
    the source of truth.  ``value_json["evidence_ids"]`` is a legacy location
    some older rows still carry and is consulted only as a fallback.
    """
    ids = _coerce_evidence_ids(safe_json_list(getattr(belief, "evidence_json", None)))
    if not ids:
        val = safe_json(getattr(belief, "value_json", None))
        ids = _coerce_evidence_ids(val.get("evidence_ids"))
    return ids


def parse_llm_json(raw: str | None, *, expect_array: bool = False) -> dict | list | None:
    """Parse LLM output that may contain markdown fences or bare arrays.

    Handles:
    - `````json ... ``` fenced blocks
    - Bare ``{...}`` objects
    - Bare ``[...]`` arrays (when *expect_array* is True)

    Returns the parsed object, or ``None`` on failure.
    """
    text = (raw or "").strip()
    if not text:
        return None

    # Strip markdown fences.
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    # Try bare array first (common LLM output format).
    if expect_array and text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting a JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Try bare array as last resort.
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    return None


def truncate_id(id_str: str | None, length: int = 8) -> str:
    """Truncate an identifier for safe logging (privacy-preserving).

    Returns the first *length* characters, or ``"?"`` if empty.
    """
    if not id_str:
        return "?"
    return str(id_str)[:length]


def coerce_datetime(value) -> datetime | None:
    """Coerce a datetime, ISO string, or ``None`` into a datetime.

    Extraction output and LLM payloads hand back strings; storage and
    comparisons need datetimes.  An unparseable string becomes ``None`` rather
    than raising, so one bad field cannot fail a whole write.
    """
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def content_tokens(text: str | None, *, min_length: int = 3) -> set[str]:
    """Significant tokens for lexical matching between a query and a belief.

    Stop words and short tokens are dropped: matching on them makes every
    belief look relevant to every question.  Chinese text has no word
    boundaries, so character bigrams are used instead -- the same trick the
    renderer's noun extraction already uses.
    """
    if not text:
        return set()
    cjk = [ch for ch in text if _CJK_RANGES[0] <= ch <= _CJK_RANGES[1]]
    if cjk:
        if len(cjk) < 2:
            return {cjk[0]}
        return {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
    return {
        word
        for word in re.findall(r"\w+", text.lower())
        if len(word) >= min_length and word not in _STOP_WORDS
    }


def text_overlap(tokens: set[str], text: str | None) -> float:
    """Fraction of *tokens* that appear in *text*, in [0, 1].

    Used as the content-relevance signal in retrieval scoring.  A belief that
    carries more of what the question asks about scores higher, regardless of
    how the question happens to be phrased.
    """
    if not tokens or not text:
        return 0.0
    haystack = text.lower()
    return sum(1 for token in tokens if token in haystack) / len(tokens)
