"""Shared utility functions for mirror-memory.

Extracted to eliminate duplication across modules:
- ``safe_json`` — parse JSON with fallback (used 6+ times)
- ``parse_llm_json`` — parse LLM output that may contain markdown fences (used 3 times)
- ``truncate_id`` — truncate identifiers for logging (used 4+ times)
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


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
