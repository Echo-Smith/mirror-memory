"""Canonicalization — normalize predicate and object to canonical forms.

Without canonicalization, ``likes coffee``, ``loves coffee``, and
``enjoys coffee`` would produce three separate atoms.  The canonicalizer
maps them all to ``predicate=likes, object=coffee``.

Two layers:
1. **Rule layer** (fast, deterministic): lowercase, strip punctuation,
   synonym mapping from config.
2. **LLM layer** (optional, higher quality): asks the LLM to canonicalize
   complex or ambiguous phrases.

Only the rule layer is used by default.  The LLM layer is opt-in.
"""

from __future__ import annotations

import re
from typing import Any


def canonicalize_predicate(predicate: str, synonym_map: dict[str, str] | None = None) -> str:
    """Normalize a predicate to its canonical form.

    Steps:
    1. lowercase + strip
    2. replace spaces/hyphens with underscores
    3. look up in synonym_map (if provided)
    """
    if not predicate:
        return ""
    norm = predicate.lower().strip()
    norm = re.sub(r"[\s\-]+", "_", norm)
    if synonym_map:
        return synonym_map.get(norm, norm)
    return norm


def canonicalize_object(obj: str) -> str:
    """Normalize an object to its canonical form.

    Steps:
    1. lowercase + strip
    2. collapse whitespace
    3. remove leading articles (a, an, the)
    """
    if not obj:
        return ""
    norm = obj.lower().strip()
    norm = re.sub(r"\s+", "_", norm)
    # Remove leading articles.
    for article in ("the_", "a_", "an_"):
        if norm.startswith(article):
            norm = norm[len(article):]
            break
    return norm


def canonicalize_atom(
    subject: str,
    predicate: str,
    obj: str,
    synonym_map: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    """Canonicalize all three parts of a cognitive triple.

    Returns (subject, canonical_predicate, canonical_object).
    """
    return (
        subject.lower().strip(),
        canonicalize_predicate(predicate, synonym_map),
        canonicalize_object(obj),
    )
