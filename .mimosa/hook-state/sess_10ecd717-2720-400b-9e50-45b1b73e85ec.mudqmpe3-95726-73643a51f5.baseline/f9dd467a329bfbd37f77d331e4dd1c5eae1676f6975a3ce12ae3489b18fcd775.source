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
    if not synonym_map:
        return norm
    direct = synonym_map.get(norm)
    if direct:
        return direct
    # Auxiliary-verb stripping: "was_hired_by" is the same slot as "hired_by".
    # Without this every tense/voice variant needs its own synonym entry, and
    # each miss silently splits one attribute into two unrelated beliefs.
    for prefix in ("was_", "is_", "are_", "been_", "being_", "have_", "has_",
                   "had_", "will_", "would_", "do_", "does_", "did_"):
        if norm.startswith(prefix):
            stripped = norm[len(prefix):]
            hit = synonym_map.get(stripped)
            return hit or _tense_fallback(stripped, synonym_map)
    return _tense_fallback(norm, synonym_map)


def _tense_fallback(norm: str, synonym_map: dict[str, str]) -> str:
    """Strip a trailing -ed from the verb root and retry the lookup.

    "worked_at" is the same employment slot as "works_at"; without this the
    past-tense form becomes its own predicate and never gets superseded, so
    a current-state query keeps returning the employer the user left.
    """
    head, sep, tail = norm.partition("_")
    for candidate in (head + sep + tail, head[:-2] + sep + tail if head.endswith("ed") else None):
        if candidate and candidate in synonym_map:
            return synonym_map[candidate]
    if head.endswith("ed") and not sep:
        stem = head[:-2]
        if stem in synonym_map:
            return synonym_map[stem]
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
