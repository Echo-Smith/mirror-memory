"""Query intent and retrieval eligibility — filter *before* ranking.

The recall path is "intent → eligibility → candidates → ranking", not
"all beliefs → score → top-k".  Eligibility is where storage tiers and
query intent meet: a question about the present should not pay for scanning
cold storage, and a question about the past must be allowed to reach it::

    current          active + hot/warm
    historical       superseded history + warm/dormant (via the default scan)
    evidence         belief + its evidence graph (all non-archived tiers)
    experience       episodic history + session summaries
    all_occurrences  every interval, no currency filter
    all              no intent signal; the default scan

Query-mode detection lives here (public and testable); the renderer imports
it.  Archived beliefs are excluded from every default scan — reaching them
is an explicit restore (PR4), never a silent side effect of a query.

Marker tables cover English and Chinese surface forms.  Detection order
matters: enumeration/all-occurrences wins (a count question must not be
narrowed), then historical (a past question must reach superseded rows),
then evidence, then experience, then current, else "all".
"""

from __future__ import annotations

from mirror_memory.metabolism.tiers import TIER_DORMANT, TIER_HOT, TIER_WARM

# -- Query modes ---------------------------------------------------------------

MODE_CURRENT = "current"
MODE_HISTORICAL = "historical"
MODE_ALL_OCCURRENCES = "all_occurrences"
MODE_EVIDENCE = "evidence"
MODE_EXPERIENCE = "experience"
MODE_ALL = "all"

QUERY_MODES = (
    MODE_CURRENT,
    MODE_HISTORICAL,
    MODE_ALL_OCCURRENCES,
    MODE_EVIDENCE,
    MODE_EXPERIENCE,
    MODE_ALL,
)

# Everything the default scan may see.  ARCHIVED is reachable only through
# an explicit restore, never through a routine query.
DEFAULT_SCAN_TIERS = (TIER_HOT, TIER_WARM, TIER_DORMANT)
# A question about the present never pays for cold storage.
CURRENT_SCAN_TIERS = (TIER_HOT, TIER_WARM)


def tiers_for_mode(mode: str) -> tuple[str, ...]:
    """Storage tiers a query in *mode* may scan."""
    if mode == MODE_CURRENT:
        return CURRENT_SCAN_TIERS
    return DEFAULT_SCAN_TIERS


def tier_eligible(memory_tier: str, mode: str) -> bool:
    """Is a belief in *memory_tier* eligible for a query in *mode*?"""
    return memory_tier in tiers_for_mode(mode)


# -- Surface markers -----------------------------------------------------------

_CURRENT_MARKERS = (
    "now", "currently", "these days", "at the moment", "right now",
    "现在", "目前", "当前",
)
_HISTORICAL_MARKERS = (
    "used to", "previously", "before", "earlier", "in the past", "formerly",
    "no longer", "used to live", "did they", "did you",
    "以前", "曾经", "之前", "过去", "曾经住",
)
_ALL_OCCURRENCES_MARKERS = (
    "how many times", "list all", "all the times", "every time",
    "what all", "which all", "all the places", "all the cities",
    "哪些次", "所有次", "各次", "总共",
)
_EVIDENCE_MARKERS = (
    "why", "based on what", "what evidence", "according to what",
    "what makes you", "how do you know",
    "为什么", "根据什么", "依据", "证据", "凭什么",
)
_EXPERIENCE_MARKERS = (
    "what happened", "what did they experience", "tell me about the time",
    "what have they been through",
    "发生过什么", "经历过", "经历", "怎么样了",
)
_ENUMERATION_MARKERS = (
    "what all", "list all", "all the", "every ", "which all", "what languages",
    "what sports", "what pets", "what instruments", "what food", "what books",
    "what cities", "what places", "what countries", "what schools",
    "what companies", "what jobs", "what races", "what classes",
    "what workshops", "what courses", "what appointments", "what gifts",
    "what did they buy", "what did they visit", "what did they attend",
    "what did they read", "what did they watch", "what did they eat",
    "what hobbies", "what skills", "what are their", "what are the",
    "哪些", "所有", "各种", "列出",
)


def is_enumeration_query(query: str) -> bool:
    """Does this query ask for every matching item rather than one value?"""
    if not query:
        return False
    lowered = query.lower()
    return any(marker in lowered for marker in _ENUMERATION_MARKERS)


def detect_query_mode(query: str) -> str:
    """Map a query onto a retrieval mode.

    Returns one of :data:`QUERY_MODES`.  Historical wins over current on a
    tie ("where did you used to live" is a past question despite "did");
    all-occurrences wins over both (a count needs every interval); evidence
    and experience only claim queries the temporal markers did not already
    resolve, so existing interval behaviour is unchanged.
    """
    if not query:
        return MODE_ALL
    lowered = query.lower()
    if is_enumeration_query(query):
        return MODE_ALL_OCCURRENCES
    if any(marker in lowered for marker in _ALL_OCCURRENCES_MARKERS):
        return MODE_ALL_OCCURRENCES
    for marker in _HISTORICAL_MARKERS:
        if marker in lowered:
            return MODE_HISTORICAL
    for marker in _EVIDENCE_MARKERS:
        if marker in lowered:
            return MODE_EVIDENCE
    for marker in _EXPERIENCE_MARKERS:
        if marker in lowered:
            return MODE_EXPERIENCE
    for marker in _CURRENT_MARKERS:
        if marker in lowered:
            return MODE_CURRENT
    return MODE_ALL
