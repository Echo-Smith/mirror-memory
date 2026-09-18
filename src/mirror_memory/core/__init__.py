"""Mirror Memory core -- structured belief lifecycle, confidence weighting,
retrieval scoring, and behavioural signal detection.

All modules are domain-agnostic.  Domain-specific configuration (question
templates, dimension tiers, blocked key prefixes, etc.) is loaded at runtime
by the caller.
"""

# -- Models ------------------------------------------------------------------
from mirror_memory.core.models import (
    Base,
    Belief,
    BeliefEvent,
    EvolutionJob,
    ExtractionStats,
    MemoryPreference,
    Snapshot,
    User,
    utcnow,
)

# -- Constants ---------------------------------------------------------------
from mirror_memory.core.constants import (
    ACTIVITY_HALF_LIFE_DAYS,
    CONFIDENCE_CEILING,
    CONTRADICT_FACTOR,
    DEFAULT_QUESTION_TIER,
    DEFAULT_SLEEP_WINDOW,
    L4_QUESTION_THRESHOLD,
    L4_RENDER_THRESHOLD,
    LOW_CONFIDENCE_PATTERN_CEIL,
    LOW_CONFIDENCE_PATTERN_FLOOR,
    MAX_EVIDENCE_REFS,
    MAX_CLAIMS_PER_TURN,
    MESSAGE_LENGTH_DROP_RATIO,
    PROFILE_SIGNAL_THRESHOLD,
    QUESTION_TIER_WEIGHT_MAX,
    QUESTION_TIER_WEIGHT_MIN,
    RESPONSE_DELAY_MAX_SECONDS,
    RESPONSE_DELAY_MIN_SECONDS,
    SUPPORT_GAIN,
    TIME_LABEL_ACTIVE_DAYS,
    TIME_LABEL_MONTHLY_DAYS,
    TIME_LABEL_WEEKLY_DAYS,
    VAD_BRIEF_SPEECH_MS,
    VAD_HESITATION_MIN_SPEECH_MS,
    VAD_HESITATION_PAUSE_COUNT,
    VAD_PAUSE_THRESHOLD_MS,
)

# -- Pure functions ----------------------------------------------------------
from mirror_memory.core.confidence import compute_confidence_weight
from mirror_memory.core.activity import belief_activity
from mirror_memory.core.budget import compute_profile_budget
from mirror_memory.core.retrieval import score_belief
from mirror_memory.core.behavioral import detect_behavioral_signals

# -- Repository --------------------------------------------------------------
from mirror_memory.core.repository import (
    confirm_belief,
    delete_user_memories,
    get_active_belief,
    get_belief,
    get_belief_events,
    is_memory_enabled,
    list_active_beliefs,
    list_question_candidates,
    list_rejected_beliefs,
    record_claim,
    record_extraction_stats,
    reject_belief,
    set_memory_enabled,
    update_belief_value,
)

__all__ = [
    # Models
    "Base",
    "Belief",
    "BeliefEvent",
    "EvolutionJob",
    "ExtractionStats",
    "MemoryPreference",
    "Snapshot",
    "User",
    "utcnow",
    # Constants
    "ACTIVITY_HALF_LIFE_DAYS",
    "CONFIDENCE_CEILING",
    "CONTRADICT_FACTOR",
    "DEFAULT_QUESTION_TIER",
    "DEFAULT_SLEEP_WINDOW",
    "L4_QUESTION_THRESHOLD",
    "L4_RENDER_THRESHOLD",
    "LOW_CONFIDENCE_PATTERN_CEIL",
    "LOW_CONFIDENCE_PATTERN_FLOOR",
    "MAX_EVIDENCE_REFS",
    "MAX_CLAIMS_PER_TURN",
    "MESSAGE_LENGTH_DROP_RATIO",
    "PROFILE_SIGNAL_THRESHOLD",
    "QUESTION_TIER_WEIGHT_MAX",
    "QUESTION_TIER_WEIGHT_MIN",
    "RESPONSE_DELAY_MAX_SECONDS",
    "RESPONSE_DELAY_MIN_SECONDS",
    "SUPPORT_GAIN",
    "TIME_LABEL_ACTIVE_DAYS",
    "TIME_LABEL_MONTHLY_DAYS",
    "TIME_LABEL_WEEKLY_DAYS",
    "VAD_BRIEF_SPEECH_MS",
    "VAD_HESITATION_MIN_SPEECH_MS",
    "VAD_HESITATION_PAUSE_COUNT",
    "VAD_PAUSE_THRESHOLD_MS",
    # Pure functions
    "compute_confidence_weight",
    "belief_activity",
    "compute_profile_budget",
    "score_belief",
    "detect_behavioral_signals",
    # Repository
    "confirm_belief",
    "delete_user_memories",
    "get_active_belief",
    "get_belief",
    "get_belief_events",
    "is_memory_enabled",
    "list_active_beliefs",
    "list_question_candidates",
    "list_rejected_beliefs",
    "record_claim",
    "record_extraction_stats",
    "reject_belief",
    "set_memory_enabled",
    "update_belief_value",
]
