"""Mirror Memory core constants.

Centralised thresholds and parameters -- no magic numbers scattered in code.
All values are generic (zero domain coupling). Domain-specific overrides
should be loaded from configuration at runtime.

Design principles:
- Every threshold has a name and a comment.
- Grouped by function: confidence / time / behaviour / limits.
- Change a value here; it takes effect everywhere.
"""

# -- Confidence thresholds ---------------------------------------------------

# Base signal floor: beliefs below this value are excluded from rendering,
# retrieval, and curiosity injection.  Corresponds to roughly one supporting
# evidence item (0.3 initial + SUPPORT_GAIN 0.15 = 0.45).
PROFILE_SIGNAL_THRESHOLD = 0.4

# Low-confidence pattern floor: beliefs above this but below the render
# threshold may trigger curiosity injection ("I'm sensing a pattern").
LOW_CONFIDENCE_PATTERN_FLOOR = 0.35

# Low-confidence pattern ceiling: beliefs at or above this value have
# already crossed the render watermark and no longer need curiosity.
LOW_CONFIDENCE_PATTERN_CEIL = 0.5

# L4 render watermark: L4 beliefs with confidence >= this value are
# included in prompt rendering.  Roughly 2 cross-session evidence items.
L4_RENDER_THRESHOLD = 0.55

# Question-candidate watermark: beliefs at or above this value AND with
# cross-session evidence >= 2 are eligible for questioning.
L4_QUESTION_THRESHOLD = 0.7

# Hard ceiling on belief confidence.
CONFIDENCE_CEILING = 0.95

# Per-evidence support gain (applied after confidence weight modulation).
SUPPORT_GAIN = 0.15

# Contradiction attenuation factor (multiplied with current confidence).
# 0.8 preserves the contradicted belief at 80% confidence rather than
# halving it — keeps both sides alive for verification instead of
# silently discarding user statements that may still be valid.
CONTRADICT_FACTOR = 0.8

# -- Time constants ----------------------------------------------------------

# Life events auto-expire after this many days (renders as past, not current).
LIFE_EVENT_EXPIRY_DAYS = 30

# Belief activity half-life: after this many days without new evidence the
# activity score halves (exponential decay).
ACTIVITY_HALF_LIFE_DAYS = 60.0

# Time-label segmentation (days).
TIME_LABEL_ACTIVE_DAYS = 7   # within 7 days  -> "active"
TIME_LABEL_WEEKLY_DAYS = 30  # within 30 days -> "Nw ago"
TIME_LABEL_MONTHLY_DAYS = 90 # within 90 days -> "Nmo ago"; beyond -> no label

# -- Behavioural signal constants --------------------------------------------

# VAD pause detection: silence longer than this counts as a pause (ms).
VAD_PAUSE_THRESHOLD_MS = 500

# VAD brief speech: speech duration below this with text content triggers a
# "spoke quickly" signal (ms).
VAD_BRIEF_SPEECH_MS = 2000

# VAD hesitation: pause count >= this AND speech > this duration (ms)
# triggers a "paused several times" signal.
VAD_HESITATION_PAUSE_COUNT = 3
VAD_HESITATION_MIN_SPEECH_MS = 5000

# Response delay: gap longer than this is considered a meaningful pause (s).
RESPONSE_DELAY_MIN_SECONDS = 300   # 5 minutes
RESPONSE_DELAY_MAX_SECONDS = 3600  # 60 minutes

# Message length drop: current message shorter than this ratio of the recent
# average is considered anomalous.
MESSAGE_LENGTH_DROP_RATIO = 0.4

# Default sleep window (hours) used when no UserTimeProfile is available.
DEFAULT_SLEEP_WINDOW = (23, 5)

# -- Quantity limits ---------------------------------------------------------

# Maximum claims per extraction turn.
MAX_CLAIMS_PER_TURN = 3

# Maximum evidence message-id references kept on a single belief row.
MAX_EVIDENCE_REFS = 8

# -- Question-tier weighting -------------------------------------------------

# Static tier weights for question-candidate prioritisation.  Higher tier
# means the dimension is more valuable to verify.  Override per-domain via
# ``question_value_tiers`` config key.
DEFAULT_QUESTION_TIER = 1

# Beta prior for confirm-rate shrinkage (Laplace smoothing).
# confirm_rate = (confirm_alpha) / (confirm_alpha + confirm_beta),
# starting at 0.5 with a weak prior that requires evidence to move.
BETA_PRIOR_ALPHA = 1
BETA_PRIOR_BETA = 1

# Confirm-rate feedback weight range (static tier * learning weight).
QUESTION_TIER_WEIGHT_MIN = 0.6
QUESTION_TIER_WEIGHT_MAX = 1.4

# -- Extraction constants ---------------------------------------------------

# K1 keyword confidence (deterministic extraction).
KEYWORD_CONFIDENCE = 0.4
PATTERN_CONFIDENCE = 0.5

# Maximum claim_text length stored per belief.
CLAIM_TEXT_MAX_LENGTH = 200

# -- Session summary constants ----------------------------------------------

# Maximum snippet length per message.
SNIPPET_MAX_LENGTH = 500

# Maximum combined summary length per session.
SESSION_SUMMARY_MAX_LENGTH = 3000

# Minimum text length to store as a snippet.
SNIPPET_MIN_LENGTH = 5

# -- Logging helpers ---------------------------------------------------------

# UUID truncation length for privacy-safe logging.
LOG_ID_LENGTH = 8
UUID_TRUNCATE_LENGTH = 16

# -- Retrieval constants ----------------------------------------------------

# Freshness threshold: beliefs with evidence newer than this (hours) get a boost.
FRESHNESS_THRESHOLD_HOURS = 2

# Multi-evidence threshold: beliefs with this many evidence items get a bonus.
MULTI_EVIDENCE_THRESHOLD = 3

# Query noun match threshold: fraction of query nouns that must appear in
# rendered beliefs before triggering session summary fallback.
QUERY_NOUN_MATCH_THRESHOLD = 0.5

# Budget overflow factor for session summary fallback.
SNIPPET_BUDGET_OVERFLOW_FACTOR = 2

# -- Worker / evolution constants --------------------------------------------

# Maximum number of active beliefs loaded as evidence for evolution.
LOAD_EVIDENCE_LIMIT = 30

# Maximum number of pending jobs processed per poll cycle.
JOB_BATCH_LIMIT = 5

# Minimum confidence for beliefs fed into the formulate LLM call.
FORMULATE_MIN_CONFIDENCE = 0.3
