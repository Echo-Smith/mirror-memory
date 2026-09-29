"""Mirror Memory configuration schema.

Pydantic v2 models that define the shape of every configuration file.
Zero domain coupling -- all fields are generic and reusable.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Dimension
# ---------------------------------------------------------------------------


class DimensionConfig(BaseModel):
    """A single dimension definition (e.g. topic, preference, fact)."""

    dimension_id: str = Field(..., min_length=1, description="Unique identifier")
    name: dict[str, str] = Field(default_factory=lambda: {"zh": "", "en": ""})
    description: str = ""
    render_priority: int = Field(default=0, ge=0)
    requires_user_confirmation: bool = Field(
        default=False,
        description="Beliefs in this dimension must have source='user_confirmed' to render",
    )


# ---------------------------------------------------------------------------
# Anchor
# ---------------------------------------------------------------------------


class AnchorConfig(BaseModel):
    """A word-anchor used for deterministic extraction and closed-set checks."""

    anchor_id: str = Field(default="", description="Auto-generated if empty")
    dimension: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    phrases: list[str] = Field(default_factory=list)
    identify_only: bool = Field(
        default=False,
        description="If true, the anchor triggers identification but no automatic claim creation",
    )
    source: str = Field(
        default="",
        description="Provenance reference for the anchor phrases (e.g. 'act:core_process:typical_phrases')",
    )
    live_bind: bool = Field(
        default=False,
        description="If true, the anchor key is also accepted in K2 extraction even when "
        "no static phrases match — the anchor is 'alive' because the user has beliefs with this key",
    )

    @field_validator("anchor_id", mode="before")
    @classmethod
    def _default_anchor_id(cls, v: str, info: Any) -> str:
        if not v:
            dim = info.data.get("dimension", "?")
            key = info.data.get("key", "?")
            return f"{dim}:{key}"
        return v


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


class DisplayLabel(BaseModel):
    """A display label mapping for one (dimension, key) pair."""

    key: str = Field(..., min_length=1)
    dimension: str = Field(..., min_length=1)
    zh: str = ""
    en: str = ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class PatternRule(BaseModel):
    """A regex-based extraction pattern.

    ``predicate`` and ``object_group`` turn a pattern from a text snippet into
    a cognitive triple.  A pattern that declares a predicate asserts a slot
    (``lives_in``), and ``object_group`` names the capture group holding the
    value; without them the claim has no triple, cannot be matched against
    existing beliefs, and would sit in the current-state surface forever.
    """

    regex: str = Field(..., min_length=1, description="Python regex pattern")
    dimension: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    predicate: str = Field(
        default="",
        description=(
            "Canonical predicate this pattern asserts (e.g. 'lives_in'). "
            "Empty keeps the claim triple-less."
        ),
    )
    object_group: int = Field(
        default=1,
        ge=1,
        description="Capture group holding the object text",
    )

    @field_validator("regex", mode="after")
    @classmethod
    def _validate_regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc
        return v

    @model_validator(mode="after")
    def _validate_object_group(self) -> "PatternRule":
        if self.predicate:
            groups = re.compile(self.regex).groups
            if self.object_group > groups:
                raise ValueError(
                    f"object_group={self.object_group} exceeds the "
                    f"{groups} capture group(s) in regex {self.regex!r}"
                )
        return self


class SuppressionRule(BaseModel):
    """Suppress extraction in specific contexts.

    Example: during a questionnaire, don't extract topic claims
    because the user is answering structured items, not chatting.
    """

    context: str = Field(..., min_length=1, description="Context identifier (e.g. 'assessment', 'exercise')")
    suppress_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions to suppress in this context. Empty = suppress all.",
    )


class ExtractionConfig(BaseModel):
    """Keyword lists, regex patterns, and throttle parameters for extraction."""

    keywords: dict[str, list[str]] = Field(default_factory=dict)
    patterns: list[PatternRule] = Field(default_factory=list)
    context_tags: list[str] = Field(default_factory=list)
    # Must exceed the K2 parser's own self-cap (MAX_CLAIMS_PER_TURN = 3).  At 3
    # the two caps coincided, so a full K2 batch consumed the entire budget and
    # every K1 claim was structurally excluded -- including any carrying the
    # answer.  Raising it cut the assembler's drop rate from 16.3% to 0.9%; it
    # did not by itself move the benchmark, because most of the apparent
    # extraction loss turned out to be a measurement artifact rather than
    # claims being discarded.
    max_claims_per_turn: int = Field(default=6, ge=1)
    llm_every_turns: int = Field(default=5, ge=1, description="LLM extraction fires every N turns at minimum")
    llm_min_keyword_hits: int = Field(
        default=2,
        ge=0,
        description="Minimum keyword hits to trigger LLM; zero forces extraction every turn",
    )
    high_value_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions that get a scoring boost for extraction priority",
    )
    extraction_value_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Information-gain score ([0,1]) at or above which extraction fires early, "
        "breaking the uniform every-N schedule",
    )
    suppression_rules: list[SuppressionRule] = Field(
        default_factory=list,
        description="Context-based extraction suppression rules",
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


class PromptTemplates(BaseModel):
    """Named prompt templates for the K-pipeline stages."""

    k2_system: str = Field(default="", description="K2 semantic extraction system prompt")
    k3_system: str = Field(default="", description="K3 synthesis system prompt")
    verification: str = Field(default="", description="Verification prompt")
    formulate: str = Field(default="", description="Worker formulate prompt")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class BudgetConfig(BaseModel):
    """Memory block rendering budget parameters."""

    base: int = Field(default=480, ge=50, description="Base character budget")
    floor: int = Field(default=160, ge=50, description="Minimum budget")
    cap: int = Field(default=720, ge=100, description="Maximum budget")


class RenderConfig(BaseModel):
    """Parameters for memory-block rendering."""

    base: int = Field(default=1200, description="Nominal character budget at zero pressure")
    floor: int = Field(default=400, description="Minimum character budget")
    cap: int = Field(default=2000, description="Maximum character budget")
    max_per_dimension: int = Field(default=3, ge=1, description="Max items per dimension in rendered block")
    # Higher per-dimension cap used when the query asks for everything of a
    # kind ("what languages do they speak"). The character budget still bounds
    # the total, so this trades breadth for length rather than dropping facts.
    enumeration_max_per_dimension: int = Field(
        default=12, ge=1,
        description="Per-dimension cap for enumeration queries",
    )
    l4_render_threshold: float = Field(default=0.55, description="Min confidence for L4 beliefs to render")
    activity_half_life_days: float = Field(default=60.0, description="Half-life for belief activity time-decay")


class WorkerConfig(BaseModel):
    """Parameters for the async evolution worker."""

    poll_interval_seconds: int = Field(default=60)
    job_lease_seconds: int = Field(default=300)
    max_attempts: int = Field(default=3)
    prompt_version: str = Field(default="v1")
    forbidden_labels: list[str] = Field(default_factory=list)
    third_party_markers: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory metabolism (lifecycle runtime)
# ---------------------------------------------------------------------------


class RetentionClassPolicy(BaseModel):
    """Cooling schedule for one retention class.

    ``None`` means "no automatic move at this stage" — the canonical class
    (name, employer, long-lived facts) is never cooled or archived by the
    background runtime on age alone.
    """

    cool_after_days: int | None = Field(default=None, ge=1)
    archive_after_days: int | None = Field(default=None, ge=1)


class HeatWeights(BaseModel):
    """Weights of the heat-score signals; they should sum to ~1.0."""

    freshness: float = Field(default=0.30, ge=0, le=1)
    access: float = Field(default=0.20, ge=0, le=1)
    evidence_strength: float = Field(default=0.20, ge=0, le=1)
    authority: float = Field(default=0.15, ge=0, le=1)
    importance: float = Field(default=0.15, ge=0, le=1)


class HeatThresholds(BaseModel):
    """Heat-score cut points mapping onto storage tiers."""

    hot: float = Field(default=0.70, ge=0, le=1)
    warm: float = Field(default=0.40, ge=0, le=1)
    dormant: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "HeatThresholds":
        if not (self.hot > self.warm > self.dormant):
            raise ValueError(
                f"heat thresholds must satisfy hot > warm > dormant, got "
                f"{self.hot} > {self.warm} > {self.dormant}"
            )
        return self


class HeatConfig(BaseModel):
    """Heat-score parameters (see ``mirror_memory.metabolism.heat``)."""

    weights: HeatWeights = Field(default_factory=HeatWeights)
    half_life_days: float = Field(default=60.0, gt=0)
    thresholds: HeatThresholds = Field(default_factory=HeatThresholds)
    protected_floor: float = Field(default=0.70, ge=0, le=1)


class MetabolismPlannerConfig(BaseModel):
    """Safety rails for one metabolism cycle (see ``metabolism.planner``)."""

    max_transitions_per_run: int = Field(
        default=200, ge=1,
        description="Upper bound on tier transitions + compactions proposed per run; excess "
        "cold beliefs are handled on the following run instead",
    )


class CompactionKeepPolicy(BaseModel):
    """How many support rows survive evidence compaction, per rule."""

    oldest: int = Field(default=1, ge=1, le=10)
    recent: int = Field(default=3, ge=1, le=10)
    highest_authority: int = Field(default=2, ge=1, le=10)


class CompactionConfig(BaseModel):
    """Evidence-compaction thresholds (see ``metabolism.compact``)."""

    min_support_links: int = Field(
        default=12, ge=1, le=1000,
        description="Compaction starts strictly above this many support links on one belief",
    )
    keep: CompactionKeepPolicy = Field(default_factory=CompactionKeepPolicy)


class LedgerConfig(BaseModel):
    """Bounds on the versioned belief ledger.

    Without them the version table grows without limit: every value a slot
    has ever held is a row, and a heavily-changed slot (a goal abandoned
    and resumed monthly) would accumulate one per change forever.
    """

    max_versions_per_identity: int = Field(
        default=50, ge=2, le=1000,
        description="Newest N versions kept per slot; older ones are pruned when a new one opens",
    )
    version_ttl_days: int = Field(
        default=730, ge=30, le=3650,
        description="Versions whose belief interval closed longer ago than this are pruned",
    )


class MetabolismConfig(BaseModel):
    """Memory-metabolism settings, loaded from ``metabolism.yaml``.

    Values here drive rule-based lifecycle decisions only — no model ever
    decides what to cool, compact, or delete.
    """

    retention_classes: dict[str, RetentionClassPolicy] = Field(
        default_factory=lambda: {
            "canonical": RetentionClassPolicy(cool_after_days=None, archive_after_days=None),
            "preference": RetentionClassPolicy(cool_after_days=180, archive_after_days=720),
            "behavioral": RetentionClassPolicy(cool_after_days=60, archive_after_days=180),
            "episodic": RetentionClassPolicy(cool_after_days=30, archive_after_days=90),
            "transient": RetentionClassPolicy(cool_after_days=7, archive_after_days=30),
        },
        description="Retention class -> cooling schedule. archive_after_days is not deletion: "
        "archived data simply leaves the default retrieval scan.",
    )
    retention_predicates: dict[str, str] = Field(
        default_factory=dict,
        description="Explicit predicate -> retention class overrides; beat every derived rule",
    )
    default_retention_class: str = Field(
        default="preference",
        description="Fallback class for claims with no usable identity signal",
    )
    heat: HeatConfig = Field(default_factory=HeatConfig)
    planner: MetabolismPlannerConfig = Field(default_factory=MetabolismPlannerConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)

    @field_validator("default_retention_class")
    @classmethod
    def _known_default_class(cls, v: str) -> str:
        from mirror_memory.metabolism.policy import RETENTION_CLASSES

        if v not in RETENTION_CLASSES:
            raise ValueError(f"default_retention_class must be one of {RETENTION_CLASSES}, got {v!r}")
        return v


class MemoryConfig(BaseModel):
    """Root configuration object assembled from multiple YAML files.

    Runtime-injected fields (``llm_client``, ``session_factory``) use
    ``Any`` type and ``None`` default.  They are set after loading.
    """

    model_config = {"arbitrary_types_allowed": True}

    dimensions: list[DimensionConfig] = Field(default_factory=list)
    anchors: list[AnchorConfig] = Field(default_factory=list)
    display_labels: list[DisplayLabel] = Field(default_factory=list)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    prompts: PromptTemplates = Field(default_factory=PromptTemplates)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    blocked_key_prefixes: list[str] = Field(default_factory=list)
    render: RenderConfig = Field(default_factory=RenderConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    question_value_tiers: dict[str, int] = Field(
        default_factory=dict,
        description="Dimension -> priority tier for verification question candidates "
        "(higher = more valuable to verify). Dimensions absent from the map use "
        "DEFAULT_QUESTION_TIER.",
    )
    session_summary_enabled: bool = Field(
        default=False,
        description="Enable unstructured session summary storage for factual recall. "
        "Off by default — structured beliefs suffice for most use cases.",
    )
    strict_dimensions: bool = Field(
        default=False,
        description="If True, record_claim raises ValueError for dimensions "
        "not present in config.dimensions.",
    )
    # Identity policy — maps predicates to cardinality (single/multi/event).
    identity_policy: dict[str, str] = Field(
        default_factory=dict,
        description="Predicate → cardinality mapping for IdentityResolver",
    )
    # Predicate → temporal scope (current_state / persistent / episodic).
    # Read from identity_policy.yaml's `temporal` key alongside cardinality;
    # drives the lifecycle transition instead of the old "SINGLE → UPDATE".
    temporal_policy: dict[str, str] = Field(
        default_factory=dict,
        description="Predicate → temporal scope mapping for the lifecycle policy",
    )
    # Predicate synonym map — canonicalizes variant predicates.
    predicate_synonyms: dict[str, str] = Field(
        default_factory=dict,
        description="Variant predicate → canonical predicate (e.g. loves → likes)",
    )
    # Memory metabolism — retention classes, heat scoring, tier eligibility.
    metabolism: MetabolismConfig = Field(
        default_factory=MetabolismConfig,
        description="Memory-metabolism settings from metabolism.yaml",
    )
    llm_client: Any = Field(default=None, description="LLM client; must expose generate()")
    session_factory: Any = Field(default=None, description="Callable returning a new SQLAlchemy Session")

    # -- Convenience lookups ---------------------------------------------------

    def get_dimension(self, dimension_id: str) -> DimensionConfig | None:
        """Return the dimension with the given id, or None."""
        for d in self.dimensions:
            if d.dimension_id == dimension_id:
                return d
        return None

    def anchors_for_dimension(self, dimension: str) -> list[AnchorConfig]:
        """Return all anchors belonging to *dimension*."""
        return [a for a in self.anchors if a.dimension == dimension]

    def labels_for_dimension(self, dimension: str) -> list[DisplayLabel]:
        """Return all display labels belonging to *dimension*."""
        return [lbl for lbl in self.display_labels if lbl.dimension == dimension]

    def anchor_phrase_set(self, dimension: str) -> set[str]:
        """Lower-cased union of all anchor phrases for a dimension."""
        phrases: set[str] = set()
        for a in self.anchors_for_dimension(dimension):
            phrases.update(p.lower() for p in a.phrases)
        return phrases
