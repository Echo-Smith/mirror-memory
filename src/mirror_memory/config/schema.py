"""Mirror Memory configuration schema.

Pydantic v2 models that define the shape of every configuration file.
Zero domain coupling -- all fields are generic and reusable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


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
    """A regex-based extraction pattern."""

    regex: str = Field(..., min_length=1, description="Python regex pattern")
    dimension: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)

    @field_validator("regex", mode="after")
    @classmethod
    def _validate_regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc
        return v


class ExtractionConfig(BaseModel):
    """Keyword lists, regex patterns, and throttle parameters for extraction."""

    keywords: dict[str, list[str]] = Field(default_factory=dict)
    patterns: list[PatternRule] = Field(default_factory=list)
    context_tags: list[str] = Field(default_factory=list)
    max_claims_per_turn: int = Field(default=3, ge=1)
    llm_every_turns: int = Field(default=5, ge=1, description="LLM extraction fires every N turns at minimum")
    llm_min_keyword_hits: int = Field(default=2, ge=1, description="Minimum keyword hits to trigger LLM regardless of turn count")
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
