"""Mirror Memory configuration loader.

Reads YAML files from a config directory and assembles them into a
:class:`MemoryConfig`.  Missing files fall back to empty defaults so that
partial configurations are valid.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from mirror_memory.exceptions import ConfigError

from .schema import (
    AnchorConfig,
    BudgetConfig,
    DisplayLabel,
    ExtractionConfig,
    MemoryConfig,
    DimensionConfig,
    PatternRule,
    PromptTemplates,
    SuppressionRule,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a single YAML file and return its contents as a dict.

    Returns an empty dict if the file does not exist or is empty.
    """
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    """Read a text file and return its contents.

    Returns an empty string if the file does not exist.
    """
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Parsers for individual sections
# ---------------------------------------------------------------------------


def _parse_dimensions(data: dict[str, Any]) -> list[DimensionConfig]:
    _KNOWN_DIMENSION_KEYS = {"dimensions"}
    unknown = set(data.keys()) - _KNOWN_DIMENSION_KEYS
    if unknown:
        logger.warning("Unknown keys in dimensions.yaml: %s", unknown)

    raw = data.get("dimensions", [])
    if not isinstance(raw, list):
        return []
    result: list[DimensionConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(
            DimensionConfig(
                dimension_id=item.get("id", item.get("dimension_id", "")),
                name=item.get("name", {}),
                description=item.get("description", ""),
                render_priority=item.get("render_priority", 0),
            )
        )
    return result


def _parse_anchors(data: dict[str, Any]) -> list[AnchorConfig]:
    _KNOWN_ANCHOR_KEYS = {"anchors"}
    unknown = set(data.keys()) - _KNOWN_ANCHOR_KEYS
    if unknown:
        logger.warning("Unknown keys in anchors.yaml: %s", unknown)

    raw = data.get("anchors", [])
    if not isinstance(raw, list):
        return []
    result: list[AnchorConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(
            AnchorConfig(
                dimension=item.get("dimension", ""),
                key=item.get("key", ""),
                phrases=item.get("phrases", []),
                identify_only=item.get("identify_only", False),
            )
        )
    return result


def _parse_display(data: dict[str, Any]) -> list[DisplayLabel]:
    _KNOWN_DISPLAY_KEYS = {"labels", "display"}
    unknown = set(data.keys()) - _KNOWN_DISPLAY_KEYS
    if unknown:
        logger.warning("Unknown keys in display.yaml: %s", unknown)

    raw = data.get("labels", data.get("display", []))
    if not isinstance(raw, list):
        return []
    result: list[DisplayLabel] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(
            DisplayLabel(
                key=item.get("key", ""),
                dimension=item.get("dimension", ""),
                zh=item.get("zh", ""),
                en=item.get("en", ""),
            )
        )
    return result


def _parse_extraction(data: dict[str, Any]) -> ExtractionConfig:
    _KNOWN_EXTRACTION_KEYS = {
        "keywords", "patterns", "context_tags",
        "max_claims_per_turn", "llm_every_turns", "llm_min_keyword_hits",
        "high_value_dimensions", "extraction_value_threshold",
        "suppression_rules",
    }
    unknown = set(data.keys()) - _KNOWN_EXTRACTION_KEYS
    if unknown:
        logger.warning("Unknown keys in extraction.yaml: %s", unknown)

    keywords = data.get("keywords", {})
    if not isinstance(keywords, dict):
        keywords = {}

    patterns_raw = data.get("patterns", [])
    patterns: list[PatternRule] = []
    if isinstance(patterns_raw, list):
        for p in patterns_raw:
            if not isinstance(p, dict):
                continue
            raw_regex = p.get("regex", "")
            try:
                patterns.append(
                    PatternRule(
                        regex=raw_regex,
                        dimension=p.get("dimension", ""),
                        key=p.get("key", ""),
                    )
                )
            except re.error as e:
                logger.warning("Skipping invalid regex pattern '%s': %s", raw_regex, e)
                continue
            except Exception:
                continue

    context_tags = data.get("context_tags", [])
    if not isinstance(context_tags, list):
        context_tags = []

    high_value_dimensions = data.get("high_value_dimensions", [])
    if not isinstance(high_value_dimensions, list):
        high_value_dimensions = []

    extraction_value_threshold = data.get("extraction_value_threshold", 0.5)
    try:
        extraction_value_threshold = float(extraction_value_threshold)
    except (TypeError, ValueError):
        logger.warning("Invalid extraction_value_threshold %r; using default 0.5", extraction_value_threshold)
        extraction_value_threshold = 0.5

    return ExtractionConfig(
        keywords=keywords,
        patterns=patterns,
        context_tags=[str(t) for t in context_tags],
        max_claims_per_turn=max(1, int(data.get("max_claims_per_turn", 3))),
        llm_every_turns=max(1, int(data.get("llm_every_turns", 5))),
        llm_min_keyword_hits=max(1, int(data.get("llm_min_keyword_hits", 2))),
        high_value_dimensions=[str(d) for d in high_value_dimensions],
        extraction_value_threshold=extraction_value_threshold,
        suppression_rules=[
            SuppressionRule(**r) for r in data.get("suppression_rules", [])
            if isinstance(r, dict) and r.get("context")
        ],
    )


def _parse_question_value_tiers(data: dict[str, Any]) -> dict[str, int]:
    """Parse the top-level ``question_value_tiers`` mapping (config.yaml)."""
    raw = data.get("question_value_tiers")
    if not isinstance(raw, dict):
        return {}
    tiers: dict[str, int] = {}
    for dim, tier in raw.items():
        try:
            tiers[str(dim)] = int(tier)
        except (TypeError, ValueError):
            logger.warning("Skipping invalid question_value_tiers entry %s=%r", dim, tier)
    return tiers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> MemoryConfig:
    """Load a :class:`MemoryConfig` from a directory of YAML / text files.

    Expected layout::

        config_path/
            dimensions.yaml      # dimensions list
            anchors.yaml         # anchors list
            display.yaml         # display labels
            extraction.yaml      # keywords, patterns, context_tags
            prompts/
                k2_system.txt
                k3_system.txt
                verification.txt
                formulate.txt

    Missing files are silently replaced with empty defaults.

    Returns
    -------
    MemoryConfig
        The assembled configuration object.

    Raises
    ------
    ConfigError
        If the config directory does not exist or validation fails.
    """
    path = Path(config_path)
    if not path.is_dir():
        raise ConfigError(f"Config directory does not exist: {config_path}")

    base = path

    # -- Sections from YAML files ---------------------------------------------
    dim_data = _read_yaml(base / "dimensions.yaml")
    anchor_data = _read_yaml(base / "anchors.yaml")
    display_data = _read_yaml(base / "display.yaml")
    extraction_data = _read_yaml(base / "extraction.yaml")

    # Budget may live in a top-level config.yaml
    top_data = _read_yaml(base / "config.yaml")
    budget_raw = top_data.get("budget", {})
    if isinstance(budget_raw, dict):
        budget = BudgetConfig(**{k: v for k, v in budget_raw.items() if k in {"base", "floor", "cap"}})
    else:
        budget = BudgetConfig()
    question_value_tiers = _parse_question_value_tiers(top_data)

    # -- Prompt templates from text files -------------------------------------
    prompts_dir = base / "prompts"
    prompts = PromptTemplates(
        k2_system=_read_text(prompts_dir / "k2_system.txt"),
        k3_system=_read_text(prompts_dir / "k3_system.txt"),
        verification=_read_text(prompts_dir / "verification.txt"),
        formulate=_read_text(prompts_dir / "formulate.txt"),
    )

    # -- Post-load validations ------------------------------------------------
    if not prompts.k2_system:
        logger.info("k2_system prompt is empty — K2 semantic extraction is disabled")

    # -- Assemble -------------------------------------------------------------
    config = MemoryConfig(
        dimensions=_parse_dimensions(dim_data),
        anchors=_parse_anchors(anchor_data),
        display_labels=_parse_display(display_data),
        extraction=_parse_extraction(extraction_data),
        prompts=prompts,
        budget=budget,
        question_value_tiers=question_value_tiers,
    )

    if config.render.floor > config.render.cap:
        raise ConfigError(
            f"render.floor ({config.render.floor}) must be <= render.cap ({config.render.cap})"
        )

    return config
