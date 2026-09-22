"""Tests for config/loader.py and config/schema.py."""


import pytest
import yaml

from mirror_memory.config.loader import load_config
from mirror_memory.config.schema import (
    AnchorConfig,
    BudgetConfig,
    DimensionConfig,
    MemoryConfig,
    PatternRule,
)
from mirror_memory.exceptions import ConfigError


class TestLoadConfig:
    def test_load_default_config(self):
        """Loading the bundled config/ directory should succeed."""
        cfg = load_config("config/")
        assert len(cfg.dimensions) > 0
        assert len(cfg.anchors) > 0

    def test_missing_directory_raises_config_error(self):
        with pytest.raises(ConfigError, match="does not exist"):
            load_config("/nonexistent/path/xyz")

    def test_empty_directory_loads_defaults(self, tmp_path):
        """Empty config directory should load with empty defaults."""
        cfg = load_config(str(tmp_path))
        assert cfg.dimensions == []
        assert cfg.anchors == []

    def test_budget_config_loaded_from_yaml(self, tmp_path):
        """Budget values from config.yaml should be applied."""
        d = tmp_path / "config.yaml"
        d.write_text(yaml.dump({"budget": {"base": 600, "floor": 200, "cap": 1000}}))
        cfg = load_config(str(tmp_path))
        assert cfg.budget.base == 600
        assert cfg.budget.floor == 200
        assert cfg.budget.cap == 1000

    def test_partial_config_files(self, tmp_path):
        """Only some YAML files present should load without error."""
        (tmp_path / "dimensions.yaml").write_text(yaml.dump({
            "dimensions": [{"dimension_id": "test", "name": {"zh": "测试", "en": "Test"}}]
        }))
        cfg = load_config(str(tmp_path))
        assert len(cfg.dimensions) == 1
        assert cfg.anchors == []  # missing file -> empty

    def test_budget_from_yaml(self, tmp_path):
        d = tmp_path / "config.yaml"
        d.write_text(yaml.dump({"budget": {"base": 600, "floor": 200, "cap": 1000}}))
        cfg = load_config(str(tmp_path))
        assert cfg.budget.base == 600
        assert cfg.budget.floor == 200
        assert cfg.budget.cap == 1000

    def test_budget_defaults_when_missing(self, tmp_path):
        cfg = load_config(str(tmp_path))
        assert cfg.budget.base == 480
        assert cfg.budget.floor == 160
        assert cfg.budget.cap == 720


class TestDimensionConfig:
    def test_valid_dimension(self):
        d = DimensionConfig(dimension_id="topic", name={"zh": "话题", "en": "Topic"})
        assert d.dimension_id == "topic"
        assert d.render_priority == 0

    def test_empty_dimension_id_raises(self):
        with pytest.raises(Exception):
            DimensionConfig(dimension_id="")


class TestAnchorConfig:
    def test_default_anchor_id_empty(self):
        """anchor_id defaults to empty; auto-generation happens in loader."""
        a = AnchorConfig(dimension="topic", key="sleep", phrases=["sleep", "insomnia"])
        assert a.anchor_id == ""

    def test_explicit_anchor_id(self):
        a = AnchorConfig(anchor_id="custom", dimension="topic", key="sleep", phrases=["sleep"])
        assert a.anchor_id == "custom"

    def test_anchors_loaded_from_config(self):
        """Anchors loaded via load_config should have dimension and key."""
        cfg = load_config("config/")
        assert len(cfg.anchors) > 0
        for a in cfg.anchors[:5]:
            assert a.dimension
            assert a.key


class TestPatternRule:
    def test_valid_regex(self):
        p = PatternRule(regex=r"\d+", dimension="fact", key="number")
        assert p.regex == r"\d+"

    def test_invalid_regex_raises(self):
        with pytest.raises(Exception, match="regex"):
            PatternRule(regex="[invalid", dimension="fact", key="bad")


class TestBudgetConfig:
    def test_defaults(self):
        b = BudgetConfig()
        assert b.base == 480
        assert b.floor == 160
        assert b.cap == 720

    def test_custom_values(self):
        b = BudgetConfig(base=600, floor=200, cap=1000)
        assert b.base == 600

    def test_negative_base_raises(self):
        with pytest.raises(Exception):
            BudgetConfig(base=-1)

    def test_floor_below_minimum_raises(self):
        with pytest.raises(Exception):
            BudgetConfig(floor=10)  # ge=50


class TestMemoryConfig:
    def test_get_dimension_found(self):
        cfg = load_config("config/")
        dim = cfg.get_dimension("topic")
        assert dim is not None
        assert dim.dimension_id == "topic"

    def test_get_dimension_not_found(self):
        cfg = load_config("config/")
        assert cfg.get_dimension("nonexistent") is None

    def test_anchors_for_dimension(self):
        cfg = load_config("config/")
        anchors = cfg.anchors_for_dimension("topic")
        assert len(anchors) > 0
        assert all(a.dimension == "topic" for a in anchors)

    def test_labels_for_dimension(self):
        cfg = load_config("config/")
        labels = cfg.labels_for_dimension("topic")
        assert isinstance(labels, list)

    def test_session_summary_default_false(self):
        cfg = MemoryConfig()
        assert cfg.session_summary_enabled is False


class TestOpenAILLMExtraBody:
    """extra_body passes provider-specific fields through to the API.

    Reasoning models spend their token budget on thinking and can return an
    empty completion, which the extractor's fallback silently turns into
    "no claims" -- so a misconfigured thinking mode looks like a model that
    finds nothing.
    """

    @staticmethod
    def _mock_openai():
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))],
            usage=None,
        )
        return patch("openai.OpenAI", return_value=mock_client), mock_client

    def test_extra_body_is_forwarded(self):
        from mirror_memory.llm import OpenAILLM

        patcher, mock_client = self._mock_openai()
        with patcher:
            llm = OpenAILLM(
                api_key="k", model="mimo-v2.5",
                extra_body={"thinking": {"type": "disabled"}},
            )
            llm.generate(system_prompt="s", payload_text="p")

            kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_no_extra_body_omits_the_field(self):
        from mirror_memory.llm import OpenAILLM

        patcher, mock_client = self._mock_openai()
        with patcher:
            llm = OpenAILLM(api_key="k", model="m")
            llm.generate(system_prompt="s", payload_text="p")

            kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert "extra_body" not in kwargs
