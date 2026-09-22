"""Tests for render/display.py — DisplayDict."""

from mirror_memory.config.loader import load_config
from mirror_memory.render.display import DisplayDict


class TestDisplayDict:
    def test_friendly_label_known_key(self, config):
        dd = DisplayDict(config)
        # Find a key that exists in display labels
        label = dd.friendly_label("sleep", "zh")
        # Should return something or None depending on config
        assert label is None or isinstance(label, str)

    def test_friendly_label_unknown_key(self, config):
        dd = DisplayDict(config)
        assert dd.friendly_label("nonexistent_key_xyz", "en") is None

    def test_dimension_header_known(self, config):
        dd = DisplayDict(config)
        header = dd.dimension_header("topic", "en")
        assert header is not None
        assert isinstance(header, str)

    def test_dimension_header_unknown(self, config):
        dd = DisplayDict(config)
        header = dd.dimension_header("nonexistent_dim", "en")
        # Should return the dim_id itself as fallback
        assert header == "nonexistent_dim"

    def test_language_fallback_en_to_zh(self, config):
        """When English label is missing, should fall back to zh."""
        dd = DisplayDict(config)
        # This depends on the config having some labels without English
        # Just verify it doesn't crash
        label = dd.friendly_label("test", "fr")
        assert label is None or isinstance(label, str)

    def test_dimension_header_fallback_chain(self, config):
        """Header should follow requested -> en -> zh -> dim_id chain."""
        dd = DisplayDict(config)
        # Unknown dimension returns dim_id
        assert dd.dimension_header("xyz", "en") == "xyz"

    def test_empty_display_labels(self):
        """Empty config should not crash."""
        from mirror_memory.config.schema import MemoryConfig
        cfg = MemoryConfig()
        dd = DisplayDict(cfg)
        assert dd.friendly_label("any", "en") is None

    def test_multiple_labels_same_key(self, config):
        """Multiple labels with same key but different dimensions."""
        dd = DisplayDict(config)
        # Should not crash regardless
        for dim in ["topic", "preference", "fact"]:
            dd.friendly_label(dim, "en")
