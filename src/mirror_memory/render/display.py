"""Display dictionary -- friendly label lookup from config.

Zero domain coupling.  All labels come from ``config.display_labels``
and ``config.dimensions``.
"""

from __future__ import annotations

from mirror_memory.config.schema import MemoryConfig


class DisplayDict:
    """Lookup-friendly wrapper around ``config.display_labels``.

    Parameters
    ----------
    config:
        The ``MemoryConfig`` providing display labels and dimension
        definitions.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        # Build a fast lookup: key -> {lang: label}
        self._label_index: dict[str, dict[str, str]] = {}
        for lbl in config.display_labels:
            self._label_index[lbl.key] = {"zh": lbl.zh, "en": lbl.en}

        # Build dimension header index: dimension_id -> {lang: header}
        self._dim_headers: dict[str, dict[str, str]] = {}
        for dim in config.dimensions:
            self._dim_headers[dim.dimension_id] = dim.name

    def friendly_label(self, key: str, lang: str = "zh") -> str | None:
        """Return the friendly label for *key* in *lang*, or ``None``.

        Parameters
        ----------
        key:
            The belief key (e.g. ``"social_anxiety"``).
        lang:
            ``"zh"`` or ``"en"``.

        Returns
        -------
        str or None
            The human-readable label, or ``None`` if no label is
            configured for this key.
        """
        labels = self._label_index.get(key)
        if not labels:
            return None
        lang = lang.strip().lower()
        return labels.get(lang) or labels.get("en") or labels.get("zh") or None

    def dimension_header(self, dim_id: str, lang: str = "zh") -> str:
        """Return the section header for a dimension.

        Parameters
        ----------
        dim_id:
            The dimension identifier (e.g. ``"D1"``).
        lang:
            ``"zh"`` or ``"en"``.

        Returns
        -------
        str
            The header string, or ``dim_id`` if no header is configured.
        """
        headers = self._dim_headers.get(dim_id, {})
        lang = lang.strip().lower()
        return headers.get(lang) or headers.get("en") or headers.get("zh") or dim_id
