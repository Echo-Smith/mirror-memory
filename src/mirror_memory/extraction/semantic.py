"""K2 LLM semantic extraction -- domain-agnostic.

Allowed keys and system prompt are loaded from ``config``.  The LLM
client is injected via ``config.llm_client``.  Zero domain coupling.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from mirror_memory.config.schema import MemoryConfig
from mirror_memory.core.constants import CLAIM_TEXT_MAX_LENGTH, MAX_CLAIMS_PER_TURN

logger = logging.getLogger(__name__)


def _build_allowed_keys(config: MemoryConfig) -> dict[str, frozenset[str]]:
    """Build dimension -> allowed keys mapping from config anchors.

    All configured dimensions are included, even if they have no anchors.
    Dimensions without anchors have an empty key set, which means the
    parser accepts any key within that dimension.
    """
    allowed: dict[str, set[str]] = {}
    # Include all configured dimensions (even without anchors).
    for dim in config.dimensions:
        allowed.setdefault(dim.dimension_id, set())
    # Populate keys from anchors.
    for anchor in config.anchors:
        if anchor.identify_only:
            continue
        allowed.setdefault(anchor.dimension, set()).add(anchor.key)
    return {dim: frozenset(keys) for dim, keys in allowed.items()}


def _anchor_prompt_lines(config: MemoryConfig) -> list[str]:
    """Build the allowed-keys section for the extraction prompt."""
    lines: list[str] = []
    for anchor in config.anchors:
        if anchor.identify_only:
            continue
        # Find a display label for this anchor
        label = ""
        for lbl in config.display_labels:
            if lbl.key == anchor.key and lbl.dimension == anchor.dimension:
                label = lbl.zh or lbl.en or ""
                break
        sample = anchor.phrases[0] if anchor.phrases else ""
        lines.append(f"{anchor.dimension} {anchor.key} = {label} | e.g. {sample}")
    return lines


def _empty_result() -> dict:
    """The no-claims result, in the same shape the success path returns."""
    return {"claims": [], "subject": "user", "context_tags": []}


def _parse_extraction_full(
    raw: str, allowed_keys: dict[str, frozenset[str]]
) -> dict:
    """Parse LLM output into a structured result with claims, subject, and context_tags.

    Returns ``{"claims": [...], "subject": "user"|"third_party", "context_tags": [...]}``.

    ``parse_error`` classifies why no claims came back: ``None`` when claims
    were produced (or the model legitimately found none), ``"invalid_json"``
    when the text was not parseable, ``"wrong_shape"`` when it parsed to
    something other than the claim container, and ``"no_valid_claims"`` when
    the container held only rejected items.  A zero-claim response that is
    not an error and a zero-claim response that is one must not be counted
    the same way, or the run health cannot tell a quiet model from a broken
    one.
    """
    from mirror_memory.core.utils import parse_llm_json

    data = parse_llm_json(raw, expect_array=True)
    if data is None:
        return {"claims": [], "subject": "user", "context_tags": [],
                "parse_error": "invalid_json"}

    # Accept bare [...] (legacy), {"claims": [...]}, and full structured formats.
    if isinstance(data, list):
        raw_claims = data
        subject = "user"
        context_tags = []
    elif isinstance(data, dict):
        raw_claims = data.get("claims", [])
        subject = str(data.get("subject") or "user")
        raw_tags = data.get("context_tags") or []
        context_tags = [str(t) for t in raw_tags if isinstance(t, str)][:3] if isinstance(raw_tags, list) else []
    else:
        return {"claims": [], "subject": "user", "context_tags": [],
                "parse_error": "wrong_shape"}

    if not isinstance(raw_claims, list):
        return {"claims": [], "subject": subject, "context_tags": context_tags,
                "parse_error": "wrong_shape"}

    validated: list[dict] = []
    for item in raw_claims:
        if not isinstance(item, dict) or len(validated) >= MAX_CLAIMS_PER_TURN:
            break
        dimension = str(item.get("dimension") or "")
        key = str(item.get("key") or "")
        relation = str(item.get("relation") or "supports")
        if dimension not in allowed_keys or not key:
            continue
        # Normalize relations: "new"/"first_mention"/"updates" → "supports"
        # "updates" is intentionally excluded from the prompt — state UPDATE
        # is decided by the IdentityResolver based on cardinality, not the LLM.
        if relation in {"new", "first_mention", "updates"}:
            relation = "supports"
        if relation not in {"supports", "contradicts"}:
            continue
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        # Cognitive triple fields (optional, for richer structured memory).
        predicate = str(item.get("predicate") or "")
        obj = str(item.get("object") or "")
        temporal = str(item.get("temporal") or "")
        # Validity window.  Without these the whole Temporal V2 path is inert:
        # every belief gets valid_from=valid_to=None, no interval is ever
        # closed, and a current-state question cannot be told apart from a
        # historical one.
        valid_from = str(item.get("valid_from") or "")
        valid_to = str(item.get("valid_to") or "")
        value: dict = {"via": "llm_semantic"}
        if predicate:
            value["predicate"] = predicate
        if obj:
            value["object"] = obj
        if temporal:
            value["temporal"] = temporal
        if valid_from:
            value["valid_from"] = valid_from
        if valid_to:
            value["valid_to"] = valid_to
        validated.append({
            "dimension": dimension,
            "key": key,
            "claim_text": str(item.get("claim_text") or item.get("claim_zh") or "")[:CLAIM_TEXT_MAX_LENGTH],
            "confidence": confidence,
            "relation": relation,
            "source": "extracted",
            "extractor_stage": "k2",
            "value": value,
        })

    return {
        "claims": validated,
        "subject": subject,
        "context_tags": context_tags,
        # Every item was rejected by the schema gate: the model answered, but
        # nothing it said was usable.  That is a different failure from a
        # quiet model and has to be counted separately.
        "parse_error": "no_valid_claims" if not validated and raw_claims else None,
    }


def _parse_extraction_json(raw: str, allowed_keys: dict[str, frozenset[str]]) -> list[dict]:
    """Backward-compatible wrapper: parse claims only (no side output)."""
    return _parse_extraction_full(raw, allowed_keys)["claims"]


class SemanticExtractor:
    """K2 LLM semantic extractor.

    Parameters
    ----------
    config:
        The ``MemoryConfig`` providing prompts, anchors, and the LLM
        client.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._allowed_keys = _build_allowed_keys(config)
        # LLM health counters.  A failed call returns empty claims rather than
        # raising -- that is what keeps a turn usable -- so the failure has to
        # be counted here or a run whose every call failed (a rate-limited
        # "forced" run, say) is indistinguishable from a deterministic one.
        # The three quality counters separate "the model said nothing usable"
        # from "the model was never heard", which the release gate needs.
        self.llm_calls = 0
        self.llm_failures = 0
        self.empty_responses = 0
        self.parse_failures = 0
        self.schema_rejections = 0

    def extract(
        self,
        text: str,
        session: object | None,
        user_id: str,
        config: MemoryConfig,
    ) -> list[dict]:
        """Run LLM semantic extraction on *text*.

        Parameters
        ----------
        text:
            The user's current message.
        session:
            Database session (for reading current beliefs, if needed).
        user_id:
            The user identifier.
        config:
            The ``MemoryConfig`` (may differ from constructor config if
            the caller overrides at call time).

        Returns
        -------
        list[dict]
            Validated claim dicts (up to 3).
        """
        llm = config.llm_client
        if llm is None:
            logger.debug("semantic: no LLM client configured, skipping")
            return []

        system_prompt = config.prompts.k2_system
        if not system_prompt:
            logger.warning("semantic: k2_system prompt is empty, skipping")
            return []

        # Build payload: allowed keys + today's date + user text.
        # The date is required, not decorative: without it the model resolves
        # "last month" against its own training cutoff, so every relative date
        # lands years in the past and no interval is ever closed correctly.
        anchor_lines = _anchor_prompt_lines(config)
        today = datetime.now(UTC).date().isoformat()
        payload = (
            "[Allowed keys]\n"
            + "\n".join(anchor_lines)
            + f"\n\n[Current date]\n{today}"
            + "\n\n[User utterance]\n"
            + (text or "").strip()
        )

        self.llm_calls += 1
        failures_before = getattr(llm, "call_failures", 0)
        try:
            raw = llm.generate(
                system_prompt=system_prompt,
                payload_text=payload,
                fallback=lambda: '{"claims": []}',
            )
        except Exception:
            self.llm_failures += 1
            logger.warning("semantic: LLM call failed, returning empty claims")
            return _empty_result()
        # A client may swallow its own error and answer with the fallback
        # string instead of raising.  Its failure counter is the only signal
        # that the empty claims are a failed call, not a model that found
        # nothing -- without this a rate-limited run measures the fallback.
        failures_after = getattr(llm, "call_failures", 0)
        if failures_after > failures_before:
            self.llm_failures += failures_after - failures_before
            logger.warning(
                "semantic: LLM client reported a failed call, returning empty claims"
            )
            return _empty_result()

        if not (raw or "").strip():
            self.empty_responses += 1
            logger.warning("semantic: LLM returned an empty response")
            return _empty_result()

        result = _parse_extraction_full(raw, self._allowed_keys)
        parse_error = result.get("parse_error")
        if parse_error == "invalid_json":
            self.parse_failures += 1
        elif parse_error in ("wrong_shape", "no_valid_claims"):
            self.schema_rejections += 1
        logger.info(
            "semantic: extracted %d claims (subject=%s, tags=%s, parse_error=%s)",
            len(result["claims"]), result["subject"], result["context_tags"],
            parse_error,
        )
        return result
