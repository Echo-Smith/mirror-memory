"""Memory metabolism (PR1: tiers + heat + eligibility) — rule-level and
integration tests.

Covers:
- retention-class derivation (identity axes → class, overrides)
- heat score math and the protected floor
- protection invariants (every rule)
- query-mode detection and storage-tier eligibility
- repository integration: classification at write time, access telemetry,
  tier-filtered retrieval, correction protection
- config loading of metabolism.yaml
- the legacy-schema guard
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy import (
    Column as SAColumn,
    DateTime as SADateTime,
    Integer as SAInteger,
    MetaData,
    String as SAString,
    Table,
)

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base
from mirror_memory.core.repository import (
    mark_belief_accessed,
    recall_candidates,
    record_claim,
)
from mirror_memory.exceptions import ConfigError
from mirror_memory.metabolism import (
    DEFAULT_SCAN_TIERS,
    CURRENT_SCAN_TIERS,
    LEGAL_TIER_TRANSITIONS,
    TIER_ARCHIVED,
    TIER_DORMANT,
    TIER_HOT,
    TIER_WARM,
    classify_retention,
    detect_query_mode,
    heat_score,
    heat_tier,
    is_legal_tier_transition,
    is_protected,
    protection_reason,
    tier_eligible,
    tiers_for_mode,
)


# ---------------------------------------------------------------------------
# Retention classification
# ---------------------------------------------------------------------------


class TestClassifyRetention:
    def test_single_cardinality_is_canonical(self):
        # lives_in / works_at / age style slots: one current value, long-lived.
        assert classify_retention(predicate="lives_in", cardinality="single") == "canonical"

    def test_event_cardinality_is_episodic(self):
        assert classify_retention(predicate="went_to", cardinality="event") == "episodic"

    def test_goal_lifecycle_state_is_behavioral(self):
        assert classify_retention(predicate="anything", cardinality="multi", lifecycle_state="active") == "behavioral"
        assert classify_retention(predicate="anything", cardinality="multi", lifecycle_state="paused") == "behavioral"

    def test_goal_predicate_is_behavioral(self):
        assert classify_retention(predicate="wants_to", cardinality="multi") == "behavioral"

    def test_default_is_preference(self):
        assert classify_retention(predicate="likes", cardinality="multi") == "preference"

    def test_override_beats_derived_rules(self):
        overrides = {"likes": "transient"}
        assert classify_retention(predicate="likes", cardinality="multi", overrides=overrides) == "transient"
        assert classify_retention(predicate="lives_in", cardinality="single", overrides=overrides) == "canonical"

    def test_unknown_override_is_ignored(self):
        assert classify_retention(predicate="likes", cardinality="multi", overrides={"likes": "nope"}) == "preference"


# ---------------------------------------------------------------------------
# Heat score
# ---------------------------------------------------------------------------


def _belief_stub(**kwargs):
    base = dict(
        last_accessed_at=None,
        last_evidence_at=datetime.now(UTC),
        access_count=0,
        evidence_json="[]",
        layer="L4",
        source="extracted",
        importance_score=0.5,
        value_json="{}",
        status="active",
        retention_class="preference",
        confidence=0.5,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestHeatScore:
    def test_fresh_strongly_supported_belief(self):
        now = datetime.now(UTC)
        score = heat_score(_belief_stub(), now=now, evidence_count=5)
        # freshness=1, access=0, evidence=1, authority=0.5, importance=0.5
        expected = 0.30 * 1 + 0.20 * 0 + 0.20 * 1 + 0.15 * 0.5 + 0.15 * 0.5
        assert score == pytest.approx(expected, abs=1e-6)

    def test_user_confirmed_beats_extracted(self):
        now = datetime.now(UTC)
        extracted = heat_score(_belief_stub(), now=now, evidence_count=5)
        confirmed = heat_score(_belief_stub(layer="L2", source="user_confirmed"), now=now, evidence_count=5)
        assert confirmed > extracted

    def test_freshness_decays_exponentially(self):
        now = datetime.now(UTC)
        fresh = heat_score(_belief_stub(last_evidence_at=now), now=now, evidence_count=0)
        stale = heat_score(
            _belief_stub(last_evidence_at=now - timedelta(days=120)), now=now, evidence_count=0
        )
        # Two half-lives (120d at 60d) → freshness falls by e^-2.
        assert fresh > stale
        assert stale < 0.30 * 0.14 + 0.20 + 0.15 * 0.5 + 0.15 * 0.5 + 0.001

    def test_access_count_lifts_score_log_scaled(self):
        now = datetime.now(UTC)
        never = heat_score(_belief_stub(access_count=0), now=now, evidence_count=0)
        once = heat_score(_belief_stub(access_count=1), now=now, evidence_count=0)
        often = heat_score(_belief_stub(access_count=100), now=now, evidence_count=0)
        assert once > never
        assert often < once + 0.21  # saturates: log1p(100)/5 capped at 1.0
        assert often <= 1.0

    def test_last_accessed_at_anchors_freshness(self):
        now = datetime.now(UTC)
        old_evidence = _belief_stub(last_evidence_at=now - timedelta(days=365))
        recently_recalled = _belief_stub(
            last_evidence_at=now - timedelta(days=365), last_accessed_at=now
        )
        assert heat_score(recently_recalled, now=now, evidence_count=0) > heat_score(
            old_evidence, now=now, evidence_count=0
        )

    def test_protected_floor(self):
        now = datetime.now(UTC)
        cold_and_thin = _belief_stub(last_evidence_at=now - timedelta(days=400))
        plain = heat_score(cold_and_thin, now=now, evidence_count=0)
        floored = heat_score(cold_and_thin, now=now, evidence_count=0, protected=True)
        assert plain < 0.70
        assert floored == pytest.approx(0.70)

    def test_heat_tier_thresholds(self):
        assert heat_tier(0.75) == "hot"
        assert heat_tier(0.70) == "hot"
        assert heat_tier(0.55) == "warm"
        assert heat_tier(0.40) == "warm"
        assert heat_tier(0.20) == "dormant"
        assert heat_tier(0.15) == "dormant"
        assert heat_tier(0.05) == "archive_candidate"


# ---------------------------------------------------------------------------
# Protection invariants
# ---------------------------------------------------------------------------


class TestProtection:
    def test_unresolved_conflict_is_protected(self):
        belief = _belief_stub(value_json='{"clarification_status": "needs_clarification"}')
        assert protection_reason(belief, support_count=5) == "unresolved_conflict"

    def test_user_correction_link_is_protected(self):
        belief = _belief_stub()
        assert protection_reason(belief, support_count=5, correct_count=1) == "user_correction"

    def test_user_corrected_source_is_protected(self):
        belief = _belief_stub(source="user_corrected")
        assert protection_reason(belief, support_count=5) == "user_correction"

    def test_pending_verification_is_protected(self):
        belief = _belief_stub()
        assert protection_reason(belief, support_count=5, pending_verification=True) == "pending_verification"

    def test_user_confirmed_layer_is_protected(self):
        assert protection_reason(_belief_stub(layer="L2"), support_count=5) == "user_confirmed"
        assert protection_reason(_belief_stub(source="user_confirmed"), support_count=5) == "user_confirmed"

    def test_canonical_high_confidence_is_protected(self):
        belief = _belief_stub(retention_class="canonical", confidence=0.85)
        assert protection_reason(belief, support_count=5) == "canonical_high_confidence"
        # Below the confidence bar it is not protected by this rule.
        weak = _belief_stub(retention_class="canonical", confidence=0.6)
        assert protection_reason(weak, support_count=5) != "canonical_high_confidence"

    def test_thin_evidence_is_protected(self):
        belief = _belief_stub()
        assert protection_reason(belief, support_count=1) == "thin_evidence"
        assert protection_reason(belief, support_count=0) == "thin_evidence"
        assert protection_reason(belief, support_count=2) is None

    def test_is_protected_wrapper(self):
        assert is_protected(_belief_stub(), support_count=0)
        assert not is_protected(_belief_stub(), support_count=3)


# ---------------------------------------------------------------------------
# Tiers and eligibility
# ---------------------------------------------------------------------------


class TestTiers:
    def test_cooling_and_reheat_edges_exist(self):
        for edge in ((TIER_HOT, TIER_WARM), (TIER_WARM, TIER_DORMANT), (TIER_DORMANT, TIER_HOT),
                     (TIER_ARCHIVED, TIER_WARM)):
            assert edge in LEGAL_TIER_TRANSITIONS

    def test_no_tier_skipping(self):
        assert not is_legal_tier_transition(TIER_HOT, TIER_DORMANT)
        assert not is_legal_tier_transition(TIER_HOT, TIER_ARCHIVED)
        assert not is_legal_tier_transition(TIER_WARM, TIER_ARCHIVED)


class TestEligibility:
    def test_current_scans_only_hot_and_warm(self):
        assert tiers_for_mode("current") == CURRENT_SCAN_TIERS == (TIER_HOT, TIER_WARM)

    def test_other_modes_scan_everything_but_archived(self):
        for mode in ("historical", "all", "all_occurrences", "evidence", "experience"):
            assert tiers_for_mode(mode) == DEFAULT_SCAN_TIERS
            assert TIER_ARCHIVED not in tiers_for_mode(mode)
            assert TIER_DORMANT in tiers_for_mode(mode)

    def test_tier_eligible(self):
        assert tier_eligible(TIER_DORMANT, "current") is False
        assert tier_eligible(TIER_DORMANT, "historical") is True
        assert tier_eligible(TIER_ARCHIVED, "all") is False


class TestQueryModeDetection:
    # Regression: the modes that existed before PR1 are unchanged.
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Where do they live now?", "current"),
            ("Where do they currently work?", "current"),
            ("现在住哪里？", "current"),
            ("Where did they live before?", "historical"),
            ("Where did they used to live?", "historical"),
            ("以前住哪里？", "historical"),
            ("Where do they live?", "all"),
            ("", "all"),
            ("What did they buy?", "all_occurrences"),
            ("How many times did they visit Berlin?", "all_occurrences"),
        ],
    )
    def test_existing_modes_unchanged(self, query, expected):
        assert detect_query_mode(query) == expected

    def test_evidence_mode(self):
        assert detect_query_mode("Why do they like coffee?") == "evidence"
        assert detect_query_mode("根据什么说他住在上海？") == "evidence"

    def test_experience_mode(self):
        assert detect_query_mode("What happened at the conference?") == "experience"
        assert detect_query_mode("他们经历过什么困难？") == "experience"

    def test_historical_beats_evidence(self):
        # A past question must still reach superseded rows.
        assert detect_query_mode("Why did they used to live in Shanghai?") == "historical"


# ---------------------------------------------------------------------------
# Repository integration
# ---------------------------------------------------------------------------


def _claim(session, key="k", **kwargs):
    params = dict(
        dimension="preference",
        key=key,
        claim_text="likes coffee",
        predicate="likes",
        object="coffee",
        cardinality="multi",
        # Above the L4 render watermark (0.55) so rendering tests see it.
        confidence=0.6,
    )
    params.update(kwargs)
    belief, event = record_claim(session, "u1", **params)
    assert event == "created"
    return belief


class TestRepositoryMetabolism:
    def test_new_belief_starts_hot_with_classified_retention(self, db_session):
        belief = _claim(db_session, key="lives_in", predicate="lives_in", cardinality="single")
        assert belief.memory_tier == "hot"
        assert belief.retention_class == "canonical"

    def test_episodic_and_behavioral_classification(self, db_session):
        event = _claim(db_session, key="e1", predicate="went_to", cardinality="event")
        goal = _claim(db_session, key="g1", predicate="wants_to", lifecycle_state="active")
        assert event.retention_class == "episodic"
        assert goal.retention_class == "behavioral"

    def test_explicit_retention_class_override(self, db_session):
        belief = record_claim(
            db_session, "u1", dimension="preference", key="k", claim_text="x",
            predicate="likes", object="coffee", cardinality="multi",
            retention_class="transient",
        )[0]
        assert belief.retention_class == "transient"

    def test_support_updates_last_supported_at(self, db_session):
        belief = _claim(db_session, key="k")
        assert belief.last_supported_at is not None
        before = belief.last_supported_at
        belief.last_supported_at = before - timedelta(days=30)
        record_claim(
            db_session, "u1", dimension="preference", key="k",
            claim_text="likes coffee a lot", relation="supports",
        )
        assert belief.last_supported_at > before - timedelta(days=30)

    def test_current_mode_skips_dormant_beliefs(self, db_session):
        hot = _claim(db_session, key="coffee", claim_text="likes coffee")
        dormant = _claim(db_session, key="tea", claim_text="likes tea")
        dormant.memory_tier = "dormant"
        db_session.flush()

        current = recall_candidates(db_session, "u1", query="coffee tea", temporal_mode="current")
        assert hot in current
        assert dormant not in current

        everything = recall_candidates(db_session, "u1", query="coffee tea", temporal_mode="all")
        assert dormant in everything

    def test_archived_never_enters_the_default_scan(self, db_session):
        archived = _claim(db_session, key="old", claim_text="likes old things")
        archived.memory_tier = "archived"
        db_session.flush()
        for mode in ("current", "historical", "all"):
            assert archived not in recall_candidates(db_session, "u1", query="old things", temporal_mode=mode)

    def test_historical_mode_reaches_superseded_dormant(self, db_session):
        old = _claim(db_session, key="lives_in:shanghai", predicate="lives_in", cardinality="single")
        old.status = "superseded"
        old.memory_tier = "dormant"
        db_session.flush()
        found = recall_candidates(db_session, "u1", query="shanghai", temporal_mode="historical")
        assert old in found

    def test_mark_belief_accessed(self, db_session):
        a = _claim(db_session, key="a")
        b = _claim(db_session, key="b")
        touched = mark_belief_accessed(db_session, [a.id, b.id, b.id])
        assert touched == 2  # deduplicated by row, not by call
        db_session.flush()
        assert a.access_count == 1
        assert b.access_count == 1  # one UPDATE per belief per call
        assert a.last_accessed_at is not None
        assert mark_belief_accessed(db_session, []) == 0

    def test_correction_marks_protection(self, db_session):
        from mirror_memory.core.repository import correct_belief

        belief = _claim(db_session, key="works_at:acme", predicate="works_at", cardinality="single")
        corrected = correct_belief(
            db_session, "u1", belief.id, new_claim_text="works at Globex",
            new_object="Globex",
        )
        assert corrected is not None
        assert corrected.protected_reason == "user_correction"
        assert corrected.retention_class == belief.retention_class
        assert is_protected(corrected, support_count=99)

    def test_update_inherits_retention_class(self, db_session):
        from mirror_memory.core.repository import update_belief_by_id

        old = _claim(db_session, key="lives_in:shanghai", predicate="lives_in", cardinality="single")
        _old, new = update_belief_by_id(
            db_session, old.id,
            new_predicate="lives_in", new_object="Beijing",
            new_dimension=old.dimension, new_key="lives_in:beijing",
            new_claim_text="lives in Beijing", new_confidence=0.6,
            new_cardinality="single",
        )
        assert new.retention_class == "canonical"

    def test_explain_includes_metabolism_fields(self, db_session):
        from mirror_memory.core.repository import explain_belief

        belief = _claim(db_session, key="k")
        info = explain_belief(db_session, "u1", belief.id)
        assert info["memory_tier"] == "hot"
        assert info["retention_class"] == "preference"
        assert info["protected_reason"] == ""


class TestPublisherPassthrough:
    def test_create_proposal_carries_retention_class(self, db_session):
        from mirror_memory.core.proposal import TRANSITION_CREATE, StateTransitionProposal
        from mirror_memory.core.publisher import Publisher

        publisher = Publisher(db_session)
        decision = publisher.publish(
            StateTransitionProposal(
                transition=TRANSITION_CREATE,
                user_id="u1",
                payload={
                    "dimension": "preference",
                    "key": "likes:cycling",
                    "claim_text": "likes cycling",
                    "predicate": "likes",
                    "object": "cycling",
                    "cardinality": "multi",
                    "confidence": 0.5,
                    "retention_class": "transient",
                },
            )
        )
        assert decision.committed
        belief = decision.detail  # unused; fetch directly
        from mirror_memory.core.repository import get_belief

        stored = get_belief(db_session, "u1", "likes:cycling")
        assert stored.retention_class == "transient"


class TestRenderAccessTelemetry:
    def test_rendered_beliefs_get_access_marked(self, db_session, config):
        from mirror_memory.render.renderer import render_memory_block

        belief = _claim(db_session, key="coffee", claim_text="likes coffee")
        assert belief.access_count == 0
        block = render_memory_block(
            db_session, "u1", config=config, language="en",
            user_message="what coffee do they like",
        )
        assert block is not None and "coffee" in block
        assert belief.access_count == 1
        assert belief.last_accessed_at is not None

    def test_unrendered_beliefs_are_not_marked(self, db_session, config):
        from mirror_memory.render.renderer import render_memory_block

        # Below the L4 render watermark, so the renderer gates it out before
        # the budget-packing stage where access is recorded.  (A recent,
        # high-confidence but irrelevant belief *does* render -- the recent
        # floor keeps context non-empty -- and is rightly marked.)
        other = _claim(
            db_session, key="unrelated", claim_text="collects stamps from Paraguay",
            confidence=0.3,
        )
        render_memory_block(
            db_session, "u1", config=config, language="en",
            user_message="what coffee do they like",
        )
        assert other.access_count == 0

    def test_repeated_reheat_after_recalls(self, db_session, config):
        from mirror_memory.render.renderer import render_memory_block

        belief = _claim(db_session, key="coffee", claim_text="likes coffee")
        for _ in range(3):
            render_memory_block(
                db_session, "u1", config=config, language="en",
                user_message="what coffee do they like",
            )
        assert belief.access_count == 3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestMetabolismConfig:
    def test_loads_from_config_dir(self):
        cfg = load_config("config/")
        assert cfg.metabolism.retention_classes["canonical"].cool_after_days is None
        assert cfg.metabolism.retention_classes["preference"].cool_after_days == 180
        assert cfg.metabolism.retention_classes["transient"].archive_after_days == 30
        assert cfg.metabolism.heat.thresholds.hot == 0.70
        assert cfg.metabolism.default_retention_class == "preference"

    def test_missing_file_yields_defaults(self, tmp_path):
        cfg = load_config(tmp_path)
        assert cfg.metabolism.retention_classes["behavioral"].cool_after_days == 60

    def test_invalid_default_class_rejected(self):
        from mirror_memory.config.schema import MetabolismConfig

        with pytest.raises(Exception):
            MetabolismConfig(default_retention_class="nope")


# ---------------------------------------------------------------------------
# Schema guard
# ---------------------------------------------------------------------------


class TestSchemaGuard:
    def test_fresh_database_passes(self):
        from mirror_memory.core.migrate import check_schema

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        check_schema(engine)  # no raise

    def test_empty_database_passes(self):
        from mirror_memory.core.migrate import check_schema

        check_schema(create_engine("sqlite://"))

    def test_legacy_beliefs_table_raises(self):
        from mirror_memory.core.migrate import check_schema

        md = MetaData()
        Table(
            "mm_beliefs", md,
            SAColumn("id", SAInteger, primary_key=True),
            SAColumn("user_id", SAString(64)),
            SAColumn("dimension", SAString(64)),
            SAColumn("key", SAString(128)),
            SAColumn("claim_text", SAString(200)),
            SAColumn("last_evidence_at", SADateTime),
        )
        engine = create_engine("sqlite://")
        md.create_all(engine)
        with pytest.raises(ConfigError, match="memory-metabolism"):
            check_schema(engine)
