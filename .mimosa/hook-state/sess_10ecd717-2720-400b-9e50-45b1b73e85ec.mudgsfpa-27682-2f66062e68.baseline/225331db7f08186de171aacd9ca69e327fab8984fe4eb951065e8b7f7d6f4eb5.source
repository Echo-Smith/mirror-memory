"""Tests for Temporal V2 — validity intervals and lifecycle transitions.

The rule being replaced was ``SINGLE cardinality → UPDATE``: a new value for
``lives_in`` destroyed the old one, so "where did they live before?" became
unanswerable.  These tests pin the replacement::

    identity (same subject + predicate?)
      + temporal relation (does the new fact start after the old ends?)
      → lifecycle transition

and, critically, that a TEMPORAL_UPDATE *closes* the old interval instead of
erasing it.
"""

from datetime import UTC, datetime

from mirror_memory.core.repository import (
    list_active_beliefs,
    record_claim,
    update_belief_by_id,
)
from mirror_memory.memory.atom import (
    ACTION_CONTRADICT,
    ACTION_CREATE,
    ACTION_SUPPORT,
    ACTION_UPDATE,
    CandidateAtom,
)
from mirror_memory.memory.identity import resolve_identity
from mirror_memory.memory.temporal import (
    SCOPE_CURRENT_STATE,
    SCOPE_EPISODIC,
    SCOPE_PERSISTENT,
    LifecycleAction,
    TemporalRelation,
    TemporalWindow,
    compare_windows,
    lifecycle_transition,
    window_from_belief,
)

POLICY = {"lives_in": "single", "works_at": "single", "likes": "multi", "went_to": "event"}
TEMPORAL_POLICY = {
    "lives_in": SCOPE_CURRENT_STATE,
    "works_at": SCOPE_CURRENT_STATE,
    "likes": SCOPE_PERSISTENT,
    "went_to": SCOPE_EPISODIC,
}


def _dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Interval comparison
# ---------------------------------------------------------------------------


class TestCompareWindows:
    def test_new_starts_after_old_ends_is_follows(self):
        new = TemporalWindow(valid_from=_dt(2026, 5), valid_to=_dt(2027, 5))
        old = TemporalWindow(valid_from=_dt(2024, 5), valid_to=_dt(2026, 5))
        assert compare_windows(new, old) is TemporalRelation.FOLLOWS

    def test_adjacent_intervals_are_follows(self):
        """A handover with no gap: Shanghai ends exactly when Beijing starts."""
        new = TemporalWindow(valid_from=_dt(2026, 5), valid_to=_dt(2027, 5))
        old = TemporalWindow(valid_from=_dt(2024, 5), valid_to=_dt(2026, 5))
        assert compare_windows(new, old) is TemporalRelation.FOLLOWS

    def test_new_ends_before_old_starts_is_precedes(self):
        new = TemporalWindow(valid_from=_dt(2020, 1), valid_to=_dt(2021, 1))
        old = TemporalWindow(valid_from=_dt(2024, 1), valid_to=_dt(2025, 1))
        assert compare_windows(new, old) is TemporalRelation.PRECEDES

    def test_same_window(self):
        window = TemporalWindow(valid_from=_dt(2024, 1), valid_to=_dt(2025, 1))
        assert compare_windows(window, window) is TemporalRelation.SAME

    def test_open_ended_new_start_decides_handover(self):
        """"Started May 2026, still true" vs "ended May 2026" is a handover."""
        new = TemporalWindow(valid_from=_dt(2026, 5), valid_to=None)
        old = TemporalWindow(valid_from=_dt(2024, 5), valid_to=_dt(2026, 5))
        assert compare_windows(new, old) is TemporalRelation.FOLLOWS

    def test_open_ended_new_start_before_old_end_overlaps(self):
        """Began while the old fact was still true."""
        new = TemporalWindow(valid_from=_dt(2025, 1), valid_to=None)
        old = TemporalWindow(valid_from=_dt(2024, 1), valid_to=_dt(2026, 1))
        assert compare_windows(new, old) is TemporalRelation.OVERLAPS

    def test_no_bounds_at_all_is_unknown(self):
        new = TemporalWindow()
        old = TemporalWindow()
        assert compare_windows(new, old) is TemporalRelation.UNKNOWN

    def test_one_sided_unknown(self):
        """Only a start on the new side and nothing on the old: undecidable."""
        new = TemporalWindow(valid_from=_dt(2026, 5), valid_to=None)
        old = TemporalWindow()
        assert compare_windows(new, old) is TemporalRelation.UNKNOWN

    def test_overlapping_windows(self):
        new = TemporalWindow(valid_from=_dt(2025, 1), valid_to=_dt(2027, 1))
        old = TemporalWindow(valid_from=_dt(2024, 1), valid_to=_dt(2026, 1))
        assert compare_windows(new, old) is TemporalRelation.OVERLAPS

    def test_containing_window(self):
        new = TemporalWindow(valid_from=_dt(2023, 1), valid_to=_dt(2028, 1))
        old = TemporalWindow(valid_from=_dt(2024, 1), valid_to=_dt(2026, 1))
        assert compare_windows(new, old) is TemporalRelation.CONTAINS


class TestWindowIsCurrent:
    def test_open_valid_to_is_current(self):
        assert TemporalWindow(valid_from=_dt(2024, 1)).is_current(at=_dt(2026, 1))

    def test_closed_valid_to_is_not_current(self):
        assert not TemporalWindow(
            valid_from=_dt(2024, 1), valid_to=_dt(2026, 5)
        ).is_current(at=_dt(2026, 6))

    def test_future_valid_from_is_not_current(self):
        assert not TemporalWindow(valid_from=_dt(2027, 1)).is_current(at=_dt(2026, 1))

    def test_null_window_is_current(self):
        assert TemporalWindow().is_current(at=_dt(2026, 1))


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------


class TestLifecycleTransition:
    def test_current_state_handover_is_temporal_update(self):
        action = lifecycle_transition(
            same_subject=True, same_predicate=True, same_object=False,
            scope=SCOPE_CURRENT_STATE, relation=TemporalRelation.FOLLOWS,
        )
        assert action is LifecycleAction.TEMPORAL_UPDATE

    def test_current_state_no_interval_evidence_still_supersedes(self):
        """Backwards compatible: the old SINGLE → UPDATE behaviour."""
        action = lifecycle_transition(
            same_subject=True, same_predicate=True, same_object=False,
            scope=SCOPE_CURRENT_STATE, relation=TemporalRelation.UNKNOWN,
        )
        assert action is LifecycleAction.TEMPORAL_UPDATE

    def test_overlapping_periods_coexist(self):
        action = lifecycle_transition(
            same_subject=True, same_predicate=True, same_object=False,
            scope=SCOPE_CURRENT_STATE, relation=TemporalRelation.OVERLAPS,
        )
        assert action is LifecycleAction.COEXIST

    def test_same_object_supports(self):
        action = lifecycle_transition(
            same_subject=True, same_predicate=True, same_object=True,
            scope=SCOPE_CURRENT_STATE, relation=TemporalRelation.SAME,
        )
        assert action is LifecycleAction.SUPPORT

    def test_same_object_contradiction_conflicts(self):
        action = lifecycle_transition(
            same_subject=True, same_predicate=True, same_object=True,
            scope=SCOPE_CURRENT_STATE, relation=TemporalRelation.SAME,
            claims_contradiction=True,
        )
        assert action is LifecycleAction.CONTRADICT

    def test_contradiction_against_different_value_is_a_handover(self):
        """"I don't live in Shanghai, I live in Berlin" replaces; it doesn't dispute."""
        action = lifecycle_transition(
            same_subject=True, same_predicate=True, same_object=False,
            scope=SCOPE_CURRENT_STATE, relation=TemporalRelation.FOLLOWS,
            claims_contradiction=True,
        )
        assert action is LifecycleAction.TEMPORAL_UPDATE

    def test_persistent_different_object_conflicts(self):
        action = lifecycle_transition(
            same_subject=True, same_predicate=True, same_object=False,
            scope=SCOPE_PERSISTENT, relation=TemporalRelation.FOLLOWS,
        )
        assert action is LifecycleAction.CONTRADICT

    def test_episodic_different_occurrence_creates(self):
        action = lifecycle_transition(
            same_subject=True, same_predicate=True, same_object=False,
            scope=SCOPE_EPISODIC, relation=TemporalRelation.FOLLOWS,
        )
        assert action is LifecycleAction.CREATE

    def test_different_subject_creates(self):
        action = lifecycle_transition(
            same_subject=False, same_predicate=True, same_object=False,
            scope=SCOPE_CURRENT_STATE, relation=TemporalRelation.FOLLOWS,
        )
        assert action is LifecycleAction.CREATE

    def test_different_predicate_creates(self):
        action = lifecycle_transition(
            same_subject=True, same_predicate=False, same_object=False,
            scope=SCOPE_CURRENT_STATE, relation=TemporalRelation.FOLLOWS,
        )
        assert action is LifecycleAction.CREATE


# ---------------------------------------------------------------------------
# Resolver integration
# ---------------------------------------------------------------------------


class TestResolverTemporal:
    def test_handover_reports_temporal_update(self):
        candidate = CandidateAtom(
            predicate="lives_in", object="beijing", valid_from=_dt(2026, 5)
        )
        existing = [{
            "id": 1, "predicate": "lives_in", "object": "shanghai",
            "status": "active", "valid_from": _dt(2024, 5), "valid_to": _dt(2026, 5),
        }]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.action == ACTION_UPDATE
        assert result.lifecycle == LifecycleAction.TEMPORAL_UPDATE.value
        assert result.temporal_relation == TemporalRelation.FOLLOWS.value
        assert result.target_belief_id == 1

    def test_no_interval_evidence_falls_back_to_update(self):
        candidate = CandidateAtom(predicate="lives_in", object="beijing")
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.action == ACTION_UPDATE
        assert result.temporal_relation == TemporalRelation.UNKNOWN.value

    def test_overlap_coexists_as_create(self):
        candidate = CandidateAtom(
            predicate="lives_in", object="beijing", valid_from=_dt(2025, 1)
        )
        existing = [{
            "id": 1, "predicate": "lives_in", "object": "shanghai",
            "status": "active", "valid_from": _dt(2024, 1), "valid_to": _dt(2026, 1),
        }]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.action == ACTION_CREATE
        assert result.lifecycle == LifecycleAction.COEXIST.value

    def test_same_object_supports(self):
        candidate = CandidateAtom(predicate="lives_in", object="shanghai")
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.action == ACTION_SUPPORT

    def test_multi_predicate_unaffected_by_scope(self):
        """likes is persistent but multi: coffee and tea coexist."""
        candidate = CandidateAtom(predicate="likes", object="tea")
        existing = [{"id": 1, "predicate": "likes", "object": "coffee", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.action == ACTION_CREATE

    def test_episodic_predicate_uses_event_logic(self):
        candidate = CandidateAtom(predicate="went_to", object="museum", temporal="monday")
        existing = [{
            "id": 1, "predicate": "went_to", "object": "museum",
            "status": "active", "temporal": "friday",
        }]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.action == ACTION_CREATE

    def test_picks_the_currently_true_belief_as_target(self):
        """Two Shanghai rows: the closed one is history, the open one is current."""
        candidate = CandidateAtom(predicate="lives_in", object="beijing")
        existing = [
            {"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active",
             "valid_from": _dt(2020, 1), "valid_to": _dt(2024, 1)},
            {"id": 2, "predicate": "lives_in", "object": "shanghai", "status": "active",
             "valid_from": _dt(2024, 1), "valid_to": None},
        ]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.target_belief_id == 2

    def test_contradiction_against_different_value_is_update(self):
        candidate = CandidateAtom(
            predicate="lives_in", object="beijing", relation="contradicts"
        )
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.action == ACTION_UPDATE

    def test_contradiction_against_same_value_conflicts(self):
        candidate = CandidateAtom(
            predicate="lives_in", object="shanghai", relation="contradicts"
        )
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY, TEMPORAL_POLICY)
        assert result.action == ACTION_CONTRADICT

    def test_empty_predicate_is_noop(self):
        result = resolve_identity(CandidateAtom(), [], POLICY, TEMPORAL_POLICY)
        assert result.action == "NOOP"


# ---------------------------------------------------------------------------
# Persistence: the old interval is closed, not destroyed
# ---------------------------------------------------------------------------


class TestUpdateClosesInterval:
    def test_old_belief_gets_valid_to(self, db_session):
        from mirror_memory.core.repository import set_memory_enabled

        set_memory_enabled(db_session, "u1", True)
        belief, _ = record_claim(
            db_session, "u1", dimension="location", key="lives_in",
            claim_text="lives in Shanghai", confidence=0.8,
            predicate="lives_in", object="shanghai", session_id="s1",
        )
        old, new = update_belief_by_id(
            db_session, belief.id,
            new_object="beijing", new_claim_text="lives in Berlin",
            valid_from=datetime(2026, 5, 1),
        )

        assert old.valid_to == datetime(2026, 5, 1)
        assert new.valid_from == datetime(2026, 5, 1)
        assert new.valid_to is None
        assert new.object == "beijing"

    def test_without_valid_from_interval_stays_open(self, db_session):
        """No interval evidence means the new fact is simply current."""
        from mirror_memory.core.repository import set_memory_enabled

        set_memory_enabled(db_session, "u1", True)
        belief, _ = record_claim(
            db_session, "u1", dimension="location", key="lives_in",
            claim_text="lives in Shanghai", confidence=0.8,
            predicate="lives_in", object="shanghai", session_id="s1",
        )
        old, new = update_belief_by_id(
            db_session, belief.id, new_object="beijing", new_claim_text="lives in Berlin",
        )
        assert old.valid_to is None
        assert new.valid_to is None
        assert new.valid_from is None

    def test_existing_valid_to_is_not_overwritten(self, db_session):
        from mirror_memory.core.repository import set_memory_enabled

        set_memory_enabled(db_session, "u1", True)
        belief, _ = record_claim(
            db_session, "u1", dimension="location", key="lives_in",
            claim_text="lives in Shanghai", confidence=0.8,
            predicate="lives_in", object="shanghai", session_id="s1",
        )
        belief.valid_to = datetime(2025, 1, 1)
        db_session.flush()

        old, _new = update_belief_by_id(
            db_session, belief.id, new_object="beijing", valid_from=datetime(2026, 5, 1),
        )
        assert old.valid_to == datetime(2025, 1, 1)

    def test_window_from_belief_round_trips(self, db_session):
        from mirror_memory.core.repository import set_memory_enabled

        set_memory_enabled(db_session, "u1", True)
        belief, _ = record_claim(
            db_session, "u1", dimension="location", key="lives_in",
            claim_text="lives in Shanghai", confidence=0.8,
            predicate="lives_in", object="shanghai", session_id="s1",
        )
        belief.valid_from = datetime(2024, 5, 1)
        belief.valid_to = datetime(2026, 5, 1)
        db_session.flush()

        window = window_from_belief(belief)
        assert window.valid_from == datetime(2024, 5, 1)
        assert window.valid_to == datetime(2026, 5, 1)
        assert not window.is_current(at=datetime(2027, 1, 1))


# ---------------------------------------------------------------------------
# Temporal retrieval
# ---------------------------------------------------------------------------


def _location_belief(session, key, obj, valid_from=None, valid_to=None):
    from mirror_memory.core.repository import set_memory_enabled

    set_memory_enabled(session, "u1", True)
    belief, _ = record_claim(
        session, "u1", dimension="location", key=key,
        claim_text=f"lives in {obj}", confidence=0.8,
        predicate="lives_in", object=obj, session_id="s1",
    )
    belief.valid_from = valid_from
    belief.valid_to = valid_to
    session.flush()
    return belief


class TestTemporalRetrieval:
    def test_current_excludes_closed_interval(self, db_session):
        past = _location_belief(
            db_session, "lives_in:shanghai", "shanghai",
            valid_from=datetime(2020, 1, 1), valid_to=datetime(2024, 1, 1),
        )
        present = _location_belief(
            db_session, "lives_in:beijing", "beijing",
            valid_from=datetime(2024, 1, 1), valid_to=None,
        )
        current = list_active_beliefs(db_session, "u1", temporal_mode="current")
        assert present in current
        assert past not in current

    def test_historical_excludes_open_interval(self, db_session):
        past = _location_belief(
            db_session, "lives_in:shanghai", "shanghai",
            valid_from=datetime(2020, 1, 1), valid_to=datetime(2024, 1, 1),
        )
        present = _location_belief(
            db_session, "lives_in:beijing", "beijing",
            valid_from=datetime(2024, 1, 1), valid_to=None,
        )
        historical = list_active_beliefs(db_session, "u1", temporal_mode="historical")
        assert past in historical
        assert present not in historical

    def test_all_returns_both(self, db_session):
        _location_belief(
            db_session, "lives_in:shanghai", "shanghai",
            valid_from=datetime(2020, 1, 1), valid_to=datetime(2024, 1, 1),
        )
        _location_belief(
            db_session, "lives_in:beijing", "beijing",
            valid_from=datetime(2024, 1, 1), valid_to=None,
        )
        assert len(list_active_beliefs(db_session, "u1")) == 2
        assert len(list_active_beliefs(db_session, "u1", temporal_mode="all")) == 2

    def test_null_valid_from_never_excludes(self, db_session):
        belief = _location_belief(db_session, "lives_in:shanghai", "shanghai")
        assert belief in list_active_beliefs(db_session, "u1", temporal_mode="current")

    def test_undated_belief_passes_both_filters(self, db_session):
        """An episodic event with no interval answers past questions too.

        Excluding undated beliefs from "historical" would make every question
        about the past return nothing, because K1 never sets valid_from.
        """
        undated = _location_belief(db_session, "went_to:museum", "museum")
        assert undated in list_active_beliefs(db_session, "u1", temporal_mode="current")
        assert undated in list_active_beliefs(db_session, "u1", temporal_mode="historical")

    def test_closed_interval_excluded_from_current(self, db_session):
        past = _location_belief(
            db_session, "lives_in:shanghai", "shanghai",
            valid_from=datetime(2020, 1, 1), valid_to=datetime(2024, 1, 1),
        )
        assert past not in list_active_beliefs(db_session, "u1", temporal_mode="current")
        assert past in list_active_beliefs(db_session, "u1", temporal_mode="historical")

    def test_disabled_memory_returns_nothing(self, db_session):
        from mirror_memory.core.repository import set_memory_enabled

        _location_belief(db_session, "lives_in:shanghai", "shanghai")
        set_memory_enabled(db_session, "u1", False)
        assert list_active_beliefs(db_session, "u1", temporal_mode="current") == []


# ---------------------------------------------------------------------------
# Query intent detection
# ---------------------------------------------------------------------------


class TestTemporalIntent:
    def test_now_is_current(self):
        from mirror_memory.render.renderer import _detect_temporal_mode

        assert _detect_temporal_mode("Where do they live now?") == "current"
        assert _detect_temporal_mode("Where do they currently work?") == "current"
        assert _detect_temporal_mode("\u73b0\u5728\u4f4f\u54ea\u91cc\uff1f") == "current"

    def test_before_is_historical(self):
        from mirror_memory.render.renderer import _detect_temporal_mode

        assert _detect_temporal_mode("Where did they live before?") == "historical"
        assert _detect_temporal_mode("Where did they used to live?") == "historical"
        assert _detect_temporal_mode("\u4ee5\u524d\u4f4f\u54ea\u91cc\uff1f") == "historical"

    def test_neutral_is_all(self):
        from mirror_memory.render.renderer import _detect_temporal_mode

        assert _detect_temporal_mode("Where do they live?") == "all"
        assert _detect_temporal_mode("") == "all"

    def test_historical_wins_on_a_tie(self):
        from mirror_memory.render.renderer import _detect_temporal_mode

        # "did they used to live" contains both a past marker and "now"-ish
        # phrasing; the past question is the one being asked.
        assert _detect_temporal_mode("Where did they used to live?") == "historical"


# ---------------------------------------------------------------------------
# The spec's worked example
# ---------------------------------------------------------------------------


class TestMoveScenario:
    def test_move_answers_both_present_and_past(self, db_session):
        """lives_in Shanghai → lives_in Beijing must answer both questions."""
        from mirror_memory.core.repository import set_memory_enabled
        from mirror_memory.render.renderer import render_memory_block

        set_memory_enabled(db_session, "u1", True)
        shanghai, _ = record_claim(
            db_session, "u1", dimension="location", key="lives_in:shanghai",
            claim_text="lives in Shanghai", confidence=0.8,
            predicate="lives_in", object="shanghai", session_id="s1",
        )
        shanghai.valid_from = datetime(2020, 1, 1)
        db_session.flush()

        old, new = update_belief_by_id(
            db_session, shanghai.id,
            new_object="beijing", new_claim_text="lives in Berlin",
            new_confidence=0.8, valid_from=datetime(2026, 5, 1),
        )

        # Shanghai is closed, not deleted.
        assert old.status == "superseded"
        assert old.valid_to == datetime(2026, 5, 1)
        assert new.status == "active"
        assert new.valid_from == datetime(2026, 5, 1)

        # "Where do they live now?" sees Berlin only.
        now_block = render_memory_block(
            db_session, "u1", config=_config(),
            user_message="Where do they live now?", language="en",
        )
        assert now_block and "Berlin" in now_block
        assert "Shanghai" not in now_block

        # "Where did they live before?" sees Shanghai only.
        past_block = render_memory_block(
            db_session, "u1", config=_config(),
            user_message="Where did they live before?", language="en",
        )
        assert past_block and "Shanghai" in past_block
        assert "Berlin" not in past_block


def _config():
    from mirror_memory.config.loader import load_config

    return load_config("config/")
