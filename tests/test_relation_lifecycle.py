"""Tests for the three-way split: Identity / Relation / Lifecycle.

The old ``IdentityResolver`` answered "is this the same thing?", "what is
their relationship?", and "what should happen?" in one function.  These tests
pin each concern separately, so a change to the lifecycle policy cannot
silently change what the relation *is*.
"""


from datetime import UTC, datetime

from mirror_memory.memory.atom import (
    ACTION_CONTRADICT,
    ACTION_CREATE,
    ACTION_SUPPORT,
    ACTION_UPDATE,
    CandidateAtom,
)
from mirror_memory.memory.identity import resolve_identity
from mirror_memory.memory.lifecycle import LifecyclePolicy, to_action
from mirror_memory.memory.relation import RelationJudgement, judge_relation
from mirror_memory.memory.temporal import (
    SCOPE_CURRENT_STATE,
    SCOPE_EPISODIC,
    SCOPE_PERSISTENT,
    LifecycleAction,
    TemporalRelation,
)


def _dt(year, month=1, day=1):
    return datetime(year, month, day, tzinfo=UTC)


CARDINALITY = {"lives_in": "single", "likes": "multi", "went_to": "event"}
SCOPES = {
    "lives_in": SCOPE_CURRENT_STATE,
    "likes": SCOPE_PERSISTENT,
    "went_to": SCOPE_EPISODIC,
}


# ---------------------------------------------------------------------------
# RelationReasoner
# ---------------------------------------------------------------------------


class TestRelationReasoner:
    def test_same_predicate_and_object(self):
        judgement = judge_relation(
            CandidateAtom(predicate="likes", object="coffee"),
            {"predicate": "likes", "object": "coffee", "status": "active"},
        )
        assert judgement.same_identity
        assert judgement.same_object
        assert not judgement.claims_contradiction

    def test_different_object(self):
        judgement = judge_relation(
            CandidateAtom(predicate="likes", object="tea"),
            {"predicate": "likes", "object": "coffee", "status": "active"},
        )
        assert judgement.same_identity
        assert not judgement.same_object

    def test_different_predicate_is_not_same_identity(self):
        judgement = judge_relation(
            CandidateAtom(predicate="likes", object="coffee"),
            {"predicate": "dislikes", "object": "coffee", "status": "active"},
        )
        assert not judgement.same_identity

    def test_case_and_whitespace_insensitive(self):
        judgement = judge_relation(
            CandidateAtom(predicate="Lives_In", object=" Shanghai "),
            {"predicate": "lives_in", "object": "shanghai", "status": "active"},
        )
        assert judgement.same_identity
        assert judgement.same_object

    def test_contradiction_flag(self):
        judgement = judge_relation(
            CandidateAtom(predicate="likes", object="coffee", relation="contradicts"),
            {"predicate": "likes", "object": "coffee", "status": "active"},
        )
        assert judgement.claims_contradiction

    def test_temporal_relation_only_for_same_identity(self):
        """Two unrelated facts have no ordering to report."""
        judgement = judge_relation(
            CandidateAtom(predicate="likes", object="coffee", valid_from=_dt(2026, 1)),
            {"predicate": "dislikes", "object": "tea", "valid_from": _dt(2020, 1),
             "valid_to": _dt(2021, 1)},
        )
        assert judgement.temporal_relation is TemporalRelation.UNKNOWN

    def test_temporal_relation_reported_for_same_identity(self):
        judgement = judge_relation(
            CandidateAtom(predicate="lives_in", object="beijing", valid_from=_dt(2026, 5)),
            {"predicate": "lives_in", "object": "shanghai", "valid_from": _dt(2024, 5),
             "valid_to": _dt(2026, 5)},
        )
        assert judgement.temporal_relation is TemporalRelation.FOLLOWS

    def test_blank_qualifier_is_not_a_difference(self):
        judgement = judge_relation(
            CandidateAtom(predicate="went_to", object="museum"),
            {"predicate": "went_to", "object": "museum", "status": "active",
             "temporal": "monday"},
        )
        assert judgement.same_temporal_qualifier

    def test_different_qualifiers_differ(self):
        judgement = judge_relation(
            CandidateAtom(predicate="went_to", object="museum", temporal="friday"),
            {"predicate": "went_to", "object": "museum", "status": "active",
             "temporal": "monday"},
        )
        assert not judgement.same_temporal_qualifier

    def test_windows_derived_when_not_supplied(self):
        judgement = judge_relation(
            CandidateAtom(predicate="lives_in", object="beijing", valid_from=_dt(2026, 5)),
            {"predicate": "lives_in", "object": "shanghai", "valid_to": _dt(2026, 5)},
        )
        assert judgement.temporal_relation is TemporalRelation.FOLLOWS


# ---------------------------------------------------------------------------
# LifecyclePolicy
# ---------------------------------------------------------------------------


def _judgement(**overrides):
    base = dict(
        same_subject=True, same_predicate=True, same_object=False,
        temporal_relation=TemporalRelation.UNKNOWN, claims_contradiction=False,
    )
    base.update(overrides)
    return RelationJudgement(**base)


class TestLifecyclePolicy:
    def test_scope_from_config(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        assert policy.scope_for("lives_in") == SCOPE_CURRENT_STATE
        assert policy.scope_for("unknown_predicate") == ""

    def test_single_cardinality_implies_current_state(self):
        policy = LifecyclePolicy(cardinality={"lives_in": "single"})
        assert policy.scope_for("lives_in") == SCOPE_CURRENT_STATE
        assert policy.uses_temporal_lifecycle("lives_in")

    def test_multi_predicate_does_not_use_temporal_lifecycle(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        assert not policy.uses_temporal_lifecycle("likes")

    def test_different_identity_creates(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(_judgement(same_predicate=False), "lives_in")
        assert action is LifecycleAction.CREATE

    def test_handover_is_temporal_update(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(
            _judgement(temporal_relation=TemporalRelation.FOLLOWS), "lives_in"
        )
        assert action is LifecycleAction.TEMPORAL_UPDATE

    def test_overlap_coexists(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(
            _judgement(temporal_relation=TemporalRelation.OVERLAPS), "lives_in"
        )
        assert action is LifecycleAction.COEXIST

    def test_same_object_supports(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(_judgement(same_object=True), "lives_in")
        assert action is LifecycleAction.SUPPORT

    def test_same_object_contradiction_conflicts(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(
            _judgement(same_object=True, claims_contradiction=True), "lives_in"
        )
        assert action is LifecycleAction.CONTRADICT

    def test_multi_new_object_creates(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(_judgement(), "likes")
        assert action is LifecycleAction.CREATE

    def test_multi_contradiction_on_new_object_creates(self):
        """A negative claim about an object with no belief is its own fact."""
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(_judgement(claims_contradiction=True), "likes")
        assert action is LifecycleAction.CREATE

    def test_multi_contradiction_on_same_object_conflicts(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(
            _judgement(same_object=True, claims_contradiction=True), "likes"
        )
        assert action is LifecycleAction.CONTRADICT

    def test_event_different_qualifier_creates(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(
            _judgement(same_object=True, same_temporal_qualifier=False), "went_to"
        )
        assert action is LifecycleAction.CREATE

    def test_event_same_qualifier_supports(self):
        policy = LifecyclePolicy(cardinality=CARDINALITY, scopes=SCOPES)
        action = policy.decide(
            _judgement(same_object=True, same_temporal_qualifier=True), "went_to"
        )
        assert action is LifecycleAction.SUPPORT


class TestActionMapping:
    def test_temporal_update_maps_to_update(self):
        assert to_action(LifecycleAction.TEMPORAL_UPDATE) == ACTION_UPDATE

    def test_coexist_maps_to_create(self):
        assert to_action(LifecycleAction.COEXIST) == ACTION_CREATE

    def test_support_maps_to_support(self):
        assert to_action(LifecycleAction.SUPPORT) == ACTION_SUPPORT

    def test_contradict_maps_to_contradict(self):
        assert to_action(LifecycleAction.CONTRADICT) == ACTION_CONTRADICT

    def test_create_maps_to_create(self):
        assert to_action(LifecycleAction.CREATE) == ACTION_CREATE


# ---------------------------------------------------------------------------
# The resolver is only an orchestrator
# ---------------------------------------------------------------------------


class TestResolverOrchestration:
    def test_first_claim_creates(self):
        result = resolve_identity(
            CandidateAtom(predicate="lives_in", object="shanghai"), [], CARDINALITY
        )
        assert result.action == ACTION_CREATE

    def test_resolution_carries_lifecycle_and_relation(self):
        result = resolve_identity(
            CandidateAtom(predicate="lives_in", object="beijing", valid_from=_dt(2026, 5)),
            [{"id": 7, "predicate": "lives_in", "object": "shanghai", "status": "active",
              "valid_from": _dt(2024, 5), "valid_to": _dt(2026, 5)}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_UPDATE
        assert result.lifecycle == LifecycleAction.TEMPORAL_UPDATE.value
        assert result.temporal_relation == TemporalRelation.FOLLOWS.value

    def test_opposite_polarity_replaces_the_current_value(self):
        """One current polarity per object: the withdrawal closes the stance.

        "I do not like coffee" against an active "likes coffee" is a
        replacement of the current value, not a second coexisting belief.
        """
        result = resolve_identity(
            CandidateAtom(predicate="dislikes", object="coffee"),
            [{"id": 7, "predicate": "likes", "object": "coffee", "status": "active",
              "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_UPDATE
        assert result.target_belief_id == 7
        assert result.lifecycle == LifecycleAction.TEMPORAL_UPDATE.value

    def test_same_polarity_about_other_objects_coexists(self):
        result = resolve_identity(
            CandidateAtom(predicate="likes", object="painting"),
            [{"id": 7, "predicate": "likes", "object": "coffee", "status": "active",
              "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_CREATE

    def test_neutral_predicate_is_untouched_by_polarity_replacement(self):
        """lives_in is neutral; no opposing polarity exists to replace."""
        result = resolve_identity(
            CandidateAtom(predicate="lives_in", object="berlin"),
            [{"id": 7, "predicate": "lives_in", "object": "shanghai", "status": "active",
              "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_UPDATE
        assert "TEMPORAL_UPDATE" in (result.lifecycle or "")
        assert result.target_belief_id == 7

    def test_self_correction_closes_the_corrected_belief(self):
        """An explicit retraction ends the belief it retracts.

        The extractor spells the object differently each turn and often under
        a different predicate, so the referent is found by object tokens.
        """
        result = resolve_identity(
            CandidateAtom(
                predicate="skills", object="rust",
                claim_text="Correction: I have never learned Rust.",
            ),
            [{"id": 7, "predicate": "skills", "object": "rust_coding",
              "status": "active", "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_UPDATE
        assert result.target_belief_id == 7
        assert result.lifecycle == "SELF_CORRECTION"

    def test_termination_closes_the_current_goal(self):
        """Cancelling a goal ends its current stage, it does not add a row."""
        result = resolve_identity(
            CandidateAtom(
                predicate="gave_up", object="marathon_goal",
                claim_text="I gave up the marathon goal.",
            ),
            [{"id": 7, "predicate": "wants_to", "object": "run_marathon",
              "status": "active", "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_UPDATE
        assert result.target_belief_id == 7
        assert result.lifecycle == "END_CURRENT"

    def test_retraction_without_a_referent_falls_through(self):
        """A marker with no overlapping active belief is an ordinary claim."""
        result = resolve_identity(
            CandidateAtom(
                predicate="gave_up", object="marathon_goal",
                claim_text="I gave up the marathon goal.",
            ),
            [{"id": 7, "predicate": "likes", "object": "coffee",
              "status": "active", "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_CREATE

    def test_ordinary_restatement_is_not_a_retraction(self):
        result = resolve_identity(
            CandidateAtom(predicate="likes", object="coffee", claim_text="I like coffee"),
            [{"id": 7, "predicate": "likes", "object": "coffee", "status": "active",
              "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_SUPPORT

    def test_generic_goal_referent_closes_the_active_goal(self):
        """"I gave up on the goal" names no object that overlaps the goal."""
        result = resolve_identity(
            CandidateAtom(
                predicate="experiences", object="unnamed_goal",
                claim_text="I gave up on the goal.",
            ),
            [{"id": 7, "predicate": "wants_to", "object": "run_marathon",
              "status": "active", "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_UPDATE
        assert result.target_belief_id == 7
        assert result.lifecycle == "END_CURRENT"

    def test_resumption_retires_the_ended_stage(self):
        """The cancellation row must not stay current once the goal is back.

        The match is by the ended belief's identity, which the pipeline stamps
        on the ended row -- the row's own object ("unnamed_goal") says nothing
        about which goal it ended.
        """
        result = resolve_identity(
            CandidateAtom(
                predicate="went_to", object="marathon_training",
                claim_text="I have started marathon training again.",
            ),
            [
                {"id": 7, "predicate": "wants_to", "object": "run_marathon",
                 "status": "superseded", "confidence": 0.9},
                {"id": 8, "predicate": "experiences", "object": "unnamed_goal",
                 "status": "active", "confidence": 0.9,
                 "transition": "END_CURRENT", "ended_object": "run_marathon"},
            ],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_UPDATE
        assert result.target_belief_id == 8
        assert result.lifecycle == "RESUME"

    def test_resumption_ignores_an_unrelated_ended_stage(self):
        """"I like coffee again" must not retire the sold-bicycle row."""
        result = resolve_identity(
            CandidateAtom(predicate="likes", object="coffee", claim_text="I like coffee again."),
            [
                {"id": 7, "predicate": "sold", "object": "red_bicycle",
                 "status": "active", "confidence": 0.9,
                 "transition": "END_CURRENT", "ended_object": "red_bicycle"},
            ],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_CREATE

    def test_generic_goal_cancellation_with_several_goals_closes_nothing(self):
        """"the goal" is ambiguous when more than one goal is live."""
        result = resolve_identity(
            CandidateAtom(
                predicate="experiences", object="unnamed_goal",
                claim_text="I gave up on the goal.",
            ),
            [
                {"id": 7, "predicate": "wants_to", "object": "learn_japanese",
                 "status": "active", "confidence": 0.9},
                {"id": 8, "predicate": "wants_to", "object": "run_marathon",
                 "status": "active", "confidence": 0.9},
            ],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_CREATE

    def test_resumption_without_an_ended_stage_is_an_ordinary_claim(self):
        result = resolve_identity(
            CandidateAtom(predicate="likes", object="coffee", claim_text="I like coffee again."),
            [{"id": 7, "predicate": "likes", "object": "coffee", "status": "active",
              "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_SUPPORT

    def test_unrelated_termination_leaves_the_goal_alone(self):
        """"I no longer work at Acme" must not cancel an unrelated goal."""
        result = resolve_identity(
            CandidateAtom(predicate="works_at", object="acme", claim_text="I no longer work at Acme."),
            [{"id": 7, "predicate": "wants_to", "object": "learn_japanese",
              "status": "active", "confidence": 0.9}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_CREATE

    def test_picks_currently_true_belief_as_target(self):
        result = resolve_identity(
            CandidateAtom(predicate="lives_in", object="beijing"),
            [
                {"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active",
                 "valid_from": _dt(2020, 1), "valid_to": _dt(2024, 1)},
                {"id": 2, "predicate": "lives_in", "object": "shanghai", "status": "active",
                 "valid_from": _dt(2024, 1), "valid_to": None},
            ],
            CARDINALITY, SCOPES,
        )
        assert result.target_belief_id == 2

    def test_contradiction_same_object_short_circuits_policy(self):
        result = resolve_identity(
            CandidateAtom(predicate="lives_in", object="shanghai", relation="contradicts"),
            [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}],
            CARDINALITY, SCOPES,
        )
        assert result.action == ACTION_CONTRADICT
        assert result.lifecycle == "CONTRADICT"

    def test_empty_predicate_is_noop(self):
        result = resolve_identity(CandidateAtom(), [], CARDINALITY)
        assert result.action == "NOOP"

    def test_policy_change_alone_does_not_change_the_relation(self):
        """The same facts, judged the same way, under a different policy."""
        candidate = CandidateAtom(predicate="lives_in", object="beijing", valid_from=_dt(2026, 5))
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active",
                     "valid_from": _dt(2024, 5), "valid_to": _dt(2026, 5)}]

        as_current_state = resolve_identity(candidate, existing, CARDINALITY, SCOPES)
        as_legacy = resolve_identity(candidate, existing, CARDINALITY, {})

        # The relation is identical; only the policy differs.
        assert as_current_state.temporal_relation == as_legacy.temporal_relation
        assert as_current_state.lifecycle == LifecycleAction.TEMPORAL_UPDATE.value
        assert as_legacy.lifecycle == LifecycleAction.TEMPORAL_UPDATE.value
