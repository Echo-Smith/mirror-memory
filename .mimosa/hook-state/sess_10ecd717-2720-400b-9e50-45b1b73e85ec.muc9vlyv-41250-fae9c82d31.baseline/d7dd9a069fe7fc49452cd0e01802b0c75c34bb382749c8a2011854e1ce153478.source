"""Tests for memory/identity.py — IdentityResolver.

The resolver decides what happens when a new CandidateAtom arrives:
CREATE / SUPPORT / UPDATE / CONTRADICT / NOOP.
"""

import pytest

from mirror_memory.memory.atom import (
    ACTION_CONTRADICT,
    ACTION_CREATE,
    ACTION_NOOP,
    ACTION_SUPPORT,
    ACTION_UPDATE,
    CandidateAtom,
)
from mirror_memory.memory.identity import resolve_identity


# Default test policy
POLICY = {
    "lives_in": "single",
    "works_at": "single",
    "age": "single",
    "likes": "multi",
    "has": "multi",
    "went_to": "event",
    "attended": "event",
}


class TestMultiCardinality:
    """likes coffee + likes painting → 2 atoms."""

    def test_new_predicate(self):
        """First claim for a predicate → CREATE."""
        candidate = CandidateAtom(predicate="likes", object="coffee")
        result = resolve_identity(candidate, [], POLICY)
        assert result.action == ACTION_CREATE

    def test_same_object_supports(self):
        """Same predicate + same object → SUPPORT."""
        candidate = CandidateAtom(predicate="likes", object="coffee")
        existing = [{"id": 1, "predicate": "likes", "object": "coffee", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_SUPPORT
        assert result.target_belief_id == 1

    def test_different_object_multi_create(self):
        """MULTI: different object → CREATE."""
        candidate = CandidateAtom(predicate="likes", object="painting")
        existing = [{"id": 1, "predicate": "likes", "object": "coffee", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_CREATE

    def test_case_insensitive_match(self):
        """Object matching is case-insensitive."""
        candidate = CandidateAtom(predicate="likes", object="Coffee")
        existing = [{"id": 1, "predicate": "likes", "object": "coffee", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_SUPPORT

    def test_whitespace_normalized(self):
        """Object matching normalizes whitespace."""
        candidate = CandidateAtom(predicate="likes", object="  coffee  ")
        existing = [{"id": 1, "predicate": "likes", "object": "coffee", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_SUPPORT


class TestSingleCardinality:
    """lives_in Shanghai + lives_in Beijing → UPDATE."""

    def test_new_single_create(self):
        """First claim for SINGLE predicate → CREATE."""
        candidate = CandidateAtom(predicate="lives_in", object="shanghai")
        result = resolve_identity(candidate, [], POLICY)
        assert result.action == ACTION_CREATE

    def test_same_object_supports(self):
        """SINGLE + same object → SUPPORT."""
        candidate = CandidateAtom(predicate="lives_in", object="shanghai")
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_SUPPORT

    def test_different_object_updates(self):
        """SINGLE + different object → UPDATE (supersedes)."""
        candidate = CandidateAtom(predicate="lives_in", object="beijing")
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_UPDATE
        assert result.target_belief_id == 1

    def test_age_update(self):
        """Age is SINGLE: 28 → 29 = UPDATE."""
        candidate = CandidateAtom(predicate="age", object="29")
        existing = [{"id": 1, "predicate": "age", "object": "28", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_UPDATE


class TestEventCardinality:
    """went_to museum Mon + went_to museum Fri → 2 events."""

    def test_new_event_create(self):
        """New event → CREATE."""
        candidate = CandidateAtom(predicate="went_to", object="museum_monday")
        result = resolve_identity(candidate, [], POLICY)
        assert result.action == ACTION_CREATE

    def test_different_event_create(self):
        """Different event → CREATE (not UPDATE)."""
        candidate = CandidateAtom(predicate="went_to", object="museum_friday")
        existing = [{"id": 1, "predicate": "went_to", "object": "museum_monday", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_CREATE

    def test_exact_duplicate_supports(self):
        """Exact same event → SUPPORT (not duplicate)."""
        candidate = CandidateAtom(predicate="went_to", object="museum_monday")
        existing = [{"id": 1, "predicate": "went_to", "object": "museum_monday", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_SUPPORT


class TestEventTemporal:
    """Same predicate+object but different temporal = different events."""

    def test_different_temporal_create(self):
        """went_to museum Monday vs Friday → different events → CREATE."""
        candidate = CandidateAtom(predicate="went_to", object="museum", temporal="monday", dimension="event")
        existing = [{"id": 1, "predicate": "went_to", "object": "museum", "status": "active", "temporal": "friday"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_CREATE

    def test_same_temporal_supports(self):
        """went_to museum Monday vs Monday → same event → SUPPORT."""
        candidate = CandidateAtom(predicate="went_to", object="museum", temporal="monday", dimension="event")
        existing = [{"id": 1, "predicate": "went_to", "object": "museum", "status": "active", "temporal": "monday"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_SUPPORT

    def test_no_temporal_supports(self):
        """Both lack temporal → fall back to object matching → SUPPORT."""
        candidate = CandidateAtom(predicate="went_to", object="museum", dimension="event")
        existing = [{"id": 1, "predicate": "went_to", "object": "museum", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_SUPPORT


class TestContradiction:
    """relation=contradicts on same identity → CONTRADICT."""

    def test_contradiction_multi(self):
        """MULTI: likes coffee + relation=contradicts → CONTRADICT."""
        candidate = CandidateAtom(predicate="likes", object="coffee", relation="contradicts")
        existing = [{"id": 1, "predicate": "likes", "object": "coffee", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_CONTRADICT
        assert result.target_belief_id == 1

    def test_contradiction_single(self):
        """SINGLE: same object + contradicts → CONTRADICT."""
        candidate = CandidateAtom(predicate="lives_in", object="shanghai", relation="contradicts")
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_CONTRADICT

    def test_contradiction_no_existing_creates(self):
        """Contradiction with no existing belief → CREATE."""
        candidate = CandidateAtom(predicate="likes", object="coffee", relation="contradicts")
        existing = []
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_CREATE

    def test_contradiction_different_object_still_update(self):
        """SINGLE + different object + contradicts → still UPDATE."""
        candidate = CandidateAtom(predicate="lives_in", object="beijing", relation="contradicts")
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_UPDATE

    def test_support_not_strengthened_by_contradiction(self):
        """CRITICAL: contradicts must NEVER be treated as SUPPORT."""
        candidate = CandidateAtom(predicate="likes", object="coffee", relation="contradicts")
        existing = [{"id": 1, "predicate": "likes", "object": "coffee", "status": "active", "confidence": 0.9}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action != "SUPPORT"
        assert result.action == ACTION_CONTRADICT


class TestCityRoundTrip:
    """Shanghai → Beijing → Shanghai (key collision regression)."""

    def test_resolver_round_trip(self):
        """Round trip: resolver correctly identifies UPDATE back to Shanghai."""
        # Current: Beijing active, Shanghai superseded.
        # New claim: I live in Shanghai again → should UPDATE Beijing → Shanghai
        candidate = CandidateAtom(predicate="lives_in", object="shanghai")
        existing = [{"id": 2, "predicate": "lives_in", "object": "beijing", "status": "active"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_UPDATE
        assert result.target_belief_id == 2


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_predicate_noop(self):
        """Empty predicate → NOOP."""
        candidate = CandidateAtom(predicate="", object="coffee")
        result = resolve_identity(candidate, [], POLICY)
        assert result.action == ACTION_NOOP

    def test_unknown_predicate_defaults_to_multi(self):
        """Unknown predicate defaults to MULTI."""
        candidate = CandidateAtom(predicate="custom_verb", object="something")
        result = resolve_identity(candidate, [], POLICY)
        assert result.action == ACTION_CREATE

    def test_rejected_belief_ignored(self):
        """Rejected beliefs are not matched (they're dead)."""
        candidate = CandidateAtom(predicate="likes", object="coffee")
        existing = [{"id": 1, "predicate": "likes", "object": "coffee", "status": "rejected"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_CREATE  # treated as new

    def test_superseded_belief_ignored(self):
        """Superseded beliefs are not matched."""
        candidate = CandidateAtom(predicate="lives_in", object="beijing")
        existing = [{"id": 1, "predicate": "lives_in", "object": "shanghai", "status": "superseded"}]
        result = resolve_identity(candidate, existing, POLICY)
        assert result.action == ACTION_CREATE
