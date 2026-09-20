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
