"""Regression tests for predicate-variant identity resolution.

StateBench's A-class failures: the extractor emits a different verb for the
same attribute on different turns -- ``works_at``, ``started_at``, ``joined``,
``returned_to`` for employment; ``lives_in``, ``lived_in``, ``moved_to`` for
residence.  Identity resolution matched on the literal canonical predicate, so
a variant matched nothing, the claim was stored as a new belief instead of
superseding the old one, and the stale value stayed active with no validity
interval.  A current-state question then returned it as current.

Two things had to line up, and both were missing:

1. the synonym map did not resolve the variant to the canonical predicate;
2. ``works_as`` was absent from the identity policy, so it defaulted to
   ``multi`` cardinality and never superseded at all.
"""

import pytest

from mirror_memory.config.loader import load_config
from mirror_memory.memory.canonicalize import canonicalize_atom


@pytest.fixture
def config():
    return load_config("config/")


# ---------------------------------------------------------------------------
# The synonym map must resolve the variants the extractor actually emits
# ---------------------------------------------------------------------------


class TestPredicateVariantSynonyms:
    @pytest.mark.parametrize("variant,canonical", [
        # Employment -- observed on "I just started at Globex", "I joined
        # Globex", "Acme offered me my old role back".
        ("started_at", "works_at"),
        ("joined", "works_at"),
        ("returned_to", "works_at"),
        ("employed_at", "works_at"),
        ("employed_by", "works_at"),
        ("hired_by", "works_at"),
        ("works_for", "works_at"),
        ("began_at", "works_at"),
        # Residence -- observed on "I lived in Shanghai".
        ("lived_in", "lives_in"),
        ("moved_to", "lives_in"),
        ("relocated_to", "lives_in"),
        # Profession.
        ("retrained", "profession"),
        ("became", "profession"),
    ])
    def test_variant_resolves_to_canonical(self, config, variant, canonical):
        assert config.predicate_synonyms.get(variant) == canonical

    def test_canonicalize_atom_applies_the_map(self, config):
        synonyms = config.predicate_synonyms
        _subject, pred, obj = canonicalize_atom(
            "user", "started_at", "globex", synonyms
        )
        assert pred == "works_at"
        assert obj == "globex"

    def test_unknown_verb_is_left_alone(self, config):
        """An unmapped verb must not be silently folded into another slot."""
        _subject, pred, obj = canonicalize_atom(
            "user", "invented_verb", "thing", config.predicate_synonyms
        )
        assert pred == "invented_verb"

    def test_negation_verb_is_not_mapped_to_the_slot(self, config):
        """"left" describes a departure, not an employer.

        Mapping it to ``works_at`` would create a belief claiming the user
        works at their previous job -- worse than not closing the interval.
        """
        assert "left" not in config.predicate_synonyms


# ---------------------------------------------------------------------------
# Single-valued attributes must be declared, or they never supersede
# ---------------------------------------------------------------------------


class TestSingleValuedPredicates:
    @pytest.mark.parametrize("predicate", [
        "lives_in", "works_at", "works_as", "profession", "age", "studies_at",
    ])
    def test_declared_single(self, config, predicate):
        assert config.identity_policy.get(predicate) == "single", (
            f"{predicate} defaults to multi, so a new value creates a second "
            "belief instead of superseding the old one"
        )

    @pytest.mark.parametrize("predicate", [
        "lives_in", "works_at", "works_as", "profession",
    ])
    def test_declared_current_state(self, config, predicate):
        assert config.temporal_policy.get(predicate) == "current_state"

    def test_multi_predicates_stay_multi(self, config):
        for predicate in ("likes", "loves", "has", "skills"):
            assert config.identity_policy.get(predicate) == "multi"


# ---------------------------------------------------------------------------
# End to end: a variant verb must supersede, not duplicate
# ---------------------------------------------------------------------------


class TestVariantSupersedesEndToEnd:
    @pytest.fixture
    def session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from mirror_memory.core.models import Base

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)
        yield factory()
        Base.metadata.drop_all(engine)
        engine.dispose()

    @staticmethod
    def _observe(session, pipeline, user_id, text, turn):
        from mirror_memory.core.repository import set_memory_enabled

        set_memory_enabled(session, user_id, True)
        pipeline.observe(session, user_id, "s1", text, turn)

    def test_started_at_supersedes_works_at(self, session, config):
        """The exact A-class failure: variant verb must not create a duplicate."""
        from mirror_memory.memory.atom import CandidateAtom
        from mirror_memory.memory.identity import resolve_identity

        synonyms = config.predicate_synonyms
        from mirror_memory.memory.canonicalize import canonicalize_atom

        _s, canon, obj = canonicalize_atom("user", "started_at", "globex", synonyms)
        existing = [{
            "id": 1, "predicate": "works_at", "object": "acme_corp",
            "status": "active", "confidence": 0.9,
            "valid_from": None, "valid_to": None,
        }]
        result = resolve_identity(
            CandidateAtom(subject="user", predicate=canon, object=obj),
            existing,
            config.identity_policy,
            config.temporal_policy,
        )
        assert result.action == "UPDATE", (
            "a variant verb for the same slot must supersede, not create"
        )
        assert result.target_belief_id == 1

    def test_lived_in_supersedes_lives_in(self, config):
        from mirror_memory.memory.atom import CandidateAtom
        from mirror_memory.memory.canonicalize import canonicalize_atom
        from mirror_memory.memory.identity import resolve_identity

        _s, canon, obj = canonicalize_atom(
            "user", "lived_in", "shanghai", config.predicate_synonyms
        )
        result = resolve_identity(
            CandidateAtom(subject="user", predicate=canon, object=obj),
            [{"id": 1, "predicate": "lives_in", "object": "berlin",
              "status": "active", "confidence": 0.9,
              "valid_from": None, "valid_to": None}],
            config.identity_policy,
            config.temporal_policy,
        )
        assert result.action == "UPDATE"

    def test_works_as_supersedes_itself(self, config):
        """`works_as` was absent from the policy and defaulted to multi."""
        from mirror_memory.memory.atom import CandidateAtom
        from mirror_memory.memory.identity import resolve_identity

        result = resolve_identity(
            CandidateAtom(subject="user", predicate="works_as", object="data_analyst"),
            [{"id": 1, "predicate": "works_as", "object": "graphic_designer",
              "status": "active", "confidence": 0.9,
              "valid_from": None, "valid_to": None}],
            config.identity_policy,
            config.temporal_policy,
        )
        assert result.action == "UPDATE"
        assert result.lifecycle == "TEMPORAL_UPDATE"
