"""Tests for memory/canonicalize.py — predicate/object normalization."""

from mirror_memory.memory.canonicalize import (
    canonicalize_atom,
    canonicalize_object,
    canonicalize_predicate,
)


class TestCanonicalizePredicate:
    def test_lowercase(self):
        assert canonicalize_predicate("Likes") == "likes"

    def test_spaces_to_underscores(self):
        assert canonicalize_predicate("lives in") == "lives_in"

    def test_hyphens_to_underscores(self):
        assert canonicalize_predicate("went-to") == "went_to"

    def test_synonym_mapping(self):
        synonyms = {"loves": "likes", "enjoys": "likes", "moved_to": "lives_in"}
        assert canonicalize_predicate("loves", synonyms) == "likes"
        assert canonicalize_predicate("enjoys", synonyms) == "likes"
        assert canonicalize_predicate("moved_to", synonyms) == "lives_in"

    def test_no_synonym_keeps_original(self):
        synonyms = {"loves": "likes"}
        assert canonicalize_predicate("works_at", synonyms) == "works_at"

    def test_empty(self):
        assert canonicalize_predicate("") == ""


class TestCanonicalizeObject:
    def test_lowercase(self):
        assert canonicalize_object("Coffee") == "coffee"

    def test_collapse_whitespace(self):
        assert canonicalize_object("  New   York  ") == "new_york"

    def test_remove_articles(self):
        assert canonicalize_object("the coffee") == "coffee"
        assert canonicalize_object("a painting") == "painting"
        assert canonicalize_object("an art show") == "art_show"

    def test_empty(self):
        assert canonicalize_object("") == ""


class TestCanonicalizeAtom:
    def test_full_triple(self):
        s, p, o = canonicalize_atom("User", "Loves", "The Coffee", {"loves": "likes"})
        assert s == "user"
        assert p == "likes"
        assert o == "coffee"

    def test_no_synonyms(self):
        s, p, o = canonicalize_atom("User", "went_to", "art show")
        assert p == "went_to"
        assert o == "art_show"
