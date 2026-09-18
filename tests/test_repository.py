"""Tests for core/repository.py — belief CRUD, merge strategy, lifecycle."""

import json

import pytest

from mirror_memory.core.models import Base, Belief, BeliefEvent
from mirror_memory.core.repository import (
    confirm_belief,
    delete_user_memories,
    get_belief,
    is_memory_enabled,
    list_active_beliefs,
    list_rejected_beliefs,
    record_claim,
    record_extraction_stats,
    reject_belief,
    set_memory_enabled,
    store_session_summary,
)


class TestMemoryPreference:
    def test_missing_row_means_enabled(self, db_session):
        assert is_memory_enabled(db_session, "user1") is True

    def test_enabled_row(self, db_session):
        set_memory_enabled(db_session, "user1", True)
        assert is_memory_enabled(db_session, "user1") is True

    def test_disabled_row(self, db_session):
        set_memory_enabled(db_session, "user1", False)
        assert is_memory_enabled(db_session, "user1") is False

    def test_re_enable(self, db_session):
        set_memory_enabled(db_session, "user1", False)
        set_memory_enabled(db_session, "user1", True)
        assert is_memory_enabled(db_session, "user1") is True


class TestRecordClaim:
    def test_new_belief_created(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        belief, event = record_claim(
            db_session, "u1", dimension="topic", key="sleep",
            claim_text="trouble sleeping", confidence=0.6,
        )
        assert belief is not None
        assert event == "created"
        assert belief.key == "sleep"
        assert belief.dimension == "topic"
        assert belief.layer == "L4"  # extracted always L4

    def test_support_increases_confidence(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b1, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                             claim_text="sleep issue", confidence=0.5)
        conf1 = b1.confidence
        b2, event = record_claim(db_session, "u1", dimension="topic", key="sleep",
                                 claim_text="still sleeping badly", confidence=0.6,
                                 relation="supports")
        assert event == "supported"
        assert b2.confidence > conf1

    def test_contradict_attenuates_confidence(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b1, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                             claim_text="sleep issue", confidence=0.8)
        conf1 = b1.confidence
        b2, event = record_claim(db_session, "u1", dimension="topic", key="sleep",
                                 claim_text="sleeping better now", confidence=0.5,
                                 relation="contradicts")
        assert event == "contradicted"
        assert b2.confidence < conf1

    def test_resurrection_guard(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                            claim_text="sleep issue", confidence=0.5)
        reject_belief(db_session, "u1", b.id)
        # Try to create new claim for rejected key
        result, event = record_claim(db_session, "u1", dimension="topic", key="sleep",
                                     claim_text="new sleep", confidence=0.5)
        assert result is None
        assert event == "resurrection_guard"

    def test_disabled_memory_returns_none(self, db_session):
        set_memory_enabled(db_session, "u1", False)
        result, event = record_claim(db_session, "u1", dimension="topic", key="sleep",
                                     claim_text="test", confidence=0.5)
        assert result is None
        assert event == "profile_memory_disabled"

    def test_invalid_relation_raises(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        with pytest.raises(ValueError, match="invalid"):
            record_claim(db_session, "u1", dimension="topic", key="sleep",
                         claim_text="test", confidence=0.5, relation="invalid")

    def test_layer_clamping_extracted(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                            claim_text="test", confidence=0.5, source="extracted", layer="L2")
        assert b.layer == "L4"  # forced to L4

    def test_evidence_json_stored(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                            claim_text="test", confidence=0.5, evidence_message_ids=[1, 2, 3])
        evidence = json.loads(b.evidence_json)
        assert 1 in evidence

    def test_context_tags_merged(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b1, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                             claim_text="test", confidence=0.5, context_tags=["work"])
        b2, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                             claim_text="test2", confidence=0.5, relation="supports",
                             context_tags=["family"])
        val = json.loads(b2.value_json)
        tags = val.get("context_tags", [])
        assert "work" in tags
        assert "family" in tags

    def test_claim_text_update_longer(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b1, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                             claim_text="short", confidence=0.5)
        b2, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                             claim_text="much longer and more detailed description",
                             confidence=0.5, relation="supports")
        assert b2.claim_text == "much longer and more detailed description"

    def test_confidence_ceiling(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                            claim_text="test", confidence=0.9)
        # Many supports shouldn't exceed ceiling
        for i in range(20):
            record_claim(db_session, "u1", dimension="topic", key="sleep",
                         claim_text=f"support {i}", confidence=0.9, relation="supports")
        db_session.refresh(b)
        assert b.confidence <= 0.95

    def test_blocked_key_prefix_raises(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        with pytest.raises(ValueError, match="blocked"):
            record_claim(db_session, "u1", dimension="topic", key="distortion.test",
                         claim_text="test", confidence=0.5, blocked_key_prefixes=("distortion.",))


class TestConfirmReject:
    def _make_belief(self, db_session, user_id="u1"):
        set_memory_enabled(db_session, user_id, True)
        b, _ = record_claim(db_session, user_id, dimension="topic", key="sleep",
                            claim_text="test", confidence=0.5)
        return b

    def test_confirm_upgrades_layer(self, db_session):
        b = self._make_belief(db_session)
        result = confirm_belief(db_session, "u1", b.id)
        assert result is not None
        assert result.layer == "L2"

    def test_reject_sets_status(self, db_session):
        b = self._make_belief(db_session)
        result = reject_belief(db_session, "u1", b.id)
        assert result is not None
        assert result.status == "rejected"
        assert result.confidence == 0.0

    def test_confirm_wrong_user(self, db_session):
        b = self._make_belief(db_session, "u1")
        result = confirm_belief(db_session, "u2", b.id)
        assert result is None

    def test_reject_wrong_user(self, db_session):
        b = self._make_belief(db_session, "u1")
        result = reject_belief(db_session, "u2", b.id)
        assert result is None


class TestListBeliefs:
    def test_list_active(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5)
        beliefs = list_active_beliefs(db_session, "u1")
        assert len(beliefs) >= 1

    def test_list_rejected(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(db_session, "u1", dimension="topic", key="sleep",
                            claim_text="test", confidence=0.5)
        reject_belief(db_session, "u1", b.id)
        rejected = list_rejected_beliefs(db_session, "u1")
        assert len(rejected) >= 1

    def test_filter_by_dimensions(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5)
        record_claim(db_session, "u1", dimension="preference", key="like",
                     claim_text="test", confidence=0.5)
        topics = list_active_beliefs(db_session, "u1", dimensions=("topic",))
        assert all(b.dimension == "topic" for b in topics)


class TestDeleteUserMemories:
    def test_delete_returns_counts(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5)
        counts = delete_user_memories(db_session, "u1")
        assert counts["beliefs"] >= 1

    def test_delete_empty_user(self, db_session):
        counts = delete_user_memories(db_session, "nonexistent")
        assert all(v == 0 for v in counts.values())


class TestStoreSessionSummary:
    def test_first_call_creates_row(self, db_session):
        store_session_summary(db_session, "u1", "s1", "hello world", 1)
        from mirror_memory.core.models import SessionSummary
        rows = db_session.query(SessionSummary).filter_by(user_id="u1").all()
        assert len(rows) == 1
        assert rows[0].summary_text == "hello world"

    def test_subsequent_calls_append(self, db_session):
        store_session_summary(db_session, "u1", "s1", "hello", 1)
        store_session_summary(db_session, "u1", "s1", "world", 2)
        from mirror_memory.core.models import SessionSummary
        row = db_session.query(SessionSummary).filter_by(user_id="u1").first()
        assert "hello" in row.summary_text
        assert "world" in row.summary_text

    def test_different_sessions_separate_rows(self, db_session):
        store_session_summary(db_session, "u1", "s1", "hello", 1)
        store_session_summary(db_session, "u1", "s2", "world", 1)
        from mirror_memory.core.models import SessionSummary
        rows = db_session.query(SessionSummary).filter_by(user_id="u1").all()
        assert len(rows) == 2


class TestRecordExtractionStats:
    def test_creates_stats_row(self, db_session):
        record_extraction_stats(db_session, "u1", session_id="s1", trigger="topic_flow")
        from mirror_memory.core.models import ExtractionStats
        rows = db_session.query(ExtractionStats).filter_by(user_id="u1").all()
        assert len(rows) == 1
        assert rows[0].trigger == "topic_flow"