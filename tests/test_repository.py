"""Tests for core/repository.py — belief CRUD, merge strategy, lifecycle."""

import json

import pytest

from mirror_memory.core.models import Base, Belief, BeliefEvent, EvolutionJob, Snapshot
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
    update_belief_by_id,
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


class TestUpdateRevival:
    """Shanghai → Beijing → Shanghai: revive the superseded row."""

    def test_round_trip_no_unique_violation(self, db_session):
        """Key collision regression: reviving a superseded key must not violate UNIQUE."""
        set_memory_enabled(db_session, "u_revive", True)

        # Simulate pipeline: create Shanghai, then UPDATE to Beijing (superseded).
        from mirror_memory.core.repository import update_belief_by_id

        b1, _ = record_claim(
            db_session, "u_revive", dimension="fact", key="loc_shanghai",
            claim_text="lives in Shanghai", confidence=0.8,
            predicate="lives_in", object="shanghai", cardinality="single",
        )
        old, b2 = update_belief_by_id(
            db_session, b1.id, new_key="loc_beijing", new_object="beijing",
            new_claim_text="lives in Beijing",
            new_value={"predicate": "lives_in", "object": "beijing"},
        )
        db_session.commit()
        assert old.status == "superseded"
        assert b2.status == "active"

        # Round trip: UPDATE back to Shanghai → should revive, not collide.
        old2, new2 = update_belief_by_id(
            db_session, b2.id,  # Beijing is active
            new_key="loc_shanghai",
            new_object="shanghai",
            new_claim_text="lives in Shanghai again",
            new_predicate="lives_in",
            new_confidence=0.7,
            new_value={"via": "llm_semantic", "predicate": "lives_in", "object": "shanghai"},
        )
        db_session.commit()

        assert new2 is not None
        assert new2.object == "shanghai"
        assert new2.status == "active"
        assert old2.id == b2.id
        assert old2.status == "superseded"
        # The original Shanghai row should be revived (same id).
        assert new2.id == b1.id

    def test_update_preserves_metadata(self, db_session):
        """UPDATE path should preserve value metadata (temporal, context_tags)."""
        set_memory_enabled(db_session, "u_meta", True)

        b1, _ = record_claim(
            db_session, "u_meta", dimension="fact", key="loc_a",
            claim_text="lives in A", confidence=0.8,
            predicate="lives_in", object="a", cardinality="single",
        )
        b2, _ = record_claim(
            db_session, "u_meta", dimension="fact", key="loc_b",
            claim_text="lives in B", confidence=0.8,
            predicate="lives_in", object="b", cardinality="single",
        )

        old, new = update_belief_by_id(
            db_session, b2.id,
            new_key="loc_c",
            new_object="c",
            new_claim_text="lives in C",
            new_value={"via": "llm_semantic", "predicate": "lives_in", "object": "c", "temporal": "current"},
        )
        db_session.commit()
        val = json.loads(new.value_json)
        assert val.get("predicate") == "lives_in"
        assert val.get("object") == "c"
        assert val.get("temporal") == "current"


class TestForget:
    """Targeted forget: delete specific beliefs and session summaries."""

    def test_forget_belief(self, db_session):
        set_memory_enabled(db_session, "u_forget", True)
        b, _ = record_claim(
            db_session, "u_forget", dimension="topic", key="sleep",
            claim_text="trouble sleeping", confidence=0.8,
        )
        assert get_belief(db_session, "u_forget", "sleep") is not None

        from mirror_memory.core.repository import forget_belief
        result = forget_belief(db_session, "u_forget", b.id)
        assert result is True
        assert get_belief(db_session, "u_forget", "sleep") is None

    def test_forget_belief_wrong_user(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(
            db_session, "u1", dimension="topic", key="sleep",
            claim_text="test", confidence=0.5,
        )
        from mirror_memory.core.repository import forget_belief
        result = forget_belief(db_session, "u2", b.id)
        assert result is False

    def test_forget_session_summary(self, db_session):
        from mirror_memory.core.repository import forget_session_summary, store_session_summary
        store_session_summary(db_session, "u1", "s1", "hello world", 1)

        result = forget_session_summary(db_session, "u1", "s1")
        assert result is True

        # Verify deleted
        from mirror_memory.core.models import SessionSummary
        row = db_session.query(SessionSummary).filter_by(user_id="u1", session_id="s1").first()
        assert row is None

    def test_forget_nonexistent_session(self, db_session):
        from mirror_memory.core.repository import forget_session_summary
        result = forget_session_summary(db_session, "u1", "nonexistent")
        assert result is False

    def test_delete_user_memories_includes_summaries(self, db_session):
        from mirror_memory.core.repository import store_session_summary
        set_memory_enabled(db_session, "u_del", True)
        record_claim(db_session, "u_del", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5)
        store_session_summary(db_session, "u_del", "s1", "snippet", 1)

        counts = delete_user_memories(db_session, "u_del")
        assert counts["beliefs"] >= 1
        assert counts["session_summaries"] >= 1


class TestCorrect:
    """User-initiated correction: update belief in-place with corrected text."""

    def test_correct_updates_belief(self, db_session):
        set_memory_enabled(db_session, "u_corr", True)
        b, _ = record_claim(
            db_session, "u_corr", dimension="fact", key="age",
            claim_text="User is 28 years old", confidence=0.9,
            predicate="age", object="28", cardinality="single",
        )

        from mirror_memory.core.repository import correct_belief
        result = correct_belief(
            db_session, "u_corr", b.id,
            new_claim_text="User is actually 30 years old",
            new_object="30",
            correction_note="user corrected age",
        )
        db_session.commit()

        assert result is not None
        assert result.id == b.id  # same belief, updated in-place
        assert result.claim_text == "User is actually 30 years old"
        assert result.source == "user_corrected"
        assert result.object == "30"

        # Check correction event was logged
        from mirror_memory.core.repository import get_belief_events
        events = get_belief_events(db_session, "u_corr", b.id)
        correction_events = [e for e in events if e.event_type == "corrected"]
        assert len(correction_events) >= 1

    def test_correct_wrong_user_returns_none(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(
            db_session, "u1", dimension="fact", key="age",
            claim_text="28", confidence=0.9,
        )
        from mirror_memory.core.repository import correct_belief
        result = correct_belief(db_session, "u2", b.id, new_claim_text="30")
        assert result is None


class TestExplain:
    """Provenance chain for a belief."""

    def test_explain_returns_chain(self, db_session):
        set_memory_enabled(db_session, "u_explain", True)
        b, _ = record_claim(
            db_session, "u_explain", dimension="topic", key="sleep",
            claim_text="trouble sleeping", confidence=0.8,
            predicate="has", object="insomnia", cardinality="multi",
        )

        from mirror_memory.core.repository import explain_belief
        result = explain_belief(db_session, "u_explain", b.id)
        assert result is not None
        assert result["belief_id"] == b.id
        assert result["claim_text"] == "trouble sleeping"
        assert result["predicate"] == "has"
        assert result["object"] == "insomnia"
        assert result["status"] == "active"
        assert isinstance(result["events"], list)

    def test_explain_wrong_user_returns_none(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        b, _ = record_claim(
            db_session, "u1", dimension="topic", key="sleep",
            claim_text="test", confidence=0.5,
        )
        from mirror_memory.core.repository import explain_belief
        result = explain_belief(db_session, "u2", b.id)
        assert result is None


# ---------------------------------------------------------------------------
# Safety Closure regression tests
# ---------------------------------------------------------------------------


class TestStaleWritePrevention:
    """Worker publish must be blocked when memory state has changed."""

    def test_revision_bumps_on_forget(self, db_session):
        """forget_belief bumps state_revision, invalidating stale workers."""
        from mirror_memory.core.repository import get_state_revision, forget_belief

        set_memory_enabled(db_session, "u_stale", True)
        b, _ = record_claim(
            db_session, "u_stale", dimension="topic", key="sleep",
            claim_text="trouble sleeping", confidence=0.8,
        )
        rev_before = get_state_revision(db_session, "u_stale")
        forget_belief(db_session, "u_stale", b.id)
        rev_after = get_state_revision(db_session, "u_stale")
        assert rev_after > rev_before

    def test_revision_bumps_on_correct(self, db_session):
        """correct_belief bumps state_revision."""
        from mirror_memory.core.repository import get_state_revision, correct_belief

        set_memory_enabled(db_session, "u_rev", True)
        b, _ = record_claim(
            db_session, "u_rev", dimension="fact", key="age",
            claim_text="28", confidence=0.9,
            predicate="age", object="28",
        )
        rev_before = get_state_revision(db_session, "u_rev")
        correct_belief(db_session, "u_rev", b.id, new_claim_text="30",
                        new_object="30")
        rev_after = get_state_revision(db_session, "u_rev")
        assert rev_after > rev_before


class TestCorrectTripleConsistency:
    """correct_belief must update claim_text AND structured triple fields."""

    def test_correct_updates_object(self, db_session):
        """age 28 → 30: claim_text=30 AND object=30."""
        set_memory_enabled(db_session, "u_trip", True)
        b, _ = record_claim(
            db_session, "u_trip", dimension="fact", key="age",
            claim_text="User is 28", confidence=0.9,
            predicate="age", object="28", cardinality="single",
        )
        from mirror_memory.core.repository import correct_belief
        correct_belief(
            db_session, "u_trip", b.id,
            new_claim_text="User is 30",
            new_object="30",
        )
        db_session.refresh(b)
        assert b.claim_text == "User is 30"
        assert b.object == "30"

    def test_correct_preserves_before_state_in_event(self, db_session):
        """Correction event must record the previous state."""
        set_memory_enabled(db_session, "u_prov", True)
        b, _ = record_claim(
            db_session, "u_prov", dimension="fact", key="loc",
            claim_text="lives in Shanghai", confidence=0.9,
            predicate="lives_in", object="shanghai",
        )
        from mirror_memory.core.repository import correct_belief, get_belief_events
        correct_belief(
            db_session, "u_prov", b.id,
            new_claim_text="lives in Beijing",
            new_object="beijing",
            new_predicate="lives_in",
        )
        events = get_belief_events(db_session, "u_prov", b.id)
        correction = [e for e in events if e.event_type == "corrected"]
        assert len(correction) >= 1
        detail = json.loads(correction[-1].detail_json)
        assert detail.get("before", {}).get("object") == "shanghai"
        assert detail.get("after", {}).get("object") == "beijing"


class TestForgetInvalidatesDerived:
    """forget_belief must invalidate snapshots and cancel worker jobs."""

    def test_forget_deletes_snapshots(self, db_session):
        from mirror_memory.core.repository import forget_belief
        from mirror_memory.worker.snapshot import persist_snapshot

        set_memory_enabled(db_session, "u_snap", True)
        b, _ = record_claim(
            db_session, "u_snap", dimension="topic", key="sleep",
            claim_text="trouble sleeping", confidence=0.8,
        )
        persist_snapshot(db_session, "u_snap", {"patterns": []}, {}, "wm1")
        # Verify snapshot exists
        snaps = db_session.query(Snapshot).filter_by(user_id="u_snap").all()
        assert len(snaps) >= 1

        forget_belief(db_session, "u_snap", b.id)
        # Snapshot should be deleted
        snaps = db_session.query(Snapshot).filter_by(user_id="u_snap").all()
        assert len(snaps) == 0

    def test_forget_cancels_pending_jobs(self, db_session):
        from mirror_memory.core.repository import forget_belief

        set_memory_enabled(db_session, "u_job", True)
        b, _ = record_claim(
            db_session, "u_job", dimension="topic", key="sleep",
            claim_text="test", confidence=0.5,
        )
        # Create a pending job
        job = EvolutionJob(
            id="test-job-1", user_id="u_job",
            evidence_watermark="{}", prompt_version="v1",
            idempotency_key="u_job:none:v1", status="pending",
        )
        db_session.add(job)
        db_session.commit()

        forget_belief(db_session, "u_job", b.id)
        db_session.refresh(job)
        assert job.status == "cancelled"
        assert job.error_code == "belief_forgotten"


class TestConcurrentStaleWrite:
    """Two-session regression: revision barrier under real concurrency."""

    def test_fresh_read_sees_concurrent_revision(self, db_session):
        """get_state_revision must see changes from another session."""
        from mirror_memory.core.repository import get_state_revision, touch_memory_state

        set_memory_enabled(db_session, "u_conc", True)
        rev1 = get_state_revision(db_session, "u_conc")
        touch_memory_state(db_session, "u_conc")
        rev2 = get_state_revision(db_session, "u_conc")
        assert rev2 > rev1

    def test_worker_blocks_on_revision_mismatch(self, db_session):
        """Worker publish must be blocked when revision changes mid-compute."""
        from mirror_memory.core.repository import get_state_revision, touch_memory_state

        set_memory_enabled(db_session, "u_worker", True)
        record_claim(
            db_session, "u_worker", dimension="topic", key="sleep",
            claim_text="test", confidence=0.8, session_id="s1",
        )
        claimed = get_state_revision(db_session, "u_worker")
        # Simulate user action bumping revision
        touch_memory_state(db_session, "u_worker")
        current = get_state_revision(db_session, "u_worker")
        assert current != claimed, "Worker should detect revision change"

    def test_two_session_concurrent_bump(self, db_session):
        """Real two-session test: Session B bumps while Session A computes."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session as SASession
        from mirror_memory.core.models import Base as MemBase
        from mirror_memory.core.repository import (
            get_state_revision, touch_memory_state, set_memory_enabled, record_claim,
        )

        # Use a shared file-based DB so two sessions see the same data.
        import tempfile, os
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        db_url = f"sqlite:///{db_path}"
        eng = create_engine(db_url)
        MemBase.metadata.create_all(eng)

        # Session A: worker
        sA = SASession(eng)
        set_memory_enabled(sA, "u2", True)
        record_claim(sA, "u2", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.8, session_id="s1")
        sA.commit()

        # Worker claims revision
        claimed = get_state_revision(sA, "u2")
        assert claimed >= 1

        # Session B: user action (forget) in a separate session
        sB = SASession(eng)
        touch_memory_state(sB, "u2")
        sB.commit()
        sB.close()

        # Worker checks fresh revision (must see Session B's change)
        current = get_state_revision(sA, "u2")
        assert current > claimed, f"Worker should see revision bump: {current} > {claimed}"
        sA.close()
        eng.dispose()

    def test_atomic_increment_no_lost_updates(self, db_session):
        """Atomic bump: 10 concurrent threads each bump once → 10 unique revisions."""
        import threading
        import tempfile
        import os
        from sqlalchemy import create_engine as ce
        from sqlalchemy.orm import Session as SASession
        from mirror_memory.core.models import Base as MemBase

        db_path = os.path.join(tempfile.mkdtemp(), "atomic.db")
        eng = ce(f"sqlite:///{db_path}")
        MemBase.metadata.create_all(eng)

        # Setup user
        with SASession(eng) as s:
            from mirror_memory.core.repository import set_memory_enabled as sme
            sme(s, "u_atomic", True)
            s.commit()

        results = []
        def bump():
            from mirror_memory.core.repository import bump_state_revision as bsr
            e = ce(f"sqlite:///{db_path}")
            with SASession(e) as s:
                r = bsr(s, "u_atomic")
                s.commit()
                results.append(r)
            e.dispose()

        threads = [threading.Thread(target=bump) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 10, f"Expected 10 unique revisions, got {len(set(results))}"
        from mirror_memory.core.repository import get_state_revision as gsr
        with SASession(eng) as s:
            final = gsr(s, "u_atomic")
            assert final == 11, f"Expected 11, got {final}"
        eng.dispose()