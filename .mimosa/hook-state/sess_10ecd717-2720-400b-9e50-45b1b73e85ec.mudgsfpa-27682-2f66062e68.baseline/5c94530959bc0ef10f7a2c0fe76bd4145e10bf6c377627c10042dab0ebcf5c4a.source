"""Tests for worker modules — evolution, snapshot, pipeline, synthesis."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base, Belief, EvolutionJob, Snapshot
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.worker.evolution import (
    _compile_policy,
    _load_evidence,
    _parse_formulate_json,
    _validate,
    enqueue_job,
    process_pending,
)
from mirror_memory.worker.snapshot import persist_snapshot, promote_shadow_if_ready


class TestEnqueueJob:
    @pytest.fixture
    def config(self):
        return load_config("config/")

    def test_creates_pending_job(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5, session_id="s1")
        job = enqueue_job(db_session, "u1", config)
        assert job is not None
        assert job.status == "pending"

    def test_disabled_memory_returns_none(self, db_session, config):
        set_memory_enabled(db_session, "u1", False)
        assert enqueue_job(db_session, "u1", config) is None

    def test_existing_pending_returns_same(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5, session_id="s1")
        j1 = enqueue_job(db_session, "u1", config)
        j2 = enqueue_job(db_session, "u1", config)
        assert j1.id == j2.id

    def test_updates_watermark_on_existing(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5, session_id="s1")
        j1 = enqueue_job(db_session, "u1", config)
        wm1 = j1.evidence_watermark
        record_claim(db_session, "u1", dimension="topic", key="work",
                     claim_text="new evidence", confidence=0.5, session_id="s1")
        j2 = enqueue_job(db_session, "u1", config)
        assert j2.id == j1.id
        assert j2.evidence_watermark != wm1

    def test_recycles_completed_job(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5, session_id="s1")
        j1 = enqueue_job(db_session, "u1", config)
        j1.status = "completed"
        db_session.commit()
        j2 = enqueue_job(db_session, "u1", config)
        assert j2.id == j1.id
        assert j2.status == "pending"


class TestProcessPending:
    @pytest.fixture
    def config(self):
        cfg = load_config("config/")
        cfg.llm_client = lambda **kw: '{"patterns": [], "support_policy": {}}'
        # Need to make the lambda look like it has a generate method
        class FakeLLM:
            def generate(self, **kw):
                return '{"patterns": [], "support_policy": {}}'
        cfg.llm_client = FakeLLM()
        return cfg

    def test_processes_pending_job(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="User has trouble sleeping", confidence=0.8,
                     session_id="s1")
        enqueue_job(db_session, "u1", config)

        def factory():
            return db_session

        process_pending(factory, config)
        job = db_session.query(EvolutionJob).filter_by(user_id="u1").first()
        assert job.status == "completed"

    def test_crash_recovery(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5, session_id="s1")
        job = enqueue_job(db_session, "u1", config)
        job_id = job.id
        job.status = "running"
        job.started_at = datetime.now(UTC) - timedelta(seconds=600)
        db_session.commit()

        def factory():
            return db_session

        process_pending(factory, config)
        # Re-query from DB (process_pending closes session internally)
        updated_job = db_session.query(EvolutionJob).filter_by(id=job_id).first()
        assert updated_job.status in ("pending", "completed")

    def test_empty_queue(self, db_session, config):
        def factory():
            return db_session
        # Should not raise
        process_pending(factory, config)


class TestLoadEvidence:
    def test_returns_beliefs_and_sessions(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5, session_id="s1")
        record_claim(db_session, "u1", dimension="topic", key="work",
                     claim_text="test", confidence=0.5, session_id="s2")
        evidence = _load_evidence(db_session, "u1")
        assert len(evidence["beliefs"]) >= 2
        assert evidence["distinct_sessions"] == 2

    def test_empty_user(self, db_session):
        evidence = _load_evidence(db_session, "nonexistent")
        assert evidence["beliefs"] == []


class TestValidate:
    def _config(self, forbidden=None, third_party=None):
        cfg = load_config("config/")
        if forbidden is not None:
            cfg.worker.forbidden_labels = forbidden
        if third_party is not None:
            cfg.worker.third_party_markers = third_party
        return cfg

    def test_forbidden_labels_filtered(self):
        config = self._config(forbidden=["narcissist", "borderline"])
        candidate = {"patterns": [{"description": "shows narcissist tendencies"}]}
        result = _validate(candidate, {"beliefs": []}, config)
        assert len(result["patterns"]) == 0

    def test_empty_description_filtered(self):
        config = self._config()
        candidate = {"patterns": [{"description": "  "}]}
        result = _validate(candidate, {"beliefs": []}, config)
        assert len(result["patterns"]) == 0

    def test_third_party_filtered(self):
        config = self._config(third_party=["his friend", "her friend"])
        candidate = {"patterns": [{"description": "his friend causes him anxiety"}]}
        result = _validate(candidate, {"beliefs": []}, config)
        assert len(result["patterns"]) == 0

    def test_valid_pattern_passes(self):
        config = self._config()
        candidate = {"patterns": [{"description": "tends to avoid social situations", "confidence": 0.7}]}
        result = _validate(candidate, {"beliefs": [], "distinct_sessions": 2}, config)
        assert len(result["patterns"]) == 1

    def test_single_session_forces_verification(self):
        config = self._config()
        candidate = {"patterns": [{"description": "tends to avoid", "confidence": 0.7}]}
        result = _validate(candidate, {"beliefs": [], "distinct_sessions": 1}, config)
        assert result["patterns"][0]["needs_verification"] is True

    def test_max_three_patterns(self):
        config = self._config()
        candidate = {"patterns": [
            {"description": f"pattern {i}", "confidence": 0.5} for i in range(5)
        ]}
        result = _validate(candidate, {"beliefs": [], "distinct_sessions": 2}, config)
        assert len(result["patterns"]) == 3

    def test_evidence_key_check(self):
        config = self._config()
        candidate = {"patterns": [
            {"description": "test", "confidence": 0.5, "evidence_keys": ["nonexistent"]}
        ]}
        result = _validate(candidate, {"beliefs": [{"key": "sleep"}]}, config)
        assert len(result["patterns"]) == 0


class TestCompilePolicy:
    def test_defaults(self):
        policy = _compile_policy({})
        assert policy["response_length"] == "normal"

    def test_valid_override(self):
        candidate = {"support_policy": {"response_length": "brief", "pacing": "slow"}}
        policy = _compile_policy(candidate)
        assert policy["response_length"] == "brief"

    def test_invalid_value_kept_default(self):
        candidate = {"support_policy": {"response_length": "verbose"}}
        policy = _compile_policy(candidate)
        assert policy["response_length"] == "normal"


class TestParseFormulateJson:
    def test_valid_dict(self):
        raw = '{"patterns": [], "support_policy": {}}'
        result = _parse_formulate_json(raw)
        assert result["patterns"] == []

    def test_markdown_fenced(self):
        raw = '```json\n{"patterns": [], "support_policy": {}}\n```'
        result = _parse_formulate_json(raw)
        assert "patterns" in result

    def test_invalid_returns_defaults(self):
        result = _parse_formulate_json("not json")
        assert result["patterns"] == []


class TestPersistSnapshot:
    def test_first_snapshot_version_1(self, db_session):
        persist_snapshot(db_session, "u1", {"test": 1}, {}, "wm1")
        snap = db_session.query(Snapshot).filter_by(user_id="u1").first()
        assert snap is not None
        assert snap.version == 1

    def test_version_increments(self, db_session):
        persist_snapshot(db_session, "u1", {"v": 1}, {}, "wm1")
        snap2 = persist_snapshot(db_session, "u1", {"v": 2}, {}, "wm2")
        assert snap2.version == 2

    def test_old_retired(self, db_session):
        s1 = persist_snapshot(db_session, "u1", {"v": 1}, {}, "wm1")
        persist_snapshot(db_session, "u1", {"v": 2}, {}, "wm2")
        db_session.refresh(s1)
        assert s1.status == "retired"

    def test_shadow_mode(self, db_session):
        snap = persist_snapshot(db_session, "u1", {"test": 1}, {}, "wm1", shadow=True)
        assert snap.status == "shadow"


class TestPromoteShadow:
    def test_no_shadow_returns_false(self, db_session):
        assert promote_shadow_if_ready(db_session, "u1") is False

    def test_shadow_with_multi_session_promotes(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5, session_id="s1")
        record_claim(db_session, "u1", dimension="topic", key="work",
                     claim_text="test", confidence=0.5, session_id="s2")
        snap = persist_snapshot(db_session, "u1", {"test": 1}, {}, "wm1", shadow=True)
        result = promote_shadow_if_ready(db_session, "u1")
        assert result is True
        db_session.refresh(snap)
        assert snap.status == "active"

    def test_shadow_with_single_session_stays(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        record_claim(db_session, "u1", dimension="topic", key="sleep",
                     claim_text="test", confidence=0.5, session_id="s1")
        snap = persist_snapshot(db_session, "u1", {"test": 1}, {}, "wm1", shadow=True)
        result = promote_shadow_if_ready(db_session, "u1")
        assert result is False
        db_session.refresh(snap)
        assert snap.status == "shadow"


class TestPipeline:
    def test_observe_extracts_claims(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        from mirror_memory.extraction.pipeline import ExtractionPipeline
        pipeline = ExtractionPipeline(config=config)
        claims = pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping", 1)
        assert isinstance(claims, list)

    def test_observe_empty_text(self, db_session, config):
        set_memory_enabled(db_session, "u1", True)
        from mirror_memory.extraction.pipeline import ExtractionPipeline
        pipeline = ExtractionPipeline(config=config)
        claims = pipeline.observe(db_session, "u1", "s1", "", 1)
        assert claims == []