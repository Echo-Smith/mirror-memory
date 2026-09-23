"""Tests for the verification loop -- candidate selection, injection,
judgment, and belief lifecycle (confirm/reject)."""

import pytest

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core.models import InterventionEvent
from mirror_memory.core.repository import (
    get_belief,
    get_pending_verification,
    has_unanswered_injection,
    record_claim,
    record_intervention_event,
)
from mirror_memory.extraction.verification import (
    build_verification_question,
    get_question_candidates,
    parse_verdict,
    record_question_injection,
    run_verification_judgment,
)


class ScriptedLLM:
    """LLM test double returning a canned verification verdict."""

    model = "test-model"

    def __init__(self, response: str = "unclear"):
        self.response = response
        self.calls: list[dict] = []

    def generate(self, *, system_prompt, payload_text, fallback=None):
        self.calls.append({"system_prompt": system_prompt, "payload_text": payload_text})
        return self.response


@pytest.fixture
def cfg():
    return load_config("config/")


def _seed_hypothesis(session, user_id="u1", *, key="i_want", confidence=0.85, origin="s1", last="s2"):
    """Seed an L4 hypothesis with cross-session evidence."""
    belief, _ = record_claim(
        session,
        user_id,
        dimension="goal",
        key=key,
        claim_text="User wants to write a book",
        confidence=confidence,
        session_id=origin,
    )
    belief.last_evidence_session_id = last
    session.flush()
    return belief


class TestParseVerdict:
    def test_bare_word_confirm(self):
        assert parse_verdict("confirm") == "confirm"

    def test_bare_word_deny_with_punctuation(self):
        assert parse_verdict("deny.") == "deny"

    def test_json_envelope(self):
        assert parse_verdict('{"verdict": "deny", "reason": "no"}') == "deny"

    def test_json_fenced(self):
        assert parse_verdict('```json\n{"verdict": "confirm"}\n```') == "confirm"

    def test_garbage_is_unclear(self):
        assert parse_verdict("The user said something unrelated") == "unclear"

    def test_empty_is_unclear(self):
        assert parse_verdict("") == "unclear"
        assert parse_verdict(None) == "unclear"


class TestQuestionCandidates:
    def test_eligible_l4_cross_session(self, db_session, cfg):
        _seed_hypothesis(db_session)
        candidates = get_question_candidates(db_session, "u1", cfg)
        assert len(candidates) == 1
        assert candidates[0].key == "i_want"
        assert candidates[0].layer == "L4"

    def test_single_session_evidence_excluded(self, db_session, cfg):
        _seed_hypothesis(db_session, origin="s1", last="s1")
        assert get_question_candidates(db_session, "u1", cfg) == []

    def test_low_confidence_excluded(self, db_session, cfg):
        _seed_hypothesis(db_session, confidence=0.5)
        assert get_question_candidates(db_session, "u1", cfg) == []

    def test_confirmed_belief_excluded(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)
        from mirror_memory.core.repository import confirm_belief

        confirm_belief(db_session, "u1", belief.id)  # L4 -> L2
        assert get_question_candidates(db_session, "u1", cfg) == []

    def test_sorted_by_tier_times_confirm_rate(self, db_session, cfg):
        low = _seed_hypothesis(db_session, key="i_want")  # dimension goal
        high = _seed_hypothesis(db_session, key="i_have", origin="s3", last="s4")
        high.dimension = "fact"  # tier 2 beats goal tier 1 (default tiers)
        db_session.flush()
        cfg.question_value_tiers = {"goal": 1, "fact": 2}
        candidates = get_question_candidates(db_session, "u1", cfg, limit=2)
        assert [c.id for c in candidates] == [high.id, low.id]

    def test_disabled_memory_returns_empty(self, db_session, cfg):
        from mirror_memory.core.repository import set_memory_enabled

        _seed_hypothesis(db_session)
        set_memory_enabled(db_session, "u1", False)
        assert get_question_candidates(db_session, "u1", cfg) == []


class TestInjection:
    def test_record_injection_event(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)
        event = record_question_injection(db_session, "u1", "s1", belief, cfg)
        assert event is not None
        assert event.kind == "question_injected"
        assert has_unanswered_injection(db_session, "u1") is True

    def test_no_double_inject(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)
        first = record_question_injection(db_session, "u1", "s1", belief, cfg)
        second = record_question_injection(db_session, "u1", "s1", belief, cfg)
        assert first is not None
        assert second is None  # previous injection still unanswered

    def test_question_text_uses_display_label(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)  # goal/i_want has a display label
        question = build_verification_question(belief, cfg)
        assert question == 'Quick check — "愿望" — does that sound right?'

    def test_question_text_falls_back_to_claim(self, db_session, cfg):
        belief = _seed_hypothesis(db_session, key="unlisted_key")  # no display label
        question = build_verification_question(belief, cfg)
        assert "User wants to write a book" in question

    def test_answer_closes_pending_state(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)
        record_question_injection(db_session, "u1", "s1", belief, cfg)
        record_intervention_event(
            db_session, "u1", "s1",
            kind="question_answered", detail={"verdict": "confirm", "belief_key": belief.key},
        )
        assert has_unanswered_injection(db_session, "u1") is False


class TestPendingVerification:
    def test_no_injection_returns_none(self, db_session):
        assert get_pending_verification(db_session, "u1", "s1") is None

    def test_answered_injection_returns_none(self, db_session):
        record_intervention_event(db_session, "u1", "s1", kind="question_injected", detail={})
        record_intervention_event(db_session, "u1", "s1", kind="question_answered", detail={})
        assert get_pending_verification(db_session, "u1", "s1") is None

    def test_current_session_in_window(self, db_session):
        event = record_intervention_event(db_session, "u1", "sA", kind="question_injected", detail={})
        pending = get_pending_verification(db_session, "u1", "sA")
        assert pending is not None and pending.id == event.id

    def test_cross_session_within_window(self, db_session):
        event = record_intervention_event(db_session, "u1", "sA", kind="question_injected", detail={})
        pending = get_pending_verification(db_session, "u1", "sB", max_window_turns=3)
        assert pending is not None and pending.id == event.id

    def test_zero_window_rejects_other_session(self, db_session):
        record_intervention_event(db_session, "u1", "sA", kind="question_injected", detail={})
        assert get_pending_verification(db_session, "u1", "sB", max_window_turns=0) is None

    def test_injection_pushed_out_of_window(self, db_session):
        record_intervention_event(db_session, "u1", "sA", kind="question_injected", detail={})
        # Later activity in newer sessions pushes sA out of the 3-session window.
        record_intervention_event(db_session, "u1", "sB", kind="practice_note", detail={})
        record_intervention_event(db_session, "u1", "sC", kind="practice_note", detail={})
        record_intervention_event(db_session, "u1", "sD", kind="practice_note", detail={})
        assert get_pending_verification(db_session, "u1", "sE", max_window_turns=3) is None


class TestRunVerificationJudgment:
    def test_disabled_without_llm(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)
        record_question_injection(db_session, "u1", "s1", belief, cfg)
        assert run_verification_judgment(db_session, "u1", "s1", "yes", cfg, None) is False
        assert has_unanswered_injection(db_session, "u1") is True  # untouched

    def test_no_pending_returns_false(self, db_session, cfg):
        llm = ScriptedLLM("confirm")
        assert run_verification_judgment(db_session, "u1", "s1", "yes", cfg, llm) is False
        assert llm.calls == []

    def test_confirm_upgrades_to_l2(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)
        record_question_injection(db_session, "u1", "s1", belief, cfg)
        llm = ScriptedLLM("confirm")
        consumed = run_verification_judgment(db_session, "u1", "s1", "yeah that's right", cfg, llm)
        assert consumed is True
        db_session.refresh(belief)
        assert belief.layer == "L2"
        assert belief.source == "user_confirmed"
        assert belief.confidence >= 0.9
        # verification prompt template is consumed as the system prompt
        assert llm.calls[0]["system_prompt"] == cfg.prompts.verification
        # answered event closes the loop
        assert has_unanswered_injection(db_session, "u1") is False
        answered = (
            db_session.query(InterventionEvent)
            .filter(InterventionEvent.kind == "question_answered")
            .first()
        )
        assert answered is not None

    def test_deny_rejects_belief(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)
        record_question_injection(db_session, "u1", "s1", belief, cfg)
        consumed = run_verification_judgment(db_session, "u1", "s1", "no, that's not me", cfg, ScriptedLLM("deny"))
        assert consumed is True
        db_session.refresh(belief)
        assert belief.status == "rejected"
        assert belief.confidence == 0.0

    def test_unclear_keeps_belief_state(self, db_session, cfg):
        belief = _seed_hypothesis(db_session)
        record_question_injection(db_session, "u1", "s1", belief, cfg)
        consumed = run_verification_judgment(db_session, "u1", "s1", "nice weather today", cfg, ScriptedLLM("unclear"))
        assert consumed is True
        db_session.refresh(belief)
        assert belief.status == "active"
        assert belief.layer == "L4"
        assert has_unanswered_injection(db_session, "u1") is False

    def test_stale_injection_consumed(self, db_session, cfg):
        # Injection references a belief key that no longer exists.
        record_intervention_event(
            db_session, "u1", "s1",
            kind="question_injected", detail={"belief_key": "ghost.key", "belief_label": "ghost"},
        )
        consumed = run_verification_judgment(db_session, "u1", "s1", "hello", cfg, ScriptedLLM("confirm"))
        assert consumed is False
        assert has_unanswered_injection(db_session, "u1") is False

    def test_llm_failure_falls_back_to_unclear(self, db_session, cfg):
        class ExplodingLLM:
            model = "boom"

            def generate(self, **kwargs):
                raise RuntimeError("llm down")

        belief = _seed_hypothesis(db_session)
        record_question_injection(db_session, "u1", "s1", belief, cfg)
        consumed = run_verification_judgment(db_session, "u1", "s1", "yes", cfg, ExplodingLLM())
        assert consumed is True  # fail-open: judged unclear, turn still consumed
        db_session.refresh(belief)
        assert belief.status == "active"

    def test_empty_prompt_disables_loop(self, db_session, cfg):
        from mirror_memory.config.schema import MemoryConfig

        bare = MemoryConfig()
        belief = _seed_hypothesis(db_session)
        record_question_injection(db_session, "u1", "s1", belief, cfg)
        assert run_verification_judgment(db_session, "u1", "s1", "yes", bare, ScriptedLLM("confirm")) is False
        assert get_belief(db_session, "u1", belief.key).layer == "L4"


class TestPipelineVerificationIntegration:
    def test_judgment_consumes_turn(self, db_session, cfg):
        from mirror_memory.extraction.pipeline import ExtractionPipeline

        belief = _seed_hypothesis(db_session)
        record_question_injection(db_session, "u1", "s1", belief, cfg)
        pipeline = ExtractionPipeline(config=cfg, llm_client=ScriptedLLM("confirm"))
        claims = pipeline.observe(db_session, "u1", "s1", "yes exactly", 1)
        assert claims == []  # turn consumed by verification
        db_session.refresh(belief)
        assert belief.layer == "L2"

    def test_no_pending_extraction_proceeds(self, db_session, cfg):
        from mirror_memory.extraction.pipeline import ExtractionPipeline

        pipeline = ExtractionPipeline(config=cfg, llm_client=ScriptedLLM("confirm"))
        claims = pipeline.observe(db_session, "u1", "s1", "I want to learn Japanese", 1)
        assert isinstance(claims, list)  # normal extraction path
        assert len(claims) >= 1


class TestMemoryEngineVerification:
    @pytest.fixture
    def engine(self):
        return MemoryEngine(
            config_path="config/",
            database_url="sqlite://",
            llm_client=ScriptedLLM("confirm"),
        )

    def _seed(self, engine):
        engine._ensure_db()
        with engine._session() as session:
            _seed_hypothesis(session)
            session.commit()

    def test_candidates_record_injection(self, engine):
        self._seed(engine)
        candidates = engine.get_verification_candidates(user_id="u1", session_id="s3")
        assert len(candidates) == 1
        assert candidates[0]["key"] == "i_want"
        assert candidates[0]["injected"] is True
        assert "question" in candidates[0]

    def test_no_double_inject_across_calls(self, engine):
        self._seed(engine)
        first = engine.get_verification_candidates(user_id="u1", session_id="s3")
        second = engine.get_verification_candidates(user_id="u1", session_id="s3")
        assert first[0]["injected"] is True
        assert all(c["injected"] is False for c in second)

    def test_full_loop_confirm(self, engine):
        self._seed(engine)
        assert engine.get_verification_candidates(user_id="u1", session_id="s3")
        consumed = engine.run_verification(user_id="u1", session_id="s3", user_text="yes that's right")
        assert consumed is True
        beliefs = engine.get_beliefs(user_id="u1")
        assert len(beliefs) == 1
        assert beliefs[0].layer == "L2"
        assert beliefs[0].source == "user_confirmed"

    def test_run_verification_without_pending(self, engine):
        assert engine.run_verification(user_id="u1", session_id="s3", user_text="hello") is False

    def test_candidates_empty_without_llm(self):
        engine = MemoryEngine(config_path="config/", database_url="sqlite://")
        assert engine.get_verification_candidates(user_id="u1") == []

    def test_engine_disabled_loop(self, engine):
        self._seed(engine)
        engine.set_enabled(user_id="u1", enabled=False)
        assert engine.get_verification_candidates(user_id="u1") == []
        assert engine.run_verification(user_id="u1", session_id="s3", user_text="yes") is False

    def test_delete_cascades_intervention_events(self, engine):
        self._seed(engine)
        engine.get_verification_candidates(user_id="u1", session_id="s3")
        result = engine.delete_memories(user_id="u1")
        assert result.intervention_events >= 1
