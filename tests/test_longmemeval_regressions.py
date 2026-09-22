"""Regression tests for the two LongMemEval ingestion failures.

1. **Key collision** — 19 LongMemEval users were lost to
   ``UNIQUE constraint failed: mm_beliefs.user_id, mm_beliefs.key``.  The
   extractor produced the same key for two different values of a
   single-cardinality predicate; ``update_belief_by_id`` then tried to insert
   a row under a key the row being superseded still held, and the constraint
   -- which ignores ``status`` -- rejected it.

2. **Throttle blindness** — ``_belief_dimensions`` split keys on ``"."`` while
   K1 namespaces them with ``":"``, so a stored belief never mapped back to
   its dimension.  ``compute_extraction_value`` returned the same score with
   and without beliefs, the suppression gate never fired, and every signal
   turn triggered an LLM call.  That is the >80% ingestion runtime.

3. **CLI wiring** — ``_build_extractor`` was called with (model, key) in the
   (key, model) slots, so the extractor authenticated with the model name.
   Every call 401'd, the fallback returned empty claims, and the run looked
   like "K2 found nothing" instead of "K2 is misconfigured".
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import Base, Belief
from mirror_memory.core.repository import (
    record_claim,
    set_memory_enabled,
    update_belief_by_id,
)
from mirror_memory.extraction.throttle import (
    _belief_dimensions,
    _hit_dimensions,
    compute_extraction_value,
    should_extract,
)


@pytest.fixture
def config():
    return load_config("config/")


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory()
    Base.metadata.drop_all(engine)
    engine.dispose()


# ---------------------------------------------------------------------------
# 1. Key collision
# ---------------------------------------------------------------------------


class TestKeyCollision:
    def test_update_with_same_key_and_new_object_does_not_collide(self, session):
        """The exact LongMemEval failure: same key, different value."""
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact",
            key="works_as_digital_marketing_specialist",
            claim_text="works as a digital marketing specialist",
            confidence=0.6, predicate="works_as",
            object="digital_marketing_specialist", session_id="s1",
        )

        old, new = update_belief_by_id(
            session, belief.id,
            new_predicate="works_as", new_object="data_analyst",
            # The extractor reused the key -- this is what crashed.
            new_key="works_as_digital_marketing_specialist",
            new_claim_text="works as a data analyst",
            new_confidence=0.6, session_id="s2",
        )

        assert old.status == "superseded"
        assert new.status == "active"
        assert new.key != old.key, "a value change must move to a new key"
        assert new.object == "data_analyst"
        assert session.query(Belief).count() == 2

    def test_repeated_updates_never_collide(self, session):
        """A long conversation changes the same predicate many times."""
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="job",
            claim_text="works as a designer", confidence=0.6,
            predicate="works_as", object="designer", session_id="s1",
        )
        current = belief
        for i, obj in enumerate(("analyst", "manager", "director", "consultant")):
            old, new = update_belief_by_id(
                session, current.id,
                new_predicate="works_as", new_object=obj,
                new_key=current.key,  # the extractor keeps reusing it
                new_claim_text=f"works as a {obj}",
                new_confidence=0.6, session_id=f"s{i + 2}",
            )
            assert new is not None, f"update to {obj} failed"
            assert new.key != old.key
            current = new
        assert session.query(Belief).filter_by(status="active").count() == 1

    def test_disambiguation_suffixes_when_derived_key_is_taken(self, session):
        """Even a derived key can be taken; the suffix keeps the insert legal."""
        set_memory_enabled(session, "u1", True)
        first, _ = record_claim(
            session, "u1", dimension="fact", key="job:analyst",
            claim_text="works as an analyst", confidence=0.6,
            predicate="works_as", object="analyst", session_id="s1",
        )
        second, _ = record_claim(
            session, "u1", dimension="fact", key="job",
            claim_text="works as a designer", confidence=0.6,
            predicate="works_as", object="designer", session_id="s1",
        )
        # Both derive toward "job:analyst"; the second must not collide.
        old, new = update_belief_by_id(
            session, second.id,
            new_predicate="works_as", new_object="analyst",
            new_claim_text="back to analyst", new_confidence=0.6, session_id="s2",
        )
        assert new is not None
        assert new.key != first.key or new.id != first.id
        assert session.query(Belief).filter_by(key=new.key).count() == 1

    def test_revival_path_still_works_after_the_fix(self, session):
        """Shanghai -> Berlin -> Shanghai must revive, not accumulate rows."""
        set_memory_enabled(session, "u1", True)
        shanghai, _ = record_claim(
            session, "u1", dimension="fact", key="lives_in:shanghai",
            claim_text="lives in Shanghai", confidence=0.8,
            predicate="lives_in", object="shanghai", session_id="s1",
        )
        _old, beijing = update_belief_by_id(
            session, shanghai.id,
            new_predicate="lives_in", new_object="beijing",
            new_claim_text="lives in Berlin", new_confidence=0.8, session_id="s2",
        )
        _old2, revived = update_belief_by_id(
            session, beijing.id,
            new_predicate="lives_in", new_object="shanghai",
            new_claim_text="back in Shanghai", new_confidence=0.8, session_id="s3",
        )
        assert revived.id == shanghai.id
        assert revived.status == "active"
        assert session.query(Belief).filter_by(status="active").count() == 1

    def test_explicit_distinct_key_is_respected(self, session):
        """A caller-supplied key that is genuinely free is used as given."""
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="job_a",
            claim_text="works as a designer", confidence=0.6,
            predicate="works_as", object="designer", session_id="s1",
        )
        _old, new = update_belief_by_id(
            session, belief.id,
            new_predicate="works_as", new_object="analyst",
            new_key="job_b",
            new_claim_text="works as an analyst", new_confidence=0.6, session_id="s2",
        )
        assert new.key == "job_b"

    def test_superseded_row_keeps_its_own_key(self, session):
        """History stays addressable: the closed row keeps the key it had."""
        set_memory_enabled(session, "u1", True)
        belief, _ = record_claim(
            session, "u1", dimension="fact", key="works_as_designer",
            claim_text="works as a designer", confidence=0.6,
            predicate="works_as", object="designer", session_id="s1",
        )
        old, new = update_belief_by_id(
            session, belief.id,
            new_predicate="works_as", new_object="analyst",
            new_key="works_as_designer",
            new_claim_text="works as an analyst", new_confidence=0.6, session_id="s2",
        )
        assert old.key == "works_as_designer"
        assert new.key != "works_as_designer"


# ---------------------------------------------------------------------------
# 2. Dimension coverage / throttle blindness
# ---------------------------------------------------------------------------


class TestBeliefDimensionMapping:
    @pytest.mark.parametrize("key,dimension", [
        ("topic:sleep", "topic"),      # K1 keyword namespace
        ("age:30", "fact"),            # K1 keyword namespace, dotless dimension
        ("always:abc123", "pattern"),  # K1 pattern namespace
        ("goal", "goal"),              # dimension id used directly
        ("goal.i_want", "goal"),       # config-style dot namespace
        ("topic", "topic"),
    ])
    def test_key_maps_to_its_dimension(self, config, key, dimension):
        assert dimension in _belief_dimensions({key}, config)

    def test_unknown_key_covers_nothing(self, config):
        """A key we cannot place must not claim coverage it does not have."""
        assert _belief_dimensions({"lives_in_shanghai"}, config) == set()

    def test_empty_covers_nothing(self, config):
        assert _belief_dimensions(set(), config) == set()
        assert _belief_dimensions(None, config) == set()

    def test_colon_namespace_is_the_regression(self, config):
        """The bug: splitting on "." alone never recovered the dimension."""
        # topic:sleep split on "." yields ["topic:sleep"] -- not "topic".
        assert "topic:sleep".split(".", 1)[0] == "topic:sleep"
        assert "topic" in _belief_dimensions({"topic:sleep"}, config)


class TestValueRespondsToBeliefs:
    def test_score_drops_once_the_dimension_is_covered(self, config):
        text = "I have trouble sleeping"
        cold = compute_extraction_value(text=text, config=config, active_belief_keys=set())
        warm = compute_extraction_value(
            text=text, config=config, active_belief_keys={"topic:sleep"}
        )
        assert warm < cold, "an existing belief must lower the extraction value"

    def test_score_is_identical_with_and_without_belief_is_the_bug(self, config):
        """Pins the regression: the score used to ignore beliefs entirely."""
        text = "I have trouble sleeping"
        cold = compute_extraction_value(text=text, config=config, active_belief_keys=set())
        warm = compute_extraction_value(
            text=text, config=config, active_belief_keys={"topic:sleep"}
        )
        assert cold != warm

    def test_fresh_user_scores_full(self, config):
        value = compute_extraction_value(
            text="I want to learn Japanese", config=config, active_belief_keys=set()
        )
        assert value == pytest.approx(0.45 + 0.25 + 0.30)

    def test_no_keyword_signal_is_zero_regardless_of_state(self, config):
        assert compute_extraction_value(
            text="xyzzy plugh", config=config, active_belief_keys=set()
        ) == 0.0
        assert compute_extraction_value(
            text="xyzzy plugh", config=config, active_belief_keys={"topic:sleep"}
        ) == 0.0


class TestThrottleActuallyThrottles:
    def test_mature_coverage_suppresses_repeated_dimension(self, session, config):
        config.extraction.llm_min_keyword_hits = 2
        record_claim(
            session, "u1", dimension="goal", key="goal",
            claim_text="I want to learn Japanese", confidence=0.9, session_id="s1",
        )
        assert should_extract(
            "I want to learn Japanese", 1, config, session=session, user_id="u1"
        ) is False

    def test_cold_profile_extracts(self, session, config):
        config.extraction.llm_min_keyword_hits = 2
        assert should_extract(
            "I want to learn Japanese", 1, config, session=session, user_id="u1"
        ) is True

    def test_no_keyword_hit_never_calls(self, session, config):
        config.extraction.llm_min_keyword_hits = 2
        assert should_extract(
            "xyzzy plugh", 1, config, session=session, user_id="u1"
        ) is False

    def test_benchmark_escape_hatch_is_untouched(self, session, config):
        """min_hits=0 means "always extract" and must stay that way.

        LongMemEval ran with this setting; the coverage fix must not silently
        start throttling a configuration that asked for every turn.
        """
        config.extraction.llm_min_keyword_hits = 0
        record_claim(
            session, "u1", dimension="goal", key="goal",
            claim_text="I want to learn Japanese", confidence=0.9, session_id="s1",
        )
        assert should_extract(
            "I want to learn Japanese", 1, config, session=session, user_id="u1"
        ) is True
        assert should_extract(
            "I want to learn Japanese", 99, config, session=session, user_id="u1"
        ) is True

    def test_suppression_beats_early_trigger(self, session, config):
        config.extraction.llm_min_keyword_hits = 2
        record_claim(
            session, "u1", dimension="goal", key="goal",
            claim_text="I want to learn Japanese", confidence=0.9, session_id="s1",
        )
        value = compute_extraction_value(
            text="I want to learn Japanese", config=config, active_belief_keys={"goal"}
        )
        assert value >= config.extraction.extraction_value_threshold
        assert should_extract(
            "I want to learn Japanese", 1, config, session=session, user_id="u1"
        ) is False

    def test_low_confidence_belief_does_not_suppress(self, session, config):
        config.extraction.llm_min_keyword_hits = 2
        record_claim(
            session, "u1", dimension="goal", key="goal",
            claim_text="user goals", confidence=0.3, session_id="s1",
        )
        assert should_extract(
            "I want to learn Japanese", 1, config, session=session, user_id="u1"
        ) is True

    def test_hit_dimensions_unchanged(self, config):
        assert _hit_dimensions("I have trouble sleeping and I work a lot", config) == {
            "topic", "fact",
        }


# ---------------------------------------------------------------------------
# 3. CLI wiring: the extractor must get the key, not the model name
# ---------------------------------------------------------------------------


class TestBenchCliWiring:
    def test_extractor_receives_key_and_model_in_the_right_slots(self):
        """A swapped pair authenticates with the model name and 401s.

        The fallback turns that into "no claims", which is indistinguishable
        from a turn with nothing to extract -- so the run silently reports a
        healthy-looking extraction recall with K2 contributing nothing.
        """
        from mirror_memory.bench.run import _build_extractor

        client = _build_extractor(
            "mimo-v2.5", "sk-test-key", "https://api.xiaomimimo.com/v1",
        )
        assert client._model == "mimo-v2.5"
        assert client._client.api_key == "sk-test-key"

    def test_answerer_and_extractor_agree_on_argument_order(self):
        """Both builders must take the model in the same position."""
        import inspect

        from mirror_memory.bench.run import _build_answerer, _build_extractor

        answerer_params = list(inspect.signature(_build_answerer).parameters)
        extractor_params = list(inspect.signature(_build_extractor).parameters)
        assert answerer_params[:3] == extractor_params[:3], (
            "the two builders disagree on argument order -- they are called "
            "interchangeably and a swap is invisible until runtime"
        )

    def test_no_thinking_passes_thinking_disabled(self):
        from mirror_memory.bench.run import _build_extractor

        client = _build_extractor(
            "mimo-v2.5", "sk-k", None, {"thinking": {"type": "disabled"}}
        )
        assert client._extra_body == {"thinking": {"type": "disabled"}}

    def test_extra_body_defaults_to_none(self):
        from mirror_memory.bench.run import _build_extractor

        client = _build_extractor("mimo-v2.5", "sk-k", None)
        assert client._extra_body is None

    def test_cli_builds_extractor_config_with_the_key(self, monkeypatch):
        """End to end through main(): the config handed to run_cases carries a
        client whose api_key is the supplied key, not the model name."""
        import mirror_memory.bench.run as run_module

        captured = {}

        def fake_run_cases(cases, **kwargs):
            captured["config"] = kwargs.get("config")
            return {"summary": {"cases": 0, "metrics": {}}, "by_category": {},
                    "cases": []}

        monkeypatch.setattr(run_module, "run_cases", fake_run_cases)
        # load_locomo_cases is imported into run.py's namespace; patch it there.
        # It must return something, or main() bails before building the config.
        from mirror_memory.bench.runner import BenchCase

        monkeypatch.setattr(
            run_module, "load_locomo_cases",
            lambda *a, **k: [BenchCase(case_id="c1", turns=["hi"], question="q",
                                      answer="a", evidence="a")],
        )
        run_module.main([
            "--dataset", "unused.json",
            "--api-key", "sk-from-cli",
            "--answerer-model", "mimo-v2.5",
            "--base-url", "https://api.xiaomimimo.com/v1",
            "--no-thinking",
        ])

        config = captured["config"]
        assert config is not None
        assert config.llm_client._client.api_key == "sk-from-cli"
        assert config.llm_client._model == "mimo-v2.5"
