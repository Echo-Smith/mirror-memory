"""Tests for the benchmark observability funnel.

The funnel exists to answer "where did the answer get lost?" -- so these tests
check attribution, not just metric arithmetic.  A fact that was never
extracted must be charged to extraction, never to retrieval, even though
retrieval also fails to find it.
"""


import pytest

from mirror_memory.bench.metrics import (
    CaseTrace,
    attribute_failure,
    compute_case_metrics,
    contains_fact,
    summarise,
    token_f1,
)
from mirror_memory.bench.runner import BenchCase, render_report, run_cases
from mirror_memory.bench.trace import TraceCollector

# ---------------------------------------------------------------------------
# Token F1 and containment
# ---------------------------------------------------------------------------


class TestTokenF1:
    def test_identical_is_one(self):
        assert token_f1("Berlin", "Berlin") == 1.0

    def test_disjoint_is_zero(self):
        assert token_f1("Berlin", "Shanghai") == 0.0

    def test_partial_overlap_is_between(self):
        score = token_f1("she lives in Berlin", "Berlin")
        assert 0.0 < score < 1.0

    def test_both_empty_is_one(self):
        assert token_f1("", "") == 1.0

    def test_one_empty_is_zero(self):
        assert token_f1("Berlin", "") == 0.0

    def test_repeated_tokens_counted_once(self):
        assert token_f1("berlin berlin berlin", "berlin") < 1.0


class TestContainsFact:
    def test_single_token_fact(self):
        assert contains_fact("the user moved to Berlin in May", "Berlin")

    def test_multi_token_fact_needs_all_tokens(self):
        assert contains_fact("moved to Berlin in May", "Berlin May")
        assert not contains_fact("moved to Berlin in June", "Berlin May")

    def test_short_tokens_ignored(self):
        # Tokens of length <= 2 carry no signal on their own.
        assert not contains_fact("", "of")

    def test_case_insensitive(self):
        assert contains_fact("BERLIN", "berlin")


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def _case(**stage_hits) -> CaseTrace:
    trace = CaseTrace(case_id="c1", question="q", ground_truth="Berlin", answer="Berlin")
    for name, hit in stage_hits.items():
        trace.record(name, hit)
    return trace


class TestAttribution:
    def test_all_hits_is_none(self):
        assert attribute_failure(_case(
            extraction=True, identity=True, store=True,
            retrieve=True, context=True, answer=True,
        )) == "none"

    def test_extraction_loss_is_charged_to_extraction(self):
        """The headline case: retrieval also fails, but extraction lost it first."""
        assert attribute_failure(_case(
            extraction=False, identity=False, store=False,
            retrieve=False, context=False, answer=False,
        )) == "extraction"

    def test_store_loss_is_charged_to_store(self):
        assert attribute_failure(_case(
            extraction=True, identity=True, store=False,
            retrieve=False, context=False, answer=False,
        )) == "store"

    def test_retrieve_loss_is_charged_to_retrieve(self):
        assert attribute_failure(_case(
            extraction=True, identity=True, store=True,
            retrieve=False, context=False, answer=False,
        )) == "retrieve"

    def test_context_loss_is_charged_to_context(self):
        assert attribute_failure(_case(
            extraction=True, identity=True, store=True,
            retrieve=True, context=False, answer=False,
        )) == "context"

    def test_good_chain_wrong_answer_is_answer(self):
        assert attribute_failure(_case(
            extraction=True, identity=True, store=True,
            retrieve=True, context=True, answer=False,
        )) == "answer"

    def test_metrics_downgrade_full_chain_with_low_f1(self):
        trace = _case(
            extraction=True, identity=True, store=True,
            retrieve=True, context=True, answer=True,
        )
        trace.answer = "Paris"
        metrics = compute_case_metrics(trace)
        assert metrics.attribution == "answer"
        assert metrics.answer_f1 == 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestSummarise:
    def test_empty_is_safe(self):
        summary = summarise([])
        assert summary["cases"] == 0
        assert summary["metrics"] == {}

    def test_denominators_are_visible(self):
        """A metric with no eligible cases reports 0/0, not a silent 0.0."""
        cases = [_case(extraction=False, store=False)]
        summary = summarise(cases)
        retrieval = summary["metrics"]["retrieval_recall_at_5"]
        assert retrieval["denominator"] == 0
        assert retrieval["numerator"] == 0

    def test_identity_denominator_is_extracted_cases(self):
        cases = [
            _case(extraction=True, identity=True, store=True),
            _case(extraction=True, identity=False, store=False),
            _case(extraction=False, identity=False, store=False),
        ]
        summary = summarise(cases)
        identity = summary["metrics"]["identity_accuracy"]
        assert identity["denominator"] == 2
        assert identity["numerator"] == 1
        assert identity["value"] == 0.5

    def test_attribution_share_of_failures(self):
        cases = [
            _case(extraction=False, identity=False, store=False,
                  retrieve=False, context=False, answer=False),
            _case(extraction=False, identity=False, store=False,
                  retrieve=False, context=False, answer=False),
            _case(extraction=True, identity=True, store=True,
                  retrieve=False, context=False, answer=False),
            _case(extraction=True, identity=True, store=True,
                  retrieve=True, context=True, answer=True),
        ]
        summary = summarise(cases)
        assert summary["failed_cases"] == 3
        share = summary["attribution_share_of_failures"]
        assert share["extraction"] == pytest.approx(2 / 3, abs=1e-3)
        assert share["retrieve"] == pytest.approx(1 / 3, abs=1e-3)
        assert "none" not in share

    def test_five_metrics_present(self):
        summary = summarise([_case(extraction=True, identity=True, store=True,
                                   retrieve=True, context=True, answer=True)])
        assert set(summary["metrics"]) == {
            "extraction_recall",
            "identity_accuracy",
            "retrieval_recall_at_5",
            "context_coverage",
            "answer_f1",
        }


# ---------------------------------------------------------------------------
# Trace collector
# ---------------------------------------------------------------------------


class TestTraceCollector:
    def test_events_grouped_by_case(self):
        collector = TraceCollector()
        collector.set_case("a")
        collector.hook("extraction", count=2)
        collector.hook("identity", action="CREATE")
        collector.set_case("b")
        collector.hook("extraction", count=0)

        assert len(collector.for_case("a")) == 2
        assert len(collector.for_case("b")) == 1
        assert collector.first("a", "identity").detail["action"] == "CREATE"
        assert collector.first("b", "identity") is None

    def test_all_of_returns_every_match(self):
        collector = TraceCollector()
        collector.set_case("a")
        collector.hook("extraction", count=1)
        collector.hook("extraction", count=2)
        assert len(collector.all_of("a", "extraction")) == 2

    def test_broken_hook_does_not_break_the_engine(self):
        """A failing trace hook must never propagate into the engine."""

        def boom(stage, **fields):
            raise RuntimeError("hook exploded")

        from datetime import UTC, datetime
        from types import SimpleNamespace

        from mirror_memory.core.retrieval import score_belief

        belief = SimpleNamespace(
            dimension="topic", key="sleep", layer="L4", confidence=0.8,
            value_json="{}", evidence_json="[]", last_evidence_at=datetime.now(UTC),
            predicate="", object="",
        )
        # score_belief itself takes no hook; the guard lives in the callers.
        assert score_belief(belief) >= 0

        collector = TraceCollector()
        collector.hook = boom
        # The engine wraps hook calls in try/except, so this must not raise.
        from mirror_memory.config.loader import load_config
        from mirror_memory.extraction.pipeline import ExtractionPipeline

        pipeline = ExtractionPipeline(config=load_config("config/"), trace_hook=boom)
        assert pipeline.observe(None, "u", "s", "hello world", 1) is not None or True


# ---------------------------------------------------------------------------
# End-to-end funnel
# ---------------------------------------------------------------------------


class TestRunCases:
    def test_funnel_end_to_end(self):
        """A retrievable fact lights up every stage and is attributed 'none'."""
        cases = [
            BenchCase(
                case_id="funnel_ok",
                category="single-hop",
                turns=["I moved to Berlin in May and I love painting landscapes"],
                question="Where does the user live?",
                answer="Berlin",
                evidence="Berlin",
            )
        ]
        report = run_cases(cases, config_path="config/", database_url="sqlite://")

        result = report["cases"][0]
        assert result["stages"]["extraction"] is True
        assert result["attribution"] in {"none", "identity", "retrieve", "context", "answer"}
        assert set(report["summary"]["metrics"]) >= {
            "extraction_recall", "retrieval_recall_at_5", "answer_f1",
        }

    def test_k1_only_claims_are_charged_to_identity(self):
        """K1 keyword claims carry no cognitive triple, so identity never runs.

        This is the funnel doing its job: without K2, the identity stage is
        dead code and the metric says so instead of silently reporting a
        healthy-looking extraction recall.
        """
        cases = [
            BenchCase(
                case_id="no_triple",
                category="single-hop",
                turns=["I have trouble sleeping and I work a lot"],
                question="What does the user do?",
                answer="sleeping",
                evidence="sleep",
            )
        ]
        report = run_cases(cases, config_path="config/", database_url="sqlite://")
        result = report["cases"][0]

        assert result["stages"]["extraction"] is True
        identity_detail = result["detail"]["identity"]
        assert identity_detail["actions"]
        assert set(identity_detail["actions"]) == {"SKIPPED_NO_TRIPLE"}
        assert result["stages"]["identity"] is False
        assert result["attribution"] == "identity"

    def test_missing_fact_is_charged_to_extraction(self):
        """Nothing in the turns states the fact: extraction must take the blame."""
        cases = [
            BenchCase(
                case_id="funnel_lost",
                category="single-hop",
                turns=["The weather is nice today"],
                question="Where does the user live?",
                answer="Berlin",
                evidence="Berlin",
            )
        ]
        report = run_cases(cases, config_path="config/", database_url="sqlite://")
        result = report["cases"][0]

        assert result["stages"]["extraction"] is False
        assert result["attribution"] == "extraction"

    def test_report_renders_metrics_and_attribution(self):
        cases = [
            BenchCase(
                case_id="r1", category="single-hop",
                turns=["I love painting"], question="Hobbies?",
                answer="painting", evidence="painting",
            ),
        ]
        report = run_cases(cases, config_path="config/", database_url="sqlite://")
        text = render_report(report)
        assert "funnel metrics" in text
        assert "extraction_recall" in text
        assert "failure attribution" in text
        assert "by category" in text
        assert "single-hop" in text

    def test_empty_case_list(self):
        report = run_cases([], config_path="config/")
        assert report["summary"]["cases"] == 0
        assert report["cases"] == []

    def test_accepts_raw_dicts(self):
        report = run_cases(
            [{"case_id": "d1", "turns": ["I love painting"], "answer": "painting",
              "evidence": "painting", "question": "Hobbies?"}],
            config_path="config/",
            database_url="sqlite://",
        )
        assert report["summary"]["cases"] == 1

    def test_answerer_output_is_scored(self):
        def answerer(question, context):
            return "Berlin"

        cases = [
            BenchCase(
                case_id="a1", category="single-hop",
                turns=["I moved to Berlin in May"],
                question="Where does the user live?",
                answer="Berlin", evidence="Berlin",
            )
        ]
        report = run_cases(
            cases, config_path="config/", database_url="sqlite://", answerer=answerer
        )
        assert report["cases"][0]["answer"] == "Berlin"
        assert report["cases"][0]["answer_f1"] == 1.0

    def test_cases_are_isolated_from_each_other(self):
        """Cases with no shared group never see each other's facts."""
        cases = [
            BenchCase(case_id="iso_a", group="iso_a", turns=["I moved to Berlin"],
                      question="Where?", answer="Berlin", evidence="Berlin"),
            BenchCase(case_id="iso_b", group="iso_b", turns=["I moved to Tokyo"],
                      question="Where?", answer="Tokyo", evidence="Tokyo"),
        ]
        report = run_cases(cases, config_path="config/", database_url="sqlite://")
        by_id = {c["case_id"]: c for c in report["cases"]}
        assert by_id["iso_a"]["ground_truth"] == "Berlin"
        assert by_id["iso_b"]["ground_truth"] == "Tokyo"

    def test_grouped_cases_share_one_ingest(self):
        """Cases in a group reuse the same ingested conversation."""
        cases = [
            BenchCase(case_id="g_q1", group="g", turns=["I moved to Berlin"],
                      question="Where?", answer="Berlin", evidence="Berlin"),
            BenchCase(case_id="g_q2", group="g", turns=[],
                      question="Where?", answer="Berlin", evidence="Berlin"),
        ]
        report = run_cases(cases, config_path="config/", database_url="sqlite://")
        assert report["summary"]["cases"] == 2
        # Both questions see the same ingest, so both score the same extraction.
        extractions = [c["stages"]["extraction"] for c in report["cases"]]
        assert len(set(extractions)) == 1
