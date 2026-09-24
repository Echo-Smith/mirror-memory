"""Tests for Evidence as a first-class entity.

Before this, a belief carried a bare list of message ids in ``evidence_json``:
enough to say *which* messages, nothing about *what kind* of source they were
or how the belief was obtained.  These tests pin the new graph -- typed links,
idempotent evidence, and orphan pruning on forget.
"""


import pytest

from mirror_memory.core.models import Belief, BeliefEvidenceLink, Evidence
from mirror_memory.core.repository import (
    evidence_counts_for_beliefs,
    evidence_for_belief,
    evidence_summary,
    forget_belief,
    link_evidence,
    record_claim,
    record_evidence,
    set_memory_enabled,
)
from mirror_memory.core.utils import belief_evidence_ids


def _belief(session, key="sleep", **kwargs):
    belief, _ = record_claim(
        session, "u1", dimension="topic", key=key,
        claim_text="trouble sleeping", confidence=0.8,
        session_id="s1", evidence_message_ids=[1], **kwargs,
    )
    return belief


# ---------------------------------------------------------------------------
# Evidence entity
# ---------------------------------------------------------------------------


class TestRecordEvidence:
    def test_creates_row_with_provenance(self, db_session):
        ev = record_evidence(
            db_session, "u1", ref="m1", content="I have trouble sleeping",
            session_id="s1", source_type="message",
            extraction_method="k1_keyword", authority="user",
        )
        assert ev.id is not None
        assert ev.ref == "m1"
        assert ev.source_type == "message"
        assert ev.extraction_method == "k1_keyword"
        assert ev.authority == "user"
        assert ev.session_id == "s1"
        assert ev.observed_at is not None

    def test_idempotent_on_user_and_ref(self, db_session):
        first = record_evidence(db_session, "u1", ref="m1", content="short")
        second = record_evidence(db_session, "u1", ref="m1", content="a longer message")
        assert second.id == first.id
        assert db_session.query(Evidence).count() == 1
        # The longer content wins -- provenance should not lose detail.
        assert second.content == "a longer message"

    def test_same_ref_different_user_is_distinct(self, db_session):
        a = record_evidence(db_session, "u1", ref="m1")
        b = record_evidence(db_session, "u2", ref="m1")
        assert a.id != b.id
        assert db_session.query(Evidence).count() == 2

    def test_refresh_updates_provenance_not_identity(self, db_session):
        ev = record_evidence(
            db_session, "u1", ref="m1", extraction_method="k1_keyword", authority="user",
        )
        record_evidence(
            db_session, "u1", ref="m1",
            extraction_method="k2_llm", authority="assistant",
        )
        db_session.refresh(ev)
        assert ev.extraction_method == "k2_llm"
        assert ev.authority == "assistant"
        assert ev.ref == "m1"  # identity unchanged


# ---------------------------------------------------------------------------
# Typed links
# ---------------------------------------------------------------------------


class TestLinkEvidence:
    def test_support_link(self, db_session):
        belief = _belief(db_session)
        ev = record_evidence(db_session, "u1", ref="m1")
        link = link_evidence(db_session, belief.id, ev.id, relation="support")
        assert link.relation == "support"
        assert link.belief_id == belief.id

    def test_same_evidence_can_support_and_contradict(self, db_session):
        """The whole point of typed links: one message, two opposite edges."""
        belief = _belief(db_session, key="sleep")
        other = _belief(db_session, key="rest")
        ev = record_evidence(db_session, "u1", ref="m1", content="I sleep badly")

        link_evidence(db_session, belief.id, ev.id, relation="support")
        link_evidence(db_session, other.id, ev.id, relation="contradict")

        assert [r for _e, r in evidence_for_belief(db_session, belief.id)] == ["support"]
        assert [r for _e, r in evidence_for_belief(db_session, other.id)] == ["contradict"]

    def test_idempotent_on_belief_evidence_relation(self, db_session):
        belief = _belief(db_session)
        ev = record_evidence(db_session, "u1", ref="m1")
        link_evidence(db_session, belief.id, ev.id, relation="support")
        link_evidence(db_session, belief.id, ev.id, relation="support")
        assert db_session.query(BeliefEvidenceLink).count() == 1

    def test_invalid_relation_rejected(self, db_session):
        belief = _belief(db_session)
        ev = record_evidence(db_session, "u1", ref="m1")
        with pytest.raises(ValueError, match="invalid evidence relation"):
            link_evidence(db_session, belief.id, ev.id, relation="endorses")

    def test_all_four_relations_accepted(self, db_session):
        belief = _belief(db_session)
        for i, relation in enumerate(("support", "contradict", "verify", "correct")):
            ev = record_evidence(db_session, "u1", ref=f"m{i}")
            link_evidence(db_session, belief.id, ev.id, relation=relation)
        assert db_session.query(BeliefEvidenceLink).count() == 4

    def test_filter_by_relation(self, db_session):
        belief = _belief(db_session)
        for i, relation in enumerate(("support", "contradict", "verify")):
            ev = record_evidence(db_session, "u1", ref=f"m{i}")
            link_evidence(db_session, belief.id, ev.id, relation=relation)
        supporting = evidence_for_belief(db_session, belief.id, relations=("support",))
        assert len(supporting) == 1
        assert supporting[0][1] == "support"

    def test_counts_typed_edges(self, db_session):
        belief = _belief(db_session)
        for i, relation in enumerate(("support", "contradict")):
            ev = record_evidence(db_session, "u1", ref=f"m{i}")
            link_evidence(db_session, belief.id, ev.id, relation=relation)
        assert evidence_counts_for_beliefs(db_session, [belief.id]) == {belief.id: 2}

    def test_counts_empty_list(self, db_session):
        assert evidence_counts_for_beliefs(db_session, []) == {}


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


class TestEvidenceSummary:
    def test_groups_by_relation_with_provenance(self, db_session):
        belief = _belief(db_session)
        ev = record_evidence(
            db_session, "u1", ref="m1", content="I sleep badly",
            extraction_method="k1_keyword", authority="user", session_id="s1",
        )
        link_evidence(db_session, belief.id, ev.id, relation="support")

        summary = evidence_summary(db_session, "u1", belief.id)
        assert summary["counts"] == {"support": 1}
        item = summary["by_relation"]["support"][0]
        assert item["ref"] == "m1"
        assert item["extraction_method"] == "k1_keyword"
        assert item["authority"] == "user"
        assert item["session_id"] == "s1"

    def test_empty_graph(self, db_session):
        belief = _belief(db_session)
        summary = evidence_summary(db_session, "u1", belief.id)
        assert summary["counts"] == {}
        assert summary["by_relation"] == {}


# ---------------------------------------------------------------------------
# Lifecycle: forget prunes orphans
# ---------------------------------------------------------------------------


class TestEvidenceLifecycle:
    def test_forget_belief_drops_its_links(self, db_session):
        belief = _belief(db_session)
        ev = record_evidence(db_session, "u1", ref="m1")
        link_evidence(db_session, belief.id, ev.id, relation="support")

        assert forget_belief(db_session, "u1", belief.id) is True
        assert db_session.query(BeliefEvidenceLink).count() == 0

    def test_shared_evidence_survives_forget(self, db_session):
        """Evidence backing another belief is provenance, not garbage."""
        a = _belief(db_session, key="sleep")
        b = _belief(db_session, key="rest")
        ev = record_evidence(db_session, "u1", ref="m1", content="shared message")
        link_evidence(db_session, a.id, ev.id, relation="support")
        link_evidence(db_session, b.id, ev.id, relation="support")

        forget_belief(db_session, "u1", a.id)

        assert db_session.query(BeliefEvidenceLink).count() == 1
        assert db_session.query(Evidence).count() == 1
        assert evidence_for_belief(db_session, b.id)

    def test_orphan_evidence_is_pruned(self, db_session):
        belief = _belief(db_session)
        ev = record_evidence(db_session, "u1", ref="m1")
        link_evidence(db_session, belief.id, ev.id, relation="support")
        forget_belief(db_session, "u1", belief.id)
        assert db_session.query(Evidence).count() == 0

    def test_evidence_for_other_user_untouched(self, db_session):
        belief = _belief(db_session)
        record_evidence(db_session, "u2", ref="m9")
        forget_belief(db_session, "u1", belief.id)
        assert db_session.query(Evidence).filter_by(user_id="u2").count() == 1


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineCreatesEvidence:
    def test_observe_records_evidence_and_links(self, db_session):
        from mirror_memory.config.loader import load_config
        from mirror_memory.extraction.pipeline import ExtractionPipeline

        set_memory_enabled(db_session, "u1", True)
        pipeline = ExtractionPipeline(config=load_config("config/"))
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping lately", 1)

        evidence = db_session.query(Evidence).all()
        assert evidence, "observe produced no Evidence rows"
        assert all(e.user_id == "u1" for e in evidence)
        assert all(e.session_id == "s1" for e in evidence)
        assert all(e.authority == "user" for e in evidence)
        assert all(e.extraction_method for e in evidence)
        assert all(e.content for e in evidence)

        links = db_session.query(BeliefEvidenceLink).all()
        assert links
        assert all(link.relation == "support" for link in links)

    def test_each_turn_is_its_own_evidence(self, db_session):
        from mirror_memory.config.loader import load_config
        from mirror_memory.extraction.pipeline import ExtractionPipeline

        set_memory_enabled(db_session, "u1", True)
        pipeline = ExtractionPipeline(config=load_config("config/"))
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping lately", 1)
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping again", 2)

        refs = {e.ref for e in db_session.query(Evidence).all()}
        assert refs == {"s1:1", "s1:2"}

    def test_repeating_a_turn_reuses_the_evidence_row(self, db_session):
        from mirror_memory.config.loader import load_config
        from mirror_memory.extraction.pipeline import ExtractionPipeline

        set_memory_enabled(db_session, "u1", True)
        pipeline = ExtractionPipeline(config=load_config("config/"))
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping lately", 1)
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping again", 1)

        assert db_session.query(Evidence).count() == 1

    def test_support_accumulates_typed_links(self, db_session):
        from mirror_memory.config.loader import load_config
        from mirror_memory.extraction.pipeline import ExtractionPipeline

        set_memory_enabled(db_session, "u1", True)
        pipeline = ExtractionPipeline(config=load_config("config/"))
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping lately", 1)
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping again", 2)

        belief = db_session.query(Belief).first()
        counts = evidence_summary(db_session, "u1", belief.id)["counts"]
        assert counts.get("support", 0) >= 2

    def test_evidence_survives_when_legacy_json_is_empty(self, db_session):
        """The graph is authoritative even though evidence_json is still written."""
        from mirror_memory.config.loader import load_config
        from mirror_memory.extraction.pipeline import ExtractionPipeline

        set_memory_enabled(db_session, "u1", True)
        pipeline = ExtractionPipeline(config=load_config("config/"))
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping lately", 1)

        belief = db_session.query(Belief).first()
        # Legacy field is untouched by K1 (no message ids were supplied)...
        assert belief_evidence_ids(belief) == []
        # ...but the graph still records where the belief came from.
        assert evidence_summary(db_session, "u1", belief.id)["counts"]


# ---------------------------------------------------------------------------
# Retrieval uses the graph
# ---------------------------------------------------------------------------


class TestRetrievalUsesEvidenceGraph:
    def test_evidence_count_overrides_row_reading(self, db_session):
        from datetime import UTC, datetime
        from types import SimpleNamespace

        from mirror_memory.core.retrieval import score_belief

        belief = SimpleNamespace(
            dimension="topic", key="sleep", layer="L4", confidence=0.8,
            value_json="{}", evidence_json="[]", last_evidence_at=datetime.now(UTC),
            predicate="", object="",
        )
        # No evidence on the row at all.
        assert score_belief(belief, now=datetime.now(UTC)) == pytest.approx(
            score_belief(belief, now=datetime.now(UTC), evidence_count=0)
        )
        # The graph says three links -> multi-evidence bonus applies.
        with_bonus = score_belief(belief, now=datetime.now(UTC), evidence_count=3)
        assert with_bonus > score_belief(belief, now=datetime.now(UTC), evidence_count=0)

    def test_renderer_counts_links(self, db_session):
        """The renderer reads link counts and does not crash on the graph."""
        from mirror_memory.config.loader import load_config
        from mirror_memory.extraction.pipeline import ExtractionPipeline
        from mirror_memory.render.renderer import render_memory_block

        set_memory_enabled(db_session, "u1", True)
        pipeline = ExtractionPipeline(config=load_config("config/"))
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping lately", 1)

        belief = db_session.query(Belief).first()
        assert evidence_counts_for_beliefs(db_session, [belief.id])[belief.id] >= 1

        # A single K1 keyword claim is L4 at 0.4 confidence, below the 0.55
        # render watermark, so nothing renders.  What matters here is that the
        # link-count lookup runs inside the render path without error.
        block = render_memory_block(
            db_session, "u1", config=load_config("config/"),
            user_message="sleep", language="en",
        )
        assert block is None or "sleep" in block.lower()

    def test_renderer_renders_once_watermark_is_met(self, db_session):
        """With enough support the belief clears the watermark and renders."""
        from mirror_memory.config.loader import load_config
        from mirror_memory.core.repository import support_belief_by_id
        from mirror_memory.extraction.pipeline import ExtractionPipeline
        from mirror_memory.render.renderer import render_memory_block

        set_memory_enabled(db_session, "u1", True)
        pipeline = ExtractionPipeline(config=load_config("config/"))
        pipeline.observe(db_session, "u1", "s1", "I have trouble sleeping lately", 1)

        belief = db_session.query(Belief).first()
        for _ in range(3):
            support_belief_by_id(db_session, belief.id, evidence_message_ids=[])
        db_session.flush()

        block = render_memory_block(
            db_session, "u1", config=load_config("config/"),
            user_message="sleep", language="en",
        )
        assert block is not None
        assert "sleep" in block.lower()


# ---------------------------------------------------------------------------
# Explain surfaces the graph
# ---------------------------------------------------------------------------


class TestExplainIncludesEvidenceGraph:
    def test_explain_returns_graph(self, db_session):
        from mirror_memory.core.repository import explain_belief

        belief = _belief(db_session)
        ev = record_evidence(
            db_session, "u1", ref="m1", content="I sleep badly",
            extraction_method="k2_llm", authority="user",
        )
        link_evidence(db_session, belief.id, ev.id, relation="support")

        explained = explain_belief(db_session, "u1", belief.id)
        assert "evidence_graph" in explained
        assert explained["evidence_graph"]["counts"] == {"support": 1}
        assert explained["evidence_graph"]["by_relation"]["support"][0]["ref"] == "m1"
        # The legacy field is still reported for rows that predate the graph.
        assert "evidence_ids" in explained
