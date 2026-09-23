"""Runtime invariant tests.

These tests pin down the runtime holes that make every other score
untrustworthy.  They are deliberately about *side effects*, not return
values: a stale write, a consent change, or a Compute node that quietly
persists must leave the database in exactly the state it was in.

The invariant under test is the Publisher rule: only ``worker.evolution``
Node 6 may write derived state, and it may do so only when both the consent
gate and the revision guard pass.  Everything upstream is pure Compute.
"""

import json

from mirror_memory.config.loader import load_config
from mirror_memory.core.models import (
    Belief,
    BeliefEvent,
    ConsentGrant,
    EvolutionJob,
    ExtractionStats,
    InterventionEvent,
    MemoryPreference,
    SessionSummary,
    Snapshot,
)
from mirror_memory.core.repository import (
    bump_state_revision,
    get_state_revision,
    record_claim,
    set_memory_enabled,
    touch_memory_state,
)
from mirror_memory.worker.evolution import _process_single_job, enqueue_job
from mirror_memory.worker.snapshot import persist_snapshot

# Tables that hold user-visible memory state.  EvolutionJob is excluded: a
# blocked job must still record *why* it was blocked, or the same job would
# be retried forever with no way to tell it was refused.
MEMORY_STATE_MODELS = (
    Belief,
    BeliefEvent,
    Snapshot,
    MemoryPreference,
    ConsentGrant,
    SessionSummary,
    InterventionEvent,
    ExtractionStats,
)


def _dump_memory_state(session):
    """Return a fully comparable snapshot of every memory-state table.

    ``mm_memory_preferences`` is projected down to ``user_id`` only: it is a
    control record, not memory content, and the simulated concurrent action
    under test legitimately changes ``enabled`` / ``state_revision`` /
    ``updated_at`` on it.  Everything the worker might wrongly write --
    beliefs, events, snapshots, summaries, stats -- is compared in full.
    """
    dump = {}
    for model in MEMORY_STATE_MODELS:
        if model is MemoryPreference:
            dump[model.__table__.name] = sorted(
                (row.user_id,) for row in session.query(model).all()
            )
            continue
        rows = []
        for row in session.query(model).all():
            rows.append(
                tuple(
                    "" if (value := getattr(row, col.key)) is None else str(value)
                    for col in model.__table__.columns
                )
            )
        dump[model.__table__.name] = sorted(rows)
    return dump


def _dump_job(session, job_id):
    job = session.get(EvolutionJob, job_id)
    assert job is not None
    return {
        col.key: "" if (value := getattr(job, col.key)) is None else str(value)
        for col in EvolutionJob.__table__.columns
    }


def _fake_config():
    """Config with an LLM that always answers -- drives the K3/formulate paths."""
    cfg = load_config("config/")

    class FakeLLM:
        def generate(self, **kwargs):
            payload = kwargs.get("payload_text", "")
            if "[Current understanding" in payload:
                return json.dumps({"summary": "understood", "traits": ["curious"]})
            return json.dumps({"patterns": [], "support_policy": {}})

    cfg.llm_client = FakeLLM()
    return cfg


def _seed_user(session, user_id="u1", *, sessions=("s1", "s2")):
    set_memory_enabled(session, user_id, True)
    record_claim(
        session, user_id, dimension="topic", key="sleep",
        claim_text="User has trouble sleeping", confidence=0.8,
        session_id=sessions[0], evidence_message_ids=[1, 2],
    )
    record_claim(
        session, user_id, dimension="topic", key="work",
        claim_text="User works on a research team", confidence=0.8,
        session_id=sessions[1], evidence_message_ids=[3],
    )
    return get_state_revision(session, user_id)


def _bump_during_formulate(monkeypatch):
    """Make the worker's compute phase look like a concurrent user action."""

    def _formulate_with_concurrent_write(session, user_id, evidence, cfg):
        touch_memory_state(session, user_id)
        return {"patterns": [], "support_policy": {}}

    monkeypatch.setattr(
        "mirror_memory.worker.evolution._formulate",
        _formulate_with_concurrent_write,
    )


# ---------------------------------------------------------------------------
# Revision guard: a stale write must not touch memory state
# ---------------------------------------------------------------------------


class TestStaleWriteLeavesDatabaseUnchanged:
    def test_revision_mismatch_does_not_persist_snapshot(self, db_session, monkeypatch):
        """The headline regression: revision bumped mid-job → nothing published."""
        _seed_user(db_session)
        config = _fake_config()
        job = enqueue_job(db_session, "u1", config)
        assert job is not None

        before = _dump_memory_state(db_session)
        _bump_during_formulate(monkeypatch)

        _process_single_job(db_session, job, config)

        assert _dump_memory_state(db_session) == before, "stale write changed memory state"
        assert session_count(db_session, Snapshot) == 0
        assert job.status == "cancelled"
        assert job.error_code.startswith("stale_revision")

    def test_stale_job_only_touches_its_own_bookkeeping(self, db_session, monkeypatch):
        """Even the job row may change only in status / completion / reason."""
        _seed_user(db_session)
        config = _fake_config()
        job = enqueue_job(db_session, "u1", config)

        before_job = _dump_job(db_session, job.id)
        _bump_during_formulate(monkeypatch)

        _process_single_job(db_session, job, config)
        after_job = _dump_job(db_session, job.id)

        changed = {k for k in before_job if before_job[k] != after_job[k]}
        assert changed <= {"status", "completed_at", "error_code"}, (
            f"blocked job mutated unexpected columns: {sorted(changed)}"
        )

    def test_stale_write_does_not_create_or_retire_snapshot(self, db_session, monkeypatch):
        """An existing snapshot must survive a blocked publish untouched."""
        _seed_user(db_session)
        persist_snapshot(db_session, "u1", {"v": 1}, {}, "wm1", shadow=True)
        config = _fake_config()
        job = enqueue_job(db_session, "u1", config)

        before = _dump_memory_state(db_session)
        snapshot_before = _dump_memory_state(db_session)["mm_snapshots"]
        _bump_during_formulate(monkeypatch)

        _process_single_job(db_session, job, config)

        assert _dump_memory_state(db_session) == before
        snapshots = db_session.query(Snapshot).all()
        assert len(snapshots) == 1
        assert _dump_memory_state(db_session)["mm_snapshots"] == snapshot_before

    def test_k3_output_is_not_persisted_when_guard_fires(self, db_session, monkeypatch):
        """The exact old hole: K3 used to write its own snapshot before the guard."""
        _seed_user(db_session)
        config = _fake_config()
        job = enqueue_job(db_session, "u1", config)

        before = _dump_memory_state(db_session)
        _bump_during_formulate(monkeypatch)

        _process_single_job(db_session, job, config)

        assert _dump_memory_state(db_session) == before
        assert session_count(db_session, Snapshot) == 0, "K3 leaked a snapshot past the guard"


def session_count(session, model):
    return session.query(model).count()


# ---------------------------------------------------------------------------
# Consent gate
# ---------------------------------------------------------------------------


class TestMemoryDisabledRuntime:
    def test_disable_bumps_revision(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        before = get_state_revision(db_session, "u1")
        set_memory_enabled(db_session, "u1", False)
        assert get_state_revision(db_session, "u1") > before

    def test_reenable_also_bumps(self, db_session):
        set_memory_enabled(db_session, "u1", False)
        before = get_state_revision(db_session, "u1")
        set_memory_enabled(db_session, "u1", True)
        assert get_state_revision(db_session, "u1") > before

    def test_noop_toggle_does_not_bump(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        before = get_state_revision(db_session, "u1")
        set_memory_enabled(db_session, "u1", True)
        assert get_state_revision(db_session, "u1") == before

    def test_first_write_of_default_does_not_bump(self, db_session):
        """A missing row already means enabled; recording the default is a no-op."""
        assert get_state_revision(db_session, "u1") == 1
        set_memory_enabled(db_session, "u1", True)
        assert get_state_revision(db_session, "u1") == 1

    def test_disabled_mid_job_blocks_publish(self, db_session, monkeypatch):
        """Disabling memory while the worker computes must stop the publish."""
        _seed_user(db_session)
        config = _fake_config()
        job = enqueue_job(db_session, "u1", config)

        before = _dump_memory_state(db_session)

        def _formulate_then_disable(session, user_id, evidence, cfg):
            set_memory_enabled(session, user_id, False)
            return {"patterns": [], "support_policy": {}}

        monkeypatch.setattr(
            "mirror_memory.worker.evolution._formulate", _formulate_then_disable
        )

        _process_single_job(db_session, job, config)

        assert _dump_memory_state(db_session) == before
        assert job.status == "cancelled"
        assert job.error_code == "memory_disabled"

    def test_job_never_starts_when_disabled(self, db_session, monkeypatch):
        """The consent gate runs before any evidence is read into a payload."""
        _seed_user(db_session)
        config = _fake_config()
        job = enqueue_job(db_session, "u1", config)
        set_memory_enabled(db_session, "u1", False)

        called = []

        def _formulate_spy(session, user_id, evidence, cfg):
            called.append(user_id)
            return {"patterns": [], "support_policy": {}}

        monkeypatch.setattr("mirror_memory.worker.evolution._formulate", _formulate_spy)

        _process_single_job(db_session, job, config)

        assert called == [], "evidence was computed for a user with memory disabled"
        assert job.status == "cancelled"
        assert job.error_code == "memory_disabled"

    def test_enqueue_refuses_when_disabled(self, db_session):
        set_memory_enabled(db_session, "u1", False)
        config = _fake_config()
        assert enqueue_job(db_session, "u1", config) is None


# ---------------------------------------------------------------------------
# Compute purity
# ---------------------------------------------------------------------------


class TestComputeNodesArePure:
    def test_k3_synthesize_writes_nothing(self, db_session):
        """K3 is a Compute node: it returns understanding and persists nothing."""
        from mirror_memory.extraction.synthesis import Synthesizer

        _seed_user(db_session)
        config = _fake_config()
        before = _dump_memory_state(db_session)

        result = Synthesizer(config).synthesize(db_session, "u1", config)

        assert result == {"summary": "understood", "traits": ["curious"]}
        assert _dump_memory_state(db_session) == before, "K3 performed a write"

    def test_k3_returns_none_without_llm(self, db_session):
        from mirror_memory.extraction.synthesis import Synthesizer

        _seed_user(db_session)
        config = load_config("config/")
        config.llm_client = None
        assert Synthesizer(config).synthesize(db_session, "u1", config) is None

    def test_identity_resolution_writes_nothing(self, db_session):
        from mirror_memory.memory.atom import CandidateAtom
        from mirror_memory.memory.identity import resolve_identity

        _seed_user(db_session)
        before = _dump_memory_state(db_session)
        existing = [
            {"id": 1, "predicate": "lives_in", "object": "shanghai",
             "status": "active", "confidence": 0.7},
        ]
        resolution = resolve_identity(
            CandidateAtom(subject="user", predicate="lives_in", object="beijing"),
            existing,
            {"lives_in": "single"},
        )
        assert resolution.action == "UPDATE"
        assert _dump_memory_state(db_session) == before

    def test_successful_publish_is_the_only_writer(self, db_session):
        """A clean run publishes exactly one snapshot and nothing else."""
        _seed_user(db_session)
        config = _fake_config()
        job = enqueue_job(db_session, "u1", config)

        before = _dump_memory_state(db_session)
        _process_single_job(db_session, job, config)
        after = _dump_memory_state(db_session)

        for table in before:
            if table == "mm_snapshots":
                continue
            assert before[table] == after[table], f"unexpected write to {table}"

        snapshots = db_session.query(Snapshot).all()
        assert len(snapshots) == 1
        assert snapshots[0].status == "active"  # two sessions of evidence
        content = json.loads(snapshots[0].content_json)
        assert content["understanding"] == {"summary": "understood", "traits": ["curious"]}


# ---------------------------------------------------------------------------
# Evidence accounting
# ---------------------------------------------------------------------------


class TestEvidenceAccounting:
    def test_multi_evidence_bonus_reads_evidence_json(self, db_session):
        """The counting bug: evidence lives in evidence_json, not value_json."""
        from datetime import UTC, datetime

        from mirror_memory.core.retrieval import score_belief

        record_claim(
            db_session, "u1", dimension="topic", key="sleep",
            claim_text="trouble sleeping", confidence=0.8,
            evidence_message_ids=[1, 2, 3],
        )
        belief = db_session.query(Belief).filter_by(key="sleep").one()

        assert json.loads(belief.evidence_json) == [1, 2, 3]
        assert "evidence_ids" not in json.loads(belief.value_json)

        scored = score_belief(belief, now=datetime.now(UTC))
        # Same belief with fewer than MULTI_EVIDENCE_THRESHOLD evidence items.
        belief.evidence_json = json.dumps([1])
        plain = score_belief(belief, now=datetime.now(UTC))
        assert scored > plain

    def test_legacy_value_json_evidence_still_counted(self):
        from types import SimpleNamespace

        from mirror_memory.core.utils import belief_evidence_ids

        legacy = SimpleNamespace(
            evidence_json="[]",
            value_json=json.dumps({"evidence_ids": [4, 5, 6]}),
        )
        assert belief_evidence_ids(legacy) == [4, 5, 6]

    def test_evidence_json_takes_precedence(self):
        from types import SimpleNamespace

        from mirror_memory.core.utils import belief_evidence_ids

        both = SimpleNamespace(
            evidence_json=json.dumps([7, 8, 9]),
            value_json=json.dumps({"evidence_ids": [1]}),
        )
        assert belief_evidence_ids(both) == [7, 8, 9]

    def test_malformed_evidence_json_does_not_raise(self):
        from types import SimpleNamespace

        from mirror_memory.core.utils import belief_evidence_ids

        broken = SimpleNamespace(evidence_json="{not json", value_json="{}")
        assert belief_evidence_ids(broken) == []

    def test_support_accumulates_evidence_ids(self, db_session):
        from mirror_memory.core.repository import support_belief_by_id

        record_claim(
            db_session, "u1", dimension="topic", key="sleep",
            claim_text="trouble sleeping", confidence=0.8,
            evidence_message_ids=[1],
        )
        belief = db_session.query(Belief).filter_by(key="sleep").one()
        support_belief_by_id(db_session, belief.id, evidence_message_ids=[2, 3])
        assert json.loads(belief.evidence_json) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Revision plumbing
# ---------------------------------------------------------------------------


class TestRevisionPlumbing:
    def test_bump_is_monotonic(self, db_session):
        seen = [bump_state_revision(db_session, "u1") for _ in range(5)]
        assert seen == sorted(seen)
        assert len(set(seen)) == 5

    def test_missing_row_defaults_to_one(self, db_session):
        assert get_state_revision(db_session, "nobody") == 1

    def test_state_mutations_bump(self, db_session):
        set_memory_enabled(db_session, "u1", True)
        before = get_state_revision(db_session, "u1")
        record_claim(
            db_session, "u1", dimension="topic", key="sleep",
            claim_text="x", confidence=0.5, session_id="s1",
        )
        assert get_state_revision(db_session, "u1") > before
