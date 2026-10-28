"""M3 batch-2 task 1: rows that record where memory came from carry the
purpose they were collected under.

``mm_evidence.source_purpose`` and ``mm_belief_events.source_purpose`` are
set from the SESSION's bound provenance (core.provenance): the entry binds
the purpose it just authorized, the worker binds its job's purpose, and the
writers read it at insert time.  Rows written outside an authorized entry
carry the empty "not recorded" marker.  Recorded only: nothing filters on
it yet (the cross-purpose recall semantics are docs/plans material).
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import provenance
from mirror_memory.core.models import Base, Belief, BeliefEvent, Evidence, EvolutionJob, Snapshot
from mirror_memory.core.repository import record_claim, record_evidence, set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.worker.evolution import enqueue_job, process_pending

USER = "u-prov"
TENANT = ("default", "default")


@pytest.fixture
def db(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/prov.db")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _engine(db) -> MemoryEngine:
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    return MemoryEngine(
        config=load_config("config/"), database_url=str(db.url),
        tenant_id=TENANT[0], app_id=TENANT[1],
    )


def _rows(db, model):
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            return s.execute(select(model).where(model.user_id == USER).order_by(model.id)).scalars().all()
        finally:
            s.close()


# ---------------------------------------------------------------------------
# the binding itself
# ---------------------------------------------------------------------------


def test_bind_validates_and_unbind_clears(db):
    F = sessionmaker(bind=db)
    s = F()
    try:
        assert provenance.current(s) == provenance.UNKNOWN == ""
        for bad in ("", "  ", "two words", None, 3):
            with pytest.raises(ValueError):
                provenance.bind(s, bad)  # type: ignore[arg-type]
        provenance.bind(s, "api")
        assert provenance.current(s) == "api"
        provenance.unbind(s)
        assert provenance.current(s) == ""
    finally:
        s.close()


# ---------------------------------------------------------------------------
# observe: evidence and ledger events carry the call's purpose
# ---------------------------------------------------------------------------


def test_observe_stamps_evidence_and_events_with_its_purpose(db):
    eng = _engine(db)
    eng.observe(user_id=USER, session_id="s1", text="I prefer concise answers and I like coffee.", purpose="api")
    evidence = _rows(db, Evidence)
    events = _rows(db, BeliefEvent)
    assert evidence, "observe must have recorded evidence"
    assert events, "observe must have appended ledger events"
    assert {e.source_purpose for e in evidence} == {"api"}
    assert {e.source_purpose for e in events} == {"api"}


def test_each_entry_stamps_its_own_purpose(db):
    """Two purposes on one user: the rows each entry writes are labelled
    with that entry's purpose, not the first one seen."""
    eng = _engine(db)
    eng.observe(user_id=USER, session_id="s1", text="I like coffee.", purpose="api")
    eng.observe(user_id=USER, session_id="s2", text="I live in Berlin and work as a nurse.", purpose="support")
    by_purpose = {}
    for e in _rows(db, Evidence):
        by_purpose.setdefault(e.source_purpose, set()).add(e.session_id)
    assert by_purpose == {"api": {"s1"}, "support": {"s2"}}
    assert {e.source_purpose for e in _rows(db, BeliefEvent)} == {"api", "support"}


def test_correct_stamps_the_correction_events_with_its_purpose(db):
    eng = _engine(db)
    eng.observe(user_id=USER, session_id="s1", text="I like coffee.", purpose="api")
    belief = _rows(db, Belief)[0]
    before = {e.id for e in _rows(db, BeliefEvent)}
    corrected = eng.correct(
        user_id=USER, belief_id=belief.id, new_claim_text="I like tea",
        new_object="tea" if getattr(belief, "predicate", "") else None, purpose="support",
    )
    assert corrected is not None, "the correction must have been applied"
    new_events = [e for e in _rows(db, BeliefEvent) if e.id not in before]
    assert new_events, "correct must append ledger events"
    assert {e.source_purpose for e in new_events} == {"support"}


# ---------------------------------------------------------------------------
# re-observation and the unknown marker
# ---------------------------------------------------------------------------


def test_reobserving_a_message_keeps_the_first_purpose(db):
    """Evidence is idempotent on (user, ref); a second collection under
    another purpose refreshes provenance fields but not the source
    purpose -- it records the FIRST collection."""
    eng = _engine(db)
    # the evidence ref is "<session>:<turn>", so the same turn observed
    # twice is the same message
    eng.observe(user_id=USER, session_id="s1", text="I like coffee.", turn_count=1, purpose="api")
    eng.observe(user_id=USER, session_id="s1", text="I like coffee.", turn_count=1, purpose="support")
    refs = {(e.ref, e.source_purpose) for e in _rows(db, Evidence)}
    assert refs == {("s1:1", "api")}


def test_unknown_marker_outside_an_entry_and_filled_on_next_collection(db):
    """Direct repository use (bench seeding, tests) has no bound purpose:
    the row carries the empty marker, the same value step 7 backfills.
    When the same message is next collected through an authorized entry
    the marker is replaced by that purpose."""
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("seed evidence directly"):
        s = F()
        set_memory_enabled(s, USER, True)
        row = record_evidence(s, USER, ref="s0:1", content="I like coffee.", session_id="s0")
        assert row.source_purpose == ""
        s.commit()
        s.close()
    eng = _engine(db)
    eng.observe(user_id=USER, session_id="s0", text="I like coffee.", turn_count=1, purpose="api")
    assert {(e.ref, e.source_purpose) for e in _rows(db, Evidence)} == {("s0:1", "api")}


# ---------------------------------------------------------------------------
# the worker: the job's purpose, and no leak into the next job
# ---------------------------------------------------------------------------


def test_worker_publish_stamps_the_snapshot_with_the_job_purpose(db):
    """The worker's publish writes one provenance row: the synthesised
    snapshot.  It carries the JOB's purpose (the purpose of the observe
    that enqueued it), not whatever purpose later touched the user."""
    cfg = load_config("config/")
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            set_memory_enabled(s, USER, True)
            s.commit()
            for i, text in enumerate(("SEED_A", "SEED_B", "SEED_C")):
                record_claim(s, USER, dimension="topic", key=f"k{i}", claim_text=text,
                             confidence=0.8, session_id=f"s{i}")
            s.commit()
            assert enqueue_job(s, USER, cfg, purpose="batch") is not None
            s.commit()
            assert provenance.current(s) == "", "enqueue must not bind provenance"
        finally:
            s.close()

    process_pending(F, cfg)

    with tenant_scope(*TENANT):
        s = F()
        try:
            jobs = s.execute(select(EvolutionJob)).scalars().all()
            assert [j.status for j in jobs] == ["completed"], [(j.status, j.error_code) for j in jobs]
            snapshots = s.execute(select(Snapshot).where(Snapshot.user_id == USER)).scalars().all()
            seeded = s.execute(select(BeliefEvent)).scalars().all()
        finally:
            s.close()
    assert {e.source_purpose for e in seeded} == {""}, "direct seeding carries the unknown marker"
    assert [snap.source_purpose for snap in snapshots] == ["batch"]


def test_worker_unbinds_the_session_after_the_job(db):
    """A session reused across jobs (tests pass one in) must not carry the
    previous job's purpose into the next one."""
    from mirror_memory.worker.evolution import _process_single_job

    cfg = load_config("config/")
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            set_memory_enabled(s, USER, True)
            s.commit()
            record_claim(s, USER, dimension="topic", key="k", claim_text="SEED", confidence=0.8, session_id="s1")
            s.commit()
            job = enqueue_job(s, USER, cfg, purpose="batch")
            s.commit()
            _process_single_job(s, s.get(EvolutionJob, job.id), cfg)
            assert provenance.current(s) == ""
        finally:
            s.close()
