"""M1 task 4 -- Publisher write-path收口 tests."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief, BeliefEvent, ProposalLog
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import WriteGuardViolation, tenant_scope


def session_at(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_confirm_rate_goes_through_proposal_log(tmp_path):
    from mirror_memory.extraction.verification import _update_confirm_rate

    factory = session_at(f"sqlite:///{tmp_path}/cr.db")
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        belief, _ = record_claim(
            s, "u1", dimension="topic", key="sleep",
            claim_text="bad sleep", confidence=0.8,
        )
        s.commit()
        from mirror_memory.core.repository import get_state_revision

        revision_before = get_state_revision(s, "u1")

        assert _update_confirm_rate(s, "u1", belief.id, "confirm", None)

        # The write is audited in the proposal log (one CONFIRM_RATE row).
        logs = s.query(ProposalLog).filter(
            ProposalLog.transition == "CONFIRM_RATE"
        ).all()
        assert len(logs) == 1
        assert logs[0].target_belief_id == belief.id

        # Telemetry updated, revision untouched.
        val = json.loads(belief.value_json)
        assert val["confirm_alpha"] == 2.0  # Beta(1,1) prior + 1 confirm
        assert val["confirm_rate"] == round(2.0 / 3.0, 4)
        assert get_state_revision(s, "u1") == revision_before
        s.commit()
        s.close()


def test_confirm_rate_unclear_is_a_count_noop(tmp_path):
    from mirror_memory.extraction.verification import _update_confirm_rate

    factory = session_at(f"sqlite:///{tmp_path}/cr2.db")
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        belief, _ = record_claim(
            s, "u1", dimension="topic", key="k", claim_text="c", confidence=0.8,
        )
        s.commit()
        _update_confirm_rate(s, "u1", belief.id, "unclear", None)
        val = json.loads(belief.value_json)
        assert val["confirm_alpha"] == 1.0 and val["confirm_beta"] == 1.0
        s.close()


def test_write_guard_blocks_direct_belief_writes(tmp_path, monkeypatch):
    # strict mode disables the tests/ exemption so the guard itself is
    # exercised even though the write originates from test code.
    monkeypatch.setenv("MM_WRITE_AUDIT", "strict")
    factory = session_at(f"sqlite:///{tmp_path}/guard.db")
    with tenant_scope("default", "default"):
        s = factory()
        try:
            with pytest.raises(WriteGuardViolation):
                s.add(Belief(user_id="u1", dimension="d", key="k", claim_text="x"))
                s.commit()
        finally:
            s.rollback()
            s.close()


def test_write_guard_blocks_direct_event_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("MM_WRITE_AUDIT", "strict")
    factory = session_at(f"sqlite:///{tmp_path}/guard2.db")
    with tenant_scope("default", "default"):
        s = factory()
        try:
            with pytest.raises(WriteGuardViolation):
                s.add(BeliefEvent(belief_id=1, user_id="u1", event_type="created"))
                s.commit()
        finally:
            s.rollback()
            s.close()


def test_sanctioned_entries_still_write(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/ok.db")
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        belief, event_type = record_claim(
            s, "u1", dimension="topic", key="k", claim_text="c", confidence=0.8,
        )
        s.commit()
        assert belief is not None and event_type == "created"
        assert s.query(Belief).count() == 1
        s.close()
