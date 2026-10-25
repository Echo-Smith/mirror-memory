"""M1 patch -- whitelist write-guard tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief, BeliefEvent, ProposalLog
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import (
    WriteGuardViolation,
    fixture_write_scope,
    tenant_scope,
)


def session_at(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_direct_belief_write_is_refused(tmp_path):
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


def test_direct_event_write_is_refused(tmp_path):
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


def test_non_engine_module_write_is_refused(tmp_path):
    """The evaluation's counterexample: a module OUTSIDE the engine tree
    (a future server / script / plugin) writes a belief row directly.
    The whitelist guard has no source-path exemption, so it must be
    intercepted regardless of where the writing code lives."""
    mod_dir = tmp_path / "some_plugin_pkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "writer.py").write_text(
        "from mirror_memory.core.models import Belief\n"
        "\n"
        "def write_belief(session, user_id, key):\n"
        "    session.add(Belief(user_id=user_id, dimension='d', key=key, claim_text='plugin write'))\n"
        "    session.commit()\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import some_plugin_pkg.writer as plugin  # noqa: E402

        factory = session_at(f"sqlite:///{tmp_path}/guard3.db")
        with tenant_scope("default", "default"):
            s = factory()
            try:
                with pytest.raises(WriteGuardViolation):
                    plugin.write_belief(s, "u1", "plugin-k")
            finally:
                s.rollback()
                s.close()
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("some_plugin_pkg.writer", None)
        sys.modules.pop("some_plugin_pkg", None)


def test_fixture_scope_allows_seeding(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/ok.db")
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)
        s.commit()
        with fixture_write_scope("test seeding"):
            belief, event_type = record_claim(
                s, "u1", dimension="topic", key="k", claim_text="c", confidence=0.8,
            )
        s.commit()
        assert belief is not None and event_type == "created"
        assert s.query(Belief).count() == 1
        s.close()


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

        logs = s.query(ProposalLog).filter(
            ProposalLog.transition == "CONFIRM_RATE"
        ).all()
        assert len(logs) == 1
        assert logs[0].target_belief_id == belief.id

        val = json.loads(belief.value_json)
        assert val["confirm_alpha"] == 2.0
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
