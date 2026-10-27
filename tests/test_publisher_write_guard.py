"""M1 batch-2 task 2 -- whitelist write guard, classified tables.

Canary suite: every test here runs WITHOUT the blanket test write scope
(@pytest.mark.no_write_fixture_scope), so the guard's teeth are proven
directly:

- each write class has a counterexample: writing a table of that class
  outside its sanctioned scope raises
- the correct scope allows the write
- a table missing from the classification mapping is refused (new tables
  must be classified first)
- an out-of-tree plugin module writing a protected table is intercepted
- engine-scoped Publisher paths keep working (confirm_rate proposal)
"""

from __future__ import annotations

import json
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief, BeliefEvent, EvolutionJob
from mirror_memory.core.repository import record_claim, set_memory_enabled
from mirror_memory.core.tenancy import (
    TABLE_WRITE_CLASSES,
    WriteGuardViolation,
    consent_write_scope,
    engine_write_scope,
    fixture_write_scope,
    queue_write_scope,
    telemetry_write_scope,
    tenant_scope,
)

pytestmark = pytest.mark.no_write_fixture_scope


def session_at(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


# ---------------------------------------------------------------------------
# Classification completeness: mapping is the single source of truth
# ---------------------------------------------------------------------------


def test_every_tenant_table_is_classified():
    from mirror_memory.core.models import Base
    from mirror_memory.core.tenancy import _iter_tenant_classes

    tables = {cls.__tablename__ for cls in _iter_tenant_classes() if hasattr(cls, "__tablename__")}
    unclassified = tables - set(TABLE_WRITE_CLASSES)
    assert not unclassified, (
        f"new tenant table(s) {sorted(unclassified)} lack a write class; "
        "add them to TABLE_WRITE_CLASSES before writing code against them"
    )
    ghosts = set(TABLE_WRITE_CLASSES) - tables
    assert not ghosts, f"TABLE_WRITE_CLASSES names unknown tables: {sorted(ghosts)}"


def test_mapping_covers_exactly_the_tenant_tables():
    from mirror_memory.core.models import Base

    assert len(TABLE_WRITE_CLASSES) == len(Base.metadata.tables) == 20


# ---------------------------------------------------------------------------
# Per-class counterexamples (wrong scope -> refused; right scope -> allowed)
# ---------------------------------------------------------------------------


def test_engine_class_wrong_scope_refused(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/e.db")
    with tenant_scope("default", "default"):
        s = factory()
        with pytest.raises(WriteGuardViolation):
            s.add(Belief(user_id="u1", dimension="d", key="k", claim_text="x"))
        # the right scope allows it
        with engine_write_scope():
            s.add(Belief(user_id="u1", dimension="d", key="k2", claim_text="ok"))
        s.rollback()
        s.close()


def test_queue_class_wrong_scope_refused(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/q.db")
    with tenant_scope("default", "default"):
        s = factory()
        with pytest.raises(WriteGuardViolation):
            s.add(EvolutionJob(id="j1", user_id="u1", idempotency_key="ik"))
        with queue_write_scope():
            s.add(EvolutionJob(id="j2", user_id="u1", idempotency_key="ik2"))
        s.rollback()
        s.close()


def test_consent_class_wrong_scope_refused(tmp_path):
    from mirror_memory.core.models import MemoryPreference

    factory = session_at(f"sqlite:///{tmp_path}/c.db")
    with tenant_scope("default", "default"):
        s = factory()
        with pytest.raises(WriteGuardViolation):
            s.add(MemoryPreference(user_id="u1", enabled=False))
        with consent_write_scope():
            s.add(MemoryPreference(user_id="u1", enabled=False))
        s.rollback()
        s.close()


def test_telemetry_class_wrong_scope_refused(tmp_path):
    from mirror_memory.core.models import ExtractionStats

    factory = session_at(f"sqlite:///{tmp_path}/t.db")
    with tenant_scope("default", "default"):
        s = factory()
        with pytest.raises(WriteGuardViolation):
            s.add(ExtractionStats(user_id="u1", trigger="test"))
        with telemetry_write_scope():
            s.add(ExtractionStats(user_id="u1", trigger="test"))
        s.rollback()
        s.close()


def test_direct_event_write_is_refused(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/g2.db")
    with tenant_scope("default", "default"):
        s = factory()
        try:
            with pytest.raises(WriteGuardViolation):
                s.add(BeliefEvent(belief_id=1, user_id="u1", event_type="created"))
                s.commit()
        finally:
            s.rollback()
            s.close()


# ---------------------------------------------------------------------------
# Out-of-tree plugin bypass (evaluation counterexample from batch 1)
# ---------------------------------------------------------------------------


def test_non_engine_module_write_is_refused(tmp_path):
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

        factory = session_at(f"sqlite:///{tmp_path}/g3.db")
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


# ---------------------------------------------------------------------------
# Sanctioned entries still work under the canary (no blanket scope)
# ---------------------------------------------------------------------------


def test_sanctioned_entries_still_write(tmp_path):
    factory = session_at(f"sqlite:///{tmp_path}/ok.db")
    with tenant_scope("default", "default"):
        s = factory()
        set_memory_enabled(s, "u1", True)  # consent-decorated
        s.commit()
        belief, event_type = record_claim(  # engine-decorated
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

        from mirror_memory.core.models import ProposalLog

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
