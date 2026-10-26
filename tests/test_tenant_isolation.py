"""Cross-tenant isolation tests for the spike session-hook tenancy layer.

Two tenants ``T_A=("tA","app")`` and ``T_B=("tB","app")`` share the *same*
``user_id "u1"`` — the easiest way to bleed data across tenants, because
every pre-existing query filters on ``user_id`` alone.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core.models import (
    Base,
    Belief,
    BeliefIdentity,
    DeletionTombstone,
    EvolutionJob,
    Evidence,
    ExtractionStats,
    Snapshot,
)
from mirror_memory.core.repository import get_state_revision, record_claim, set_memory_enabled
from mirror_memory.core.tenancy import (
    TenantContextError,
    TenantMismatchError,
    WriteGuardViolation,
    no_tenant_scope,
    system_scope,
    tenant_scope,
)
from mirror_memory.extraction.verification import _update_confirm_rate
from mirror_memory.worker.evolution import enqueue_job, process_pending

T_A = ("tA", "app")
T_B = ("tB", "app")
USER = "u1"

CLAIM = '{"claims": [{"dimension": "topic", "key": "sleep", "claim_text": "TENANT_MARKER", "confidence": 0.9}]}'


class MockLLM:
    def __init__(self, response: str = CLAIM):
        self._response = response

    def generate(self, *, system_prompt, payload_text, fallback=None):
        return self._response


def make_config(**overrides):
    cfg = load_config("config/")
    # Lower the compaction threshold so the metabolism cycle actually folds
    # evidence instead of scanning past everything.
    cfg.metabolism.compaction.min_support_links = 2
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


@pytest.fixture
def shared_db_url(tmp_path):
    return f"sqlite:///{tmp_path}/tenants.db"


@pytest.fixture
def raw_session_factory(shared_db_url):
    """A plain sessionmaker against the shared database (no tenant baked in)."""
    return session_at(shared_db_url)


def session_at(url):
    """Session factory against the shared database, tables ensured."""
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def make_engine(shared_db_url, tenant) -> MemoryEngine:
    return MemoryEngine(
        config_path="config/",
        database_url=shared_db_url,
        llm_client=MockLLM(),
        tenant_id=tenant[0],
        app_id=tenant[1],
    )


def seed_belief_via_claim(session_factory, tenant, key="sleep", claim_text="SEED", session_id="s1"):
    """Insert one belief row through the repository, under a tenant context."""
    with tenant_scope(*tenant):
        s = session_factory()
        try:
            set_memory_enabled(s, USER, True)
            s.commit()
            record_claim(
                s, USER,
                dimension="topic", key=key,
                claim_text=claim_text, confidence=0.8,
                session_id=session_id,
            )
            s.commit()
        finally:
            s.close()


# ---------------------------------------------------------------------------
# 1. Same (user_id, key) can live in both tenants without clobbering
# ---------------------------------------------------------------------------


def test_same_user_key_coexists_across_tenants(raw_session_factory):
    seed_belief_via_claim(raw_session_factory, T_A, claim_text="FROM_TENANT_A")
    seed_belief_via_claim(raw_session_factory, T_B, claim_text="FROM_TENANT_B")

    with tenant_scope(*T_A):
        s = raw_session_factory()
        rows = s.query(Belief).filter(Belief.user_id == USER, Belief.key == "sleep").all()
        s.close()
        assert len(rows) == 1
        assert rows[0].claim_text == "FROM_TENANT_A"

    with tenant_scope(*T_B):
        s = raw_session_factory()
        rows = s.query(Belief).filter(Belief.user_id == USER, Belief.key == "sleep").all()
        s.close()
        assert len(rows) == 1
        assert rows[0].claim_text == "FROM_TENANT_B"


# ---------------------------------------------------------------------------
# 2. After T_A observes, T_B sees nothing of it in recall/beliefs/panel/stats
# ---------------------------------------------------------------------------


@pytest.mark.fixture_writes(reason="isolation test: seed deterministic ExtractionStats row")
def test_tenant_b_cannot_read_tenant_a_content(shared_db_url):
    engine_a = make_engine(shared_db_url, T_A)
    engine_b = make_engine(shared_db_url, T_B)

    # Seed a belief directly (the K2 pipeline only promotes claims to
    # beliefs past its own evidence thresholds, which would couple this
    # isolation test to extraction behaviour).
    with tenant_scope(*T_A):
        s = session_at(shared_db_url)()
        try:
            record_claim(
                s, USER, dimension="topic", key="sleep",
                claim_text="TENANT_MARKER", confidence=0.9,
                session_id="sA1",
            )
            s.commit()
        finally:
            s.close()

    count = engine_a.observe(user_id=USER, session_id="sA2", text="I also drink coffee")
    assert count >= 1  # the engine entry ran the extraction pipeline

    # Deterministic stats row for the isolation check (the pipeline's own
    # stats write is throttled, so the test does not depend on it).
    with tenant_scope(*T_A):
        s = session_at(shared_db_url)()
        try:
            s.add(ExtractionStats(user_id=USER, trigger="test"))
            s.commit()
        finally:
            s.close()

    # recall: T_B gets nothing that mentions T_A's content
    recall_b = engine_b.recall(user_id=USER, query="How is your sleep?")
    assert recall_b is None or "TENANT_MARKER" not in recall_b

    # beliefs
    assert engine_b.get_beliefs(user_id=USER) == []

    # panel: nothing derived from T_A data reaches T_B
    panel_b = engine_b.panel(user_id=USER)
    rendered = json.dumps(panel_b.sections, default=str, ensure_ascii=False)
    assert "TENANT_MARKER" not in rendered

    # stats: T_A's extraction stats row is invisible
    with tenant_scope(*T_B):
        s = session_at(shared_db_url)()
        try:
            assert s.query(ExtractionStats).all() == []
        finally:
            s.close()

    # ...while T_A itself sees everything
    assert len(engine_a.get_beliefs(user_id=USER)) == 1
    with tenant_scope(*T_A):
        s = session_at(shared_db_url)()
        try:
            assert len(s.query(ExtractionStats).all()) >= 1
        finally:
            s.close()


# ---------------------------------------------------------------------------
# 3. T_B delete/forget leaves T_A intact; tombstones are per-tenant
# ---------------------------------------------------------------------------


def test_delete_and_forget_are_tenant_scoped(shared_db_url):
    seed_belief_via_claim(session_at(shared_db_url), T_A,
                          claim_text="FROM_TENANT_A", session_id="sA1")
    seed_belief_via_claim(session_at(shared_db_url), T_B,
                          claim_text="FROM_TENANT_B", session_id="sB1")

    engine_a = make_engine(shared_db_url, T_A)
    engine_b = make_engine(shared_db_url, T_B)

    # Full-user delete in T_B only.
    result_b = engine_b.delete_memories(user_id=USER)
    assert result_b.beliefs == 1

    # T_A's data survives untouched.
    surviving = engine_a.get_beliefs(user_id=USER)
    assert len(surviving) == 1

    # Targeted forget in T_B (nothing left to forget -> False) and in T_A.
    with tenant_scope(*T_A):
        s = session_at(shared_db_url)()
        try:
            belief_id_a = s.query(Belief).first().id
        finally:
            s.close()
    assert engine_b.forget_belief(user_id=USER, belief_id=belief_id_a) is False
    assert engine_a.forget_belief(user_id=USER, belief_id=belief_id_a) is True
    assert engine_a.get_beliefs(user_id=USER) == []

    # Tombstones: every tombstone written belongs to T_B (T_A's forget
    # tombstone is created inside T_A's context and carries T_A's tenant).
    with system_scope(reason="test: cross-tenant verification read"):
        s = session_at(shared_db_url)()
        try:
            tombstones = s.query(DeletionTombstone).all()
        finally:
            s.close()
    assert {t.tenant_id for t in tombstones} <= {"tA", "tB"}
    assert any(t.tenant_id == "tB" for t in tombstones)
    assert any(t.tenant_id == "tA" for t in tombstones)


# ---------------------------------------------------------------------------
# 4. A deliberately unfiltered session.query(Belief).all() in T_B context
# ---------------------------------------------------------------------------


def test_unfiltered_query_returns_only_own_tenant(raw_session_factory):
    seed_belief_via_claim(raw_session_factory, T_A, claim_text="FROM_TENANT_A")
    seed_belief_via_claim(raw_session_factory, T_B, claim_text="FROM_TENANT_B")

    with tenant_scope(*T_B):
        s = raw_session_factory()
        try:
            rows = s.query(Belief).all()
        finally:
            s.close()
    assert len(rows) == 1
    assert rows[0].claim_text == "FROM_TENANT_B"
    assert rows[0].tenant_id == "tB"


# ---------------------------------------------------------------------------
# 5. session.get() across tenants returns nothing and mutates nothing
# ---------------------------------------------------------------------------


def test_session_get_cannot_reach_other_tenant_row(raw_session_factory):
    seed_belief_via_claim(raw_session_factory, T_A, claim_text="FROM_TENANT_A")
    with tenant_scope(*T_A):
        s = raw_session_factory()
        try:
            id_a = s.query(Belief).first().id
        finally:
            s.close()

    with tenant_scope(*T_B):
        s = raw_session_factory()
        try:
            fetched = s.get(Belief, id_a)
            assert fetched is None
        finally:
            s.close()

    with tenant_scope(*T_A):
        s = raw_session_factory()
        try:
            row = s.get(Belief, id_a)
            assert row is not None
            assert row.claim_text == "FROM_TENANT_A"
        finally:
            s.close()


# ---------------------------------------------------------------------------
# 6. Bulk UPDATE / DELETE in T_B context leaves T_A alone
# ---------------------------------------------------------------------------


def test_bulk_update_and_delete_are_tenant_scoped(raw_session_factory):
    from sqlalchemy import delete as sa_delete
    from sqlalchemy import update as sa_update

    seed_belief_via_claim(raw_session_factory, T_A, claim_text="FROM_TENANT_A")
    seed_belief_via_claim(raw_session_factory, T_B, claim_text="FROM_TENANT_B")

    with tenant_scope(*T_B):
        from mirror_memory.core.sql_audit import no_audit_scope

        s = raw_session_factory()
        try:
            bulk_update = sa_update(Belief).values(claim_text="BULKED").execution_options(
                synchronize_session=False
            )
            s.execute(bulk_update)
            s.commit()
        finally:
            s.close()

        s = raw_session_factory()
        try:
            bulk_delete = sa_delete(Belief).where(Belief.key == "nonexistent")
            s.execute(bulk_delete)
            # query-API bulk delete of everything visible in this tenant
            # (mechanic test of hook-level isolation: audit off for the
            # hand-rolled per-row delete)
            with no_audit_scope():
                visible = s.query(Belief).all()
                for row in visible:
                    s.delete(row)
                s.commit()
        finally:
            s.close()

    with tenant_scope(*T_A):
        s = raw_session_factory()
        try:
            rows = s.query(Belief).all()
            assert len(rows) == 1
            assert rows[0].claim_text == "FROM_TENANT_A"
        finally:
            s.close()


# ---------------------------------------------------------------------------
# 7. Background work (evolution worker, metabolism, compaction) stays
#    tenant-scoped even though it runs across the shared queue
# ---------------------------------------------------------------------------


def test_background_tasks_do_not_cross_tenants(raw_session_factory):
    cfg = make_config()

    for tenant, marker in ((T_A, "A_CONTENT"), (T_B, "B_CONTENT")):
        seed_belief_via_claim(raw_session_factory, tenant, claim_text=marker, session_id=f"s1{marker}")
        seed_belief_via_claim(raw_session_factory, tenant, claim_text=marker, session_id=f"s2{marker}")
        seed_belief_via_claim(raw_session_factory, tenant, claim_text=marker, session_id=f"s3{marker}")
        with tenant_scope(*tenant):
            s = raw_session_factory()
            try:
                assert enqueue_job(s, USER, cfg) is not None
                s.commit()
            finally:
                s.close()

    # The worker scans the shared queue (system_scope) and processes each
    # job under its row's tenant.
    process_pending(raw_session_factory, cfg)

    # Metabolism (with evidence compaction) runs per tenant through its own
    # session entry.
    with tenant_scope(*T_A):
        s = raw_session_factory()
        try:
            from mirror_memory.metabolism.planner import run_metabolism

            run_metabolism(s, USER, cfg)
            s.commit()
        finally:
            s.close()
    with tenant_scope(*T_B):
        s = raw_session_factory()
        try:
            from mirror_memory.metabolism.planner import run_metabolism

            run_metabolism(s, USER, cfg)
            s.commit()
        finally:
            s.close()

    # Verify no cross-tenant bleed anywhere.
    with system_scope(reason="test: cross-tenant verification read"):
        s = raw_session_factory()
        try:
            jobs = s.query(EvolutionJob).all()
            assert {j.tenant_id for j in jobs} == {"tA", "tB"}
            assert all(j.status == "completed" for j in jobs)

            snapshots = s.query(Snapshot).all()
            assert {sn.tenant_id for sn in snapshots} == {"tA", "tB"}
            for sn in snapshots:
                content = json.dumps(json.loads(sn.content_json or "{}"))
                other = "B_CONTENT" if sn.tenant_id == "tA" else "A_CONTENT"
                assert other not in content

            # Compaction digests (one per belief) stay on their own tenant.
            from mirror_memory.core.models import EvidenceDigest

            digests = s.query(EvidenceDigest).all()
            beliefs_by_id = {b.id: b for b in s.query(Belief).all()}
            for d in digests:
                assert beliefs_by_id[d.belief_id].tenant_id == d.tenant_id
        finally:
            s.close()


# ---------------------------------------------------------------------------
# 8. The Publisher-bypassing _update_confirm_rate only touches own tenant
# ---------------------------------------------------------------------------


def test_update_confirm_rate_cannot_cross_tenants(raw_session_factory):
    seed_belief_via_claim(raw_session_factory, T_A, claim_text="FROM_TENANT_A")
    seed_belief_via_claim(raw_session_factory, T_B, claim_text="FROM_TENANT_B")

    with tenant_scope(*T_B):
        s = raw_session_factory()
        try:
            belief_b = s.query(Belief).filter(Belief.user_id == USER).first()
            _update_confirm_rate(s, "u1", belief_b.id, "confirm", None)
            s.commit()
        finally:
            s.close()

    with tenant_scope(*T_A):
        s = raw_session_factory()
        try:
            belief_a = s.query(Belief).filter(Belief.user_id == USER).first()
            val = json.loads(belief_a.value_json or "{}")
            assert "confirm_alpha" not in val or val["confirm_alpha"] == 1.0
        finally:
            s.close()

    with tenant_scope(*T_B):
        s = raw_session_factory()
        try:
            belief_b = s.query(Belief).filter(Belief.user_id == USER).first()
            val = json.loads(belief_b.value_json or "{}")
            assert val["confirm_alpha"] == 2.0  # Beta(1,1) prior + 1 confirm
        finally:
            s.close()


# ---------------------------------------------------------------------------
# 9. Missing tenant context fails closed on tenant tables
# ---------------------------------------------------------------------------


def test_missing_context_raises(raw_session_factory):
    seed_belief_via_claim(raw_session_factory, T_A, claim_text="FROM_TENANT_A")

    with no_tenant_scope():
        s = raw_session_factory()
        try:
            with pytest.raises(TenantContextError):
                s.query(Belief).all()
            with pytest.raises(TenantContextError):
                s.query(Belief).filter(Belief.user_id == USER).first()
            with pytest.raises((TenantContextError, WriteGuardViolation)):
                # construction may trip the whitelist guard first; either
                # way the write must not succeed without a tenant context
                s.add(Belief(user_id=USER, dimension="topic", key="k"))
                s.commit()
        finally:
            s.rollback()
            s.close()


# ---------------------------------------------------------------------------
# 10. Writing a tenant that contradicts the context raises
# ---------------------------------------------------------------------------


def test_mismatched_tenant_write_raises(raw_session_factory):
    with tenant_scope(*T_B):
        s = raw_session_factory()
        try:
            with pytest.raises((TenantMismatchError, WriteGuardViolation)):
                # construction may trip the whitelist guard first; either
                # way the mismatched write must be refused
                s.add(Belief(
                    user_id=USER, dimension="topic", key="k",
                    tenant_id="tA", app_id="app",
                ))
                s.commit()
        finally:
            s.rollback()
            s.close()

    # And the row was not written.
    with system_scope(reason="test: cross-tenant verification read"):
        s = raw_session_factory()
        try:
            assert s.query(Belief).count() == 0
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Ledger identities stay tenant-scoped (belt-and-braces on the uq slot)
# ---------------------------------------------------------------------------


@pytest.mark.fixture_writes(reason="isolation test: seed BeliefIdentity rows directly")
def test_belief_identity_slot_is_per_tenant(raw_session_factory):
    with tenant_scope(*T_A):
        s = raw_session_factory()
        try:
            s.add(BeliefIdentity(user_id=USER, subject="user", predicate="works_at", slot_key="lives_in"))
            s.commit()
        finally:
            s.close()
    with tenant_scope(*T_B):
        s = raw_session_factory()
        try:
            s.add(BeliefIdentity(user_id=USER, subject="user", predicate="works_at", slot_key="lives_in"))
            s.commit()
            assert s.query(BeliefIdentity).count() == 1
        finally:
            s.close()
    with system_scope(reason="test: cross-tenant verification read"):
        s = raw_session_factory()
        try:
            assert s.query(BeliefIdentity).count() == 2
        finally:
            s.close()
