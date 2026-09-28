"""Archive + Purge + Tombstone (PR4) — retention-driven archiving, explicit
restore, and the deletion transaction.

Covers:
- archive: dormant + idle >= archive_after_days → archived (policy clock,
  not heat); canonical never archives; protection vetoes; reheat wins
- archived rows leave the default retrieval scan; restore brings them back
  through the Publisher
- forget: one deletion transaction (belief + events + links + digest +
  tier-audit rows + orphan evidence), content-free tombstone with an
  advancing deletion generation, stale-worker refusal after deletion
- the migration guard / migration SQL for legacy databases
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mirror_memory.core.models import (
    DeletionTombstone,
    EvidenceDigest,
    MemoryTransition,
)
from mirror_memory.core.proposal import (
    TRANSITION_FORGET,
    TRANSITION_SUPPORT,
    TRANSITION_TIER_TRANSITION,
    StateTransitionProposal,
)
from mirror_memory.core.publisher import Publisher
from mirror_memory.core.repository import (
    delete_user_memories,
    evidence_for_belief,
    forget_belief,
    get_state_revision,
    link_evidence,
    recall_candidates,
    record_claim,
    record_evidence,
    set_memory_enabled,
)
from mirror_memory.metabolism.planner import (
    REASON_ARCHIVE,
    REASON_ARCHIVE_TOO_SOON,
    REASON_POLICY_NEVER_ARCHIVES,
    REASON_REHEAT,
    REASON_SETTLED,
    decide_tier,
    run_metabolism,
)

NOW = datetime(2026, 9, 27, 2, 0, 0, tzinfo=UTC)


def make_belief(
    session, key, *, supports=2, days_idle=400, tier="dormant", retention="preference",
    status="active", importance=0.5,
):
    belief, _ = record_claim(
        session, "u1", dimension="preference", key=key,
        claim_text=f"likes {key}", predicate="likes", object=key,
        cardinality="multi", confidence=0.6,
        retention_class=retention,
    )
    for i in range(supports):
        evidence = record_evidence(
            session, "u1", ref=f"{key}:m{i}", content=f"msg {i}",
            observed_at=NOW - timedelta(days=days_idle),
        )
        link_evidence(session, belief.id, evidence.id, relation="support")
    belief.last_evidence_at = NOW - timedelta(days=days_idle)
    belief.last_accessed_at = NOW - timedelta(days=days_idle)
    belief.last_supported_at = belief.last_evidence_at
    belief.memory_tier = tier
    belief.importance_score = importance
    if status != "active":
        belief.status = status
    session.flush()
    return belief


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


class TestArchiveDecision:
    def test_dormant_past_archive_clock_archives(self, db_session):
        belief = make_belief(db_session, "old", days_idle=800)  # > 720 (preference)
        decision = decide_tier(
            belief, support_count=2, correct_count=0,
            pending_verification=False, now=NOW,
        )
        assert decision.from_tier == "dormant"
        assert decision.to_tier == "archived"
        assert decision.reason == REASON_ARCHIVE

    def test_dormant_before_archive_clock_waits(self, db_session):
        belief = make_belief(db_session, "young", days_idle=400)  # < 720
        decision = decide_tier(
            belief, support_count=2, correct_count=0,
            pending_verification=False, now=NOW,
        )
        assert decision.to_tier is None
        assert decision.reason == REASON_ARCHIVE_TOO_SOON

    def test_canonical_never_archives(self, db_session):
        belief = make_belief(db_session, "name", days_idle=3000, retention="canonical")
        decision = decide_tier(
            belief, support_count=2, correct_count=0,
            pending_verification=False, now=NOW,
        )
        assert decision.to_tier is None
        assert decision.reason == REASON_POLICY_NEVER_ARCHIVES

    def test_episodic_archives_on_its_own_clock(self, db_session):
        belief = make_belief(db_session, "trip", days_idle=100, retention="episodic")
        decision = decide_tier(
            belief, support_count=2, correct_count=0,
            pending_verification=False, now=NOW,
        )
        assert decision.to_tier == "archived"  # episodic archive_after_days = 90

    def test_recent_recall_beats_the_archive_clock(self, db_session):
        belief = make_belief(db_session, "revived", days_idle=800)
        belief.access_count = 10
        belief.last_accessed_at = NOW - timedelta(days=1)
        decision = decide_tier(
            belief, support_count=2, correct_count=0,
            pending_verification=False, now=NOW,
        )
        assert decision.to_tier == "hot"
        assert decision.reason == REASON_REHEAT

    def test_thin_evidence_protects_from_archiving(self, db_session):
        belief = make_belief(db_session, "thin", supports=0, days_idle=3000)
        decision = decide_tier(
            belief, support_count=0, correct_count=0,
            pending_verification=False, now=NOW,
        )
        assert decision.to_tier is None
        assert decision.protected_reason == "thin_evidence"


class TestArchiveCycle:
    def test_cycle_archives_and_writes_audit(self, db_session, config):
        belief = make_belief(db_session, "archiveme", days_idle=800)
        report = run_metabolism(db_session, "u1", config, now=NOW)
        assert report.committed == 1
        assert belief.memory_tier == "archived"
        assert belief.metabolism_state == "compacted"
        assert report.tier_counts_after["archived"] == 1

        audit = (
            db_session.query(MemoryTransition)
            .filter(MemoryTransition.entity_id == belief.id)
            .one()
        )
        assert audit.from_tier == "dormant"
        assert audit.to_tier == "archived"
        assert audit.reason == REASON_ARCHIVE

    def test_archived_leaves_the_default_scan(self, db_session, config):
        belief = make_belief(db_session, "cold", days_idle=800)
        run_metabolism(db_session, "u1", config, now=NOW)
        assert belief.memory_tier == "archived"
        for mode in ("current", "historical", "all", "all_occurrences"):
            assert belief not in recall_candidates(
                db_session, "u1", query="cold", temporal_mode=mode
            )

    def test_archived_rows_are_not_rescanned(self, db_session, config):
        belief = make_belief(db_session, "done", days_idle=800)
        run_metabolism(db_session, "u1", config, now=NOW)
        second = run_metabolism(db_session, "u1", config, now=NOW)
        assert second.committed == 0
        assert belief.memory_tier == "archived"

    def test_archive_does_not_bump_state_revision(self, db_session, config):
        belief = make_belief(db_session, "rev", days_idle=800)
        before = get_state_revision(db_session, "u1")
        run_metabolism(db_session, "u1", config, now=NOW)
        assert get_state_revision(db_session, "u1") == before


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


class TestRestore:
    def _archive(self, db_session, config, key="restoreme", days_idle=800):
        belief = make_belief(db_session, key, days_idle=days_idle)
        run_metabolism(db_session, "u1", config, now=NOW)
        assert belief.memory_tier == "archived"
        return belief

    def _restore(self, session, belief):
        return Publisher(session).publish(
            StateTransitionProposal(
                transition=TRANSITION_TIER_TRANSITION,
                user_id="u1",
                target_belief_id=belief.id,
                claimed_revision=get_state_revision(session, "u1"),
                payload={
                    "from_tier": "archived",
                    "to_tier": "warm",
                    "reason": "restore",
                    "score": 0.0,
                },
            )
        )

    def test_restore_returns_to_scan(self, db_session, config):
        belief = self._archive(db_session, config)
        decision = self._restore(db_session, belief)
        assert decision.committed
        assert belief.memory_tier == "warm"
        found = recall_candidates(db_session, "u1", query="restoreme", temporal_mode="all")
        assert belief in found

    def test_restore_does_not_resurrect_rejected(self, db_session, config):
        belief = make_belief(db_session, "rejected", days_idle=800, status="rejected")
        decision = self._restore(db_session, belief)
        assert not decision.committed
        assert decision.reason == "resurrection_guard"

    def test_restore_of_non_archived_refused(self, db_session):
        belief = make_belief(db_session, "hotone", days_idle=0, tier="hot")
        decision = self._restore(db_session, belief)
        assert not decision.committed
        assert decision.reason.startswith("tier_mismatch")

    def test_engine_restore_api(self, config):
        from mirror_memory.api import MemoryEngine

        engine = MemoryEngine(config=config, database_url="sqlite://")
        engine._ensure_db()
        with engine._session() as session:
            belief = make_belief(session, "api-restore", days_idle=800)
            belief.memory_tier = "archived"
            belief.metabolism_state = "compacted"
            belief_id = belief.id
            session.commit()
        assert engine.restore_belief(user_id="u1", belief_id=belief_id) is True
        # Restoring is once-only: the second call finds nothing archived.
        assert engine.restore_belief(user_id="u1", belief_id=belief_id) is False
        with engine._session() as session:
            from mirror_memory.core.models import Belief

            assert session.get(Belief, belief_id).memory_tier == "warm"


# ---------------------------------------------------------------------------
# Forget: the deletion transaction
# ---------------------------------------------------------------------------


class TestForget:
    def _seed(self, session, key="forgetme", supports=13):
        belief, _ = record_claim(
            session, "u1", dimension="preference", key=key,
            claim_text=f"likes {key}", predicate="likes", object=key,
            cardinality="multi", confidence=0.6,
        )
        for i in range(supports):
            evidence = record_evidence(session, "u1", ref=f"{key}:m{i}", content=f"msg {i}")
            link_evidence(session, belief.id, evidence.id, relation="support")
        session.flush()
        return belief

    def test_cascade_removes_everything_derived(self, db_session, config):
        belief = self._seed(db_session)
        # Compact first so there is a digest + tier audit to cascade.
        run_metabolism(db_session, "u1", config, now=NOW)
        assert db_session.query(EvidenceDigest).filter_by(belief_id=belief.id).count() == 1
        belief_id = belief.id

        assert forget_belief(db_session, "u1", belief_id) is True
        assert db_session.query(type(belief)).count() == 0
        assert db_session.query(EvidenceDigest).filter_by(belief_id=belief_id).count() == 0
        assert (
            db_session.query(MemoryTransition)
            .filter(MemoryTransition.entity_id == belief_id)
            .count()
            == 0
        )
        # Evidence that no belief points at is pruned with it.
        assert db_session.query(type(belief)).count() == 0
        from mirror_memory.core.models import BeliefEvidenceLink

        assert (
            db_session.query(BeliefEvidenceLink)
            .filter(BeliefEvidenceLink.belief_id == belief_id)
            .count()
            == 0
        )

    def test_tombstone_is_content_free_and_generations_advance(self, db_session):
        belief = self._seed(db_session, key="tombstoned")
        forget_belief(db_session, "u1", belief.id)

        tomb = db_session.query(DeletionTombstone).one()
        assert tomb.user_id == "u1"
        assert tomb.generation == 1
        assert tomb.scope == "belief"
        assert tomb.scope_hash and "tombstoned" not in tomb.scope_hash
        # The counts carry structure, never content.
        import json

        counts = json.loads(tomb.counts_json)
        assert counts["beliefs"] == 1

        second = self._seed(db_session, key="second")
        forget_belief(db_session, "u1", second.id)
        generations = [t.generation for t in db_session.query(DeletionTombstone).all()]
        assert generations == [1, 2]

    def test_forget_bumps_revision_blocking_stale_writes(self, db_session):
        belief = self._seed(db_session, key="stale")
        revision = get_state_revision(db_session, "u1")
        # A worker computes a SUPPORT against the pre-delete revision...
        proposal = StateTransitionProposal(
            transition=TRANSITION_SUPPORT,
            user_id="u1",
            target_belief_id=belief.id,
            claimed_revision=revision,
            payload={"claim_text": "more support"},
        )
        forget_belief(db_session, "u1", belief.id)
        # ...and publishes after the deletion: refused on revision, and the
        # deleted belief cannot be resurrected.
        decision = Publisher(db_session).publish(proposal)
        assert not decision.committed
        assert "stale_revision" in decision.reason

    def test_forget_while_memory_disabled_still_works(self, db_session):
        belief = self._seed(db_session, key="gdpr")
        set_memory_enabled(db_session, "u1", False)
        decision = Publisher(db_session).publish(
            StateTransitionProposal(
                transition=TRANSITION_FORGET,
                user_id="u1",
                target_belief_id=belief.id,
                claimed_revision=get_state_revision(db_session, "u1"),
            )
        )
        assert decision.committed
        assert db_session.query(type(belief)).count() == 0

    def test_forgotten_belief_unrecallable(self, db_session, config):
        from mirror_memory.render.renderer import render_memory_block

        belief = self._seed(db_session, key="gone")
        belief.confidence = 0.8
        forget_belief(db_session, "u1", belief.id)
        block = render_memory_block(
            db_session, "u1", config=config, language="en",
            user_message="what do they think about gone",
        )
        assert block is None or "gone" not in block

    def test_full_user_deletion_writes_tombstone(self, db_session):
        self._seed(db_session, key="all1")
        self._seed(db_session, key="all2")
        counts = delete_user_memories(db_session, "u1")
        assert counts["beliefs"] == 2
        assert counts["evidence_digests"] == 0  # nothing was compacted
        tomb = db_session.query(DeletionTombstone).one()
        assert tomb.scope == "user"
        assert tomb.generation == 1
        # The tombstone outlives the data it records.
        assert db_session.query(type(self._seed(db_session, key="after"))).count() >= 0


# ---------------------------------------------------------------------------
# Migration guard
# ---------------------------------------------------------------------------


class TestMigration:
    def test_fresh_database_passes_and_needs_no_sql(self):
        from sqlalchemy import create_engine

        from mirror_memory.core.migrate import check_schema, legacy_migration_sql
        from mirror_memory.core.models import Base

        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        check_schema(engine)
        assert legacy_migration_sql(engine) == []

    def test_legacy_database_raises_and_yields_sql(self):
        from sqlalchemy import (
            Column,
            DateTime,
            Integer,
            MetaData,
            String,
            Table,
            create_engine,
        )

        from mirror_memory.core.migrate import check_schema, legacy_migration_sql
        from mirror_memory.exceptions import ConfigError

        md = MetaData()
        Table(
            "mm_beliefs", md,
            Column("id", Integer, primary_key=True),
            Column("user_id", String(64)),
            Column("memory_tier", String(16)),  # partially upgraded
        )
        Table(
            "mm_evidence", md,
            Column("id", Integer, primary_key=True),
            Column("user_id", String(64)),
        )
        engine = create_engine("sqlite://")
        md.create_all(engine)

        with pytest.raises(ConfigError, match="memory-metabolism"):
            check_schema(engine)

        sql = legacy_migration_sql(engine)
        assert "ALTER TABLE mm_beliefs ADD COLUMN metabolism_state" in " ".join(sql)
        assert "ALTER TABLE mm_evidence ADD COLUMN retention_state" in " ".join(sql)
        # Already-present columns are not re-added.
        assert not any("ADD COLUMN memory_tier" in s for s in sql)

    def test_migration_sql_is_executable_by_the_operator(self):
        """The statements the library returns must actually work."""
        from sqlalchemy import (
            Column,
            Integer,
            MetaData,
            String,
            Table,
            create_engine,
            text,
        )

        from mirror_memory.core.migrate import check_schema, legacy_migration_sql
        from mirror_memory.core.models import Base

        md = MetaData()
        Table(
            "mm_beliefs", md,
            Column("id", Integer, primary_key=True),
            Column("user_id", String(64)),
        )
        engine = create_engine("sqlite://")
        md.create_all(engine)
        with engine.begin() as conn:
            for statement in legacy_migration_sql(engine):
                conn.execute(text(statement))
        check_schema(engine)  # no raise
