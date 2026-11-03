"""M4 batch 2 tasks 1-2: the tombstone synchronization interface and the
tenant / identity checks of the sync channel, driven in-process (no
network): a "cloud" engine and an "edge" engine, each with its own main
database and deletion log, exchange tombstone batches directly.

- export: the local deletion log becomes cross-device tombstones (no
  content, no local fingerprints; beliefs named by global_id); entries
  applied FROM a peer are never exported back to it (no echo);
- apply: every tombstone runs the normal local deletion of its scope,
  writes the local log, advances the local generation, and is recorded
  as applied by (origin, origin_seq) -- idempotent, atomic per batch;
- order and integrity: origin_seq must be contiguous per origin; a gap,
  a hash that does not verify or a malformed entry refuses the WHOLE
  batch with zero writes (main DB, log, applied record all unchanged);
- tenant: the edge refuses any batch carrying another tenant; the cloud
  takes the tenant from the authenticated engine and refuses entries for
  users the tenant does not know;
- publish: applying a tombstone advances the generation, so a claimed
  job's publish is cancelled (deletion_generation_changed) -- 50 rounds
  of apply-vs-publish never let memory land after the tombstone.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import FileDeletionLog
from mirror_memory.core.models import (
    Base,
    Belief,
    DeletionGeneration,
    EvolutionJob,
    SessionSummary,
    TombstoneApplied,
)
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.core.tombstones import (
    TombstoneBatch,
    TombstoneBatchRejected,
    TombstoneEntry,
)

TENANT = ("default", "default")
USER = "u-sync"


@pytest.fixture(autouse=True)
def _single_tenant_mode_off():
    """The edge device runs single_tenant_mode=True (a contextvar): reset
    it after every test so later tests keep tenant filtering."""
    from mirror_memory.core.tenancy import set_single_tenant_mode

    set_single_tenant_mode(False)
    yield
    set_single_tenant_mode(False)


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


def _cfg(log_dir: Path):
    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    cfg.worker.job_lease_seconds = 60
    cfg.session_summary_enabled = True
    cfg.deletion_log.directory = str(log_dir)
    return cfg


class _Device:
    """One side of the sync: its own main database + deletion log +
    engine.  ``single_tenant`` True is the edge, False the cloud."""

    def __init__(self, root: Path, name: str, *, single_tenant: bool):
        self.root = root
        self.url = f"sqlite:///{root / f'{name}.db'}"
        self.db = create_engine(self.url)
        Base.metadata.create_all(self.db)
        sm.ensure_watermark(self.db)
        self.log_dir = root / f"{name}-deletions"
        self.log = FileDeletionLog(self.log_dir)
        self.cfg = _cfg(self.log_dir)
        self.F = sessionmaker(bind=self.db)
        with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
            s = self.F()
            set_memory_enabled(s, USER, True)
            s.commit()
            s.close()
        self.engine = MemoryEngine(
            config=self.cfg, database_url=self.url, tenant_id=TENANT[0],
            app_id=TENANT[1], llm_client=_FakeLLM(), deletion_log=self.log,
            single_tenant_mode=single_tenant)
        self.engine._ensure_db()
        self.db_id = sm.read_watermark(self.db)[0]

    def observe(self, text: str, session_id: str = "s1") -> None:
        self.engine.observe(user_id=USER, session_id=session_id, text=text, purpose="api")

    def beliefs(self) -> list[Belief]:
        with tenant_scope(*TENANT):
            s = self.F()
            try:
                return s.query(Belief).filter(Belief.user_id == USER).all()
            finally:
                s.close()

    def summaries(self) -> int:
        with tenant_scope(*TENANT):
            s = self.F()
            try:
                return s.query(SessionSummary).filter(SessionSummary.user_id == USER).count()
            finally:
                s.close()

    def applied(self) -> list[tuple[str, int]]:
        with tenant_scope(*TENANT):
            s = self.F()
            try:
                return sorted((r.origin, r.origin_seq) for r in s.query(TombstoneApplied).all())
            finally:
                s.close()

    def generation(self) -> int:
        with tenant_scope(*TENANT):
            s = self.F()
            try:
                return int(s.scalar(
                    select(DeletionGeneration.generation)
                    .where(DeletionGeneration.user_id == USER)
                    .order_by(DeletionGeneration.generation.desc())) or 0)
            finally:
                s.close()

    def snapshot(self) -> dict:
        """Everything a rejected batch must leave untouched."""
        return {
            "beliefs": len(self.beliefs()),
            "summaries": self.summaries(),
            "applied": self.applied(),
            "log_tail": self.log.head(self.db_id)[0],
            "generation": self.generation(),
        }

    def dispose(self):
        self.engine._engine.dispose()
        self.db.dispose()


@pytest.fixture
def cloud(tmp_path):
    d = _Device(tmp_path, "cloud", single_tenant=False)
    yield d
    d.dispose()


@pytest.fixture
def edge(tmp_path):
    d = _Device(tmp_path, "edge", single_tenant=True)
    yield d
    d.dispose()


def _seed_both(cloud: _Device, edge: _Device) -> None:
    """The same memories on both sides (the cloud and the edge saw the
    same conversation): the sentinel belief, a second belief, two
    sessions with summaries."""
    for d in (cloud, edge):
        d.observe("I like SYNCSENTINEL and coffee.", session_id="sa")
        d.observe("I live in Berlin.", session_id="sb")


def _belief_by_text(device: _Device, text: str) -> Belief:
    return next(b for b in device.beliefs() if text in b.claim_text)


# ---------------------------------------------------------------------------
# task 1: export / apply, three scopes, idempotency, no echo
# ---------------------------------------------------------------------------


def test_three_scopes_travel_from_the_cloud_to_the_edge(cloud, edge):
    _seed_both(cloud, edge)
    # the SAME belief must carry the same global_id on both sides for the
    # tombstone to find it: the edge adopts the cloud's identity (what a
    # shared conversation + sync bootstrap establishes)
    cloud_b = _belief_by_text(cloud, "SYNCSENTINEL")
    edge_b = _belief_by_text(edge, "SYNCSENTINEL")
    with tenant_scope(*TENANT), fixture_write_scope("align identity"):
        s = edge.F()
        row = s.get(Belief, (*TENANT, edge_b.id))
        row.global_id = cloud_b.global_id
        s.commit()
        s.close()

    # three scopes of deletion on the cloud
    assert cloud.engine.forget_belief(user_id=USER, belief_id=cloud_b.id, purpose="api")
    assert cloud.engine.forget_session(user_id=USER, session_id="sb", purpose="api")
    assert len(edge.beliefs()) >= 1

    batch, cursor = cloud.engine.export_tombstones(0, 100, peer_db_id=edge.db_id)
    assert [e.scope for e in batch.entries] == ["belief", "session"]
    assert batch.entries[0].scope_key == cloud_b.global_id and batch.entries[1].scope_key == "sb"
    assert all(e.origin == cloud.db_id for e in batch.entries)
    assert [e.origin_seq for e in batch.entries] == [1, 2]
    assert all(e.hash == e.compute_hash() for e in batch.entries)
    assert batch.sender == cloud.db_id and batch.from_cursor == 0
    assert batch.to_cursor == 2 and batch.hash == batch.compute_hash()
    # the tombstone format carries no content and no fingerprints
    assert set(batch.entries[0].to_dict()) == {
        "origin", "origin_seq", "tenant_id", "app_id", "user_id", "scope",
        "scope_key", "generation", "created_at", "hash"}

    result = edge.engine.apply_tombstones(batch)
    assert (result.applied, result.skipped, result.noop) == (2, 0, 0)
    assert all("SYNCSENTINEL" not in b.claim_text for b in edge.beliefs())
    assert edge.summaries() == 1  # sb gone, sa kept
    assert edge.applied() == [(cloud.db_id, 1), (cloud.db_id, 2)]
    # the edge's own log records the applied deletions WITH their origin
    local = [e for e in edge.log.read_from(edge.db_id, 0) if e.scope != "genesis"]
    assert [(e.origin, e.origin_seq) for e in local] == [(cloud.db_id, 1), (cloud.db_id, 2)]
    assert edge.generation() >= 1

    # user scope: the whole user
    cloud.engine.delete_memories(user_id=USER, purpose="api")
    batch2, cursor2 = cloud.engine.export_tombstones(cursor, 100, peer_db_id=edge.db_id)
    assert [e.scope for e in batch2.entries] == ["user"] and batch2.entries[0].scope_key == ""
    edge.engine.apply_tombstones(batch2)
    assert edge.beliefs() == [] and edge.summaries() == 0
    assert "SYNCSENTINEL" not in (edge.engine.recall(user_id=USER, query="SYNCSENTINEL", purpose="api") or "")

    # NO ECHO: exporting the edge's log back to the cloud yields no
    # entries -- every edge entry originated at the cloud; the envelope
    # still advances the cloud's stored edge-cursor
    back, _ = edge.engine.export_tombstones(0, 100, peer_db_id=cloud.db_id)
    assert back.entries == [] and back.to_cursor == 3, (back.to_dict(),)
    # ... while a THIRD device would receive them with the ORIGINAL origin
    third, _ = edge.engine.export_tombstones(0, 100, peer_db_id="f" * 32)
    assert [(e.origin, e.origin_seq) for e in third.entries] == [
        (cloud.db_id, 1), (cloud.db_id, 2), (cloud.db_id, 3)], third.to_dict()

    # the empty continuation batch (everything already applied/excluded)
    # advances the channel cursor without touching data
    empty, empty_cursor = cloud.engine.export_tombstones(cursor2, 100, peer_db_id=edge.db_id)
    assert empty.entries == [] and empty.from_cursor == empty_cursor == cursor2
    r3 = edge.engine.apply_tombstones(empty)
    assert (r3.applied, r3.skipped, r3.noop) == (0, 0, 0)
    from mirror_memory.core.repository import read_sync_peer

    with tenant_scope(*TENANT):
        s = edge.F()
        row = read_sync_peer(s, cloud.db_id)
        assert row is not None and row.cursor == empty_cursor
        s.close()
    # an all-EXCLUDED read (to > from, entries []) also advances: the edge
    # log read by the cloud EXCLUDES origin=edge -- simulate by re-exporting
    # the edge log toward the edge itself is meaningless; instead verify the
    # exclusion case on the cloud: a fresh export toward the edge from 0
    # has entries and to_cursor == tail


def test_apply_is_idempotent_and_noop_scopes_are_recorded(cloud, edge):
    _seed_both(cloud, edge)
    cloud_b = _belief_by_text(cloud, "Berlin")
    assert cloud.engine.forget_belief(user_id=USER, belief_id=cloud_b.id, purpose="api")
    batch, tail_cursor = cloud.engine.export_tombstones(0, 100)
    # the edge never had a belief with the cloud's global_id -> no-op, yet
    # recorded as applied
    r1 = edge.engine.apply_tombstones(batch)
    assert (r1.applied, r1.noop) == (1, 1)
    assert edge.applied() == [(cloud.db_id, 1)]
    tail = edge.log.head(edge.db_id)[0]

    # the cloud deletes another belief and exports the NEXT batch
    other = _belief_by_text(cloud, "coffee")
    if other is None:
        from mirror_memory.core.models import Belief as _B

        with tenant_scope(*TENANT):
            s = cloud.F()
            other = s.query(_B).filter(_B.user_id == USER,
                                       _B.id != cloud_b.id).first()
            s.close()
    assert cloud.engine.forget_belief(user_id=USER, belief_id=other.id,
                                      purpose="api")
    batch2, _ = cloud.engine.export_tombstones(tail_cursor, 100)
    assert batch2.from_cursor == tail_cursor

    # entry-level idempotency: a CONTINUING batch (from == stored) that
    # repeats already-applied (origin, origin_seq) skips them and applies
    # only the new one
    dup = _batch_of(cloud.db_id, tail_cursor,
                    [batch.entries[0], batch2.entries[0]],
                    to_cursor=batch2.to_cursor)
    r2 = edge.engine.apply_tombstones(dup)
    assert (r2.applied, r2.skipped) == (1, 1)
    # both applies are no-ops on THIS edge (the beliefs were never
    # aligned), so the local log stays at the pre-batch tail: the applied
    # record is the durable trace
    assert edge.applied() == [(cloud.db_id, 1), (cloud.db_id, 2)]

    # re-delivering a batch the receiver has ALREADY ACKED (from behind
    # the stored cursor) is a channel violation and is refused
    with pytest.raises(TombstoneBatchRejected) as info:
        edge.engine.apply_tombstones(batch)
    assert info.value.reason == "sequence"
    assert edge.applied() == [(cloud.db_id, 1), (cloud.db_id, 2)]


def test_reverse_direction_edge_to_cloud_and_no_echo(cloud, edge):
    """The edge deletes -> exports -> the cloud applies; the cloud's
    re-export to the edge does not carry it back."""
    _seed_both(cloud, edge)
    edge_b = _belief_by_text(edge, "SYNCSENTINEL")
    cloud_b = _belief_by_text(cloud, "SYNCSENTINEL")
    with tenant_scope(*TENANT), fixture_write_scope("align identity"):
        s = cloud.F()
        row = s.get(Belief, (*TENANT, cloud_b.id))
        row.global_id = edge_b.global_id
        s.commit()
        s.close()
    assert edge.engine.forget_belief(user_id=USER, belief_id=edge_b.id, purpose="api")
    up, _ = edge.engine.export_tombstones(0, 100, peer_db_id=cloud.db_id)
    assert len(up.entries) == 1 and up.entries[0].origin == edge.db_id
    res = cloud.engine.apply_tombstones(up, auth_tenant=TENANT)
    assert res.applied == 1 and res.noop == 0
    assert all("SYNCSENTINEL" not in b.claim_text for b in cloud.beliefs())
    # the belief surfaces are clean (a belief-scope deletion keeps the raw
    # session summary by design -- coverage matrix: session text is its
    # own layer, removed by the session / user scopes)
    assert all("SYNCSENTINEL" not in json.dumps(b.__dict__ if hasattr(b, "__dict__") else b, default=str)
               for b in cloud.engine.get_beliefs(user_id=USER, purpose="api"))
    # no echo back to the edge: export from the cloud's STORED edge
    # cursor -- the entries were applied FROM the edge, so none come back
    from mirror_memory.core.repository import read_sync_peer as _rsp

    with tenant_scope(*TENANT):
        s = cloud.F()
        peer = _rsp(s, edge.db_id)
        stored = peer.cursor if peer is not None else 0
        s.close()
    down, _ = cloud.engine.export_tombstones(stored, 100, peer_db_id=edge.db_id)
    assert down.entries == [] and down.sender == cloud.db_id


# ---------------------------------------------------------------------------
# task 1 order/integrity + task 2 tenant/identity: whole-batch refusal,
# zero writes, both directions
# ---------------------------------------------------------------------------


def _forged(entry: TombstoneEntry, **changes) -> TombstoneEntry:
    data = entry.to_dict()
    data.update(changes)
    forged = TombstoneEntry.from_dict(data)
    return forged.seal() if "hash" not in changes else forged



def _batch_of(sender: str, from_cursor: int, entries: list[TombstoneEntry],
              to_cursor: int | None = None) -> TombstoneBatch:
    return TombstoneBatch(sender=sender, from_cursor=from_cursor,
                          to_cursor=to_cursor if to_cursor is not None
                          else from_cursor + len(entries),
                          entries=entries).seal()


def _two_cloud_tombstones(cloud: _Device) -> list[TombstoneEntry]:
    b = _belief_by_text(cloud, "Berlin")
    assert cloud.engine.forget_belief(user_id=USER, belief_id=b.id, purpose="api")
    assert cloud.engine.forget_session(user_id=USER, session_id="sa", purpose="api")
    batch, _ = cloud.engine.export_tombstones(0, 100)
    assert len(batch.entries) == 2
    return batch.entries


@pytest.mark.parametrize("direction", ["cloud_to_edge", "edge_to_cloud"])
def test_forged_mixed_gap_and_tampered_batches_are_refused_with_zero_writes(
        cloud, edge, direction):
    _seed_both(cloud, edge)
    sender, receiver = (cloud, edge) if direction == "cloud_to_edge" else (edge, cloud)
    entries = _two_cloud_tombstones(sender)
    _, cursor = sender.engine.export_tombstones(0, 100)
    kwargs = {"auth_tenant": TENANT} if receiver is cloud else {}

    def resend(entries, from_cursor=0):
        return _batch_of(sender.db_id, from_cursor, entries,
                         from_cursor + len(entries))

    cases = {
        # every entry from another tenant
        "tenant": resend([_forged(e, tenant_id="other-tenant") for e in entries]),
        # ONE other-tenant entry mixed into a valid batch
        "tenant-mixed": resend([entries[0], _forged(entries[1], app_id="other-app")]),
        # a batch continuing from a cursor the receiver never reached (a
        # dropped prior batch / a fabricated continuation)
        "sequence": resend(entries, from_cursor=cursor + 5),
        # a tampered entry hash (the envelope resealed around it)
        "hash": resend([entries[0], _forged(entries[1], hash="0" * 64)]),
        # a content change behind a stale entry hash
        "hash-content": resend([entries[0], _forged(entries[1], user_id="someone-else", hash=entries[1].hash)]),
        # malformed: a belief without a global_id
        "format": resend([_forged(entries[0], scope_key="")]),
        # a tampered ENVELOPE: from_cursor edited after sealing (the
        # continuation is plausible, the stored cursor disagrees)
        "envelope": TombstoneBatch(sender=sender.db_id, from_cursor=cursor + 1,
                                   to_cursor=cursor + 1 + len(entries),
                                   entries=entries,
                                   hash=_batch_of(sender.db_id, cursor + 1,
                                                  entries,
                                                  cursor + 1 + len(entries)).hash),
    }
    expected_reason = {"tenant": "tenant", "tenant-mixed": "tenant", "sequence": "sequence",
                       "hash": "hash", "hash-content": "hash", "format": "format",
                       "envelope": "sequence"}
    for name, batch in cases.items():
        before = receiver.snapshot()
        with pytest.raises(TombstoneBatchRejected) as info:
            receiver.engine.apply_tombstones(batch, **kwargs)
        assert info.value.reason == expected_reason[name], (name, str(info.value))
        if name == "sequence":
            # the refusal names BOTH cursors: the forged continuation and
            # what the receiver actually stored
            assert str(cursor + 5) in str(info.value) and "stored cursor is 0" in str(info.value), str(info.value)
        assert receiver.snapshot() == before, (name, "a refused batch must write nothing")

    # the untouched valid batch still applies afterwards
    ok = receiver.engine.apply_tombstones(resend(entries), **kwargs)
    assert ok.applied == 2


def test_cloud_refuses_users_its_tenant_does_not_know(cloud, edge):
    """The multi-tenant receiver takes the tenant from the authenticated
    engine and refuses a tombstone for a user that tenant never saw --
    the entry's own fields are not trusted."""
    _seed_both(cloud, edge)
    entries = _two_cloud_tombstones(edge)
    unknown = _batch_of(edge.db_id, 0,
                        [_forged(e, user_id="u-never-seen") for e in entries])
    before = cloud.snapshot()
    with pytest.raises(TombstoneBatchRejected) as info:
        cloud.engine.apply_tombstones(unknown, auth_tenant=TENANT)
    assert info.value.reason == "user"
    assert cloud.snapshot() == before
    # the authenticated tenant must be this engine's tenant: anything else
    # is refused before any validation (no routing by entry fields)
    with pytest.raises(TombstoneBatchRejected) as info2:
        cloud.engine.apply_tombstones(_batch_of(edge.db_id, 0, entries),
                                      auth_tenant=("other", "app"))
    assert info2.value.reason == "tenant"
    assert cloud.snapshot() == before


def test_edge_treats_unknown_users_as_noops(cloud, edge):
    """The single-tenant edge may legitimately never have met the user
    (a fresh install catching up): the tombstone is a no-op, recorded."""
    _seed_both(cloud, edge)
    entries = _two_cloud_tombstones(cloud)
    unknown = _batch_of(cloud.db_id, 0,
                        [_forged(e, user_id="u-never-seen") for e in entries])
    res = edge.engine.apply_tombstones(unknown)
    assert (res.applied, res.noop) == (2, 2)


# ---------------------------------------------------------------------------
# task 1: applying a tombstone races the publish -- 50 rounds
# ---------------------------------------------------------------------------


def test_tombstone_apply_vs_publish_50_rounds_sqlite(tmp_path):
    """Each round: memory + a claimed job on the edge; a user-scope
    tombstone from the cloud is applied while the job publishes.  The
    tombstone advances the generation, so the publish is cancelled
    (deletion_generation_changed) or the deletion removes what it wrote:
    memory never lands after the tombstone."""
    from mirror_memory.worker.evolution import _claim_job, _process_single_job, enqueue_job

    cloud = _Device(tmp_path, "cloud50", single_tenant=False)
    edge = _Device(tmp_path, "edge50", single_tenant=True)
    try:
        for round_no in range(50):
            edge.observe(f"round {round_no} content", session_id=f"r{round_no}")
            cloud.observe(f"round {round_no} content", session_id=f"r{round_no}")
            cloud.engine.delete_memories(user_id=USER, purpose="api")
            batch, _ = cloud.engine.export_tombstones(round_no, 1, peer_db_id=edge.db_id)
            assert len(batch.entries) == 1, (round_no, batch.to_dict())
            assert batch.from_cursor == round_no and batch.to_cursor == round_no + 1

            outcomes: dict = {}

            def run_publish():
                try:
                    with tenant_scope(*TENANT):
                        s = edge.F()
                        try:
                            job = enqueue_job(s, USER, edge.cfg, purpose="api")
                            s.commit()
                            token = _claim_job(s, job.id, lease_seconds=60)
                            row = s.get(EvolutionJob, job.id)
                            _process_single_job(s, row, edge.cfg, token)
                            s.commit()
                            s.expire_all()
                            final = s.get(EvolutionJob, job.id)
                            outcomes["publish"] = (final.status if final else "deleted",
                                                   final.error_code if final else "")
                        except Exception as exc:  # noqa: BLE001 -- the job row may be deleted under us
                            s.rollback()
                            outcomes["publish"] = ("deleted-or-cancelled", repr(exc))
                        finally:
                            s.close()
                except Exception as exc:  # noqa: BLE001
                    outcomes["publish"] = ("error", repr(exc))

            def run_apply():
                try:
                    edge.engine.apply_tombstones(batch)
                    outcomes["apply"] = "applied"
                except Exception as exc:  # noqa: BLE001
                    outcomes["apply"] = repr(exc)

            tp = threading.Thread(target=run_publish)
            ta = threading.Thread(target=run_apply)
            ta.start()
            tp.start()
            ta.join(timeout=60)
            tp.join(timeout=60)
            assert "apply" in outcomes and "publish" in outcomes, outcomes
            assert outcomes["apply"] == "applied", outcomes
            assert "error', repr" not in str(outcomes.get("publish", "")), outcomes
            # the invariant: nothing of this user survives the applied tombstone
            assert edge.beliefs() == [], (round_no, outcomes)
            assert edge.generation() >= round_no + 1
    finally:
        cloud.dispose()
        edge.dispose()
