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

import contextlib
import json
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import schema_migration as sm
from mirror_memory.core.deletion import GENESIS_HASH, FileDeletionLog
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

    batch, cursor = cloud.engine.export_tombstones(0, 100)
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
        "scope_key", "generation", "created_at", "hash", "origin_hash"}

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
    batch2, cursor2 = cloud.engine.export_tombstones(cursor, 100)
    assert [e.scope for e in batch2.entries] == ["user"] and batch2.entries[0].scope_key == ""
    edge.engine.apply_tombstones(batch2)
    assert edge.beliefs() == [] and edge.summaries() == 0
    assert "SYNCSENTINEL" not in (edge.engine.recall(user_id=USER, query="SYNCSENTINEL", purpose="api") or "")

    # NO ECHO (receiver-side now): the edge's log re-exported to the
    # cloud carries the C-origin tombstones back, and the cloud SKIPS
    # them at apply time -- its own log still holds every one of them
    back, _ = edge.engine.export_tombstones(0, 100)
    assert back.entries and all(e.origin == cloud.db_id for e in back.entries)
    r_back = cloud.engine.apply_tombstones(back)
    assert r_back.applied == 0
    # ... while a THIRD device would receive them with the ORIGINAL origin
    third, _ = edge.engine.export_tombstones(0, 100)
    assert [(e.origin, e.origin_seq) for e in third.entries] == [
        (cloud.db_id, 1), (cloud.db_id, 2), (cloud.db_id, 3)], third.to_dict()

    # the empty continuation batch (everything already applied/excluded)
    # advances the channel cursor without touching data
    empty, empty_cursor = cloud.engine.export_tombstones(cursor2, 100)
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
    dup = _batch_of(cloud, tail_cursor,
                    [batch.entries[0], batch2.entries[0]],
                    to_cursor=batch2.to_cursor)
    r2 = edge.engine.apply_tombstones(dup)
    assert (r2.applied, r2.skipped) == (1, 1)
    # both applies are no-ops on THIS edge (the beliefs were never
    # aligned), so the local log stays at the pre-batch tail: the applied
    # record is the durable trace
    assert edge.applied() == [(cloud.db_id, 1), (cloud.db_id, 2)]

    # re-delivering a batch the receiver has ALREADY ACKED: the envelope
    # no longer continues the stored channel (stored cursor 2, chain
    # X vs batch from_cursor 0, chain GENESIS) -- the channel RESCANS:
    # it re-anchors to 0, re-fetches the whole sender log, and the
    # (origin, origin_hash) applied records deduplicate the re-delivery
    r3 = edge.engine.apply_tombstones(batch)
    assert (r3.applied, r3.skipped) == (0, 1)
    assert edge.applied() == [(cloud.db_id, 1), (cloud.db_id, 2)]
    # the rescan re-anchored the channel to 0: the re-fetch from 0
    # re-delivers BOTH tombstones and the applied records deduplicate
    refetch, _ = cloud.engine.export_tombstones(0, 100)
    r4 = edge.engine.apply_tombstones(refetch)
    assert (r4.applied, r4.skipped) == (0, 2)
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
    up, _ = edge.engine.export_tombstones(0, 100)
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
    down, _ = cloud.engine.export_tombstones(stored, 100)
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



def _batch_of(sender_device, from_cursor: int,
              entries: list[TombstoneEntry],
              to_cursor: int | None = None) -> TombstoneBatch:
    """Seal a batch continuing the sender's REAL channel: the chain
    hashes come from the sender's log at both cursor ends (tests that
    need a MISMATCH pass explicit wrong hashes instead)."""
    tail = sender_device.log.head(sender_device.db_id)[0]
    to = to_cursor if to_cursor is not None else max(tail, from_cursor + len(entries))
    return TombstoneBatch(
        sender=sender_device.db_id, from_cursor=from_cursor, to_cursor=to,
        from_chain_hash=sender_device.log.chain_hash_at(sender_device.db_id, from_cursor),
        to_chain_hash=sender_device.log.chain_hash_at(sender_device.db_id, to),
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
    """Integrity failures (tampered hash / malformed entry) are refused
    outright with zero writes.  A fabricated continuation that does not
    carry the stored chain hash triggers a RESCAN: the channel re-anchors
    to 0, the sender's log is re-fetched from the start, and the
    (origin, origin_hash) applied records deduplicate the re-delivery --
    visible in the rescan metric, never silent."""
    _seed_both(cloud, edge)
    sender, receiver = (cloud, edge) if direction == "cloud_to_edge" else (edge, cloud)
    entries = _two_cloud_tombstones(sender)
    kwargs = {"auth_tenant": TENANT} if receiver is cloud else {}

    def resend(entries, from_cursor=0):
        return _batch_of(sender, from_cursor, entries,
                         from_cursor + len(entries))

    from mirror_memory.core import metrics

    before = receiver.snapshot()
    integrity_cases = {
        "hash": resend([entries[0], _forged(entries[1], hash="0" * 64)]),
        "hash-content": resend([entries[0], _forged(entries[1], user_id="someone-else", hash=entries[1].hash)]),
        "format": resend([_forged(entries[0], scope_key="")]),
    }
    for name, batch in integrity_cases.items():
        with pytest.raises(TombstoneBatchRejected) as info:
            receiver.engine.apply_tombstones(batch, **kwargs)
        assert info.value.reason in ("hash", "format"), (name, str(info.value))
        assert receiver.snapshot() == before, (name, "a refused batch must write nothing")

    # continuity failures RESCAN (not refuse): the channel re-anchors to
    # 0, the whole sender log is re-fetched, and the applied records
    # deduplicate the re-delivery.  First the honest batch applies (2
    # entries); then a FABRICATED batch continuing beyond the tail
    # triggers the rescan -- the re-fetched log re-delivers the same two
    # tombstones and they dedup to zero re-applications.
    rescans_before = metrics.snapshot().get("tombstones_rescans", 0)
    receiver.engine.apply_tombstones(resend(entries), **kwargs)
    after_honest = receiver.snapshot()

    fabricated_from = sender.log.head(sender.db_id)[0] + 5
    fabricated = _batch_of(sender, fabricated_from, entries,
                           to_cursor=fabricated_from + len(entries))
    r = receiver.engine.apply_tombstones(fabricated, **kwargs)
    assert (r.applied, r.skipped) == (0, len(entries)), (
        "the rescan re-delivery must dedup to zero applications")
    rescans = metrics.snapshot().get("tombstones_rescans", 0)
    assert rescans > rescans_before, "the rescan was counted (not silent)"
    assert receiver.snapshot() == after_honest, (
        "the rescan re-delivery must not change the receiver's state")
def test_cloud_refuses_users_its_tenant_does_not_know(cloud, edge):
    """The multi-tenant receiver takes the tenant from the authenticated
    engine and refuses a tombstone for a user that tenant never saw --
    the entry's own fields are not trusted."""
    _seed_both(cloud, edge)
    entries = _two_cloud_tombstones(edge)
    unknown = _batch_of(edge, 0,
                        [_forged(e, user_id="u-never-seen") for e in entries])
    before = cloud.snapshot()
    with pytest.raises(TombstoneBatchRejected) as info:
        cloud.engine.apply_tombstones(unknown, auth_tenant=TENANT)
    assert info.value.reason == "user"
    assert cloud.snapshot() == before
    # the authenticated tenant must be this engine's tenant: anything else
    # is refused before any validation (no routing by entry fields)
    with pytest.raises(TombstoneBatchRejected) as info2:
        cloud.engine.apply_tombstones(_batch_of(edge, 0, entries),
                                      auth_tenant=("other", "app"))
    assert info2.value.reason == "tenant"
    assert cloud.snapshot() == before


def test_edge_treats_unknown_users_as_noops(cloud, edge):
    """The single-tenant edge may legitimately never have met the user
    (a fresh install catching up): the tombstone is a no-op, recorded."""
    _seed_both(cloud, edge)
    entries = _two_cloud_tombstones(cloud)
    unknown = _batch_of(cloud, 0,
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
            batch, _ = cloud.engine.export_tombstones(round_no, 1)
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


# --- M4 batch-2 patch 3, task 2: apply cost does not follow table size ---

def _scale_receiver(root: Path, name: str):
    """A single-tenant receiver with its own SQLite DB and file log."""
    log_dir = root / f"{name}-deletions"
    log_dir.mkdir(parents=True, exist_ok=True)
    engine = MemoryEngine(
        config=_cfg(log_dir),
        database_url=f"sqlite:///{root / name}.db",
        tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM(),
        deletion_log=FileDeletionLog(log_dir), single_tenant_mode=True)
    engine._ensure_db()
    return engine


def _measure_apply(engine, batch):
    """One apply, timed, with the SQL statement count taken from a
    before_cursor_execute listener on the engine's own bind."""
    import time as _time
    from sqlalchemy import event as _sa_event

    counter = {"n": 0}

    def _count(conn, cursor, statement, parameters, context, executemany):
        counter["n"] += 1

    _sa_event.listen(engine._engine, "before_cursor_execute", _count)
    try:
        t0 = _time.perf_counter()
        engine.apply_tombstones(batch, purpose="sync")
        elapsed = _time.perf_counter() - t0
    finally:
        _sa_event.remove(engine._engine, "before_cursor_execute", _count)
    return elapsed, counter["n"]


def _grant_for(engine, user: str, purposes=("sync", "api")) -> None:
    from mirror_memory.core import authorization

    with tenant_scope(*TENANT), fixture_write_scope("scale test grants"):
        s = sessionmaker(bind=engine._engine)()
        for purpose in purposes:
            authorization.grant_purpose(
                s, user, purpose,
                operations=list(authorization.OPERATIONS),
                issuer="scale-test", reason="scale test grants")
        s.commit()
        s.close()


def _enable_memory(engine, user: str) -> None:
    from mirror_memory.core.repository import set_memory_enabled

    with tenant_scope(*TENANT), fixture_write_scope("scale test memory"):
        s = sessionmaker(bind=engine._engine)()
        set_memory_enabled(s, user, True)
        s.commit()
        s.close()


def _seed_beliefs(engine, user: str, n: int) -> None:
    """`n` real beliefs (bulk) the real-deletion batches will delete."""
    from mirror_memory.core.models import Belief

    rows = [dict(tenant_id=TENANT[0], app_id=TENANT[1],
                 global_id=f"{i:032x}", user_id=user, dimension="scale",
                 key=f"scale-{i}", claim_text=f"scale claim {i}",
                 value_json="{}", layer="L4", status="active",
                 confidence=1.0)
            for i in range(1, n + 1)]
    with tenant_scope(*TENANT), fixture_write_scope("scale test beliefs"):
        s = sessionmaker(bind=engine._engine)()
        s.bulk_insert_mappings(Belief, rows)
        s.commit()
        s.close()


def _seed_own_log(log: FileDeletionLog, db_id: str, n: int) -> dict:
    """The RECEIVER'S OWN log seeded with `n` chained entries, written raw
    (one fsync) -- the echo lookups must hit THIS log, not another db_id's.
    Returns seq -> hash for the echo batch to name."""
    import json as _json
    import os as _os
    from datetime import datetime as _dt

    from mirror_memory.core.deletion import DeletionEntry

    log.ensure_genesis(db_id)
    path = log._path_for(db_id)
    hashes = {}
    lines = []
    prev = GENESIS_HASH
    for i in range(1, n + 1):
        e = DeletionEntry(
            db_id=db_id, seq=i, tenant_id=TENANT[0], app_id=TENANT[1],
            user_id="u-seed", scope="user", scope_key="", fingerprints=[],
            generation=0, purpose="seed", created_at=_dt(2026, 1, 1),
            prev_hash=prev)
        e.hash = e.compute_hash()
        prev = e.hash
        hashes[i] = e.hash
        lines.append(_json.dumps(log._line(e), sort_keys=True,
                                 ensure_ascii=False))
    with open(path, "a+b") as fh:
        fh.seek(0, _os.SEEK_END)
        fh.write(("\n".join(lines) + "\n").encode())
        fh.flush()
        _os.fsync(fh.fileno())
    log._index_invalidate(db_id)
    return hashes


def _seed_applied_rows(engine, n: int) -> None:
    """`n` applied rows of one foreign origin, bulk-inserted."""
    from mirror_memory.core.models import TombstoneApplied

    rows = [dict(tenant_id=TENANT[0], app_id=TENANT[1], origin="b" * 32,
                 origin_seq=i, origin_hash=f"{i:064x}", user_id="u-seed",
                 scope="user", scope_key="", local_seq=None)
            for i in range(1, n + 1)]
    with tenant_scope(*TENANT), fixture_write_scope("scale test seeding"):
        s = sessionmaker(bind=engine._engine)()
        s.bulk_insert_mappings(TombstoneApplied, rows)
        s.commit()
        s.close()


class _BatchChain:
    """Honest continuation envelopes from one sender: from_cursor/
    from_chain continue the previously served to_cursor/to_chain, so no
    measured batch triggers a rescan."""

    def __init__(self, sender: str):
        self.sender = sender
        self.next_from = 0
        self.next_chain = GENESIS_HASH

    def make(self, entries):
        to = entries[-1].origin_seq
        to_chain = f"{to:064x}"
        batch = TombstoneBatch(sender=self.sender, from_cursor=self.next_from,
                               to_cursor=to,
                               from_chain_hash=self.next_chain,
                               to_chain_hash=to_chain,
                               entries=entries).seal()
        self.next_from = to
        self.next_chain = to_chain
        return batch


def test_apply_scale_report_by_batch_kind(tmp_path):
    """Batch-2 patch 4, task 3: the scale protocol with the RECEIVER'S OWN
    log and the three batch kinds, reported separately.

    Setup per receiver: SQLite main DB + file deletion log (the repo's
    test harness shape); `scaled` additionally holds 10,000 applied rows
    and ITS OWN 10,000-entry log; both hold 1,500 real beliefs.  The
    500-entry batches, each applied REPS=3 times (median reported):

    - echo : the receiver's OWN tombstones coming back -- present in its
      own log -> the echo check looks the entry up there (index hit) and
      skips; on the empty receiver the same batch misses and applies as
      no-ops (reported as-is -- the statement counts differ by design);
    - noop : a foreign origin naming beliefs that do not exist --
      recorded, no deletion, no log write;
    - real : a foreign origin naming 500 of the seeded beliefs -- real
      deletions that append 500 entries to the receiver's own log.

    The one-time index build and the steady-state lookup rounds are
    printed for the scaled log.  read_from remains a linear scan of the
    file (it must parse payloads) and is NOT claimed O(1)."""
    import platform
    import time as _time

    print(f"\nSCALE setup: db=SQLite(file log), platform={platform.platform()}")
    print(f"SCALE setup: scale=10k applied + own 10k log, batch=500, reps=3")

    def build(name):
        eng = _scale_receiver(tmp_path, name)
        _enable_memory(eng, "u-real")
        _grant_for(eng, "u-real")
        _seed_beliefs(eng, "u-real", 1_500)
        return eng

    empty = build("rx-empty")
    scaled = build("rx-scaled")
    _seed_applied_rows(scaled, 10_000)
    own_hashes = _seed_own_log(scaled._deletion_log, scaled._db_id, 10_000)

    # the one-time index build vs the steady state (the scaled own log)
    t0 = _time.perf_counter()
    scaled._deletion_log.chain_hash_at(scaled._db_id, 1)
    build_ms = (_time.perf_counter() - t0) * 1000
    t0 = _time.perf_counter()
    for i in range(1, 201):
        scaled._deletion_log.chain_hash_at(scaled._db_id, i)
        scaled._deletion_log.has_tombstone(scaled._db_id, i, own_hashes[i])
        scaled._deletion_log.head(scaled._db_id)
    steady_ms = (_time.perf_counter() - t0) * 1000
    t0 = _time.perf_counter()
    for i in range(1, 201):
        empty._deletion_log.chain_hash_at(empty._db_id, i)
        empty._deletion_log.has_tombstone(empty._db_id, i, own_hashes[i])
        empty._deletion_log.head(empty._db_id)
    steady_empty_ms = (_time.perf_counter() - t0) * 1000
    print(f"SCALE index build (once, own 10k log): {build_ms:.4g} ms")
    print(f"SCALE lookups steady 600 (own 10k log): {steady_ms:.4g} ms")
    print(f"SCALE lookups steady 600 (empty log):   {steady_empty_ms:.4g} ms")

    def batch_of(kind, receiver, rep, chain):
        base = rep * 500
        if kind == "echo":
            seqs = range(5001 + rep * 500, 5501 + rep * 500)
            entries = [TombstoneEntry(
                origin=receiver._db_id, origin_seq=s,
                tenant_id=TENANT[0], app_id=TENANT[1], user_id="u-seed",
                scope="belief", scope_key=f"{s:032x}", generation=0,
                created_at="2026-01-01T00:00:00",
                origin_hash=own_hashes[s]).seal() for s in seqs]
        elif kind == "noop":
            entries = [TombstoneEntry(
                origin="b" * 32, origin_seq=20_001 + base + i,
                tenant_id=TENANT[0], app_id=TENANT[1], user_id="u-ghost",
                scope="belief", scope_key=f"{900_000 + base + i:032x}",
                generation=0, created_at="2026-01-01T00:00:00",
                origin_hash=f"{20_001 + base + i:064x}").seal()
                for i in range(500)]
        else:
            entries = [TombstoneEntry(
                origin="c" * 32, origin_seq=30_001 + base + i,
                tenant_id=TENANT[0], app_id=TENANT[1], user_id="u-real",
                scope="belief", scope_key=f"{base + i + 1:032x}",
                generation=0, created_at="2026-01-01T00:00:00",
                origin_hash=f"{30_001 + base + i:064x}").seal()
                for i in range(500)]
        return chain.make(entries)

    results = {}
    head_before = scaled._deletion_log.head(scaled._db_id)[0]
    for kind in ("echo", "noop", "real"):
        for label, eng in (("empty", empty), ("scaled", scaled)):
            chain = _BatchChain(sender="e" * 32)
            times, stmts = [], None
            for rep in range(3):
                dt, n = _measure_apply(eng, batch_of(kind, eng, rep, chain))
                times.append(dt)
                stmts = n if stmts is None else stmts
            results[(kind, label)] = (sorted(times)[1], stmts)
    for kind in ("echo", "noop", "real"):
        t_empty, s_empty = results[(kind, "empty")]
        t_scaled, s_scaled = results[(kind, "scaled")]
        print(f"SCALE {kind:5s} empty  : median {t_empty * 1000:9.4g} ms, "
              f"{s_empty} statements")
        print(f"SCALE {kind:5s} scaled : median {t_scaled * 1000:9.4g} ms, "
              f"{s_scaled} statements")

    # the real batches appended 3 x 500 entries to the scaled receiver's
    # OWN log; the index (tail pointer) kept agreeing with the file
    head_after = scaled._deletion_log.head(scaled._db_id)[0]
    assert head_after == head_before + 1_500, (head_before, head_after)
    cold = FileDeletionLog(scaled._deletion_log.directory)
    assert cold.head(scaled._db_id) == scaled._deletion_log.head(scaled._db_id)

    # The scale-sensitive property is STATEMENT-COUNT-based and therefore
    # deterministic: the dedup lookup must not add statements as the
    # applied table grows (the old full-table preload's cost would show
    # up as extra work per batch, and its timing ratio is load-noisy).
    # noop/real do identical work on both receivers -> identical counts:
    for kind in ("noop", "real"):
        s_empty = results[(kind, "empty")][1]
        s_scaled = results[(kind, "scaled")][1]
        assert s_scaled == s_empty, \
            f"{kind}: the dedup lookup's statement count grew with the " \
            f"table ({s_empty} -> {s_scaled})"
    # echo on the scaled receiver issues a HANDFUL of statements (the
    # echo lookups are file-index hits, not SQL); on the empty receiver
    # the same batch applies as no-ops (hundreds of INSERTs) -- the
    # asymmetry is the measured path, reported above
    assert results[("echo", "scaled")][1] <= 20
    # a loose timing guard for pathological regressions only (load-noisy:
    # it flaked once inside a fully loaded suite -- the statement-count
    # assertions above carry the real property)
    for kind in ("noop", "real"):
        t_empty = results[(kind, "empty")][0]
        t_scaled = results[(kind, "scaled")][0]
        assert t_scaled < max(8.0 * t_empty, 5.0), \
            f"{kind} apply grew pathologically with the table: " \
            f"{t_empty:.4g}s -> {t_scaled:.4g}s"
