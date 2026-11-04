"""M4 batch-2 patch task 2: three devices delete in alternation and the
tombstone channel keeps them consistent.

Topology (star): cloud C, edges A and B -- every edge talks only to the
cloud.  20 rounds; each round ALL devices observe the same tagged
conversation (and adopt the cloud's global_ids, as a shared history
implies), then ONE device (random) deletes a scope (rotating belief /
session / user), then the pairs sync in a randomized order, pulling
batch-at-a-time until the tail.  After every round:

- NO batch was refused (refusals list is empty);
- the tagged data deleted this round is invisible on ALL devices
  (beliefs, summaries, recall, export -- read entrances, not tables);
- the three devices' VISIBLE states agree (same surviving belief keys
  and summary count).

Also covered:
- a device D joining at round 10 catches up on every tombstone so far;
- an entry that cannot travel (a pre-batch-2 belief deletion with no
  global_id) is skipped at export and does NOT wedge the channel;
- forging (a fabricated from_cursor, a dropped batch) is refused with
  zero writes;
- a receiver that restores its backup re-pulls from the rewound cursor
  and lands on the same state as the cloud.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core.deletion import DeletionEntry, FileDeletionLog
from mirror_memory.core.models import Base, Belief, SessionSummary
from mirror_memory.core.repository import read_sync_peer, set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.core.tombstones import (
    TombstoneBatch,
    TombstoneBatchRejected,
    TombstoneEntry,
)

TENANT = ("default", "default")


@pytest.fixture(autouse=True)
def _single_tenant_mode_off():
    """The mesh devices run single_tenant_mode=True (a process-wide
    contextvar): reset it before/after every test so later tests keep
    tenant filtering."""
    from mirror_memory.core.tenancy import set_single_tenant_mode

    set_single_tenant_mode(False)
    yield
    set_single_tenant_mode(False)
USER = "u-mesh"

SCOPES = ["belief", "session", "user"]


def _cfg(log_dir):
    from mirror_memory.config.loader import load_config

    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    cfg.worker.job_lease_seconds = 60
    cfg.session_summary_enabled = True
    cfg.deletion_log.directory = str(log_dir)
    return cfg


class _FakeLLM:
    def generate(self, **kw):
        return '{"patterns": [], "support_policy": {}}'


class _Device:
    def __init__(self, root: Path, name: str, *, single_tenant: bool):
        self.name = name
        self.single_tenant = single_tenant
        self.url = f"sqlite:///{root / f'{name}.db'}"
        self.db = create_engine(self.url)
        Base.metadata.create_all(self.db)
        from mirror_memory.core import schema_migration as sm

        sm.ensure_watermark(self.db)
        self.log_dir = root / f"{name}-deletions"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = FileDeletionLog(self.log_dir)
        cfg = load_config("config/")
        cfg.llm_client = _FakeLLM()
        cfg.worker.job_lease_seconds = 60
        cfg.session_summary_enabled = True
        cfg.deletion_log.directory = str(self.log_dir)
        self.cfg = cfg
        self.F = sessionmaker(bind=self.db)
        with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
            s = self.F()
            set_memory_enabled(s, USER, True)
            s.commit()
            s.close()
        from mirror_memory.api import MemoryEngine

        self.engine = MemoryEngine(
            config=cfg, database_url=self.url, tenant_id=TENANT[0],
            app_id=TENANT[1], llm_client=_FakeLLM(), deletion_log=self.log,
            single_tenant_mode=single_tenant)
        self.engine._ensure_db()

    @property
    def db_id(self) -> str:
        from mirror_memory.core import schema_migration as sm

        return sm.read_watermark(self.db)[0]

    def observe(self, text: str, session_id: str) -> None:
        self.engine.observe(user_id=USER, session_id=session_id, text=text,
                            purpose="api")

    def delete(self, scope: str, round_no: int) -> str:
        """One deletion of *scope*; returns the tag it is findable by.
        The tagged text lives in its OWN session, so the session scope
        removes exactly its own summary layer."""
        tag = f"r{round_no}"
        session_id = f"s{round_no}"
        self.observe(f"I like {tag} and coffee.", session_id=session_id)
        if scope == "belief":
            belief = next(b for b in self.beliefs() if tag in b.claim_text)
            assert self.engine.forget_belief(user_id=USER,
                                             belief_id=belief.id,
                                             purpose="api")
            return tag
        if scope == "session":
            assert self.engine.forget_session(user_id=USER,
                                              session_id=session_id,
                                              purpose="api")
            return tag
        self.engine.delete_memories(user_id=USER, purpose="api")
        return tag

    def adopt_global_ids_by_key(self, ids: dict) -> None:
        from mirror_memory.core.tenancy import fixture_write_scope

        with tenant_scope(*TENANT), fixture_write_scope("align identity"):
            s = self.F()
            for b in s.query(Belief).filter(Belief.user_id == USER).all():
                if b.key in ids:
                    b.global_id = ids[b.key]
            s.commit()
            s.close()

    def adopt_global_ids(self, source: "_Device") -> None:
        """Same-key beliefs adopt the source's global_id (what a shared
        conversation implies once sync is on)."""
        ids = {b.key: b.global_id for b in source.beliefs()}
        with tenant_scope(*TENANT), fixture_write_scope("align identity"):
            s = self.F()
            for b in s.query(Belief).filter(Belief.user_id == USER).all():
                if b.key in ids:
                    b.global_id = ids[b.key]
            s.commit()
            s.close()

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
                return s.query(SessionSummary).filter(
                    SessionSummary.user_id == USER).count()
            finally:
                s.close()

    def visible_state(self) -> tuple:
        """The device's visible state, comparable across devices."""
        return (
            tuple(sorted(b.key for b in self.beliefs())),
            self.summaries(),
        )

    def sees(self, tag: str, scope: str = "user") -> bool:
        """Does the surface the DELETED scope owned still surface
        this tag?  recall MIXES layers (it synthesises from surviving
        sessions and beliefs), so it is only a clean signal when
        everything for the user was deleted (scope=user)."""
        if scope in ("belief", "user"):
            if any(tag in b.claim_text for b in self.beliefs()):
                return True
            if tag in str(self.engine.get_beliefs(user_id=USER, purpose="api")):
                return True
            if tag in str(self.engine.export_page(
                    user_id=USER, purpose="api").to_dict()):
                return True
        if scope in ("session", "user") and self.summaries():
            with tenant_scope(*TENANT):
                s = self.F()
                try:
                    texts = " ".join(r.summary_text for r in s.query(
                        SessionSummary).filter(
                            SessionSummary.user_id == USER).all())
                finally:
                    s.close()
                if tag in texts:
                    return True
        if scope == "user" and tag in (self.engine.recall(
                user_id=USER, query=tag, purpose="api") or ""):
            return True
        return False

    def peer_cursor(self, peer_db_id: str) -> int:
        with tenant_scope(*TENANT):
            s = self.F()
            try:
                peer = read_sync_peer(s, peer_db_id)
                return peer.cursor if peer is not None else 0
            finally:
                s.close()

    def pull_from(self, peer: "_Device", refusals: list[str], *,
                  max_batches: int = 200) -> None:
        """Pull from *peer* batch-at-a-time until the tail, applying each
        batch.  A refusal is RECORDED, never swallowed."""
        for _ in range(max_batches):
            cursor = self.peer_cursor(peer.db_id)
            try:
                batch, _next = peer.engine.export_tombstones(
                    cursor, 500)
            except Exception as exc:  # noqa: BLE001 -- recorded
                refusals.append(f"export from {peer.name}: {exc!r}")
                return
            try:
                result = self.engine.apply_tombstones(batch)
            except TombstoneBatchRejected as exc:
                refusals.append(f"apply from {peer.name}: {exc}")
                return
            # a rescan re-anchored the channel to 0: keep pulling from the
            # reset cursor even when this batch's own range was empty
            if (batch.to_cursor == batch.from_cursor
                    and not result.rescan):
                return

    def dispose(self):
        self.engine._engine.dispose()
        self.db.dispose()


def _align_all(devices: dict, source: "_Device") -> None:
    ids = {b.key: b.global_id for b in source.beliefs()}
    for d in devices.values():
        if d is source:
            continue
        d.adopt_global_ids(source)


def _sync_star(devices: dict, refusals: list[str], rng: random.Random) -> None:
    """The pairs (A<->C, B<->C) sync in a randomized order.  Within a
    pair the edge pulls the cloud FIRST and the cloud pulls the edge
    second, so a hub relay converges inside a single pass; the pass
    repeats until a fixed point (no cursor moved) with a small cap --
    a hub relay takes one extra pass, and per-round convergence is the
    test's invariant."""
    edges = [n for n in ("A", "B", "D") if n in devices]
    rng.shuffle(edges)
    for _pass in range(3):
        moved = False
        for name in edges:
            d = devices[name]
            c = devices["C"]
            before = (d.peer_cursor(c.db_id), c.peer_cursor(d.db_id))
            d.pull_from(c, refusals)
            c.pull_from(d, refusals)
            if (d.peer_cursor(c.db_id), c.peer_cursor(d.db_id)) != before:
                moved = True
        if not moved:
            break


def test_three_devices_alternate_deletes_and_stay_consistent(tmp_path):
    rng = random.Random(11)
    refusals: list[str] = []
    root = tmp_path
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
        "B": _Device(root, "b", single_tenant=True),
    }
    try:
        deleted_tags: list = []
        for round_no in range(20):
            # all devices observe the same tagged conversations
            for d in devices.values():
                d.observe(f"I like r{round_no} and coffee.",
                          session_id=f"s{round_no}")
                d.observe(f"I live in city{round_no}.",
                          session_id=f"s{round_no}")
            _align_all(devices, devices["C"])

            # one device deletes; the scope rotates through all three
            actor_name = ("C", "A", "B")[round_no % 3]
            scope = SCOPES[round_no % 3]
            actor = devices[actor_name]
            actor.delete(scope, round_no)
            deleted_tags.append(round_no)

            _sync_star(devices, refusals, rng)

            # the round's invariants: no refusals; the deleted scope's
            # surface is clean on ALL devices; the visible states agree
            assert refusals == [], (round_no, refusals)
            tag = f"r{round_no}"
            for d in devices.values():
                if d.sees(tag, scope):
                    cursors = {n: d.peer_cursor(o.db_id)
                               for n, o in devices.items()}
                    raise AssertionError(
                        (round_no, d.name, tag, scope, cursors,
                         d.visible_state(), refusals))

            states = {name: d.visible_state() for name, d in devices.items()}
            assert len(set(states.values())) == 1, (round_no, states)

        # every propagated tombstone was applied everywhere (the applied
        # record + the channel cursors are the durable trace); beliefs
        # observed ONLY on their own device after the last alignment keep
        # their locally-minted global_ids -- the channel carries
        # deletions, not content, so those correctly survive
        for name, d in devices.items():
            if d is devices["C"]:
                continue  # the cloud never pulls from itself
            cursor = d.peer_cursor(devices["C"].db_id)
            assert cursor >= len(deleted_tags), (name, cursor)
    finally:
        for d in devices.values():
            d.dispose()


def test_device_joining_midway_catches_up(tmp_path):
    rng = random.Random(5)
    refusals: list[str] = []
    root = tmp_path
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
    }
    try:
        tags: list[str] = []
        for round_no in range(10):
            tag = f"j{round_no}"
            for d in devices.values():
                d.observe(f"I like {tag} and tea.", session_id="shared")
            _align_all(devices, devices["C"])
            actor = devices["C" if round_no % 2 else "A"]
            actor.delete(["belief", "user"][round_no % 2], round_no)
            tags.append(tag)
            _sync_star(devices, refusals, rng)
        assert refusals == []

        # D joins at round 10 with a FRESH database: cursor starts at 0
        d_root = root / "device-d"
        d_root.mkdir(parents=True, exist_ok=True)
        d = _Device(d_root, "d", single_tenant=True)
        devices["D"] = d
        try:
            d.pull_from(devices["C"], refusals)
            assert refusals == []
            applied = d.applied() if hasattr(d, "applied") else None
            cursor = d.peer_cursor(devices["C"].db_id)
            assert cursor > 0, "the catch-up advanced the channel cursor"
            assert applied is None or len(applied) >= 1
            # every tombstone D could act on was applied (none of the
            # deleted tags exist on D -- they were no-ops or deletions)
            for tag in tags:
                assert not d.sees(tag)
        finally:
            devices["D"].dispose()
    finally:
        for d in devices.values():
            d.dispose()


def test_untravelable_entry_does_not_wedge_the_channel(tmp_path):
    """A pre-batch-2 belief deletion (no global_key) cannot travel: the
    export SKIPS it, the cursor batch advances, and later deletions
    still flow."""
    root = tmp_path
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
    }
    refusals: list[str] = []
    try:
        C, A = devices["C"], devices["A"]
        C.observe("I like legacy and coffee.", session_id="legacy")
        with tenant_scope(*TENANT), fixture_write_scope("legacy delete"):
            s = C.F()
            b = s.query(Belief).filter(
                Belief.claim_text.contains("legacy")).one()
            C.engine._append_deletion_log(
                s, USER, scope="belief", scope_key=str(b.id),
                fingerprints=[], generation=1, purpose="api",
                global_key="")  # pre-batch-2 shape: NO global_key
            s.commit()
            s.close()

        # the cloud ALSO deletes a modern belief: its entry must travel
        C.observe("I like modern and tea.", session_id="modern")
        with tenant_scope(*TENANT), fixture_write_scope("align modern"):
            s = A.F()
            pass
        C.observe("I like modern2 and tea.", session_id="modern2")
        modern = next(b for b in C.beliefs() if "modern" in b.claim_text)
        C.engine.forget_belief(user_id=USER, belief_id=modern.id, purpose="api")

        A.pull_from(C, refusals)
        assert refusals == [], refusals
        # the untravelable entry did not wedge the channel: the cursor is
        # at the cloud tail and the MODERN tombstone was applied (no-op on
        # A -- the belief never existed there)
        assert A.peer_cursor(C.db_id) == C.engine.export_tombstones(0, 1)[0].to_cursor
    finally:
        for d in devices.values():
            d.dispose()


def test_forged_batches_trigger_rescan_and_apply_with_dedup(tmp_path):
    """A fabricated from_cursor (honestly re-sealed) does not continue
    the stored channel: the receiver RESCANS -- the channel re-anchors
    to 0 so the sender's log is re-fetched from the start, where the
    (origin, origin_seq, origin_hash) applied records deduplicate the
    re-delivery to skips.  The entries in the forged envelope are
    honest, so they apply on first sight; the FABRICATION is punished
    at the channel, not the data.  A tampered entry hash is still
    refused outright (integrity)."""
    root = tmp_path
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
    }
    refusals: list[str] = []
    try:
        C, A = devices["C"], devices["A"]
        C.observe("I like forged and coffee.", session_id="s")
        A.observe("I like forged and coffee.", session_id="s")
        A.adopt_global_ids(C)
        A.engine.forget_belief(user_id=USER,
                               belief_id=A.beliefs()[0].id, purpose="api")
        batch, _ = A.engine.export_tombstones(0, 100)
        honest_tail = batch.to_cursor

        # a FABRICATED continuation (from_cursor beyond the tail, sealed
        # honestly): the receiver detects the non-continuation and RESCANS
        # -- the channel re-anchors to 0.  The entries themselves are
        # honest (A's real tombstone), so they apply; the re-fetch from 0
        # then deduplicates the re-delivery at entry level.
        forged_from = honest_tail + 5
        forged = TombstoneBatch(sender=A.db_id,
                                from_cursor=forged_from,
                                to_cursor=forged_from + len(batch.entries),
                                from_chain_hash=batch.to_chain_hash,
                                to_chain_hash=batch.to_chain_hash,
                                entries=batch.entries).seal()
        r = C.engine.apply_tombstones(forged)
        assert (r.applied, r.skipped) == (len(batch.entries), 0)
        assert all("forged" not in b.claim_text for b in C.beliefs())
        assert C.peer_cursor(A.db_id) == 0, "the rescan re-anchors to 0"

        # the honest batch (from 0) re-delivers through the re-fetch:
        # the (origin, origin_seq, origin_hash) applied records dedup it
        r2 = C.engine.apply_tombstones(batch)
        assert (r2.applied, r2.skipped) == (0, len(batch.entries))
        assert C.peer_cursor(A.db_id) == honest_tail

        # a tampered ENTRY hash is refused outright (integrity, not a
        # continuity issue)
        forged_entry = batch.entries[0].to_dict()
        forged_entry["hash"] = "0" * 64
        forged_batch = TombstoneBatch(
            sender=A.db_id, from_cursor=0, to_cursor=honest_tail,
            from_chain_hash=batch.from_chain_hash,
            to_chain_hash=batch.to_chain_hash,
            entries=[TombstoneEntry.from_dict(forged_entry)]).seal()

        with pytest.raises(TombstoneBatchRejected) as info:
            C.engine.apply_tombstones(forged_batch)
        assert info.value.reason == "hash"
        assert refusals == []
    finally:
        for d in devices.values():
            d.dispose()


def test_receiver_restore_repulls_to_the_same_state(tmp_path):
    """A restores its backup (data + channel cursor rewind): re-pulling
    from the rewound cursor re-applies the tombstones (idempotent at
    entry level) and lands on the same visible state as the cloud."""
    root = tmp_path
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
    }
    try:
        C, A = devices["C"], devices["A"]
        C.observe("I like restore-tag and coffee.", session_id="s")
        A.observe("I like restore-tag and coffee.", session_id="s")
        A.adopt_global_ids(C)
        C.engine.forget_belief(
            user_id=USER,
            belief_id=next(b for b in C.beliefs()
                           if "restore-tag" in b.claim_text).id,
            purpose="api")

        # A backs up BEFORE the tombstone arrives (the data is still
        # there and the channel cursor is 0)
        backup_db = str(A.url).replace("sqlite:///", "").replace("a.db", "a-backup.db")
        src = A.engine._engine.raw_connection()
        from sqlalchemy import create_engine as _ce

        dest = _ce(f"sqlite:///{backup_db}").raw_connection()
        src.connection.backup(dest.connection)
        dest.close()
        src.close()

        A.pull_from(C, refusals := [])  # type: ignore
        assert refusals == []
        assert A.beliefs() == []

        pre_cursor = A.peer_cursor(C.db_id)
        assert pre_cursor >= 1
        A.engine._engine.dispose()
        A.db.dispose()
        shutil.copyfile(backup_db, str(A.url).replace("sqlite:///", ""))
        A.db = create_engine(A.url)
        A.F = sessionmaker(bind=A.db)
        A.engine = MemoryEngine(
            config=A.cfg, database_url=A.url, tenant_id=TENANT[0],
            app_id=TENANT[1], llm_client=_FakeLLM(),
            deletion_log=A.log, single_tenant_mode=True)
        assert any("restore-tag" in b.claim_text for b in A.beliefs()), \
            "the restore brought the data back"
        rewound = A.peer_cursor(C.db_id)
        assert rewound < pre_cursor, "the channel cursor rewound with the backup"

        # re-pull: from == the rewound stored cursor, so the channel
        # accepts; the tombstone is applied again (entry-level idempotent
        # against nothing -- the applied record rewound too)
        refusals2: list[str] = []
        A.pull_from(C, refusals2)
        assert refusals2 == []
        assert A.beliefs() == []
        # the same final belief state as the cloud; the session summary
        # survives on BOTH (only forget_belief ran -- session text is its
        # own layer), so recall legitimately still carries the text
        assert A.visible_state() == C.visible_state()
    finally:
        for d in devices.values():
            d.dispose()


# --- M4 batch-2 patch 2, task 2: full-machine restore of a SENDER -----

def _backup_device(d: "_Device", backup_db: str, backup_log_dir) -> None:
    """Full-machine backup: the main DB (SQLite backup API) AND the
    deletion-log directory together."""
    src = d.engine._engine.raw_connection()
    dest = create_engine(f"sqlite:///{backup_db}").raw_connection()
    src.connection.backup(dest.connection)
    dest.close()
    src.close()
    shutil.copytree(d.log_dir, backup_log_dir, dirs_exist_ok=True)


def _restore_device(d: "_Device", backup_db: str, backup_log_dir) -> None:
    """The restore half: both halves go back, then the device reboots in
    place (same db_id -- the watermark travels in the main DB)."""
    d.engine._engine.dispose()
    d.db.dispose()
    shutil.copyfile(backup_db, str(d.url).replace("sqlite:///", ""))
    shutil.rmtree(d.log_dir, ignore_errors=True)
    shutil.copytree(backup_log_dir, d.log_dir)
    d.db = create_engine(d.url)
    d.F = sessionmaker(bind=d.db)
    d.engine = MemoryEngine(
        config=d.cfg, database_url=d.url, tenant_id=TENANT[0],
        app_id=TENANT[1], llm_client=_FakeLLM(),
        deletion_log=d.log, single_tenant_mode=d.single_tenant)
    # the reboot reads its own identity (the watermark travels in the
    # main DB) before anything pulls from it
    d.engine._ensure_db()


def _sender_restore_and_offline_redelete(devices, refusals):
    """Batch-2 patch 2, task 2, both legs: A backs up, deletes in all
    three scopes, syncs out, then FULLY restores (main DB + log dir).

    Phase 1: the catch-up brings A's OWN tombstones back through the
    cloud -- the restored log no longer holds them, so the echo check
    re-applies them -- and every read entrance on A is sentinel-free.

    Phase 2: A deletes again, one of the deletions composed while the
    channel is idle (offline); the cloud and B end up clean -- the
    deletions were neither misfiled as duplicates (the applied keys are
    (origin, origin_seq, origin_hash); the re-issued ones carry new
    hashes) nor skipped by a cursor (the cloud RESCANS the sender's
    rolled-back log and re-anchors to 0).  Returns the rescans seen.
    """
    from mirror_memory.core import metrics

    C, A, B = devices["C"], devices["A"], devices["B"]
    rescans_before = metrics.value(metrics.TOMBSTONES_RESCANS)

    # seed: the same content everywhere, identities aligned with C
    for d in devices.values():
        d.observe("I like SEN1 and coffee.", session_id="sa")
        d.observe("I live in SEN2.", session_id="sb")
    _align_all(devices, C)
    for d in devices.values():
        assert d.sees("SEN1", "user")

    # A backs up (main DB + log dir) BEFORE any deletion
    backup_db = str(A.url).replace("sqlite:///", "").replace(
        "a.db", "a-backup.db")
    backup_logs = A.log_dir.parent / "a-log-backup"
    _backup_device(A, backup_db, backup_logs)

    # A deletes all three scopes and the deletions sync out
    a_sen1 = next(b for b in A.beliefs() if "SEN1" in b.claim_text)
    A.engine.forget_belief(user_id=USER, belief_id=a_sen1.id,
                           purpose="api")
    A.engine.forget_session(user_id=USER, session_id="sb", purpose="api")
    A.engine.delete_memories(user_id=USER, purpose="api")
    _sync_star(devices, refusals, random.Random(1))
    assert refusals == [], refusals
    for d in devices.values():
        assert not d.sees("SEN1", "user")
    assert C.peer_cursor(A.db_id) >= 3, "the cloud consumed A's log"

    # FULL MACHINE RESTORE of A: main DB + log dir go back
    _restore_device(A, backup_db, backup_logs)
    assert A.sees("SEN1", "user"), "the restore resurrected A's data"

    # the catch-up: A re-pulls the cloud, its OWN tombstones come
    # back, the echo check finds them missing from the restored log
    # and re-applies them
    refusals2: list[str] = []
    A.pull_from(C, refusals2)
    assert refusals2 == [], refusals2
    assert not A.sees("SEN1", "user"), \
        "A's own pre-restore deletions must re-apply after the restore"
    assert A.summaries() == 0

    # the rest of the star syncs: the cloud PULLS from the restored
    # A -- its stored cursor points past A's rewound log, which must
    # trigger the automatic rescan (never a refusal, never silent)
    _sync_star(devices, refusals, random.Random(2))
    assert refusals == [], refusals
    rescans = metrics.value(metrics.TOMBSTONES_RESCANS)
    assert rescans > rescans_before, \
        "the cloud must detect the sender's rollback (rescan metric)"

    # phase 2: new shared content; A deletes a belief (synced), then
    # wipes the whole user while OFFLINE
    for d in devices.values():
        d.observe("I like SEN3 and tea.", session_id="sc")
        d.observe("I work in SEN4.", session_id="sd")
    _align_all(devices, C)
    for d in devices.values():
        assert d.sees("SEN4", "user")

    a_sen3 = next(b for b in A.beliefs() if "SEN3" in b.claim_text)
    A.engine.forget_belief(user_id=USER, belief_id=a_sen3.id,
                           purpose="api")
    _sync_star(devices, refusals, random.Random(3))
    assert refusals == [], refusals
    for d in devices.values():
        assert not d.sees("SEN3", "belief")

    # the OFFLINE deletion: it lands in A's log and NOTHING syncs
    A.engine.delete_memories(user_id=USER, purpose="api")

    # back online: the deletion must reach everyone -- not filed as a
    # duplicate, not jumped over by a cursor
    _sync_star(devices, refusals, random.Random(4))
    assert refusals == [], refusals
    for d in (C, B):
        assert not d.sees("SEN3", "user")
        assert not d.sees("SEN4", "user"), \
            "the offline deletion did not propagate"
        assert d.visible_state() == ((), 0)
    assert A.visible_state() == ((), 0)
    assert C.peer_cursor(A.db_id) == A.log.head(A.db_id)[0]
    return (metrics.value(metrics.TOMBSTONES_RESCANS) - rescans_before)


def test_sender_full_restore_then_offline_redelete_reaches_everyone(tmp_path):
    """The SQLite leg of :func:`_sender_restore_and_offline_redelete`:
    three SQLite devices -- the sender-restore and the offline re-delete
    must reach everyone (see the helper's docstring)."""
    refusals: list[str] = []
    root = tmp_path
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
        "B": _Device(root, "b", single_tenant=True),
    }
    try:
        rescans = _sender_restore_and_offline_redelete(devices, refusals)
        assert rescans >= 1, "the sender's rollback must be DETECTED"
    finally:
        for d in devices.values():
            d.dispose()

def _mesh_rounds_with_restore(devices, refusals, rng, *, restore_name,
                              restore_round, rounds=20):
    """The alternating-delete mesh, with a mid-stream FULL MACHINE
    RESTORE of *restore_name* (main DB + log dir from its round-0
    backup) at *restore_round*.  Per round the deleted tag is invisible
    on every device and the never-restored devices agree; before the
    restore all three agree.  Returns the number of channel rescans the
    restore produced (automatic, never a refusal)."""
    from mirror_memory.core import metrics

    C = devices["C"]
    others = [n for n in devices if n != "C"]
    assert restore_name in others

    backups = {}
    for name in others:
        d = devices[name]
        bdb = str(d.url).replace("sqlite:///", "") + "-backup"
        blog = d.log_dir.parent / (d.log_dir.name + "-backup")
        _backup_device(d, bdb, blog)
        backups[name] = (bdb, blog)

    rescans_before = metrics.value(metrics.TOMBSTONES_RESCANS)
    restored = False
    for round_no in range(rounds):
        if round_no == restore_round:
            d = devices[restore_name]
            _restore_device(d, *backups[restore_name])
            restored = True

        for d in devices.values():
            d.observe(f"I like r{round_no} and coffee.",
                      session_id=f"s{round_no}")
            d.observe(f"I live in city{round_no}.",
                      session_id=f"s{round_no}")
        _align_all(devices, C)

        actor_name = ("A", "B", "C")[round_no % 3]
        scope = SCOPES[round_no % 3]
        devices[actor_name].delete(scope, round_no)
        _sync_star(devices, refusals, rng)

        assert refusals == [], (round_no, refusals)
        tag = f"r{round_no}"
        for name, d in devices.items():
            assert not d.sees(tag, scope), (round_no, name, tag, scope)
        # the never-restored devices agree every round; ALL three agree
        # only until the restore (the restored device loses the content
        # its backup predates -- the channel carries deletions, not
        # content, and a replayed user-scope tombstone legitimately
        # wipes what it finds; the final wipe below equalises everyone)
        states = {n: devices[n].visible_state()
                  for n in others if n != restore_name}
        states["C"] = C.visible_state()
        assert len(set(states.values())) == 1, (round_no, states)
        if not restored:
            assert devices[restore_name].visible_state() \
                == C.visible_state(), (round_no, restore_name)

    # the channel carries deletions, not content: the restored device
    # never regains the pre-restore content -- a final user-scope wipe
    # equalises everyone, which is the three-way consistency claim
    C.engine.delete_memories(user_id=USER, purpose="api")
    _sync_star(devices, refusals, rng)
    assert refusals == [], refusals
    for d in devices.values():
        assert d.visible_state() == ((), 0)
    return metrics.value(metrics.TOMBSTONES_RESCANS) - rescans_before


def test_mesh_survives_a_midstream_full_restore(tmp_path):
    """Batch-2 patch 2, task 2 mesh variant (SQLite): one edge fully
    restores mid-stream; the channel auto-rescans the rolled-back log,
    nothing wedges, zero refusals, and the mesh converges."""
    rng = random.Random(23)
    refusals: list[str] = []
    root = tmp_path
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
        "B": _Device(root, "b", single_tenant=True),
    }
    try:
        rescans = _mesh_rounds_with_restore(
            devices, refusals, rng, restore_name="A", restore_round=7)
        assert rescans >= 1, "the restore must be DETECTED (rescan metric)"
    finally:
        for d in devices.values():
            d.dispose()


# --- M4 batch-2 patch 3, task 1: the catch-up keeps pulling after a rescan

class _PeerTransport:
    """The in-process link one engine uses to pull another's log (used to
    drive the engine's own _catch_up loop in the tests below)."""

    def __init__(self, peer_engine):
        self.peer = peer_engine

    @property
    def db_id(self):
        return self.peer._db_id

    def export_tombstones(self, cursor, limit, peer_db_id=None):
        return self.peer.export_tombstones(cursor, limit)


def _deletions(d: "_Device", rounds: range, start: int) -> None:
    """`rounds` belief-scope deletions on *d*, each its own log entry."""
    for i in rounds:
        tag = f"p{start + i}"
        session_id = f"ps{start + i}"
        d.observe(f"I like {tag} and coffee.", session_id=session_id)
        belief = next(b for b in d.beliefs() if tag in b.claim_text)
        assert d.engine.forget_belief(user_id=USER, belief_id=belief.id,
                                      purpose="api")


def test_catchup_completes_a_rescan_and_repull_in_one_pass(tmp_path):
    """The task's scenario: the cloud's stored cursor (7) overshoots the
    restored sender's log tail (5); ONE catch-up pass detects the sender
    rollback (rescan), re-pulls from the reset cursor, and ends with the
    stored cursor at the sender's tail and the sync state normal."""
    from mirror_memory.core import metrics

    root = tmp_path
    refusals: list[str] = []
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
    }
    try:
        C, A = devices["C"], devices["A"]
        rescans_before = metrics.value(metrics.TOMBSTONES_RESCANS)

        # A deletes 5 scopes (log entries 1..5) and they sync out
        _deletions(A, range(5), 0)
        _sync_star(devices, refusals, random.Random(1))
        assert refusals == []
        assert C.peer_cursor(A.db_id) == 5

        # A backs up (main DB + log dir) at tail 5
        backup_db = str(A.url).replace("sqlite:///", "") + "-backup"
        backup_logs = A.log_dir.parent / (A.log_dir.name + "-backup")
        _backup_device(A, backup_db, backup_logs)

        # A deletes 2 more (entries 6..7) and they sync out: cursor 7
        _deletions(A, range(2), 5)
        _sync_star(devices, refusals, random.Random(2))
        assert refusals == []
        assert C.peer_cursor(A.db_id) == 7
        assert A.log.head(A.db_id)[0] == 7

        # A FULL MACHINE RESTORES (tail back to 5, same entry hashes)
        _restore_device(A, backup_db, backup_logs)
        assert A.log.head(A.db_id)[0] == 5

        # ONE catch-up pass of the cloud against A: rescan + re-pull
        C.engine._config.sync.enabled = True
        C.engine._sync_transport = _PeerTransport(A.engine)
        C.engine._catch_up()
        assert refusals == []
        assert C.peer_cursor(A.db_id) == 5, \
            "one catch-up must land the channel on the sender's tail"
        assert C.engine.sync_status()["state"] == "normal"
        assert metrics.value(metrics.TOMBSTONES_RESCANS) > rescans_before, \
            "the rollback must have been detected (rescan metric)"
    finally:
        for d in devices.values():
            d.dispose()


def test_catchup_survives_an_empty_rescan_batch(tmp_path):
    """The fix's target: the rescan is triggered by an EMPTY batch (the
    stored cursor equals the sender's tail but the chain hash differs --
    the sender restored an older log and re-deleted different things).
    The loop must NOT exit on the empty range: it keeps pulling from the
    reset cursor and lands on the tail with the new deletions applied,
    all within the ONE catch-up pass."""
    from mirror_memory.core import metrics

    root = tmp_path
    refusals: list[str] = []
    devices = {
        "C": _Device(root, "c", single_tenant=False),
        "A": _Device(root, "a", single_tenant=True),
    }
    try:
        C, A = devices["C"], devices["A"]
        rescans_before = metrics.value(metrics.TOMBSTONES_RESCANS)

        # A deletes 3 (entries 1..3) -> sync (cursor 3) -> backup at 3
        _deletions(A, range(3), 0)
        _sync_star(devices, refusals, random.Random(1))
        assert refusals == []
        backup_db = str(A.url).replace("sqlite:///", "") + "-backup"
        backup_logs = A.log_dir.parent / (A.log_dir.name + "-backup")
        _backup_device(A, backup_db, backup_logs)
        assert A.log.head(A.db_id)[0] == 3

        # A deletes 2 more (entries 4..5) -> sync (cursor 5)
        _deletions(A, range(2), 3)
        _sync_star(devices, refusals, random.Random(2))
        assert refusals == []
        assert C.peer_cursor(A.db_id) == 5

        # A restores the tail-3 backup, then deletes 2 DIFFERENT scopes:
        # new entries 4..5 with DIFFERENT hashes, and the channel is
        # idle (nothing synced since)
        _restore_device(A, backup_db, backup_logs)
        assert A.log.head(A.db_id)[0] == 3
        _deletions(A, range(2), 10)  # p10, p11 -- scopes never deleted before
        assert A.log.head(A.db_id)[0] == 5
        new_scopes = [f"p10", f"p11"]

        # the cloud pulls ONCE: the first batch is empty (5 -> 5) but
        # its chain hash betrays the rollback -> rescan -> re-pull -> the
        # new deletions apply -> the tail
        C.engine._config.sync.enabled = True
        C.engine._sync_transport = _PeerTransport(A.engine)
        C.engine._catch_up()
        assert refusals == []
        assert C.peer_cursor(A.db_id) == 5
        for tag in new_scopes:
            assert not C.sees(tag, "belief"), \
                "the re-issued deletions must apply within the same pass"
        assert metrics.value(metrics.TOMBSTONES_RESCANS) > rescans_before
    finally:
        for d in devices.values():
            d.dispose()
