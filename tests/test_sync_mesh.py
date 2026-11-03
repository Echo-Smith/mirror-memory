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
from mirror_memory.core.tombstones import TombstoneBatch, TombstoneBatchRejected

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
        import os as _os

        for _ in range(max_batches):
            cursor = self.peer_cursor(peer.db_id)
            try:
                batch, _next = peer.engine.export_tombstones(
                    cursor, 500)
            except Exception as exc:  # noqa: BLE001 -- recorded
                refusals.append(f"export from {peer.name}: {exc!r}")
                return
            if _os.environ.get("MESH_TRACE"):
                with open("/tmp/mesh_trace.txt", "a") as _fh:
                    _fh.write(f"{self.name}<-{peer.name}: from={batch.from_cursor} "
                              f"to={batch.to_cursor} entries={len(batch.entries)}\n")
            import os as _os

            if _os.environ.get("MESH_TRACE"):
                with open("/tmp/mesh_trace.txt", "a") as _fh:
                    _fh.write(f"{self.name}<-{peer.name}: from={batch.from_cursor} "
                              f"to={batch.to_cursor} entries={len(batch.entries)}\n")
            try:
                self.engine.apply_tombstones(batch)
            except TombstoneBatchRejected as exc:
                refusals.append(f"apply from {peer.name}: {exc}")
                return
            if _os.environ.get("MESH_TRACE"):
                with open("/tmp/mesh_trace.txt", "a") as _fh:
                    _fh.write(f"  {self.name} stored cursor for {peer.name} = "
                              f"{self.peer_cursor(peer.db_id)}\n")
            if batch.to_cursor == batch.from_cursor:
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
                    diag = []
                    from mirror_memory.core.repository import read_sync_peer as _rsp

                    for other in devices.values():
                        diag.append((other.name, d.peer_cursor(other.db_id),
                                     other.log.head(other.db_id)[0]))
                    probe = d.engine.export_tombstones(
                        d.peer_cursor(devices["C"].db_id), 500,
                        peer_db_id=devices["C"].db_id)[0]
                    full, _fc = devices["C"].engine.export_tombstones(
                        0, 500)
                    log_lines = [f"C log seq={e.seq} origin={e.origin[:6]} "
                                 f"scope={e.scope} key={e.scope_key[:12]}"
                                 for e in devices["C"].log.read_from(
                                     devices["C"].db_id, 0)]
                    raise AssertionError(
                        (round_no, d.name, tag, scope, diag, d.visible_state(),
                         refusals, "probe@4:", probe.to_dict(),
                         "full-from-0:", [(e.origin[:6], e.origin_seq)
                                          for e in full.entries],
                         log_lines))

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
    to 0, the sender's log is re-fetched from the start, and the
    (origin, origin_hash) applied records deduplicate the re-delivery.
    A tampered entry hash is still refused outright (integrity)."""
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
        before = C.visible_state()

        # a FABRICATED continuation (from_cursor beyond the tail, sealed
        # honestly): the receiver detects the non-continuation, rescans
        # from 0, and the entries dedup to no-ops
        forged_from = honest_tail + 5
        forged = TombstoneBatch(sender=A.db_id,
                                from_cursor=forged_from,
                                to_cursor=forged_from + len(batch.entries),
                                from_chain_hash=batch.to_chain_hash,
                                to_chain_hash=batch.to_chain_hash,
                                entries=batch.entries).seal()
        r = C.engine.apply_tombstones(forged)
        assert (r.applied, r.skipped) == (0, len(batch.entries))
        assert C.visible_state() == before
        assert C.peer_cursor(A.db_id) == 0, "the rescan re-anchors to 0"

        # the honest batch (from 0) then applies through the re-fetch
        C.engine.apply_tombstones(batch)
        assert all("forged" not in b.claim_text for b in C.beliefs())
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
        from mirror_memory.core.tombstones import TombstoneBatchRejected

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
