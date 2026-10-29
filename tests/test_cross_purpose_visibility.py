"""M3 batch 3: cross-purpose visibility (the A2 rule, no readable_sources).

A belief is visible under purpose Q through its PROVENANCE only:

- with evidence links: at least one linked evidence was collected under a
  purpose that is still live (or under the unknown marker while the global
  switch is on);
- without links: same rule over the created / corrected events only;
- support / worker-publish / confirmation events NEVER count -- a Q event
  must not launder memory collected under a since-revoked P (pinned);
- with no recorded provenance at all: the unknown marker.

Applied to recall_candidates, list_active_beliefs, explain_belief,
belief_history, the verification candidates, export pages, the server's
belief_by_id, and the worker's evidence load / K3 context.  Snapshots are
gated read-time by their own provenance (batch-3 snapshot decision).
"""

from __future__ import annotations

import json
import threading

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import provenance
from mirror_memory.core.authorization import PurposeDenied
from mirror_memory.core.models import (
    Base, Belief, BeliefEvent, BeliefEvidenceLink, Evidence, EvolutionJob, Snapshot,
)
from mirror_memory.core.repository import (
    _append_event, get_belief_events, list_active_beliefs, recall_candidates,
    record_evidence, set_memory_enabled,
)
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.core.export import read_page
from mirror_memory.worker.evolution import _claim_job, _process_single_job, enqueue_job

USER = "u-vis"
TENANT = ("default", "default")
P, Q = "p-src", "q-reader"
SENTINEL = "SENTINEL-VIS-77"


class _FakeLLM:
    def __init__(self, reply=None):
        self.reply = reply

    def generate(self, **kw):
        return self.reply or '{"patterns": [], "support_policy": {}}'


@pytest.fixture
def db(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/vis.db")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _engine(db) -> MemoryEngine:
    """*db* is an Engine (the sqlite fixture) or a URL (the postgres leg's
    app-role runtime)."""
    url = str(db.url) if hasattr(db, "url") else str(db)
    F = sessionmaker(bind=create_engine(url) if isinstance(db, str) else db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    return MemoryEngine(config=load_config("config/"), database_url=url,
                        tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM())


def _grant(eng: MemoryEngine, purpose: str, operations=("observe", "recall", "correct",
                                                        "forget", "export", "explain", "extract")):
    eng.grant_purpose(user_id=USER, purpose=purpose, operations=list(operations), issuer="t")


def _observe(eng: MemoryEngine, text: str, purpose: str, session_id: str) -> None:
    eng.observe(user_id=USER, session_id=session_id, text=text, purpose=purpose)


def _beliefs(db) -> list[Belief]:
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            return s.execute(select(Belief).where(Belief.user_id == USER).order_by(Belief.id)).scalars().all()
        finally:
            s.close()


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------


def test_visibility_rule_reads_live_purposes_once_and_without_caching(db):
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            rule = provenance.visibility(s, USER, Q)
            assert rule.live == {P, Q}
            assert rule.purpose_visible(P) and rule.purpose_visible(Q)
            assert rule.purpose_visible("") and not rule.purpose_visible("other")
            with pytest.raises(ValueError):
                provenance.visibility(s, USER, "two words")
        finally:
            s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="t", reason="withdrawn")
    with tenant_scope(*TENANT):
        s = F()
        try:
            # re-read inside a fresh transaction: no cache
            assert provenance.visibility(s, USER).live == {Q}
        finally:
            s.close()


def test_expired_grant_is_not_live(db):
    eng = _engine(db)
    _grant(eng, P, operations=["observe", "recall"])
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            from datetime import datetime, timedelta

            from mirror_memory.core.models import PurposeGrant
            from mirror_memory.core.tenancy import consent_write_scope

            with consent_write_scope():
                row = s.query(PurposeGrant).filter_by(user_id=USER, purpose=P).one()
                # the engine stores naive UTC; a naive LOCAL now would be
                # hours away from it
                row.expires_at = datetime.utcnow() - timedelta(hours=1)
            s.commit()
            assert provenance.visibility(s, USER).live == set()
        finally:
            s.close()


# ---------------------------------------------------------------------------
# belief visibility through the engine's read paths
# ---------------------------------------------------------------------------


def _p_q_setup(db) -> MemoryEngine:
    """Memory collected under P and under Q; P granted then revoked."""
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I like {SENTINEL} and coffee.", P, "s1")
    _observe(eng, "I live in Berlin and work as a nurse.", Q, "s2")
    return eng


def test_p_only_memory_hides_from_q_after_revocation_across_read_paths(db):
    eng = _p_q_setup(db)
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")

    assert f"{SENTINEL}" not in (eng.recall(user_id=USER, query=SENTINEL, purpose=Q) or "")
    beliefs = eng.get_beliefs(user_id=USER, purpose=Q)
    claims = [b.claim_text for b in beliefs]
    assert all(SENTINEL not in c for c in claims) and claims, claims
    panel = eng.panel(user_id=USER, purpose=Q)
    assert SENTINEL not in repr(panel.sections)
    for item in eng.get_verification_candidates(user_id=USER, purpose=Q):
        assert SENTINEL not in item["claim_text"]
    page = eng.export_page(user_id=USER, purpose=Q)
    assert SENTINEL not in json.dumps(page.to_dict(), default=str)
    # Q's own memory still works
    assert "Berlin" in (eng.recall(user_id=USER, query="Berlin", purpose=Q) or "")
    # and P is refused outright for its own purpose
    with pytest.raises(PurposeDenied) as info:
        eng.recall(user_id=USER, query=SENTINEL, purpose=P)
    assert info.value.decision.reason == "revoked"


def test_p_memory_visible_under_q_while_p_still_live(db):
    """The A2 upside: while P is granted, Q reads P-collected memory."""
    eng = _p_q_setup(db)
    block = eng.recall(user_id=USER, query=SENTINEL, purpose=Q)
    assert SENTINEL in (block or "")


def test_event_bypass_is_locked(db):
    """A P-only-evidence belief gains a Q support event (what the worker
    publish writes); after P is revoked it stays invisible under Q --
    support events are consequences, never provenance."""
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I like {SENTINEL} and coffee.", P, "s1")
    belief = _beliefs(db)[0]
    # a Q support event on the P belief, written the way the publisher does
    from mirror_memory.core.tenancy import engine_write_scope

    with tenant_scope(*TENANT), engine_write_scope():
        s = sessionmaker(bind=db)()
        provenance.bind(s, Q)
        _append_event(s, belief, "supported", evidence=[])
        s.commit()
        s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    assert SENTINEL not in (eng.recall(user_id=USER, query=SENTINEL, purpose=Q) or "")
    assert all(SENTINEL not in b.claim_text for b in eng.get_beliefs(user_id=USER, purpose=Q))


def test_unlinked_belief_follows_its_created_and_corrected_events(db):
    """A corrected belief has no evidence links: its provenance is the
    created / corrected events.  Created under P, corrected under Q: after
    P is revoked it stays visible under Q through the Q correction."""
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I like {SENTINEL} and coffee.", P, "s1")
    belief = _beliefs(db)[0]
    corrected = eng.correct(
        user_id=USER, belief_id=belief.id, new_claim_text=f"I love {SENTINEL} lots",
        new_object=getattr(belief, "object", "") or None, purpose=Q,
    )
    assert corrected is not None, "the correction must land for the scenario to mean anything"
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    claims = [b.claim_text for b in eng.get_beliefs(user_id=USER, purpose=Q)]
    assert claims and all(SENTINEL in c for c in claims), claims


def test_unlinked_belief_created_under_p_hides_after_revocation(db):
    """A K3-style pattern belief (no evidence links, created by the worker
    under P) disappears from Q once P is revoked."""
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, "I like coffee.", P, "s1")
    from mirror_memory.core.tenancy import engine_write_scope

    with tenant_scope(*TENANT), engine_write_scope():
        s = sessionmaker(bind=db)()
        provenance.bind(s, P)
        s.add(Belief(user_id=USER, dimension="topic", key="pattern",
                     claim_text=f"pattern about {SENTINEL}", confidence=0.8, layer="L2"))
        s.flush()
        belief = s.query(Belief).filter_by(user_id=USER, key="pattern").one()
        _append_event(s, belief, "created", evidence=[])
        s.commit()
        s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    assert all(SENTINEL not in b.claim_text for b in eng.get_beliefs(user_id=USER, purpose=Q))


def test_unrecorded_provenance_behaves_as_the_unknown_marker(db):
    """Direct seeding (no links, no events) = source not recorded: visible
    while the global switch is on, exactly like '' evidence."""
    eng = _engine(db)
    _grant(eng, Q)
    with tenant_scope(*TENANT), fixture_write_scope("seed bare belief"):
        s = sessionmaker(bind=db)()
        s.add(Belief(user_id=USER, dimension="topic", key="bare",
                     claim_text="bare seeded claim", confidence=0.9, layer="L2"))
        s.commit()
        s.close()
    assert "bare seeded claim" in (eng.recall(user_id=USER, query="bare", purpose=Q) or "")


# ---------------------------------------------------------------------------
# explain / history / export: only visible provenance is listed
# ---------------------------------------------------------------------------


def test_explain_hides_invisible_beliefs_and_filters_visible_provenance(db):
    eng = _p_q_setup(db)
    p_belief = next(b for b in _beliefs(db) if SENTINEL in b.claim_text)
    q_belief = next(b for b in _beliefs(db) if "Berlin" in b.claim_text)
    # a Q event on the P belief so explain's event list has a mix
    from mirror_memory.core.tenancy import engine_write_scope

    with tenant_scope(*TENANT), engine_write_scope():
        s = sessionmaker(bind=db)()
        provenance.bind(s, Q)
        _append_event(s, p_belief, "supported", evidence=[])
        s.commit()
        s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")

    # invisible belief: "not found", no content, no existence leak
    assert eng.explain(user_id=USER, belief_id=p_belief.id, purpose=Q) is None
    # visible belief: provenance filtered to visible purposes
    explained = eng.explain(user_id=USER, belief_id=q_belief.id, purpose=Q)
    assert explained is not None
    assert explained["source_purposes"] == ["q-reader"]
    assert explained["authorization"]["purpose"] == "q-reader"
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=db)()
        try:
            visible_ids = {e.id for e in s.execute(select(Evidence).where(
                Evidence.user_id == USER, Evidence.source_purpose == Q)).scalars()}
        finally:
            s.close()
    assert set(explained["evidence_ids"]) <= visible_ids
    for relation_items in explained["evidence_graph"]["by_relation"].values():
        assert all(i["evidence_id"] in visible_ids for i in relation_items)
    # re-grant P through the product API (CAS update of the revoked row):
    # while P is live again the P belief is visible under Q once more (and
    # q_belief's own provenance listing is unchanged -- it is Q-only)
    from mirror_memory.core.models import PurposeGrant

    with tenant_scope(*TENANT):
        s = sessionmaker(bind=db)()
        version = s.query(PurposeGrant).filter_by(user_id=USER, purpose=P).one().version
        s.close()
    eng.grant_purpose(user_id=USER, purpose=P, operations=[
        "observe", "recall", "correct", "forget", "export", "explain", "extract"],
        issuer="t", expected_version=version)
    assert eng.explain(user_id=USER, belief_id=p_belief.id, purpose=Q) is not None
    assert eng.explain(user_id=USER, belief_id=q_belief.id, purpose=Q)["source_purposes"] == ["q-reader"]


def test_explain_omits_the_digest_when_any_provenance_is_invisible(db):
    """A compacted digest folds text from every linked evidence; with mixed
    visibility it is withheld entirely (fail-closed on the aggregate)."""
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I like {SENTINEL} and coffee.", P, "s1")
    _observe(eng, "I live in Berlin.", Q, "s2")
    _observe(eng, "I also like tea.", Q, "s3")
    evidence_rows = None
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=db)()
        evidence_rows = {e.ref: e.id for e in s.execute(select(Evidence).where(Evidence.user_id == USER)).scalars()}
        s.close()
    # a belief linked to BOTH a P and a Q evidence: visible via the Q link
    with tenant_scope(*TENANT), fixture_write_scope("seed mixed-link belief + digest"):
        from mirror_memory.core.models import EvidenceDigest

        s = sessionmaker(bind=db)()
        s.add(Belief(user_id=USER, dimension="topic", key="mixed",
                     claim_text="mixed provenance claim", confidence=0.9, layer="L2"))
        s.flush()
        mixed = s.query(Belief).filter_by(user_id=USER, key="mixed").one()
        s.add(BeliefEvidenceLink(belief_id=mixed.id, evidence_id=evidence_rows["s1:1"], relation="support"))
        s.add(BeliefEvidenceLink(belief_id=mixed.id, evidence_id=evidence_rows["s2:1"], relation="support"))
        s.add(EvidenceDigest(user_id=USER, belief_id=mixed.id,
                             summary=f"digest mentions {SENTINEL}"))
        s.commit()
        s.close()
    mixed_id = _beliefs(db)[-1].id

    # while P is live: digest present (all provenance visible)
    explained = eng.explain(user_id=USER, belief_id=mixed_id, purpose=Q)
    assert explained is not None
    assert SENTINEL in json.dumps(explained["evidence_graph"])
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    # after the revoke: still visible via the Q link, but the digest (which
    # folds the P evidence's text) and the P evidence row are withheld
    explained = eng.explain(user_id=USER, belief_id=mixed_id, purpose=Q)
    assert explained is not None
    assert "digest" not in json.dumps(explained["evidence_graph"])
    assert SENTINEL not in json.dumps(explained["evidence_graph"])
    assert explained["source_purposes"] == ["q-reader"]


def test_belief_history_is_gated_on_visibility(db):
    eng = _p_q_setup(db)
    p_belief = next(b for b in _beliefs(db) if SENTINEL in b.claim_text)
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    assert eng.belief_history(user_id=USER, belief_id=p_belief.id, purpose=Q) == []


def test_export_pages_filter_by_visibility_and_cursors_stay_sound(db):
    eng = _p_q_setup(db)
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            rule = provenance.visibility(s, USER)
            section, items, ids, cursor = read_page(s, USER, cursor=None, page_size=1, visibility=rule)
            assert section == "beliefs" and len(items) <= 1
            assert SENTINEL not in json.dumps(items)
            seen = list(items)
            fetched = 1
            while cursor is not None:
                section, items, ids, cursor = read_page(s, USER, cursor=cursor, page_size=1, visibility=rule)
                seen += items
                fetched += 1
            assert section == "events"
            assert SENTINEL not in json.dumps(seen, default=str)
            assert any("Berlin" in json.dumps(i) for i in seen)
        finally:
            s.close()


# ---------------------------------------------------------------------------
# snapshots: read-time gating (the batch-3 snapshot decision)
# ---------------------------------------------------------------------------


def test_snapshot_context_is_gated_by_its_own_provenance(db):
    """A snapshot synthesized under P is not used as K3 context by a job
    under Q once P is revoked -- and no older snapshot is resurrected."""
    from mirror_memory.extraction.synthesis import _get_current_understanding

    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, "I like coffee.", P, "s1")
    with tenant_scope(*TENANT), fixture_write_scope("seed snapshot"):
        s = sessionmaker(bind=db)()
        s.add(Snapshot(id="snap-p", user_id=USER, version=1, status="active",
                       content_json='{"note": "shaped by P"}', source_purpose=P,
                       source_generation=0))
        s.commit()
        s.close()
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=db)()
        try:
            rule = provenance.visibility(s, USER)
            assert "shaped by P" in _get_current_understanding(s, USER, visibility=rule)
        finally:
            s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=db)()
        try:
            rule = provenance.visibility(s, USER)
            assert _get_current_understanding(s, USER, visibility=rule) == ""
        finally:
            s.close()


def test_worker_synthesis_never_reads_invisible_memory(db):
    """Job under Q after P was revoked: the evidence load and K3 context
    see nothing P-only, so nothing P-only can be republished."""
    cfg = load_config("config/")
    # the LLM invents nothing: if the sentinel shows up in what the job
    # writes, it can only have come from memory the job read
    cfg.llm_client = _FakeLLM()
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I like {SENTINEL} and coffee.", P, "s1")
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")

    beliefs_before = {b.id for b in _beliefs(db)}
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            job = enqueue_job(s, USER, cfg, purpose=Q)
            s.commit()
            job_id = job.id
            token = _claim_job(s, job_id, lease_seconds=60)
            row = s.get(EvolutionJob, job_id)
            _process_single_job(s, row, cfg, token)
            s.commit()
            s.expire_all()
            job = s.get(EvolutionJob, job_id)
            assert job.status in ("completed", "cancelled"), (job.status, job.error_code)
            # the P belief row itself persists (hidden, not deleted); the
            # invariant is that the JOB wrote nothing carrying the sentinel
            beliefs = [b.claim_text for b in s.execute(select(Belief)).scalars() if b.id not in beliefs_before]
            snapshots = [snap.content_json for snap in s.execute(select(Snapshot)).scalars()
                         if snap.source_purpose == Q]
        finally:
            s.close()
    assert all(SENTINEL not in text for text in beliefs), beliefs
    assert all(SENTINEL not in text for text in snapshots), snapshots


# ---------------------------------------------------------------------------
# concurrent revocation: "revoked AND committed" must imply "not visible"
# ---------------------------------------------------------------------------


def _visibility_race_round(eng, round_no) -> None:
    """One round: grant a fresh P (a revoked grant row cannot be re-granted
    without the CAS version -- unique purposes per round), seed, then a
    reader recalls under Q while the main thread commits the revocation.
    Every recall that STARTED after the commit must be sentinel-free (the
    invariant); a recall overlapping the commit may still see the
    pre-revoke state."""
    cfg = load_config("config/")
    p_round = f"{P}-{round_no}"
    _grant(eng, p_round, operations=["observe", "recall", "extract"])
    _observe(eng, f"round{round_no} likes {SENTINEL}", p_round, f"sr{round_no}")

    commit_instant = {}
    reader_ready = threading.Event()
    stop = threading.Event()
    observations: list[tuple[float, bool]] = []
    errors: list[BaseException] = []

    def read_loop():
        try:
            reader_ready.set()
            while not stop.is_set():
                started = __import__("time").monotonic()
                try:
                    block = eng.recall(user_id=USER, query=SENTINEL, purpose=Q) or ""
                except Exception as exc:  # noqa: BLE001 -- a lock error is not visibility
                    if "locked" in str(exc):
                        continue
                    raise
                observations.append((started, SENTINEL in block))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def revoke_loop():
        # wait until the reader is running, commit the revocation, then keep
        # the reader running a grace window so post-commit recalls happen
        reader_ready.wait(timeout=10)
        for _ in range(500):
            if observations:
                break
            __import__("time").sleep(0.01)
        eng.revoke_purpose(user_id=USER, purpose=p_round, actor="user", reason=f"round{round_no}")
        commit_instant["t"] = __import__("time").monotonic()
        __import__("time").sleep(0.3)
        stop.set()

    threads = [threading.Thread(target=read_loop), threading.Thread(target=revoke_loop)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, [repr(e) for e in errors]
    assert commit_instant and observations
    late = [seen for started, seen in observations if started > commit_instant["t"]]
    assert late, "the reader should have kept recalling after the commit"
    assert not any(late), (
        "a recall that started after the revocation committed saw revoked memory"
    )
    # the next round uses a fresh purpose; nothing to re-grant


def test_revocation_vs_visibility_50_rounds_sqlite(tmp_path):
    eng0 = create_engine(f"sqlite:///{tmp_path / 'vis-race.db'}", connect_args={"timeout": 5.0})
    Base.metadata.create_all(eng0)
    eng0.dispose()
    db = create_engine(f"sqlite:///{tmp_path / 'vis-race.db'}", connect_args={"timeout": 5.0})
    try:
        eng = _engine(db)
        _grant(eng, Q)
        for round_no in range(50):
            _visibility_race_round(eng, round_no)
    finally:
        db.dispose()


def test_session_summary_fallback_is_visibility_filtered(db):
    """recall's snippet fallback returns raw message text: summaries are
    provenance like evidence, so P's snippets vanish from Q once P is
    revoked (session summaries need session_summary_enabled)."""
    cfg = load_config("config/")
    cfg.session_summary_enabled = True
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    eng = MemoryEngine(config=cfg, database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM())
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I keep a journal about {SENTINEL} in the attic.", P, "s1")
    _observe(eng, "I keep a journal about gardening in the shed.", Q, "s2")
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    # both snippets exist, but only Q's may surface
    with tenant_scope(*TENANT):
        s = F()
        try:
            summaries = s.execute(select(__import__("mirror_memory.core.models", fromlist=["SessionSummary"]).SessionSummary)).scalars().all()
        finally:
            s.close()
    assert len(summaries) == 2
    block = eng.recall(user_id=USER, query="journal attic", purpose=Q) or ""
    assert SENTINEL not in block
    assert "gardening" in block or "shed" in block


def test_revocation_vs_visibility_50_rounds_postgres(pg_app_url):
    """The same invariant under real PostgreSQL concurrency (MVCC: readers
    never block the revoker): every recall that STARTED after the
    revocation committed is sentinel-free."""
    eng = _engine(pg_app_url)
    _grant(eng, Q)
    for round_no in range(50):
        _visibility_race_round(eng, round_no)


# ---------------------------------------------------------------------------
# batch-3 patch task 4: filter before limit (no starvation)
# ---------------------------------------------------------------------------


def test_no_starvation_when_the_revoked_purpose_bulks_the_limit(db):
    """P has 200 NEWER beliefs, Q has 5 older ones.  After P is revoked,
    recall / get_beliefs / panel / the summary fallback must still return
    Q's content -- the visibility rule runs inside the query, before the
    limit."""
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    # deterministic seeding through the repository with bound provenance:
    # 200 NEWER P beliefs (created events under P) and 5 OLDER Q beliefs
    # -- the created-event branch of the rule
    from mirror_memory.core.repository import record_claim

    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("seed the bulk"):
        s = F()
        provenance.bind(s, P)
        for i in range(200):
            record_claim(s, USER, dimension="topic", key=f"p{i}",
                         claim_text=f"P fact {i} about {SENTINEL}", confidence=0.8,
                         session_id=f"sp{i}")
        s.commit()
        s.close()
    with tenant_scope(*TENANT), fixture_write_scope("seed the Q rows"):
        s = F()
        provenance.bind(s, Q)
        # anchor keys: the panel only renders keys with display labels
        for i, key in enumerate(("work", "family", "health", "travel", "hobby")):
            record_claim(s, USER, dimension="topic", key=key,
                         claim_text=f"Q gardening fact {i}", confidence=0.8,
                         session_id="sq")
        s.commit()
        s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")

    beliefs = eng.get_beliefs(user_id=USER, purpose=Q, limit=10)
    assert beliefs and all("gardening" in b.claim_text for b in beliefs), (
        f"get_beliefs starved: {[b.claim_text for b in beliefs]}"
    )
    block = eng.recall(user_id=USER, query="gardening", purpose=Q) or ""
    assert "gardening" in block and SENTINEL not in block, block
    panel = eng.panel(user_id=USER, purpose=Q)
    items = [i for sec in panel.sections for i in sec.get("items", [])]
    assert items, "panel starved: no visible items"
    assert all(SENTINEL not in repr(i) for i in items)
    page = eng.export_page(user_id=USER, purpose=Q)
    assert SENTINEL not in json.dumps(page.to_dict(), default=str)


# ---------------------------------------------------------------------------
# batch-3 patch task 1: session summaries are per-purpose rows
# ---------------------------------------------------------------------------


def test_session_summary_rows_are_split_by_purpose(db):
    """The same session observed under Q then P produces two rows; the P
    row's text never reaches a Q recall after P is revoked (the AND/OR
    rule collapses to the row's single purpose)."""
    cfg = load_config("config/")
    cfg.session_summary_enabled = True
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    eng = MemoryEngine(config=cfg, database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM())
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I keep notes about {SENTINEL} in the attic.", P, "shared")
    _observe(eng, "I keep notes about gardening in the shed.", Q, "shared")
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")

    from mirror_memory.core.models import SessionSummary

    with tenant_scope(*TENANT):
        s = F()
        try:
            rows = s.execute(select(SessionSummary)).scalars().all()
        finally:
            s.close()
    assert {(r.source_purpose, SENTINEL in r.summary_text) for r in rows} == {
        ("p-src", True), ("q-reader", False),
    }, [(r.source_purpose, r.summary_text) for r in rows]
    block = eng.recall(user_id=USER, query="notes attic", purpose=Q) or ""
    assert SENTINEL not in block


# ---------------------------------------------------------------------------
# batch-3 patch task 2: snapshots record their INPUT provenance union
# ---------------------------------------------------------------------------


def test_snapshot_records_the_input_union_and_k3_applies_the_and_rule(db):
    """A Q job synthesizes while P is live: the snapshot's union records
    BOTH purposes (the inputs it read).  After P is revoked the snapshot
    is AND-invisible to K3 even though the job purpose (Q) still is --
    merged free text needs every input visible."""
    from mirror_memory.extraction.synthesis import _get_current_understanding
    from mirror_memory.core.models import Snapshot

    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I like {SENTINEL} and coffee.", P, "s1")
    _observe(eng, "I live in Berlin and work as a nurse.", Q, "s2")

    # a Q job: its evidence load sees BOTH beliefs (P live), so the
    # snapshot's union is {P, Q}, not just {Q}
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            rule = provenance.visibility(s, USER)
            from mirror_memory.worker.evolution import _previous_snapshot_purposes

            assert _previous_snapshot_purposes(s, USER) == set(), "no snapshots yet"
        finally:
            s.close()
    beliefs = _beliefs(db)
    with tenant_scope(*TENANT), fixture_write_scope("persist a snapshot"):
        s = F()
        from mirror_memory.worker.snapshot import persist_snapshot

        union: set[str] = {Q}
        link_purposes, event_purposes = provenance.provenance_purposes(
            s, USER, [b.id for b in beliefs])
        for b in beliefs:
            union |= link_purposes.get(b.id, set()) | event_purposes.get(b.id, set())
        from mirror_memory.core.repository import current_deletion_generation

        persist_snapshot(s, USER, {"note": "merged"}, {}, "{}", source_purposes=sorted(union),
                         source_generation=current_deletion_generation(s, USER))
        s.commit()
        s.close()
    with tenant_scope(*TENANT):
        s = F()
        try:
            snap = s.execute(select(Snapshot)).scalars().one()
            assert sorted(json.loads(snap.source_purposes_json)) == [P, Q], snap.source_purposes_json
            rule = provenance.visibility(s, USER)
            assert "merged" in _get_current_understanding(s, USER, visibility=rule)
        finally:
            s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    with tenant_scope(*TENANT):
        s = F()
        try:
            rule = provenance.visibility(s, USER)
            # AND rule: Q alone is not enough -- the union had P in it
            assert _get_current_understanding(s, USER, visibility=rule) == ""
        finally:
            s.close()


def test_digest_hint_gates_on_the_digest_union(db):
    """The renderer's evidence hint prints the digest's folded TEXT: with a
    digest whose union includes a revoked purpose, the hint falls back to
    the observation count."""
    from mirror_memory.core.models import EvidenceDigest
    from mirror_memory.render.renderer import _digest_hint_visible

    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I like {SENTINEL} and coffee.", P, "s1")
    belief = _beliefs(db)[0]
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("seed digest"):
        s = F()
        s.add(EvidenceDigest(user_id=USER, belief_id=belief.id,
                             summary=f"digest mentions {SENTINEL}",
                             source_purposes_json=json.dumps([P, Q])))
        s.commit()
        s.close()
    with tenant_scope(*TENANT):
        s = F()
        try:
            rule = provenance.visibility(s, USER)
            from mirror_memory.core.models import EvidenceDigest as ED

            digest = s.execute(select(ED)).scalars().one()
            assert _digest_hint_visible(digest, rule) is True
        finally:
            s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    with tenant_scope(*TENANT):
        s = F()
        try:
            rule = provenance.visibility(s, USER)
            from mirror_memory.core.models import EvidenceDigest as ED

            digest = s.execute(select(ED)).scalars().one()
            assert _digest_hint_visible(digest, rule) is False
        finally:
            s.close()


# ---------------------------------------------------------------------------
# batch-3 patch task 3: the residual entries
# ---------------------------------------------------------------------------


def test_correct_and_restore_treat_invisible_as_not_found_with_zero_writes(db):
    eng = _p_q_setup(db)
    p_belief = next(b for b in _beliefs(db) if SENTINEL in b.claim_text)
    events_before = len(_events(db))
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")

    # correct: None, exactly like a missing belief
    missing = eng.correct(user_id=USER, belief_id=999_999,
                          new_claim_text="x", purpose=Q)
    corrected = eng.correct(user_id=USER, belief_id=p_belief.id,
                            new_claim_text="rewritten", purpose=Q)
    assert corrected is None and missing is None
    # restore: False, exactly like a missing belief
    assert eng.restore_belief(user_id=USER, belief_id=999_999, purpose=Q) is False
    assert eng.restore_belief(user_id=USER, belief_id=p_belief.id, purpose=Q) is False
    assert len(_events(db)) == events_before, "an invisible belief must not be touched"
    # and the belief itself is unchanged
    assert SENTINEL in next(b for b in _beliefs(db) if b.id == p_belief.id).claim_text


def test_rejected_beliefs_are_visibility_filtered(db):
    eng = _p_q_setup(db)
    p_belief = next(b for b in _beliefs(db) if SENTINEL in b.claim_text)
    q_belief = next(b for b in _beliefs(db) if "Berlin" in b.claim_text)
    from mirror_memory.core.repository import reject_belief

    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("reject both"):
        s = F()
        provenance.bind(s, Q)
        reject_belief(s, USER, p_belief.id)
        reject_belief(s, USER, q_belief.id)
        s.commit()
        s.close()
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    rejected = eng.get_rejected(user_id=USER, purpose=Q) if hasattr(eng, "get_rejected") else None
    from mirror_memory.core.repository import list_rejected_beliefs

    with tenant_scope(*TENANT):
        s = F()
        try:
            rule = provenance.visibility(s, USER)
            rows = list_rejected_beliefs(s, USER, limit=5, visibility=rule)
        finally:
            s.close()
    assert [b.id for b in rows] == [q_belief.id]


def _events(db) -> list[BeliefEvent]:
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            return s.execute(select(BeliefEvent)).scalars().all()
        finally:
            s.close()


# ---------------------------------------------------------------------------
# pre-merge fix: forget_session removes EVERY purpose's summary rows
# ---------------------------------------------------------------------------


def test_forget_session_deletes_all_purpose_rows_and_every_read_is_clean(db):
    """The per-purpose split made one (user, session) hold several rows;
    forget must remove ALL of them -- unconditionally (any purpose, any
    authorization state, no visibility filter) -- and afterwards no read
    path may surface the session's text."""
    cfg = load_config("config/")
    cfg.session_summary_enabled = True
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    eng = MemoryEngine(config=cfg, database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM())
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I keep notes about {SENTINEL} in the attic.", P, "shared")
    _observe(eng, "I keep notes about gardening in the shed.", Q, "shared")

    from mirror_memory.core.models import SessionSummary

    def summary_rows():
        with tenant_scope(*TENANT):
            s = F()
            try:
                return s.execute(select(SessionSummary)).scalars().all()
            finally:
                s.close()

    assert len(summary_rows()) == 2

    # forget under Q (the session was observed under Q, but the call is
    # unconditional): both rows go
    assert eng.forget_session(user_id=USER, session_id="shared", purpose=Q) is True
    assert summary_rows() == []

    # observe again under P, then revoke P and forget with P revoked: the
    # forget still runs (unconditional) and still removes everything
    _observe(eng, f"more notes about {SENTINEL} upstairs.", P, "shared")
    assert len(summary_rows()) == 1
    eng.revoke_purpose(user_id=USER, purpose=P, actor="user", reason="withdrawn")
    assert eng.forget_session(user_id=USER, session_id="shared", purpose=Q) is True
    assert summary_rows() == []

    # nothing was ever forgotten: False, and no exception (a missing
    # session forgets nothing)
    assert eng.forget_session(user_id=USER, session_id="never-existed", purpose=Q) is False

    # the whole read surface carries zero sentinel text (P itself is
    # revoked: it must be refused outright)
    assert SENTINEL not in (eng.recall(user_id=USER, query="attic notes", purpose=Q) or "")
    with pytest.raises(PurposeDenied):
        eng.recall(user_id=USER, query="attic notes", purpose=P)
    beliefs = eng.get_beliefs(user_id=USER, purpose=Q)
    assert all(SENTINEL not in b.claim_text for b in beliefs)
    page = eng.export_page(user_id=USER, purpose=Q)
    assert SENTINEL not in json.dumps(page.to_dict(), default=str)


# ---------------------------------------------------------------------------
# M4 task 0: snapshot invalidation on deletion (sentinel tests)
# ---------------------------------------------------------------------------


def _worker_rebuild(db, purpose: str):
    """Run one worker job under *purpose* and return what it wrote."""
    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    cfg.worker.job_lease_seconds = 60
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            job = enqueue_job(s, USER, cfg, purpose=purpose)
            s.commit()
            token = _claim_job(s, job.id, lease_seconds=60)
            _process_single_job(s, s.get(EvolutionJob, job.id), cfg, token)
            s.commit()
            s.expire_all()
            return s.get(EvolutionJob, job.id)
        finally:
            s.close()


def test_belief_forget_invalidates_snapshots_rebuild_carries_no_sentinel(db):
    """A snapshot synthesized from a belief survives only until that
    belief is forgotten: forget_belief deletes ALL the user's snapshots in
    the deletion transaction (stronger than read-skip), the worker's
    rebuild contains none of the forgotten content, and no read path
    surfaces it."""
    eng = _engine(db)
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I like {SENTINEL} and coffee.", P, "s1")
    _observe(eng, "I live in Berlin and work as a nurse.", Q, "s2")
    _worker_rebuild(db, P)
    from mirror_memory.core.models import Snapshot

    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            assert s.execute(select(Snapshot)).scalars().all(), "sanity: a snapshot exists"
        finally:
            s.close()
    belief = next(b for b in _beliefs(db) if SENTINEL in b.claim_text)
    assert eng.forget_belief(user_id=USER, belief_id=belief.id, purpose=P) is True
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            assert s.execute(select(Snapshot)).scalars().all() == [], (
                "the deletion transaction must invalidate every snapshot"
            )
        finally:
            s.close()
    _worker_rebuild(db, Q)
    with tenant_scope(*TENANT):
        s = F()
        try:
            snaps = [x.content_json for x in s.execute(select(Snapshot)).scalars()]
            jobs = [j.error_code for j in s.execute(select(EvolutionJob)).scalars()]
        finally:
            s.close()
    assert all(SENTINEL not in text for text in snaps), snaps
    # the whole read surface too
    assert SENTINEL not in (eng.recall(user_id=USER, query=SENTINEL, purpose=Q) or "")
    assert SENTINEL not in json.dumps(eng.export_page(user_id=USER, purpose=Q).to_dict(), default=str)


def test_session_forget_leaves_no_summary_text_and_snapshots_unaffected(db):
    """Session scope: the raw text lives only in the summary rows; the
    forget removes every purpose's rows, and snapshots (which synthesize
    from beliefs, not summaries) are untouched by design."""
    cfg = load_config("config/")
    cfg.session_summary_enabled = True
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    eng = MemoryEngine(config=cfg, database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM())
    _grant(eng, P)
    _grant(eng, Q)
    _observe(eng, f"I keep notes about {SENTINEL} in the attic.", P, "shared")
    _observe(eng, "I keep notes about gardening in the shed.", Q, "shared")
    _worker_rebuild(db, Q)
    from mirror_memory.core.models import SessionSummary, Snapshot

    eng.forget_session(user_id=USER, session_id="shared", purpose=Q)
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            assert s.execute(select(SessionSummary)).scalars().all() == []
            snaps = [x.content_json for x in s.execute(select(Snapshot)).scalars()]
        finally:
            s.close()
    assert all(SENTINEL not in text for text in snaps)
    assert SENTINEL not in (eng.recall(user_id=USER, query="attic notes", purpose=Q) or "")
