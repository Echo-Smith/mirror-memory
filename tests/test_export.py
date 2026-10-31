"""M3 batch-2 task 3: export and the explain authorization source.

Export is ``mm-export/1``: beliefs, evidence and ledger events in fixed
order, keyset-paginated, every item with its source purpose (beliefs with
the derived set over their evidence).  Every page is its own transaction,
authorized again for ``export`` and audited; a revocation during the
export refuses the next page.  Explain reports the authorization that
allowed the call and the source purposes of the belief's evidence.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import export as export_mod
from mirror_memory.core.authorization import PurposeDenied
from mirror_memory.core.models import AccessAudit, Base, Belief, BeliefEvent, Evidence
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.server import create_app, register_tenant_key

USER = "u-export"
TENANT = ("default", "default")


@pytest.fixture
def db(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path}/export.db")
    Base.metadata.create_all(eng)
    from mirror_memory.core import schema_migration as _sm_mint

    _sm_mint.ensure_watermark(eng)
    yield eng
    eng.dispose()


def _engine(db) -> MemoryEngine:
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    return MemoryEngine(config=load_config("config/"), database_url=str(db.url),
                        tenant_id=TENANT[0], app_id=TENANT[1])


def _populate(eng: MemoryEngine) -> None:
    """Memory collected under two purposes: 'api' and 'support'."""
    eng.observe(user_id=USER, session_id="s1", text="I prefer concise answers and I like coffee.", purpose="api")
    eng.observe(user_id=USER, session_id="s2", text="I live in Berlin and work as a nurse.", purpose="support")
    eng.observe(user_id=USER, session_id="s3", text="Actually I don't like coffee anymore, I prefer tea.", purpose="api")


def _counts(db) -> dict[str, int]:
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            return {
                "beliefs": len(s.execute(select(Belief).where(Belief.user_id == USER)).scalars().all()),
                "evidence": len(s.execute(select(Evidence).where(Evidence.user_id == USER)).scalars().all()),
                "events": len(s.execute(select(BeliefEvent).where(BeliefEvent.user_id == USER)).scalars().all()),
            }
        finally:
            s.close()


def _audit_rows(db):
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            return s.execute(select(AccessAudit).order_by(AccessAudit.id)).scalars().all()
        finally:
            s.close()


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


def test_export_covers_the_three_sections_with_source_purposes(db):
    eng = _engine(db)
    _populate(eng)
    pages = list(eng.export(user_id=USER, purpose="api", page_size=1000))
    assert [p.section for p in pages] == ["beliefs", "evidence", "events"]
    assert {p.schema_version for p in pages} == {"mm-export/1"}
    assert pages[-1].next_cursor is None
    counts = _counts(db)
    assert {p.section: len(p.items) for p in pages} == counts
    assert all(counts[k] > 0 for k in counts), counts

    beliefs, evidence, events = (p.items for p in pages)
    assert {e["source_purpose"] for e in evidence} == {"api", "support"}
    assert {e["source_purpose"] for e in events} == {"api", "support"}
    for item in beliefs:
        assert set(item) >= {"id", "dimension", "key", "claim_text", "confidence", "layer",
                             "status", "source", "created_at", "source_purposes"}
        assert isinstance(item["source_purposes"], list)
    # a belief's derived provenance is the set over its linked evidence
    seen = {sp for item in beliefs for sp in item["source_purposes"]}
    assert seen <= {"api", "support"} and seen, seen
    for item in evidence:
        assert set(item) >= {"id", "ref", "session_id", "content", "source_type",
                             "extraction_method", "authority", "source_purpose", "observed_at"}
    for item in events:
        assert set(item) >= {"id", "belief_id", "event_type", "evidence_ids", "detail",
                             "source_purpose", "created_at"}
    # the authorization that allowed each page, not the grant itself
    assert all(p.authorization["purpose"] == "api" and isinstance(p.authorization["grant_version"], int)
               for p in pages)
    assert all(isinstance(p.revision, int) for p in pages)
    assert "to_dict" in dir(pages[0]) and pages[0].to_dict()["section"] == "beliefs"


def test_pages_are_bounded_and_cover_every_item_exactly_once(db):
    eng = _engine(db)
    _populate(eng)
    counts = _counts(db)
    pages = list(eng.export(user_id=USER, purpose="api", page_size=2))
    assert all(len(p.items) <= 2 for p in pages)
    by_section: dict[str, list[int]] = {}
    for p in pages:
        by_section.setdefault(p.section, []).extend(i["id"] for i in p.items)
    for section, ids in by_section.items():
        assert ids == sorted(ids) and len(ids) == len(set(ids)) == counts[section], section
    # cursors are opaque but parseable, and a full page points into the same section
    full = [p for p in pages if len(p.items) == 2 and p.next_cursor]
    assert full
    for p in full:
        section, last = export_mod.parse_cursor(p.next_cursor)
        assert section == p.section and last == p.items[-1]["id"]


def test_cursor_validation(db):
    eng = _engine(db)
    for bad in ("nope", "beliefs", "beliefs:x", "beliefs:-1", "grants:0"):
        with pytest.raises(export_mod.InvalidCursor):
            eng.export_page(user_id=USER, purpose="api", cursor=bad)
    with pytest.raises(ValueError):
        eng.export_page(user_id=USER, purpose="api", page_size=0)
    with pytest.raises(ValueError):
        eng.export_page(user_id=USER, purpose="api", page_size=export_mod.MAX_PAGE_SIZE + 1)


def test_empty_user_still_walks_the_sections(db):
    eng = _engine(db)
    pages = list(eng.export(user_id="nobody", purpose="api"))
    assert [(p.section, p.items, p.next_cursor) for p in pages] == [
        ("beliefs", [], "evidence:0"), ("evidence", [], "events:0"), ("events", [], None),
    ]


# ---------------------------------------------------------------------------
# authorization: the export operation, and re-authorization per page
# ---------------------------------------------------------------------------


def test_export_needs_the_export_operation(db):
    eng = _engine(db)
    _populate(eng)
    eng.grant_purpose(user_id=USER, purpose="reader", operations=["recall", "explain"], issuer="t")
    with pytest.raises(PurposeDenied) as info:
        eng.export_page(user_id=USER, purpose="reader")
    assert info.value.decision.reason == "operation_not_allowed"
    assert info.value.operation == "export"
    row = _audit_rows(db)[-1]
    assert (row.decision, row.reason, row.operation, row.entry) == (
        "denied", "operation_not_allowed", "export", "mirror_memory.api.MemoryEngine.export_page")


def test_revocation_mid_export_refuses_the_next_page(db):
    eng = _engine(db)
    _populate(eng)
    eng.grant_purpose(user_id=USER, purpose="exporter", operations=["export"], issuer="t")
    first = eng.export_page(user_id=USER, purpose="exporter", page_size=1)
    assert first.items and first.next_cursor
    eng.revoke_purpose(user_id=USER, purpose="exporter", actor="t", reason="user withdrew")
    with pytest.raises(PurposeDenied) as info:
        eng.export_page(user_id=USER, purpose="exporter", cursor=first.next_cursor, page_size=1)
    assert info.value.decision.reason == "revoked"
    rows = _audit_rows(db)
    assert [(r.decision, r.reason) for r in rows[-2:]] == [("allowed", ""), ("denied", "revoked")]
    assert json.loads(rows[-2].memory_ids_json) == [i["id"] for i in first.items]


def test_the_stream_stops_at_a_revocation(db):
    """The generator form: pages before the revocation come out, the page
    after it raises -- nothing after the revoke is delivered."""
    eng = _engine(db)
    _populate(eng)
    eng.grant_purpose(user_id=USER, purpose="exporter", operations=["export"], issuer="t")
    delivered = []
    stream = eng.export(user_id=USER, purpose="exporter", page_size=1)
    delivered.append(next(stream))
    delivered.append(next(stream))
    eng.revoke_purpose(user_id=USER, purpose="exporter", actor="t", reason="user withdrew")
    with pytest.raises(PurposeDenied):
        next(stream)
    assert len(delivered) == 2
    assert {r.entry for r in _audit_rows(db)[-3:]} == {"mirror_memory.api.MemoryEngine.export"}


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


def test_server_export_pages_and_stops_after_revocation(db):
    eng = _engine(db)
    _populate(eng)
    from mirror_memory import server as server_mod

    server_mod._tenant_key_registry.clear()
    register_tenant_key("key-a", *TENANT, purposes={"exporter"})
    eng.grant_purpose(user_id=USER, purpose="exporter", operations=["export"], issuer="t")
    app = create_app(config=load_config("config/"), database_url=str(db.url))
    h = {"Authorization": "Bearer key-a", "X-MM-Purpose": "exporter"}
    try:
        with TestClient(app) as client:
            r = client.get("/export", params={"user_id": USER, "page_size": 1}, headers=h)
            assert r.status_code == 200, r.text
            page = r.json()
            assert page["schema_version"] == "mm-export/1" and page["section"] == "beliefs"
            assert len(page["items"]) == 1 and page["next_cursor"]
            assert page["authorization"]["purpose"] == "exporter"
            r = client.get("/export", params={"user_id": USER, "cursor": "bogus"}, headers=h)
            assert r.status_code == 400
            eng.revoke_purpose(user_id=USER, purpose="exporter", actor="t", reason="withdrawn")
            r = client.get("/export", params={"user_id": USER, "cursor": page["next_cursor"]}, headers=h)
            assert r.status_code == 403
            assert "revoked" in r.json()["detail"]
    finally:
        server_mod._tenant_key_registry.clear()
    assert _audit_rows(db)[-1].entry == "mirror_memory.server.export"
    assert _audit_rows(db)[-1].decision == "denied"


# ---------------------------------------------------------------------------
# explain: authorization source + evidence source purposes
# ---------------------------------------------------------------------------


def test_explain_reports_the_authorization_and_the_evidence_source_purposes(db):
    eng = _engine(db)
    _populate(eng)
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT):
        s = F()
        try:
            beliefs = s.execute(select(Belief).where(Belief.user_id == USER).order_by(Belief.id)).scalars().all()
        finally:
            s.close()
    explained = eng.explain(user_id=USER, belief_id=beliefs[0].id, purpose="support")
    assert explained is not None
    assert explained["authorization"]["purpose"] == "support"
    assert isinstance(explained["authorization"]["grant_version"], int)
    assert isinstance(explained["evidence_source_purposes"], list)
    assert set(explained["evidence_source_purposes"]) <= {"api", "support", ""}
    # a belief collected under one purpose names exactly that purpose
    first_belief_purposes = {
        b.id: set(eng.explain(user_id=USER, belief_id=b.id, purpose="api")["evidence_source_purposes"])
        for b in beliefs
    }
    assert any(v == {"api"} for v in first_belief_purposes.values()), first_belief_purposes
    assert any(v == {"support"} for v in first_belief_purposes.values()), first_belief_purposes
    # a miss has no authorization block to leak through
    assert eng.explain(user_id=USER, belief_id=999_999, purpose="api") is None
