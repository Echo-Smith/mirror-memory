"""M3 acceptance (batch-2 task 5 + batch-3 additions): after revocation --
and under a purpose that was never granted -- NO path returns memory, and
after a revocation the OTHER purposes see nothing of the revoked one.

Three scenarios, all driven by ENTRY_REGISTRY so a new entry is pulled in
automatically.  The suite runs on BOTH dialects: on PostgreSQL the engine
and the server run through the app role under RLS, and the scenario
grants are issued through the PRODUCT API (MemoryEngine.grant_purpose)
-- no test-side grant plumbing for the setup path (task 4: at least one
grant per acceptance run goes through product authorization).

* **Revoked everywhere**: memory collected under purpose X (a sentinel
  claim), then X revoked.  Every memory-reading entry -- the library
  methods, the server routes, export, and the worker's publish -- is
  called with X and must refuse without a trace of the sentinel in what
  it returns or raises.
* **I5 (wrong purpose)**: X granted, every entry called with Y (never
  granted).  All refused.
* **Cross-purpose (batch 3)**: X revoked, Q granted; every read entry
  runs under Q and must SUCCEED while carrying zero sentinel -- the A2
  visibility rule hides the X-only memory from every surface, and the
  worker under Q writes nothing carrying it.

The driver table must cover every registry entry flagged ``reads_memory``
plus the worker's job entry; an entry without a driver fails the test.

v2 merge-plan invariant mapping (I5 and the related ones):

| invariant | this suite |
|---|---|
| I5 purpose mismatch refused on every path | ``test_i5_wrong_purpose_is_refused_on_every_path`` (here) |
| revocation immediate, every path | ``test_after_revocation_no_path_returns_memory`` (here) |
| cross-purpose hiding after revocation | ``test_cross_purpose_q_reads_nothing_of_the_revoked_p`` (here) |
| I2 / PI revoke between claim and publish | tests/test_purpose_revocation.py (PG, 50 rounds), tests/test_purpose_revocation_sqlite.py |
| refusal issues zero business SQL | tests/test_purpose_entries.py::test_refused_recall_issues_only_authorization_queries |
| missing purpose refused before any SQL | tests/test_purpose_entries.py::test_missing_purpose_refused_before_any_sql |
| every entry registered or explicitly ungated | tests/test_purpose_entries.py::test_every_public_engine_method_is_registered_or_explicitly_ungated |
| every read entry audited | tests/test_access_audit.py::test_every_memory_reading_entry_in_the_registry_is_audited |
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core.authorization import ENTRY_REGISTRY, PurposeDenied
from mirror_memory.core.models import Base, Belief, Evidence, EvolutionJob, Snapshot
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.server import create_app, register_tenant_key
from mirror_memory.worker.evolution import _claim_job, _process_single_job, enqueue_job
from tests.conftest import refusing_authz

USER = "u-accept"
TENANT = ("default", "default")
X, Y, Q = "granted-x", "never-y", "other-q"
SENTINEL = "SENTINEL-SECRET-4242"
WORKER_ENTRY = "mirror_memory.worker.evolution._process_single_job"


class _FakeLLM:
    def generate(self, **kw):
        # invents nothing: a sentinel in anything the job writes can only
        # have come from memory the job read
        return '{"patterns": [], "support_policy": {}}'


@pytest.fixture(params=["sqlite", "postgres"])
def db_url(request, tmp_path):
    """The acceptance suite runs on both dialects; the postgres param is
    the suite's RLS-protected app-role runtime (skipped on the SQLite
    leg).  The pg fixture is pulled LAZILY so the sqlite param does not
    demand a database."""
    if request.param == "sqlite":
        eng = create_engine(f"sqlite:///{tmp_path / 'accept.db'}")
        Base.metadata.create_all(eng)
        yield str(eng.url)
        eng.dispose()
    else:
        yield request.getfixturevalue("pg_app_url")


def _config():
    cfg = load_config("config/")
    cfg.llm_client = _FakeLLM()
    cfg.worker.job_lease_seconds = 60
    return cfg


def _engine(db_url) -> MemoryEngine:
    F = sessionmaker(bind=create_engine(db_url))
    with tenant_scope(*TENANT), fixture_write_scope("enable memory"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    return MemoryEngine(config=_config(), database_url=db_url,
                        tenant_id=TENANT[0], app_id=TENANT[1], llm_client=_FakeLLM())


def _memory_counts(db_url) -> tuple[int, int, int]:
    F = sessionmaker(bind=create_engine(db_url))
    with tenant_scope(*TENANT):
        s = F()
        try:
            return (
                len(s.execute(select(Belief).where(Belief.user_id == USER)).scalars().all()),
                len(s.execute(select(Evidence).where(Evidence.user_id == USER)).scalars().all()),
                len(s.execute(select(Snapshot).where(Snapshot.user_id == USER)).scalars().all()),
            )
        finally:
            s.close()


def _first_belief_id(db_url) -> int:
    F = sessionmaker(bind=create_engine(db_url))
    with tenant_scope(*TENANT):
        s = F()
        try:
            return s.execute(select(Belief).where(Belief.user_id == USER).order_by(Belief.id)).scalars().first().id
        finally:
            s.close()


def _populate_under_x(eng: MemoryEngine) -> None:
    # the setup grant goes through the PRODUCT API: on the postgres leg
    # this INSERT runs as the app role under RLS in the consent write scope
    eng.grant_purpose(user_id=USER, purpose=X, operations=[
        "observe", "recall", "correct", "forget", "export", "explain", "extract"], issuer="t")
    eng.observe(user_id=USER, session_id="s1", text=f"I like {SENTINEL} and coffee.", purpose=X)
    eng.observe(user_id=USER, session_id="s2", text=f"I live in Berlin, {SENTINEL} is my street.", purpose=X)


# ---------------------------------------------------------------------------
# drivers (library + server + worker), keyed by ENTRY_REGISTRY key
# ---------------------------------------------------------------------------


def _library_drivers(eng: MemoryEngine, purpose: str, belief_id: int) -> dict[str, object]:
    return {
        "mirror_memory.api.MemoryEngine.recall": lambda: eng.recall(user_id=USER, query=SENTINEL, purpose=purpose),
        "mirror_memory.api.MemoryEngine.panel": lambda: eng.panel(user_id=USER, purpose=purpose),
        "mirror_memory.api.MemoryEngine.get_beliefs": lambda: eng.get_beliefs(user_id=USER, purpose=purpose),
        "mirror_memory.api.MemoryEngine.get_verification_candidates": lambda: eng.get_verification_candidates(
            user_id=USER, purpose=purpose),
        "mirror_memory.api.MemoryEngine.belief_history": lambda: eng.belief_history(
            user_id=USER, belief_id=belief_id, purpose=purpose),
        "mirror_memory.api.MemoryEngine.explain": lambda: eng.explain(
            user_id=USER, belief_id=belief_id, purpose=purpose),
        "mirror_memory.api.MemoryEngine.export_page": lambda: eng.export_page(user_id=USER, purpose=purpose),
        "mirror_memory.api.MemoryEngine.export": lambda: list(eng.export(user_id=USER, purpose=purpose)),
    }


def _server_drivers(client: TestClient, purpose: str, belief_id: int) -> dict[str, object]:
    h = {"Authorization": "Bearer key-a", "X-MM-Purpose": purpose}
    return {
        "mirror_memory.server.recall": lambda: client.post(
            "/recall", json={"user_id": USER, "query": SENTINEL}, headers=h),
        "mirror_memory.server.panel": lambda: client.get("/panel", params={"user_id": USER}, headers=h),
        "mirror_memory.server.beliefs": lambda: client.get("/beliefs", params={"user_id": USER}, headers=h),
        "mirror_memory.server.belief_by_id": lambda: client.get(
            f"/belief/{belief_id}", params={"user_id": USER}, headers=h),
        "mirror_memory.server.stats": lambda: client.get("/stats", params={"user_id": USER}, headers=h),
        "mirror_memory.server.export": lambda: client.get("/export", params={"user_id": USER}, headers=h),
    }


def _beliefs_all(db_url) -> list[Belief]:
    F = sessionmaker(bind=create_engine(db_url))
    with tenant_scope(*TENANT):
        s = F()
        try:
            return s.execute(select(Belief).where(Belief.user_id == USER)).scalars().all()
        finally:
            s.close()


def _worker_publish(db_url, purpose: str) -> EvolutionJob:
    """Enqueue a job carrying *purpose*, claim it, run the publish; return
    the job row as it ends up."""
    cfg = _config()
    F = sessionmaker(bind=create_engine(db_url))
    with tenant_scope(*TENANT):
        s = F()
        try:
            job = enqueue_job(s, USER, cfg, purpose=purpose)
            assert job is not None
            s.commit()
            job_id = job.id
            token = _claim_job(s, job_id, lease_seconds=60)
            row = s.get(EvolutionJob, job_id)
            _process_single_job(s, row, cfg, token)
            s.commit()
            s.expire_all()
            return s.get(EvolutionJob, job_id)
        finally:
            s.close()


def _assert_registry_covered(drivers: dict[str, object]) -> None:
    flagged = {k for k, v in ENTRY_REGISTRY.items() if v.get("reads_memory")}
    expected = flagged | {WORKER_ENTRY}
    assert set(drivers) == expected, (
        f"registry entries without an acceptance driver: {sorted(expected - set(drivers))}; "
        f"drivers for unknown entries: {sorted(set(drivers) - expected)}"
    )


def _run_all_refused(db_url, eng, drivers: dict[str, object], *, purpose: str, reason: str) -> None:
    before = _memory_counts(db_url)
    for key, run in drivers.items():
        if key == WORKER_ENTRY:
            job = run()
            assert job.status == "cancelled", (key, job.status, job.error_code)
            assert job.error_code == reason, (key, job.error_code)
            continue
        if key.startswith("mirror_memory.server."):
            r = run()
            assert r.status_code == 403, (key, r.status_code, r.text)
            assert SENTINEL not in r.text, key
            body = r.json()
            assert set(body) == {"detail"}, (key, body)
            assert reason in body["detail"], (key, body)
            continue
        with pytest.raises(PurposeDenied) as info:
            run()
        assert info.value.decision.reason == reason, (key, info.value.decision.reason)
        assert SENTINEL not in str(info.value), key
    assert _memory_counts(db_url) == before, "a refused path must not write memory either"


# ---------------------------------------------------------------------------
# scenario 1: revoked everywhere
# ---------------------------------------------------------------------------


def test_after_revocation_no_path_returns_memory(db_url):
    eng = _engine(db_url)
    _populate_under_x(eng)
    belief_id = _first_belief_id(db_url)
    assert SENTINEL in (eng.recall(user_id=USER, query=SENTINEL, purpose=X) or ""), "sanity: X sees it before"
    eng.revoke_purpose(user_id=USER, purpose=X, actor="user", reason="withdrawn")

    from mirror_memory import server as server_mod

    server_mod._tenant_key_registry.clear()
    register_tenant_key("key-a", *TENANT, purposes={X, Y})
    app = create_app(config=_config(), database_url=db_url)
    try:
        with TestClient(app) as client:
            drivers = {
                **_library_drivers(eng, X, belief_id),
                **_server_drivers(client, X, belief_id),
                WORKER_ENTRY: lambda: _worker_publish(db_url, X),
            }
            _assert_registry_covered(drivers)
            _run_all_refused(db_url, eng, drivers, purpose=X, reason="revoked")
    finally:
        server_mod._tenant_key_registry.clear()


# ---------------------------------------------------------------------------
# scenario 2: I5 -- granted X, called with Y
# ---------------------------------------------------------------------------


def test_i5_wrong_purpose_is_refused_on_every_path(db_url):
    eng = _engine(db_url)
    _populate_under_x(eng)
    belief_id = _first_belief_id(db_url)

    from mirror_memory import server as server_mod

    server_mod._tenant_key_registry.clear()
    register_tenant_key("key-a", *TENANT, purposes={X, Y})
    app = create_app(config=_config(), database_url=db_url)
    try:
        # the harness would lazily grant Y on first touch; the whole
        # scenario runs with that off so Y is a genuine no_grant
        with refusing_authz(), TestClient(app) as client:
            drivers = {
                **_library_drivers(eng, Y, belief_id),
                **_server_drivers(client, Y, belief_id),
                WORKER_ENTRY: lambda: _worker_publish(db_url, Y),
            }
            _assert_registry_covered(drivers)
            _run_all_refused(db_url, eng, drivers, purpose=Y, reason="no_grant")
            # and X still works: the refusals were about Y, not the user
            assert SENTINEL in (eng.recall(user_id=USER, query=SENTINEL, purpose=X) or "")
    finally:
        server_mod._tenant_key_registry.clear()


# ---------------------------------------------------------------------------
# scenario 3 (batch 3): X revoked, Q reads -- success, zero sentinel
# ---------------------------------------------------------------------------


def test_cross_purpose_q_reads_nothing_of_the_revoked_p(db_url):
    eng = _engine(db_url)
    _populate_under_x(eng)
    belief_id = _first_belief_id(db_url)
    counts_before = _memory_counts(db_url)
    eng.revoke_purpose(user_id=USER, purpose=X, actor="user", reason="withdrawn")
    # Q's grant: the product API again (app role + RLS on the postgres leg)
    eng.grant_purpose(user_id=USER, purpose=Q, operations=[
        "observe", "recall", "correct", "forget", "export", "explain", "extract"], issuer="t")

    from mirror_memory import server as server_mod

    server_mod._tenant_key_registry.clear()
    register_tenant_key("key-a", *TENANT, purposes={X, Y, Q})
    app = create_app(config=_config(), database_url=db_url)
    try:
        with TestClient(app) as client:
            drivers = {
                **_library_drivers(eng, Q, belief_id),
                **_server_drivers(client, Q, belief_id),
            }
            _assert_registry_covered(drivers | {WORKER_ENTRY: object()})
            for key, run in drivers.items():
                result = run()
                if hasattr(result, "status_code"):
                    # 200 (a visible-result surface) or 404 (an invisible
                    # belief) -- never 403 (Q is authorized) and never a leak
                    assert result.status_code in (200, 404), (key, result.status_code, result.text)
                    assert SENTINEL not in result.text, key
                elif key.endswith(".export"):
                    pages = result
                    for page in pages:
                        assert SENTINEL not in page.to_dict().__str__(), key
                else:
                    assert SENTINEL not in repr(result), key
            # the worker under Q: the job runs (Q is authorized) but reads
            # nothing X-only, so nothing X-only is republished.  The X
            # belief ROWS persist (hidden, not deleted) -- only what the
            # job itself writes may carry content.
            beliefs_before = {b.id for b in _beliefs_all(db_url)}
            job = _worker_publish(db_url, Q)
            assert job.status in ("completed", "cancelled"), (job.status, job.error_code)
            F = sessionmaker(bind=create_engine(db_url))
            with tenant_scope(*TENANT):
                s = F()
                try:
                    new_beliefs = [
                        b.claim_text for b in s.execute(select(Belief).where(Belief.user_id == USER)).scalars()
                        if b.id not in beliefs_before
                    ]
                    snapshots = [
                        snap.content_json for snap in s.execute(select(Snapshot)).scalars()
                        if snap.source_purpose == Q
                    ]
                finally:
                    s.close()
            assert all(SENTINEL not in text for text in new_beliefs), new_beliefs
            assert all(SENTINEL not in text for text in snapshots), snapshots
            # and Q's own memory (none seeded here) simply reads empty
            assert f"{SENTINEL}" not in (eng.recall(user_id=USER, query=SENTINEL, purpose=Q) or "")
    finally:
        server_mod._tenant_key_registry.clear()
    assert _memory_counts(db_url)[:2] == counts_before[:2], "reads must not create beliefs or evidence"
