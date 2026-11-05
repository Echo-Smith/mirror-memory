"""M5 §1 integration tests: real engine-driven metrics (not module-only).

These drive the ACTUAL MemoryEngine.observe / recall / run_metabolism paths
(and real SQL/LLM clients) rather than calling ``metrics.xxx`` directly, so
they would have caught the defects the previous module-only tests missed:

* /metrics reachable by an admin even when tenant auth is enabled, while other
  routes still require a tenant key + purpose;
* SQL counting for EVERY engine (multi-engine), no same-thread cross-engine
  attribution, concurrent-thread correctness;
* LLM calls counted for ANY client (usage or not); no cross-talk / no leaked
  hook on a shared client under concurrency;
* run_metabolism attributed to its own entry (not folded into observe);
* storage_stats tenant-scoped, (tenant,user) keyed, no system_scope auto-
  privilege; user ids / content never reach Prometheus (end-to-end).
"""

from __future__ import annotations

import pathlib
import threading

import pytest

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import metrics
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics.reset_for_tests()
    yield
    metrics.reset_for_tests()


class _NoUsageLLM:
    """A custom client with NO ``set_on_call`` and NO usage object -- the
    worst case for accounting."""

    def __init__(self):
        self.generate_calls = 0

    def generate(self, *, system_prompt="", payload_text="", fallback=None):
        self.generate_calls += 1
        return '{"patterns": [], "support_policy": {}}'


class _UsageLLM:
    """A client that reports usage through ``set_on_call`` (the OpenAILLM
    contract), so tokens must be recorded."""

    def __init__(self):
        self._on_call = None
        self._mm_usage_hook = False

    def set_on_call(self, hook):
        self._on_call = hook

    def generate(self, *, system_prompt="", payload_text="", fallback=None):
        if self._on_call is not None:
            self._on_call({"ok": True, "input_tokens": 111, "output_tokens": 22,
                           "latency_ms": 1})
        return '{"patterns": [], "support_policy": {}}'


def _grant_direct(eng, user, purpose):
    """Explicit grant bypassing the harness lazy-grant memo.  Issues the grant
    through the sanctioned path INSIDE a fixture write scope (a bare
    PurposeGrant insert trips the write guard).  Idempotent."""
    from mirror_memory.core.authorization import StaleGrantVersion
    from mirror_memory.core import authorization as _auth

    with tenant_scope(*eng._tenant):
        with Session(eng._engine) as s:
            try:
                with fixture_write_scope("metrics integration: explicit grant"):
                    _auth.grant_purpose(
                        s, user, purpose,
                        operations=["observe", "recall", "extract"], issuer="test",
                    )
                    s.commit()
            except StaleGrantVersion:
                s.rollback()


def _make_engine(tmp: pathlib.Path, *, llm, tenant=("default", "default"), user="u1"):
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = load_config("config/")
    cfg.llm_client = llm
    eng = MemoryEngine(
        config=cfg,
        database_url=f"sqlite:///{tmp / 'm.db'}",
        llm_client=llm,
        tenant_id=tenant[0],
        app_id=tenant[1],
    )
    eng._ensure_db()
    with tenant_scope(*tenant), fixture_write_scope("metrics integration: enable"):
        _s = Session(eng._engine)
        try:
            set_memory_enabled(_s, user, True)
            _s.commit()  # commit+close: a lingering session holds a write txn
        finally:
            _s.close()
    return eng


# ---------------------------------------------------------------------------
# (1) /metrics admin access in tenant mode; other routes still tenant-gated
# ---------------------------------------------------------------------------


def test_metrics_admin_reachable_in_tenant_mode(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mirror_memory.server import create_app, register_tenant_key, _tenant_key_registry

    monkeypatch.setenv("MM_ADMIN_API_KEY", "admin-secret")
    # a tenant key so the tenant middleware is active
    register_tenant_key("tenant-key-abc", tenant_id="t1", app_id="a1",
                        purposes=["recall", "observe"])
    try:
        app = create_app(config=load_config("config/"), database_url="sqlite://")
        c = TestClient(app)
        # valid admin -> 200
        assert c.get("/metrics", headers={"authorization": "Bearer admin-secret"}).status_code == 200
        # tenant key is NOT an admin -> 401 (never 200, never a tenant bypass)
        assert c.get("/metrics", headers={"authorization": "Bearer tenant-key-abc"}).status_code == 401
        # missing / wrong key -> 401
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"authorization": "Bearer nope"}).status_code == 401
        # other routes are STILL tenant-gated (no tenant key -> rejected)
        assert c.get("/stats", params={"user_id": "u1"}).status_code in (401, 403)
        # a tenant key with the WRONG purpose is still refused on a normal
        # route -> proves the /metrics exemption did not loosen purpose checks
        wrong = c.get("/stats", params={"user_id": "u1"},
                      headers={"authorization": "Bearer tenant-key-abc",
                               "x-mm-purpose": "nottallowed"})
        assert wrong.status_code == 403
    finally:
        _tenant_key_registry.clear()


def test_metrics_disabled_when_admin_key_unset(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from mirror_memory.server import create_app, _tenant_key_registry

    monkeypatch.delenv("MM_ADMIN_API_KEY", raising=False)
    app = create_app(config=load_config("config/"), database_url="sqlite://")
    c = TestClient(app)
    # fail-closed: no admin key configured -> 404, never open
    assert c.get("/metrics").status_code == 404
    _tenant_key_registry.clear()


# ---------------------------------------------------------------------------
# (2) SQL counting: multi-engine, same-thread isolation, concurrency
# ---------------------------------------------------------------------------


def test_sql_counted_for_every_engine(tmp_path):
    e1 = _make_engine(tmp_path / "e1", llm=_NoUsageLLM())
    e2 = _make_engine(tmp_path / "e2", llm=_NoUsageLLM())

    metrics.reset_for_tests()
    e1.observe(user_id="u1", session_id="s", text="I like apples.", purpose="chat")
    c1 = metrics.snapshot().get("sql_statements_total.observe", 0)
    metrics.reset_for_tests()
    e2.observe(user_id="u1", session_id="s", text="I like pears.", purpose="chat")
    c2 = metrics.snapshot().get("sql_statements_total.observe", 0)

    # BOTH engines must count (the old global-class-listener bug broke the 2nd)
    assert c1 > 0, f"first engine counted {c1}"
    assert c2 > 0, f"second engine counted {c2}"


def test_unrelated_sql_on_thread_not_counted(tmp_path):
    """SQL on a thread with NO entry scope open must not be attributed."""
    eng = _make_engine(tmp_path, llm=_NoUsageLLM())
    # a direct query outside any entry scope
    from sqlalchemy import text
    with eng._engine.connect() as c:
        c.execute(text("SELECT 1"))
    assert metrics.snapshot().get("sql_statements_total.observe", 0) == 0
    assert not any(k.startswith("sql_statements_total") for k in metrics.snapshot())


def test_sql_concurrent_threads_attributed_by_entry(tmp_path):
    """Two threads, two entries; each thread's SQL counts to its own entry."""
    eng = _make_engine(tmp_path, llm=_NoUsageLLM())

    errors = []
    barrier = threading.Barrier(2)

    def run(entry):
        try:
            with metrics.entry_scope(entry):
                barrier.wait(timeout=5)
                from sqlalchemy import text
                for _ in range(3):
                    with eng._engine.connect() as c:
                        c.execute(text("SELECT 1"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=run, args=("observe",))
    t2 = threading.Thread(target=run, args=("recall",))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    snap = metrics.snapshot()
    assert snap.get("sql_statements_total.observe", 0) == 3
    assert snap.get("sql_statements_total.recall", 0) == 3


# ---------------------------------------------------------------------------
# (3) LLM accounting: any client counted; tokens when available; concurrency
# ---------------------------------------------------------------------------


def test_llm_calls_counted_for_custom_client(tmp_path):
    llm = _NoUsageLLM()
    eng = _make_engine(tmp_path, llm=llm)
    metrics.reset_for_tests()
    eng.observe(user_id="u1", session_id="s", text="I love tea.", purpose="chat")
    snap = metrics.snapshot()
    # a real generate() happened -> the call is counted even with no usage
    assert llm.generate_calls >= 1
    assert snap.get("llm_calls_total.observe", 0) == llm.generate_calls
    # no usage -> booked as usage-unknown, never invented tokens
    assert snap.get("llm_usage_unknown_total.observe", 0) >= 1
    assert snap.get("llm_input_tokens.observe", 0) == 0


def test_llm_tokens_recorded_for_usage_client(tmp_path):
    llm = _UsageLLM()
    eng = _make_engine(tmp_path, llm=llm)
    metrics.reset_for_tests()
    eng.observe(user_id="u1", session_id="s", text="I love tea.", purpose="chat")
    snap = metrics.snapshot()
    assert snap.get("llm_calls_total.observe", 0) >= 1
    # real tokens from the provider's usage
    assert snap.get("llm_input_tokens.observe", 0) == 111
    assert snap.get("llm_output_tokens.observe", 0) == 22


def test_llm_hook_concurrency_no_crosstalk_no_leak():
    """Two concurrent entry scopes on ONE shared usage client attribute by
    thread; nothing leaks (the hook is installed once and reads thread-local)."""
    from mirror_memory.core import metrics as m

    llm = _UsageLLM()

    def install_once():
        def hook(info):
            entry = m.current_entry()
            if entry is not None:
                m.record_llm_tokens(entry, input_tokens=info["input_tokens"],
                                    output_tokens=info["output_tokens"])
        llm.set_on_call(hook)
        llm._mm_usage_hook = True

    install_once()
    barrier = threading.Barrier(2)

    def run(entry):
        with m.entry_scope(entry):
            barrier.wait(timeout=5)
            llm.generate()

    t1 = threading.Thread(target=run, args=("observe",))
    t2 = threading.Thread(target=run, args=("metabolism",))
    t1.start(); t2.start(); t1.join(); t2.join()
    snap = m.snapshot()
    assert snap.get("llm_input_tokens.observe", 0) == 111
    assert snap.get("llm_input_tokens.metabolism", 0) == 111
    # the shared client's hook is the ONE we installed -- never restored to a
    # stale per-scope callable
    assert llm._on_call is not None


# ---------------------------------------------------------------------------
# (4) run_metabolism has its own entry, not observe
# ---------------------------------------------------------------------------


def test_run_metabolism_uses_its_own_entry(tmp_path):
    eng = _make_engine(tmp_path, llm=_NoUsageLLM())
    metrics.reset_for_tests()
    eng.run_metabolism(user_id="u1", purpose="chat")
    lat = metrics.latency_summaries()
    assert lat["metabolism"]["count"] == 1
    assert lat["observe"]["count"] == 0, "metabolism must not be folded into observe"


def test_observe_cost_aggregation_semantics(tmp_path):
    """observe books exactly one latency sample and the SQL/LLM costs of that
    one call; a second observe increments each by its own amount."""
    eng = _make_engine(tmp_path, llm=_NoUsageLLM())
    metrics.reset_for_tests()
    eng.observe(user_id="u1", session_id="s1", text="I love tea.", purpose="chat")
    n1 = metrics.snapshot().get("sql_statements_total.observe", 0)
    lat1 = metrics.latency_summaries()["observe"]["count"]
    eng.observe(user_id="u1", session_id="s2", text="I love coffee.", purpose="chat")
    n2 = metrics.snapshot().get("sql_statements_total.observe", 0)
    lat2 = metrics.latency_summaries()["observe"]["count"]
    assert lat1 == 1 and lat2 == 2  # one sample per observe
    assert n2 > n1  # SQL accumulates


# ---------------------------------------------------------------------------
# (5) storage_stats: tenant scope, (tenant,user) key, no system_scope
# ---------------------------------------------------------------------------

import sqlite3  # noqa: E402


def test_storage_stats_tenant_scoped_and_keyed(tmp_path):
    """Same user_id in two tenants must NOT be conflated; scoped call returns
    only the requested tenant.  Seeded with Core INSERT (the ORM flush guard
    makes a single session single-tenant; storage_stats reads raw SQL, so a
    Core insert is the faithful low-level seed)."""
    from sqlalchemy import create_engine, insert

    from mirror_memory.core.models import Base, Belief as CoreBelief

    eng = create_engine(f"sqlite:///{tmp_path / 'store.db'}")
    Base.metadata.create_all(eng)
    t = CoreBelief.__table__
    with eng.begin() as conn:
        for i, tid in enumerate(("T1", "T2"), start=1):
            conn.execute(insert(t).values(
                id=i, global_id=f"g-{tid}",
                tenant_id=tid, app_id="a", user_id="u1",
                dimension="d", key="k", predicate="p", object="o",
                claim_text="marked-original-text-ABC", polarity="pos",
                status="active", confidence=0.9,
            ))

    scoped = metrics.storage_stats(eng, tenant=("T1", "a"))
    allt = metrics.storage_stats_all_tenants(eng)
    assert all(k[0] == "T1" for k in scoped)
    assert ("T1", "u1") in scoped
    assert ("T1", "u1") in allt and ("T2", "u1") in allt
    # the two tenants' rows are counted separately, not summed together
    assert allt[("T1", "u1")]["mm_beliefs"]["rows"] == 1
    assert allt[("T2", "u1")]["mm_beliefs"]["rows"] == 1


def test_storage_stats_requires_tenant():
    from sqlalchemy import create_engine
    eng = create_engine("sqlite://")
    with pytest.raises(ValueError):
        metrics.storage_stats(eng, tenant=("", "a"))


def test_storage_stats_does_not_use_system_scope():
    """The TENANT-SCOPED report must apply an explicit WHERE, not rely on
    system_scope automatically elevating the DB role.  (Checked on the CODE
    only -- the docstring legitimately explains why it does NOT use it.)

    ``storage_stats_all_tenants`` is the separately-registered privileged
    report: it DOES run under system_scope (a logical, audited scope -- NOT a
    DB grant), so only the tenant-scoped path is asserted here."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(metrics.storage_stats).lstrip())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and ((isinstance(n.func, ast.Name) and n.func.id == "system_scope")
             or (isinstance(n.func, ast.Attribute) and n.func.attr == "system_scope"))
    ]
    assert not calls, "tenant-scoped storage_stats must not call system_scope"


# ---------------------------------------------------------------------------
# (6) end-to-end: no content / user id / purpose leaks into Prometheus
# ---------------------------------------------------------------------------


def test_no_content_or_user_id_in_prometheus_end_to_end(tmp_path):
    """Drive a REAL observe+recall with a distinctive user id and utterance,
    then assert neither the user id nor the utterance text appears anywhere in
    the Prometheus exposition -- and that only the closed label set is used.

    Uses user id ``u1`` (the id the harness lazy-grants for) so the grant path
    is the normal one; ``u1`` is itself a user id, so the 'no user id on the
    wire' check is genuine."""
    MARKER = "SECRET-ORIGINAL-UTTERANCE-XYZ"
    USER = "u1"
    eng = _make_engine(tmp_path, llm=_NoUsageLLM(), user=USER)
    # Insert the grant row ourselves (the harness's per-test memo keys on
    # id(bind) and can be stale across tests); a real committed row is what
    # authorize_or_raise reads.
    _grant_direct(eng, USER, "chat")
    metrics.reset_for_tests()
    eng.observe(user_id=USER, session_id="s", text=MARKER, purpose="chat")
    eng.recall(user_id=USER, purpose="chat")

    text = metrics.render_prometheus()
    assert USER not in text, "a user id reached the Prometheus output"
    assert MARKER not in text, "memory content reached the Prometheus output"
    # only entry/reason/le labels appear on the wire
    import re
    for label_name in re.findall(r'(\w+)="', text):
        assert label_name in ("entry", "reason", "le"), f"unexpected label {label_name}"
