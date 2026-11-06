"""M5 §3 patch: metrics attribution over the REAL paths.

Regression coverage for the behaviours the handover asks for explicitly:

* the host's own LLM callback is still invoked (chained, never replaced);
* no usage / partial usage / failure / real usage are told apart per call;
* a shared client used from several threads attributes per thread;
* calls made outside an entry scope record nothing;
* SQL is attributed to the engine that issued it -- an unrelated engine's
  statement on the same thread during an open scope is NOT counted;
* the listener installs once (concurrent engine construction cannot double it);
* the entry span covers initialization and the sync gate;
* per-call cost is available and distinct from the aggregate counters;
* Prometheus labels are a closed set and are escaped;
* no content or user identifier reaches the exposition.
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
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session


@pytest.fixture(autouse=True)
def _clean():
    metrics.reset_for_tests()
    yield
    metrics.reset_for_tests()


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _PlainLLM:
    """No usage hook at all (the worst case for accounting)."""

    def __init__(self):
        self.calls = 0

    def generate(self, **_kw):
        self.calls += 1
        return '{"patterns": [], "support_policy": {}}'


class _HookLLM:
    """Reports usage through ``set_on_call`` and holds a HOST callback."""

    def __init__(self, *, tokens):
        self._on_call = None
        self._mm_usage_hook = False
        self.host_seen: list[dict] = []
        self._tokens = tokens
        self.calls = 0

    def set_on_call(self, hook):
        self._on_call = hook

    def generate(self, **_kw):
        self.calls += 1
        if self._on_call is not None:
            self._on_call(self._tokens)
        return '{"patterns": [], "support_policy": {}}'


# ---------------------------------------------------------------------------
# LLM accounting: host callback preserved, four call outcomes
# ---------------------------------------------------------------------------


def test_host_callback_is_preserved_and_still_called():
    llm = _HookLLM(tokens={"input_tokens": 10, "output_tokens": 4})
    host_seen: list[dict] = []
    llm.set_on_call(host_seen.append)          # the host's own callback

    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate()

    # instrumentation did not swallow the host callback ...
    assert host_seen and host_seen[0]["input_tokens"] == 10
    # ... and the usage was recorded for the entry
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_input_tokens.observe"] == 10
    assert snap["llm_output_tokens.observe"] == 4
    assert snap.get("llm_usage_unknown_total.observe", 0) == 0
    assert snap.get("llm_usage_partial_total.observe", 0) == 0


def test_no_usage_client_books_unknown():
    llm = _PlainLLM()
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate()
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_usage_unknown_total.observe"] == 1
    assert snap.get("llm_input_tokens.observe", 0) == 0


def test_partial_usage_is_distinguished():
    llm = _HookLLM(tokens={"input_tokens": 7, "output_tokens": None})
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_RECALL):
        proxy.generate()
    snap = metrics.snapshot()
    assert snap["llm_calls_total.recall"] == 1
    assert snap["llm_input_tokens.recall"] == 7
    assert snap["llm_usage_partial_total.recall"] == 1
    assert snap.get("llm_usage_unknown_total.recall", 0) == 0


def test_failed_call_is_counted_as_failed_not_pass():
    class _Boom:
        def generate(self, **_kw):
            raise RuntimeError("provider down")

    proxy = metrics.instrument_llm_client(_Boom())
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        with pytest.raises(RuntimeError):
            proxy.generate()
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_calls_failed_total.observe"] == 1


def test_calls_outside_any_entry_scope_record_nothing():
    llm = _HookLLM(tokens={"input_tokens": 5, "output_tokens": 5})
    proxy = metrics.instrument_llm_client(llm)
    proxy.generate()                            # no entry scope on this thread
    assert not any(k.startswith("llm_") for k in metrics.snapshot())


def test_shared_client_concurrent_threads_attribute_per_thread():
    llm = _HookLLM(tokens={"input_tokens": 3, "output_tokens": 1})
    proxy = metrics.instrument_llm_client(llm)
    barrier = threading.Barrier(2)

    def run(entry):
        with metrics.entry_scope(entry):
            barrier.wait(timeout=5)
            proxy.generate()

    t1 = threading.Thread(target=run, args=(metrics.ENTRY_OBSERVE,))
    t2 = threading.Thread(target=run, args=(metrics.ENTRY_RECALL,))
    t1.start(); t2.start(); t1.join(); t2.join()
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_calls_total.recall"] == 1
    assert snap["llm_input_tokens.observe"] == 3
    assert snap["llm_input_tokens.recall"] == 3


# ---------------------------------------------------------------------------
# SQL attribution: the issuing engine, not merely the thread
# ---------------------------------------------------------------------------


def test_unrelated_engine_sql_is_not_counted_inside_a_scope(tmp_path):
    _ensure_listener()
    shared = create_engine("sqlite://")
    other = create_engine("sqlite://")
    with metrics.entry_scope(metrics.ENTRY_RECALL, engine=shared):
        with shared.connect() as c:
            for _ in range(2):
                c.execute(text("SELECT 1"))
        with other.connect() as c:            # unrelated engine, same thread
            for _ in range(3):
                c.execute(text("SELECT 1"))
    assert metrics.snapshot()["sql_statements_total.recall"] == 2


def test_scope_without_engine_counts_nothing():
    """``engine=None`` never means "all engines": an unbound scope attributes
    NO SQL (cold start before ``MemoryEngine._engine`` exists), so unrelated
    engines on the same thread can never be miscounted.  ``bind_scope_engine``
    upgrades the scope once the engine exists."""
    _ensure_listener()
    eng = create_engine("sqlite://")
    with metrics.entry_scope(metrics.ENTRY_EXTRACTION_WORKER):
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        # still unbound: the unrelated-engine statement was NOT counted
        assert not any(k.startswith("sql_statements_total")
                       for k in metrics.snapshot())
        # binding the scope to THIS engine makes its own SQL count
        assert metrics.bind_scope_engine(eng) is True
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    assert metrics.snapshot()["sql_statements_total.extraction_worker"] == 1


def test_bind_scope_engine_only_upgrades_unbound():
    eng = create_engine("sqlite://")
    other = create_engine("sqlite://")
    with metrics.entry_scope(metrics.ENTRY_OBSERVE, engine=eng):
        # already bound -> a later bind must NOT re-bind to another engine
        assert metrics.bind_scope_engine(other) is False
        with other.connect() as c:
            c.execute(text("SELECT 1"))
    assert metrics.snapshot().get("sql_statements_total.observe", 0) == 0


def test_sql_outside_scope_is_not_counted():
    _ensure_listener()
    eng = create_engine("sqlite://")
    with eng.connect() as c:
        c.execute(text("SELECT 1"))
    assert not any(k.startswith("sql_statements_total") for k in metrics.snapshot())


def test_listener_installs_once_under_concurrent_engine_construction(tmp_path):
    """Two engines built concurrently must not attach two listeners (which
    would double every count)."""
    import mirror_memory.api as api_mod

    errors: list[BaseException] = []

    def make(i):
        try:
            cfg = load_config("config/")
            cfg.llm_client = _PlainLLM()
            MemoryEngine(config=cfg, database_url=f"sqlite:///{tmp_path}/{i}.db",
                         llm_client=_PlainLLM())
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=make, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    eng = create_engine("sqlite://")
    with metrics.entry_scope(metrics.ENTRY_OBSERVE, engine=eng):
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    # exactly one count for one statement == exactly one listener
    assert metrics.snapshot()["sql_statements_total.observe"] == 1


# ---------------------------------------------------------------------------
# Entry span and per-call cost
# ---------------------------------------------------------------------------


def _ensure_listener() -> None:
    """Install the process-wide listener exactly the way ``MemoryEngine`` does.

    Some tests here use bare SQLAlchemy engines (to prove engine-level
    attribution) without constructing a ``MemoryEngine``, which is what
    normally installs the listener at construction."""
    from sqlalchemy import event
    from sqlalchemy.engine import Engine as _SAEngine

    def _install() -> None:
        def _count(conn, cursor, statement, parameters, context, executemany):
            metrics.record_sql_current(getattr(conn, "engine", None))

        event.listen(_SAEngine, "after_cursor_execute", _count)

    metrics.ensure_sql_listener(_install)


def _engine_at(tmp: pathlib.Path) -> tuple[MemoryEngine, object]:
    tmp.mkdir(parents=True, exist_ok=True)
    llm = _PlainLLM()
    cfg = load_config("config/")
    cfg.llm_client = llm
    eng = MemoryEngine(config=cfg, database_url=f"sqlite:///{tmp / 'm.db'}",
                       llm_client=llm)
    eng._ensure_db()
    with tenant_scope(*eng._tenant), fixture_write_scope("metrics paths: enable"):
        s = Session(eng._engine)
        try:
            set_memory_enabled(s, "u1", True)
            s.commit()
        finally:
            s.close()
    return eng, llm


def test_entry_span_covers_init_and_gate(tmp_path):
    """The FIRST observe pays initialization inside the measured span, so its
    sample exists and the entry cost reflects the whole call."""
    eng, _llm = _engine_at(tmp_path / "e")
    metrics.reset_for_tests()
    eng.observe(user_id="u1", session_id="s1", text="I like tea.", purpose="chat")
    lat = metrics.latency_summaries()["observe"]
    assert lat["count"] == 1
    # initialization issues SQL that belongs to this entry
    assert metrics.snapshot()["sql_statements_total.observe"] > 0


def test_per_call_cost_differs_from_aggregate(tmp_path):
    eng, _llm = _engine_at(tmp_path / "e")
    metrics.reset_for_tests()
    eng.observe(user_id="u1", session_id="s1", text="I like tea.", purpose="chat")
    eng.observe(user_id="u1", session_id="s2", text="I like coffee.", purpose="chat")

    cost = metrics.entry_cost(metrics.ENTRY_OBSERVE)
    totals = metrics.totals()
    assert cost["calls"] == 2                       # two invocations
    assert cost["sql_statements"] == totals["sql_statements_total.observe"]
    assert cost["sql_statements_per_call"] == pytest.approx(
        cost["sql_statements"] / 2, rel=1e-6)
    assert cost["llm_calls_per_call"] == pytest.approx(
        cost["llm_calls"] / 2, rel=1e-6)
    # the aggregate is NOT the per-call figure
    assert cost["sql_statements"] != cost["sql_statements_per_call"]
    # cost for a never-run entry is None, never a fabricated zero
    assert metrics.entry_cost(metrics.ENTRY_METABOLISM)["calls"] == 0
    assert metrics.entry_cost(metrics.ENTRY_METABOLISM)["llm_calls_per_call"] is None


def test_unknown_entry_is_rejected_by_the_closed_set():
    with pytest.raises(ValueError):
        metrics.entry_scope("not-an-entry").__enter__()
    with pytest.raises(ValueError):
        metrics.entry_cost("not-an-entry")


# ---------------------------------------------------------------------------
# Prometheus: closed labels, escaping, no content / identifiers
# ---------------------------------------------------------------------------


def test_prometheus_rejects_arbitrary_entry_labels():
    metrics.observe_latency("evil-entry", 1.0)      # dropped: not a known entry
    metrics.record_llm("evil-entry", input_tokens=1, output_tokens=1)
    metrics.record_sql("evil-entry", 1)
    text_out = metrics.render_prometheus()
    assert "evil-entry" not in text_out


def test_prometheus_escapes_label_values():
    assert metrics.escape_label_value('a"b\\c\nd') == 'a\\"b\\\\c\\nd'


def test_no_content_or_user_id_reaches_prometheus(tmp_path):
    marker = "SECRET-UTTERANCE-ABC"
    eng, _llm = _engine_at(tmp_path / "e")
    metrics.reset_for_tests()
    eng.observe(user_id="u1", session_id="s", text=marker, purpose="chat")
    eng.recall(user_id="u1", purpose="chat")
    out = metrics.render_prometheus()
    assert marker not in out
    assert "u1" not in out
    import re
    for label in re.findall(r'(\w+)="', out):
        assert label in ("entry", "reason", "le")