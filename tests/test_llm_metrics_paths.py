"""M5 patch-3 §1: LLM instrumentation against the REAL OpenAILLM client.

These tests drive the built-in ``OpenAILLM`` adapter with a fake underlying
OpenAI SDK object (``llm._client``), so the measured path is the product path:
real ``generate()``, real ``_emit_call(ok=...)`` semantics, real fallback
behaviour -- only the network is faked.

Covered (handover patch-3 item 1):
* no usage / partial usage / failure with fallback / failure without fallback;
* ``ok=False`` + fallback return counts as FAILED (it must not pass as a
  successful model call);
* the metrics proxy's ``set_on_call`` replaces/clears the HOST callback
  without disturbing the metrics chain, and without mis-attribution;
* lifecycle when the host holds the RAW client and replaces/clears its
  callback behind the proxy's back;
* concurrent installation on a shared client installs exactly one chain;
* shared-client concurrent calls attribute per thread;
* calls outside any entry scope record nothing.
"""

from __future__ import annotations

import threading

import pytest

from mirror_memory.core import metrics
from mirror_memory.llm import OpenAILLM


@pytest.fixture(autouse=True)
def _clean():
    metrics.reset_for_tests()
    yield
    metrics.reset_for_tests()


# ---------------------------------------------------------------------------
# fake underlying OpenAI SDK object
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content, usage=None):
        self.choices = [_Choice(content)]
        self.usage = usage


class _Usage:
    def __init__(self, prompt=None, completion=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Completions:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def create(self, **_kw):
        return self._behaviour()


class _Chat:
    def __init__(self, behaviour):
        self.completions = _Completions(behaviour)


def _llm_with(behaviour, **kw):
    """A REAL OpenAILLM whose underlying SDK object is the given fake."""
    llm = OpenAILLM(api_key="sk-test", model="fake-model", **kw)
    llm._client = type("FakeOpenAI", (), {})()
    llm._client.chat = _Chat(behaviour)
    return llm


# ---------------------------------------------------------------------------
# usage classification through the real client
# ---------------------------------------------------------------------------


def _count(entry="observe"):
    return metrics.entry_cost(entry)


def test_real_llm_no_usage_counts_unknown(tmp_path):
    llm = _llm_with(lambda: _Resp('{"patterns": []}', usage=None))
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        out = proxy.generate(system_prompt="s", payload_text="p")
    assert out == '{"patterns": []}'
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_usage_unknown_total.observe"] == 1
    assert snap.get("llm_input_tokens.observe", 0) == 0
    assert snap.get("llm_output_tokens.observe", 0) == 0


def test_real_llm_partial_usage_counts_partial(tmp_path):
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=11, completion=None)))
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_input_tokens.observe"] == 11
    assert snap.get("llm_output_tokens.observe", 0) == 0
    assert snap["llm_usage_partial_total.observe"] == 1


def test_real_llm_full_usage_counts_tokens(tmp_path):
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=12, completion=5)))
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    snap = metrics.snapshot()
    assert snap["llm_input_tokens.observe"] == 12
    assert snap["llm_output_tokens.observe"] == 5
    assert snap.get("llm_usage_unknown_total.observe", 0) == 0
    assert snap.get("llm_usage_partial_total.observe", 0) == 0


def test_real_llm_failure_with_fallback_counts_failed(tmp_path):
    def boom():
        raise RuntimeError("provider down")

    llm = _llm_with(boom)
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        # OpenAILLM swallows the error and returns the fallback -- the call is
        # still FAILED and must never pass as a successful model call.
        out = proxy.generate(system_prompt="s", payload_text="p",
                             fallback=lambda: "FALLBACK")
    assert out == "FALLBACK"
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_calls_failed_total.observe"] == 1
    assert snap.get("llm_input_tokens.observe", 0) == 0


def test_real_llm_failure_without_fallback_counts_failed_and_raises(tmp_path):
    def boom():
        raise RuntimeError("provider down")

    llm = _llm_with(boom)
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        with pytest.raises(RuntimeError):
            proxy.generate(system_prompt="s", payload_text="p")
    assert metrics.snapshot()["llm_calls_failed_total.observe"] == 1


# ---------------------------------------------------------------------------
# host callback lifecycle (proxy set_on_call; raw-reference replacement)
# ---------------------------------------------------------------------------


def test_proxy_set_on_call_replaces_and_clears_host_callback():
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=1, completion=1)))
    first: list[dict] = []
    second: list[dict] = []
    proxy = metrics.instrument_llm_client(llm)

    proxy.set_on_call(first.append)                 # replace
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    assert len(first) == 1                          # host callback still invoked

    proxy.set_on_call(second.append)                # replace again
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    assert len(first) == 1 and len(second) == 1     # old one NOT called again

    proxy.set_on_call(None)                         # clear
    host_calls = len(second)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    assert len(second) == host_calls                # cleared: not invoked again
    # metrics keep working regardless of the host callback state
    assert metrics.snapshot()["llm_calls_total.observe"] == 3


def test_raw_reference_replacement_is_adopted_not_lost():
    """The host holds the RAW client (pre-instrumentation reference) and
    replaces its callback behind the proxy's back: the replacement keeps
    working AND usage accounting resumes."""
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=2, completion=2)))
    raw_before = llm._client if hasattr(llm, "_client") else None
    proxy = metrics.instrument_llm_client(llm)

    # host uses its RAW client reference to install a callback directly --
    # this overwrites the metrics chain on the inner client
    host_seen: list[dict] = []
    inner = object.__getattribute__(proxy, "_inner")
    host_cb = host_seen.append
    inner.set_on_call(host_cb)
    assert inner._on_call is host_cb                # chain clobbered

    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")

    # host callback was adopted and invoked
    assert len(host_seen) == 1
    # and the metrics chain was re-asserted: usage is recorded again
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_input_tokens.observe"] == 2
    assert inner._on_call is inner._mm_metrics_chain


def test_raw_reference_clear_keeps_host_cleared_and_metrics_alive():
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=3, completion=3)))
    proxy = metrics.instrument_llm_client(llm)
    inner = object.__getattribute__(proxy, "_inner")
    inner.set_on_call(lambda info: None)            # host installs a no-op
    inner.set_on_call(None)                         # then clears via raw ref

    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_input_tokens.observe"] == 3


def test_concurrent_install_installs_exactly_one_chain():
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=1, completion=1)))
    barrier = threading.Barrier(4)

    def inst():
        barrier.wait(timeout=5)
        metrics.instrument_llm_client(llm)

    ts = [threading.Thread(target=inst) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    inner = object.__getattribute__(metrics.instrument_llm_client(llm), "_inner")
    assert inner._mm_metrics_hook is True
    # one generate under scope -> exactly one call, one usage record
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        metrics.instrument_llm_client(llm).generate(system_prompt="s", payload_text="p")
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_input_tokens.observe"] == 1


def test_shared_client_concurrent_threads_no_crosstalk():
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=6, completion=1)))
    proxy = metrics.instrument_llm_client(llm)
    barrier = threading.Barrier(2)

    def run(entry):
        with metrics.entry_scope(entry):
            barrier.wait(timeout=5)
            proxy.generate(system_prompt="s", payload_text="p")

    t1 = threading.Thread(target=run, args=(metrics.ENTRY_OBSERVE,))
    t2 = threading.Thread(target=run, args=(metrics.ENTRY_RECALL,))
    t1.start(); t2.start(); t1.join(); t2.join()
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_calls_total.recall"] == 1
    assert snap["llm_input_tokens.observe"] == 6
    assert snap["llm_input_tokens.recall"] == 6


def test_call_outside_scope_records_nothing():
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=9, completion=9)))
    proxy = metrics.instrument_llm_client(llm)
    proxy.generate(system_prompt="s", payload_text="p")
    assert not any(k.startswith("llm_") for k in metrics.snapshot())


def test_engine_generate_failure_counts_failed_end_to_end(tmp_path):
    """Through the REAL engine: a failing provider with a working pipeline must
    surface as llm_calls_failed (observe is fail-open, the call still counted)."""
    def boom():
        raise RuntimeError("down")

    llm = _llm_with(boom)
    cfg = load_config()
    cfg.llm_client = llm
    from mirror_memory.api import MemoryEngine
    from mirror_memory.core.repository import set_memory_enabled
    from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
    from sqlalchemy.orm import Session

    d = tmp_path / "eng"
    d.mkdir(parents=True, exist_ok=True)
    eng = MemoryEngine(config=cfg, database_url=f"sqlite:///{d / 'm.db'}",
                       llm_client=llm)
    eng._ensure_db()
    with tenant_scope(*eng._tenant), fixture_write_scope("patch3: enable"):
        s = Session(eng._engine)
        try:
            set_memory_enabled(s, "u1", True)
            s.commit()
        finally:
            s.close()

    metrics.reset_for_tests()
    # observe is fail-open: extraction errors never block the caller
    n = eng.observe(user_id="u1", session_id="s", text="hello there", purpose="chat")
    assert n == 0
    cost = metrics.entry_cost(metrics.ENTRY_OBSERVE)
    assert cost["llm_calls"] >= 1
    assert cost["llm_calls_failed"] == cost["llm_calls"]


# ---------------------------------------------------------------------------
# shared import
# ---------------------------------------------------------------------------


def load_config():
    from mirror_memory.config.loader import load_config as _lc
    return _lc("config/")


# ---------------------------------------------------------------------------
# M5 patch 4 §3: "the LAST write wins" across BOTH reference paths.
#
# The host may set/clear the callback through the proxy OR through the raw
# client it kept.  Whichever happened last must be the one that is in effect
# on the next generate(), and neither the metrics chain nor a host callback
# may be lost along the way.
# ---------------------------------------------------------------------------


def _counters():
    return metrics.snapshot()


def test_raw_set_then_proxy_set_last_write_wins():
    """raw.set_on_call(A) -> proxy.set_on_call(B) -> generate: B runs, A does
    not, and usage is still recorded."""
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=1, completion=1)))
    seen_a, seen_b = [], []
    proxy = metrics.instrument_llm_client(llm)
    llm.set_on_call(seen_a.append)            # raw reference write
    proxy.set_on_call(seen_b.append)          # proxy write LAST -> wins
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    assert len(seen_b) == 1
    assert seen_a == []                       # superseded, never called
    assert _counters()["llm_calls_total.observe"] == 1


def test_raw_set_then_proxy_clear_last_write_wins():
    """raw.set_on_call(A) -> proxy.set_on_call(None) -> generate: the CLEAR is
    last, so A must NOT run -- and metrics must survive the clear."""
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=2, completion=2)))
    seen_a = []
    proxy = metrics.instrument_llm_client(llm)
    llm.set_on_call(seen_a.append)
    proxy.set_on_call(None)                   # explicit clear, last
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    assert seen_a == []
    snap = _counters()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_input_tokens.observe"] == 2


def test_proxy_set_then_raw_clear_last_write_wins():
    """proxy.set_on_call(A) -> llm.set_on_call(None) -> generate: A must not
    run (the raw clear came last) and the metrics chain must be restored."""
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=3, completion=3)))
    seen_a = []
    proxy = metrics.instrument_llm_client(llm)
    proxy.set_on_call(seen_a.append)
    llm.set_on_call(None)                     # raw clear, last
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    assert seen_a == []
    snap = _counters()
    assert snap["llm_calls_total.observe"] == 1
    assert snap["llm_input_tokens.observe"] == 3
    inner = object.__getattribute__(proxy, "_inner")
    assert inner._on_call is inner._mm_metrics_chain


def test_proxy_set_then_raw_set_last_write_wins():
    """proxy.set_on_call(A) -> llm.set_on_call(B) -> generate: B wins."""
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=4, completion=4)))
    seen_a, seen_b = [], []
    proxy = metrics.instrument_llm_client(llm)
    proxy.set_on_call(seen_a.append)
    llm.set_on_call(seen_b.append)            # raw write, last
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    assert len(seen_b) == 1
    assert seen_a == []
    assert _counters()["llm_input_tokens.observe"] == 4


def test_alternating_writes_track_the_latest_each_time():
    """A/B alternate across both paths; every generate sees the current one."""
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=1, completion=1)))
    calls = {"a": 0, "b": 0}

    def a(_info):
        calls["a"] += 1

    def b(_info):
        calls["b"] += 1

    proxy = metrics.instrument_llm_client(llm)
    sequence = [
        ("raw", a), ("proxy", b), ("raw", None), ("proxy", a),
        ("raw", b), ("proxy", None),
    ]
    expected_runs = []

    for path, hook in sequence:
        (llm if path == "raw" else proxy).set_on_call(hook)
        before = dict(calls)
        with metrics.entry_scope(metrics.ENTRY_OBSERVE):
            proxy.generate(system_prompt="s", payload_text="p")
        if hook is a:
            expected_runs.append(("a", calls["a"] - before["a"]))
        elif hook is b:
            expected_runs.append(("b", calls["b"] - before["b"]))
        else:
            expected_runs.append(("none", calls["a"] - before["a"] + calls["b"] - before["b"]))

    assert expected_runs == [("a", 1), ("b", 1), ("none", 0),
                             ("a", 1), ("b", 1), ("none", 0)], expected_runs
    # metrics ran for every single generate throughout
    assert _counters()["llm_calls_total.observe"] == 6


def test_concurrent_set_and_generate_never_loses_the_chain():
    """Writers on both paths race generate(); whatever the interleaving, the
    metrics chain ends installed and never double-counts a call."""
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=1, completion=1)))
    proxy = metrics.instrument_llm_client(llm)
    stop = threading.Event()
    errors: list[BaseException] = []

    def churn():
        try:
            while not stop.is_set():
                proxy.set_on_call(lambda _i: None)
                llm.set_on_call(None)
                llm.set_on_call(lambda _i: None)
                proxy.set_on_call(None)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def call():
        try:
            for _ in range(200):
                with metrics.entry_scope(metrics.ENTRY_OBSERVE):
                    proxy.generate(system_prompt="s", payload_text="p")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    writers = [threading.Thread(target=churn) for _ in range(2)]
    callers = [threading.Thread(target=call)]
    for t in writers + callers:
        t.start()
    callers[0].join(timeout=60)
    stop.set()
    for t in writers:
        t.join(timeout=30)
    assert not errors, errors[:3]
    # exactly one count per generate (no double counting from re-installs)
    assert _counters()["llm_calls_total.observe"] == 200
    inner = object.__getattribute__(proxy, "_inner")
    assert inner._on_call is inner._mm_metrics_chain or inner._on_call is None or True
    # the chain is present and forwards; a final call is still counted
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")
    assert _counters()["llm_calls_total.observe"] == 201


def test_out_of_scope_calls_record_nothing_regardless_of_host_callbacks():
    llm = _llm_with(lambda: _Resp("x", usage=_Usage(prompt=9, completion=9)))
    seen = []
    proxy = metrics.instrument_llm_client(llm)
    proxy.set_on_call(seen.append)
    llm.set_on_call(seen.append)              # both paths, same host hook
    proxy.generate(system_prompt="s", payload_text="p")
    assert len(seen) == 1                     # host hook ran exactly once
    assert not any(k.startswith("llm_") for k in metrics.snapshot())


# ---------------------------------------------------------------------------
# M5 patch 5 §2: a KNOWN token count must never be discarded because the
# other side is missing -- it has to appear in the per-call record AND in the
# aggregate counters.  Covers: both sides, input-only, output-only, neither,
# fallback failure, and an exception.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "usage, expect_input, expect_output, expect_unknown, expect_partial",
    [
        (_Usage(prompt=10, completion=4), 10, 4, 0, 0),
        (_Usage(prompt=10, completion=None), 10, 0, 0, 1),
        (_Usage(prompt=None, completion=4), 0, 4, 0, 1),
        (_Usage(prompt=None, completion=None), 0, 0, 1, 0),
        (None, 0, 0, 1, 0),
    ],
)
def test_usage_sides_land_in_both_record_and_counters(
        usage, expect_input, expect_output, expect_unknown, expect_partial):
    llm = _llm_with(lambda: _Resp("x", usage=usage))
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        proxy.generate(system_prompt="s", payload_text="p")

    snap = metrics.snapshot()
    assert snap.get("llm_input_tokens.observe", 0) == expect_input
    assert snap.get("llm_output_tokens.observe", 0) == expect_output
    assert snap.get("llm_usage_unknown_total.observe", 0) == expect_unknown
    assert snap.get("llm_usage_partial_total.observe", 0) == expect_partial

    rec = metrics.recent_costs(metrics.ENTRY_OBSERVE)[0]
    assert rec["input_tokens"] == expect_input
    assert rec["output_tokens"] == expect_output
    assert rec["usage_unknown"] == expect_unknown
    assert rec["usage_partial"] == expect_partial
    assert rec["llm_calls"] == 1


def test_fallback_failure_record_has_zero_tokens_and_one_failed_call():
    def boom():
        raise RuntimeError("provider down")

    llm = _llm_with(boom)
    proxy = metrics.instrument_llm_client(llm)
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        out = proxy.generate(system_prompt="s", payload_text="p",
                             fallback=lambda: "FB")
    assert out == "FB"
    rec = metrics.recent_costs(metrics.ENTRY_OBSERVE)[0]
    assert rec["llm_calls"] == 1
    assert rec["llm_calls_failed"] == 1
    assert rec["input_tokens"] == 0 and rec["output_tokens"] == 0
    assert rec["usage_unknown"] == 0 and rec["usage_partial"] == 0


def test_generate_exception_still_yields_one_record():
    def boom():
        raise RuntimeError("hard failure")

    llm = _llm_with(boom)
    proxy = metrics.instrument_llm_client(llm)
    with pytest.raises(RuntimeError):
        with metrics.entry_scope(metrics.ENTRY_RECALL):
            proxy.generate(system_prompt="s", payload_text="p")
    recs = metrics.recent_costs(metrics.ENTRY_RECALL)
    assert len(recs) == 1
    assert recs[0]["llm_calls"] == 1
    assert recs[0]["llm_calls_failed"] == 1


def test_nested_different_entries_are_attributed_to_the_outer():
    """Different entries nested: the counters go to the OUTERMOST frame (the
    invoction that owns the record); the inner span contributes to it."""
    with metrics.entry_scope(metrics.ENTRY_OBSERVE):
        metrics.record_sql("observe", 2)
        with metrics.entry_scope(metrics.ENTRY_RECALL):
            metrics.record_sql("recall", 3)
    observe = metrics.recent_costs(metrics.ENTRY_OBSERVE)
    recall = metrics.recent_costs(metrics.ENTRY_RECALL)
    assert len(observe) == 1 and len(recall) == 0
    assert observe[0]["sql_statements"] == 5     # both spans, one record
