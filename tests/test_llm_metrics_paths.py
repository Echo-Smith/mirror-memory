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
