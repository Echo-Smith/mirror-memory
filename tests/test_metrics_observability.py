"""Metrics observability tests (M5 task 2).

Locks the privacy/label invariants and the accounting/latency/admin-auth
behaviour the observability task promises:

(a) a user id string never appears in Prometheus output;
(b) memory text / content never appears in any metric;
(c) only ``entry`` (and fixed reason tokens) is used as a label;
(d) counters increment correctly;
(e) ``/metrics`` requires admin auth.

Also: snapshot() stays backward-compatible (dict[str,int]), storage_stats is
per-user/per-table and on-demand, and the histogram quantiles are sane.
"""

from __future__ import annotations

import os

import pytest

from mirror_memory.core import metrics


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics.reset_for_tests()
    yield
    metrics.reset_for_tests()


# ---------------------------------------------------------------------------
# (d) counters / cost accounting
# ---------------------------------------------------------------------------


def test_counters_increment_correctly():
    assert metrics.value(metrics.MISSING_PURPOSE) == 0
    metrics.inc(metrics.MISSING_PURPOSE)
    metrics.inc(metrics.MISSING_PURPOSE, 2)
    assert metrics.value(metrics.MISSING_PURPOSE) == 3


def test_record_llm_attributes_tokens_and_unknown():
    metrics.record_llm("observe", input_tokens=120, output_tokens=40)
    metrics.record_llm("observe", input_tokens=None, output_tokens=None)
    snap = metrics.snapshot()
    assert snap["llm_calls_total.observe"] == 2
    assert snap["llm_input_tokens.observe"] == 120
    assert snap["llm_output_tokens.observe"] == 40
    assert snap["llm_usage_unknown_total.observe"] == 1


def test_record_llm_rejects_non_int_tokens_as_unknown():
    metrics.record_llm("recall", input_tokens="unknown", output_tokens=5)
    snap = metrics.snapshot()
    assert "llm_input_tokens.recall" not in snap
    assert snap["llm_output_tokens.recall"] == 5
    assert snap["llm_usage_unknown_total.recall"] == 1


def test_record_sql_counts_by_entry():
    metrics.record_sql("recall")
    metrics.record_sql("recall", 3)
    assert metrics.snapshot()["sql_statements_total.recall"] == 4


def test_snapshot_backward_compatible_dict_of_int():
    metrics.inc("whatever_counter", 5)
    snap = metrics.snapshot()
    assert isinstance(snap, dict)
    assert snap.get("whatever_counter", 0) == 5
    assert snap.get("missing", 0) == 0
    assert all(isinstance(v, int) for v in snap.values())


def test_gauges_set_and_read():
    metrics.set_gauge(metrics.REPLAY_ACTIVE, 1)
    assert metrics.gauge(metrics.REPLAY_ACTIVE) == 1.0


# ---------------------------------------------------------------------------
# (2) latency histograms
# ---------------------------------------------------------------------------


def test_latency_quantiles_are_ordered():
    for ms in (3, 8, 15, 30, 60, 120, 240, 480):
        metrics.observe_latency("recall", ms)
    s = metrics.latency_summaries()["recall"]
    assert s["count"] == 8
    assert 0 < s["p50"] <= s["p95"] <= s["p99"]


def test_latency_bounded_memory():
    metrics.observe_latency("observe", 5.0)
    before = len(metrics.histogram("observe")["buckets"])
    for _ in range(1000):
        metrics.observe_latency("observe", 5.0)
    after = len(metrics.histogram("observe")["buckets"])
    assert before == after
    assert metrics.histogram("observe")["count"] == 1001


def test_latency_thread_safe_all_samples_counted():
    import threading

    def worker():
        for _ in range(500):
            metrics.observe_latency("recall", 10.0)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert metrics.histogram("recall")["count"] == 4000


# ---------------------------------------------------------------------------
# (a)/(b)/(c) Prometheus label & content privacy
# ---------------------------------------------------------------------------


def _sample_prometheus_with_user_data(user_id: str, secret_text: str) -> str:
    metrics.inc(metrics.MISSING_PURPOSE)
    metrics.set_gauge(metrics.REPLAY_ACTIVE, 1)
    metrics.record_llm("observe", input_tokens=42, output_tokens=7)
    metrics.record_llm("observe", input_tokens=None, output_tokens=None)
    metrics.record_sql("observe")
    metrics.observe_latency("observe", 12.0)
    metrics.observe_latency("recall", 88.0)
    metrics.observe_latency("extraction_worker", 300.0)
    return metrics.render_prometheus()


def test_user_id_never_in_prometheus():
    user_id = "USER-SECRET-MARKER-9f3a2b"
    text = "my deeply personal memory text about a break-up"
    out = _sample_prometheus_with_user_data(user_id, text)
    assert user_id not in out
    assert text not in out


def test_memory_content_never_in_any_metric():
    text = "TENDRSWEET-COMMITMENT-content-xyz"
    _sample_prometheus_with_user_data("u1", text)
    import json

    assert text not in metrics.render_prometheus()
    assert text not in json.dumps(metrics.report())
    assert text not in json.dumps(metrics.snapshot())


def test_only_entry_and_reason_labels_used():
    import re

    out = _sample_prometheus_with_user_data("u1", "t")
    label_keys = set(re.findall(r'(\w+)="', out))
    assert label_keys <= {"entry", "reason", "le"}, label_keys


def test_prometheus_is_valid_exposition_shape():
    out = _sample_prometheus_with_user_data("u1", "t")
    types = {}
    for line in out.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            assert kind in ("counter", "gauge", "histogram")
            types[name] = kind
    assert types.get("mirror_memory_missing_purpose_refusals_total") == "counter"
    assert types.get("mirror_memory_latency_ms") == "histogram"
    assert 'mirror_memory_latency_ms_bucket{entry="recall"' in out


# ---------------------------------------------------------------------------
# (3) storage_stats: on-demand, per-user, per-table (library only)
# ---------------------------------------------------------------------------


def test_storage_stats_per_user_per_table(tmp_path):
    """storage_stats is now TENANT-SCOPED (explicit WHERE), keyed by
    (tenant_id, user_id); the privileged all-tenant report is separate."""
    from mirror_memory.api import MemoryEngine
    from mirror_memory.config.loader import load_config
    from mirror_memory.core.models import Evidence
    from mirror_memory.core.tenancy import fixture_write_scope

    cfg = load_config("config/")
    cfg.deletion_log.directory = str(tmp_path / "del")
    db = f"sqlite:///{tmp_path}/m.db"
    eng = MemoryEngine(config=cfg, database_url=db, default_purpose="benchmark")
    eng._ensure_db()
    from mirror_memory.bench import grant_bench_purpose

    grant_bench_purpose(eng, ["u-storage"])

    with fixture_write_scope(reason="metrics storage_stats test: seed one evidence row"):
        with eng._session() as s:
            s.add(Evidence(user_id="u-storage", content="x" * 100,
                           source_type="message", session_id="s1"))
            s.commit()

    tenant = eng._tenant
    stats = metrics.storage_stats(eng._engine, tenant=tenant)
    assert (tenant[0], "u-storage") in stats
    cell = stats[(tenant[0], "u-storage")]
    assert "mm_evidence" in cell
    assert cell["mm_evidence"]["rows"] >= 1
    assert cell["mm_evidence"]["bytes"] >= 100
    # a DIFFERENT tenant sees nothing for this call (explicit filter)
    other = metrics.storage_stats(eng._engine, tenant=("other-tenant", "x"))
    assert (("other-tenant"), "u-storage") not in other
    assert "u-storage" not in metrics.render_prometheus()


# ---------------------------------------------------------------------------
# (e) /metrics admin auth
# ---------------------------------------------------------------------------


def _make_app(tmp_path):
    from mirror_memory.server import create_app

    return create_app(
        config_path="config/",
        database_url=f"sqlite:///{tmp_path}/m.db",
    )


def test_metrics_requires_admin_auth(tmp_path):
    from fastapi.testclient import TestClient

    os.environ["MM_ADMIN_API_KEY"] = "admin-secret"
    try:
        with TestClient(_make_app(tmp_path)) as c:
            assert c.get("/metrics").status_code == 401
            assert c.get(
                "/metrics", headers={"Authorization": "Bearer nope"}
            ).status_code == 401
            r = c.get("/metrics", headers={"Authorization": "Bearer admin-secret"})
            assert r.status_code == 200
            assert "# TYPE" in r.text
            assert "mirror_memory_latency_ms" in r.text
    finally:
        del os.environ["MM_ADMIN_API_KEY"]


def test_metrics_disabled_without_admin_key(tmp_path):
    from fastapi.testclient import TestClient

    os.environ.pop("MM_ADMIN_API_KEY", None)
    with TestClient(_make_app(tmp_path)) as c:
        assert c.get("/metrics").status_code == 404


def test_metrics_never_leaks_user_ids_through_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    os.environ["MM_ADMIN_API_KEY"] = "admin-secret"
    try:
        with TestClient(_make_app(tmp_path)) as c:
            c.post(
                "/observe",
                json={"user_id": "u-EP-9c1", "session_id": "s1",
                      "text": "CONFIDENTIAL-BREAKUP-DETAIL-7712"},
            )
            r = c.get("/metrics", headers={"Authorization": "Bearer admin-secret"})
            assert r.status_code == 200
            assert "u-EP-9c1" not in r.text
            assert "CONFIDENTIAL-BREAKUP-DETAIL-7712" not in r.text
    finally:
        del os.environ["MM_ADMIN_API_KEY"]
