"""M3 batch 3 task 3: the missing_purpose counter (no SQL) and the audit
retention maintenance pass (per tenant, in batches, on the worker's
cadence)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mirror_memory.api import MemoryEngine
from mirror_memory.config.loader import load_config
from mirror_memory.core import access_audit, metrics
from mirror_memory.core.authorization import PurposeDenied
from mirror_memory.core.models import AccessAudit, Base, Belief, EvolutionJob
from mirror_memory.core.repository import set_memory_enabled
from mirror_memory.core.tenancy import fixture_write_scope, tenant_scope
from mirror_memory.worker import maintenance

USER = "u-maint"
TENANT = ("default", "default")


@pytest.fixture
def db(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'maint.db'}")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# the missing_purpose counter
# ---------------------------------------------------------------------------


def test_missing_purpose_refusal_counts_without_sql(db):
    cfg = load_config("config/")
    with tenant_scope(*TENANT), fixture_write_scope("enable"):
        s = sessionmaker(bind=db)()
        set_memory_enabled(s, USER, True)
        s.commit()
        s.close()
    eng = MemoryEngine(config=cfg, database_url=str(db.url),
                       tenant_id=TENANT[0], app_id=TENANT[1], default_purpose=None)
    before = metrics.value(metrics.MISSING_PURPOSE)
    statements: list[str] = []
    from sqlalchemy import event

    eng._ensure_db()
    event.listen(eng._engine, "before_cursor_execute",
                 lambda c, cur, st, p, ctx, m: statements.append(st))
    with pytest.raises(PurposeDenied, match="missing_purpose"):
        eng.recall(user_id=USER, query="x")
    assert statements == [], "the refusal must stay a Python-level zero-SQL decision"
    assert metrics.value(metrics.MISSING_PURPOSE) == before + 1


def test_worker_missing_purpose_cancellation_counts(tmp_path):
    from mirror_memory.worker.evolution import _process_single_job

    cfg = load_config("config/")
    db = create_engine(f"sqlite:///{tmp_path / 'w.db'}")
    Base.metadata.create_all(db)

    from mirror_memory.core import schema_migration as _sm_mint
    _sm_mint.ensure_watermark(db)
    F = sessionmaker(bind=db)
    with tenant_scope(*TENANT), fixture_write_scope("seed job"):
        s = F()
        set_memory_enabled(s, USER, True)
        s.commit()
        # a pre-M3 style row: the empty purpose cannot get past enqueue's
        # own validation, so write the row the way the migration left it
        s.add(EvolutionJob(id="legacy-job", user_id=USER, evidence_watermark="{}",
                           idempotency_key="ik-legacy", status="pending", purpose=""))
        s.commit()
        job_id = "legacy-job"
        s.close()
    before = metrics.value(metrics.MISSING_PURPOSE)
    with tenant_scope(*TENANT):
        s = F()
        try:
            _process_single_job(s, s.get(EvolutionJob, job_id), cfg)
            s.expire_all()
            assert s.get(EvolutionJob, job_id).status == "cancelled"
        finally:
            s.close()
    assert metrics.value(metrics.MISSING_PURPOSE) == before + 1
    db.dispose()


# ---------------------------------------------------------------------------
# the retention maintenance pass
# ---------------------------------------------------------------------------


def _seed_audit(db, tenant, n: int) -> list[int]:
    with tenant_scope(*tenant), fixture_write_scope("seed audit rows"):
        s = sessionmaker(bind=db)()
        ids = []
        for _ in range(n):
            row = AccessAudit(user_id="u", purpose="p", operation="recall",
                              entry="e", decision="allowed")
            s.add(row)
            s.flush()
            ids.append(row.id)
        s.commit()
        s.close()
        return ids


def _backdate(db, ids, days: int) -> None:
    from mirror_memory.core.sql_audit import no_audit_scope

    if not ids:
        return
    placeholders = ", ".join(f":id{i}" for i in range(len(ids)))
    params = {f"id{n}": v for n, v in enumerate(ids)}
    params["t"] = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
    with no_audit_scope(), db.begin() as conn:
        conn.execute(text(
            f"UPDATE mm_access_audit SET created_at = :t WHERE id IN ({placeholders})"
        ), params)


def test_maintenance_purges_expired_rows_per_tenant_in_batches(tmp_path):
    db = create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    Base.metadata.create_all(db)
    from mirror_memory.core import schema_migration as _sm_mint

    _sm_mint.ensure_watermark(db)
    t_a, t_b = ("tA", "app"), ("tB", "app")
    a_expired = _seed_audit(db, t_a, 5)
    a_live = _seed_audit(db, t_a, 2)
    b_expired = _seed_audit(db, t_b, 3)
    _backdate(db, a_expired, 100)
    _backdate(db, b_expired, 100)
    _backdate(db, a_live, 1)

    cfg = load_config("config/")
    cfg.access_audit.retention_days = 90
    cfg.access_audit.purge_batch_size = 2  # forces several batches per tenant

    def factory():
        return sessionmaker(bind=db)()

    removed = maintenance.purge_expired_audit(factory, cfg)
    assert removed == 8

    from mirror_memory.core.sql_audit import no_audit_scope

    with no_audit_scope(), db.connect() as conn:
        left = conn.execute(text("SELECT id FROM mm_access_audit ORDER BY id")).all()
    assert [r[0] for r in left] == a_live, "only expired rows, across both tenants"


def test_maintenance_with_no_rows_is_a_no_op(tmp_path):
    db = create_engine(f"sqlite:///{tmp_path / 'm2.db'}")
    Base.metadata.create_all(db)
    cfg = load_config("config/")
    assert maintenance.purge_expired_audit(sessionmaker(bind=db), cfg) == 0
    db.dispose()


def test_maintenance_cadence_is_configurable_and_first_call_is_due():
    cfg = load_config("config/")
    cfg.access_audit.purge_interval_seconds = 3600
    state: dict[str, float] = {}
    assert maintenance.next_due(state, cfg, now=100.0) is True
    assert state["next"] == 100.0 + 3600
    assert maintenance.next_due(state, cfg, now=200.0) is False
    assert maintenance.next_due(state, cfg, now=100.0 + 3600) is True


def test_purge_limit_validation(db):
    with tenant_scope(*TENANT):
        s = sessionmaker(bind=db)()
        with pytest.raises(ValueError):
            access_audit.purge_access_audit(s, older_than=datetime.now(UTC), limit=0)
        s.rollback()
        s.close()
