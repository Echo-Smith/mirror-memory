"""EXPLAIN review of the key queries on a tenant-scale dataset (M2 batch-2 task 4).

Builds a fresh RLS-protected schema, loads 100 tenants x 10k beliefs with
events / evidence / links / jobs in realistic ratios (set-based
``INSERT ... SELECT generate_series`` -- seconds, not minutes), runs
``ANALYZE``, then ``EXPLAIN (ANALYZE, BUFFERS)`` for each key query in
three shapes:

- ``C``  application tenant predicates, superuser connection (RLS bypassed)
- ``A``  application tenant predicates + RLS (app role, per-transaction
         tenant setting) -- today's production shape
- ``B``  RLS alone (app role, no application predicates) -- the perf-final
         candidate

User ids repeat across tenants on purpose (``u000``..``u099`` in every
tenant): that is the realistic multi-tenant namespace and the worst case
for the single-column ``user_id`` indexes M1 kept.

Usage::

    python tools/explain_review.py --admin-url postgresql+psycopg://postgres:...@host/db \
        --app-url postgresql+psycopg://mm_app:...@host/db --out docs/artifacts/explain.md
    python tools/explain_review.py ... --tenants 5 --beliefs-per-tenant 500   # smoke

The optional ``--candidate-indexes`` flag creates the composite indexes
proposed in the report (CONCURRENTLY, outside any transaction) after the
first pass and re-explains every query, so the before/after plans sit side
by side.  Timings are indicative (plan shape is the decision input);
the harness-wide performance numbers come from CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

from sqlalchemy import Index, create_engine, func, literal, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateIndex

sys.path.insert(0, "src")

from mirror_memory.core.models import (  # noqa: E402
    Base,
    Belief,
    BeliefEvent,
    BeliefEvidenceLink,
    Evidence,
    EvolutionJob,
    MemoryPreference,
)
from mirror_memory.core.schema_migration import install_row_level_security  # noqa: E402
from mirror_memory.core.tenancy import tenant_scope  # noqa: E402

APP_ID = "app"


# ---------------------------------------------------------------------------
# Data generation (owner side, set-based)
# ---------------------------------------------------------------------------


def _series(n: int):
    return func.generate_series(1, n).table_valued("n")


def _fill(table, select_stmt, provided: set[str]):
    """INSERT ... SELECT with every NOT NULL column satisfied.

    Columns the caller did not provide get the model's scalar default, a
    neutral literal of the right type, or ``now()`` -- Python-side defaults
    do not apply to INSERT ... SELECT, so this is what ``create_all`` + the
    ORM would have written.
    """
    cols = []
    exprs = list(select_stmt.selected_columns)
    for col in table.columns:
        if col.name in provided or col.nullable:
            continue
        default = getattr(col.default, "arg", None)
        if default is not None and not callable(default):
            cols.append(col.name)
            exprs.append(literal(default))
        elif callable(default):
            cols.append(col.name)
            exprs.append(func.now())
        else:
            py = col.type.python_type
            cols.append(col.name)
            exprs.append(literal({str: "", int: 0, float: 0.0, bool: False}.get(py, "")))
    stmt = select(*exprs)
    names = list(provided) + cols
    return table.insert().from_select(names, stmt)


def load_dataset(admin: Engine, tenants: int, beliefs_per_tenant: int, users_per_tenant: int) -> dict[str, int]:
    per_user = beliefs_per_tenant // users_per_tenant
    t = _series(tenants).alias("t")
    u = _series(users_per_tenant).alias("u")
    b = _series(per_user).alias("b")
    tenant_id = func.format("t%03s", t.c.n - 1)
    user_id = func.format("u%03s", u.c.n - 1)
    belief_id = (t.c.n - 1) * beliefs_per_tenant + (u.c.n - 1) * per_user + b.c.n
    counts = {}
    with admin.begin() as conn:
        prefs = select(
            tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"), user_id.label("user_id"),
            literal(True).label("enabled"), literal(1).label("state_revision"),
        ).select_from(t, u)
        counts["mm_memory_preferences"] = conn.execute(
            _fill(MemoryPreference.__table__, prefs, {"tenant_id", "app_id", "user_id", "enabled", "state_revision"})
        ).rowcount
        beliefs = select(
            belief_id.label("id"), tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"),
            user_id.label("user_id"), literal("topic").label("dimension"),
            func.format("k%s", b.c.n).label("key"),
            func.format("claim %s about topic %s", b.c.n, b.c.n % 17).label("claim_text"),
            literal("active").label("status"), literal(0.7).label("confidence"),
            literal("hot").label("memory_tier"),
            (func.now() - func.make_interval(0, 0, 0, 0, 0, 0, b.c.n * 37)).label("last_evidence_at"),
        ).select_from(t, u, b)
        counts["mm_beliefs"] = conn.execute(
            _fill(Belief.__table__, beliefs, {
                "id", "tenant_id", "app_id", "user_id", "dimension", "key", "claim_text",
                "status", "confidence", "memory_tier", "last_evidence_at",
            })
        ).rowcount
        conn.execute(text(
            "SELECT setval(pg_get_serial_sequence('mm_beliefs', 'id'), (SELECT max(id) FROM mm_beliefs))"
        ))
        e2 = _series(2).alias("e")
        events = select(
            tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"), belief_id.label("belief_id"),
            user_id.label("user_id"), literal("support").label("event_type"),
        ).select_from(t, u, b, e2)
        counts["mm_belief_events"] = conn.execute(
            _fill(BeliefEvent.__table__, events, {"tenant_id", "app_id", "belief_id", "user_id", "event_type"})
        ).rowcount
        evidence = select(
            belief_id.label("id"), tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"),
            user_id.label("user_id"), func.format("s%s", b.c.n % 7).label("session_id"),
            func.format("ref-%s", belief_id).label("ref"), literal("message").label("source_type"),
        ).select_from(t, u, b)
        counts["mm_evidence"] = conn.execute(
            _fill(Evidence.__table__, evidence, {"id", "tenant_id", "app_id", "user_id", "session_id", "ref", "source_type"})
        ).rowcount
        conn.execute(text(
            "SELECT setval(pg_get_serial_sequence('mm_evidence', 'id'), (SELECT max(id) FROM mm_evidence))"
        ))
        links = select(
            tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"), belief_id.label("belief_id"),
            belief_id.label("evidence_id"), literal("supports").label("relation"),
        ).select_from(t, u, b)
        counts["mm_belief_evidence_links"] = conn.execute(
            _fill(BeliefEvidenceLink.__table__, links, {"tenant_id", "app_id", "belief_id", "evidence_id", "relation"})
        ).rowcount
        jobs = select(
            func.format("job-%s-%s", tenant_id, user_id).label("id"), tenant_id.label("tenant_id"),
            literal(APP_ID).label("app_id"), user_id.label("user_id"),
            func.format("%s:%s:v1", user_id, u.c.n).label("idempotency_key"),
            func.case((u.c.n % 10 == 0, literal("pending")), else_=literal("completed")).label("status"),
        ).select_from(t, u)
        counts["mm_evolution_jobs"] = conn.execute(
            _fill(EvolutionJob.__table__, jobs, {"id", "tenant_id", "app_id", "user_id", "idempotency_key", "status"})
        ).rowcount
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("ANALYZE"))
    return counts


# ---------------------------------------------------------------------------
# Key queries (Core mirrors of the ORM shapes; tenant predicates optional)
# ---------------------------------------------------------------------------

T, A, U, BID = "t042", APP_ID, "u017", None


@dataclass
class Q:
    name: str
    build: object  # (with_tenant: bool) -> statement


def _tenant(stmt, table, with_tenant: bool):
    return stmt.where(table.c.tenant_id == T, table.c.app_id == A) if with_tenant else stmt


def queries(belief_id: int, job_id: str) -> list[Q]:
    b, ev, e, l, j = (Belief.__table__, BeliefEvent.__table__, Evidence.__table__,
                      BeliefEvidenceLink.__table__, EvolutionJob.__table__)
    return [
        Q("recall: relevant slice (ILIKE, ORDER BY last_evidence_at LIMIT 150)",
          lambda wt: _tenant(
              select(b).where(b.c.user_id == U, b.c.status.in_(("active",)), b.c.memory_tier.in_(("hot", "warm")),
                              (b.c.claim_text.ilike("%topic 3%") | b.c.key.ilike("%topic 3%")))
              .order_by(b.c.last_evidence_at.desc()).limit(150), b, wt)),
        Q("recall: recent slice (ORDER BY last_evidence_at LIMIT 25)",
          lambda wt: _tenant(
              select(b).where(b.c.user_id == U, b.c.status == "active", b.c.memory_tier.in_(("hot", "warm")))
              .order_by(b.c.last_evidence_at.desc()).limit(25), b, wt)),
        Q("belief by id (session.get, PK)",
          lambda wt: _tenant(select(b).where(b.c.id == belief_id), b, wt)),
        Q("events by belief (ORDER BY created_at DESC LIMIT 50)",
          lambda wt: _tenant(select(ev).where(ev.c.belief_id == belief_id, ev.c.user_id == U)
                             .order_by(ev.c.created_at.desc()).limit(50), ev, wt)),
        Q("evidence by belief (links JOIN evidence)",
          lambda wt: _tenant(_tenant(
              select(e).select_from(l.join(e, (l.c.evidence_id == e.c.id) & (l.c.tenant_id == e.c.tenant_id) & (l.c.app_id == e.c.app_id)))
              .where(l.c.belief_id == belief_id), l, wt), e, wt)),
        Q("job claim (conditional UPDATE pending->running)",
          lambda wt: _tenant(j.update().where(j.c.id == job_id, j.c.status == "pending")
                             .values(status="running", lease_token=j.c.lease_token + 1), j, wt)),
        Q("pending-job scan (maintenance: status='pending' ORDER BY created_at LIMIT 100)",
          lambda wt: select(j).where(j.c.status == "pending").order_by(j.c.created_at).limit(100)),
        Q("delete path: beliefs by user",
          lambda wt: _tenant(b.delete().where(b.c.user_id == U), b, wt)),
        Q("delete path: events by user",
          lambda wt: _tenant(ev.delete().where(ev.c.user_id == U), ev, wt)),
    ]


def _compile(stmt, engine: Engine) -> str:
    return str(stmt.compile(engine, compile_kwargs={"literal_binds": True}))


def _explain(conn, sql: str) -> dict:
    row = conn.exec_driver_sql("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql).scalar()
    plan = row[0] if isinstance(row, list) else json.loads(row)[0]
    return plan


def _walk(node, acc):
    acc.append(node)
    for child in node.get("Plans", []):
        _walk(child, acc)
    return acc


def summarize(plan: dict) -> dict:
    root = plan["Plan"]
    nodes = _walk(root, [])
    scans = [n for n in nodes if "Scan" in n.get("Node Type", "")]
    index_names = sorted({n.get("Index Name") for n in scans if n.get("Index Name")})
    seq = sorted({n.get("Relation Name") for n in scans if n.get("Node Type") == "Seq Scan"})
    return {
        "top": root["Node Type"],
        "scan": " / ".join(sorted({n["Node Type"] for n in scans})) or root["Node Type"],
        "index": ", ".join(index_names) or ("SEQ:" + ",".join(seq) if seq else "-"),
        "est_rows": root.get("Plan Rows"),
        "rows": root.get("Actual Rows"),
        "ms": round(plan.get("Execution Time", 0.0), 2),
        "hit": sum(n.get("Shared Hit Blocks", 0) for n in nodes),
        "read": sum(n.get("Shared Read Blocks", 0) for n in nodes),
        "rows_removed": sum(n.get("Rows Removed by Filter", 0) for n in nodes),
    }


def run_pass(admin: Engine, app: Engine, qs: list[Q]) -> list[tuple[str, str, dict]]:
    out = []
    for q in qs:
        # C: app predicates, superuser (RLS bypassed)
        with admin.begin() as conn:
            out.append((q.name, "C", summarize(_explain(conn, _compile(q.build(True), admin)))))
            conn.rollback()
        # A: app predicates + RLS; B: RLS alone (same transaction setting)
        with tenant_scope(T, A):
            for mode, with_tenant in (("A", True), ("B", False)):
                with app.begin() as conn:
                    out.append((q.name, mode, summarize(_explain(conn, _compile(q.build(with_tenant), app)))))
                    conn.rollback()
    return out


CANDIDATE_INDEXES = [
    Index("ix_mm_beliefs_tenant_user_recent", Belief.__table__.c.tenant_id, Belief.__table__.c.app_id,
          Belief.__table__.c.user_id, Belief.__table__.c.last_evidence_at.desc(), postgresql_concurrently=True),
    Index("ix_mm_belief_events_tenant_belief", BeliefEvent.__table__.c.tenant_id, BeliefEvent.__table__.c.app_id,
          BeliefEvent.__table__.c.belief_id, postgresql_concurrently=True),
    Index("ix_mm_belief_evidence_links_tenant_belief", BeliefEvidenceLink.__table__.c.tenant_id,
          BeliefEvidenceLink.__table__.c.app_id, BeliefEvidenceLink.__table__.c.belief_id, postgresql_concurrently=True),
    Index("ix_mm_evolution_jobs_pending", EvolutionJob.__table__.c.created_at,
          postgresql_where=text("status = 'pending'"), postgresql_concurrently=True),
]


def create_candidates(admin: Engine) -> None:
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for index in CANDIDATE_INDEXES:
            conn.execute(CreateIndex(index, if_not_exists=True))
        conn.execute(text("ANALYZE"))


def render(title: str, rows: list[tuple[str, str, dict]], counts: dict[str, int], elapsed: float) -> str:
    lines = [f"## {title}", "", "Rows: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()),
             f"; load {elapsed:.1f}s", "",
             "| query | mode | scan | index | est rows | rows | removed by filter | ms | shared hit/read |",
             "|---|---|---|---|---:|---:|---:|---:|---|"]
    for name, mode, s in rows:
        lines.append(
            f"| {name} | {mode} | {s['scan']} | {s['index']} | {s['est_rows']} | {s['rows']} | "
            f"{s['rows_removed']} | {s['ms']} | {s['hit']}/{s['read']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-url", required=True)
    ap.add_argument("--app-url", required=True)
    ap.add_argument("--tenants", type=int, default=100)
    ap.add_argument("--beliefs-per-tenant", type=int, default=10_000)
    ap.add_argument("--users-per-tenant", type=int, default=100)
    ap.add_argument("--candidate-indexes", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    admin = create_engine(args.admin_url)
    app = create_engine(args.app_url)
    t0 = time.monotonic()
    Base.metadata.drop_all(admin)
    Base.metadata.create_all(admin)
    install_row_level_security(admin)
    counts = load_dataset(admin, args.tenants, args.beliefs_per_tenant, args.users_per_tenant)
    elapsed = time.monotonic() - t0
    belief_id = 42 * args.beliefs_per_tenant + 17 * (args.beliefs_per_tenant // args.users_per_tenant) + 3
    job_id = f"job-{T}-u019"  # u019 -> n=20 -> pending (n % 10 == 0)
    qs = queries(belief_id, job_id)
    report = render("Baseline indexes (as shipped)", run_pass(admin, app, qs), counts, elapsed)
    if args.candidate_indexes:
        t1 = time.monotonic()
        create_candidates(admin)
        report += "\n" + render(
            f"With candidate composite indexes ({time.monotonic() - t1:.1f}s to build CONCURRENTLY)",
            run_pass(admin, app, qs), counts, elapsed,
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
    print(report)
    admin.dispose()
    app.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
