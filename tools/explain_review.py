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
the review considered (CONCURRENTLY, outside any transaction) after the
first pass and re-explains every query, so the before/after plans sit side
by side; the one index the review adopted (the pending-jobs partial index)
ships on the model and is present in the baseline pass.  Timings are indicative (plan shape is the decision input);
the harness-wide performance numbers come from CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, case, cast, create_engine, func, literal, select, text, true
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateIndex

sys.path.insert(0, "src")
warnings.filterwarnings("ignore", message=".*cartesian product.*")  # the generators cross-join on purpose

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


def _series(n: int, name: str):
    """generate_series(1, n) AS <name>(n)"""
    return func.generate_series(1, n).table_valued("n").render_derived(name=name)


def _fill(table, select_stmt, provided: set[str]):
    """INSERT ... SELECT with every NOT NULL column satisfied.

    Columns the caller did not provide get the model's scalar default, a
    neutral literal of the right type, or ``now()`` -- Python-side defaults
    do not apply to INSERT ... SELECT, so this is what ``create_all`` + the
    ORM would have written.  Column names follow the SELECT list order.
    """
    names = [c.key for c in select_stmt.selected_columns]
    assert set(names) == provided, (sorted(names), sorted(provided))
    exprs = []  # extra columns appended to the caller's SELECT (its FROM is kept)
    for col in table.columns:
        if col.name in provided or col.nullable or col.autoincrement is True:
            continue  # serial ids come from the sequence
        default = getattr(col.default, "arg", None)
        names.append(col.name)
        if default is not None and not callable(default):
            exprs.append(literal(default))
        elif isinstance(col.type, DateTime) or callable(default):
            exprs.append(func.now())
        elif isinstance(col.type, Boolean):
            exprs.append(literal(False))
        elif isinstance(col.type, Integer):
            exprs.append(literal(0))
        elif isinstance(col.type, Float):
            exprs.append(literal(0.0))
        else:
            exprs.append(literal(""))
    return table.insert().from_select(names, select_stmt.add_columns(*exprs))


def load_dataset(admin: Engine, tenants: int, beliefs_per_tenant: int, users_per_tenant: int) -> dict[str, int]:
    per_user = beliefs_per_tenant // users_per_tenant
    t = _series(tenants, "t")
    u = _series(users_per_tenant, "u")
    b = _series(per_user, "b")
    tenant_id = func.concat("t", func.lpad(cast(t.c.n - 1, String), 3, "0"))
    user_id = func.concat("u", func.lpad(cast(u.c.n - 1, String), 3, "0"))
    belief_id = (t.c.n - 1) * beliefs_per_tenant + (u.c.n - 1) * per_user + b.c.n
    counts = {}
    with admin.begin() as conn:
        prefs = select(
            tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"), user_id.label("user_id"),
            literal(True).label("enabled"), literal(1).label("state_revision"),
        ).select_from(t.join(u, true()))
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
        ).select_from(t.join(u, true()).join(b, true()))
        counts["mm_beliefs"] = conn.execute(
            _fill(Belief.__table__, beliefs, {
                "id", "tenant_id", "app_id", "user_id", "dimension", "key", "claim_text",
                "status", "confidence", "memory_tier", "last_evidence_at",
            })
        ).rowcount
        conn.execute(text(
            "SELECT setval(pg_get_serial_sequence('mm_beliefs', 'id'), (SELECT max(id) FROM mm_beliefs))"
        ))
        e2 = _series(2, "e")
        events = select(
            tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"), belief_id.label("belief_id"),
            user_id.label("user_id"), literal("support").label("event_type"),
        ).select_from(t.join(u, true()).join(b, true()).join(e2, true()))
        counts["mm_belief_events"] = conn.execute(
            _fill(BeliefEvent.__table__, events, {"tenant_id", "app_id", "belief_id", "user_id", "event_type"})
        ).rowcount
        evidence = select(
            belief_id.label("id"), tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"),
            user_id.label("user_id"), func.format("s%s", b.c.n % 7).label("session_id"),
            func.format("ref-%s", belief_id).label("ref"), literal("message").label("source_type"),
        ).select_from(t.join(u, true()).join(b, true()))
        counts["mm_evidence"] = conn.execute(
            _fill(Evidence.__table__, evidence, {"id", "tenant_id", "app_id", "user_id", "session_id", "ref", "source_type"})
        ).rowcount
        conn.execute(text(
            "SELECT setval(pg_get_serial_sequence('mm_evidence', 'id'), (SELECT max(id) FROM mm_evidence))"
        ))
        links = select(
            tenant_id.label("tenant_id"), literal(APP_ID).label("app_id"), belief_id.label("belief_id"),
            belief_id.label("evidence_id"), literal("supports").label("relation"),
        ).select_from(t.join(u, true()).join(b, true()))
        counts["mm_belief_evidence_links"] = conn.execute(
            _fill(BeliefEvidenceLink.__table__, links, {"tenant_id", "app_id", "belief_id", "evidence_id", "relation"})
        ).rowcount
        jobs = select(
            func.format("job-%s-%s", tenant_id, user_id).label("id"), tenant_id.label("tenant_id"),
            literal(APP_ID).label("app_id"), user_id.label("user_id"),
            func.format("%s:%s:v1", user_id, u.c.n).label("idempotency_key"),
            case((u.c.n % 10 == 0, literal("pending")), else_=literal("completed")).label("status"),
        ).select_from(t.join(u, true()))
        counts["mm_evolution_jobs"] = conn.execute(
            _fill(EvolutionJob.__table__, jobs, {"id", "tenant_id", "app_id", "user_id", "idempotency_key", "status"})
        ).rowcount
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("ANALYZE"))
    with admin.connect() as conn:
        for name in list(counts):
            counts[name] = conn.execute(
                select(func.count()).select_from(Base.metadata.tables[name])
            ).scalar_one()
    return counts


# ---------------------------------------------------------------------------
# Key queries (Core mirrors of the ORM shapes; tenant predicates optional)
# ---------------------------------------------------------------------------

T, A, U = "t042", APP_ID, "u017"  # re-pointed by main() for small datasets


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
        with admin.connect() as conn:
            out.append((q.name, "C", summarize(_explain(conn, _compile(q.build(True), admin)))))
            conn.rollback()
        # A: app predicates + RLS; B: RLS alone (same transaction setting)
        with tenant_scope(T, A):
            for mode, with_tenant in (("A", True), ("B", False)):
                with app.connect() as conn:
                    out.append((q.name, mode, summarize(_explain(conn, _compile(q.build(with_tenant), app)))))
                    conn.rollback()
    return out


def _detached(name: str):
    """A copy of the model table on a throwaway MetaData: an Index built on
    the live table's columns would attach itself to Base.metadata and make
    every later create_all emit it (CONCURRENTLY, inside a transaction)."""
    from sqlalchemy import MetaData

    return Base.metadata.tables[name].to_metadata(MetaData())


def candidate_indexes() -> list[Index]:
    """Composite indexes considered and NOT adopted (the before/after plans
    in the report are the evidence).  The pending-jobs partial index that
    the review did adopt now ships on the model (migration step 4)."""
    b, ev, l = (_detached("mm_beliefs"), _detached("mm_belief_events"),
                _detached("mm_belief_evidence_links"))
    return [
        Index("ix_mm_beliefs_tenant_user_recent", b.c.tenant_id, b.c.app_id, b.c.user_id,
              b.c.last_evidence_at.desc(), postgresql_concurrently=True),
        Index("ix_mm_belief_events_tenant_user", ev.c.tenant_id, ev.c.app_id, ev.c.user_id,
              postgresql_concurrently=True),
        Index("ix_mm_belief_evidence_links_tenant_belief", l.c.tenant_id, l.c.app_id, l.c.belief_id,
              postgresql_concurrently=True),
    ]


def create_candidates(admin: Engine) -> None:
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for index in candidate_indexes():
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


def bootstrap_app_login(admin_url: str, password: str) -> None:
    """CI only: make sure the app role exists as a LOGIN with *password*
    (psycopg identifier/literal composition -- no string assembly)."""
    import psycopg
    from psycopg import sql
    from sqlalchemy.engine import make_url

    from mirror_memory.core import rls

    conninfo = make_url(admin_url).set(drivername="postgresql").render_as_string(hide_password=False)
    name = rls.app_role()
    with psycopg.connect(conninfo, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = %s", (name,)).fetchone()
        verb = "ALTER ROLE {} WITH LOGIN PASSWORD {}" if exists else "CREATE ROLE {} LOGIN PASSWORD {}"
        conn.execute(sql.SQL(verb).format(sql.Identifier(name), sql.Literal(password)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-url", required=True)
    ap.add_argument("--app-url", required=True)
    ap.add_argument("--tenants", type=int, default=100)
    ap.add_argument("--beliefs-per-tenant", type=int, default=10_000)
    ap.add_argument("--users-per-tenant", type=int, default=100)
    ap.add_argument("--candidate-indexes", action="store_true")
    ap.add_argument("--bootstrap-app-password", default="",
                    help="create/alter the app login with this password first (CI: no prior role setup)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.bootstrap_app_password:
        bootstrap_app_login(args.admin_url, args.bootstrap_app_password)
    admin = create_engine(args.admin_url)
    app = create_engine(args.app_url)
    t0 = time.monotonic()
    Base.metadata.drop_all(admin)
    Base.metadata.create_all(admin)
    install_row_level_security(admin)
    counts = load_dataset(admin, args.tenants, args.beliefs_per_tenant, args.users_per_tenant)
    elapsed = time.monotonic() - t0
    global T, U
    per_user = args.beliefs_per_tenant // args.users_per_tenant
    t_idx, u_idx = min(42, args.tenants - 1), min(17, args.users_per_tenant - 1)
    T, U = f"t{t_idx:03d}", f"u{u_idx:03d}"
    belief_id = t_idx * args.beliefs_per_tenant + u_idx * per_user + 3
    pending_idx = 9 if args.users_per_tenant >= 10 else args.users_per_tenant - 1
    job_id = f"job-{T}-u{pending_idx:03d}"  # (idx+1) % 10 == 0 -> seeded as pending
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
