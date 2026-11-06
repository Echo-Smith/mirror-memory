"""Invariant → test mapping checks that the real code must PASS (not xfail).

Companion to ``docs/invariants.md``.  Where ``test_invariants_xfail.py`` guards
the ONE literal v2 shape v1 lacks (a unique ``source_event_id`` column), this
module pins the behaviour v1 *actually* provides — **and its limits** — so the
I4 row in the docs is backed by runnable, dual-leg tests rather than prose and
is not overstated.

## What v2's I4 says (see ``docs/v2-inherited/invariants.md``)

> I4: 同一 source_event_id 在同一 Scope 内全局唯一  (verification: **Claim
> 阶段唯一约束** — a claim-time unique constraint).
> Violation ⇒ idempotency break / duplicate memory.  R03 adds: the same event
> re-sent must not produce a duplicate result, and the same key with different
> content must return a conflict.

``Scope`` in v2 is the trusted ``(tenant_id, app_id, subject_id)`` triple
(R01; the Claim SQL filters ``scope_control`` on exactly those three); its
``scope_id`` identifies that scope-control row.  ``source_event_id`` is the
caller's event identifier (produced by the event source), written when the
event is claimed, and uniqueness is enforced **at claim time**.

## What v1 has, and how close it is (per mechanism)

* **``mm_evidence.ref`` + ``uq_mm_evidence_user_ref`` UNIQUE(tenant_id, app_id,
  user_id, ref) + ``record_evidence`` idempotency** — the closest analogue.
  ``ref`` is the caller's source identifier (``evidence_message_ids`` from the
  extraction pipeline, i.e. the source event/message ids).  The scope is
  ``(tenant_id, app_id, user_id)``.
  - *Serial* re-record of the same ``ref`` → same row (idempotent): **tested &
    equivalent** (``test_i4_v1_serial_idempotency_ref_uniqueness``).
  - *Concurrent* re-record → the DB constraint still guarantees **at most one
    persisted row**, but ``record_evidence`` is a read-then-write (SELECT then
    INSERT); it is **not** an atomic upsert.  A racing loser hits an
    ``IntegrityError`` at flush (the pipeline swallows it and drops that
    evidence), rather than both callers receiving the shared row.  So the
    *result* (one row) holds, the *graceful-dedup semantics* do not.
    **Partially equivalent** (``test_i4_concurrent_same_ref_yields_one_row``).
  - *Scope* boundary → same ``ref`` under a different user / tenant is a
    distinct source, no collision, no cross-tenant visibility: **equivalent at
    the isolation level** (``test_i4_same_ref_different_user_is_distinct`` /
    ``test_i4_same_ref_other_tenant_does_not_collide_or_leak``).
* **``mm_proposal_log.proposal_id`` UNIQUE** — ``proposal_id`` is a random
  ``uuid4`` generated per proposal object, NOT derived from the source event.
  It dedups **replays of the same proposal object**; a fresh proposal built
  from the same source event gets a new id and is not deduped.  **Not
  equivalent** to source-event uniqueness.
* **``mm_evolution_jobs`` UNIQUE(tenant_id, app_id, idempotency_key)** — the key
  is ``f"{user_id}:{evidence_watermark}:{prompt_version}"``, i.e. a *job/batch*
  identity, not a per-``source_event_id`` identity.  It prevents duplicate
  *jobs* for the same watermark; it does not make individual source events
  unique.  **Partially covers** (job-level dedup, not source-event-level).

**The gap** (stated, not faked): v2's *claim-time* unique constraint on
``source_event_id`` has **no v1 counterpart** — v1 has no claim phase for
evidence and no ``source_event_id`` column.  What v1 guarantees is a
*persist-time* row-level uniqueness on ``ref``.  That is why the verdict is
**partial equivalence**, not "equivalent".

The tests below execute the guarantee that DOES exist (and pass only when it
holds).  The v2-only shape is guarded as a **structural warning** (not a
behaviour test) in ``test_invariants_xfail.py``.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Evidence
from mirror_memory.core.repository import record_evidence
from mirror_memory.core.tenancy import tenant_scope


# ---------------------------------------------------------------------------
# static check: the constraint that carries the v1 guarantee
# ---------------------------------------------------------------------------


def _evidence_ref_unique_constraint_present() -> bool:
    """The DB-level constraint behind v1's source-ref uniqueness."""
    for c in Evidence.__table__.constraints:
        if c.__class__.__name__ != "UniqueConstraint":
            continue
        cols = {col.name for col in c.columns}
        if cols == {"tenant_id", "app_id", "user_id", "ref"}:
            return True
    return False


# ---------------------------------------------------------------------------
# serial idempotency (dual-leg: db_session -> SQLite + PostgreSQL)
# ---------------------------------------------------------------------------


def test_i4_v1_serial_idempotency_ref_uniqueness(db_session):
    """I4 (serial, v1): the same source ``ref`` in one scope is one row.

    Runs on both SQLite and PostgreSQL via ``db_session``.  This is the
    serial-idempotency half of the equivalence.
    """
    assert _evidence_ref_unique_constraint_present(), (
        "mm_evidence must keep uq_mm_evidence_user_ref(tenant_id, app_id, "
        "user_id, ref) — it is the v1 realisation of I4's serial half"
    )

    with tenant_scope("default", "default"):
        first = record_evidence(db_session, "u-i4", ref="evt-123", content="hello")
        db_session.commit()
        second = record_evidence(db_session, "u-i4", ref="evt-123", content="hello again")
        db_session.commit()

        rows = (
            db_session.query(Evidence)
            .filter_by(user_id="u-i4", ref="evt-123")
            .all()
        )

    assert first.id == second.id
    assert len(rows) == 1, [r.id for r in rows]

    # A different ref in the same scope is a distinct source.
    with tenant_scope("default", "default"):
        third = record_evidence(db_session, "u-i4", ref="evt-999", content="other")
        db_session.commit()
        count = db_session.query(Evidence).filter_by(user_id="u-i4").count()
    assert third.id != first.id
    assert count == 2


# ---------------------------------------------------------------------------
# real concurrency: same ref, same scope, multiple threads -> exactly one row
# ---------------------------------------------------------------------------


def _sqlite_file_sessionmaker(tmp_path, name: str):
    """A file-backed SQLite DB so several threads share one database.

    In-memory ``sqlite://`` is per-connection and cannot be shared across
    threads; concurrency tests therefore use a ``tmp_path`` file.
    ``check_same_thread=False`` lets each thread open/hold its own session
    against the same engine.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path}/{name}",
        connect_args={"check_same_thread": False},
    )
    from mirror_memory.core import schema_migration as _sm

    Base.metadata.create_all(engine)
    _sm.ensure_watermark(engine)
    return engine, sessionmaker(bind=engine)


@pytest.fixture(params=["sqlite", "postgres"])
def conc_backend(request, tmp_path):
    """The concurrency backend: file-SQLite always, PostgreSQL when available.

    Mirrors ``test_m3_acceptance.py::db_url``: the ``postgres`` param pulls the
    PG fixture LAZILY, so a SQLite-only run does not demand a database (the
    param skips with "needs MM_DATABASE_URL").  On a PG leg BOTH params run.
    Yields ``(engine, sessionmaker)``; the caller disposes the engine.
    """
    if request.param == "sqlite":
        engine, factory = _sqlite_file_sessionmaker(tmp_path, "i4-conc.db")
        try:
            yield engine, factory
        finally:
            engine.dispose()
    else:
        # pg_app_url is the RLS-protected app-role runtime URL, and it builds a
        # fresh per-test schema; skip automatically when no PG is configured.
        app_url = request.getfixturevalue("pg_app_url")
        engine = create_engine(app_url)
        factory = sessionmaker(bind=engine)
        try:
            yield engine, factory
        finally:
            engine.dispose()


def test_i4_concurrent_same_ref_yields_one_row(conc_backend):
    """I4 (concurrent, v1): N threads record the SAME ref in one scope →
    exactly ONE persisted Evidence row.

    Parametrized over ``sqlite`` (file DB) and ``postgres`` (app role under
    RLS, skipped when no PG is configured).  This executes the real
    concurrency path.  ``record_evidence`` is a read-then-write, not an atomic
    upsert, so a racing thread can lose on the UNIQUE constraint
    (``IntegrityError``) — the pipeline swallows that.  The *guarantee* under
    test is the outcome: exactly one row survives, i.e. the invariant is not
    broken by concurrent ingestion.
    """
    assert _evidence_ref_unique_constraint_present()
    engine, factory = conc_backend
    REF = "evt-concurrent-1"
    TENANT = ("default", "default")
    N = 8
    barrier = threading.Barrier(N)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        # Sessions are single-tenant: the session MUST be created inside the
        # tenant scope it will run under (bound tenant is captured at creation).
        with tenant_scope(*TENANT):
            session = factory()
            try:
                barrier.wait(timeout=10)
                try:
                    record_evidence(session, "u-conc", ref=REF, content=f"c{idx}")
                    session.commit()
                    result = "inserted"
                except Exception as exc:  # noqa: BLE001 — the race loser
                    session.rollback()
                    result = type(exc).__name__
                with lock:
                    outcomes.append(result)
            finally:
                session.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
        assert not t.is_alive(), "a worker thread hung"

    # Inspect the final table state in a fresh session/tenant scope.
    with tenant_scope(*TENANT):
        check = factory()
        try:
            rows = check.query(Evidence).filter_by(user_id="u-conc", ref=REF).all()
        finally:
            check.close()

    assert len(rows) == 1, (
        "I4 broken under concurrency: same ref produced "
        f"{len(rows)} rows (outcomes={outcomes})"
    )
    # At least one caller succeeded; the rest either shared the row (sequenced)
    # or lost the race on the constraint — either way, never a second row.
    assert any(o == "inserted" for o in outcomes), outcomes


# ---------------------------------------------------------------------------
# scope semantics: same ref, DIFFERENT user -> distinct sources, no collision
# ---------------------------------------------------------------------------


def test_i4_same_ref_different_user_is_distinct(db_session):
    """The same ``ref`` under a different ``user_id`` in the same tenant/app is
    a DIFFERENT source — it must not collide with the first user's row.

    Runs on both legs.  This checks that the "scope" in the v1 constraint is
    genuinely ``(tenant, app, user)`` — the source identifier is unique per
    scope, not globally across users.
    """
    with tenant_scope("default", "default"):
        a = record_evidence(db_session, "alice", ref="evt-shared", content="from alice")
        b = record_evidence(db_session, "bob", ref="evt-shared", content="from bob")
        db_session.commit()

        alice_rows = db_session.query(Evidence).filter_by(user_id="alice", ref="evt-shared").all()
        bob_rows = db_session.query(Evidence).filter_by(user_id="bob", ref="evt-shared").all()

    assert a.id != b.id, "same ref across users must NOT collapse to one row"
    assert len(alice_rows) == 1
    assert len(bob_rows) == 1
    assert alice_rows[0].content == "from alice"
    assert bob_rows[0].content == "from bob"


# ---------------------------------------------------------------------------
# scope semantics: same ref, DIFFERENT tenant -> no collision, no cross-tenant
# visibility
# ---------------------------------------------------------------------------


def test_i4_same_ref_other_tenant_does_not_collide_or_leak(conc_backend):
    """The same ``ref`` (and same ``user_id``) under a DIFFERENT tenant/app is a
    separate source: no collision on write, and no cross-tenant read leak.

    Parametrized over ``sqlite`` and ``postgres`` (skipped on the SQLite leg).
    It needs its OWN database (the shared ``db_session`` is bound to one tenant)
    so both tenants live in the same tables; on SQLite the ORM tenant filter
    enforces isolation, on PostgreSQL row-level security is the second line of
    defence.  Each session is created inside the tenant scope it runs under
    (sessions are single-tenant).
    """
    engine, factory = conc_backend

    with tenant_scope("tenant-a", "app-1"):
        sa = factory()
        try:
            a = record_evidence(sa, "shared-user", ref="evt-x", content="tenant A secret")
            sa.commit()
            a_id = a.id  # capture before the session closes
        finally:
            sa.close()

    # Same user id + ref, different tenant -> must be a distinct row and a
    # distinct scope, not an update of tenant A's row.
    with tenant_scope("tenant-b", "app-1"):
        sb = factory()
        try:
            b = record_evidence(sb, "shared-user", ref="evt-x", content="tenant B data")
            sb.commit()
            b_id = b.id
            b_visible = sb.query(Evidence).filter_by(user_id="shared-user", ref="evt-x").all()
        finally:
            sb.close()

    # Under tenant B the query must see ONLY tenant B's row (no leak of A).
    assert len(b_visible) == 1, [r.id for r in b_visible]
    assert b_visible[0].content == "tenant B data"
    assert b_visible[0].tenant_id == "tenant-b"

    # Tenant A still sees only its own row.
    with tenant_scope("tenant-a", "app-1"):
        sa2 = factory()
        try:
            a_visible = sa2.query(Evidence).filter_by(user_id="shared-user", ref="evt-x").all()
        finally:
            sa2.close()
    assert len(a_visible) == 1, [r.id for r in a_visible]
    assert a_visible[0].content == "tenant A secret"
    assert a_visible[0].tenant_id == "tenant-a"

    assert a_id != b_id, "same ref in different tenants must NOT collide"


# ---------------------------------------------------------------------------
# STRUCTURAL WARNING counterexamples (NOT behaviour tests)
#
# These drive the *structural* checker
# ``test_invariants_xfail.source_event_id_uniqueness_shape`` against fabricated
# table shapes.  Their whole point is that a structural warning must NOT be
# fooled into reporting I4's claim-time behaviour as implemented by (a) a bare
# column name or (b) a *superset* unique constraint that contains the target
# column but has extra columns.  A superset is WEAKER than the exact in-scope
# identity and must be judged "does not meet the target".
# ---------------------------------------------------------------------------


def _fabricated_evidence_table(*, columns, unique_sets, name="mm_evidence_fake"):
    """Build a throwaway ``Table`` mirroring the Evidence shape for the checker.

    ``columns`` is the full column list; ``unique_sets`` is a list of column-name
    iterables, each becoming one ``UniqueConstraint``.
    """
    from sqlalchemy import Column, MetaData, String, Table, UniqueConstraint

    md = MetaData()
    cols = [Column(c, String(64), primary_key=(i == 0)) for i, c in enumerate(columns)]
    t = Table(name, md, *cols)
    for uset in unique_sets:
        t.append_constraint(UniqueConstraint(*[t.c[c] for c in uset]))
    return t


def test_i4_structural_warning_accepts_exact_target_shape():
    """Positive control: the checker DOES accept the exact v2 shape — so the
    counterexample below is meaningful (the checker is not just always-False)."""
    from tests.test_invariants_xfail import source_event_id_uniqueness_shape

    table = _fabricated_evidence_table(
        columns=["id", "tenant_id", "app_id", "user_id", "source_event_id"],
        unique_sets=[("tenant_id", "app_id", "user_id", "source_event_id")],
    )
    present, why = source_event_id_uniqueness_shape(table)
    assert present is True, why


def test_i4_structural_warning_rejects_superset_constraint():
    """COUNTEREXAMPLE: a composite unique constraint that CONTAINS
    ``source_event_id`` but has EXTRA columns must be judged NOT implementing
    the target — a superset constraint must never pass the structural warning.

    This is the guard against claiming equivalence from "an arbitrary superset
    unique constraint": adding ``observed_at`` (or any extra column) makes the
    constraint weaker (many rows may share one ``source_event_id`` as the extra
    column varies), so the exact in-scope identity is NOT enforced.
    """
    from tests.test_invariants_xfail import source_event_id_uniqueness_shape

    superset = _fabricated_evidence_table(
        columns=["id", "tenant_id", "app_id", "user_id", "source_event_id", "observed_at"],
        unique_sets=[
            ("tenant_id", "app_id", "user_id", "source_event_id", "observed_at"),
        ],
    )
    present, why = source_event_id_uniqueness_shape(superset)
    assert present is False, (
        "a SUPERSET unique constraint containing source_event_id plus an extra "
        f"column must NOT be accepted as implementing I4 (got {present}: {why})"
    )
    assert "extra column" in why.lower(), why


def test_i4_structural_warning_rejects_superset_even_with_target_also_present():
    """A table that has BOTH the exact target constraint AND a superset one must
    still be accepted (the exact one is present), proving the check keys on the
    exact shape rather than merely rejecting anything with extras."""
    from tests.test_invariants_xfail import source_event_id_uniqueness_shape

    both = _fabricated_evidence_table(
        columns=["id", "tenant_id", "app_id", "user_id", "source_event_id", "observed_at"],
        unique_sets=[
            ("tenant_id", "app_id", "user_id", "source_event_id"),
            ("tenant_id", "app_id", "user_id", "source_event_id", "observed_at"),
        ],
    )
    present, why = source_event_id_uniqueness_shape(both)
    assert present is True, why


def test_i4_structural_warning_rejects_bare_column_and_missing_scope():
    """Negative controls: a bare ``source_event_id`` column with no unique
    constraint, and a constraint missing the scope/subject columns, are both
    rejected — a field name is not proof of the behaviour."""
    from tests.test_invariants_xfail import source_event_id_uniqueness_shape

    # bare column, no unique constraint at all
    bare = _fabricated_evidence_table(
        columns=["id", "tenant_id", "app_id", "user_id", "source_event_id"],
        unique_sets=[],
    )
    assert source_event_id_uniqueness_shape(bare)[0] is False

    # constraint exists but omits the scope (no tenant/app) -> not scope-scoped
    no_scope = _fabricated_evidence_table(
        columns=["id", "user_id", "source_event_id"],
        unique_sets=[("user_id", "source_event_id")],
    )
    assert source_event_id_uniqueness_shape(no_scope)[0] is False

    # constraint omits the subject dimension -> not the in-scope identity
    no_subject = _fabricated_evidence_table(
        columns=["id", "tenant_id", "app_id", "source_event_id"],
        unique_sets=[("tenant_id", "app_id", "source_event_id")],
    )
    assert source_event_id_uniqueness_shape(no_subject)[0] is False
