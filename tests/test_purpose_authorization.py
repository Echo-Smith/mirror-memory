"""M3 batch-1 task 1: the purpose authorization model.

Covers: the authorize() refusal matrix (each reason), the fixed operation
vocabulary, the fail-closed counterexample (missing purpose is refused,
never fail-open like v2), no caching (a committed revocation bites the
next call, even from a session that saw the allow), the version CAS
(first issuance and updates, 50 real-concurrency rounds each on
PostgreSQL), the audit events, migration step 6, and the RLS posture on
the new tables.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from mirror_memory.core import authorization as auth
from mirror_memory.core.models import Base
from mirror_memory.core.authorization import (
    Decision,
    StaleGrantVersion,
    UnknownOperation,
    authorize,
    grant_purpose,
    revoke_purpose,
)
from mirror_memory.core.models import PurposeGrant, PurposeGrantEvent
from mirror_memory.core.repository import set_feature_consent, set_memory_enabled
from mirror_memory.core.tenancy import no_tenant_scope, system_scope, tenant_scope


def _migration_cfg():
    """A deletion-log config for step-11 migrations that don't exercise
    the log itself (M4 patch 2: the migration refuses to mint a
    data-bearing database without a log to consult)."""
    import tempfile

    from mirror_memory.config.loader import load_config

    cfg = load_config("config/")
    cfg.deletion_log.directory = tempfile.mkdtemp(prefix="mm-migration-log-")
    return cfg


PURPOSE = "api"


def _grant(session, user_id="u1", purpose=PURPOSE, operations=("recall", "observe"), **kw):
    grant = grant_purpose(
        session, user_id, purpose,
        operations=list(operations), issuer="reviewer", **kw,
    )
    session.commit()
    return grant


# ---------------------------------------------------------------------------
# The authorize() matrix
# ---------------------------------------------------------------------------


def test_allowed_decision_carries_the_grant_version(db_session):
    _grant(db_session)
    decision = authorize(db_session, "u1", PURPOSE, "recall")
    assert decision.allowed and decision.reason == "granted"
    assert decision.grant_version == 1 and decision.purpose == PURPOSE


def test_missing_purpose_is_refused_not_fail_open(db_session):
    """The explicit counterexample to v2's fail-open: no purpose at all,
    an empty one, and a blank one are refusals -- the user HAS a grant for
    other purposes, so a fail-open would silently allow here."""
    _grant(db_session)
    for purpose in (None, "", "   "):
        decision = authorize(db_session, "u1", purpose, "recall")
        assert not decision.allowed and decision.reason == "missing_purpose"
        assert decision.grant_version is None


def test_no_grant_is_refused(db_session):
    decision = authorize(db_session, "u1", "never-granted", "recall")
    assert not decision.allowed and decision.reason == "no_grant"
    assert decision.grant_version is None


def test_revoked_is_refused(db_session):
    _grant(db_session)
    revoke_purpose(db_session, "u1", PURPOSE, actor="reviewer", reason="offboarding")
    db_session.commit()
    decision = authorize(db_session, "u1", PURPOSE, "recall")
    assert not decision.allowed and decision.reason == "revoked"
    assert decision.grant_version == 2  # the revocation bumped the version


def test_expired_is_refused(db_session):
    grant = _grant(db_session)
    grant_purpose(
        db_session, "u1", PURPOSE,
        operations=["recall"], issuer="reviewer",
        expected_version=1,
        expires_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1),
    )
    db_session.commit()
    decision = authorize(db_session, "u1", PURPOSE, "recall")
    assert not decision.allowed and decision.reason == "expired"
    assert decision.grant_version == 2


def test_operation_not_allowed_is_refused(db_session):
    _grant(db_session, operations=("observe",))
    decision = authorize(db_session, "u1", PURPOSE, "recall")
    assert not decision.allowed and decision.reason == "operation_not_allowed"
    # an operation the grant DOES allow passes
    assert authorize(db_session, "u1", PURPOSE, "observe").allowed


def test_feature_disabled_is_refused(db_session):
    """Layer 2: the purpose grant allows, the feature consent forbids."""
    _grant(db_session, operations=("recall",))
    set_feature_consent(db_session, "u1", "recall", granted=False)
    db_session.commit()
    decision = authorize(db_session, "u1", PURPOSE, "recall")
    assert not decision.allowed and decision.reason == "feature_disabled"
    assert decision.grant_version == 1
    # observe inherits the global memory preference, which is still on
    assert authorize(db_session, "u1", PURPOSE, "observe").reason == "operation_not_allowed"


def test_missing_purpose_beats_no_grant_and_feature_state(db_session):
    set_memory_enabled(db_session, "u1", False)
    db_session.commit()
    assert authorize(db_session, "u1", None, "recall").reason == "missing_purpose"


def test_unknown_operation_raises_everywhere(db_session):
    with pytest.raises(UnknownOperation):
        authorize(db_session, "u1", PURPOSE, "admin")
    with pytest.raises(UnknownOperation):
        grant_purpose(db_session, "u1", PURPOSE, operations=["admin"], issuer="x")
    with pytest.raises(UnknownOperation):
        grant_purpose(db_session, "u1", PURPOSE, operations=[], issuer="x")


def test_authorize_refuses_under_system_scope(db_session):
    _grant(db_session)
    with system_scope("test: the empty-pass shape"):
        with pytest.raises(RuntimeError, match="system_scope"):
            authorize(db_session, "u1", PURPOSE, "recall")


def test_authorize_refuses_without_a_tenant_context(db_session):
    _grant(db_session)
    with no_tenant_scope():
        with pytest.raises(Exception) as excinfo:  # TenantContextMissing from the hooks
            authorize(db_session, "u1", PURPOSE, "recall")
    assert "tenant" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# No caching: a committed revocation bites immediately
# ---------------------------------------------------------------------------


def test_revocation_bites_within_the_same_session_after_commit(db_session):
    _grant(db_session)
    assert authorize(db_session, "u1", PURPOSE, "recall").allowed
    revoke_purpose(db_session, "u1", PURPOSE, actor="reviewer")
    db_session.commit()
    assert authorize(db_session, "u1", PURPOSE, "recall").reason == "revoked"


def test_revocation_bites_a_session_that_saw_the_allow(pg_app_url):
    """The no-caching contract across sessions AND transactions: session A
    authorizes; session B revokes and commits; A's next transaction must
    see the revocation (authorize re-reads the database on every call, and
    the query -- unlike session.get -- cannot be answered from a stale
    identity map)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(pg_app_url)
    factory = sessionmaker(bind=engine)
    try:
        with tenant_scope("default", "default"):
            sa = factory()
            _grant(sa)
            assert authorize(sa, "u1", PURPOSE, "recall").allowed
            sb = factory()
            revoke_purpose(sb, "u1", PURPOSE, actor="reviewer")
            sb.commit()
            sb.close()
            sa.commit()  # A's next transaction
            assert authorize(sa, "u1", PURPOSE, "recall").reason == "revoked"
            sa.close()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# The version CAS: first issuance and updates, 50 real-concurrency rounds
# ---------------------------------------------------------------------------


def _pg_factory(pg_app_url):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(pg_app_url)
    return sessionmaker(bind=engine), engine


def test_cas_first_issuance_exactly_one_winner(pg_app_url):
    factory, engine = _pg_factory(pg_app_url)
    try:
        for round_no in range(50):
            user, purpose = f"cas1-{round_no}", "p"
            outcomes = []
            barrier = threading.Barrier(2)

            def issue(n):
                with tenant_scope("default", "default"):
                    s = factory()
                    try:
                        barrier.wait(timeout=10)
                        try:
                            grant_purpose(s, user, purpose, operations=["recall"], issuer=f"i{n}")
                            s.commit()
                            outcomes.append("won")
                        except StaleGrantVersion:
                            s.rollback()
                            outcomes.append("lost")
                    finally:
                        s.close()

            ts = [threading.Thread(target=issue, args=(n,)) for n in (1, 2)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=15)
            assert outcomes.count("won") == 1 and outcomes.count("lost") == 1, (round_no, outcomes)
            with tenant_scope("default", "default"):
                s = factory()
                grants = s.query(PurposeGrant).filter_by(user_id=user, purpose=purpose).all()
                assert len(grants) == 1 and grants[0].version == 1
                s.close()
    finally:
        engine.dispose()


def test_cas_update_exactly_one_winner(pg_app_url):
    factory, engine = _pg_factory(pg_app_url)
    try:
        for round_no in range(50):
            user = f"cas2-{round_no}"
            with tenant_scope("default", "default"):
                s = factory()
                grant_purpose(s, user, "p", operations=["recall"], issuer="seed")
                s.commit()
                s.close()
            outcomes = []
            barrier = threading.Barrier(2)

            def update(n):
                with tenant_scope("default", "default"):
                    s = factory()
                    try:
                        barrier.wait(timeout=10)
                        try:
                            grant_purpose(
                                s, user, "p", operations=["recall", "observe"],
                                issuer=f"i{n}", expected_version=1,
                            )
                            s.commit()
                            outcomes.append("won")
                        except StaleGrantVersion:
                            s.rollback()
                            outcomes.append("lost")
                    finally:
                        s.close()

            ts = [threading.Thread(target=update, args=(n,)) for n in (1, 2)]
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=15)
            assert outcomes.count("won") == 1 and outcomes.count("lost") == 1, (round_no, outcomes)
            with tenant_scope("default", "default"):
                s = factory()
                row = s.query(PurposeGrant).filter_by(user_id=user, purpose="p").first()
                assert row.version == 2
                s.close()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------


def test_every_change_appends_an_event_row(db_session):
    from mirror_memory.core.models import PurposeGrantEvent

    _grant(db_session)
    grant_purpose(db_session, "u1", PURPOSE, operations=["recall", "explain"],
                  issuer="reviewer-2", expected_version=1, reason="widen")
    db_session.commit()
    revoke_purpose(db_session, "u1", PURPOSE, actor="reviewer-3", reason="offboard")
    db_session.commit()
    events = (
        db_session.query(PurposeGrantEvent)
        .filter_by(user_id="u1", purpose=PURPOSE)
        .order_by(PurposeGrantEvent.id)
        .all()
    )
    assert [(e.action, e.old_version, e.new_version, e.actor, e.reason) for e in events] == [
        ("granted", None, 1, "reviewer", ""),
        ("updated", 1, 2, "reviewer-2", "widen"),
        ("revoked", 2, 3, "reviewer-3", "offboard"),
    ]


def test_revoke_is_idempotent_and_grant_requires_expected_version(db_session):
    _grant(db_session)
    # updating without the expected version is refused
    with pytest.raises(StaleGrantVersion):
        grant_purpose(db_session, "u1", PURPOSE, operations=["recall"], issuer="x")
    # a wrong expected version is refused
    with pytest.raises(StaleGrantVersion):
        grant_purpose(db_session, "u1", PURPOSE, operations=["recall"],
                      issuer="x", expected_version=99)
    assert revoke_purpose(db_session, "u1", PURPOSE, actor="a") is not None
    db_session.commit()
    assert revoke_purpose(db_session, "u1", PURPOSE, actor="a").revoked_at is not None
    # a first issuance expecting a version that cannot exist is refused
    with pytest.raises(StaleGrantVersion):
        grant_purpose(db_session, "u2", "p", operations=["recall"],
                      issuer="x", expected_version=1)
    db_session.rollback()


# ---------------------------------------------------------------------------
# Migration step 6 and the RLS posture on the new tables
# ---------------------------------------------------------------------------


def test_step6_creates_tables_column_and_records_version(tmp_path):
    from sqlalchemy import create_engine, inspect

    from mirror_memory.core import schema_migration as sm

    eng = create_engine(f"sqlite:///{tmp_path}/old.db")
    Base.metadata.create_all(eng)
    # simulate a version-5 database: drop the step-6 artifacts
    with eng.begin() as conn:
        conn.exec_driver_sql("DROP TABLE mm_purpose_grants")
        conn.exec_driver_sql("DROP TABLE mm_purpose_grant_events")
        conn.exec_driver_sql("ALTER TABLE mm_evolution_jobs DROP COLUMN purpose")
    from mirror_memory.core.schema_migration import _ensure_version_table, _version_table
    from sqlalchemy import MetaData

    _ensure_version_table(eng)
    vt = _version_table(MetaData())
    with eng.begin() as conn:
        conn.execute(vt.delete())
        conn.execute(vt.insert().values(version=5, note="simulated v5"))
    sm.upgrade_database(eng, default_tenant="legacy", default_app="app", config=_migration_cfg())
    assert sm.current_version(eng) == sm.LATEST_VERSION
    step = sm.Step6PurposeAuthorization()
    insp = inspect(eng)
    assert {"mm_purpose_grants", "mm_purpose_grant_events"} <= set(insp.get_table_names())
    assert "purpose" in {c["name"] for c in insp.get_columns("mm_evolution_jobs")}
    assert not step.needed(eng)
    step.apply(eng)  # re-entrant
    step.verify(eng)
    eng.dispose()


def test_new_tables_have_the_rls_posture_and_cross_tenant_reads_are_empty(pg_admin_engine, pg_app_url):
    from sqlalchemy import create_engine, select

    from mirror_memory.core import rls
    from mirror_memory.core.models import PurposeGrant

    rls.verify_posture(pg_admin_engine, tables=["mm_purpose_grants", "mm_purpose_grant_events"])
    # tenant A seeds a grant through the app role
    from mirror_memory.core.tenancy import tenant_scope

    engine = create_engine(pg_app_url)
    try:
        with tenant_scope("tA", "app"):
            seeder = Session(engine)
            grant_purpose(seeder, "u1", "api", operations=["recall"], issuer="t")
            seeder.commit()
            seeder.close()
        # tenant B (same user id, on purpose) sees zero rows
        with tenant_scope("tB", "app"):
            session = Session(engine)
            assert session.query(PurposeGrant).count() == 0
            assert authorize(session, "u1", "api", "recall").reason == "no_grant"
            session.close()
        with tenant_scope("tA", "app"):
            session = Session(engine)
            assert session.query(PurposeGrant).count() == 1
            session.close()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# The two-layer truth table (purpose grant x feature consent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "purpose_granted,feature_granted,expected_reason",
    [
        (True, True, "granted"),
        (True, False, "feature_disabled"),
        (False, True, "no_grant"),
        (False, False, "no_grant"),  # the purpose layer runs first
    ],
)
def test_two_layer_truth_table(db_session, purpose_granted, feature_granted, expected_reason):
    set_memory_enabled(db_session, "u1", True)
    if purpose_granted:
        _grant(db_session, operations=("recall",))
    if feature_granted:
        set_feature_consent(db_session, "u1", "recall", granted=True)
    else:
        set_feature_consent(db_session, "u1", "recall", granted=False)
    db_session.commit()
    decision = authorize(db_session, "u1", "api" if purpose_granted else "other", "recall")
    assert (decision.allowed, decision.reason) == (expected_reason == "granted", expected_reason)
