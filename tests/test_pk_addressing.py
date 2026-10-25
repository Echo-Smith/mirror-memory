"""M1 patch -- cross-tenant PK-addressing counterexamples.

The evaluation called out that ORM writes addressed purely by primary key
(UOW statements) are exempt from the SQL auditor, and that objects do not
necessarily come from a tenant-filtered load: session.merge(), hand-built
objects with a foreign PK (make_transient_to_detached + add), and deletes
of such objects all bypass load-time filtering.

Composite primary keys (which would put tenant_id into every UOW WHERE)
are not available on SQLite: SQLAlchemy raises ``CompileError: SQLite does
not support autoincrement for composite primary keys``.  The surface is
therefore closed in three layers, each locked by a test here:

1. ``session.merge()`` of a foreign-PK object: its internal PK fetch does
   not pass through loader criteria or the Session.get wrapper -- the
   merge wrapper checks the merged instance's tenant and refuses.
2. Hand-built objects carrying a foreign PK: intercepted at *construction*
   time by the whitelist write guard (attribute set on ``id`` etc.).
3. Anything that slips to flush: the flush-time tenant checks refuse
   dirty/deleted objects whose tenant disagrees with the ambient context.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, orm
from sqlalchemy.orm import sessionmaker

from mirror_memory.core.models import Base, Belief
from mirror_memory.core.tenancy import (
    TenantMismatchError,
    WriteGuardViolation,
    fixture_write_scope,
    tenant_scope,
)


def session_at(url):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def seeded_tB(tmp_path):
    """One belief row in tenant B; returns (factory, its id)."""
    factory = session_at(f"sqlite:///{tmp_path}/pk.db")
    with tenant_scope("tB", "app"), fixture_write_scope("seed tB row"):
        s = factory()
        b = Belief(user_id="u1", dimension="topic", key="k", claim_text="TB-SECRET")
        s.add(b)
        s.commit()
        tb_id = b.id
        s.close()
    return factory, tb_id


def test_merge_foreign_pk_is_refused(seeded_tB):
    factory, tb_id = seeded_tB
    with tenant_scope("tA", "app"):
        s = factory()
        try:
            with pytest.raises((WriteGuardViolation, TenantMismatchError)):
                # Construction of the foreign-PK object trips the whitelist
                # guard first; if a scope was somehow active, the merge
                # wrapper refuses the cross-tenant row.
                s.merge(Belief(id=tb_id, user_id="u1", dimension="topic", key="k"))
        finally:
            s.rollback()
            s.close()
    _assert_tB_row_intact(factory, tb_id)


def test_detached_attach_foreign_pk_is_refused(seeded_tB):
    factory, tb_id = seeded_tB
    with tenant_scope("tA", "app"):
        s = factory()
        try:
            with pytest.raises(WriteGuardViolation):
                # Constructing the object already sets its columns -- the
                # whitelist guard fires at authoring time, before any
                # attach or flush can happen.
                obj = Belief(id=tb_id, user_id="u1", dimension="topic", key="k")
                orm.make_transient_to_detached(obj)
                s.add(obj)
                obj.claim_text = "HIJACKED-B"
                s.commit()
        finally:
            s.rollback()
            s.close()
    _assert_tB_row_intact(factory, tb_id)


def test_delete_foreign_pk_is_refused(seeded_tB):
    factory, tb_id = seeded_tB
    with tenant_scope("tA", "app"):
        s = factory()
        try:
            with pytest.raises((WriteGuardViolation, TenantMismatchError)):
                obj = Belief(id=tb_id, user_id="u1", dimension="topic", key="k")
                orm.make_transient_to_detached(obj)
                s.add(obj)
                s.delete(obj)
                s.commit()
        finally:
            s.rollback()
            s.close()
    _assert_tB_row_intact(factory, tb_id)


def _assert_tB_row_intact(factory, tb_id):
    with tenant_scope("tB", "app"):
        s = factory()
        try:
            row = s.query(Belief).filter(Belief.id == tb_id).one()
            assert row.claim_text == "TB-SECRET"
        finally:
            s.close()
